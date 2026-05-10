"""AIFS on Modal — single app for ingestion (ARCO/us-central1) and inference (GPU).

Two images on the same app:

``ingest_image``
    Lightweight CPU image used by ARCO-ERA5 ingestion functions, which run in
    ``us-central1`` for in-region GCS bandwidth.
``infer_image``
    CUDA + torch + flash-attn image used by inference functions and the
    CPU-side orchestrators that share its dependencies.

Open-data and CDS-based ingestion (``ifs-ekd``, ``era5-cds``) runs inline in
``run_forecast`` when ICs are missing. ARCO and ArrayLake sources are dispatched
to their dedicated co-located functions (``ingest_era5_arco``,
``ingest_ifs_arraylake``).
"""

import concurrent.futures
import contextlib
import datetime
import os
import random
import time
from os import path

import earthkit.regrid as ekr
import icechunk
import modal
import numpy as np
import xarray as xr
import zarr

from aifs_modal import _ic as ic
from aifs_modal import _ingest_ekd as ingest_ekd
from aifs_modal import settings, utils

# volumes
data_volume = modal.Volume.from_name(settings.DATA_VOLUME_NAME, create_if_missing=True)
models_volume = modal.Volume.from_name(
    settings.MODELS_VOLUME_NAME, create_if_missing=True
)
ic_volume = modal.Volume.from_name(settings.IC_VOLUME_NAME, create_if_missing=True)

# secrets
_secrets = [
    modal.Secret.from_name("aws-credentials"),
    modal.Secret.from_name("arraylake-api-token"),
    modal.Secret.from_name("huggingface-secret"),
]

app = modal.App(settings.APP_NAME)


@app.local_entrypoint()
def create_volumes() -> None:
    """Create Modal Volumes required by aifs-modal (idempotent, safe to re-run).

    Run once during first-time setup::

        modal run -m aifs_modal._app

    This forces each volume to be hydrated (created if missing) so that
    subsequent ``app.run()`` calls can resolve all dependency object IDs.
    """
    for name, vol in [
        (settings.DATA_VOLUME_NAME, data_volume),
        (settings.MODELS_VOLUME_NAME, models_volume),
        (settings.IC_VOLUME_NAME, ic_volume),
    ]:
        vol.hydrate()
        print(f"  {name}: ok")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

ingest_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "arraylake",
        "boto3",
        "dask",
        "earthkit-data",
        "earthkit-regrid>=0.5.1,<0.6",
        "gcsfs",
        "icechunk>=2.0.3,<3",
        "numpy",
        "xarray",
        "zarr",
    )
    .run_commands(
        # this avoids `ValueError: No matrix found! in_grid={'grid': (0.25, 0.25)}
        # out_grid={'grid': 'N320'} method='linear'`
        # TODO: migrate regridding to earthkit-geo
        "python -c 'import earthkit.regrid.db as db; "
        'db.SYS_DB.find({"grid": [0.25, 0.25]}, {"grid": "N320"}, "linear")\''
    )
)


flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
infer_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
    .uv_pip_install(
        "anemoi-inference[huggingface]==0.6.3",
        "anemoi-models==0.5.1",
        "anemoi-utils==0.4.22",
        "arraylake",
        "boto3",
        "earthkit-regrid>=0.5.1,<0.6",
        "ecmwf-opendata",
        flash_attn_release,
        "icechunk>=2.0.3,<3",
        "numpy",
        "torch==2.9.0",
        "torch-geometric==2.4.0",
        "xarray",
        "zarr",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .env(
        {
            "HF_HUB_CACHE": path.join(settings.MODELS_DIR, "hf_hub_cache"),
            "TORCH_HOME": path.join(settings.MODELS_DIR, "torch"),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "ANEMOI_INFERENCE_NUM_CHUNKS": "16",
            **settings.INFER_IMAGE_ZARR_ENV,
        }
    )
)


# ---------------------------------------------------------------------------
# Output grid constants (0.25° lat/lon, AIFS pressure levels)
# ---------------------------------------------------------------------------

LAT = 90 - 0.25 * np.arange(721)
LON = 0.25 * np.arange(1440)
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PRESSURE_VAR_PREFIXES = ("q", "t", "u", "v", "w", "z")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _without_aws_env():
    keys = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_ENDPOINT_URL",
    ]
    saved = {key: os.environ.pop(key) for key in keys if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def get_gpu_regridder(source_grid, target_grid, method="linear"):
    """Create a GPU regridder using weights from Earthkit regrid."""
    import torch

    class GPU_Regridder:
        def __init__(self, source_grid, target_grid, method="linear"):
            weights_csr, self.target_shape = ekr.db.find(
                source_grid, target_grid, method
            )
            self.weights = torch.sparse_csr_tensor(
                torch.from_numpy(weights_csr.indptr),
                torch.from_numpy(weights_csr.indices),
                torch.from_numpy(weights_csr.data),
                size=weights_csr.shape,
            ).cuda()

        def regrid(self, data):
            tensor = torch.from_numpy(data.astype("f8")).cuda()
            regridded = self.weights.matmul(tensor)
            return regridded.cpu().numpy().astype("f4").reshape(self.target_shape)

    return GPU_Regridder(source_grid, target_grid, method)


def state_to_xarray(state, regridder, init_time, include_pressure_levels=False):
    """Convert state fields to an xarray dataset with init_time + lead_time schema."""
    fields = state["fields"]
    valid_time = state["date"].replace(tzinfo=None)
    lead_time = valid_time - init_time

    ds = xr.Dataset(
        {
            vname: (
                ("init_time", "lead_time", "lat", "lon"),
                regridder.regrid(array)[None, None, :, :],
            )
            for vname, array in fields.items()
        },
        coords={
            "init_time": (
                "init_time",
                [init_time],
                {"standard_name": "forecast_reference_time"},
            ),
            "lead_time": (
                "lead_time",
                [lead_time],
                {"standard_name": "forecast_period"},
            ),
            "lat": ("lat", LAT, {"standard_name": "latitude", "axis": "Y"}),
            "lon": ("lon", LON, {"standard_name": "longitude", "axis": "X"}),
            "pressure": PRESSURE_LEVELS,
        },
    )

    to_drop = []
    for pvar in PRESSURE_VAR_PREFIXES:
        vnames = [f"{pvar}_{plev}" for plev in PRESSURE_LEVELS]
        if include_pressure_levels:
            ds[pvar] = xr.concat(
                [
                    ds[vname].assign_coords(pressure=plev)
                    for vname, plev in zip(vnames, PRESSURE_LEVELS)
                ],
                dim="pressure",
            ).transpose("init_time", "lead_time", ...)
        to_drop.extend(vnames)

    return ds.drop_vars(to_drop)


def _make_fetch_fn(
    source: str,
    *,
    source_repo: str | None = None,
    source_branch: str = "main",
):
    """Resolve a source label into a ``fetch_fn(date) -> dict`` callable."""
    if source == "ifs-ekd":
        return ingest_ekd.get_all_data
    if source == "era5-cds":
        return lambda d: ingest_ekd.get_all_data(d, cds=True)
    if source == "era5-arco":
        from aifs_modal import _ingest_era5_arco as ingest_arco

        # ACHTUNG: import here because of optional `gcsfs` dep
        return ingest_arco.get_all_data
    if source == "ifs-arraylake":
        import arraylake as al

        from aifs_modal import _ingest_ifs_arraylake as ingest_arraylake

        # ACHTUNG: import here because of optional `arraylake` dep
        if source_repo is None:
            raise ValueError("source_repo is required when source='ifs-arraylake'")
        with _without_aws_env():
            al_client = al.Client(token=os.environ["ARRAYLAKE_API_TOKEN"])
        al_session = al_client.get_repo(source_repo).readonly_session(source_branch)
        source_ds = xr.open_dataset(
            al_session.store, engine="zarr", zarr_format=3, chunks={}
        )
        return lambda d: ingest_arraylake.get_all_data(source_ds, d)
    raise ValueError(
        f"Unknown source: {source!r}. Must be one of: {', '.join(settings.IC_SOURCES)}"
    )


def _require_outputs_target(
    outputs_repo: str | None, outputs_prefix: str | None
) -> None:
    """Forecast outputs are bespoke artifacts: the user must name where they go."""
    if outputs_repo is None and outputs_prefix is None:
        raise ValueError(
            "outputs_prefix (or outputs_repo for arraylake) is required: "
            "forecast outputs are not given a default name to avoid silently "
            "overwriting other runs."
        )


def _ic_dates_present(date: datetime.datetime, ic_dir: str) -> bool:
    """Return True if ICs for *date* and *date−6h* are both on the volume."""
    for d in [date - datetime.timedelta(hours=6), date]:
        if not os.path.exists(os.path.join(ic_dir, utils.datetime_to_str(d))):
            return False
    return True


def _open_outputs_repo(
    storage_bucket: str,
    *,
    outputs_repo: str | None = None,
    outputs_prefix: str | None = None,
    storage_type: str = "tigris",
):
    """Return an icechunk Repository for forecast outputs."""
    if outputs_repo is not None:
        import arraylake as al

        with _without_aws_env():
            client = al.Client(token=os.environ["ARRAYLAKE_API_TOKEN"])
            return client.get_repo(
                outputs_repo, config=icechunk.RepositoryConfig.default()
            )
    return icechunk.Repository.open_or_create(
        utils.get_storage(storage_bucket, outputs_prefix, storage_type)
    )


def _run_member(
    date: datetime.datetime,
    runner,
    fields: dict,
    regridder,
    *,
    member_id: int | None = None,
    lead_time: int = 96,
    include_pressure_levels: bool = False,
):
    """Run one forecast member with a pre-built runner/regridder/fields."""
    import torch
    from anemoi.inference.outputs.printer import print_state

    date_no_tz = date.replace(tzinfo=None)
    label = f"member {member_id}" if member_id is not None else "forecast"

    expected_vars = set(runner.checkpoint.variable_to_input_tensor_index)
    extra = set(fields) - expected_vars
    if extra:
        print(f"{label}: dropping {len(extra)} extra variables: {sorted(extra)}")
        fields = {k: v for k, v in fields.items() if k in expected_vars}
    input_state = dict(date=date_no_tz, fields=fields)

    if member_id is not None:
        torch.manual_seed(member_id)

    print(f"{label}: running forecast")
    steps = []
    for state in runner.run(input_state=input_state, lead_time=lead_time):
        print_state(state)
        steps.append(
            state_to_xarray(
                state,
                regridder=regridder,
                init_time=date_no_tz,
                include_pressure_levels=include_pressure_levels,
            )
        )

    ds = xr.concat(steps, dim="lead_time")
    if member_id is not None:
        ds = ds.expand_dims(ensemble_member=[member_id])

    torch.cuda.empty_cache()
    return ds


# ---------------------------------------------------------------------------
# ARCO-ERA5 ingestion (us-central1, CPU)
# ---------------------------------------------------------------------------


@app.function(
    image=ingest_image,
    region="us-east",
    timeout=60 * 60 * 4,
    secrets=_secrets,
    volumes={settings.IC_DIR: ic_volume},
)
def ingest_ifs_arraylake(
    start_date: str,
    end_date: str,
    *,
    source_repo: str,
    source_branch: str = "main",
    source_static_branch: str = "add-static-vars",
) -> None:
    """IFS-arraylake ingestion, co-located with the Cloudflare R2 ENAM bucket."""
    import arraylake as al

    from aifs_modal import _ingest_ifs_arraylake as ingest_arraylake

    with _without_aws_env():
        al_client = al.Client(token=os.environ["ARRAYLAKE_API_TOKEN"])
    al_repo = al_client.get_repo(source_repo)
    source_ds = xr.open_dataset(
        al_repo.readonly_session(source_branch).store,
        engine="zarr",
        zarr_format=3,
        chunks={},
    )
    # TODO: remove source_static_branch once static vars are merged to main
    static_ds = xr.open_dataset(
        al_repo.readonly_session(source_static_branch).store,
        engine="zarr",
        zarr_format=3,
        chunks={},
    )
    static_data = ingest_arraylake._read_static_fields(static_ds)

    # fetch_fn = lambda d: ingest_arraylake.get_all_data(source_ds, d, static_data)
    def fetch_fn(d):
        return ingest_arraylake.get_all_data(source_ds, d, static_data)

    ic.ingest_range(
        start_date, end_date, settings.IC_DIR, fetch_fn, source="ifs-arraylake"
    )
    ic_volume.commit()


@app.function(
    image=ingest_image,
    region="us-central1",
    timeout=60 * 60 * 4,
    secrets=_secrets,
    volumes={settings.IC_DIR: ic_volume},
)
def ingest_era5_arco(
    start_date: str,
    end_date: str,
) -> None:
    """ARCO-ERA5 ingestion, co-located with the ``us-central1`` bucket."""
    from aifs_modal import _ingest_era5_arco as ingest_arco

    ic.ingest_range(
        start_date,
        end_date,
        settings.IC_DIR,
        ingest_arco.get_all_data,
        source="era5-arco",
    )
    ic_volume.commit()


# ---------------------------------------------------------------------------
# Inference (GPU + CPU orchestrators on infer_image)
# ---------------------------------------------------------------------------


def _drop_regionless_coords(ds: xr.Dataset, region_dim: str) -> xr.Dataset:
    """Drop coords that don't span ``region_dim``.

    Ensures that region-write does not try to overwrite them (the template already has
    them).
    """
    coord_names = [
        name for name, coord in ds.coords.items() if region_dim not in coord.dims
    ]
    return ds.drop_vars(coord_names)


def _ensemble_output_complete(ds: xr.Dataset, n_members: int) -> bool:
    """Return whether an ensemble output group was fully committed."""
    return (
        ds.attrs.get("aifs_modal_status") == "complete"
        and ds.sizes.get("ensemble_member", 0) >= n_members
    )


def _cancel_pending_member_handles(handles_by_member: dict[int, object]) -> None:
    """Best-effort cancellation for member calls still owned by the orchestrator."""
    for member_id, handle in handles_by_member.items():
        cancel = getattr(handle, "cancel", None)
        if not callable(cancel):
            continue
        try:
            cancel(terminate_containers=True)
        except TypeError:
            cancel()
        except Exception as exc:
            print(f"member {member_id}: failed to cancel pending call: {exc!r}")


def _collect_member_forks(
    handles: list[object],
    *,
    poll_interval: float = settings.ENSEMBLE_MEMBER_POLL_SECONDS,
    timeout: float = settings.ENSEMBLE_MEMBER_COLLECTION_TIMEOUT_SECONDS,
    clock=time.monotonic,
    sleep=time.sleep,
) -> list[object]:
    """Poll spawned member calls and return forks in member order."""
    started = clock()
    next_log = started
    pending = dict(enumerate(handles))
    returned: dict[int, object] = {}

    while pending:
        for member_id, handle in list(pending.items()):
            try:
                returned[member_id] = handle.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:
                _cancel_pending_member_handles(pending)
                raise RuntimeError(f"ensemble member {member_id} failed") from exc
            else:
                del pending[member_id]
                print(f"member {member_id}: returned fork")

        if not pending:
            break

        now = clock()
        elapsed = now - started
        if elapsed > timeout:
            pending_members = sorted(pending)
            _cancel_pending_member_handles(pending)
            raise TimeoutError(
                "timed out waiting for ensemble member forks after "
                f"{elapsed:.0f}s; pending members: {pending_members}"
            )

        if now >= next_log:
            print(
                f"waiting for {len(pending)} ensemble member(s) to return forks: "
                f"{sorted(pending)} ({elapsed:.0f}s elapsed)"
            )
            next_log = now + poll_interval

        sleep(min(poll_interval, max(0.0, timeout - elapsed)))

    return [returned[i] for i in range(len(handles))]


def _as_str_list(val) -> list[str] | None:
    """Coerce ``val`` into a list of variable-name strings, or None.

    Accepts dicts (returns whichever side is all-strings), or iterables of
    strings.  Returns None if the value isn't shaped like a list of names.
    """
    if val is None:
        return None
    if isinstance(val, dict):
        keys = list(val.keys())
        if keys and all(isinstance(k, str) for k in keys):
            return keys
        vals = list(val.values())
        if vals and all(isinstance(v, str) for v in vals):
            return vals
        return None
    try:
        seq = list(val)
    except TypeError:
        return None
    if seq and all(isinstance(x, str) for x in seq):
        return seq
    return None


def _checkpoint_output_variables(ck) -> list[str]:
    """Read the output variable list from an anemoi Checkpoint.

    Anemoi has shifted the public API for this across versions, so we try a
    few known shapes in order:
      1. ``variable_to_output_tensor_index`` / ``output_tensor_index_to_variable``
         (direct mapping of output-tensor positions to names);
      2. ``prognostic_variables + diagnostic_variables`` (prognostics are
         advanced by the model, diagnostics are side-outputs like precip /
         radiation / clouds);
      3. fall back to ``variable_to_input_tensor_index`` (covers prognostics
         only — will miss diagnostics if the model has any).
    """
    for attr in ("variable_to_output_tensor_index", "output_tensor_index_to_variable"):
        names = _as_str_list(getattr(ck, attr, None))
        if names:
            return names

    prog = _as_str_list(getattr(ck, "prognostic_variables", None)) or []
    diag = _as_str_list(getattr(ck, "diagnostic_variables", None)) or []
    if prog or diag:
        seen: set[str] = set()
        out: list[str] = []
        for v in (*prog, *diag):
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    return _as_str_list(getattr(ck, "variable_to_input_tensor_index", None)) or []


def _introspect_output_schema(
    checkpoint: dict | None,
    include_pressure_levels: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load the checkpoint on CPU once to read its output variable list.

    Returns ``(surface_vars, pressure_vars)``: surface vars are the 4D
    ``(init, lead, lat, lon)`` outputs; pressure vars are the prefixes
    (``q``, ``t``, ...) that ``state_to_xarray`` stacks into 5D when
    ``include_pressure_levels`` is true.
    """
    from anemoi.inference.runners.simple import SimpleRunner

    if checkpoint is None:
        checkpoint = settings.AIFS_ENS_CHECKPOINT
    runner = SimpleRunner(checkpoint, device="cpu")
    ck = runner.checkpoint
    out_vars = _checkpoint_output_variables(ck)
    in_vars = _as_str_list(getattr(ck, "variable_to_input_tensor_index", None)) or []
    if not out_vars or not all(isinstance(v, str) for v in out_vars):
        raise RuntimeError(
            "could not extract output variable names from anemoi Checkpoint. "
            f"got {type(out_vars).__name__} of length {len(out_vars)}: "
            f"first 5 = {out_vars[:5]!r}. "
            f"available attrs: {sorted(a for a in dir(ck) if not a.startswith('_'))}"
        )
    if set(out_vars) <= set(in_vars):
        # input-only schema means we missed the model's diagnostics; better
        # to fail here than build a template that crashes every member.
        raise RuntimeError(
            "output variable schema only contains inputs (no diagnostics). "
            f"in_vars={in_vars[:5]}..., out_vars={out_vars[:5]}.... "
            f"available attrs: {sorted(a for a in dir(ck) if not a.startswith('_'))}"
        )
    print(f"discovered {len(out_vars)} output variables (input vars: {len(in_vars)})")

    pressure_vars: list[str] = []
    surface_vars: list[str] = []
    pressure_seen: set[str] = set()
    for v in out_vars:
        prefix, _, level = v.partition("_")
        if level and level.isdigit() and prefix in PRESSURE_VAR_PREFIXES:
            if prefix not in pressure_seen:
                pressure_vars.append(prefix)
                pressure_seen.add(prefix)
        else:
            surface_vars.append(v)

    if include_pressure_levels:
        return tuple(surface_vars), tuple(pressure_vars)
    # pressure vars get dropped by state_to_xarray when include_pressure_levels=False
    return tuple(surface_vars), ()


def _empty_ensemble_template(
    date: datetime.datetime,
    *,
    n_members: int,
    lead_time: int,
    surface_vars: tuple[str, ...],
    pressure_vars: tuple[str, ...],
) -> xr.Dataset:
    """Build a Dask-backed empty template for compute=False allocation."""
    import dask.array as da

    date_no_tz = date.replace(tzinfo=None)
    n_steps = lead_time // 6
    coords = {
        "ensemble_member": np.arange(n_members, dtype=np.int64),
        "init_time": (
            "init_time",
            [date_no_tz],
            {"standard_name": "forecast_reference_time"},
        ),
        "lead_time": (
            "lead_time",
            [datetime.timedelta(hours=6 * (i + 1)) for i in range(n_steps)],
            {"standard_name": "forecast_period"},
        ),
        "lat": ("lat", LAT, {"standard_name": "latitude", "axis": "Y"}),
        "lon": ("lon", LON, {"standard_name": "longitude", "axis": "X"}),
        "pressure": PRESSURE_LEVELS,
    }
    surface_dims = ("ensemble_member", "init_time", "lead_time", "lat", "lon")
    surface_shape = (n_members, 1, n_steps, LAT.size, LON.size)
    surface_chunks = (1, 1, n_steps, 320, 320)

    pressure_dims = (
        "ensemble_member",
        "init_time",
        "lead_time",
        "pressure",
        "lat",
        "lon",
    )
    pressure_shape = (n_members, 1, n_steps, len(PRESSURE_LEVELS), LAT.size, LON.size)
    pressure_chunks = (1, 1, n_steps, len(PRESSURE_LEVELS), 320, 320)

    data_vars: dict[str, tuple] = {}
    for v in surface_vars:
        data_vars[v] = (
            surface_dims,
            da.empty(surface_shape, chunks=surface_chunks, dtype="f4"),
        )
    for v in pressure_vars:
        data_vars[v] = (
            pressure_dims,
            da.empty(pressure_shape, chunks=pressure_chunks, dtype="f4"),
        )
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs["aifs_modal_status"] = "initialized"
    ds.attrs["n_members"] = n_members
    ds.attrs["lead_time_hours"] = lead_time
    return ds


def _chunk_member_ds(ds: xr.Dataset, *, n_steps: int) -> xr.Dataset:
    """Chunk a per-member ds to the final per-region layout (ensemble_member=1)."""
    surface_chunk = {
        "init_time": 1,
        "lead_time": n_steps,
        "ensemble_member": 1,
        "lat": 320,
        "lon": 320,
    }
    pressure_chunk = {**surface_chunk, "pressure": 13}
    for v in ds.data_vars:
        target = pressure_chunk if "pressure" in ds[v].dims else surface_chunk
        ds[v] = ds[v].chunk({k: vv for k, vv in target.items() if k in ds[v].dims})
        ds[v].encoding.pop("chunks", None)
        ds[v].encoding.pop("preferred_chunks", None)
    return ds


@app.function(
    image=infer_image,
    gpu=settings.GPU_TYPE,
    timeout=60 * 60 * 4,
    retries=2,
    volumes={settings.MODELS_DIR: models_volume, settings.IC_DIR: ic_volume},
    secrets=_secrets,
)
def run_ensemble_member(
    member_id: int,
    date: datetime.datetime,
    fork: "icechunk.ForkSession",
    *,
    base_group: str,
    lead_time: int = 96,
    checkpoint: dict | None = None,
    include_pressure_levels: bool = False,
) -> "icechunk.ForkSession":
    """Run one ensemble member and region-write its slice into ``fork``.

    The output store + template are pre-allocated by the orchestrator, so each
    member just writes ``region={"ensemble_member": slice(m, m+1)}`` and
    returns the mutated fork.
    """
    from anemoi.inference.runners.simple import SimpleRunner

    ic_volume.reload()
    print(f"member {member_id}: loading initial conditions for {date}")
    fields = ic.fetch_initial_conditions(date, settings.IC_DIR)
    print(f"member {member_id}: loading model checkpoint")
    if checkpoint is None:
        checkpoint = settings.AIFS_ENS_CHECKPOINT
    runner = SimpleRunner(checkpoint, device="cuda")
    print(f"member {member_id}: initializing regridder")
    regridder = get_gpu_regridder({"grid": "N320"}, {"grid": (0.25, 0.25)})
    print(f"member {member_id}: running inference")

    ds = _run_member(
        date,
        runner,
        fields,
        regridder,
        member_id=member_id,
        lead_time=lead_time,
        include_pressure_levels=include_pressure_levels,
    )
    ds = _chunk_member_ds(ds, n_steps=lead_time // 6)
    ds = _drop_regionless_coords(ds, "ensemble_member")

    jitter = random.uniform(0, settings.ENSEMBLE_MEMBER_WRITE_JITTER_SECONDS)
    if jitter > 0:
        print(f"member {member_id}: jittering write by {jitter:.1f}s")
        time.sleep(jitter)

    print(f"member {member_id}: writing region")
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(
        ds.to_zarr,
        fork.store,
        group=base_group,
        zarr_format=3,
        consolidated=False,
        region={"ensemble_member": slice(member_id, member_id + 1)},
    )
    done, _ = concurrent.futures.wait(
        [fut], timeout=settings.ENSEMBLE_MEMBER_WRITE_TIMEOUT_SECONDS
    )
    ex.shutdown(wait=False)
    if not done:
        raise TimeoutError(
            f"member {member_id}: zarr write timed out after "
            f"{settings.ENSEMBLE_MEMBER_WRITE_TIMEOUT_SECONDS}s"
        )
    fut.result()
    print(f"member {member_id}: wrote region")
    print(f"member {member_id}: returning fork")
    return fork


@app.function(
    image=infer_image,
    timeout=60 * 60 * 6,
    volumes={settings.MODELS_DIR: models_volume, settings.IC_DIR: ic_volume},
    secrets=_secrets,
)
def run_forecast(
    date: datetime.datetime,
    storage_bucket: str,
    *,
    source: str | None = None,
    source_repo: str | None = None,
    source_branch: str = "main",
    source_static_branch: str = "add-static-vars",
    lead_time: int | None = None,
    outputs_repo: str | None = None,
    outputs_prefix: str | None = None,
    outputs_branch: str = "main",
    checkpoint: dict | None = None,
    n_members: int | None = None,
    parallel_members: bool = True,
    include_pressure_levels: bool = False,
    overwrite: bool = False,
    keep_ics: bool = False,
    storage_type: str = "tigris",
) -> None:
    """CPU orchestrator: ingest initial conditions if absent, then run AIFS inference.

    If ICs are missing, ingestion is dispatched to the appropriate Modal
    function: ``ingest_era5_arco`` (us-central1) for ``era5-arco``,
    ``ingest_ifs_arraylake`` (us-east) for ``ifs-arraylake``, or inline for
    ``ifs-ekd``/``era5-cds``. AIFS inference is dispatched to a separate GPU
    container, so GPU billing is limited to inference only.

    Modes:

    - ``n_members=None`` — deterministic forecast (AIFS-Single by default).
    - ``n_members=k, parallel_members=False`` — ``k`` ensemble members
      sequentially on a single GPU (cheaper for small ``k``; reuses one runner).
    - ``n_members=k, parallel_members=True`` — ``k`` ensemble members in
      parallel, one GPU per member, using icechunk fork/merge writes.

    Parameters
    ----------
    source : str, optional
        One of :data:`aifs_modal.settings.IC_SOURCES`. Defaults to
        :data:`aifs_modal.settings.DEFAULT_IC_SOURCE`.  ICs already present in
        the target volume are skipped.
    source_repo : str, optional
        ArrayLake repository name. Required when ``source="ifs-arraylake"``.
    source_branch : str, optional
        Branch in the ArrayLake source repository. Default ``"main"``.
    keep_ics : bool, optional
        If ``False`` (default), IC data is deleted from the Modal Volume after
        successful inference. Set to ``True`` to retain ICs for reuse.
    n_members : int, optional
        If set, run an ensemble forecast with this many members. If ``None``,
        run a single deterministic forecast.
    parallel_members : bool, optional
        Only meaningful when ``n_members`` is set. If ``True`` (default), run
        members in parallel on separate GPUs; if ``False``, run them
        sequentially on one GPU.
    """
    _require_outputs_target(outputs_repo, outputs_prefix)
    if source is None:
        source = settings.DEFAULT_IC_SOURCE

    ic_volume.reload()
    if not _ic_dates_present(date, settings.IC_DIR):
        start = (date - datetime.timedelta(hours=6)).isoformat()
        end = date.isoformat()
        if source == "era5-arco":
            ingest_era5_arco.remote(start, end)
        elif source == "ifs-arraylake":
            ingest_ifs_arraylake.remote(
                start,
                end,
                source_repo=source_repo,
                source_branch=source_branch,
                source_static_branch=source_static_branch,
            )
        else:
            fetch_fn = _make_fetch_fn(
                source, source_repo=source_repo, source_branch=source_branch
            )
            ic.ingest_range(start, end, settings.IC_DIR, fetch_fn, source=source)
            ic_volume.commit()

    if n_members is None or not parallel_members:
        if checkpoint is None:
            checkpoint = (
                settings.AIFS_SINGLE_CHECKPOINT
                if n_members is None
                else settings.AIFS_ENS_CHECKPOINT
            )
        run_inference.remote(
            date,
            storage_bucket,
            checkpoint,
            lead_time=lead_time,
            outputs_repo=outputs_repo,
            outputs_prefix=outputs_prefix,
            outputs_branch=outputs_branch,
            n_members=n_members,
            include_pressure_levels=include_pressure_levels,
            overwrite=overwrite,
            keep_ics=keep_ics,
            storage_type=storage_type,
        )
        return

    # parallel ensemble path: allocate empty store, fan out one GPU per member,
    # merge forks
    outputs_repo_obj = _open_outputs_repo(
        storage_bucket,
        outputs_repo=outputs_repo,
        outputs_prefix=outputs_prefix,
        storage_type=storage_type,
    )
    if outputs_branch not in outputs_repo_obj.list_branches():
        base = outputs_repo_obj.readonly_session("main").snapshot_id
        outputs_repo_obj.create_branch(outputs_branch, base)

    base_group = utils.datetime_to_str(date)
    if not overwrite:
        try:
            existing = xr.open_dataset(
                outputs_repo_obj.readonly_session(outputs_branch).store,
                group=base_group,
                engine="zarr",
                zarr_format=3,
                chunks=None,
            )
            if _ensemble_output_complete(existing, n_members):
                print(
                    f"Ensemble forecast already complete for {date.isoformat()} "
                    f"({n_members} members); skipping"
                )
                return
            if existing.sizes.get("ensemble_member", 0) >= n_members:
                print(
                    f"Existing ensemble output for {date.isoformat()} has "
                    f"status={existing.attrs.get('aifs_modal_status')!r}; rerunning"
                )
        except Exception:
            pass

    lead_time = lead_time or 96

    print("introspecting checkpoint output schema")
    surface_vars, pressure_vars = _introspect_output_schema(
        checkpoint, include_pressure_levels
    )
    print(
        f"output schema: {len(surface_vars)} surface vars, "
        f"{len(pressure_vars)} pressure-level prefixes"
    )

    template = _empty_ensemble_template(
        date,
        n_members=n_members,
        lead_time=lead_time,
        surface_vars=surface_vars,
        pressure_vars=pressure_vars,
    )
    init_session = outputs_repo_obj.writable_session(outputs_branch)
    template.to_zarr(
        init_session.store,
        group=base_group,
        zarr_format=3,
        consolidated=False,
        compute=False,
        mode="w",
    )
    init_session.commit(
        f"initialized ensemble forecast for {base_group} ({n_members} members)"
    )
    print(f"initialized ensemble store for {base_group}")

    session = outputs_repo_obj.writable_session(outputs_branch)
    fork = session.fork()
    member_kwargs = dict(
        base_group=base_group,
        lead_time=lead_time,
        checkpoint=checkpoint,
        include_pressure_levels=include_pressure_levels,
    )
    print(f"spawning {n_members} ensemble members in parallel")
    handles = [
        run_ensemble_member.spawn(m, date, fork, **member_kwargs)
        for m in range(n_members)
    ]
    returned_forks = _collect_member_forks(handles)
    print(f"all {n_members} ensemble members returned forks")

    print(f"merging {len(returned_forks)} member forks")
    session.merge(*returned_forks)
    print("merged member forks")
    group = zarr.open_group(
        session.store,
        path=base_group,
        mode="a",
        zarr_format=3,
    )
    group.attrs["aifs_modal_status"] = "complete"
    group.attrs["n_members"] = n_members
    group.attrs["lead_time_hours"] = lead_time
    session.commit(f"ensemble forecast for {base_group} ({n_members} members)")
    print(f"ensemble forecast for {base_group} ({n_members} members) complete")

    if not keep_ics:
        ic_volume.reload()
        ic.delete_ic_dates(date, settings.IC_DIR)
        ic_volume.commit()


@app.function(
    image=infer_image,
    gpu=settings.GPU_TYPE,
    timeout=60 * 60 * 4,
    volumes={
        settings.DATA_DIR: data_volume,
        settings.MODELS_DIR: models_volume,
        settings.IC_DIR: ic_volume,
    },
    secrets=_secrets,
)
def run_inference(
    date: datetime.datetime,
    storage_bucket: str,
    checkpoint: dict,
    *,
    lead_time: int | None = None,
    outputs_repo: str | None = None,
    outputs_prefix: str | None = None,
    outputs_branch: str = "main",
    n_members: int | None = None,
    include_pressure_levels: bool = False,
    overwrite: bool = False,
    keep_ics: bool = False,
    storage_type: str = "tigris",
) -> None:
    """GPU primitive: deterministic or sequential ensemble AIFS forecast.

    With ``n_members=None`` runs a single deterministic forecast.  With
    ``n_members=k`` runs ``k`` ensemble members sequentially on one GPU,
    reusing the same loaded runner across members, and concatenates them along
    ``ensemble_member`` before a single write.

    Initial conditions must be pre-ingested. Use :func:`run_forecast` to ingest
    (if needed) and dispatch inference in one call.
    """
    from anemoi.inference.runners.simple import SimpleRunner

    _require_outputs_target(outputs_repo, outputs_prefix)

    outputs_repo_obj = _open_outputs_repo(
        storage_bucket,
        outputs_repo=outputs_repo,
        outputs_prefix=outputs_prefix,
        storage_type=storage_type,
    )
    if outputs_branch not in outputs_repo_obj.list_branches():
        base = outputs_repo_obj.readonly_session("main").snapshot_id
        outputs_repo_obj.create_branch(outputs_branch, base)

    base_group = utils.datetime_to_str(date)

    if not overwrite:
        readonly_session = outputs_repo_obj.readonly_session(outputs_branch)
        try:
            existing = zarr.open_group(
                readonly_session.store,
                path=base_group,
                mode="r",
                zarr_format=3,
            )
            if n_members is None or existing.attrs.get("n_members", 0) >= n_members:
                print(
                    f"Forecast already exists for {date.isoformat()} "
                    f"(group: {base_group}); skipping"
                )
                return
        except zarr.errors.GroupNotFoundError:
            pass

    ic_volume.reload()
    print("loading initial conditions for", date)
    fields = ic.fetch_initial_conditions(date, settings.IC_DIR)
    print("loading model checkpoint")
    runner = SimpleRunner(checkpoint, device="cuda")
    print("initializing regridder")
    regridder = get_gpu_regridder({"grid": "N320"}, {"grid": (0.25, 0.25)})
    print("running inference")

    if n_members is None:
        ds = _run_member(
            date,
            runner,
            fields,
            regridder,
            member_id=None,
            lead_time=lead_time,
            include_pressure_levels=include_pressure_levels,
        )
    else:
        member_dss = [
            _run_member(
                date,
                runner,
                fields,
                regridder,
                member_id=m,
                lead_time=lead_time,
                include_pressure_levels=include_pressure_levels,
            )
            for m in range(n_members)
        ]
        ds = xr.concat(member_dss, dim="ensemble_member")

    outputs_session = outputs_repo_obj.writable_session(outputs_branch)
    ds.to_zarr(
        outputs_session.store,
        group=base_group,
        zarr_format=3,
        consolidated=False,
        mode="w",
    )

    member_str = f" ({n_members} members, sequential)" if n_members else ""
    commit_msg = (
        f"{lead_time} hour forecast{member_str} for "
        f"{date.strftime('%Y-%m-%d %H:%M')} written to {outputs_repo_obj}"
    )
    outputs_session.commit(commit_msg)
    print(commit_msg)

    if not keep_ics:
        ic.delete_ic_dates(date, settings.IC_DIR)
        ic_volume.commit()
