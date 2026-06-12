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

import contextlib
import datetime
import os
from collections.abc import Callable
from os import path

import earthkit.regrid as ekr
import icechunk
import modal
import numpy as np
import xarray as xr

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
        if not ic.ic_date_complete(os.path.join(ic_dir, utils.datetime_to_str(d))):
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
# IFS (Brighband) and ERA5 (ARCO GCS) co-located ingestion (CPU)
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
    ic_source_repo: str,
    *,
    ic_source_branch: str = "main",
    ic_source_static_branch: str = "add-static-vars",
) -> None:
    """IFS-arraylake ingestion, co-located with the Cloudflare R2 ENAM bucket."""
    import arraylake as al

    from aifs_modal import _ingest_ifs_arraylake

    with _without_aws_env():
        al_client = al.Client(token=os.environ["ARRAYLAKE_API_TOKEN"])
    al_repo = al_client.get_repo(ic_source_repo)
    source_ds = xr.open_dataset(
        al_repo.readonly_session(ic_source_branch).store,
        engine="zarr",
        zarr_format=3,
        chunks={},
    )
    # TODO: remove ic_source_static_branch once static vars are merged to main
    static_ds = xr.open_dataset(
        al_repo.readonly_session(ic_source_static_branch).store,
        engine="zarr",
        zarr_format=3,
        chunks={},
    )
    static_data = _ingest_ifs_arraylake._read_static_fields(static_ds)
    _ingest_ifs_arraylake.ingest(
        start_date, end_date, settings.IC_DIR, source_ds, static_data
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

    ingest_arco.ingest(start_date, end_date, settings.IC_DIR)
    ic_volume.commit()


# ---------------------------------------------------------------------------
# Inference (GPU + CPU orchestrators on infer_image)
# ---------------------------------------------------------------------------
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
    ic_source: str | None = None,
    ic_source_repo: str | None = None,
    ic_source_branch: str = "main",
    ic_source_static_branch: str = "add-static-vars",
    lead_time: int | None = None,
    outputs_repo: str | None = None,
    outputs_prefix: str | None = None,
    outputs_branch: str = "main",
    checkpoint: dict | None = None,
    n_members: int | None = None,
    include_pressure_levels: bool = False,
    chunk_layout: Callable[[xr.Dataset], xr.Dataset] | None = None,
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

    Parameters
    ----------
    ic_source : str, optional
        One of :data:`aifs_modal.settings.IC_SOURCES`. Defaults to
        :data:`aifs_modal.settings.DEFAULT_IC_SOURCE`.  ICs already present in
        the target volume are skipped.
    ic_source_repo : str, optional
        ArrayLake repository name. Required when ``ic_source="ifs-arraylake"``.
    ic_source_branch : str, optional
        Branch in the ArrayLake source repository. Default ``"main"``.
    keep_ics : bool, optional
        If ``False`` (default), IC data is deleted from the Modal Volume after
        successful inference. Set to ``True`` to retain ICs for reuse.
    n_members : int, optional
        If set, run an AIFS-ENS ensemble forecast with this many members run
        sequentially on a single GPU. If ``None``, run a single deterministic
        AIFS-Single forecast.
    chunk_layout : callable or None, optional
        Function ``(ds: xr.Dataset) -> xr.Dataset`` that sets per-variable
        chunk encoding before writing. Defaults to
        :data:`aifs_modal.settings.DEFAULT_CHUNK_LAYOUT` (default:
        :func:`aifs_modal.apply_trajectory_chunks`): one chunk per
        ``init_time``, 241×240 spatial tiles, full extent on all other
        dimensions. Pass ``None`` to skip explicit chunking (Zarr defaults).
    """

    def _ingest_ekd_with_commit(start, end, ic_dir, source):
        ingest_ekd.ingest(start, end, ic_dir, source)
        ic_volume.commit()

    _run_forecast_impl(
        date,
        storage_bucket,
        ic_dir=settings.IC_DIR,
        ingest_era5_arco_fn=ingest_era5_arco.remote,
        ingest_ifs_arraylake_fn=ingest_ifs_arraylake.remote,
        ingest_ekd_fn=_ingest_ekd_with_commit,
        run_inference_fn=run_inference.remote,
        on_ic_reload=ic_volume.reload,
        ic_source=ic_source,
        ic_source_repo=ic_source_repo,
        ic_source_branch=ic_source_branch,
        ic_source_static_branch=ic_source_static_branch,
        lead_time=lead_time,
        outputs_repo=outputs_repo,
        outputs_prefix=outputs_prefix,
        outputs_branch=outputs_branch,
        checkpoint=checkpoint,
        n_members=n_members,
        include_pressure_levels=include_pressure_levels,
        chunk_layout=chunk_layout,
        overwrite=overwrite,
        keep_ics=keep_ics,
        storage_type=storage_type,
    )


def _run_forecast_impl(
    date: datetime.datetime,
    storage_bucket: str,
    *,
    ic_dir: str,
    ingest_era5_arco_fn: Callable[[str, str], None],
    ingest_ifs_arraylake_fn: Callable[..., None],
    ingest_ekd_fn: Callable[[str, str, str, str], None],
    run_inference_fn: Callable[..., None],
    on_ic_reload: Callable[[], None] = lambda: None,
    open_outputs_repo_fn: Callable[..., object] = _open_outputs_repo,
    ic_source: str | None = None,
    ic_source_repo: str | None = None,
    ic_source_branch: str = "main",
    ic_source_static_branch: str = "add-static-vars",
    lead_time: int | None = None,
    outputs_repo: str | None = None,
    outputs_prefix: str | None = None,
    outputs_branch: str = "main",
    checkpoint: dict | None = None,
    n_members: int | None = None,
    include_pressure_levels: bool = False,
    chunk_layout: Callable[[xr.Dataset], xr.Dataset] | None = None,
    overwrite: bool = False,
    keep_ics: bool = False,
    storage_type: str = "tigris",
) -> None:
    """Core orchestration logic for :func:`run_forecast`, free of Modal calls.

    Tests inject recorder callables for ``ingest_*_fn`` and ``run_inference_fn``
    and a local-FS icechunk repo via ``open_outputs_repo_fn``.
    """
    _require_outputs_target(outputs_repo, outputs_prefix)
    if ic_source is None:
        ic_source = settings.DEFAULT_IC_SOURCE

    if not overwrite:
        base_group = utils.datetime_to_str(date)
        try:
            outputs_repo_obj = open_outputs_repo_fn(
                storage_bucket,
                outputs_repo=outputs_repo,
                outputs_prefix=outputs_prefix,
                storage_type=storage_type,
            )
            if outputs_branch not in outputs_repo_obj.list_branches():
                raise ValueError
            readonly_session = outputs_repo_obj.readonly_session(outputs_branch)
            existing = xr.open_dataset(
                readonly_session.store,
                group=base_group,
                engine="zarr",
                zarr_format=3,
                chunks=None,
            )
            if (
                n_members is None
                or existing.sizes.get("ensemble_member", 0) >= n_members
            ):
                print(
                    f"Forecast already exists for {date.isoformat()} "
                    f"(group: {base_group}); skipping"
                )
                return
        except Exception:
            pass

    on_ic_reload()
    if not _ic_dates_present(date, ic_dir):
        start = (date - datetime.timedelta(hours=6)).isoformat()
        end = date.isoformat()
        if ic_source == "era5-arco":
            ingest_era5_arco_fn(start, end)
        elif ic_source == "ifs-arraylake":
            if ic_source_repo is None:
                raise ValueError(
                    "ic_source_repo is required when ic_source='ifs-arraylake'"
                )
            ingest_ifs_arraylake_fn(
                start,
                end,
                ic_source_repo,
                ic_source_branch=ic_source_branch,
                ic_source_static_branch=ic_source_static_branch,
            )
        else:  # ifs-ekd or era5-cds
            ingest_ekd_fn(start, end, ic_dir, ic_source)

    if checkpoint is None:
        checkpoint = (
            settings.AIFS_SINGLE_CHECKPOINT
            if n_members is None
            else settings.AIFS_ENS_CHECKPOINT
        )
    run_inference_fn(
        date,
        storage_bucket,
        checkpoint,
        lead_time=lead_time,
        outputs_repo=outputs_repo,
        outputs_prefix=outputs_prefix,
        outputs_branch=outputs_branch,
        n_members=n_members,
        include_pressure_levels=include_pressure_levels,
        chunk_layout=chunk_layout,
        overwrite=overwrite,
        keep_ics=keep_ics,
        storage_type=storage_type,
    )


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
    chunk_layout: Callable[[xr.Dataset], xr.Dataset] | None = None,
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

    Parameters
    ----------
    chunk_layout : callable or None, optional
        See :func:`run_forecast` for full description.
    """
    from anemoi.inference.runners.simple import SimpleRunner

    _require_outputs_target(outputs_repo, outputs_prefix)
    outputs_repo_obj = _open_outputs_repo(
        storage_bucket,
        outputs_repo=outputs_repo,
        outputs_prefix=outputs_prefix,
        storage_type=storage_type,
    )

    ran = _run_inference_impl(
        date,
        outputs_repo_obj=outputs_repo_obj,
        runner_factory=lambda: SimpleRunner(checkpoint, device="cuda"),
        regridder_factory=lambda: get_gpu_regridder(
            {"grid": "N320"}, {"grid": (0.25, 0.25)}
        ),
        ic_dir=settings.IC_DIR,
        on_ic_reload=ic_volume.reload,
        outputs_branch=outputs_branch,
        lead_time=lead_time,
        n_members=n_members,
        include_pressure_levels=include_pressure_levels,
        chunk_layout=chunk_layout,
        overwrite=overwrite,
        keep_ics=keep_ics,
    )
    if ran and not keep_ics:
        ic_volume.commit()


def _run_inference_impl(
    date: datetime.datetime,
    *,
    outputs_repo_obj,
    runner_factory: Callable[[], object],
    regridder_factory: Callable[[], object],
    ic_dir: str,
    on_ic_reload: Callable[[], None] = lambda: None,
    outputs_branch: str = "main",
    lead_time: int | None = None,
    n_members: int | None = None,
    include_pressure_levels: bool = False,
    chunk_layout: Callable[[xr.Dataset], xr.Dataset] | None = None,
    overwrite: bool = False,
    keep_ics: bool = False,
) -> bool:
    """Core inference logic for :func:`run_inference`, free of Modal calls.

    Tests inject a ``runner_factory`` returning a fake AIFS runner and a
    ``regridder_factory`` returning a CPU regridder. ``outputs_repo_obj`` may
    be a local-FS icechunk repo.

    Returns
    -------
    bool
        ``True`` if a forecast was written, ``False`` if the existing-forecast
        skip path was taken.
    """
    if outputs_branch not in outputs_repo_obj.list_branches():
        base = outputs_repo_obj.readonly_session("main").snapshot_id
        outputs_repo_obj.create_branch(outputs_branch, base)

    base_group = utils.datetime_to_str(date)

    if not overwrite:
        readonly_session = outputs_repo_obj.readonly_session(outputs_branch)
        try:
            existing = xr.open_dataset(
                readonly_session.store,
                group=base_group,
                engine="zarr",
                zarr_format=3,
                chunks=None,
            )
            if (
                n_members is None
                or existing.sizes.get("ensemble_member", 0) >= n_members
            ):
                print(
                    f"Forecast already exists for {date.isoformat()} "
                    f"(group: {base_group}); skipping"
                )
                return False
        except Exception:
            pass

    on_ic_reload()
    print("loading initial conditions for", date)
    fields = ic.fetch_initial_conditions(date, ic_dir)
    print("loading model checkpoint")
    runner = runner_factory()
    print("initializing regridder")
    regridder = regridder_factory()
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

    layout = chunk_layout if chunk_layout is not None else settings.DEFAULT_CHUNK_LAYOUT
    if layout is not None:
        ds = layout(ds)

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
        ic.delete_ic_dates(date, ic_dir)
    return True
