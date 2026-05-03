"""Ingest initial conditions from the Brightband ECMWF IFS dataset on ArrayLake.

    https://app.earthmover.io/marketplace/697162921880507a6587c31b

The source is a co-aligned zarr cube with chunks per ``init_time`` on the 0.25°
regular lat/lon grid (721×1440). Unlike the other backends this module does one
batched read over the whole date window, then regrids each field to the N320
reduced Gaussian grid and writes per date.

Transformations applied on top of the source data:
- variable name mapping to the AIFS convention
- regrid from 0.25° lat/lon to N320 via :func:`ic._regrid_n320`
- geopotential unit conversion (z in m² s⁻² → gh = z / g in m)
"""

import datetime

import icechunk
import numpy as np
import xarray as xr
import zarr

from aifs_modal import ic, settings, utils
from aifs_modal.ic import _G, LEVELS, _regrid_n320
from aifs_modal.utils import _STORAGE_TYPES

# Surface / single-level vars. Brightband uses ECMWF short names.
_SFC_MAP = {
    "u10": "10u",
    "v10": "10v",
    "d2m": "2d",
    "t2m": "2t",
    "msl": "msl",
    "skt": "skt",
    "sp": "sp",
    "tcw": "tcwv",
    "tcwv": "tcwv",
}

# TODO: once static vars are merged to the main Brightband branch, read them
# there and remove source_static_branch from the ingest API.
_STATIC_BRANCH = "add-static-vars"
_STATIC_VARS = ["lsm", "z_sfc", "slor", "sdor"]
_STATIC_RENAME = {"z_sfc": "z"}  # z_sfc → z (surface orography)

_SOIL_MAP = {
    "stl1": "sot_1",
    "stl2": "sot_2",
    "swvl1": "vsw_1",
    "swvl2": "vsw_2",
}

# Pressure-level vars. Source `z` (m² s⁻²) is stored as `gh = z / g` (m).
_PL_PREFIX_MAP = {
    "u": "u",
    "v": "v",
    "t": "t",
    "q": "q",
    "w": "w",
    "z": "gh",
}


def _read_static_fields(ds_static: xr.Dataset) -> dict[str, np.ndarray]:
    """Read time-invariant surface fields from the static-vars branch dataset."""
    data = {}
    for src in _STATIC_VARS:
        arr = ds_static[src].values.astype("f4")
        while arr.ndim > 2:
            arr = arr.squeeze(axis=0)
        data[_STATIC_RENAME.get(src, src)] = _regrid_n320(arr)
    return data


def _build_rename_map(ds: xr.Dataset) -> dict[str, str]:
    rename_map = {}
    for src, tgt in _SFC_MAP.items():
        if src in ds.data_vars and "level" not in ds[src].dims:
            rename_map[src] = tgt
    if "tcw" in rename_map and "tcwv" in rename_map:
        del rename_map["tcw"]
    for src, tgt in _SOIL_MAP.items():
        if src in ds.data_vars:
            rename_map[src] = tgt
    return rename_map


def _pl_src_prefixes(ds: xr.Dataset) -> list[str]:
    return [s for s in _PL_PREFIX_MAP if s in ds.data_vars and "level" in ds[s].dims]


def _flatten_date(
    date_ds: xr.Dataset, pl_src_prefixes: list[str]
) -> dict[str, np.ndarray]:
    """One date's already-renamed cube → flat 1D name→array dict (raveled per field)."""
    data: dict[str, np.ndarray] = {}
    sfc_names = [v for v in date_ds.data_vars if v not in pl_src_prefixes]
    for name in sfc_names:
        data[name] = _regrid_n320(date_ds[name].values.astype("f4"))

    if pl_src_prefixes:
        ds_levels = date_ds["level"].values
        level_positions = [int(np.where(ds_levels == lv)[0][0]) for lv in LEVELS]
        pl_ds = date_ds[pl_src_prefixes].isel(level=level_positions)
        for src_prefix in pl_src_prefixes:
            aifs_prefix = _PL_PREFIX_MAP[src_prefix]
            arr = pl_ds[src_prefix].values.astype("f4")  # (n_levels, lat, lon)
            if aifs_prefix == "gh":
                arr = arr / _G
            for i, level in enumerate(LEVELS):
                data[f"{aifs_prefix}_{level}"] = _regrid_n320(arr[i])
    return data


def get_all_data(
    ds: xr.Dataset,
    init_time: datetime.datetime,
    static_data: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Fetch one date's variables. Per-date path used at forecast time."""
    init_time = init_time.replace(tzinfo=None)
    src = ds.sel(init_time=init_time).isel(lead_time=0)
    src = src.rename(_build_rename_map(src))
    src = src.compute()
    data = _flatten_date(src, _pl_src_prefixes(src))
    if static_data:
        data.update(static_data)
    return data


def ingest(
    start_date: str,
    end_date: str,
    storage_bucket: str,
    *,
    client,
    source_repo: str,
    source_branch: str = "main",
    source_static_branch: str = _STATIC_BRANCH,
    initial_conditions_prefix: str | None = None,
    initial_conditions_branch: str = "main",
    storage_type: str = "tigris",
) -> None:
    """Ingest Brightband IFS initial conditions into a target icechunk store.

    Parameters
    ----------
    client : arraylake.Client
        Authenticated ArrayLake client.
    source_repo : str
        ArrayLake repository name (e.g. ``"brightband/ecmwf-ifs-initial-conditions"``).
    source_branch : str, optional
        Branch in the source repository. Default ``"main"``.
    source_static_branch : str, optional
        Branch that carries the time-invariant surface fields (``lsm``, ``z``,
        ``slor``, ``sdor``). Default ``"add-static-vars"``.
    """
    from aifs_modal.ic import _iter_dates_6h, _parse_utc_date

    start = _parse_utc_date(start_date)
    end = _parse_utc_date(end_date)
    if end < start:
        raise ValueError(
            f"end_date must be >= start_date (got {start_date!r} -> {end_date!r})"
        )
    if storage_type not in _STORAGE_TYPES:
        raise ValueError(
            f"Unknown storage_type: {storage_type!r}. "
            f"Must be one of: {', '.join(_STORAGE_TYPES)}"
        )
    if initial_conditions_prefix is None:
        initial_conditions_prefix = settings.DEFAULT_IC_PREFIXES["ifs-arraylake"]

    al_repo = client.get_repo(source_repo)
    source_session = al_repo.readonly_session(source_branch)
    source_ds = xr.open_dataset(
        source_session.store, engine="zarr", zarr_format=3, chunks={}
    )
    static_session = al_repo.readonly_session(source_static_branch)
    static_ds = xr.open_dataset(
        static_session.store, engine="zarr", zarr_format=3, chunks={}
    )
    static_data = _read_static_fields(static_ds)

    storage = utils.get_storage(storage_bucket, initial_conditions_prefix, storage_type)
    repo = icechunk.Repository.open_or_create(storage)
    ic._ensure_source_stamp(repo, initial_conditions_branch, "ifs-arraylake")

    readonly_session = repo.readonly_session(initial_conditions_branch)
    dates = list(_iter_dates_6h(start, end))
    missing: list[datetime.datetime] = []
    for date in dates:
        group_name = utils.datetime_to_str(date)
        try:
            zarr.open_group(
                readonly_session.store, path=group_name, mode="r", zarr_format=3
            )
        except zarr.errors.GroupNotFoundError:
            missing.append(date)

    if not missing:
        print(
            f"ifs-arraylake initial conditions from {start.isoformat()} "
            f"to {end.isoformat()} already present; skipping"
        )
        return

    # batch read: one .sel() over all missing dates collapses the per-date dask
    # graph rebuild that was the main cost of the previous per-date pattern.
    print(f"batch-reading {len(missing)} dates from arraylake")
    init_times = np.array(
        [d.replace(tzinfo=None) for d in missing], dtype="datetime64[ns]"
    )
    src = source_ds.sel(init_time=init_times).isel(lead_time=0)
    src = src.rename(_build_rename_map(src))
    pl_src_prefixes = _pl_src_prefixes(src)

    print("computing source slice")
    src = src.compute()

    session = repo.writable_session(initial_conditions_branch)
    for i, date in enumerate(missing, start=1):
        print(f"[{i}/{len(missing)}] writing {date.isoformat()}")
        date_ds = src.sel(init_time=np.datetime64(date.replace(tzinfo=None), "ns"))
        data = _flatten_date(date_ds, pl_src_prefixes)
        data.update(static_data)
        ic.get_and_store_date(date, session, lambda _d, _data=data: _data)

    commit_msg = (
        f"Wrote ifs-arraylake initial conditions from {start.isoformat()} "
        f"to {end.isoformat()} ({len(missing)}/{len(dates)} new dates)"
    )
    session.commit(commit_msg)
    print(commit_msg)
