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

import os

import numpy as np
import xarray as xr

from aifs_modal import _ic as ic
from aifs_modal import utils
from aifs_modal._ic import _G, LEVELS, _regrid_n320

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

# TODO: once static vars are merged to the main Brightband branch, remove
# source_static_branch from the ingest API.
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


def _to_eastward_0_360(ds: xr.Dataset) -> xr.Dataset:
    """Reorder a dataset to ascending longitude on ``[0, 360)``.

    ``_regrid_n320`` hands raw ``.values`` to earthkit-regrid declaring the source
    as the ECMWF global ``(0.25, 0.25)`` grid, whose first column is longitude 0°
    increasing eastward (N320 point 0 sits at lon 0°). The Brightband source is
    stored on ``[-180, 180)`` (column 0 == −180°), so passing it positionally would
    rotate every field by 180° (720 columns), shifting the whole forecast in
    longitude. Normalise here, label-based, so the array columns match earthkit's
    convention. No-op (idempotent) for data already on ``[0, 360)``.
    """
    lon_name = next((c for c in ("longitude", "lon") if c in ds.coords), None)
    if lon_name is None:
        return ds
    lon = ds[lon_name]
    if float(lon.min()) < 0:
        ds = ds.assign_coords({lon_name: lon % 360}).sortby(lon_name)
    return ds


def _read_static_fields(ds_static: xr.Dataset) -> dict[str, np.ndarray]:
    """Read time-invariant surface fields from the static-vars branch dataset."""
    ds_static = _to_eastward_0_360(ds_static)
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


def ingest(
    start_date: str,
    end_date: str,
    ic_dir: str,
    source_ds: xr.Dataset,
    static_data: dict[str, np.ndarray],
) -> None:
    """Ingest Brightband IFS initial conditions into a local zarr store.

    Reads all missing dates in a single batched ``.sel()`` to avoid
    per-date dask graph rebuilding, then writes each date individually.
    """
    source_ds = _to_eastward_0_360(source_ds)
    start = ic._parse_utc_date(start_date)
    end = ic._parse_utc_date(end_date)
    if end < start:
        raise ValueError(
            f"end_date must be >= start_date (got {start_date!r} -> {end_date!r})"
        )

    dates = list(ic._iter_dates_6h(start, end))
    missing = [
        d
        for d in dates
        if not os.path.exists(os.path.join(ic_dir, utils.datetime_to_str(d)))
    ]

    if not missing:
        print(
            f"ifs-arraylake initial conditions from {start.isoformat()} "
            f"to {end.isoformat()} already present; skipping"
        )
        return

    print(f"batch-reading {len(missing)} dates from arraylake")
    init_times = np.array(
        [d.replace(tzinfo=None) for d in missing], dtype="datetime64[ns]"
    )
    src = source_ds.sel(init_time=init_times).isel(lead_time=0)
    src = src.rename(_build_rename_map(src))
    pl_prefixes = _pl_src_prefixes(src)
    print("computing source slice")
    src = src.compute()

    os.makedirs(ic_dir, exist_ok=True)
    for i, date in enumerate(missing, start=1):
        print(f"[{i}/{len(missing)}] writing ifs-arraylake {date.isoformat()}")
        date_ds = src.sel(init_time=np.datetime64(date.replace(tzinfo=None), "ns"))
        data = _flatten_date(date_ds, pl_prefixes)
        data.update(static_data)
        ic.get_and_store_date(date, ic_dir, lambda _d, _data=data: _data)

    print(
        f"Wrote ifs-arraylake initial conditions from {start.isoformat()} "
        f"to {end.isoformat()} ({len(missing)}/{len(dates)} new dates)"
    )
