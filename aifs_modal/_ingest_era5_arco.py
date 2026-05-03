"""ERA5 initial conditions ingestion from ARCO-ERA5 (Google Cloud Storage).

Reads from the public Analysis-Ready ARCO-ERA5 zarr store on GCS:

    ``gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3``

Anonymously accessible; coverage is 1979-01-01 to present (ERA5T preliminary
back-fill, ~5-day lag).

.. note::
    The AR zarr store uses ``(time=1, latitude=721, longitude=1440)`` chunks,
    so every time-step read transfers a full ~4 MB global field regardless of
    any spatial subset.  This module is most efficient when run on GCP
    infrastructure co-located with the ``us-central1`` bucket.  For use from
    outside GCP, consider :mod:`aifs_modal._ingest_ekd` with ``cds=True``.

References
----------
https://github.com/google-research/arco-era5
"""

import datetime
import functools
from concurrent.futures import ThreadPoolExecutor

import gcsfs
import numpy as np
import zarr

from aifs_modal import _ic as ic
from aifs_modal._ic import _G, LEVELS, _regrid_n320

_ARCO_PATH = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Variable name maps: AIFS storage key → ARCO variable name
_SFC_ARCO = {
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "2d": "2m_dewpoint_temperature",
    "2t": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "skt": "skin_temperature",
    "sp": "surface_pressure",
    "tcwv": "total_column_water_vapour",
    "lsm": "land_sea_mask",
    "z": "geopotential_at_surface",
    "slor": "slope_of_sub_gridscale_orography",
    "sdor": "standard_deviation_of_orography",
}

_SOIL_ARCO = {
    "vsw_1": "volumetric_soil_water_layer_1",
    "vsw_2": "volumetric_soil_water_layer_2",
    "sot_1": "soil_temperature_level_1",
    "sot_2": "soil_temperature_level_2",
}

_PL_ARCO = {
    "gh": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
}


@functools.lru_cache(maxsize=1)
def _get_time_index(time: datetime.datetime) -> int:
    """ARCO starts at 1900-01-01 00:00 with 1-hourly resolution."""
    return int((time - datetime.datetime(1900, 1, 1)).total_seconds() // 3600)


@functools.lru_cache(maxsize=1)
def _open_arco() -> zarr.Group:
    """Open the ARCO-ERA5 zarr store (cached for the process lifetime)."""
    fs = gcsfs.GCSFileSystem(
        cache_timeout=-1,
        token="anon",
        access="read_only",
        block_size=8**20,
        skip_instance_cache=True,
    )
    zstore = zarr.storage.FsspecStore(fs, path=_ARCO_PATH)
    return zarr.open(store=zstore, mode="r")


def _get_array(
    group: zarr.Group, name: str, time_index: int, level: int | None = None
) -> np.ndarray:
    zarr_array = group[name]
    shape = zarr_array.shape
    if len(shape) == 2:
        return zarr_array[:]
    if len(shape) == 3:
        if level is None:
            return zarr_array[time_index]
        level_index = int(np.searchsorted(group["level"][:], level))
        return zarr_array[time_index, level_index]
    if len(shape) == 4:
        if level is None:
            raise ValueError(f"level required for 4D array {name}")
        level_index = int(np.searchsorted(group["level"][:], level))
        return zarr_array[time_index, level_index]
    raise ValueError(f"Unexpected array shape {shape} for {name}")


def _fetch_sfc(item, group, time_index):
    aifs_key, arco_name = item
    return aifs_key, _regrid_n320(_get_array(group, arco_name, time_index))


def _fetch_pl(item, group, time_index):
    prefix, arco_name, level = item
    array = _get_array(group, arco_name, time_index, level)
    if prefix == "gh":
        array = array / _G
    return f"{prefix}_{level}", _regrid_n320(array)


def get_all_data(date: datetime.datetime) -> dict[str, np.ndarray]:
    """Fetch all variables for *date* in one thread-pool pass and regrid to N320."""
    date = date.replace(tzinfo=None)
    time_index = _get_time_index(date)
    group = _open_arco()

    sfc_tasks = [("sfc", it) for it in {**_SFC_ARCO, **_SOIL_ARCO}.items()]
    pl_tasks = [
        ("pl", (prefix, arco_name, level))
        for prefix, arco_name in _PL_ARCO.items()
        for level in LEVELS
    ]

    def run(task):
        kind, payload = task
        if kind == "sfc":
            return _fetch_sfc(payload, group, time_index)
        return _fetch_pl(payload, group, time_index)

    data: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=32) as ex:
        for key, array in ex.map(run, sfc_tasks + pl_tasks):
            data[key] = array
    return data


def ingest(
    start_date: str,
    end_date: str,
    storage_bucket: str,
    *,
    initial_conditions_prefix: str | None = None,
    initial_conditions_branch: str = "main",
    storage_type: str = "tigris",
) -> None:
    """Ingest ARCO-ERA5 initial conditions into an icechunk store."""
    ic.ingest_range(
        start_date,
        end_date,
        storage_bucket,
        get_all_data,
        source="era5-arco",
        initial_conditions_prefix=initial_conditions_prefix,
        initial_conditions_branch=initial_conditions_branch,
        storage_type=storage_type,
    )
