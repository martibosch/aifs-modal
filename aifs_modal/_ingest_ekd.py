"""Ingest initial conditions via earthkit-data.

Two backends share the same per-field regrid loop:

``"ecmwf-open-data"``
    Operational IFS analysis from the public ECMWF open-data S3 archive (AWS).
    No credentials. ~2 years of coverage.
``"cds"``
    ERA5 reanalysis from the Copernicus Climate Data Store via
    :func:`earthkit.data.from_source`. Requires a valid CDS API configuration
    (``~/.cdsapirc`` or ``CDSAPI_URL``/``CDSAPI_KEY`` env vars).
"""

import datetime

import earthkit.data as ekd
import numpy as np

from aifs_modal import _ic as ic
from aifs_modal._ic import _G, LEVELS, _regrid_n320

# --- ECMWF open-data param lists (GRIB short names) ---
PARAM_SFC = [
    "10u",
    "10v",
    "2d",
    "2t",
    "msl",
    "skt",
    "sp",
    "tcwv",
    "lsm",
    "z",
    "slor",
    "sdor",
]
PARAM_SOIL = ["vsw", "sot"]
SOIL_LEVELS = [1, 2]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]

# --- CDS variable long names ---
_CDS_SFC_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "skin_temperature",
    "surface_pressure",
    "total_column_water_vapour",
    "land_sea_mask",
    "geopotential",
    "slope_of_sub_gridscale_orography",
    "standard_deviation_of_orography",
]
_CDS_SOIL_VARS = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
]
_CDS_PL_VARS = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
]
_CDS_SOIL_SHORT_TO_KEY = {
    "swvl1": "vsw_1",
    "swvl2": "vsw_2",
    "stl1": "sot_1",
    "stl2": "sot_2",
}


def _field_to_n320(field) -> np.ndarray:
    """Earthkit field → (N320,) f4, rolling -180..180 → 0..360 if needed."""
    array = field.to_numpy(dtype="float32")
    assert array.shape == (721, 1440), f"Unexpected shape: {array.shape}"
    if field.metadata("longitudeOfFirstGridPointInDegrees") < 0:
        array = np.roll(array, -array.shape[1] // 2, axis=1)
    return _regrid_n320(array)


# ---------------------------------------------------------------------------
# ECMWF open-data
# ---------------------------------------------------------------------------


def _get_open_data(
    date: datetime.datetime, param: str, levelist: list[int] | None = None
) -> dict[str, np.ndarray]:
    if levelist is None:
        levelist = []
    # ACHTUNG: need to make the date naive here
    kwargs = dict(
        date=date.replace(tzinfo=None), param=param, levelist=levelist, source="aws"
    )
    try:
        data = ekd.from_source("ecmwf-open-data", **kwargs)
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to fetch ecmwf-open-data {kwargs}") from e
    fields = {}
    for f in data:
        name = (
            f"{f.metadata('param')}_{f.metadata('levelist')}"
            if levelist
            else f.metadata("param")
        )
        fields[name] = _field_to_n320(f)
    return fields


def _get_all_open_data(date: datetime.datetime) -> dict[str, np.ndarray]:
    data_dict = {}
    for param in PARAM_SFC:
        data_dict.update(_get_open_data(date, param))
    for param in PARAM_SOIL:
        data_dict.update(_get_open_data(date, param, SOIL_LEVELS))
    for param in PARAM_PL:
        data_dict.update(_get_open_data(date, param, LEVELS))
    return data_dict


# ---------------------------------------------------------------------------
# CDS / ERA5
# ---------------------------------------------------------------------------


def _cds_request(date: datetime.datetime, **extra) -> dict:
    return {
        "product_type": "reanalysis",
        "grid": [0.25, 0.25],
        "date": date.strftime("%Y-%m-%d"),
        "time": date.strftime("%H:%M"),
        **extra,
    }


def _get_cds_sfc(date: datetime.datetime) -> dict[str, np.ndarray]:
    data = ekd.from_source(
        "cds",
        "reanalysis-era5-single-levels",
        request=_cds_request(date, variable=_CDS_SFC_VARS),
    )
    return {f.metadata("shortName"): _field_to_n320(f) for f in data}


def _get_cds_soil(date: datetime.datetime) -> dict[str, np.ndarray]:
    data = ekd.from_source(
        "cds",
        "reanalysis-era5-single-levels",
        request=_cds_request(date, variable=_CDS_SOIL_VARS),
    )
    return {
        _CDS_SOIL_SHORT_TO_KEY[f.metadata("shortName")]: _field_to_n320(f) for f in data
    }


def _get_cds_pl(date: datetime.datetime) -> dict[str, np.ndarray]:
    data = ekd.from_source(
        "cds",
        "reanalysis-era5-pressure-levels",
        request=_cds_request(
            date,
            variable=_CDS_PL_VARS,
            pressure_level=[str(lev) for lev in LEVELS],
        ),
    )
    fields = {}
    for f in data:
        short = f.metadata("shortName")
        level = int(f.metadata("level"))
        array = _field_to_n320(f)
        if short == "z":
            fields[f"gh_{level}"] = array / _G
        else:
            fields[f"{short}_{level}"] = array
    return fields


def _get_all_cds(date: datetime.datetime) -> dict[str, np.ndarray]:
    return {**_get_cds_sfc(date), **_get_cds_soil(date), **_get_cds_pl(date)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest(
    start_date: str,
    end_date: str,
    ic_dir: str,
    source: str,
) -> None:
    """Ingest IFS-open-data or ERA5/CDS initial conditions into a local zarr store."""
    if source == "ifs-ekd":
        fetch_fn = _get_all_open_data
    else:  # source == "era5-cds"
        fetch_fn = _get_all_cds

    ic._ingest_range(start_date, end_date, ic_dir, fetch_fn, source=source)
