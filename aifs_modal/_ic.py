"""Initial-conditions store: shared constants, fetch, and ingest helpers.

All ingest backends produce ``dict[str, np.ndarray]`` for one date and hand it
to :func:`get_and_store_date`, which writes a uniform zarr group under *ic_dir*.
:func:`ingest_range` wraps the per-date loop so each backend module is just a
variable map + ``get_all_data``.
"""

import datetime
import os
import shutil
from collections.abc import Callable

import earthkit.regrid as ekr
import numpy as np
import zarr
from earthkit.data import config

from aifs_modal import utils

config.set("cache-policy", "off")

_G = 9.80665  # m s⁻²

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

FetchFn = Callable[[datetime.datetime], dict[str, np.ndarray]]


def _regrid_n320(array: np.ndarray) -> np.ndarray:
    """Regrid a (721, 1440) f4 array from 0.25° lat/lon to N320."""
    assert array.shape == (721, 1440), f"Unexpected shape: {array.shape}"
    values = ekr.interpolate(
        array.astype("float32"), {"grid": (0.25, 0.25)}, {"grid": "N320"}
    )
    return values.astype("f4")


def _stack_fields(data_dict: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = list(data_dict.keys())
    arrays = [data_dict[name] for name in names]
    shape = arrays[0].shape
    assert len(shape) == 1
    assert all(v.shape == shape for v in arrays)
    return names, np.stack(arrays, axis=0)


def _store_data(group: zarr.Group, variable_names: list[str], data: np.ndarray):
    assert data.ndim == 2
    nvars = len(variable_names)
    assert data.shape[0] == nvars
    npoints = data.shape[1]
    var_array = group.create_array(
        "variable",
        dtype=str,
        shape=(nvars,),
        chunks=(nvars,),
        compressors=[],
        dimension_names=["variable"],
    )
    var_array[:] = variable_names
    data_array = group.create_array(
        "fields",
        dtype=data.dtype,
        shape=data.shape,
        chunks=(10, npoints),
        dimension_names=["variable", "point"],
    )
    data_array[:] = data


def get_and_store_date(
    date: datetime.datetime,
    ic_dir: str,
    fetch_fn: FetchFn,
) -> None:
    """Fetch one date with *fetch_fn* and write its zarr group into *ic_dir*."""
    date_path = os.path.join(ic_dir, utils.datetime_to_str(date))
    group = zarr.open_group(date_path, mode="w", zarr_format=3)
    data_dict = fetch_fn(date)
    names, stacked = _stack_fields(data_dict)
    _store_data(group, names, stacked)


def fetch_initial_conditions(
    date: datetime.datetime, ic_dir: str
) -> dict[str, np.ndarray]:
    """Fetch initial conditions for *date* (and *date*−6h) from *ic_dir*."""
    prev_path = os.path.join(
        ic_dir, utils.datetime_to_str(date - datetime.timedelta(hours=6))
    )
    curr_path = os.path.join(ic_dir, utils.datetime_to_str(date))
    group_prev = zarr.open_group(prev_path, mode="r", zarr_format=3)
    group_curr = zarr.open_group(curr_path, mode="r", zarr_format=3)

    vnames_curr = group_curr["variable"][:]
    vnames_prev = group_prev["variable"][:]
    np.testing.assert_equal(vnames_curr, vnames_prev)

    fields_prev = group_prev["fields"][:]
    fields_curr = group_curr["fields"][:]
    data = np.stack([fields_prev, fields_curr], axis=1)

    mapping = {
        "sot_1": "stl1",
        "sot_2": "stl2",
        "vsw_1": "swvl1",
        "vsw_2": "swvl2",
        "tcwv": "tcw",
    }

    fields = {
        mapping.get(vnames_curr[n], vnames_curr[n]): data[n]
        for n in range(len(vnames_curr))
    }

    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * _G

    return fields


def delete_ic_dates(date: datetime.datetime, ic_dir: str) -> None:
    """Delete IC directories for *date* and *date*−6h from *ic_dir*."""
    for d in [date - datetime.timedelta(hours=6), date]:
        path = os.path.join(ic_dir, utils.datetime_to_str(d))
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"deleted IC directory {path}")


def ingest_range(
    start_date: str,
    end_date: str,
    ic_dir: str,
    fetch_fn: FetchFn,
    *,
    source: str,
) -> None:
    """Loop over 6-hourly dates in [start, end] and write each into *ic_dir*."""
    start = _parse_utc_date(start_date)
    end = _parse_utc_date(end_date)
    if end < start:
        raise ValueError(
            f"end_date must be >= start_date (got {start_date!r} -> {end_date!r})"
        )

    os.makedirs(ic_dir, exist_ok=True)
    dates = list(_iter_dates_6h(start, end))
    ingested_dates = []
    for i, date in enumerate(dates, start=1):
        date_path = os.path.join(ic_dir, utils.datetime_to_str(date))
        if os.path.exists(date_path):
            print(
                f"[{i}/{len(dates)}] {source} already ingested for "
                f"{date.isoformat()}; skipping"
            )
            continue
        print(f"[{i}/{len(dates)}] ingesting {source} {date.isoformat()}")
        get_and_store_date(date, ic_dir, fetch_fn)
        ingested_dates.append(date)

    if not ingested_dates:
        print(
            f"{source} initial conditions from {start.isoformat()} "
            f"to {end.isoformat()} already present; skipping"
        )
        return

    print(
        f"Wrote {source} initial conditions from {start.isoformat()} "
        f"to {end.isoformat()} ({len(ingested_dates)}/{len(dates)} new dates)"
    )


def _parse_utc_date(date_str: str) -> datetime.datetime:
    date = datetime.datetime.fromisoformat(date_str)
    if date.tzinfo is None:
        date = date.replace(tzinfo=datetime.UTC)
    else:
        date = date.astimezone(datetime.UTC)
    if date.minute != 0 or date.second != 0 or date.microsecond != 0:
        raise ValueError(f"Date must be hourly with no minutes/seconds: {date_str!r}")
    if date.hour not in [0, 6, 12, 18]:
        raise ValueError(f"Date hour must be one of 00, 06, 12 or 18 UTC: {date_str!r}")
    return date


def _iter_dates_6h(start_date: datetime.datetime, end_date: datetime.datetime):
    step = datetime.timedelta(hours=6)
    date = start_date
    while date <= end_date:
        yield date
        date += step
