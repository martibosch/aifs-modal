"""Ingestion utils."""

import datetime

import earthkit.data as ekd
import earthkit.regrid as ekr
import icechunk
import numpy as np
import zarr
from earthkit.data import config

from aifs_modal import utils
from aifs_modal.utils import _STORAGE_TYPES

config.set("cache-policy", "off")

# constants
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
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def get_data_for_date(
    date: datetime.datetime, param: str, levelist: list[int] | None = None
) -> dict[str, np.ndarray]:
    """Get data for a given date and parameter."""
    if levelist is None:
        levelist = []
    fields = {}
    try:
        arg = "ecmwf-open-data"
        kwargs = dict(date=date, param=param, levelist=levelist, source="aws")
        data = ekd.from_source(arg, **kwargs)
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to fetch data {arg} {kwargs}") from e
    for f in data:
        # Open data is between -180 and 180, we need to shift it to 0-360
        array = f.to_numpy(dtype="float32")  # no need for 64 bit precision
        assert array.shape == (721, 1440)
        values = np.roll(array, -array.shape[1] // 2, axis=1)
        # Interpolate the data to from 0.25 to N320
        values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
        # no need for 64-bit precision
        values = values.astype("f4")
        name = (
            f"{f.metadata('param')}_{f.metadata('levelist')}"
            if levelist
            else f.metadata("param")
        )
        fields[name] = values
    return fields


def get_all_data(date: datetime.datetime) -> dict[str, np.ndarray]:
    """Get all parameters for a given date."""
    data_dict = {}
    for param in PARAM_SFC:
        data_dict.update(get_data_for_date(date, param))
    for param in PARAM_SOIL:
        data_dict.update(get_data_for_date(date, param, SOIL_LEVELS))
    for param in PARAM_PL:
        data_dict.update(get_data_for_date(date, param, LEVELS))
    return data_dict


def stack_fields(data_dict: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Turn many numpy arrays into a single numpy array."""
    # merge the dicts into a single dict
    names = list(data_dict.keys())
    arrays = [data_dict[name] for name in names]
    shape = arrays[0].shape

    assert len(shape) == 1
    assert all(v.shape == shape for v in arrays)
    # stack the arrays into a single array with a new dimension
    stacked = np.stack(arrays, axis=0)
    return names, stacked


def store_data(group: zarr.Group, variable_names: list[str], data: np.ndarray):
    """Store given data into a zarr group."""
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
    date: datetime.datetime, session: icechunk.Session
) -> icechunk.Session:
    """Get and store data for a given date."""
    store = session.store
    group_name = utils.datetime_to_str(date)
    group = zarr.group(store=store, path=group_name, overwrite=True)

    data_dict = get_all_data(date)
    names, stacked = stack_fields(data_dict)

    store_data(group, names, stacked)
    return session


def ensure_date_ingested(
    date: datetime.datetime,
    repo: icechunk.Repository,
    branch: str = "main",
) -> None:
    """Ensure that data for a given date has been ingested and committed."""
    group_name = utils.datetime_to_str(date)

    # read only session first in case the data has already been ingested
    readonly_session = repo.readonly_session(branch)
    try:
        zarr.open_group(
            readonly_session.store, path=group_name, mode="r", zarr_format=3
        )
    except zarr.errors.GroupNotFoundError:
        print(
            "Initial conditions missing for "
            f"{date.isoformat()} (group: {group_name}); ingesting now"
        )
    else:
        print(
            "Initial conditions already ingested for "
            f"{date.isoformat()} (group: {group_name}); skipping"
        )
        return

    # the data has not been ingested so we need to ingest it (thus a writable session)
    writable_session = repo.writable_session(branch)
    get_and_store_date(date, writable_session)
    # a) mega safety-first approach to committing the changes to avoid conflicts
    # try:
    #     writable_session.commit(commit_msg)
    # except icechunk.ConflictError:
    #     # if another writer committed in the meantime, treat this as success only when
    #     # the target group exists in the latest branch snapshot
    #     refreshed_readonly_session = repo.readonly_session(branch)
    #     try:
    #         zarr.open_group(
    #             refreshed_readonly_session.store,
    #             path=group_name,
    #             mode="r",
    #             zarr_format=3,
    #         )
    #     except zarr.errors.GroupNotFoundError:
    #         raise
    #     print(
    #         "Concurrent commit detected while ingesting "
    #         f"{date.isoformat()} (group: {group_name}); group exists now, skipping"
    #     )
    # else:
    #     print(commit_msg)
    # b) assume that in this case there is no possible concurrency
    commit_msg = f"Ingested initial conditions for {date.isoformat()}"
    writable_session.commit(commit_msg)
    print(commit_msg)


def fetch_initial_conditions(
    date: datetime.datetime, session: icechunk.Session
) -> dict[str, np.ndarray]:
    """Fetch initial conditions for a given date."""
    group_prev = zarr.open_group(
        session.store,
        zarr_format=3,
        path=utils.datetime_to_str(date - datetime.timedelta(hours=6)),
        mode="r",
    )
    group_curr = zarr.open_group(
        session.store, zarr_format=3, path=utils.datetime_to_str(date), mode="r"
    )

    vnames_curr = group_curr["variable"][:]
    vnames_prev = group_prev["variable"][:]
    np.testing.assert_equal(vnames_curr, vnames_prev)

    fields_prev = group_prev["fields"][:]
    fields_curr = group_curr["fields"][:]
    data = np.stack([fields_prev, fields_curr], axis=1)

    # tweak data to conform with AIFS input format
    mapping = {
        "sot_1": "stl1",
        "sot_2": "stl2",
        "vsw_1": "swvl1",
        "vsw_2": "swvl2",
        "tcwv": "tcw",
    }

    def maybe_rename_vname(vname):
        if vname in mapping:
            return mapping[vname]
        return vname

    fields = {
        maybe_rename_vname(vnames_curr[n]): data[n] for n in range(len(vnames_curr))
    }

    # convert to geopotential height
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665

    return fields


def _parse_utc_date(date_str: str) -> datetime.datetime:
    date = datetime.datetime.fromisoformat(date_str)
    if date.tzinfo is None:
        date = date.replace(tzinfo=datetime.UTC)
    else:
        date = date.astimezone(datetime.UTC)
    if date.minute != 0 or date.second != 0 or date.microsecond != 0:
        msg = f"Date must be hourly with no minutes/seconds: {date_str!r}"
        raise ValueError(msg)
    if date.hour not in [0, 6, 12, 18]:
        msg = f"Date hour must be one of 00, 06, 12 or 18 UTC: {date_str!r}"
        raise ValueError(msg)
    return date


def _iter_dates_6h(start_date: datetime.datetime, end_date: datetime.datetime):
    step = datetime.timedelta(hours=6)
    date = start_date
    while date <= end_date:
        yield date
        date += step


def ingest(
    start_date: str,
    end_date: str,
    storage_bucket: str,
    *,
    initial_conditions_prefix: str = "aifs-initial-conditions",
    initial_conditions_branch: str = "main",
    storage_type: str = "tigris",
) -> None:
    """Ingest initial conditions locally into an Icechunk-backed object store.

    Parameters
    ----------
    start_date : str
        ISO-8601 datetime string for the first analysis time to ingest. Must be
        a whole hour at 00, 06, 12, or 18 UTC (e.g. ``"2025-06-20T00:00:00"``).
    end_date : str
        ISO-8601 datetime string for the last analysis time to ingest (inclusive).
    storage_bucket : str
        Bucket (or container) name in the target object store.
    initial_conditions_prefix : str, optional
        Key prefix within ``storage_bucket`` for the initial conditions icechunk
        repository. Default ``"aifs-initial-conditions"``.
    initial_conditions_branch : str, optional
        Branch name in the initial conditions repository. Default ``"main"``.
    storage_type : {"tigris", "s3", "r2", "gcs", "azure"}, optional
        Object-storage backend. Credentials are read from standard environment
        variables for each backend (e.g. ``AWS_ACCESS_KEY_ID`` /
        ``AWS_SECRET_ACCESS_KEY`` for S3-compatible backends,
        ``AZURE_STORAGE_ACCOUNT`` / ``AZURE_STORAGE_ACCESS_KEY`` for Azure).
        Default ``"tigris"``.
    """
    start = _parse_utc_date(start_date)
    end = _parse_utc_date(end_date)
    if end < start:
        msg = f"end_date must be >= start_date (got {start_date!r} -> {end_date!r})"
        raise ValueError(msg)
    if storage_type not in _STORAGE_TYPES:
        raise ValueError(
            f"Unknown storage_type: {storage_type!r}. "
            f"Must be one of: {', '.join(_STORAGE_TYPES)}"
        )

    storage = utils.get_storage(storage_bucket, initial_conditions_prefix, storage_type)
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session(initial_conditions_branch)

    dates = list(_iter_dates_6h(start, end))
    for i, date in enumerate(dates, start=1):
        print(f"[{i}/{len(dates)}] ingesting {date.isoformat()}")
        get_and_store_date(date, session)

    commit_msg = (
        f"Wrote initial conditions from {start.isoformat()} to {end.isoformat()}"
    )
    session.commit(commit_msg)
    print(commit_msg)
