"""Initial-conditions store: shared constants, fetch, and ingest helpers.

All ingest backends produce ``dict[str, np.ndarray]`` for one date and hand it
to :func:`get_and_store_date`, which writes a uniform zarr group into an
icechunk session.  :func:`ingest_range` wraps the per-date loop, repo open,
and commit so each backend module is just a variable map + ``get_all_data``.
"""

import datetime
from typing import Callable

import earthkit.regrid as ekr
import icechunk
import numpy as np
import zarr
from earthkit.data import config

from aifs_modal import utils
from aifs_modal.utils import _STORAGE_TYPES

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
    session: icechunk.Session,
    fetch_fn: FetchFn,
) -> icechunk.Session:
    """Fetch one date with *fetch_fn* and write its zarr group into *session*."""
    group = zarr.group(
        store=session.store,
        path=utils.datetime_to_str(date),
        overwrite=True,
    )
    data_dict = fetch_fn(date)
    names, stacked = _stack_fields(data_dict)
    _store_data(group, names, stacked)
    return session


_SOURCE_ATTR = "aifs_ic_source"


def _read_source_stamp(repo: icechunk.Repository, branch: str) -> str | None:
    """Return the source label stamped on *repo*, or None if unstamped."""
    try:
        root = zarr.open_group(
            repo.readonly_session(branch).store, mode="r", zarr_format=3
        )
    except zarr.errors.GroupNotFoundError:
        return None
    return root.attrs.get(_SOURCE_ATTR)


def _ensure_source_stamp(repo: icechunk.Repository, branch: str, source: str) -> None:
    """Stamp *repo* with *source* on first use; reject mismatched sources.

    The stamp is a single attribute on the zarr root group, so different IC
    sources cannot silently share a repo (different content for the same date
    would otherwise mix).
    """
    existing = _read_source_stamp(repo, branch)
    if existing is None:
        ws = repo.writable_session(branch)
        root = zarr.group(store=ws.store, zarr_format=3)
        root.attrs[_SOURCE_ATTR] = source
        ws.commit(f"aifs-modal: stamp IC repo source={source}")
        return
    if existing != source:
        raise ValueError(
            f"IC repo is stamped with source={existing!r}; refusing to write "
            f"source={source!r}. Use a different initial_conditions_prefix."
        )


def ensure_date_ingested(
    date: datetime.datetime,
    repo: icechunk.Repository,
    fetch_fn: FetchFn,
    branch: str = "main",
    *,
    source: str,
) -> None:
    """Ensure initial conditions for *date* are committed to *repo*.

    If the zarr group for *date* already exists, this is a no-op. Otherwise
    *fetch_fn* is called to produce the data, which is stored and committed.
    Repo is stamped with *source* on first use; mismatched sources are rejected.
    """
    _ensure_source_stamp(repo, branch, source)

    group_name = utils.datetime_to_str(date)
    readonly_session = repo.readonly_session(branch)
    try:
        zarr.open_group(
            readonly_session.store, path=group_name, mode="r", zarr_format=3
        )
    except zarr.errors.GroupNotFoundError:
        print(
            f"Initial conditions missing for {date.isoformat()} "
            f"(group: {group_name}); ingesting now"
        )
    else:
        print(
            f"Initial conditions already ingested for {date.isoformat()} "
            f"(group: {group_name}); skipping"
        )
        return

    writable_session = repo.writable_session(branch)
    get_and_store_date(date, writable_session, fetch_fn)
    commit_msg = f"Ingested initial conditions for {date.isoformat()}"
    writable_session.commit(commit_msg)
    print(commit_msg)


def fetch_initial_conditions(
    date: datetime.datetime, session: icechunk.Session
) -> dict[str, np.ndarray]:
    """Fetch initial conditions for *date* (and *date*−6h) from an icechunk session."""
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


def ingest_range(
    start_date: str,
    end_date: str,
    storage_bucket: str,
    fetch_fn: FetchFn,
    *,
    source: str,
    initial_conditions_prefix: str | None = None,
    initial_conditions_branch: str = "main",
    storage_type: str = "tigris",
) -> None:
    """Loop over 6-hourly dates in [start, end], fetch each, commit at the end.

    Parameters
    ----------
    fetch_fn : callable(date) -> dict[str, ndarray]
        Source-specific fetcher returning one date's variables.
    source : str
        Canonical source label (one of :data:`settings.IC_SOURCES`). Used to
        stamp the IC repo and reject mismatched future writes. Also used to
        resolve the default ``initial_conditions_prefix``.
    initial_conditions_prefix : str, optional
        Defaults to :data:`settings.DEFAULT_IC_PREFIXES` ``[source]``.
    """
    from aifs_modal import settings

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
        initial_conditions_prefix = settings.DEFAULT_IC_PREFIXES[source]

    storage = utils.get_storage(storage_bucket, initial_conditions_prefix, storage_type)
    repo = icechunk.Repository.open_or_create(storage)
    _ensure_source_stamp(repo, initial_conditions_branch, source)
    session = repo.writable_session(initial_conditions_branch)

    readonly_session = repo.readonly_session(initial_conditions_branch)
    dates = list(_iter_dates_6h(start, end))
    ingested_dates = []
    for i, date in enumerate(dates, start=1):
        group_name = utils.datetime_to_str(date)
        try:
            zarr.open_group(
                readonly_session.store, path=group_name, mode="r", zarr_format=3
            )
        except zarr.errors.GroupNotFoundError:
            print(f"[{i}/{len(dates)}] ingesting {source} {date.isoformat()}")
        else:
            print(
                f"[{i}/{len(dates)}] {source} already ingested for "
                f"{date.isoformat()} (group: {group_name}); skipping"
            )
            continue
        get_and_store_date(date, session, fetch_fn)
        ingested_dates.append(date)

    if not ingested_dates:
        print(
            f"{source} initial conditions from {start.isoformat()} "
            f"to {end.isoformat()} already present; skipping"
        )
        return

    commit_msg = (
        f"Wrote {source} initial conditions from {start.isoformat()} "
        f"to {end.isoformat()} ({len(ingested_dates)}/{len(dates)} new dates)"
    )
    session.commit(commit_msg)
    print(commit_msg)


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
