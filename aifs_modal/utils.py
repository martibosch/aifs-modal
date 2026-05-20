"""Utils."""

import datetime
import os

import icechunk
import xarray as xr

_STORAGE_TYPES = ("tigris", "s3", "r2", "gcs", "azure")


def apply_trajectory_chunks(ds: xr.Dataset) -> xr.Dataset:
    """Set per-variable chunk encoding for efficient trajectory queries.

    One chunk along ``init_time``; full size along ``lead_time``,
    ``ensemble_member``, and ``pressure``; 241×240 spatial tiles (≈60°×60°,
    three near-equal tiles per hemisphere). The full forecast trajectory at
    any location is a single chunk read.
    """
    preferred = {"init_time": 1, "lat": 241, "lon": 240}
    for var in ds.data_vars:
        ds[var].encoding["chunks"] = tuple(
            min(preferred.get(d, ds.sizes[d]), ds.sizes[d]) for d in ds[var].dims
        )
    return ds


def get_storage(storage_bucket: str, prefix: str | None, storage_type: str = "tigris"):
    """Create an icechunk Storage object for the given backend.

    Parameters
    ----------
    storage_bucket : str
        Bucket name (or container name for Azure).
    prefix : str or None
        Key prefix within the bucket.
    storage_type : {"tigris", "s3", "r2", "gcs", "azure"}, optional
        Storage backend. Default ``"tigris"``.

    Returns
    -------
    icechunk.Storage
    """
    if storage_type == "tigris":
        return icechunk.tigris_storage(
            bucket=storage_bucket,
            prefix=prefix,
            region=os.getenv("AWS_REGION"),
            access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
    elif storage_type == "s3":
        return icechunk.s3_storage(
            bucket=storage_bucket,
            prefix=prefix,
            from_env=True,
        )
    elif storage_type == "r2":
        return icechunk.r2_storage(
            bucket=storage_bucket,
            prefix=prefix,
            from_env=True,
        )
    elif storage_type == "gcs":
        return icechunk.gcs_storage(
            bucket=storage_bucket,
            prefix=prefix,
            from_env=True,
        )
    elif storage_type == "azure":
        return icechunk.azure_storage(
            account=os.environ["AZURE_STORAGE_ACCOUNT"],
            container=storage_bucket,
            prefix=prefix,
            access_key=os.getenv("AZURE_STORAGE_ACCESS_KEY"),
        )
    else:
        raise ValueError(
            f"Unknown storage_type: {storage_type!r}. "
            f"Must be one of: {', '.join(_STORAGE_TYPES)}"
        )


def forecast_exists(
    date: datetime.datetime,
    storage_bucket: str,
    *,
    outputs_repo: str | None = None,
    outputs_prefix: str | None = None,
    outputs_branch: str = "main",
    n_members: int | None = None,
    storage_type: str = "tigris",
) -> bool:
    """Return True if a forecast for *date* already exists in the output store.

    Matches the storage parameters of :func:`aifs_modal.run_forecast` and can
    be called locally (outside Modal) to skip spinning up an app for a forecast
    that is already complete.

    Parameters
    ----------
    date : datetime.datetime
        Initialisation date (UTC, on a 6-hourly boundary).
    storage_bucket : str
        Bucket name for the icechunk output store.
    outputs_repo : str, optional
        ArrayLake repository name. Requires ``ARRAYLAKE_API_TOKEN`` in env.
    outputs_prefix : str, optional
        Key prefix within the bucket.
    outputs_branch : str, optional
        Branch to check. Default ``"main"``.
    n_members : int, optional
        If set, returns True only when the stored forecast already has at least
        this many ensemble members. ``None`` treats any complete forecast as
        present.
    storage_type : {"tigris", "s3", "r2", "gcs", "azure"}, optional
        Storage backend. Default ``"tigris"``.

    Returns
    -------
    bool
    """
    import xarray as xr

    if outputs_repo is not None:
        import arraylake as al

        client = al.Client(token=os.environ["ARRAYLAKE_API_TOKEN"])
        repo = client.get_repo(outputs_repo, config=icechunk.RepositoryConfig.default())
    else:
        repo = icechunk.Repository.open_or_create(
            get_storage(storage_bucket, outputs_prefix, storage_type)
        )

    base_group = datetime_to_str(date)

    if outputs_branch not in repo.list_branches():
        return False

    try:
        readonly_session = repo.readonly_session(outputs_branch)
        existing = xr.open_dataset(
            readonly_session.store,
            group=base_group,
            engine="zarr",
            zarr_format=3,
            chunks=None,
        )
        return (
            n_members is None or existing.sizes.get("ensemble_member", 0) >= n_members
        )
    except Exception:
        return False


def datetime_to_str(date: datetime.datetime) -> str:
    """Convert datetime to a string."""
    assert date.tzinfo == datetime.UTC
    assert date.minute == date.second == date.microsecond == 0
    assert date.hour in [0, 6, 12, 18]
    return date.strftime("%Y-%m-%d/%Hz")
