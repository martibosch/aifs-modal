"""Utils."""

import datetime
import os

import icechunk

_STORAGE_TYPES = ("tigris", "s3", "r2", "gcs", "azure")


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


def datetime_to_str(date: datetime.datetime) -> str:
    """Convert datetime to a string."""
    assert date.tzinfo == datetime.UTC
    assert date.minute == date.second == date.microsecond == 0
    assert date.hour in [0, 6, 12, 18]
    return date.strftime("%Y-%m-%d/%Hz")
