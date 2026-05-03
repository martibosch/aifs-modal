"""Multi-source initial-conditions ingestion dispatcher.

Picks the right backend module based on ``source`` and forwards to its
``ingest`` function. The backends in turn call into :mod:`aifs_modal._ic`
for the shared per-date / commit machinery.
"""

import typing

from aifs_modal import settings

if typing.TYPE_CHECKING:
    import arraylake as al

    ArraylakeClientType: typing.TypeAlias = al.Client
else:
    ArraylakeClientType: typing.TypeAlias = typing.Any


def ingest(
    start_date: str,
    end_date: str,
    storage_bucket: str,
    *,
    source: str | None = None,
    arraylake_client: ArraylakeClientType | None = None,
    source_repo: str | None = None,
    source_branch: str = "main",
    source_static_branch: str = "add-static-vars",
    initial_conditions_prefix: str | None = None,
    initial_conditions_branch: str = "main",
    storage_type: str = "tigris",
) -> None:
    """Ingest AIFS initial conditions from a supported source into an icechunk store.

    Parameters
    ----------
    start_date, end_date : str
        ISO-8601 datetime strings (inclusive). Must be at 00, 06, 12, or 18 UTC.
    storage_bucket : str
        Bucket (or container) name in the target object store.
    source : str, optional
        One of :data:`aifs_modal.settings.IC_SOURCES`. Defaults to
        :data:`aifs_modal.settings.DEFAULT_IC_SOURCE`.

        ``"ifs-ekd"``
            ECMWF operational IFS analysis from the public open-data S3 archive
            (AWS), via earthkit-data. No credentials. ~2 years of coverage.
        ``"era5-cds"``
            ERA5 reanalysis from the Copernicus CDS, via earthkit-data.
            Requires a valid CDS API configuration (``~/.cdsapirc`` or
            ``CDSAPI_URL`` / ``CDSAPI_KEY``).
        ``"era5-arco"``
            ERA5 reanalysis from the public ARCO-ERA5 zarr store on GCS. No
            credentials. Best run co-located with ``us-central1``; use
            :func:`aifs_modal.ingest_era5_arco` for the Modal-side variant.
        ``"ifs-arraylake"``
            ECMWF IFS initial conditions from the Brightband dataset on the
            Earthmover ArrayLake marketplace. Requires ``arraylake_client``
            and ``source_repo``.
    arraylake_client : arraylake.Client, optional
        Required when ``source="ifs-arraylake"``.
    source_repo : str, optional
        ArrayLake repository name. Required when ``source="ifs-arraylake"``.
    source_branch : str, optional
        Branch in the ArrayLake source repository. Default ``"main"``.
    initial_conditions_prefix, initial_conditions_branch : str, optional
        Target icechunk repo prefix/branch.
    storage_type : {"tigris", "s3", "r2", "gcs", "azure"}, optional
    """
    if source is None:
        source = settings.DEFAULT_IC_SOURCE

    common = dict(
        initial_conditions_prefix=initial_conditions_prefix,
        initial_conditions_branch=initial_conditions_branch,
        storage_type=storage_type,
    )

    if source == "ifs-ekd" or source == "era5-cds":
        from aifs_modal import ingest_ekd

        ingest_ekd.ingest(start_date, end_date, storage_bucket, source, **common)

    elif source == "era5-arco":
        from aifs_modal import ingest_arco

        ingest_arco.ingest(start_date, end_date, storage_bucket, **common)

    elif source == "ifs-arraylake":
        if arraylake_client is None:
            raise ValueError("arraylake_client is required when source='ifs-arraylake'")
        if source_repo is None:
            raise ValueError("source_repo is required when source='ifs-arraylake'")
        from aifs_modal import ingest_arraylake

        ingest_arraylake.ingest(
            start_date,
            end_date,
            storage_bucket,
            client=arraylake_client,
            source_repo=source_repo,
            source_branch=source_branch,
            source_static_branch=source_static_branch,
            **common,
        )

    else:
        raise ValueError(
            f"Unknown source: {source!r}. "
            f"Must be one of: {', '.join(settings.IC_SOURCES)}"
        )
