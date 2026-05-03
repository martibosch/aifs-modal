"""AIFS on Modal."""

from aifs_modal import settings
from aifs_modal._app import (
    app,
    ingest_era5_arco,
    ingest_ifs_arraylake,
    run_forecast,
)

__all__ = [
    "app",
    "ingest_era5_arco",
    "ingest_ifs_arraylake",
    "run_forecast",
    "settings",
]
