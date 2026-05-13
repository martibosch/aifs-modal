"""AIFS on Modal."""

from aifs_modal import settings
from aifs_modal._app import (
    app,
    ingest_era5_arco,
    ingest_ifs_arraylake,
    run_forecast,
)
from aifs_modal.utils import forecast_exists

__all__ = [
    "app",
    "forecast_exists",
    "ingest_era5_arco",
    "ingest_ifs_arraylake",
    "run_forecast",
    "settings",
]
