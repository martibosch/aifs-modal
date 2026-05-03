"""AIFS on Modal."""

from aifs_modal import settings
from aifs_modal._app import (
    app,
    ingest_arco_era5,
    run_forecast,
)
from aifs_modal._ingest import ingest

__all__ = [
    "app",
    "ingest",
    "ingest_arco_era5",
    "ingest_ifs_arraylake",
    "run_forecast",
    "settings",
]
