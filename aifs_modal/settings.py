"""Settings."""

# AIFS checkpoints
AIFS_SINGLE_CHECKPOINT = {"huggingface": "ecmwf/aifs-single-1.1"}
AIFS_ENS_CHECKPOINT = {"huggingface": "ecmwf/aifs-ens-1.0"}

# initial-conditions sources
IC_SOURCES = ("ifs-arraylake", "ifs-ekd", "era5-cds", "era5-arco")
DEFAULT_IC_SOURCE = "ifs-arraylake"

# default storage prefix per source. Per-source so that two sources don't share
# a repo (different IC content for the same date would silently mix).
DEFAULT_IC_PREFIXES = {
    "ifs-ekd": "aifs-ics-ifs",
    "era5-cds": "aifs-ics-era5-cds",
    "era5-arco": "aifs-ics-era5-arco",
    "ifs-arraylake": "aifs-ics-ifs-arraylake",
}

# parallel ingestion: use fork/merge above this many dates
ARCO_PARALLEL_THRESHOLD = 4

# default forecast args
LEAD_TIME = 96

# modal app config
GPU_TYPE = "L40S"
DATA_VOLUME_NAME = "aifs-data"
MODELS_VOLUME_NAME = "aifs-models"
DATA_DIR = "/data"
MODELS_DIR = "/models"
APP_NAME = "aifs-modal"
