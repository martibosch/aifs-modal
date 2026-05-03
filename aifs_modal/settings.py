"""Settings."""

# AIFS checkpoints
AIFS_SINGLE_CHECKPOINT = {"huggingface": "ecmwf/aifs-single-1.1"}
AIFS_ENS_CHECKPOINT = {"huggingface": "ecmwf/aifs-ens-1.0"}

# initial-conditions sources
IC_SOURCES = ("ifs-arraylake", "ifs-ekd", "era5-cds", "era5-arco")
DEFAULT_IC_SOURCE = "ifs-arraylake"

# default forecast args
LEAD_TIME = 96

# modal app config
GPU_TYPE = "L40S"
DATA_VOLUME_NAME = "aifs-data"
MODELS_VOLUME_NAME = "aifs-models"
IC_VOLUME_NAME = "aifs-ics"
DATA_DIR = "/data"
MODELS_DIR = "/models"
IC_DIR = "/ic"
APP_NAME = "aifs-modal"
