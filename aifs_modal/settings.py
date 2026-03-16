"""Settings."""

# AIFS checkpoints
AIFS_SINGLE_CHECKPOINT = {"huggingface": "ecmwf/aifs-single-1.1"}
AIFS_ENS_CHECKPOINT = {"huggingface": "ecmwf/aifs-ens-1.0"}

# modal app config
GPU_TYPE = "L40S"
DATA_VOLUME_NAME = "aifs-data"
MODELS_VOLUME_NAME = "aifs-models"
DATA_DIR = "/data"
MODELS_DIR = "/models"
APP_NAME = "aifs-modal"
