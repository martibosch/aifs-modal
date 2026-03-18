[![PyPI version fury.io](https://badge.fury.io/py/aifs-modal.svg)](https://pypi.python.org/pypi/aifs-modal)
[![Documentation Status](https://readthedocs.org/projects/aifs-modal/badge/?version=latest)](https://aifs-modal.readthedocs.io/en/latest/?badge=latest)
[![CI/CD](https://github.com/martibosch/aifs-modal/actions/workflows/tests.yml/badge.svg)](https://github.com/martibosch/aifs-modal/blob/main/.github/workflows/tests.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/aifs-modal/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/aifs-modal/main)
[![codecov](https://codecov.io/gh/martibosch/aifs-modal/branch/main/graph/badge.svg?token=hKoSSRn58a)](https://codecov.io/gh/martibosch/aifs-modal)
[![GitHub license](https://img.shields.io/github/license/martibosch/aifs-modal.svg)](https://github.com/martibosch/aifs-modal/blob/main/LICENSE)

# AIFS on Modal

`aifs-modal` is a Python library for running *serverless* [AIFS](https://www.ecmwf.int/en/forecasts/documentation-and-support/aifs) weather forecasts on [Modal](https://modal.com). Only the AIFS inference step runs on a Modal GPU: initial conditions ingestion, post-processing, and visualization all run locally without a GPU. Forecast outputs are stored in [Icechunk](https://icechunk.io) repositories on S3-compatible object storage (or [Earthmover ArrayLake](https://earthmover.io)), giving them a git-like version history you can open with xarray from anywhere.

## Features

- **Three forecast modes**: deterministic (AIFS-Single), sequential ensemble (AIFS-ENS, one GPU), and parallel ensemble (one GPU per member using cooperative distributed writes).
- **Versioned array storage via [Icechunk](https://icechunk.io)**: forecast outputs are stored with git-like branching and commit history, so you can reproduce any past run and safely extend ensemble experiments. All [storage backends supported by Icechunk](https://icechunk.io/en/stable/storage/) are available (Tigris, AWS S3, Cloudflare R2, Google Cloud Storage, Azure Blob Storage, Earthmover ArrayLake).
- **Task-based pipeline**: the ingestion of initial conditions and forecast inference are framed as steps of a computational pipeline, so they are skipped if its outputs already exist. Therefore, *re-running a notebook cell never duplicates work or wastes GPU resources*.
- **Reproducible ensembles**: each member is seeded by its index (`torch.manual_seed(member_id)`), so you can safely extend an ensemble run later by increasing `n_members`.

![AIFS-Single 96-hour forecast over Europe](https://raw.githubusercontent.com/martibosch/aifs-modal/main/docs/figures/forecast-europe.png)
*AIFS-Single deterministic 96-hour forecast of 2-meter temperature over Europe.*

![AIFS-ENS ensemble forecast for Lausanne](https://raw.githubusercontent.com/martibosch/aifs-modal/main/docs/figures/ensemble-forecast-lausanne.png)
*AIFS-ENS 10-member ensemble forecast of 2-meter temperature for Lausanne, verified against MeteoSwiss station data.*

## Usage

```python
import datetime

import icechunk
import xarray as xr

from aifs_modal._app import app, run_forecast
from aifs_modal import ingest

date = datetime.datetime(2025, 6, 20, 0, tzinfo=datetime.UTC)
storage_bucket = "my-bucket"

# 1. Ingest initial conditions from ECMWF Open Data (local, CPU)
ingest.ingest(
    date.isoformat(),
    date.isoformat(),
    storage_bucket,
    initial_conditions_prefix="aifs-ics",
)

# 2. Run a 96-hour forecast on Modal (GPU, ~$0.05)
with app.run():
    run_forecast.remote(
        date,
        storage_bucket,
        initial_conditions_prefix="aifs-ics",
        outputs_prefix="aifs-outputs",
        lead_time=96,
    )

# 3. Open the results locally with xarray
repo = icechunk.Repository.open(...)
ds = xr.open_dataset(
    repo.readonly_session("main").store,
    group="2025-06-20/00z",
    engine="zarr",
    zarr_format=3,
)
ds
```

```
<xarray.Dataset>
Dimensions:     (valid_time: 16, lat: 721, lon: 1440)
Coordinates:
  * valid_time  (valid_time) datetime64[ns] 2025-06-20T06:00:00 ... 2025-06-24T00:00:00
  * lat         (lat) float64 90.0 89.75 89.5 ... -89.75 -90.0
  * lon         (lon) float64 0.0 0.25 0.5 ... 359.5 359.75
Data variables:
    10u         (valid_time, lat, lon) float32 ...
    10v         (valid_time, lat, lon) float32 ...
    2d          (valid_time, lat, lon) float32 ...
    2t          (valid_time, lat, lon) float32 ...
    msl         (valid_time, lat, lon) float32 ...
    skt         (valid_time, lat, lon) float32 ...
    sp          (valid_time, lat, lon) float32 ...
    tcw         (valid_time, lat, lon) float32 ...
    ...
```

See the [user guide](https://aifs-modal.readthedocs.io) for more example applications (heatwave reforecasting, jet-stream free runs).

## Requirements

1. A [Modal](https://modal.com) account. The Starter plan gives you [$30/month in free credits](https://modal.com/pricing) — a 96-hour forecast costs roughly $0.05, so that's about 600 forecasts for free.
2. An object-storage bucket for Icechunk outputs. [Tigris](https://www.tigrisdata.com) is the recommended default: the [free tier](https://www.tigrisdata.com/pricing) includes 5 GB/month.

## Installation

```bash
pip install aifs-modal
```

## Development

Install [pixi](https://pixi.sh/latest) and use the generated environments and tasks:

```bash
pixi install -e dev
pixi run -e test test
pixi run -e doc build-doc
```

Commit the generated `pixi.lock` file once you have resolved the environments for your project, and update it whenever dependencies change.

## Acknowledgements

- This package was created with the [martibosch/cookiecutter-pypixi-package](https://github.com/martibosch/cookiecutter-pypixi-package) project template.
