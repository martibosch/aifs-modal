[![PyPI version fury.io](https://badge.fury.io/py/aifs-modal.svg)](https://pypi.python.org/pypi/aifs-modal)
[![Documentation Status](https://readthedocs.org/projects/aifs-modal/badge/?version=latest)](https://aifs-modal.readthedocs.io/en/latest/?badge=latest)
[![CI/CD](https://github.com/martibosch/aifs-modal/actions/workflows/tests.yml/badge.svg)](https://github.com/martibosch/aifs-modal/blob/main/.github/workflows/tests.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/aifs-modal/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/aifs-modal/main)
[![codecov](https://codecov.io/gh/martibosch/aifs-modal/branch/main/graph/badge.svg?token=hKoSSRn58a)](https://codecov.io/gh/martibosch/aifs-modal)
[![GitHub license](https://img.shields.io/github/license/martibosch/aifs-modal.svg)](https://github.com/martibosch/aifs-modal/blob/main/LICENSE)

# AIFS on Modal

`aifs-modal` is a Python library for running *serverless* [AIFS](https://www.ecmwf.int/en/forecasts/documentation-and-support/aifs) weather forecasts on [Modal](https://modal.com). Only AIFS inference runs on a Modal GPU; initial-conditions ingestion is dispatched to co-located Modal CPU workers and cached on a Modal Volume. Forecast outputs are stored in [Icechunk](https://icechunk.io) repositories on S3-compatible object storage (or [Earthmover ArrayLake](https://earthmover.io)), giving them a git-like version history you can open with xarray from anywhere.

## Features

- **Three forecast modes**: deterministic (AIFS-Single), sequential ensemble (AIFS-ENS, one GPU), and parallel ensemble (one GPU per member using cooperative distributed writes).
- **Versioned array storage via [Icechunk](https://icechunk.io)**: forecast outputs are stored with git-like branching and commit history, so you can reproduce any past run and safely extend ensemble experiments. All [storage backends supported by Icechunk](https://icechunk.io/en/stable/storage/) are available (Tigris, AWS S3, Cloudflare R2, Google Cloud Storage, Azure Blob Storage, Earthmover ArrayLake).
- **Task-based pipeline**: ingestion and inference are framed as pipeline steps, skipped when outputs already exist. Re-running a notebook cell never duplicates work or wastes GPU resources.
- **Reproducible ensembles**: each member is seeded by its index (`torch.manual_seed(member_id)`), so you can safely extend an ensemble run later by increasing `n_members`.

![AIFS-Single 96-hour forecast over Europe](https://raw.githubusercontent.com/martibosch/aifs-modal/main/docs/figures/forecast-europe.png)
*AIFS-Single deterministic 96-hour forecast of 2-meter temperature over Europe.*

![AIFS-ENS ensemble forecast for Lausanne](https://raw.githubusercontent.com/martibosch/aifs-modal/main/docs/figures/ensemble-forecast-lausanne.png)
*AIFS-ENS 10-member ensemble forecast of 2-meter temperature for Lausanne, verified against MeteoSwiss station data.*

## Why aifs-modal?

Two publicly accessible AIFS archives already exist: the [ECMWF operational archive](https://www.ecmwf.int/en/forecasts/dataset/operational-archive) and [dynamical.org](https://dynamical.org/catalog/ecmwf-aifs-single-forecast/). `aifs-modal` fills the gap for cases they don't cover:

1. **Extended lead times** — operational forecasts top out at 10–15 days; `aifs-modal` runs to any lead time. See the [jet-stream free-run notebook](https://aifs-modal.readthedocs.io/en/latest/user-guide/jet-stream-free-run.html): a 105-day NH jet-stream tracking run from ERA5 initial conditions.

2. **Retrospective reforecasts from ERA5** — ERA5 spans 80+ years of reanalysis, enabling ensemble reforecasts for any past date regardless of operational archive retention. See the [heatwave reforecast notebook](https://aifs-modal.readthedocs.io/en/latest/user-guide/heatwave-reforecast-ens.html): a 10-member AIFS-ENS reforecast of the June 2025 European heatwave verified against station observations.

3. **Custom initial conditions** — ingestion and inference are decoupled, so any field array following the AIFS variable convention can serve as initial conditions. Applications include perturbed or bias-corrected states from climate-model scenarios. See the [CMIP6 SST-patch notebook](https://aifs-modal.readthedocs.io/en/latest/user-guide/cmip6-sst-patch.html): patching ERA5 sea-surface temperatures to mimic a warmer-climate storyline before running the forecast.

## Usage

Before running, complete the [setup steps](https://aifs-modal.readthedocs.io/en/latest/user-guide/index.html) (Modal account, storage bucket, IC-source credentials, and `pip install aifs-modal`). Then a forecast runs in a few lines:

```python
import datetime
import xarray as xr
from aifs_modal import app, run_forecast

date = datetime.datetime(2025, 6, 20, 0, tzinfo=datetime.UTC)

# 1. Run a 96-hour forecast on Modal — ICs are ingested automatically (~$0.05 GPU cost)
with app.run():
    run_forecast.remote(
        date,
        storage_bucket="my-tigris-bucket",
        source_repo="org/ecmwf-ifs-hres-ics-open",  # Brightband IFS dataset
        outputs_prefix="aifs-outputs",
        lead_time=96,
    )

# 2. Open the results locally with xarray — no GPU needed
import icechunk
repo = icechunk.Repository.open(
    icechunk.tigris_storage(bucket="my-tigris-bucket", prefix="aifs-outputs", ...)
)
ds = xr.open_dataset(
    repo.readonly_session("main").store,
    group="2025-06-20/00z",
    engine="zarr",
    zarr_format=3,
)
```

See the [user guide](https://aifs-modal.readthedocs.io/en/latest/user-guide/your-first-forecast.html) for a full worked example including ensemble forecasts and output visualization.

## Requirements

A Modal account, an S3-compatible storage bucket, and credentials for your chosen initial-conditions source. See the [setup guide](https://aifs-modal.readthedocs.io/en/latest/user-guide/index.html) for step-by-step instructions.

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
