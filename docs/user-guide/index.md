# Getting Started

`aifs-modal` uses four external services. Set them up once in the order below, then jump to the [Your First Forecast](your-first-forecast.ipynb) notebook.

## 1. Modal

[Modal](https://modal.com) is the serverless GPU platform that runs AIFS inference. The Starter plan gives you $30/month in free credits — a 96-hour forecast costs roughly $0.05.

1. Create a free account at [modal.com](https://modal.com).

2. Install the CLI and authenticate:

   ```bash
   pip install modal
   modal setup
   ```

## 2. Forecast output storage

Forecast outputs are stored as [Icechunk](https://icechunk.io) repositories in object storage. [Tigris](https://www.tigrisdata.com) is the recommended default: it is co-located with Modal's primary region, so data transfer between them is fast and free, and the free tier includes 5 GB/month.

1. Create a free account at [tigrisdata.com](https://www.tigrisdata.com).
2. Create a bucket and generate an access key pair (access key ID + secret access key).

Any [backend supported by Icechunk](https://icechunk.io/en/stable/guides/storage/) works — AWS S3, Cloudflare R2, Google Cloud Storage, Azure Blob Storage, or Earthmover ArrayLake.

## 3. Initial conditions source

`aifs-modal` supports four IC sources, selected with the `source=` argument of `run_forecast`:

| `source`                      | Credentials                                  | Coverage           | Notes                                                                                                                              |
| ----------------------------- | -------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `"ifs-arraylake"` *(default)* | Earthmover account + Brightband subscription | ~last 15 days      | Fastest and most reliable; used in [Your First Forecast](your-first-forecast.ipynb)                                                |
| `"ifs-ekd"`                   | None                                         | ~last 2 years      | No credentials, but frequent `503 Slow Down` errors make it unreliable                                                             |
| `"era5-arco"`                 | None                                         | 1979 — ~5 days ago | ERA5 reanalysis; used in [Jet-stream free run](jet-stream-free-run.ipynb) and [Heatwave reforecast](heatwave-reforecast-ens.ipynb) |
| `"era5-cds"`                  | `~/.cdsapirc`                                | 1979 — ~5 days ago | ERA5 via Copernicus CDS; slower than `era5-arco`                                                                                   |

```{note}
ERA5 publication lags real time by about 5 days, so the ERA5 sources are suitable for hindcasts and reforecasts but not for operational use.
```

To set up the default `ifs-arraylake` source:

1. Create a free account at [app.earthmover.io](https://app.earthmover.io).
2. Subscribe to the [Brightband ECMWF IFS dataset](https://app.earthmover.io/marketplace/697162921880507a6587c31b) on the marketplace (free).
3. Copy your API token from the Earthmover dashboard.

## 4. Modal secrets

`aifs-modal` reads credentials from three named [Modal secrets](https://modal.com/docs/guide/secrets). Create them once with the CLI:

```bash
# Tigris / S3-compatible object storage (forecast outputs)
modal secret create aws-credentials \
  AWS_ACCESS_KEY_ID=<key> \
  AWS_SECRET_ACCESS_KEY=<secret> \
  AWS_REGION=auto

# Earthmover ArrayLake (only required for the ifs-arraylake IC source)
modal secret create arraylake-api-token \
  ARRAYLAKE_API_TOKEN=<token>

# HuggingFace (model weights download)
modal secret create huggingface-secret \
  HF_TOKEN=<token>
```

| Secret name           | Key(s)                                                     | Purpose                        |
| --------------------- | ---------------------------------------------------------- | ------------------------------ |
| `aws-credentials`     | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | Forecast output storage        |
| `arraylake-api-token` | `ARRAYLAKE_API_TOKEN`                                      | `ifs-arraylake` IC source      |
| `huggingface-secret`  | `HF_TOKEN`                                                 | Downloading AIFS model weights |

Secrets are stored in your Modal workspace and injected automatically into every function call — no need to re-create them.

## Notebooks

```{toctree}
---
maxdepth: 1
---

your-first-forecast
jet-stream-free-run
heatwave-reforecast-ens
cmip6-sst-patch
a01-data
a02-seq-vs-parallel-ens
a03-ingestion-benchmark
forecast-pipelines
```
