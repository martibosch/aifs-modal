# How it works

This page describes the architecture of `aifs_modal`, focusing on how AIFS forecasts are run within computational pipelines, highlighting what is executed locally versus on Modal and how data inputs and outputs are stored at each step.

Note that this set up is not intended for running an operational AIFS forecasting platform but rather as a playground to run AIFS forecasts for scientific experiments, where the pre-processing, forecasts and post-processing can be run interactively within the same local notebook, with only the forecasts running on a *serverless* GPU environment and with a task-based approach that prevents unnecessary duplication of work when re-running cells or iterating on the analysis (see the [Re-running existing forecasts and reproducibility](#re-running-existing-forecasts-and-reproducibility) section below). See the {doc}`example notebooks <user-guide/index>` for practical example applications of this set up.

## Key concepts

- **AIFS** (Artificial Intelligence Forecasting System): ECMWF's machine-learning weather model. Two checkpoints are available: [AIFS-Single](https://huggingface.co/ecmwf/aifs-single-1.1) (deterministic) and [AIFS-ENS](https://huggingface.co/ecmwf/aifs-ens-1.0) (ensemble, stochastic perturbations via random seeds).
- **Modal**: serverless GPU platform. Forecast inference runs on Modal containers with GPU access. Ingestion and orchestration also run on Modal (CPU), with workers co-located with IC sources for in-region bandwidth.
- **Icechunk**: versioned array storage engine backed by S3-compatible object storage. Used for forecast outputs, providing git-like branching and commit history.
- **Initial conditions**: analysis fields ingested from an IC source (Brightband, ARCO-ERA5, or ECMWF open data), regridded to the N320 reduced Gaussian grid and written to a shared Modal IC Volume. AIFS requires two consecutive 6-hourly analyses (t−6h and t) as input. The Volume is a temporary per-run cache; ICs are deleted after successful inference unless `keep_ics=True`.
- **IC sources**: four sources are supported — `ifs-arraylake` (default, Brightband ECMWF IFS on Earthmover ArrayLake, co-located in `us-east`), `era5-arco` (ARCO-ERA5 on GCS, co-located in `us-central1`), `ifs-ekd` (ECMWF open-data S3), and `era5-cds` (Copernicus CDS).

The diagrams below use a consistent four-zone color scheme:

| Zone           | Color  | Description                                    |
| -------------- | ------ | ---------------------------------------------- |
| Local          | grey   | Notebook or script on your machine             |
| Modal CPU      | blue   | Orchestration and IC ingestion on Modal        |
| Modal GPU      | orange | AIFS inference on Modal                        |
| Object storage | green  | IC source and Icechunk output repository on S3 |

## Deterministic forecast (`run_forecast`)

A single AIFS-Single run on one GPU. `run_forecast` is a Modal CPU function that acts as the orchestrator: it checks whether ICs are already on the IC Volume, dispatches ingestion to a co-located Modal CPU worker if they are missing, and then dispatches inference to a GPU container.

```{mermaid}
flowchart LR
    subgraph local ["Local"]
        A["configure\nrun_forecast.remote()"]
        J[read outputs]
    end
    subgraph modal_cpu ["Modal — CPU"]
        B["orchestrate\n+ ingest ICs → Volume"]
    end
    subgraph modal_gpu ["Modal — GPU"]
        G["AIFS-Single\ninference"]
    end
    subgraph storage ["Object storage"]
        IC[("IC source")]
        OUT[("Icechunk outputs")]
    end
    A --> B
    IC --> B
    B --> G
    G --> OUT
    OUT --> J

    style local fill:#555,stroke:#333,color:#fff
    style A fill:#777,stroke:#555,color:#fff
    style J fill:#777,stroke:#555,color:#fff
    style modal_cpu fill:#2c6fad,stroke:#1a4a7a,color:#fff
    style B fill:#4a90d9,stroke:#2c6fad,color:#fff
    style modal_gpu fill:#c44d1a,stroke:#8c3210,color:#fff
    style G fill:#ff6b35,stroke:#c44d1a,color:#fff
    style storage fill:#1a5c3a,stroke:#0d3d26,color:#fff
    style IC fill:#2e7d52,stroke:#1a5c3a,color:#fff
    style OUT fill:#2e7d52,stroke:#1a5c3a,color:#fff
```

**Flow:**

1. **Orchestrate** (Modal CPU, `run_forecast`): check the IC Volume for the target date. If ICs are missing, dispatch `ingest_ifs_arraylake` (Modal `us-east`) or `ingest_era5_arco` (Modal `us-central1`) and wait for completion; inline sources (`ifs-ekd`, `era5-cds`) run directly inside the orchestrator. Skipped if ICs are already present.
2. **Ingest** (Modal CPU, co-located): download and regrid the IC fields to N320; write them to the IC Volume.
3. **Inference** (Modal GPU, `run_inference`): load initial conditions from the IC Volume, run the AIFS model, regrid output fields from N320 to 0.25° lat/lon, and stream each 6-hourly step to the outputs Icechunk repository.
4. **Post-process** (local, CPU): open the outputs repository with xarray and analyze them.

## Sequential ensemble forecast (`run_forecast` with `n_members`)

Runs multiple AIFS-ENS members one after another on a single GPU. Each member uses a different random seed for stochastic perturbations. Ingestion and orchestration are identical to the deterministic case; the difference is that `run_inference` loops over members on the same GPU.

```{mermaid}
flowchart LR
    subgraph local ["Local"]
        A["configure\nrun_forecast.remote(n_members=k)"]
        J[read outputs]
    end
    subgraph modal_cpu ["Modal — CPU"]
        B["orchestrate\n+ ingest ICs → Volume"]
    end
    subgraph modal_gpu ["Modal — GPU"]
        G["AIFS-ENS\nmember 0 → 1 → … → k−1"]
    end
    subgraph storage ["Object storage"]
        IC[("IC source")]
        OUT[("Icechunk outputs")]
    end
    A --> B
    IC --> B
    B --> G
    G --> OUT
    OUT --> J

    style local fill:#555,stroke:#333,color:#fff
    style A fill:#777,stroke:#555,color:#fff
    style J fill:#777,stroke:#555,color:#fff
    style modal_cpu fill:#2c6fad,stroke:#1a4a7a,color:#fff
    style B fill:#4a90d9,stroke:#2c6fad,color:#fff
    style modal_gpu fill:#c44d1a,stroke:#8c3210,color:#fff
    style G fill:#ff6b35,stroke:#c44d1a,color:#fff
    style storage fill:#1a5c3a,stroke:#0d3d26,color:#fff
    style IC fill:#2e7d52,stroke:#1a5c3a,color:#fff
    style OUT fill:#2e7d52,stroke:#1a5c3a,color:#fff
```

Each member's output is appended along the `ensemble_member` dimension. This mode is simpler and cheaper (one GPU) but slower for large ensembles.

## Parallel ensemble forecast (`run_forecast` with `n_members` and `parallel_members=True`)

Runs all ensemble members simultaneously, each on its own GPU. The orchestration is handled inside `run_forecast` itself (running as a Modal CPU container): it pre-initialises the output arrays from checkpoint metadata, forks the icechunk session, spawns one `run_ensemble_member` container per member, collects the returned forks, and issues a single merge commit. Uses icechunk's [cooperative distributed writes](https://icechunk.io/en/stable/parallel/#distributed-writes).

```{mermaid}
flowchart LR
    subgraph local ["Local"]
        A["configure\nrun_forecast.remote(parallel_members=True)"]
        J[read outputs]
    end
    subgraph modal_cpu ["Modal — CPU"]
        B["orchestrate + ingest ICs → Volume\nfork icechunk session\nmerge forks + commit"]
    end
    subgraph modal_gpus ["Modal — GPU ×n_members"]
        G["AIFS-ENS members\n(each on own GPU)"]
    end
    subgraph storage ["Object storage"]
        IC[("IC source")]
        OUT[("Icechunk outputs")]
    end
    A --> B
    IC --> B
    B --> G
    G --> B
    B --> OUT
    OUT --> J

    style local fill:#555,stroke:#333,color:#fff
    style A fill:#777,stroke:#555,color:#fff
    style J fill:#777,stroke:#555,color:#fff
    style modal_cpu fill:#2c6fad,stroke:#1a4a7a,color:#fff
    style B fill:#4a90d9,stroke:#2c6fad,color:#fff
    style modal_gpus fill:#c44d1a,stroke:#8c3210,color:#fff
    style G fill:#ff6b35,stroke:#c44d1a,color:#fff
    style storage fill:#1a5c3a,stroke:#0d3d26,color:#fff
    style IC fill:#2e7d52,stroke:#1a5c3a,color:#fff
    style OUT fill:#2e7d52,stroke:#1a5c3a,color:#fff
```

**Flow:**

1. The **orchestrator** (`run_forecast`, Modal CPU) checks and ingests ICs as usual.
2. It introspects the checkpoint output schema on CPU (no GPU needed), builds a zero-filled template for all `n_members`, commits it as the base snapshot, and forks the session.
3. It spawns one `run_ensemble_member` container per member. Each gets a GPU, runs AIFS-ENS with `seed=member_id`, writes its output slice with `region="auto"` into the forked session, and returns the fork. No commit happens in the member.
4. The orchestrator collects all returned forks, merges them (`session.merge(*forks)`), and issues a **single commit**. No conflicts are possible.
5. Modal queues members beyond your plan's GPU limit automatically — e.g., with a 10-GPU limit, 50 members run in waves of 10.

```{note}
Parallel mode does **not** deliver a ×`n_members` wall-clock speedup over sequential mode. Container startup overhead and S3 write jitter mean that in practice the parallel ensemble is only modestly faster for moderate ensemble sizes. See the {doc}`sequential vs parallel ensemble benchmark <user-guide/a02-seq-vs-parallel-ens>` for measured timings. Additionally, note that write jitter will result in higher billing costs for the parallel mode than the sequential mode for the same lead time and number of members.
```

## Re-running existing forecasts and reproducibility

The set up of `aifs-modal` is designed to be used in computational pipelines. Accordingly a key feature is that every step checks whether its output already exists before doing any work, so interrupted or repeated runs pick up where they left off without duplicating effort or GPU cost.

| Step                                 | Skip condition                                                     | Override         |
| ------------------------------------ | ------------------------------------------------------------------ | ---------------- |
| IC ingestion in `run_forecast`       | Both IC dates already present on the IC Volume                     | —                |
| `run_forecast` (deterministic)       | Zarr group for the target date already exists on the output branch | `overwrite=True` |
| `run_forecast` (sequential ensemble) | Existing group already has ≥ `n_members` along `ensemble_member`   | `overwrite=True` |
| `run_forecast` (parallel ensemble)   | Same as above, checked before spawning any containers              | `overwrite=True` |

**Ensemble reproducibility.** Each ensemble member fixes the PyTorch random seed to its member index (`torch.manual_seed(member_id)`) before running AIFS-ENS. This means that member *k* always produces the same trajectory regardless of how many members are in the ensemble or how they are distributed across containers, making it straightforward to extend an existing ensemble run by increasing `n_members` (existing members are skipped, only the new ones are computed).
