# How it works

This page describes the motivation and set up of `aifs_modal`, focusing on how AIFS forecasts are run within computational pipelines, highlighting what is executed locally (CPU) versus on Modal (GPU) and how data inputs and outputs are stored at each step.

Note that this set up is not intended for running an operational AIFS forecasting platform but rather as a playground to run AIFS forecasts for scientific experiments, where the pre-processing, forecasts and post-processing can be run interactively within the same local notebook, with only the forecasts running on a *serverless* GPU environment and with a task-based approach that prevents unnecessary duplication of work when re-running cells or iterating on the analysis (see the [Re-running existing forecasts and reproducibility](#re-running-existing-forecasts-and-reproducibility) section below). See the {doc}`example notebooks <user-guide/index>` for practical example applications of this set up.

## Key concepts

- **AIFS** (Artificial Intelligence Forecasting System): ECMWF's machine-learning weather model. Two checkpoints are available: [AIFS-Single](https://huggingface.co/ecmwf/aifs-single-1.1) (deterministic) and [AIFS-ENS](https://huggingface.co/ecmwf/aifs-ens-1.0) (ensemble, stochastic perturbations via random seeds).
- **Modal**: serverless GPU platform. Forecast inference runs on Modal containers with GPU access, while everything else (ingestion, post-processing, visualization) runs locally on your machine without a GPU.
- **Icechunk**: versioned array storage engine backed by S3-compatible object storage. Used for both initial conditions and forecast outputs, providing git-like branching and commit history.
- **Initial conditions**: analysis fields from ECMWF Open Data, regridded to the N320 reduced Gaussian grid and stored in an icechunk repository. AIFS requires two consecutive 6-hourly analyses (t-6h and t) as input.
- **ArrayLake**: Earthmover's managed icechunk service. If you have an Earthmover subscription with initial conditions already stored in an ArrayLake repository, you can skip the local ingestion step entirely and have Modal containers pull ICs directly from ArrayLake.

## Deterministic forecast (`run_forecast`)

A single AIFS-Single run on one GPU. Ingestion runs locally, inference runs on Modal, and results are read back locally for analysis.

```{mermaid}
flowchart TB
    subgraph local ["Local (CPU)"]
        A[Configure forecast<br/>start date, lead time, storage params]
        B[Ingest initial conditions<br/>ECMWF Open Data → Icechunk]
        C[Submit forecast to Modal]
        F[Read outputs & postprocess]
    end

    subgraph modal ["Modal (GPU) 🔥"]
        D[Load ICs from Icechunk]
        E[Run AIFS-Single inference<br/>stream steps → Icechunk]
    end

    subgraph storage ["S3 / Icechunk"]
        IC[(Initial conditions<br/>repository)]
        OUT[(Forecast outputs<br/>repository)]
    end

    A --> B --> C
    B --> IC
    C --> D
    IC --> D
    D --> E
    E --> OUT
    OUT --> F

    style modal fill:#ff6b35,stroke:#c44d1a,color:#fff
    style D fill:#ff8c5a,stroke:#c44d1a,color:#fff
    style E fill:#ff8c5a,stroke:#c44d1a,color:#fff
```

**Flow:**

1. **Ingest** (local, CPU): download ECMWF Open Data GRIB fields for `t-6h` and `t`, regrid to N320, and commit to the initial conditions icechunk repository. Skipped if already ingested.
2. **Submit** (local → Modal): `run_forecast.remote(...)` sends the task to Modal.
3. **Inference** (Modal, GPU): load initial conditions from icechunk, run the AIFS model, regrid output fields from N320 to 0.25° lat/lon, and stream each 6-hourly step to the outputs icechunk repository.
4. **Post-process** (local, CPU): open the outputs repository with xarray and analyze them.

## Sequential ensemble forecast (`run_forecast` with `n_members`)

Runs multiple AIFS-ENS members one after another on a single GPU. Each member uses a different random seed for stochastic perturbations.

```{mermaid}
flowchart TB
    subgraph local ["Local (CPU)"]
        A[Configure forecast<br/>start date, lead time, n_members]
        B[Ingest initial conditions]
        C[Submit forecast to Modal]
        G[Read outputs & postprocess]
    end

    subgraph modal ["Modal (1 GPU) 🔥"]
        D[Load ICs from Icechunk]
        E[Run member 0<br/>seed=0]
        E1[Run member 1<br/>seed=1]
        E2[Run member ...<br/>seed=...]
        F[Commit combined output]
    end

    subgraph storage ["S3 / Icechunk"]
        IC[(Initial conditions)]
        OUT[(Forecast outputs)]
    end

    A --> B --> C
    B --> IC
    C --> D
    IC --> D
    D --> E --> E1 --> E2 --> F
    F --> OUT
    OUT --> G

    style modal fill:#ff6b35,stroke:#c44d1a,color:#fff
    style D fill:#ff8c5a,stroke:#c44d1a,color:#fff
    style E fill:#ff8c5a,stroke:#c44d1a,color:#fff
    style E1 fill:#ff8c5a,stroke:#c44d1a,color:#fff
    style E2 fill:#ff8c5a,stroke:#c44d1a,color:#fff
    style F fill:#ff8c5a,stroke:#c44d1a,color:#fff
```

Each member's output is appended along the `ensemble_member` dimension. This mode is simpler and cheaper (one GPU) but slower for large ensembles.

## Parallel ensemble forecast (`run_ensemble_forecast`)

Runs all ensemble members simultaneously, each on its own GPU. Uses icechunk's [cooperative distributed writes](https://icechunk.io/en/stable/parallel/#distributed-writes) via session forking: a dedicated CPU container pre-initialises the output arrays from checkpoint metadata, all member containers then write their slice into a forked session and return it, and the orchestrator merges all forks and issues a single commit.

```{mermaid}
flowchart TB
    subgraph local ["Local (CPU)"]
        A[Configure forecast<br/>start date, lead time, n_members]
        B[Ingest initial conditions]
        C[Submit orchestrator to Modal]
        J[Read outputs & postprocess]
    end

    subgraph modal_orch ["Modal (orchestrator, no GPU)"]
        D[Initialize output arrays<br/>CPU - checkpoint metadata]
        E[Fork session &<br/>spawn member containers]
        H[Merge fork sessions<br/>& commit once]
    end

    subgraph modal_gpus ["Modal (1 GPU per member) 🔥"]
        F0[Member 0<br/>inference → write fork]
        F1[Member 1<br/>inference → write fork]
        FN[Member N<br/>inference → write fork]
    end

    subgraph storage ["S3 / Icechunk"]
        IC[(Initial conditions)]
        OUT[(Forecast outputs)]
    end

    A --> B --> C
    B --> IC
    C --> D --> E
    E --> F0 & F1 & FN
    IC --> F0 & F1 & FN
    F0 & F1 & FN --> H
    H --> OUT
    OUT --> J

    style modal_orch fill:#555,stroke:#333,color:#fff
    style D fill:#777,stroke:#333,color:#fff
    style E fill:#777,stroke:#333,color:#fff
    style H fill:#777,stroke:#333,color:#fff
    style modal_gpus fill:#ff6b35,stroke:#c44d1a,color:#fff
    style F0 fill:#ff8c5a,stroke:#c44d1a,color:#fff
    style F1 fill:#ff8c5a,stroke:#c44d1a,color:#fff
    style FN fill:#ff8c5a,stroke:#c44d1a,color:#fff
```

**Flow:**

1. The **orchestrator** (`run_ensemble_forecast`) runs on Modal without a GPU. It first calls `_initialize_ensemble_store`, i.e., a separate CPU container that loads checkpoint metadata to derive output variable names, builds a zero-filled template with the correct shape for all `n_members`, and commits it. This establishes a base snapshot before any inference starts.
2. The orchestrator opens a writable session on that snapshot, forks it (`session.fork()`), and spawns one `run_ensemble_member` container per member, all receiving the same fork.
3. Each **member container** gets a GPU, runs AIFS-ENS with its seed, writes its output slice with `region="auto"` into the forked session, and returns the fork. No commit happens in the member.
4. The orchestrator collects all returned forks, merges them (`session.merge(*fork_sessions)`), and issues a **single commit** so that no conflicts are possible.
5. Modal queues members beyond your plan's GPU limit automatically, e.g., with a 10 GPU limit, 50 members run in waves of 10.

## Initial conditions from ArrayLake

By default, `aifs-modal` ingests initial conditions locally from ECMWF Open Data into a self-managed icechunk repository on Tigris. If you have an [Earthmover](https://earthmover.io) subscription with initial conditions already stored in an ArrayLake repository, you can skip the local ingestion step entirely. Modal containers will pull ICs directly from ArrayLake using the `arraylake` Python client.

```{note}
Ingesting data into an Earthmover/ArrayLake repository requires an Earthmover subscription and is outside the scope of this library. `aifs-modal` only supports reading from an existing ArrayLake IC repository.
```

To use this path, set `ARRAYLAKE_API_TOKEN` in your environment and pass `initial_conditions_repo="org/repo"` (instead of `initial_conditions_prefix`) to any of the forecast functions. The flow replaces the local ingestion node with a direct ArrayLake read inside the Modal container:

```{mermaid}
flowchart TB
    subgraph local ["Local (CPU)"]
        A[Configure forecast<br/>start date, lead time, storage params]
        C[Submit forecast to Modal]
        F[Read outputs & postprocess]
    end

    subgraph modal ["Modal (GPU) 🔥"]
        D[Load ICs from ArrayLake]
        E[Run AIFS inference<br/>stream steps → Icechunk]
    end

    subgraph storage ["Storage"]
        AL[(ArrayLake<br/>initial conditions)]
        OUT[(ArrayLake<br/>forecast outputs)]
    end

    A --> C --> D
    AL --> D
    D --> E
    E --> OUT
    OUT --> F

    style modal fill:#ff6b35,stroke:#c44d1a,color:#fff
    style D fill:#ff8c5a,stroke:#c44d1a,color:#fff
    style E fill:#ff8c5a,stroke:#c44d1a,color:#fff
    style storage fill:#dceefb,stroke:#2c6fad,color:#000
    style AL fill:#4a90d9,stroke:#2c6fad,color:#fff
    style OUT fill:#4a90d9,stroke:#2c6fad,color:#fff
```

**Differences from the default flow:**

- The local ingestion step (`ingest.ensure_date_ingested`) is skipped, no GRIB download or regridding happens locally.
- `ARRAYLAKE_API_TOKEN` must be set locally so that `aifs-modal` can forward it as a Modal secret to the containers.
- AWS credentials are temporarily unset inside the container before opening the ArrayLake client, to avoid conflicts between the Tigris and ArrayLake auth environments.

This alternative applies equally to all three forecast modes (deterministic, sequential ensemble, parallel ensemble): just pass `initial_conditions_repo` instead of `initial_conditions_prefix`.

## Re-running existing forecasts and reproducibility

The set up of `aifs-modal` is designed to be used in computational pipelines. Accordingly a key feature is that every step checks whether its output already exists before doing any work, so interrupted or repeated runs pick up where they left off without duplicating effort or GPU cost.

| Step                                        | Skip condition                                                     | Override         |
| ------------------------------------------- | ------------------------------------------------------------------ | ---------------- |
| `ingest.ensure_date_ingested`               | IC group for the target date already committed on the branch       | -                |
| `run_forecast` (deterministic)              | Zarr group for the target date already exists on the output branch | `overwrite=True` |
| `run_forecast` (sequential ensemble)        | Existing group already has ≥ `n_members` along `ensemble_member`   | `overwrite=True` |
| `run_ensemble_forecast` (parallel ensemble) | Same as above, checked before spawning any containers              | `overwrite=True` |

**Ensemble reproducibility.** Each ensemble member fixes the PyTorch random seed to its member index (`torch.manual_seed(member_id)`) before running AIFS-ENS. This means that member *k* always produces the same trajectory regardless of how many members are in the ensemble or how they are distributed across containers, making it straightforward to extend an existing ensemble run by increasing `n_members` (existing members are skipped, only the new ones are computed).
