# HYDRA DirectFlow

Direct hourly discharge forecasting at three Appalachian gauges using observed
USGS history and issued HRRR weather. The forecast outputs are leads 1–18 hours
at gauges `03161000`, `03164000` and `03479000`.

The experiment includes LSTM, Transformer, GRU–Transformer and
Mamba2–Transformer. NWM is an external comparator. Research skill results are
not yet available; synthetic model checks and a development data smoke test
do not establish forecast performance.

## Current work packages

| Chunk | Deliverable | Completion evidence |
|---|---|---|
| 01 | Installable package and documented study boundary | Package imports and source checks |
| 02 | Causal clocks, weather selection, history and split rules | Leakage and boundary tests |
| 03 | Hash-verified local HRRR/USGS/weight adapters | Bounded real-data H18 smoke |
| 04 | All four model implementations and offline dependencies | CPU/MPS and checkpoint qualification |
| 05 | Portable run events, resource observations and recovery state | Telemetry/resume tests |
| 06 | Development throughput and runtime projections | Measured, clearly scoped benchmark report |

Each completed chunk is committed separately. The protected assessment origins
are `[2024-07-31T00:00Z, 2025-07-31T00:00Z)`; their declared final labels extend
through `2025-07-31T17:00Z`. Development labels must stay inside their partition.
Actual artifact creation timestamps are distinct from simulated historical
information cutoffs.

## Local use

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[data,models]'
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The core package itself has no dependencies; the full test suite needs the data
and model extras. See `docs/offline-work.md` for a verified offline restoration.
Native Apple MPS and Linux/CUDA use separate environments with the same code
and data contracts. Installing the package does not download pretrained models,
fetch research data, provision cloud resources or start a training campaign.

## Data and claims

The retained HRRR calendar has 61,824 initialization slots: 61,658 complete and
166 explicitly recorded source gaps. The verified input index and weather
objects remain outside Git. A generation/hash-aware loader must resolve the
retained objects; source calendar completeness is not training-window coverage.

The initial local smoke uses existing approximate study-basin weights with
their documented geometry limitations. It is a development fixture, not a
claim of surveyed outlets or a fully frozen scientific protocol. Input delays
are explicitly simulated; final/revised USGS values do not reconstruct their
historical vintages.

Raw datasets, checkpoints, credentials, local paths and dependency caches are
excluded from this public repository. Public reports contain aggregate check
results, source hashes and logical identities. Run `scripts/publication_check.py`
before staging a release.

Hosted CI is supplied as `docs/ci/core-workflow.yml`. It has not been activated:
the current GitHub credential does not have the workflow scope. Local checks and
portable receipts are available in `reports/`.

The AGU draft target is November 6, 2026; presentation date is December 11.
Paid compute remains unbudgeted. Cloud execution requires a measured, bounded
execution proposal. See `docs/overnight-plan.md` for the current small-chunk plan.
