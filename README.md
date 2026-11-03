# HYDRA DirectFlow

Direct hourly discharge forecasting at three Appalachian gauges using observed
USGS history and issued HRRR weather. The forecast outputs are leads 1–18 hours
at gauges `03161000`, `03164000` and `03479000`.

The experiment includes LSTM, Transformer, GRU–Transformer and
Mamba2–Transformer. NWM is an external comparator. Research skill results are
not yet available; synthetic model checks and a development data smoke test
do not establish forecast performance.

The [resident execution recovery](docs/resident-execution.md) now keeps active code and a fully isolated pinned runtime outside cloud storage. [Optimized provenance processing](docs/pilot-provenance-performance-v2.md) passed exact v1 numerical equivalence and a new bounded restart/inference qualification. The six-fit cold-default proxy is now **2.03/4.06 hours at 20/50 epochs**, plus stated unmeasured work. Scientific fits and paid cloud jobs remain held. The [resident development snapshot](reports/resident-optimization-release-v3.json) is now **1,432/1,432 files across 477 shards**, with the exact snapshot integrity check passed. [CV1 real-data assembly and bounded loader checks](docs/pilot-real-data-assembly.md) are complete. The [small-pilot protocol](docs/bounded-gap-pilot-protocol.md) proposes six neural plus three ridge fits, pending user review and qualification of its bounded runner and natural-gap inference wrapper.

The user subsequently requested ridge removal, all four deep-learning families and a higher epoch ceiling with early stopping. The [200-epoch proposal](docs/high-epoch-stopping-proposal.md) records its resource implications; the separate-pilot versus direct-primary sequence remains under discussion. The earlier six-neural/three-ridge packet is retained historical planning, not the current execution scope. No scientific execution is released.

The active [local seed sequence](docs/local-seed-sequence.md) uses the user's latest instruction to select the [fastest demonstrated backends](docs/local-device-policy.md): CPU for LSTM/GRU–Transformer and MPS for Transformer/Mamba-2–Transformer. The all-MPS alternative is retained with its54 unmeasured control jobs explicitly identified. Seed order, cohorts and all180 fits remain unchanged; model execution has not started.

The [v1 implementation and qualification](docs/pilot-implementation-qualification.md), including its earlier 4.22/8.64-hour six-fit estimate, is preserved as historical evidence. Use the v2 performance summary above only for its stated LSTM pilot scope; neither timing applies to an expanded four-family pilot or a200-epoch campaign.

The requested missingness sensitivity now has a [development coverage census](docs/coverage-sensitivity-census.md), [paired experimental proposal](docs/coverage-sensitivity-design.md), and measured-rate cost scenarios. Short Q gaps alone leave substantial Sugar Grove high-flow exclusions; a separate short-gap/L24 route has broader structural support. No study-data gap filling or scientific sensitivity training has run, and the strict 180-fit primary remains unchanged.

## Current work packages

**November 2:** the [complete development corpus](docs/development-completion.md)
is built and audited: 477 shards, 53,028 origins, 159,084 cases and 146,464
candidate complete L168 cases. The user confirmed the 180-fit study scope.
Methodological adoption and scientific fitting remain pending; no protected
assessment scoring or cloud compute job has started.

The user subsequently accepted the training core subject to pre-freeze checks.
Those checks now include all three official-source spatial reproductions,
timestamp-based refit guards, qualified label-free best-checkpoint inference,
scoring/aggregation tests and seasonal/high-flow coverage diagnostics. See the
[pre-freeze findings and material tradeoffs](docs/prefreeze-checks.md) before
adopting the final protocol. The high-flow availability limitation is material;
these diagnostics are not forecast-skill results.

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

See [the completed work ledger](docs/work-ledger.md) for commits and evidence.
The work ledger and versioned release receipts record integrated and focused
test results at their executed code versions. Current reports
include four-model CPU/MPS qualification, a bounded real-data engineering probe,
fresh-process recovery and a 20-case synthetic local throughput benchmark.
Discharge-only ablation and source-aware NWM evaluation helpers are implemented.

The initial [multi-cycle reader qualification](docs/multicycle.md) processed five authenticated
development cycles across 2021–2024, preserving missing history masks. All four
models passed bounded CPU/MPS end-to-end timing and checkpoint replay. See the
[runtime and budget inputs](docs/runtime-and-budget.md) for measured stage costs,
conditional local projections and the remaining cloud-budget uncertainties.

[Older index/payload formats](docs/legacy-lineage.md) and
[resumable basin/discharge stores](docs/cache.md) are tested on resident source
data. The [August 2023 month runner](docs/month-preprocessing.md) now passes a real
744-origin / 2,232-case cold, resumed and warm processing exercise. The approved
storage download stayed below 1 GiB; no cloud compute jobs were started.
Preprocessing took 9.22 seconds with an empty derived cache including stop/resume,
5.12 seconds for warm full assembly and 0.78 seconds for verification-only resume.

The [November 2 next-steps plan](docs/next-steps-20261102.md) proposes the scientific
protocol, cohort preparation, realistic training benchmark and November 6 poster
schedule. It does not freeze scientific choices or authorize further spending.

The [reconciled scientific study design](docs/scientific-study-design.md) records
hypotheses, actual architecture counts, proposed expanding CV, component/input
controls, assessment gates and the figure/edge-research plans. The confirmed
A/A+R quality rule is implemented with independent input/target policy arguments;
persistence is excluded from the active experiment. Full scientific fitting,
CV adoption and protected assessment remain pending.

The [execution-readiness update](docs/execution-readiness.md) includes a refined
development eligibility inventory, the approved July development-only export,
fresh-process trainer recovery and the completed 24-case real-batch benchmark.
The 36-fit main matrix has conditional local runtimes of about9.5–10.5hours at
20epochs, or23.8–26.2hours at50epochs, before CV and controls. The user approved
the $150 combined staging/compute/contingency ceiling. Development staging is
active; cloud funding remains conditional on measured need and qualification.
No paid cloud job has started. The [staging guide](docs/development-staging.md)
describes the persistent budget, exact source scope and resumable corpus.

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
Cloud funds are reserved within the approved ceiling; execution still requires
the recorded phase gates. See `docs/execution-readiness.md` and the work ledger
for current dependencies. Earlier overnight plans remain historical proposals.
