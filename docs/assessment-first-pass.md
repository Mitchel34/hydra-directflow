# Saved-weight assessment first pass

The user requested assessment-year forecasts and an NWM comparison, then prioritized
the shortest qualified local path using saved weights. This supersedes the earlier
blanket assessment hold for this bounded inference/comparison scope. It does not
authorize retraining, additional experiments, new upstream acquisition or paid
compute. The completed 180-fit development campaign remains unchanged.

The separately frozen [protocol](../configs/assessment-firstpass-v1.json) admits
36 existing best-inner-stopping FULL L168 checkpoints: all four architectures,
three gauges and seeds 17, 29 and 43. No weights, transforms or seeds are selected
using assessment outcomes. Q-only, L24 and coverage-router assessment products
are deferred. Development coverage-extension findings remain separate.

## Time and information boundary

Origins are every UTC hour in `[2024-07-31, 2025-07-31)`, including the last
18 origins. Targets run through July 31, 2025 at 17:00 UTC. History is exactly
`C-169h` through `C-2h`. The two-hour availability delay is simulated; the
revised historical USGS archive does not establish historical publication times.
One eligible HRRR cycle supplies all weather horizons without stitching; soil
is a separate timestamped initial state. Original A/A+R qualifiers, masks,
source hashes, exclusions and spatial weights are retained.

Each forecast depends only on its causal inputs. An assessment observation can
be an input for a later origin and a target for an earlier origin. The barrier
therefore separates **verification-role assembly and scoring** from forecasts;
it does not claim that no observation from the assessment year is decoded.
Parquet may decode full row groups internally. Only authorized role-window rows
are returned, and that limitation is explicit in the reader receipts.

All 36 prediction products must be durably committed and hash-verified before
verification-role target assembly. Missing predictions on admitted inputs are
errors, not reasons to shrink the evaluation cohort. Unavailable inputs remain
explicitly unavailable; targets are never filled.

## Comparison and reporting

Primary HYDRA scores use strict complete-H18 targets on common admitted inputs.
Per-lead valid-target support is reported separately. Metrics include per-lead
NSE, equally weighted H1-18 mean NSE, physical MSE, RMSE, MAE, bias and negative
prediction frequency. The predeclared primary support gate remains 168 distinct
target hours and 28 UTC dates per lead, with positive observed-flow variance.
Undefined panels remain explicit; no undefined lead is silently dropped.

NWM uses the owned unique-record BigQuery short-range snapshot, already decoded
in m3/s and checked against the retained native-file parity receipt. Its issue
must be exactly `C-2h`; native leads 3-18 map to origin leads 1-16. H17-18 are
unsupported under this cutoff. No older-cycle fallback or rescaling is added.
The NWM source cannot determine independent HYDRA eligibility. Paired tables use
identical gauge/origin/valid-time keys and label their H1-16 summary separately
from HYDRA's H1-18 primary score.

Retain all three seed results and average their metrics, not their predictions.
The first release reports point estimates and descriptive seed ranges. These
are not confidence intervals. Overlapping windows, dependent events, approximate
basins, observation outages and limited years constrain interpretation. Sample
native metadata do not establish one NWM version for the whole archive.

## Bounded execution and evidence

The user approved up to 16 GiB GCS reads, 4,000 GET attempts and $5 estimated
storage/read-transfer cost within the existing $150 ceiling, plus five owned
USGS files totaling 1,371,323 encoded bytes. Metadata and source budgets persist
across processes; no allowance is reset. No VM or paid training job is started.
Thirteen native USGS partitions and their manifest have been verified locally.
The 1,463 HRRR candidate archive identities total 12,201,226,240 bytes. The first
128 archives were staged as opaque bytes before numerical execution.

Execution has its own six-hour numerical clock, 350,000 case/model forwards,
6,000 B64 batches, 36 model constructions (up to 40 only after an explicit
infrastructure diagnosis), 64 GiB logical reads, 2 GiB derived/results output,
4 GiB combined resident/MPS memory, 2 GiB MPS memory and an 8 GiB free-disk floor.
A model runs in a fresh process on its qualified CPU or MPS backend. Failure
records and partial artifacts remain; no scientific or numerical failure is
automatically retried. Source transfer retries remain independently charged.

Private execution plans bind source inventory, code, frozen weights, exact
checkpoints and gradient-fit memberships. Private data, checkpoints, paths and
bucket locators remain excluded from this public repository. Aggregates become
results only after actual inference, scoring and independent verification;
successful engineering tests alone establish no forecast skill.
