# Next steps toward the November 6 scientific poster draft

Status: **proposed plan with primary USGS quality confirmed; not a full protocol
freeze or execution budget**.
This planning update starts no training, additional storage downloads or cloud
jobs. The November 6 draft and December 11 AGU presentation remain the targets.

## Starting evidence

The acquisition calendar is accounted for, the legacy adapters and resumable
caches work, and an actual August 2023 month passes cold/warm/recovery checks.
There are 217 passing local tests. The month supplied 744 origins and 2,232 gauge
cases, of which 2,046 have complete inputs and targets. Preprocessing took
9.22 seconds with an empty derived cache including stop/resume, 5.12 seconds
for warm assembly, and 0.78 seconds for verification only. The approved source
read transferred 634.52 MiB. These are engineering results, not forecast skill.

The scientific runner, train-only transforms, full study cohort and protected
assessment reader are still required. The illustrative 2.32/3.23-hour CPU
projections used 10,000 training cases, 1,000 validation cases, 20 epochs and
12 fits; they are not estimates for the full study. The candidate dates contain
44,280 training origins and 8,784 validation origins per gauge before exclusions.

## Work packages and completion gates

| Package | Deliverable | Completion gate |
|---|---|---|
| 1. Scientific contract and coverage | A reviewable protocol, eligibility counts by gauge/split/lookback and baseline/ablation matrix | Choices explicitly resolved; source gaps and quality exclusions counted; protected outcomes remain unavailable for selection |
| 2. Boundaries and full-cohort plan | Correct late-development source admission, isolated July 2024 development export, exact source inventory and resumable staging plan | No target crosses its development split; extraction scope documented; additional read budget separately approved |
| 3. Scientific trainer and realistic timing | Train-only scalers, general per-gauge fit loop, early stopping, seed/sampler/optimizer recovery, local batch benchmark | Recovery reproduces the next update and sample order; realistic train/validation throughput measured; full runtime proposal calculated |
| 4. Development experiment | All four architectures at all three gauges, shared cohorts, declared search/seed budget and controls | Complete receipts and validation results; selection rules applied without protected scores |
| 5. Locked assessment and poster | Locked protocol/selection, one controlled assessment, figures and existing AGU layout populated | Qualified reader and labels, per-lead support counts, uncertainty and limitations; editable poster and PDF reviewed |

Each implementation package should be independently tested, committed and
pushed. Raw data, fitted transforms, checkpoints and private source locations
remain outside Git. Historical receipts remain bound to their executed code.

## Proposed scientific starting point

Confirmed scope stays H1–18 at gauges 03161000, 03164000 and 03479000 with LSTM,
Transformer, GRU–Transformer and Mamba2–Transformer. Recommendations below still
need review; they do not change `configs/development.json` or enable scoring.

- Retain the candidate chronological training interval July 12, 2018–July 30,
  2023, and validation July 31, 2023–July 30, 2024, with all cross-boundary target
  windows purged. The protected assessment origins remain July 31, 2024–July 30,
  2025, with the declared final label tail through July 31, 2025 at 17Z.
- Start with per-gauge fits. Audit complete-input retention at both L24 and L168
  before choosing the primary lookback; compare lookbacks on a shared cohort.
- **Confirmed by the user on November 2:** admit approved observations with A
  and optional R flags for the primary experiment; exclude provisional and
  estimated observations. Preserve every flag and never fill targets. Decide
  input missingness admission separately from label masking. The next adapter
  chunk will implement this rule; historical exact-A engineering receipts stay
  unchanged. The decision is recorded in `configs/scientific-decisions.json`.
- Retain the explicit two-hour availability scenario, complete single-run H18
  weather selection, independent soil state and the existing approximate basin
  weights with their limitations. Final spatial admission is a documented study
  decision; previously accepted numerical discrepancies need not be re-audited.
- Fit transformations on training data only. Freeze target transform, loss,
  output constraints and validation selection metric before full fits. A simple
  standardized-discharge MSE baseline is a candidate; log-flow sensitivity should
  not expand into an uncontrolled search.
- Use persistence from the last admissible observation and a matched
  discharge-only neural control. A comparison against the full input bundle
  measures HRRR weather plus soil information together; it cannot isolate future
  weather from soil without an additional control. Predeclare that claim scope.
- Use the same admission cohort and comparable tuning opportunities for all
  architectures. Record parameter counts and runtime; width64 does not imply
  equal parameter counts. Stage one seed per configuration for qualification,
  then budget multiple seeds for scientific comparisons. Do not declare a
  scientific winner from a single seed or choose seeds using assessment results.

USGS explicitly distinguishes approval from revision status: approved records
can be revised and re-approved. Authenticated July 2025 completeness metadata
shows Galax has A+R throughout the month; strict exact A would exclude those
observations, including its final target tail. Numerical protected values were
not read during this planning check.
[USGS revision procedures](https://water.usgs.gov/osw/RevisionsGuidance.html).

Two boundary fixes are concrete. The current source-init-plus-48-hour guard
unnecessarily rejects valid July 29–30 development forecasts; admission should
follow causal source times and the requested target window. July 2024 USGS also
needs a controlled development-only export. Parquet filters do not prove zero
internal decoding of protected rows. Define the administrative extraction
boundary explicitly, expose only preboundary rows to development, and preserve
source and partition hashes. Do not quietly weaken the protected scoring gate.
The final assessment label tail is supported by completeness metadata, but its
actual source admission must still be verified under the assessment workflow.

## Local compute and budget decision

First benchmark genuine August batches at B16/B32/B64 on CPU and MPS for all four
architectures: 24 cases, proposed five warmup plus 20 timed updates each (600
engineering updates), at most 15 minutes and 4 GiB. Use a fixed chronological,
purged development cohort and distinct cases within each batch. Record repeated
passes, first-use overhead, transfer, training, validation, checkpoint and memory
costs. Discard benchmark weights and publish no skill claim. Confirm the chosen
configuration on the other gauges before selecting scientific execution settings.

The first implementation can remain local. Scaling the observed August source
and cache sizes to 8,844 preassessment six-hour initializations gives roughly
44 GiB of source bundles and 1.1 GiB of derived stores. This is a one-month sizing
scenario, not a complete inventory. Stream bounded raw staging batches into the
compact cache; remove only newly created temporary downloads after their verified
reductions commit. Existing source archives and evidence stay preserved.

Additional source reads require a new concrete allowance; the August approval
does not authorize the rest of the corpus. Determine exact object sizes, source
locations and case counts before that request. If measured local training cannot
meet the schedule, propose a small capped CUDA qualification/timing pilot. Cloud
runtime must be measured on cloud hardware; CPU/MPS ratios are not a substitute.
Approve a full campaign ceiling only after the pilot, including storage, transfer,
idle time, failure allowance and shutdown controls. No paid compute is authorized.

Use the current local journals, artifact hashes and Prometheus metrics during
this phase. Add run comparison, data eligibility, throughput, validation loss,
checkpoint age and resource usage. A hosted observability service is optional
and should not block the scientific experiment or poster.

## Calendar and contingency

| Date | Target |
|---|---|
| Nov 2 | Resolve protocol decisions; prepare boundary fixes and source-read proposal; qualify realistic batches |
| Nov 3 | Stage the chosen development cohort; verify train-only transforms, trainer and recovery; establish baselines |
| Nov 4 | Complete the core four-architecture development runs; decide whether assessment readiness is realistic |
| Nov 5 | Lock protocol and selections, then assess only if all gates pass; produce per-gauge/per-lead figures and support counts |
| Nov 6 | Finish scientific review and the editable/PDF poster draft in the existing AGU-themed layout |

This is an ambitious target, not a guaranteed completion date. Cut exploratory
sweeps before cutting required architectures or gauges. If the data/trainer is
not ready by November 3 or the development experiment is incomplete on November
4, the November 6 draft must clearly identify preliminary development results
and unfinished assessment. Do not open the protected assessment early to rescue
the deadline. Continue full scientific completion toward the December 11
presentation.

Poster work can proceed in parallel on the research question, basin map, methods,
pipeline diagram and limitations. Verify the framing against the accepted
abstract before finalizing claims. Add results only from locked, reproducible
artifacts. No coauthor messages or submissions are sent by this plan.
