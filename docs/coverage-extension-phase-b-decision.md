# Phase B decision packet — proposed, not authorized

This packet is preparation after the separately registered Phase A coverage
extension. It does **not** authorize real repair forecasts, new training, extra
features, cloud jobs, downloads or protected assessment. Phase A has completed.
The recommendation is to prepare the development-results poster from the clean
extension first, then consider this bounded repair sensitivity after reviewing
its tolerances. The potential recovery at Galax's largest observed peak makes
the question worth studying; coverage alone does not establish safe forecasts.
No Phase B execution has been requested or started.

## Measured remaining opportunity

The [structural census](../reports/coverage-extension-phase-b-census-v1.json)
uses committed Phase A inputs and the fixed catalog, with no repairs or model
calls. It applies both the per-run and total-slot caps, excludes leading gaps,
and verifies originally admissible endpoints are finite and nonnegative.

| Gauge | Current forecast origins | K1 additional origins | K2 additional origins | K2 potential availability |
|---|---:|---:|---:|---:|
| Jefferson | 8,766/8,766 | 0 | 0 | 100.00% |
| Galax | 8,613/8,766 | 49 | 50 | 98.83% |
| Sugar Grove | 8,392/8,766 | 54 | 82 | 96.67% |

K1 and K2 are nested alternatives, not additive recoveries. Galax K1 could
recover all 422 currently unsupported known-high cells and seven unique high
hours, including H1–H3 at the January 10 peak. The extra K2 Galax origin has no
admissible verification targets, so adds issuance availability but no evaluable
cells. Sugar Grove K1 adds no known-high cells; K2 could recover 197 cells and
three unique high hours. Its 11 observed catalog peaks already have all-lead
coverage under the clean router. These are potential support counts, not repair
accuracy results or claims about unobserved peaks.

Repair-first would also replace already available clean L24 forecasts at 286
Galax and 428 Sugar Grove origins under K2. Those substitutions are a distinct
accuracy tradeoff, not recovered forecasts. Clean-L24-first avoids them. Even
K2 leaves 103 Galax and 292 Sugar Grove origins unavailable.

## Candidate policies

Keep the original non-Q rules, source identities, two-hour information boundary
and per-lead A/A+R verification. Never fill a target or apply Q repairs to weather
or soil. Only originally admissible finite, nonnegative Q readings may be endpoints.

| Candidate | Maximum consecutive missing Q slots | Maximum total repaired slots in L168 |
|---|---:|---:|
| K1 | 1 hour | 1 |
| K2 | 2 hours | 2 |

Interior interpolation uses the immediately bracketing **original** readings,
with both available by C under the C−2 boundary. A filled value cannot become
another repair's endpoint. Trailing carry-forward uses the last original
admissible reading within the declared window. Leading gaps, longer gaps,
all-missing histories and windows exceeding the total cap remain unavailable.
Carry-forward here is a historical-input treatment, not a persistence forecast
for H1–H18. Ridge is excluded.

Two candidate orders must be fixed before outcomes:

1. Complete L168 → admissible repaired L168 → clean L24 → unavailable.
2. Complete L168 → clean L24 → admissible repaired L168 → unavailable.

Their structural union may have the same coverage, while their forecast values
and costs differ. Reuse exactly identical repaired vectors/forecasts across
policy labels rather than duplicating model calls. Structural inability to
repair is explicit unavailability. A nonfinite or negative reconstructed Q
input, or a nonfinite model prediction, is a technical failure,
not a reason to silently switch route. Finite negative model predictions remain
raw and are reported; they are not silently withheld or clipped.

## Model and provenance limitation

The current networks expect complete two-feature histories and have not learned explicit
missingness indicators. A repair adapter must preserve the original observed
mask, qualifiers, values and endpoints separately from a derived numerical
usable mask. It must not relabel reconstructed values as approved observations.
Use each saved checkpoint's transforms without fitting again. The existing age
channel remains C minus the slot timestamp; it cannot be relabeled time since
last real reading.

Under such an adapter the current network would not be told that a value was
reconstructed. That robustness is unproven. Adding observed/filled indicators,
time-since-real channels or realistic missing-block training would require new
features/training and a separate experimental scope. A routed Q-only L24 control
also needs 36 new final-development fits for the full four-model/three-gauge/
three-seed matrix; it is not part of this inference-only proposal.

## Paired artificial-gap design to review

Use only original L168-input-eligible histories. Precommit a deterministic source
calendar sample and the fixed observed-event catalog; do not select cases from
forecast errors. Include every eligible origin needed for the first-three event
windows, plus at most 512 additional origins per gauge. For the additional sample,
exclude those event origins and sort the remaining eligible origins by the
SHA-256 hex digest of UTF-8 `HYDRA_PHASE_B_V1|gauge_id|YYYY-MM-DDTHH:MM:SSZ`, with
UTC origin as a tie-breaker; take the first 512. This fixed key and rule must be
recorded in the future adopted plan before repair outcomes are inspected.
Three 120-hour event windows expanded for the H18 origin range contribute at most
411 origins, so the proposed upper bound is 923 original origins per gauge before
overlap or input-admission exclusions.

Predeclare seven scenarios: one-hour old interior (C−96), recent interior (C−6),
and trailing (C−2); two-hour old interior (C−97,C−96), recent interior (C−7,C−6),
and trailing (C−3,C−2); plus two separate missing hours (C−96,C−6). The last
scenario tests K2's allowance for disjoint gaps rather than treating it as a
contiguous-gap-only policy. These patterns are diagnostic, not exhaustive.
Keep a no-gap control. Compare treatments on the same
origins, saved models and admissible per-lead targets. Report originally missing
real-gap cases separately. Retain all four architectures and seeds 17/29/43.

The artificial sample is a diagnostic sample, not an unbiased annual skill
estimate. Event timing/magnitude statistics require complete admissible observed
and predicted event windows at the lead being evaluated; otherwise report them
unavailable. For the proposed complete-window timing measure, choose the maximum
observed and maximum forecast discharge within the same fixed window, breaking
exact ties by earliest UTC time. Timing error is predicted-peak time minus
observed-peak time; magnitude error is forecast maximum minus observed maximum.
Normalize absolute magnitude error only when the observed maximum is positive.
These are new proposed definitions, not retrospective changes to Phase A's
error-at-the-observed-peak statistic. Sparse sampling must not manufacture a
forecast peak or its timing.

## Proposed screening tolerances — require explicit adoption

These are proposed scientific screening criteria, not approved operational
safety thresholds. They must be agreed before real repair outcomes are inspected.
No criterion may be relaxed merely because a candidate fails it.

Proposed application: evaluate each architecture/gauge/seed and each declared
lead separately, except equal-lead NSE, which is one H18 aggregate per fitted
model. Keep all three seeds. Any failed required screen flags the candidate for
review; do not discard that model/seed and declare a global pass. Insufficient
support is inconclusive, not a pass. No-gap controls and technical checks are
mandatory for every affected case.

| Check | Proposed rule |
|---|---|
| No-gap adapter path | Byte-exact input and forecast preservation |
| Leakage, source mismatch, nonfinite technical failure | Zero tolerated; stop and investigate |
| Paired equal-lead NSE | Degradation no worse than 0.01 where the unchanged primary support floors are met |
| Non-elevated-flow RMSE and MAE | At most 5% relative degradation on identical verified support |
| Known-high-flow RMSE and MAE | At most 10% relative degradation; retain counts and event dependence |
| Peak absolute-relative error | Increase at most 5 percentage points on complete fixed windows |
| Absolute peak-timing error | Degradation at most 1 hour on complete fixed windows |
| False-alarm cell fraction | Increase at most 1 percentage point on the same verified non-elevated-flow cells |
| Negative forecasts | Retain raw values; report changes by lead; no clipping to obtain a pass |

For review, define a false-alarm cell as predicted Q above gradient q95 while its
approved target is at or below gradient q90. This is a relative flow diagnostic,
not an official flood-warning false-alarm definition. Non-elevated means
Q≤q90 (LOW and ORDINARY combined); HIGH remains Q>q95. Paired error/false-alarm
comparisons require identical original coordinates and valid references. Report
structural unavailability separately instead of treating a withheld forecast as
a successful false-alarm reduction. A rising-limb descriptive
stratum can require Q(t)>Q(t−3h) with all intervening reference hours admissible;
its exact use must be adopted before execution. Neither stratum may affect routing.

Undefined ratios, insufficient support and incomplete event windows remain
inconclusive. The unchanged 168-distinct-hour/28-UTC-date rules govern the primary
NSE aggregate; short event summaries do not gain independence by overlapping
forecast windows. All seed results and failures must remain visible.

## Measured resources and proposed finite envelope

For the artificial upper bound, seven repaired scenarios plus one no-gap control
would require at most `8 × 923 × 3 gauges × 4 models × 3 seeds = 265,824`
case–checkpoint evaluations before safe reuse. Real-gap candidates must be
counted from the Phase A input masks under the **exact K1/K2 total-slot caps**;
the older short-gap-union census is not an exact substitute. Route-order variants
may reuse identical forecasts and should not create unnecessary calls.

The exact K2 census contains 846 repair-eligible gauge/origin histories, including
already available L24 cases. Reusing identical K1/K2 repaired vectors gives at
most 10,152 additional real-gap case/checkpoint forwards across all four models
and three seeds. A conservative budget also retains 289,440 original L168 clean
regression forwards, if needed for qualifying the new adapter, plus 265,824
artificial/control forwards: **565,416 case/checkpoint evaluations before a
20,000-case qualification/replay reserve**. Completed Phase A regression will not
be repeated merely as setup; the allowance is a conservative proposed ceiling.

The following timings were measured over nine original L168 clean jobs per
architecture (72,360 cases each), with the saved transforms and device sync:

| Model | Backend | Forward milliseconds/case | Maximum task overhead/job |
|---|---|---:|---:|
| LSTM | CPU | 0.616 | 8.56 s |
| Transformer | MPS | 0.692 | 7.21 s |
| GRU–Transformer | CPU | 0.612 | 7.37 s |
| Mamba-2–Transformer | MPS | 1.157 | 10.01 s |

Applying these rates to 141,354 cases per architecture plus 18 cold-task overheads
per architecture gives about **17.2 minutes** of model/task work. Doubling that
estimate and adding 20 minutes for adapter, scoring and reporting gives about
**54 minutes**. Repair implementation and review are not included. Small batches,
memory pressure or a different schedule can reduce throughput; the timing is a
projection, not a completed repair benchmark.

Proposed execution limits, all requiring a separate adopted plan: one local
worker; the same CPU/MPS assignments; at most 600,000 case/checkpoint forwards,
10,000 batches of at most 64, two persistent hours, 4 GiB combined memory with
2 GiB MPS, 1 GiB new artifacts within the original aggregate cap, 8 GiB logical
read reservations, and an 8 GiB free-disk floor. Include qualification, failures
and replay. Reconcile remaining global storage and the original campaign deadline
before release. Keep zero new training/scaler fits, spending, downloads and
protected assessment. Do not use unused Phase A headroom as execution authority.

The decision should weigh marginal recovered origins and known peaks against
new-case errors, artificial-gap degradation, failure risk and cost. Coverage
already above 90% is not proof of reliability, and coverage below it is not
permission to keep adding policies. A recommendation not to run repair
experiments is a valid outcome.

The large peak underprediction observed in Phase A occurs even where forecasts
are already available. Repairing missing inputs is not established as a remedy
for that separate accuracy limitation. Review the development poster and these
tolerances before deciding whether the marginal coverage benefit warrants this
additional experiment.
