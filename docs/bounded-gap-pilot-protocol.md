# Bounded CV1 historical-Q gap pilot — proposal for review

**Status: proposed, not approved or executable.** Prepared November 3, 2026
against repository baseline `043ac2f`. This document and its
[companion proposal](../configs/bounded-gap-pilot-proposal.json) define the small
experiment to review after real-data assembly. They authorize zero neural fits,
zero ridge fits, zero scientific scaler fits, zero protected scoring and zero
paid cloud jobs. Implementation/synthetic qualification is a separate completed
stage; it is not permission to run this pilot. Existing execution gates remain
unchanged.

The question is whether a separately trained missingness-aware L168 LSTM can
tolerate narrowly specified one/two-hour historical-discharge gaps without an
unacceptable clean-input penalty. The strict common-L168 **180-fit primary
experiment stays unchanged**. This pilot does not add H2 fits, reinstate a
persistence forecast, choose an architecture, or release the optional 18-fit
confirmation or the full campaign. CV1 outcomes used in this decision enter the
selection history; CV1 must not subsequently be called untouched confirmation.
NWM remains an external comparator, never a predictor or eligibility gate; no
additional NWM scoring request is introduced in this pilot allowance.

## Exact proposed scope and clocks

Use three separate gauge models (`03161000`, `03164000`, `03479000`), LSTM,
full HRRR/initial-soil inputs, H4 history, L168/H18, CPU, one Torch thread, B64,
width64 and FP32. Use neural/shuffle seed **17**. H4 has 68,161 parameters; both
pilot arms use the same architecture. Its added history channels mean an old H2
checkpoint is not a substitute clean control.

| Fit type | Per gauge | Total | Definition |
|---|---:|---:|---|
| H4 clean | 1 | 3 | Original complete gradient histories |
| H4 augmented | 1 | 3 | Same origins/targets; replace selected gradient inputs with declared short-gap scenarios |
| Fixed ridge fallback | 1 | 3 | Six original real lags, two direct hourly outputs, alpha 1; shared by both neural arms |

There are **six neural and three ancillary ridge fits**, one configuration, no
seed search and no extra H2 fit. Three per-gauge original-observation transform
definitions are shared across the paired arms. Record every actual scaler-fit
invocation: if the qualified constructor recomputes identical gradient-only
statistics per arm, disclose those six construction events and require exact
agreement; do not label them three executed fitting operations.

All calendar bounds below are UTC and half-open. Origin C equals the information
cutoff; targets are C+1…C+18; history is C−169…C−2 inclusive. Observation/weather/
soil availability uses the proposed source-time-plus-two-hours convention.
Historical publication/revision availability remains unproven. Actual 2026
assembly, fit and artifact times are stored separately.

| CV1 role | Declared interval | Last eligible origin before data exclusions | Boundary rule |
|---|---|---|---|
| Gradient | `[2018-07-12, 2020-05-01)` | 2020-04-30 05Z | All H18 targets strictly before May 1; 18 boundary origins purged |
| Inner stopping | `[2020-05-01, 2020-07-31)` | 2020-07-30 04Z | All targets precede July 31 and `C+18h+2h <= B`; 19 origins purged |
| Outer development | `[2020-07-31, 2021-07-31)` | 2021-07-30 05Z | All H18 targets strictly before July 31; 18 origins purged |

Simulated refit B is **2020-07-31 00Z**. Apply the label-availability guard to
both fitting and stopping data; it does not relax the earlier gradient boundary.
Fitting and model-selection elapsed time is not simulated, so this convention
does not establish historical operational readiness. No protected origins or
labels enter any pilot source bundle, even as unselected rows.

## Assembly acceptance before release

Use the verified resident development snapshot and original source cache, with
the confirmed A-required/optional-R policy; exclude P, e and unknown flags and
retain original qualifier strings, source IDs, masks and clocks. Input and label
admission remain separate. Keep the existing 514 weights unchanged as approximate
independent basin footprints, without exact-outlet or nesting claims. Pilot
adoption of that interpretation remains part of the user's protocol review.

The numerical primary cohort, common to both H4 arms, must match the lag-two CV1
case lists below. Counts and digests come from the `epochs=20`,
`simulated_label_delay_hours=2`, CV1/full/L168/LSTM entries in
[the admitted-count report](../reports/confirmed-180-fit-admitted-runtime.json).
The 20-epoch filter selects the authoritative count record; it does not set the
pilot epoch cap. The companion config pins each role's case-list digest.

| Gauge | Gradient | Inner | Outer clean | Fit+inner resident rows | Gradient batches/epoch | Max updates per neural fit |
|---|---:|---:|---:|---:|---:|---:|
| 03161000 | 15,717 | 2,152 | 8,556 | 17,869 | 246 | 12,300 |
| 03164000 | 13,024 | 2,152 | 8,341 | 15,176 | 204 | 10,200 |
| 03479000 | 14,621 | 2,152 | 8,516 | 16,773 | 229 | 11,450 |

Preserve partial last batches; do not drop cases to force B64. At 50 epochs,
`2 × 50 × (246+204+229) = 67,900` is the maximum total neural update allowance.
Both arms receive equal per-epoch opportunities and maximum allowances. Their
actual early-stopped epoch/update counts may differ and must remain visible.

Assembly must authenticate actual per-slot provenance, not invent metadata from
an H2 numerical array. Bind source parents, schemas, case order, masks, role
coordinates, source-version/precision regimes and all output hashes. A count or
digest mismatch is a hold for reconciliation, not permission to filter cases
until an expected total is reached.

Training-source bundles contain **only gradient plus inner rows**; the qualified
largest constructor extent is 17,869. Store/load outer input cases separately and
keep outer labels behind the reference-side interface. Do not append outer rows
to a trainer. Real-gap outer candidates and original calendar rows may have their
own separately identified assembly products; they do not enlarge neural training
membership. Record measured assembly time, payload/provenance bytes, maximum
active memory, source/derived hashes and coverage exclusions before release.

The subsequent [real assembly receipt](../reports/gap-pilot-real-assembly-v1.json)
now reports **PASSED**: all strict role counts/case hashes above match, separate
all-outer coverage parts contain 8,742 candidates per gauge, and no fit/scaler/
optimizer operation occurred. Its manifest SHA256 is
`4795efd085a4924df9b4b6b5b96933cb9c70590ebc71ec6e46484c63a259c540`.
Assembly took 57.62 seconds, read 44,407,134 bytes, wrote 57,750,943 bytes and
peaked at 135,217,152 parent-plus-child RSS bytes. This supplies actual assembly
evidence; it does not measure full scientific trainer construction or release fits.

The [actual loader qualification](../reports/gap-pilot-real-loader-qualification-v1.json)
also reports **PASSED** in 35.51 seconds, with 612,335,616 bytes peak combined
RSS and 107,710,407 logical read bytes. All three complete fit-plus-inner loads
matched their role proofs and contained zero outer rows. Separate first-at-most-64
outer/coverage batches enforced input-only access before explicitly requested
labels, and the three ridge-series stores were checked. It performed no Torch,
model, scaler, ridge, optimizer or scoring operation. This authenticates the
loader/source identities now bound in the proposed recipes; it does **not**
qualify a real scientific model constructor or fit.

## Training, fillers and checkpoint selection

Use the existing affine standardized-Q MSE and train-only input/target transform
rules. Fit Q/coordinate-age statistics using only approved original gradient
tensor cells, before artificial masking; fills and inner/outer values never fit
statistics. Both arms use exactly the same saved definitions. Binary missingness
uses 0/1 and elapsed-since-real age uses hours/168. Preserve original-observation,
artificially-hidden, effective-observation and usable-numeric masks separately.
Filled Q does not become observed Q or reset elapsed time since a real reading.

Proposed optimizer/stopping settings retain the decision packet: AdamW,
learning rate 1e-4, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01,
`foreach=False`, dropout 0.1, seeded shuffling, no scheduler. Monitor **clean inner
standardized-Q MSE** for both arms; minimum 10 epochs, patience 5, maximum 50,
strict improvement with `min_improvement=0`. Exact ties retain the earliest best
checkpoint. A best epoch below 10 is allowed although training completes at least
10 unless a failure/resource stop intervenes. Outer artificial-gap errors never
select an epoch. Export the saved best state, never silently last weights.

For every repair, both the longest missing run and the total missing slots must
meet the named one/two-hour cap. Leading gaps, all-missing windows and longer or
more numerous gaps remain excluded. Interior fills use the immediately bracketing
approved real endpoints within the history available at C. Trailing carry uses
only the last real reading. The ridge alternative uses six consecutive approved
real lags ending at that reading and predicts the next two Q values directly.
Its alpha 1 feature-standardized fit has an unpenalized intercept and raw m³/s
outputs; all original input/output pair times lie in gradient and obey the refit
bound. Fit it once per gauge; no hyperparameter search or recursive self-feeding.

Hide artificial values before constructing endpoints, ridge lags, masks or age
features. Keep hidden truth only on the reconstruction-reference side. No fill
may use a filled endpoint, target, weather proxy, or post-C observation. No
target, weather or soil filling is introduced. Negative/nonfinite ridge fills
are failed attempts, not clipped or silently replaced with carry. A failed window
is not partially rescued. Carry is an input treatment, not a persistence forecast.

Augmentation replaces rather than duplicates inputs, with proposed probability
0.5 and mask seed 1729, keyed by canonical case ID and epoch. Initially use only
interior interpolation and trailing carry. The assembled gradient-only gap census
now supports the following **concrete, still unapproved recipes**. It counts
candidate-history windows after chronology/refit guards and before full-history/
weather eligibility filtering; inner/outer outcomes never determine these weights.

| Gauge | Proposed conditional mixture among corrupted inputs | Basis |
|---|---|---|
| Jefferson 03161000 | Equal 1/11 over all 11 carry-compatible nonclean scenarios | All 15,798 original gradient candidate histories are clean: no empirical short-gap mixture can be estimated. This is an explicit balanced stress proposal, not an automatic fallback. |
| Galax 03164000 | Counts 327/330/295 for interior1 early/middle/recent; 1/1 for mixed early/recent with carry; 7/1 for trailing1/2 with carry; divide each by 962 | Conditional frequencies of the 962 representable short-gap windows; all unlisted classes have zero weight. |
| Sugar Grove 03479000 | Counts 55/55/55 for interior1 early/middle/recent and 5/4 for trailing1/2 with carry; divide each by 174 | Conditional frequencies of the 174 representable short-gap windows; all unlisted classes have zero weight. |

The 11 balanced classes are six interior1/2-band classes, two trailing1/2-carry
classes and three mixed-band-carry classes. Explicit JSON recipes, numerator/
denominator counts, role/census/source identities and zero-entry exclusions are
in the companion proposal. The 50% corruption probability is a **stress rate**,
not an estimate of natural outages. Repeated overlapping windows are not
independent outage episodes. Using an artificial balanced mixture at Jefferson
and conditional empirical mixtures at the other gauges is a material,
heterogeneous design assumption for user review; pooling or smoothing them is
not silently adopted. Unsupported two-hour/mixed classes can still appear in the
fixed inference stress grid. Ridge inference likewise remains outside the
training augmentation recipe. Freeze the recipe and the loader's actual source
identity before any fit; the three loader identities are now recorded, while
adoption of the recipes remains pending. No recipe is selected from outer errors.

## Exact artificial-gap panels

The existing implementation defines 17 requests per checkpoint. Proposed
evaluation epoch is 0; early/middle/recent bands use mask seeds 101/211/307
respectively, shared across arms and fill methods. Clean and deterministic
trailing panels use seed 101 for recorded identity, with no duplicate draws.
This mapping is a proposal, not an already frozen scientific seed policy.

| Panel IDs | Requests | Mask and method |
|---|---:|---|
| `clean` | 1 | Unmodified original complete history |
| `interior_1_{early,middle,recent}` | 3 | One hidden slot; linear interpolation |
| `interior_2_{early,middle,recent}` | 3 | Two adjacent hidden slots; linear interpolation |
| `trailing_{1,2}_{carry,ridge}` | 4 | Last one/two slots; named trailing method |
| `mixed_{early,middle,recent}_{carry,ridge}` | 6 | One interior slot plus final trailing slot; named method |

History index 0 is C−169h and index 167 is C−2h. Inclusive start-index bands are
early 1–55, middle 56–110, recent 111–165, exactly the implemented layout. These are
fixed stress placements, not a reproduction of every real missingness location.
The same band draw is reused across lengths and carry/ridge variants. Changing
batch order, Q magnitudes, target values or model errors cannot change a mask.

Freeze all 17 masks and paired-support manifests before prediction/error
inspection. For carry-versus-ridge comparisons, compute the common surviving
six-lag context from the artificial mask alone. Mixed-recent masks can remove
required real ridge lags: classify that structural exclusion beforehand and keep
its count. The broader carry-only population remains separately visible.
Construct only the explicitly declared eligible ridge batches; do not submit a
mixed unsupported batch and recover by dropping rows after an exception.

Clean references are recomputed on each corresponding scenario's fixed cohort.
Carry predictions on the broader population can be reused on the declared paired
subset; they do not require another fit. Seventeen requests × six checkpoints
means at most **102 artificial-grid requests**, not 102 fits. For B64 the full-size
upper envelope is 13,566 outer forward batches. Smaller predeclared ridge support
reduces a request's rows without erasing structural exclusions. A numerical fill
or requested prediction failure after that declaration cannot improve the score
denominator: that required paired panel is unavailable.

## Proposed tolerances — user review required

The following are **practical discussion margins**, not adopted scientific or
operational standards. Their values are deliberately explicit so approval can be
concrete. No result is permitted to choose or revise them retrospectively.
All clean comparisons below are augmented-H4 minus clean-H4 on identical
original clean outer cases, separately for every gauge.

| Proposed guard | Candidate tolerance | Interpretation |
|---|---|---|
| Clean equal-lead mean NSE | ΔNSE ≥−0.01 | At most 0.01 loss in the full H1–18 mean |
| Clean mean of lead-wise RMSEs | Augmented/clean ≤1.02 | At most 2% loss; no pooling lead SSE |
| Individual clean lead | RMSE ratio ≤1.05 for every lead | A mean cannot hide a larger lead-specific loss |
| Clean negative-prediction frequency | Increase ≤0.005 at every lead | At most 0.5 percentage points; predictions remain raw/unclipped |
| Training/inner/integrity errors | Zero | Any leakage, invalid fill, nonfinite loss/gradient/prediction or identity failure stops the affected attempt |
| Required paired artificial panel | Zero numerical failures or missing requested predictions | Otherwise report the panel unavailable; no successful-row-only score |
| Real-gap numerical fill failure | ≤1% of predeclared structurally eligible attempts, per gauge/method | Candidate review guard only; report the count/denominator and all failures |

For a zero clean RMSE denominator, the ratio passes only when augmented RMSE is
also zero; otherwise it fails. No arbitrary epsilon may turn this into a pass.
An undefined required NSE/support panel makes the relevant decision unavailable.
Structural lack of ridge context is not a numerical failure, but remains a
coverage loss on the original requested population.

Proposed decision labels are: `TOLERANCE_BREACH` if a point estimate breaches a
guard; `INCONCLUSIVE` if point estimates pass but their conditional intervals
cross a reviewed margin or required support/intervals are unavailable; and
`WITHIN_PROPOSED_MARGINS` only when point estimates and the reviewed interval
checks remain within the margins. These are review labels, never automatic
permission for further fits. Apply the decision to each gauge rather than hiding
an adverse site in an average. The directional uncertainty checks are lower
limits for ΔNSE and upper limits for RMSE ratios/negative-frequency increases.

High-flow recall, false-positive rate, false-alarm ratio and event timing/magnitude
remain mandatory diagnostic reports. A separate operational/high-flow failure
margin is **unresolved**, rather than inferred from these generic clean tolerances.
Likewise, a minimum worthwhile robustness gain over the clean arm is unresolved;
report every scenario effect and interval rather than select the best gap pattern.

## Metrics, events, coverage and conditional uncertainty

For each gauge/scenario/lead, report physical-unit RMSE, MAE, signed bias, NSE
when defined, raw-negative frequency, failures and exact counts/digests. Report
each arm's clean score, stressed score and paired degradation, and augmented
minus clean-arm effects under each identical mask. Report carry-versus-ridge
contrasts only on their predeclared common lag-support population. Reconstruction
errors use only deliberately hidden original observations. Actual missing Q has
no reconstruction truth.

Retain the proposed primary reporting floor of 168 distinct target hours on 28
distinct UTC dates at **every lead**, with positive reference variance. All 18 lead
NSEs are required for the mean. No dropped lead, zero substitution or pooled-row
NSE repairs an unavailable panel. This CV1 pilot is not a three-fold CV selection
aggregate and seed 17 is not a seed-variability estimate. Descriptive below-floor
metrics remain clearly labelled.

Reuse the fixed gradient-calendar q10/q90/q95 thresholds and the predeclared CV1
event catalog from [coverage/events](coverage-events-uncertainty.md). HIGH is
Q>q95, not an official flood stage. Regime and positive observed-rise strata are
evaluation-only and cannot route inputs, place gaps or alter augmentation.
Distinguish recall TP/(TP+FN), false-positive rate FP/(FP+TN), and false-alarm
ratio FP/(TP+FP), retaining undefined denominators.

Retain every catalogued observed peak, its clipped peak−48h/peak+72h window and
coverage reasons. The existing 168-hour peak separation is a catalog rule, not
proof of independent floods. First-three hydrographs remain selected without
errors. For lead-specific peak timing/magnitude, use the fixed lead's valid-time
forecast series and original references over the same fixed window; report the
metric unavailable if the required paired hourly support is incomplete. Do not
interpolate event truth, replace unsupported peaks or change windows after errors.

Propose the existing uncertainty recipe: 2,000 paired noncircular whole-origin
moving-block draws, stratified by contiguous UTC season segments within CV1;
14-day primary blocks with mandatory 7/28-day sensitivities; seed 20261102.
Retain calendar gaps, all 18 leads, all model/scenario outputs and all three fixed
gauges; use the same draws across comparisons. Recompute weighted metrics on
original unique coordinates, not fabricated duplicate sample IDs.

The pilot clean-review family is the three gauge clean-arm contrasts and their
required metrics. Each named stress scenario forms its own three-gauge paired
family; each named carry/ridge contrast within an arm likewise has a separate
three-gauge family. Freeze these families and masks before scoring. Record joint
invalidity within each family without redrawing; if more than 100/2,000 draws fail,
that family's primary intervals are unavailable. Otherwise use one jointly valid
pool per family and report its size. Report all 7/14/28-day outcomes; a tolerance
claim must not choose the favorable block length. The proposed margin label
requires compatibility at all three lengths; disagreement is inconclusive.

Intervals are 2.5th/97.5th percentile, conditional and pointwise/exploratory, not
simultaneous family-wise 95% coverage, independent-year uncertainty, calibrated
prediction intervals or evidence of operational reliability. Existing utilities
do not by themselves prove the complete family-level orchestration is ready;
qualify that orchestration before promising these outputs. Events remain
descriptive rather than independently resampled floods.

The proposed release also includes four **natural-gap** requests per checkpoint:
`natural_cap1_carry`, `natural_cap1_ridge`, `natural_cap2_carry` and
`natural_cap2_ridge`. These add at most 24 requests, giving **126 total inference
requests with no extra fits**. Require at least one original missing/rejected Q
slot; the cap applies to both total missing slots and maximum run. Interior gaps
still use linear interpolation; the named method applies only to trailing gaps.
Keep original non-Q eligibility and target masks, without requiring complete
labels merely to attempt a forecast. Caps overlap and are separate panels, never
fractions to add together.

Natural-gap inputs/provenance require a separate all-outer candidate assembly;
the strict clean outer bundle alone cannot supply them. The CV1 candidate
calendar has 8,742 origins per gauge after boundary purging. Its B64 ceiling
gives at most 3,288 additional forward batches, hence **16,854 total** across
artificial and natural requests. Actual natural masks/support counts must be
frozen from the assembled metadata before model errors; report structural ridge
exclusions and numerical failures separately. Both neural arms use identical
natural-gap method-specific cases and original label masks. Natural-gap
reference-only metrics do not identify reconstruction truth or a clean-input
counterfactual and do not replace the primary clean cohort. Zero-label-support
forecasts remain in input-availability counts with unavailable scores.
These natural-panel names are protocol request labels, not new members of the
existing 17-scenario artificial-mask manifest. **The required natural-gap
inference entry point is not yet qualified.** The current
`PilotBestPredictor.predict` binds the saved training fill policy (combined-two,
carry); `scenario_id="clean"` provides no cap/trailing-method override. Renaming
a request cannot create cap-one or ridge inference. Do not mutate the checkpoint
policy, fake its expected identity, or bypass policy validation.

A separately bound, label-free natural-gap inference wrapper must preserve the
training policy and saved best weights/transform, while explicitly binding the
requested natural cap and trailing method. Qualify its source/mask/clock guards,
policy separation and prediction behavior before scientific execution. It needs
no additional scientific fit and introduces no artificial hiding, but remains
an engineering release hold. The assembly's natural candidate data are ready;
that does not make this inference interface available.

Report real-gap recovery separately on that original candidate calendar:
structural candidates, usable input windows, successful numerical fills,
successful forecasts, per-lead evaluable labels, unique target hours, known-HIGH
and fixed-event support, and the non-Q ceiling. Keep one-hour and combined
two-hour caps separate. A goal of 90% must name the denominator; it is neither a
pilot pass assumption nor a reason to tune on outer errors. No automatic L24
routing or primary-cohort expansion follows.

## Proposed local execution allowance and stop rules

Release would be limited to one sequential local worker, CPU/one thread/B64,
the six listed neural fits, three ridge artifacts and their predeclared inference/
reporting work. Proposed caps are **67,900 entered optimizer updates**, **8 hours
total wall time**, **90 minutes per neural fit**, **4 GiB active parent-plus-child
RSS**, and **4 GiB of newly created pilot artifacts**, including checkpoints,
journals and predictions. Existing inputs/receipts are excluded from that output
byte total and remain immutable. Require 8 GiB free local space at launch; reaching
a cap stops work rather than deleting artifacts or increasing allocation.

The full process—from loading/construction for the released run through ridge,
training, stopping, checkpoint writes, all requested inference and scoring/
resampling—shares the eight-hour deadline. Per-fit update ceilings are 12,300,
10,200 and 11,450 for each respective arm; failed/interrupted entered steps consume
allowance. An explicit resume carries forward the same fit identity, state,
sampler, RNG, elapsed/resource ledger and consumed budget. It cannot reset a
clock, duplicate the allocation or become an additional seed/configuration.
There is no blind retry, qualification rerun, ridge refit or replacement fit.
The completed 57.62-second pre-release assembly is reported separately; reuse its
immutable products. Any actual repeated loading/construction after launch counts
against the eight-hour allowance and needs a reason rather than an automatic rerun.

Proposed retention is an immutable full restart snapshot every five completed
epochs (at most 10 per fit), plus at most one terminal restart snapshot, one
emergency valid-state snapshot and one selected-best inference export per fit.
Each full snapshot includes current and best weights, optimizer, sampler/RNG,
stopping decisions and ledger identities. The six fits therefore permit at most
78 neural checkpoint/export artifacts, plus three ridge artifacts: **81 large
artifacts**, with update journals recorded separately and the combined output
bytes still capped at four GiB. No automatic deletion is permitted. Early stopping
still saves its terminal/best evidence. A hard failure can lose work after the
last durable snapshot; replayed entered updates consume remaining allowance.

The [read-only artifact-size worksheet](../reports/gap-pilot-artifact-size-proposal-v1.json)
records that resident v2 synthetic clean and augmented baseline checkpoints each measured
1,223,524 bytes after four updates. Their four canonical JSON trace records were
2,699–2,710 bytes each and contained identities/case IDs/indices/loss/counters,
not per-slot numerical source values. These are observed small-fixture sizes,
not a measured 50-epoch serialization bound.

For a concrete conservative **planning worksheet**, allow 4,096 bytes per trace
record and four MiB of non-trace checkpoint/export overhead. The five-epoch
snapshot cadence plus terminal/emergency/best artifacts retains at most 425
epoch equivalents of trace per fit: `5+10+…+50+50+50+50`. Across the six fits this
is `425 × 2 × (246+204+229) = 577,150` trace-record occurrences. This budgets a
full-trace best-inference artifact rather than assuming a qualified stripped
export. The resulting capacity proposal is:

| New artifact allowance | Proposed bytes | Approximate GiB |
|---|---:|---:|
| All 78 neural snapshots/exports, including repeated traces | 2,691,162,112 | 2.506 |
| Three ridge artifacts, two MiB each | 6,291,456 | 0.006 |
| One 4,096-byte journal allowance per 67,900 entered updates | 278,118,400 | 0.259 |
| Predictions and derived reference views | 536,870,912 | 0.500 |
| Metrics, compact resampling/event records and manifests | 268,435,456 | 0.250 |
| Temporary atomic-write headroom | 67,108,864 | 0.063 |
| **Planned total / four-GiB hard cap** | **3,847,987,200 / 4,294,967,296** | **3.584 / 4.000** |

This leaves 446,980,096 bytes (about 0.416 GiB) unallocated within the hard cap.
Require one writer and at most 64 MiB of total live unpublished staging. The
largest planned checkpoint is 54,575,104 bytes, within that scratch reserve.
Prediction files must be chunked; the 512-MiB prediction allowance is not
permission to stage one 512-MiB temporary file.
Do not store full per-slot provenance repeatedly in checkpoints or expanded
bootstrap calendars when immutable source identities and compact draws suffice.
The 4,096-byte JSON allowance is not a proven pickle-size theorem: validate actual
record and serialized checkpoint bytes as they are produced, including all extra
journal/status records. Cadence integration and the byte worksheet still require
review, but another blind qualification run is not necessary merely to propose
the bound. If actual growth exhausts the allowance, retain evidence and stop;
do not prune checkpoints, raise limits or assume completion.

The full six-fit orchestration, checkpoint cadence, persistent update accounting
and live byte guards also require their own bounded engineering qualification.
Existing v2 trainer qualification does not certify that future orchestration.
These runner and natural-wrapper gates can be explicit conditions of a later
small-pilot authorization; they do not justify rerunning consumed qualification
allowances or beginning scientific fits now.

The [current v2 reassessment](../reports/gap-pilot-runtime-reassessment-v2.json)
models **2.03/4.06 hours at 20/50 epochs using new-origin cold-call proxies**.
Its timings are synthetic and its inner-loop/checkpoint components include
unmeasured surrogates. Actual assembly, ridge fitting, scoring/resampling,
long-run effects, larger journals and failures are not all covered. The eight-hour
cap provides a proposed allowance beyond the modeled portion, not a confidence
bound or completion guarantee. Earlier 4.22/8.64-hour v1 and H2 proxy estimates
must not drive this release.

In particular, the 4.06-hour modeled portion budgets the 102 artificial requests;
the 24 additional natural-gap model/adapter requests remain unmeasured additions
inside the proposed eight-hour allowance. Candidate assembly is now measured in
the separate pre-release receipt; it is no longer an unresolved source-restoration
task. The proposed
checkpoint cadence also differs from the old per-epoch checkpoint placeholder;
no runtime saving is assumed without measurement.

Before release, attach the measured assembly receipt and revise the workload
worksheet with exact admitted batches, construction/RSS/I/O costs, checkpoint
size and remaining unmeasured stages. If the bounded plan does not credibly fit
eight hours/four GiB, return for a smaller explicit plan or new review. Do not
change cohorts, seeds, stopping, masks or uncertainty after seeing outcomes to
make the deadline. No paid cloud resource is released by this proposal.

Stop or hold on source/hash/schema/role mismatch; missing actual provenance;
protected-data exposure; unsupported/incomplete training inputs; failed gradient-
only isolation; divergent paired transforms; nonfinite model/fill state; lost
required prediction; recovery/custody mismatch; or any time/memory/disk/update
ceiling. Preserve failure receipts, attempted denominators and last valid states.
A local failure does not authorize upstream mutation, another cloud download,
changing qualifiers, longer gaps, clipping, or a different architecture.

## Review checklist before a small-pilot release

1. Accept the assembled real per-slot provenance, role/cohort digests and separate
   outer products; reconcile any difference from the exact count table.
2. Agree the pilot-only settings above, including approximate basin interpretation,
   two-hour refit convention, band/seed mapping and the three concrete augmentation
   recipes, with Jefferson's balanced stress recipe explicitly distinguished.
3. Accept or revise the proposed clean/failure margins, the operational/high-flow
   limits still absent, and what qualifies as a worthwhile robustness gain.
4. Accept conditional event/uncertainty interpretation and qualify the exact
   family-level reporting path; no protected assessment is involved.
5. Review the measured runtime/disk worksheet and exact 67,900-update/eight-hour
   allowance, then provide a separate conditional execution release for six neural
   plus three ridge fits, contingent on the runner and natural-wrapper engineering
   gates above. Return the completed/failed/inconclusive pilot for review
   before any optional confirmation, changed primary policy or full campaign.

This proposal reuses the documented rationale in
[the sensitivity design](coverage-sensitivity-design.md),
[pilot qualification](pilot-implementation-qualification.md),
[optimized v2 evidence](pilot-provenance-performance-v2.md),
[scoring rules](development-scoring.md) and
[protocol decisions](protocol-decision-packet.md). It introduces practical margins
for discussion, not new literature claims or validated hydrological tolerances.
