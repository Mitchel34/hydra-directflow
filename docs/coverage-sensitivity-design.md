# Proposed historical-discharge coverage sensitivity

This is a design and resource proposal, not permission to fit, score, download,
or run cloud work. The strict, matched L168 **180-fit primary experiment stays
unchanged**. The separate sensitivity asks whether small, explicitly identified
historical-Q gaps can be handled without unacceptable loss of forecast quality.
It cannot establish performance in long outages or reveal unobserved peaks.
No executed forecast sensitivity or operational 90% coverage achievement exists yet.

## Information available at issuance

For an hourly origin C, the L168 history remains C−169h through C−2h. The
administrative availability convention is source timestamp +2h ≤ C. This is a
simulation: the archive does not establish the original publication/revision
vintage or prove that either endpoint was actually available then. Preserve
that limitation in every receipt and operational claim.

The eligibility census classifies **structurally repairable candidates**; it does
not fill values. Evaluate separate one-hour and two-hour caps. Both the longest
missing run and the TOTAL missing slots in the L168 window must fit the cap.
Leading gaps, all-missing histories, longer runs, and windows with too many
missing slots remain excluded. Endpoints must be approved real observations
inside the exact input window, with unchanged original flags and source IDs.
No filled value becomes a new real observation or a new endpoint.

| Gap | Proposed input treatment | Information constraint |
|---|---|---|
| Interior, ≤2 missing hours | Linear interpolation between immediately bracketing approved real observations | Both endpoints satisfy simulated availability by C; the right endpoint may follow the missing timestamp but cannot follow C−2h. This is reconstruction available at issuance, not a claim it was known at the missing timestamp. |
| Trailing, ≤2 missing hours | Compare carry-forward with a fixed learned fallback | Only surviving observations at or before the last real reading; no right endpoint, future reference, current target, or rainfall rule. |
| Mixed, two slots total | One eligible interior hour plus one trailing hour | Recompute both masks jointly; an artificially hidden trailing point cannot be an interpolation endpoint. |

Carry-forward supplies a historical **input only**. It is not a standalone
persistence forecast and does not reintroduce the excluded persistence baseline.
Weather/soil eligibility is unchanged. No policy fills or modifies C+1…C+18
targets, admits rejected target qualifiers, or invents future rainfall.

The small learned-fallback candidate is a fixed, per-gauge two-output ridge
autoregression: six consecutive approved real lags ending at t₀ predict
Q(t₀+1h) and Q(t₀+2h). Use alpha=1 with gradient-only feature standardization,
an unpenalized intercept and raw m³/s outputs, no hyperparameter search, and no
recursive self-feeding. Its training pairs must
have original approved inputs and outputs entirely within the gradient role and
obey its refit-availability bound. A model is fitted once per gauge/fold and
reused across neural seeds. This is an unqualified proposed ancillary model,
not evidence that ridge is optimal. Missing lag context or nonfinite/negative
fills are explicit failed attempts; no silent clipping, replacement, or removal
from the attempted-case denominator. If failures prevent the preregistered
paired score, report it unavailable. Report fallback support separately from
the broader structural carry-forward ceiling.

For artificial gaps, hide the selected Q values BEFORE constructing endpoints,
lag features, masks or age features. The original values remain accessible only
to the offline evaluation of reconstruction error, never to the filler or
forecast model. Do not recycle forecasts from a model trained on the same hidden
values as if they were independent reconstruction evidence. No filler or scaler
FITTING uses stopping/outer targets; applying the frozen target transform for
inner monitoring remains necessary. All model/scaler fits are counted.

## Preserve the meaning of age and missingness

Current H2 is Q plus observation-timestamp age from C. That coordinate age is
not elapsed time since the last real observation. A proposed versioned H4
adapter would keep Q and coordinate age, then add:

- `was_missing`: 1 for an originally missing/rejected history slot, even when
  a numerical fill is supplied; 0 for an approved real observation.
- `elapsed_since_last_real_observation_hours`: at history timestamp t, time
  since the latest approved REAL observation at or before t; zero on observed
  slots. Filled values never reset this clock. Unknown leading ages remain
  unsupported.

Thus when the latest eligible slot C−2h is missing for one hour, the last real
reading is C−3h: the fill's coordinate age is 2h, its elapsed-since-real age is
1h, and the latest real reading is 3h old at C. Retain original source masks,
qualifiers, fill method and endpoint IDs in separate provenance. Numerical
validity of a fill must never erase original missingness.

The current model requires a zero placeholder wherever its numerical
`history_mask` is false. A qualified fill therefore needs a NEW, explicitly
named usable-numeric-input mask that can be true for derived Q, while the
immutable original-observation mask stays separate and `was_missing=1`.
Coordinate/indicator channels can be valid for filled slots. Do not pass a
nonzero fill with the original false mask or relabel the fill as observed.
Failed fills retain a false usable-Q mask and their failure record. Coordinate
age and `was_missing` have true validity masks when their timestamps/status are
known; elapsed-since-real has a true mask only when a preceding real reading is
known. No false metadata mask may hide a derived numerical value.
The architecture already concatenates input masks; the primary H2 complete-case
mask is effectively constant, so it does not supply this missingness contrast.

Q/coordinate-age transforms use the SAME gradient-only observed statistics in
both sensitivity arms. New binary and elapsed-age channels use declared fixed
maps (0/1 and hours/168), not fitted statistics from stopping/outer periods or
synthetic fill values. Fit Q/coordinate-age statistics BEFORE augmentation using only
original approved masks; exclude every later artificial or real fill from that
fit. Existing model code/scalers/checkpoints are unchanged by this proposal.
H4 would add 256 history-projection weights at width64 to the
existing LSTM geometry; the resulting 68,161 parameters and new schema would
require qualification. It is not an exactly parameter-matched primary model.

## Small paired experiment, with separate real-gap coverage

Choose **LSTM, full inputs, L168, CPU, B64, width64, FP32** for a bounded pilot
because its local full-L168 batch throughput was the fastest measured candidate,
not because forecast performance is known. Do not expand training to all four
architectures or change the main matrix automatically.

Use CV1 first: gradient 2018-07-12…2020-05-01, inner stopping
2020-05-01…2020-07-31, outer development 2020-07-31…2021-07-31; preserve strict
H18 role boundaries and the proposed two-hour refit convention. Three gauges,
seed17, two newly trained H4 arms give **6 neural fits**, plus **3 ridge fits**.
The arms are:

1. H4 clean training, with the same indicator/age schema.
2. H4 block augmentation, with the same origins, targets, optimizer-update
   schedule, scaler statistics, stopping rule and architecture.

Here the matched schedule means equal per-epoch opportunities and maximum
allowances. Actual early-stopped epoch/update counts can differ and must be
reported; do not force one arm to match the other's outcome-selected stopping
epoch or call differently stopped fits identical compute.

The second arm replaces, rather than duplicates, selected training inputs, so
the number of batches per epoch is unchanged. Proposed corruption probability
is 0.5; length/location mixture comes only from gradient-history missingness
metadata, restricted to eligible one/two-hour gaps. If that empirical mixture is
unsupported, specify a balanced stress mixture separately instead of learning
one from outer events. Initially use interpolation plus carry-forward in the
augmentation recipe. Learned-fallback evaluation is then an explicit
out-of-recipe stress comparison, not proof of its optimal training treatment.
Comparing either H4 arm against old H2 weights alone would confound added inputs
and training changes; the paired H4 clean arm is necessary.

Both H4 arms use CLEAN inner-stopping standardized-Q MSE under the same saved
gradient-only target transform, strict-improvement rule, minimum10 epochs,
patience5 and maximum50 epochs. Artificial outer errors never choose an epoch.
This proposed monitor prioritizes clean-input generalization; it is not an
outage-optimized checkpoint criterion. Freeze that choice in the pilot protocol
before any pilot fit, rather than switching to whichever monitor looks better.

Evaluate the exact same complete primary origins and unchanged targets under
clean inputs and a fixed artificial-gap grid. Use mask seeds 101/211/307 shared
across arms and neural seeds. Interior placement is chosen independently of
future flow and model errors; deterministic trailing scenarios do not need
duplicate mask draws:

- Interior one/two-hour gaps, linear fill, three placements each: 6 passes.
- Trailing one/two-hour gaps, carry versus ridge: 4 passes.
- One interior plus one trailing hour, carry versus ridge, three placements: 6 passes.
- Unmodified clean history: 1 pass.

This is **17 inference passes per checkpoint**, not 17 fits. For the direct
carry-versus-ridge comparison, predeclare the common surviving six-lag context
from the artificial mask before any error is observed. Keep the broader
carry-only feasibility and ridge failures in separate coverage tables; never
silently improve the paired population by dropping model failures. A small
paired artificial-gap experiment identifies sensitivity to these imposed gaps;
it does not reproduce the mechanism of real observation loss.

Evaluate both reconstruction error on deliberately hidden approved observations
and downstream physical-unit H1…H18 RMSE, MAE, bias, NSE when defined, negative
frequency and failure rate. Keep each model's clean score and its paired
degradation. Stratify evaluation-only by ordinary/high flow, positive approved
hour-to-hour observed rises, and the fixed observed-peak catalog. These strata
must not drive issuance, routing, gap placement or augmentation probabilities.
Use gradient-calendar q95 for HIGH; it is not an official flood stage. Report
exceedance recall, false-positive rate FP/(FP+TN), and false-alarm ratio
FP/(TP+FP) separately with counts/undefined denominators. Report peak timing and
magnitude errors only where original references support the fixed event window;
do not select replacement events that look better.

Use paired whole-origin block resampling only under the separately reviewed
conditional uncertainty policy. Artificial-mask variation and final seed
variation remain visible. Deterministic input fills are not uncertainty
intervals. Actual gap magnitudes have no reconstruction truth; do not claim
their fill accuracy from artificial-gap tests.

After the pilot, an OPTIONAL final-development confirmation would repeat only
these two H4 training arms for three gauges × seeds17/29/43: **18 neural fits +
3 new ridge fits**. The ridge models are deterministic and shared across neural
seeds. Both stages together add **24 neural and 6 ancillary fits**, not yet
authorized. Predeclare practical clean-degradation and failure tolerances
before judging the pilot. Do not use the final-development confirmation to tune
gap length, fallback coefficients, augmentation mixture, or architecture.

Real-gap recovered origins are a distinct population. Forecast-input availability
does not require all 18 future targets. For this sensitivity, report each lead's
approved-target score support separately and retain unknown/rejected targets;
the primary strict H18 cohort remains untouched. Report the original matched
cohort, newly supported origins, and their union separately. A higher score on
the union is not a paired skill improvement over the primary cohort. No expanded
training cohort or target-masked training is authorized by this design.

## The 90% aspiration and the cheaper routing alternative

The completed census reports at least three distinct denominators per
gauge, lead and season: forecast-input support/all candidate origins;
input-plus-approved-target support/all candidate origins; and
input-plus-approved-target support/approved-target origins. Keep unique target
hours, repeated origin/lead cells, observed-peak support, and strict full-H18
cohort counts separate. A 90% claim must name which fraction and population it
means; 90% coverage is neither 90% accuracy nor evidence about hidden peaks.
The perfect-history/non-Q ceiling will expose gaps that no Q treatment can fix.
Unknown references stay UNKNOWN and cannot be assigned a high/low flow regime.

`reports/development-gap-recovery-v1.json` now provides the structural census.
For final development, each gauge has 8,766 candidate origins. The following
percentages are hypothetical input eligibility or approved-target support,
not successful numerical fills or model skill. The last column is the minimum
over 18 separate lead fractions among observed approved q95-exceeding targets.

| Gauge | History option | Input support / all origins | Worst-lead evaluable / all origins | Worst-lead known-HIGH support |
|---|---|---:|---:|---:|
| Galax | Strict L168 | 93.33% | 93.08% | 70.98% |
| Galax | Combined ≤2 total missing hours | 97.16% | 96.89% | 90.16% |
| Galax | Natural strict L24 | 98.25% | 98.00% | 87.56% |
| Sugar Grove | Strict L168 | 84.02% | 83.32% | 48.81% |
| Sugar Grove | Combined ≤2 total missing hours | 89.84% | 89.14% | 54.46% |
| Sugar Grove | Natural strict L24 | 95.73% | 94.93% | 95.24% |

Jefferson is 100% in these final-development columns for all three options.
Sugar Grove's combined-two-hour worst-lead support is 90.69% **among approved
target origins**, but only 89.14% among ALL origins. These answer different
questions. Short repair alone therefore misses Sugar Grove's named 90% goals,
and L24 alone misses Galax's known-HIGH goal. The independently reconstructed
union is now recorded in `reports/development-gap-routing-v1.json`, rather than
estimated by adding overlapping fractions:

| Gauge | Union input support / all origins | Worst-lead union evaluable / all origins | Worst-lead union evaluable / approved targets | Worst-lead known-HIGH support |
|---|---:|---:|---:|---:|
| Jefferson | 100.00% | 100.00% | 100.00% | 100.00% |
| Galax | 98.83% | 98.55% | 99.50% | 100.00% |
| Sugar Grove | 96.67% | 95.87% | 97.54% | 100.00% |

The union structurally supports all 192/193/336 known-HIGH unique hours and
9/6/11 cataloged observed peaks at every lead in this validation year. These
are named counterfactual support results, not successful fills, skill, future
coverage guarantees or evidence about unobserved peaks. They justify evaluating
the combined option; they do not adopt it. See the separate census decision
packet in `docs/coverage-sensitivity-census.md` for the full frontier.
The annual union rates do not establish 90% within every season or shorter
subperiod; a claim at either grain needs its own denominator and audit.

These per-lead denominators differ from the earlier complete-H18 aggregate
HIGH-cell coverage of 69.2% at Galax and 50.0% at Sugar Grove. The earlier figures
require every forecast target in an origin's H18 vector to be approved; the new
table counts a lead's approved reference independently. Neither is a unique-hour
or event percentage, and unknown-reference peaks are outside both known-HIGH
denominators. Preserve both rather than relabel the apparent change as recovery.

If the two-hour repair ceiling cannot meet the desired coverage, a lower-cost
alternative is input-status routing with already planned frozen checkpoints:
use the same architecture's L168 model when its strict inputs are supported,
otherwise its matched-training L24 model when the strict last24 hours and other
inputs are supported. Do not route by forecast errors, future targets, weather
thresholds, or a preferred architecture. This proposes **zero additional neural
fits**, but adds inference and a separate unmatched-coverage generalization
analysis. L24-only origins are outside the shared L168 training-comparison
population; routing does not reconstruct L168 history or establish an equivalent
L168 task. All four architectures already have these final checkpoints in the
180-fit matrix; applying the same route to each is not four new training arms.
No checkpoint exists merely because it is in the plan, and no route is adopted
before protocol approval and those authorized fits finish.

Neither short-history repair nor L24 routing alone meets every named coverage
goal; their structural UNION has now been counted separately. A possible fixed final-development
route is strict L168 first, qualified short-gap H4 L168 second, then strict L24,
without consulting targets or errors. A structural union is not measured repair
success or forecast skill. A working joint route needs a qualified H4 checkpoint
and therefore is NOT the zero-new-fit L24-only alternative. The proposed pilot
has no CV1 L24 checkpoint in the primary matrix: actual CV1 joint-routing tests
would require a separately counted/approved L24 fit extension. Do not silently
add those fits. Reusing already planned L24 weights applies to final development.

## Resources and execution gates

`reports/coverage-sensitivity-resource-proposal.json` binds the exact measured
B64 LSTM full-L168 rates and admitted fold counts. At the old H2 throughput,
the 6-fit pilot is approximately **0.28/0.70 hours** for 20/50 epochs; optional
18-fit confirmation is **2.15/5.35 hours**. Both total **2.43/6.05 hours** before
the extra artificial-gap inference passes. These are proxies for an unmeasured
H4/fill/augmentation implementation, not a measured completion time. The JSON
adds the 16 extra passes, preserves per-gauge arithmetic, and shows explicit
1×/1.5×/2× runtime factors as hypothetical overhead scenarios, not confidence
bounds. Ridge fitting, gap construction, metrics, resampling, checkpoint I/O on
new data, thermal effects and failed attempts still need timing. No cloud price
or new dollar allocation is inferred. The combined $150 ceiling is unchanged.

Before any sensitivity execution: review the census and named coverage goal;
approve a concrete fit/update/resource allowance; qualify the versioned H4
adapter and fillers on synthetic leakage tests; measure the new geometry; freeze
gap/feature/failure/augmentation/evaluation policies; bind all source/target/
case identities; and preserve primary results separately. Approval of the
180-fit primary is not approval of these additional fits.

The current action is census plus design BEFORE the main freeze. If the pilot is
approved later, first lock a pilot-specific protocol and bounded allowance, then
run only its CV1 development experiment. The main protocol remains unfrozen
pending review of the pilot tradeoffs. Any resulting policy or primary-cohort/
hyperparameter change requires an explicit recorded review; keep every
CV1-informed decision in the selection history and never describe CV1 as an
untouched confirmation set. No automatic final-stage expansion is authorized.

## Literature rationale and limits

GRU-D uses missingness masks and time intervals in a predictive model, supporting
an explicit missingness/age design rather than pretending filled values were
observed. Its demonstrations concern clinical classification; they do not
validate our streamflow fillers or turn the existing HYDRA GRU–Transformer into
GRU-D. [Che et al. (2018)](https://www.nature.com/articles/s41598-018-24271-9).

BRITS studies learned bidirectional imputation. Its access to both temporal
directions is a reminder to distinguish retrospective reconstruction from
issuance-time information. Here a right endpoint is usable only if it belongs
to the already available history; no full-record bidirectional imputer is
adopted. [Cao et al. (2018)](https://papers.nips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html).

Ren et al. compare LSTM and ARIMA gap filling in a groundwater-well network and
explicitly acknowledge future-relative information in training. Their task is
not our causal discharge forecast, so it motivates scrutiny of timing and abrupt
changes, not a superiority claim. [Ren et al. (2022)](https://hess.copernicus.org/articles/26/1727/2022/).

CSDI models conditional imputation distributions. It illustrates the distinction
between probabilistic imputations and a single deterministic replacement; it
does not establish uncertainty calibration for this study. Adding a diffusion
imputer would be a separate model, training and compute proposal and is excluded
from this small design. [Tashiro et al. (2021)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html).
