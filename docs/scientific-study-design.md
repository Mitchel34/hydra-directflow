# DirectFlow scientific continuation: reconciliation and proposal

Prepared November 2, 2026. Baseline reconciliation: **11:44 Central / 17:44 UTC**,
repository `a4d3fd1`. Subsequent implementation of the confirmed quality adapter
is recorded separately. All unconfirmed scientific settings below are proposals.
This document does not authorize data purchases, further staging, cloud compute,
protected-data access, assessment scoring, security changes or communications.

## Reconciled state

| Work | Status at reconciliation | Timestamp and evidence |
|---|---|---|
| Repository | Clean; local and remote main both `a4d3fd1`; no newer implementation found | Nov 2, 11:44 Central; Git and current handoff checked; another context-review thread reported no changes |
| HRRR acquisition | Complete accounting: 61,658 complete, 166 documented upstream gaps, zero remaining | Sealed Oct 27 09:47:57 UTC; qualified index, manifest and three final audits rehashed on Nov 2 |
| August inputs/cache | Complete; 127 archives, 127 commits and 130 derived entries reverified | Nov 2 reconciliation; 17,041,091 derived bytes; zero staging/quarantine entries |
| August preprocessing | Passed: 744 origins, 2,232 cases, 2,046 complete; missingness retained | Completed Nov 2 09:43:57 Central; `month-timing.json` and `month-data-quality.json` |
| Engineering tests | 217 recorded passes; all 50 corresponding code/test hashes matched at reconciliation | Executed Nov 2 09:43:41 Central; `month-execution-check.json`; not represented as a new rerun |
| Processes | No related acquisition, retrieval, preprocessing or timing worker observed locally | Local process snapshot Nov 2 11:44:43 Central. Cloud resource state was not queried |
| Primary USGS quality | User-confirmed, not yet implemented at `a4d3fd1` | `scientific-decisions.json`; implemented as an explicit tested adapter during this continuation |
| Scientific fits, model selection, rolling CV, protected scoring | Pending; no scientific forecast-skill findings | Execution gates disabled; earlier 52 real-data engineering and 878 synthetic updates are not scientific fits |
| Full study staging and paid compute | Pending separate inventory/budget approvals | August's completed 1 GiB/$1 read allowance covers that month only; billing estimate is not an invoice |
| Poster | Existing design shell, not a completed scientific poster | Accepted-abstract snapshot and layout references reviewed; current framing needs coauthor reconciliation |
| Monitoring/CI | Local journals and Grafana template exist; hosted service/CI not active | No hosted deployment receipt; workflow credentials remain separately controlled |

Older completion-report preparation fields and the October pause record are
superseded by newer repository/checkpoint evidence. Historical receipts are kept.
No acquisition controller was restarted. The current cloud VM inventory and
provider invoice remain unverified; absence of a local worker is not a cloud
shutdown receipt.

## Confirmed decisions and decisions still to freeze

| Decision | Status / proposed resolution |
|---|---|
| Task | Confirmed: direct discharge H1–18, three gauges; NWM external only |
| Quality | Confirmed A required, optional R, no P/e/unknown flags; preserve original qualifiers and never fill targets. Separate history and target policies in the adapter |
| Persistence | Excluded from the active study by the user's Nov 2 account of advisor Zoom guidance; not transcript-verified. Learned Q-only controls remain |
| Global dates | Preserve train `[2018-07-12,2023-07-31)`, validation `[2023-07-31,2024-07-31)`, protected origins `[2024-07-31,2025-07-31)`; no random split |
| Information boundary | C equals cutoff; exact hourly history and targets C+1…C+18. Proposed primary delay remains two hours for observations/weather/soil, explicitly simulated |
| Weather/soil | One eligible complete weather run, declared older-run fallback, no stitching. Separately timestamped soil; proposed maximum age eight hours |
| Lookback | Compare L24/L168 on common L168-admissible cases, and separately report each lookback's available coverage; select primary before assessment |
| Missing inputs | Propose complete-input primary cohort first, after coverage audit. Partial-history admission and outage augmentation require separate rules. Numerical mask placeholders are not observations |
| Targets | Preserve per-target quality masks. Decide whether training admits partially labelled H18 vectors; never change this by architecture or fill labels |
| Spatial | Document existing approximate independent basin weights and accepted discrepancies; no forced nesting/conservation. Whole-study spatial admission remains to be recorded |
| Fit grouping | Propose separate gauge models initially; shared multi-gauge fitting would be a distinct experiment |
| Transform/loss/output | Freeze train-only input/target scaling, physical-Q versus log-Q, loss weights, output constraints and inverse transformation. No post-hoc clipping chosen from assessment scores |
| Selection metric | Propose equal-weight mean lead-wise NSE per gauge, averaged equally over development folds; report RMSE/MAE/bias/NSE and support. Undefined NSE needs a predeclared handling rule; confirm before tuning |
| Search/seeds | Start one declared configuration and one development seed; any expanded search has the same allowed configurations/stopping opportunities by architecture. Propose final seeds 17/29/43; never select a favorable seed |
| Stopping | Proposal: maximum 50 epochs, minimum 10, patience 5 on an internal stopping period. These numbers remain unapproved and must be costed |
| Weights at assessment | Primary proposal preserves canonical train/validation roles: freeze selected checkpoints after development; no automatic retrain on validation. A preassessment refit on combined development data would require an approved amendment and new train-only transforms |
| Effect/uncertainty | Predeclare a meaningful-effect margin, contrasts, dependence-aware resampling and multiplicity handling before results; seed variability is separate |
| Assessment | Locked protocol/selection plus qualified inputs and explicit permission are necessary. A hash, this document or the planning prompt is insufficient |

The two-hour delay does not reconstruct historical observation revisions.
Results must be described as retrospective forecasting under simulated
availability using the retained observation vintage, not demonstrated real-time
operational performance.

## Controlled hypotheses

For each contrast, predeclare its metric direction, effect-size margin δ, gauges,
leads/regimes and comparison cohort. Support requires a paired effect with
uncertainty supporting improvement beyond δ. Evidence that excludes a meaningful
gain, or supports degradation, weakens the hypothesis. Wide intervals, opposing
folds, missing controls or too few independent events leave it unresolved.
Negative and inconclusive findings are publishable outcomes of this design.

| Hypothesis | Controlled comparison | Limit on interpretation |
|---|---|---|
| H1: weather and soil add forecast information | Full input versus matched learned Q-only models, same origins/targets/training opportunities; predeclare changing-flow/long-lead strata | Identifies the joint input bundle. Separate Q+weather and Q+soil arms are needed to isolate either source |
| H2: learned attention helps information combination | Retrained matched decoder variants with learned versus fixed/uniform retrieval, common input/fusion schema and measured capacity | Current LSTM/Transformer differences alone cannot isolate attention. Inference-time attention removal is sensitivity, not a learned-component test |
| H3: GRU–Transformer benefits from components | Hybrid versus standalone GRU and Transformer with comparable budgets | Without standalone GRU, claim only the observed four-family comparison; beating one comparator is not evidence of complementarity |
| H4: Mamba2–Transformer offers useful accuracy/cost tradeoffs | Hybrid versus standalone Mamba2 and Transformer, including full-model latency/memory | Encoder efficiency does not imply hybrid efficiency. Missing standalone control prevents a component-complementarity conclusion |
| H5: longer history helps | L24 versus L168 on common cases, plus coverage and cost tables on their natural cohorts | Different case coverage must not be mistaken for a lookback effect |

GRU gating, attention-based sequence fusion and selective state-space memory
provide methodological rationales, not hydrologic skill findings:
[Cho et al.](https://aclanthology.org/D14-1179/),
[Vaswani et al.](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html),
[Dao and Gu](https://arxiv.org/abs/2405.21060).
Mamba2's language-model/kernel speedups cannot be transferred to 24/168-hour
HYDRA sequences or its current unfused implementation.
Attention over time/horizon tokens is not automatically variable importance or
causal explanation; use ablation/perturbation and stability diagnostics:
[Jain and Wallace](https://aclanthology.org/N19-1357/) and
[Wiegreffe and Pinter](https://aclanthology.org/D19-1002/).

## Actual architectures and proposed controls

Current schema: history has discharge and age; forecasts have precipitation,
temperature, specific humidity and SWE; initial state has soil fraction and age.
Boolean masks are concatenated as input channels, not attention-padding masks.
Masked numerical placeholders must be zero. Soil is repeated as conditioning
information on horizon tokens, not represented as a predicted future soil state.
Sinusoidal history-token and actual horizon-offset encodings, dropout 0.1,
LayerNorm and a shared linear scalar head produce `[batch,18]` transformed-Q
outputs. Targets never enter the forward call. No automatic nonnegativity exists.

| Implemented family, width64 | History encoder | Forecast decoder | Trainable parameters |
|---|---|---|---:|
| LSTM | One LSTM; final hidden/cell state | One LSTM initialized from that state, consuming issued forecast tokens | 67,905 |
| Transformer | One 4-head encoder layer, FF128 | One 4-head Transformer decoder, FF128 | 85,057 |
| GRU–Transformer | One GRU; all history hidden states retained | Same Transformer decoder | 76,545 |
| Mamba2–Transformer | RMS normalization plus selective SSD block and residual | Same Transformer decoder | 86,093 |

These counts were checked by CPU initialization without forward calls, optimizer
updates, source reads or network. Position buffers are not trainable parameters.
Shared width is not matched capacity. Older smaller parameter-count receipts
used a different weather schema.

The Transformer decoder has self-attention among all 18 issued-weather horizon
tokens and cross-attention to history. All these weather forecasts must already
be admissible at C; this is not future observed-discharge access. LSTM decoding
is sequential in horizon tokens, without predicted-discharge feedback. The
Mamba encoder uses input-dependent Δ/B/C, gated selective state propagation,
causal convolution width4, state64, expansion2 and SSD chunks32. It is not
ordinary attention and does not use a demonstrated fused CUDA path.

Proposed additions, not implemented or approved:

- Standalone GRU: GRU history encoder followed by a GRU issued-weather decoder
  and the same output head; derived count 51,265 parameters. Planning allowance
  2–4 focused hours for implementation and CPU/MPS recovery/gradient tests.
- Standalone Mamba2: concatenate projected history and known forecast tokens,
  use causal Mamba2 processing, read the last 18 outputs, with no Transformer.
  One/two blocks give derived counts 35,853/70,361. Propose two blocks for a
  closer capacity comparison; allow 4–8 focused hours including qualification.
  These are engineering judgments, not measured completion or runtime promises.

The proposed causal Mamba decoder cannot see later forecast tokens as the
Transformer decoder can. This comparison tests an architecture package; a
separate controlled decoder study is needed to isolate attention itself.
If either addition is infeasible, keep the four required families and narrow
the conclusions rather than claiming hybrids beat untested components.

## Proposed expanding-window cross-validation

This design is **not implemented or approved**. It stays within the existing
training interval and preserves the separate 2023–24 validation interval for
the final development check. Dates below are UTC and half-open.

| Fold | Gradient-fit data | Inner stopping data | Candidate-scoring origins |
|---|---|---|---|
| 1 | 2018-07-12 → 2020-05-01 | 2020-05-01 → 2020-07-31 | 2020-07-31 → 2021-07-31 |
| 2 | 2018-07-12 → 2021-05-01 | 2021-05-01 → 2021-07-31 | 2021-07-31 → 2022-07-31 |
| 3 | 2018-07-12 → 2022-05-01 | 2022-05-01 → 2022-07-31 | 2022-07-31 → 2023-07-31 |

Refit independently once per annual fold, with new optimizer/scaler state and
predeclared seeds. Fit scalers, transformations and event thresholds only on the
gradient-fit partition. Inner stopping data chooses a checkpoint; outer scores
must not drive that fit's early stopping. Freeze each checkpoint for its entire
candidate-scoring year. Do not refit every forecast hour.

Every forecast advances its origin hourly and updates only admissible inputs;
that is rolling-origin forecasting. Repeated independent fit/score folds for
configuration selection are rolling-origin model-selection CV. They are distinct
operations. Validation histories may use earlier observed values—including
earlier validation observations—when available at C; this does not authorize
using future validation labels in gradient updates.

Purge any origin with a target outside its assigned fit/stopping/scoring interval.
If the two-hour label-availability assumption also governs simulated refits at
time B, require the last label to be available by B: `C+18h <= B-2h`, hence
`C <= B-20h`. This is stricter than merely excluding labels at B. Freeze that
refit-availability convention before execution; do not claim revised archives
reconstruct historical vintages. Input history may cross backward over a split
boundary; transformations still learn from the fit partition only.

Use a common per-gauge set of origin/valid-time cases and masks for all candidates
within each fold. For L24/L168, compare on the common L168-admissible cohort and
report natural-cohort coverage separately. The final development check uses the
canonical training interval, with May–July 2023 as its internal stopping tail,
and `[2023-07-31,2024-07-31)` for validation. It may support selection but is not
an untouched assessment. Primary proposal: no post-validation refit; freeze the
selected training-period checkpoints. Any combined-development refit needs a
reviewed protocol amendment before assessment access.

Initial budget: one common configuration and one CV seed, expanding only after
measured feasibility. Four models × three gauges × three folds = **36 CV fits**;
three-seed final-development fits add 36, giving **72** before input controls.
Six architectures yield 54+54 = **108**. A full matched Q-only factorial doubles
those relevant counts. With two CV configurations, four-model totals become
72 CV +36 final = **108** before controls. Unequal fold lengths and per-model
throughput mean these are fit counts, not wall-time multipliers.

This design follows forward evaluation principles; it is an inference tailored
to this study, not a literature-mandated schedule:
[time-series CV](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation),
[structured-data validation](https://florianhartig.github.io/publications/roberts2017cross/).
Random k-fold validity results under restricted autoregressive/error assumptions
do not establish validity for this issued-weather overlapping-H18 experiment:
[Bergmeir et al.](https://robjhyndman.com/publications/cv-time-series/).

## Highest-value implementation order and approvals

1. Implement/test confirmed quality admission and audit eligibility by gauge,
   split, lookback and exclusion reason using available local data. Preserve
   historic strict-A receipts; distinguish partial inventories from full coverage.
2. Fix late-development source admission and build a controlled July 2024
   development-only export. The source-init-plus-48h guard currently rejects
   valid July29–30 forecasts. Parquet filtering alone does not prove zero internal
   decoding of protected rows: review the administrative extraction scope,
   isolate outputs and hashes, and keep protected scoring locked.
3. Produce an exact full-development source inventory and a bounded streaming
   plan. About44 GiB of raw bundles/1.1 GiB of derived stores is a one-month sizing
   scenario, not a verified total. Cap temporary staging, retain reductions and
   receipts, and propose separate download/transfer allowance before fetching.
4. Build the scientific trainer and train-only transforms; qualify sampler,
   RNG, optimizer and scaler recovery through the next optimizer update. Then
   benchmark realistic CPU/MPS B16/32/64 batches with startup, validation,
   inference, checkpoint, RSS and failure accounting. The previously proposed
   24-case/600-update benchmark is not yet executed.
5. Review/freeze CV, primary metric, loss/transform/output rules, common cohorts,
   seeds, trial/epoch caps, ablations and component-control scope. Compute the
   full campaign estimate from measured batches and actual admitted counts.
6. Train/select on development only; lock all selected checkpoints and protocol.
   Access/score protected assessment only after existing gates and a separate
   explicit authorization are satisfied. NWM is matched externally and never
   determines HYDRA training eligibility; do not invent unsupported horizons.
   For example, native NWM leads1–18 at a two-hour availability lag support only
   C-relative leads1–16. Match its actual retained source clocks and the declared
   availability policy; do not relabel native lead18 as C+18.

Local implementation/tests are within the current request. Further scientific
source downloads, any paid compute/storage service, credential/security changes,
hosted observability and outbound messages require their separate approvals.
The August allowance is consumed for its scoped work, not a standing budget.

## Dependence, outages and figures

Use paired contiguous time blocks of whole 18-lead origin vectors, identical
block draws for models/gauges and original masks. Select block length/seasonality
handling on development data; propose 7-day blocks with 3/14-day sensitivity,
subject to residual/event-duration diagnostics. At least18h overlap must be
covered, but18h alone does not remove hydrologic dependence. Keep seed dispersion
separate; horizons, gauges and seeds are not independent replicates. Declare
multiple-comparison/exploratory scopes and report practical-effect uncertainty.
[Politis and Romano](https://doi.org/10.1080/01621459.1994.10476870).

Propose 6/24/72-hour development-only input-outage stresses on cases with observed
targets. Change only input availability/masks and retain target truth; all models
receive identical stress. At L24,24h and72h both remove all available history and
are not distinct within-window evidence. Inference-only stress diagnoses fragility;
training with outages is another experiment. Neither validates actual damaged-site
periods whose discharge truth is unavailable. Report error, coverage and abstention
together, including complete-case survivorship limitations.

| Figure | Evidence and selection rule |
|---|---|
| Basin/gauge map | Fixed original geometry, gauge/reach identifiers, explicit approximate-boundary caveats; no invented nesting |
| Timing/availability | C, simulated delays, one weather run, separate soil timestamp, history slots and H18 targets; distinguish actual artifact time |
| Architecture panels | Actual four encoders/decoders/masks/horizon tokens and verified parameter counts; proposed controls visually labeled |
| Splits/CV | Exact half-open intervals, purged boundaries and frozen versus refitted weights; proposed CV labeled until adopted |
| Gauge × lead metrics | Predeclared primary metric plus RMSE/MAE/bias/NSE, counts/exclusions and paired differences; NWM's genuinely supported subset separate |
| Event hydrographs | Freeze event definition, minimum separation and windows using development data. Example: top3 eligible independent peaks per gauge, ≥72h separation, fixed −48/+72h windows, deterministic ties and excluded-event list. Never select by model advantage |
| Ablation/uncertainty/cost | Matched input and component controls actually run, block uncertainty, seed variability, parameter counts and measured train/inference cost |

Event rules applied after assessment authorization may use observed peaks to
select events, but not model error. Missing or uncertain peaks must be disclosed;
there is no invented truth during outages. Flood/regime thresholds derive from
development data, not assessment-driven optimization.

## Poster, milestones and separate edge research

Narrative: motivation → hypotheses → data → methods → evaluation → results →
limitations. The accepted abstract snapshot describes NWM/ERA5 correction and
includes old improvement claims; the new study is direct HRRR forecasting.
Prepare a scope/claims comparison for coauthor review. Do not carry old findings
into the new results panels. No coauthor message or submission was sent.

Nov4 is a readiness checkpoint; Nov6 is a scientific-review/poster-draft target,
not a guarantee. If development or assessment readiness fails, provide an
explicitly preliminary/development-results poster in the existing AGU-themed
layout. Keep all four required architectures/gauges in the intended investigation,
reduce exploratory sweeps first, and never open assessment to rescue a deadline.

After the current experiment, investigate an edge system: calibrated water-level
and rain sensors, optional camera-derived water level, timestamped quality checks
and cached issued forcing feed a compact validated model locally. A stage-to-flow
rating relationship and its uncertainty need field calibration; local sensors
are not drop-in replacements for the training USGS series. Test camera occlusion,
lighting/privacy, rating changes, sensor drift, flooding, clock errors and outages.

Measure batch-one latency, energy, peak memory and export/quantization parity on
named hardware before choosing a device. Solar/battery sizing must cover poor
weather, cellular retries and brownouts. Buffer signed/versioned records locally;
send forecasts, uncertainty, source ages, data quality and device health through
authenticated store-and-forward transport. Use signed model updates, rollback,
per-device credentials, least privilege and explicit stale-input/no-forecast modes.
No autonomous safety-critical warning claim follows from the current research.

A larger regional model could combine calibrated site observations, local forecasts
and catchment/topology information, evaluated against local-only/cloud-only systems
with genuinely held-out sites/events. Explore documented data products and agreed
research APIs with hydrologic forecasting operators, including NWM teams; no existing
production NWM ingestion interface is assumed. This is future research: no new
hardware, deployment, collection, paid service or experiment expansion is authorized.
