# Primary local training protocol v1

**FROZEN_AUTHORIZED_LOCAL_PRIMARY.** On November 3, 2026, the user answered the
pending method question with “Approved 100epoch.” This adopts a **100-epoch
ceiling** with the other reviewed choices retained: minimum 10, patience 5,
strict matched L168 cases without repairs, approximate independent basins,
conditional within-year exploratory uncertainty, all four model families and
three final seeds, and raw unclipped outputs.

The [machine-readable protocol](../configs/primary-training-protocol-v1.json)
records the exact reply and `protocol_frozen=true` /
`scientific_training_authorized=true`. The
[durable-controller verification](../reports/primary-campaign-controller-check-v1.json),
model qualifications and data qualifications have passed. Local execution is
authorized; paid cloud jobs and protected assessment remain excluded.

The controller binds the resulting **file-byte SHA256** into the private plan
and durable ledger before the first scientific fit. That binding is enforced at
every worker launch. The JSON deliberately contains no self-hash. Authorization
and qualification are distinct from an observed running process or completed fit.

## Fixed roster and local execution

The authority is the existing
[fastest-local v3 roster](../reports/local-seed-sequence-fastest-v3.json), file
SHA256 `f45aa1afe114db80b194288ecb06c7757ebe3928de3702fac9ff1114238310e9` and
plan SHA256 `8e087b82797b79e96177cf11c19d5127d74c6f5c09ff6e0680cdbc4b3fc3e43b`.
It contains exactly 180 jobs, all four architecture families and all three gauges
`03161000`, `03164000`, `03479000`. Its exact role/case identities, job IDs and
order remain authoritative; this protocol introduces no additional fit.

| Stage | Fits | Required development evidence |
|---|---:|---|
| CV1, CV2, CV3, seed 17 | 72 | Full/Q-only L168 for every family/gauge/fold |
| Final development, seed 17 | 36 | Full L168, matched Q-only L168 and matched full L24; first complete seed package after 108 fits |
| Final development, seed 29 | 36 | Same declared settings and cohorts; cumulative 144 fits |
| Final development, seed 43 | 36 | Same declared settings and cohorts; cumulative 180 fits |

CV is not repeated for seeds 29/43. Seed17 results can be reviewed while later
released seeds continue; no extra human pause is inserted between unchanged
stages. Results do not permit tuning, pruning an architecture/seed, changing
masks or selecting favorable checkpoints for the remaining seeds. A method
amendment needs a new version and consistent treatment of affected results.

The selected devices are CPU for LSTM and GRU–Transformer, MPS for Transformer
and Mamba2–Transformer. This follows recorded execution timing, not forecast
skill. Use one local training worker, one CPU/Torch inter-op thread, B64 and FP32;
retain partial last batches. Set `PYTORCH_ENABLE_MPS_FALLBACK=0` before imports.
Backend failure stops the affected job; it cannot silently switch to CPU or
migrate a checkpoint between backends. No CUDA/cloud job is included.

## Data, information boundaries and spatial interpretation

The snapshot is
`b41630b7df9a216874eb4ba38a10176d4b72092bd6f1ce2ad4c4f369db56948c`;
the source plan is
`0dad073824984fd7eeba88e4c313d77b4015dcccb94e8eb27297fcf7c6b39b0f`.
Exact role case counts/hashes come from
`reports/confirmed-180-fit-admitted-runtime.json`, SHA256
`9a8b5526b0956125aecc2ebca9a29d20e90e6b262c136225505cde18cd92b2ec`.
The protocol JSON records all 27 unique fold/gauge/input/history data views.

Primary membership is `common_l168_complete`: complete original L168
history, admitted HRRR/soil inputs and all 18 original admitted targets. Full,
Q-only and L24 comparisons use the same per-gauge case membership; they do not
require a cross-gauge intersection. Natural L24's extra cases are coverage
diagnostics rather than additional matched training cases. No short-gap repair,
H4 augmentation, ridge, target fill or missingness-pilot fit is inserted. Strict
coverage is not guaranteed to exceed 90% at every gauge; excluded intervals remain
part of the coverage interpretation.

USGS admission requires A, allows only additional R flags, and rejects P, e and
unknown flags. Retain original qualifier order/duplicates, source IDs and masks.
Input and target policy applications remain distinct. Network-boundary zero
placeholders never become observed values or filled targets.

C is both the exact UTC hourly origin and information cutoff. H18 targets are
C+1…C+18; L168 history is C−169…C−2 and L24 history is C−25…C−2. Use the declared
two-hour simulated observation/weather/soil delay. Select one latest eligible
complete HRRR cycle, with explicit older extended-cycle fallback and no cycle
stitching. Initial soil is separate with maximum age 8 hours. Source cycle age and
native leads remain provenance, not extra predictor channels.

| Role set | Gradient interval | Inner stopping | Outer development |
|---|---|---|---|
| CV1 | `[2018-07-12,2020-05-01)` | `[2020-05-01,2020-07-31)` | `[2020-07-31,2021-07-31)` |
| CV2 | `[2018-07-12,2021-05-01)` | `[2021-05-01,2021-07-31)` | `[2021-07-31,2022-07-31)` |
| CV3 | `[2018-07-12,2022-05-01)` | `[2022-05-01,2022-07-31)` | `[2022-07-31,2023-07-31)` |
| Final development | `[2018-07-12,2023-05-01)` | `[2023-05-01,2023-07-31)` | `[2023-07-31,2024-07-31)` |

All bounds are UTC and half-open. Every H18 target remains strictly inside its
role. At simulated refit B, the outer-start timestamp, gradient and inner labels
also satisfy `last_target + 2h <= B`. This removes 18 final gradient/outer origins
and 19 inner origins. The last inner origin is B−20h; the last outer origin is
outer-end−19h. Bounds are checked from actual timestamps, not inferred from
surviving rows. Neither original publication/revision availability nor fitting
elapsed time is reconstructed. Store actual artifact times separately.

Use the numerically reproduced 514-row weights unchanged as **approximate
independent basin footprints**, preserving accepted outlet/area discrepancies
and original hold flags. This scientific interpretation was adopted in the
November 3 method confirmation. No surveyed exact-outlet, forced-nesting or conservation claim is
introduced. No protected `[2024-07-31,2025-07-31)` assessment data or its label
tail enters this execution, including as unselected bundle rows.

## One declared model/training configuration

Each fit is a fresh per-gauge model/optimizer with its declared seed. Common
settings are width 64, dropout 0.1 and FP32. Full-input H2 uses Q and coordinate age;
weather uses precipitation, temperature, specific humidity and SWE; initial
state uses soil fraction and age. The matched Q-only adapter retains Q/history
age, uses known horizon/18 and empty state, and keeps targets outside forward.
Model implementation hashes and all 12 qualified parameter counts are recorded
in JSON. Transformer blocks use four heads, feedforward width 128 and one
encoder/decoder layer where applicable. The explicit Mamba2 layout is also bound
there; no pretrained parameters or architecture search is used.

Fit affine input statistics only on approved observed gradient tensor cells.
Fit one pooled target mean/scale per gauge/fit across H18 gradient labels.
Use population variance (`ddof=0`), scale 1 for a constant channel, and fail on
absent training support. Overlapping tensor-cell occurrences contribute; they
are not independent observations. Full/Q at a given lookback must agree on
history and target transforms. L24 uses the same fitting rule but can have
different history statistics. Inner/outer/protected values never fit statistics.

Use AdamW: learning rate 1e-4, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01,
`foreach=False`; no scheduler, gradient clipping or automatic mixed precision.
Shuffle only within the fixed gradient membership. Loss is masked standardized-Q
MSE with equal lead contribution, without event weights or log-Q. Inverse-transform
through the saved affine map. Retain raw linear-head predictions without clipping
and report negative frequency.

The adopted stopping rule is **maximum 100 epochs, minimum 10, patience 5 and
`min_improvement=0`** on clean inner-stopping standardized-Q MSE. Improvement is
strict: `loss < best_loss − min_improvement`; exact ties retain the earliest best
state. The selected best epoch can precede 10 although training completes at least
10 unless a technical/resource stop intervenes. Evaluation uses the saved best
state. Neither gradient loss nor outer scores replace the inner stopping monitor.
A resource-exhausted run is not a successfully early-stopped fit.

## Immediate per-job development scoring

After each job, authenticate terminal training, selected-best checkpoint,
physical-unit predictions and matching metrics. Complete prediction generation
before retrieving outer labels. Produce per-gauge/lead NSE, MSE, RMSE, MAE,
signed bias (prediction minus observation), negative frequency, counts, exclusions
and exact label/sample identities. Missing or nonfinite requested predictions
fail even where reference labels are masked; no successful-row intersection is
allowed. Invalid admitted references and metric overflow also fail.

The primary panel is the equal arithmetic mean of **all 18 lead NSEs**. Every
lead requires 168 distinct target hours on 28 distinct UTC dates and positive
reference variance. One undefined required lead makes the primary unavailable;
retain descriptive below-floor metrics and reasons. Do not omit a lead/fold or
replace an unavailable score with zero.

CV uses the equal mean of all three complete fold metrics at seed 17. Final
development uses the equal mean of the three seed metrics after 17/29/43 finish,
not metrics of averaged predictions or a best-seed result. There is one fixed
configuration; any exact configuration tie uses the predeclared order, never a
rounded-score or favorable-seed choice. Poor finite performance and ordinary
early stopping are valid outcomes, not grounds to remove a job.

The seed barriers require every expected job's authenticated training,
checkpoint, prediction and metric receipts. The first complete seed package is
single-seed development evidence, not protected assessment or a poster-complete
result. It contains the 72 CV and 36 final-development jobs. The next unchanged
seed can run while that package is reviewed.

## Declared postprocessing from saved predictions

Event/uncertainty figures remain a postprocessing stage and must be marked
pending until produced. They do not block the next seed once the immediate
per-job evidence barrier is complete. They must fit the remaining active resource
allowance or receive a separate explicit bound; no hidden compute expansion is
permitted. Preserve saved predictions and source masks so this work does not
require refitting or selecting another checkpoint.

Use the predeclared unique-gradient-calendar q10/q90/q95 regime thresholds.
HIGH is Q>q95, not an official flood stage. Retain the existing observed peak
catalog: descending approved q95 exceedances, earliest-UTC ties, 168-hour separation,
peak−48h/peak+72h windows clipped to the outer interval, first-three representative
hydrographs. Keep unsupported events and unavailable metrics; do not select
replacement events using model errors. Short-window metrics are descriptive.

The adopted uncertainty interpretation is **conditional within-year exploratory**:
2,000 paired noncircular moving-block draws by contiguous UTC season segment;
14-day primary blocks and mandatory 7/28-day sensitivities; seed 20261102;
2.5th/97.5th percentiles. Keep the complete origin calendar, all 18 leads/models and
three fixed gauges, with common draws across comparisons/seeds. Recompute weighted
metrics on original unique coordinates and recheck distinct positive-weight
support. CV's joint family has 30 primary contrasts; final development has 42.
Do not redraw jointly invalid draws; more than 100 invalid draws makes that
family's primary intervals unavailable. Otherwise use one joint-valid pool per
family and report its size. Seed dispersion remains separate. No simultaneous
family-wise 95% coverage, independent-year/catchment inference or calibrated
forecast prediction intervals are claimed.

NWM remains a future separately qualified external matched analysis. It does not
gate independent training/scoring or enter the H1–18 primary aggregate by
dropping horizons. A two-hour-old native 18-lead issue supports origin H1–16 only.
No ridge, repair pilot or excluded persistence baseline is added here.

## Hard run, recovery and artifact bounds

| Bound | Adopted value |
|---|---:|
| Logical maximum epochs per job | 100 |
| Explicit resumes per job | At most 3 |
| Replay reserve per job | At most 30 epoch-equivalents |
| Logical updates across 180 jobs | 9,100,400 |
| Physical entered updates including replay/failure reserve | **11,830,520** |
| Largest per-job logical / physical update ceilings | 63,400 / 82,420 |
| Wall deadline per job / campaign | **6 hours / 336 hours** |
| Active parent/child RSS plus MPS driver allocation | **4 GiB** |
| MPS allocator cap | **2 GiB** |
| All new campaign outputs, including bookkeeping/staging | **8 GiB** |
| Bookkeeping reserved within that output total | **2 GiB** |
| Required free local disk | At least 8 GiB |

The roster sums to 91,004 gradient batches per epoch. At 100 logical epochs and
30 replay epoch-equivalents, physical maximum is `91,004 × 130 = 11,830,520`.
Each job is bounded by its own batch count times 100 logical / 130 physical epochs;
the reserve is not extra training epochs or new fitting opportunities. A different
confirmed epoch ceiling requires recomputing these ceilings before adoption.

Persist entered-step accounting before an optimizer call and confirm it after
completion. Failed/interrupted entered steps are never refunded. At most three
explicit resumes preserve the same protocol, data, backend, sampler/RNG and
budget ledger. Neither a resume nor a fresh process resets the job/campaign
clock. There is no automatic replacement fit or silent backend switch.

Write an initial checkpoint, then every 10 completed epochs and at termination:
at most 21 per job including recovery reservations (3,780 across 180). A
100-epoch uninterrupted fit needs at most 11; the larger finite cap preserves
room for recovery artifacts within the same byte budget. At an already saved checkpoint epoch, reuse
that checkpoint for the terminal reference; do not add a duplicate or overwrite
immutable evidence. Each full state retains current/best model, optimizer,
sampler/RNG, transforms, identities and stopping decisions. Resume from the latest
verified committed state; additional recovery artifacts still count toward the
same byte/checkpoint limits. Keep old artifacts; no automatic deletion is allowed.

The eight-GiB limit includes all new checkpoint/prediction/metric/log/staging
bytes. Reserve two GiB for bookkeeping, leaving six GiB for other artifacts;
check actual growth and free space before writes. Time, physical updates, memory
or disk exhaustion produces a visible unfinished stop, not a claim of complete
training. These are hard allowances, not promises that every fit reaches 100
epochs or that the complete campaign finishes within the deadline.

## Existing evidence and final release gate

The [current trainer qualification](../reports/primary-campaign-qualification-v1.json)
passed all 12 planned backend/feature/lookback geometries with 36 synthetic updates,
exact fresh-process recovery and best-state inference. Its file SHA256 is
`b69ae171e6972ae51fb141a586d59fe89938a65febfd9254fab48b0d9ffc6e1a`.
These are consumed synthetic allowances, not scientific fits to rerun blindly.

The [real data-view qualification](../reports/primary-campaign-data-qualification-v1.json)
passed all 27 declared views and paired-role checks, with zero model calls,
scaler fits and optimizer updates. Its file SHA256 is
`4d0e45a1f2b08a21c93b0355e53073401f593ed4e9bc43f2e2270096e19fc222`.
Outer arrays were authenticated; the predictor-slice exercise was limited to
the first 64 cases per view. This does not claim that scientific fits have run.

The durable controller qualification passed 908 tests with 3 optional GIS skips
and independent review. The human method reply is recorded; the controller must
still authenticate the frozen protocol-file identity and resource preflight at launch.
The root controller consumes exactly the six `training` and six `limits` keys
listed in JSON; additional methodological detail is outside those strict objects.
User review of a complete seed report does not reopen methods automatically.
