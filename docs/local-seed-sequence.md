# Local campaign with complete results after each seed

The user requested local training and evaluation one seed at a time, so a complete
first-seed result can be reviewed while the later seeds run. This changes execution
order, not the confirmed 180-fit comparison matrix or its common cohorts. CUDA
speedup remains unmeasured; choosing local execution is not a finding that GPUs
would be ineffective. Paid cloud jobs remain held.

## Order and deliverables

The [generated roster](../reports/local-seed-sequence-v1.json) expands the existing
admitted-count authority into exactly 180 unique jobs:

| Stage | Fits in stage | Complete milestone |
|---|---:|---|
| CV1, CV2 and CV3, seed17 | 72 | Three development CV folds, 24 fits each |
| Final development, seed17 | 36 | **108 fits complete; publish the first complete seed package** |
| Final development, seed29 | 36 | 144 total; publish seed29 and updated comparison |
| Final development, seed43 | 36 | 180 total; publish all three seeds and seed variability |

Every stage contains LSTM, Transformer, GRU–Transformer and Mamba-2–Transformer
at Jefferson, Galax and Sugar Grove. CV contains paired full/Q-only L168 arms;
each final seed contains those 24 fits plus 12 full-input L24 contrasts. No
Q-only L24, ridge or additional missingness-pilot fits are inserted. CV is done
once at seed17; repeating it for29/43 would expand the study beyond180 fits.

For one fixed predeclared configuration, CV need not mathematically precede final
training. This roster nevertheless preserves the declared CV-first sequence.
Moving final seed17 ahead of CV would be a separate scheduling amendment, rather
than a silent omission of the planned validation evidence.

The first complete package includes all expected training completion records,
source/cohort/role identities, best-checkpoint hashes and matching development
predictions/metrics. Report per-gauge/per-lead NSE, RMSE, MAE, bias, negative
frequency, full/Q and L168/L24 contrasts, coverage/exclusions, supported fixed
events, selected/completed epochs, stopping reasons, runtime and failures. Label
it **single-seed development evidence**. Three-seed averages and dispersion wait
for all three completed final-seed reports.

All evaluation here is inside the development period, including2023–24 final
development validation. The2024–25 protected assessment remains sealed and is
not part of these seed completion barriers.

## Review while later seeds continue

Once a released campaign completes and verifies the first seed package, later
seeds can continue under the same frozen settings while the user reviews it.
There is no additional human approval pause between unchanged seeds. Keep all
architectures and all three seeds regardless of whether a finite score is poor.
Review is for interpretation and diagnosing technical failures, not changing
hyperparameters, masks, stopping rules, event selection or cohorts for29/43.
Any methodological change needs a new experiment version and consistent reruns
of affected seeds, with the original evidence retained.

The durable execution controller must require authenticated terminal training,
selected-best checkpoint, prediction and metric receipts for every expected job.
Elapsed time, a live process or a checkpoint file alone does not prove completion.
Missing/stale/mismatched receipts, numerical failures, leakage, recovery errors
or resource exhaustion block an affected stage. Undefined score support remains
explicit; it cannot be silently dropped or replaced with zero. Poor finite skill
and ordinary early stopping are valid outcomes, not reasons to omit a job.

## Timing and readiness

The same sequential CPU/MPS measurements imply these conditional modeled hours:

| Complete seed package | 20-epoch scenario | 50-epoch scenario | Proposed200-epoch ceiling scenario |
|---|---:|---:|---:|
| Seed17 including CV | 17.98 | 44.78 | 178.79 |
| Seed29 | 7.55 | 18.83 | 75.24 |
| Seed43 | 7.55 | 18.83 | 75.24 |

These are per-fit arithmetic extensions, not measured early-stopping durations
or delivery guarantees. They retain the previous loading/scaling, large-run
checkpoint, uncertainty/figure, thermal and failure exclusions. The first
complete seed report takes longer because it includes the one-time CV work.
No CPU/MPS concurrency speedup is assumed; one local fit at a time remains the
planning basis. The proposed high epoch limit still needs matching update/time/
storage limits; the six-fit pilot storage estimate is not a180-fit allowance.

`scripts/plan_seed_sequence.py` generates this metadata roster only. Seven
focused tests passed for exact counts, deterministic order, paired memberships,
backend assignments, rejected expansions and timing arithmetic. It performs no
model/data-value reads, fits, scoring or resource allocation. It is **not yet the
durable campaign controller**, and its listed barriers are an execution contract,
not a claim that runtime enforcement has been completed.

The remaining preparation is to bind the concrete reviewed protocol/epoch policy
and per-fit/global budgets, qualify larger main-study role-limited bundles and
their local execution integration, and implement the durable scheduler and report
barriers. Reuse completed data integrity, temporal isolation, recovery,
best-inference and scoring checks. Do not reopen resolved acquisition or CUDA
requirements as prerequisites for this local campaign.
