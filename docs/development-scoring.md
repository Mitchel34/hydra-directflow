# Proposed development scoring rules

`hydra_directflow.scoring` implements a proposed development scoring policy and
a plan-by-default scoring runner. It does not freeze the protocol, fit a model,
authorize a scientific run, or enable protected assessment. Its tests use only
synthetic outcomes and fake predictors; no saved study model has been scored by
this chunk. Training/model code is unchanged.

## Metrics and aggregation

For each gauge, lead, fold and seed, errors are prediction minus observation in
physical discharge units, m³/s. The scorer reports NSE, physical MSE, RMSE, MAE,
bias, negative-prediction frequency, requested and admitted sample counts,
exclusion counts, distinct target hours/dates, and exact sample/label digests.
Negative predictions remain unclipped. Negative frequency is the fraction of
scored predictions below zero; its denominator is the admitted-label count.
Every metric at one lead uses the same admitted-label mask.

NSE is `1 − sum((prediction − observation)²) / sum((observation − mean(observation))²)`.
The reference mean is calculated separately on that panel's admitted labels.
The proposed primary panel score is the arithmetic mean of **all 18 lead NSEs**.
It does not pool lead samples or weight leads by discharge variance. This gives
each forecast lead equal influence, including low-variance leads that may have
large negative NSE.

The proposed primary support rule requires **at least 168 distinct target hours
across at least 28 distinct UTC dates at every gauge/lead**, plus a mathematically
defined NSE with positive reference variance. These are predeclared computational
and reporting gates, not evidence of independent observations, statistical power,
or event uncertainty. The annual study panels motivate a conservative reporting
floor; the exact threshold remains a choice to freeze before outcomes. Overlapping
H18 forecasts share observations and are not independent samples. Raw per-lead
descriptive metrics, counts, and mathematically defined NSE remain available below
this floor. Short event windows can therefore retain descriptive errors without
claiming an annual-support primary score.

Zero support, constant reference discharge, and reference-variance numerical
underflow receive explicit undefined status. A single unavailable primary lead
makes its whole primary panel unavailable. Missing or nonfinite requested
predictions fail the run, including cells whose labels are unavailable. The
scorer cannot silently intersect successful predictions to improve a result.
Missing or rejected reference labels are masked and counted; admitted labels
must be finite and nonnegative. Metric overflow fails explicitly.

CV uses seed 17 and requires the three chronological outer folds, with exact
candidate/gauge/feature identities. Its selection score is the **equal mean of
the three complete fold scores**, regardless of fold size. It is not a score on
pooled rows/SSE. If any fold's primary is unavailable, the aggregate is unavailable;
no fold is omitted. Per-lead descriptive metrics are also equal means across the
three folds, with `null` when any required constituent is undefined.

Final development reporting retains **all three seeds, 17, 29 and 43**, on the
same exact labels and original comparison cohort. Each reported aggregate is
the mean of the three seed metrics, including RMSE and NSE. This is neither a
best-seed selection nor a metric of averaged predictions. All seed reports remain
attached. Missing seeds are rejected; undefined primary panels propagate to an
unavailable aggregate. No across-gauge weighting or event uncertainty rule is
introduced here.

Configuration selection is restricted to comparable candidates in the same
architecture/gauge/input arm, with the same policy and per-fold label/sample
identities. Every declared configuration must have a complete defined CV primary.
The exact highest score wins. Exactly equal, unrounded floating-point scores
use a configuration order supplied before outcomes; all tied IDs and the order's
digest are retained. There is no rounded-score tie, secondary-metric tie break,
or favorable seed choice. Missing/undefined candidates require an explicit
protocol decision; this helper cannot quietly remove them or authorize retries.

## Best checkpoint and label-free prediction

Checkpoint selection and model selection have different criteria. The saved best
checkpoint is selected by **inner-stopping standardized-Q MSE**, using only the
saved train-only affine transform. Training MSE and stopping MSE are not physical
MSE and are not the primary reporting NSE. The existing stopping rule accepts
only `loss < best_loss − min_improvement`; exact ties retain the earliest accepted
best epoch. No outer outcome chooses an epoch.

`run_development_score(...)` delegates to
`BestCheckpointPredictor.load(...)` and `input_only_view(...)` from
`hydra_directflow.inference`. It requires the identified `best_model` state and
saved best epoch; it never falls back to last weights. The checkpoint is SHA-bound
and the inference module replays its recorded stopping decisions. That module
applies saved input transforms and inverse-transforms predictions to m³/s without
fitting statistics, constructing an optimizer, or reading target values/masks.

The runner binds training source/cohort identities separately from the current
inference source/cohort, plus the gradient-fit case digest, gauge, architecture,
feature schema, history length and seed. It verifies exact history/origin/H18
coordinates and model identity on every returned batch. All predictions finish
before the runner first retrieves outer target values or masks. Prediction labels
are not supplied to the model. Q-only calls require the existing Q-only bundle
adapter; the prediction path receives its declared horizon channel and empty
state, not original HRRR/soil inputs.

## Calling the runner

Call `run_development_score(...)` with explicit development outer indices/window,
checkpoint SHA, training and inference source/cohort SHA identities,
`gradient_fit_case_ids`, label-policy/source identities, and a `ScoreIdentity`.
By default it returns a metadata-only `PLANNED_NOT_EXECUTED` plan, including code
hashes and a `plan_sha256`; it loads no model and reads no numerical inputs or
labels. Actual invocation additionally requires `execution_enabled=True` and the
exact `approved_plan_sha256`. These fields structurally bind the request; they
are **not** approval authentication or protocol adoption. Policy status remains
`PROPOSED_NOT_FROZEN` in plans and results.

The caller must first use the admitted training-bundle API to prove the separate
gradient-fit and inner-stopping target availability at refit time. This runner
does not reconstruct that proof from opaque source digests. It additionally
rejects gradient-fit H18 labels that do not precede the requested outer window,
and requires consistent hourly origin/H18 clocks for the entire supplied source
bundle. Assessment/crossing origins or targets anywhere in that bundle, including
unselected rows, are forbidden before numerical input or label access.
The default bound is 20,000 selected cases (at most 200,000 source rows),
128 cases per prediction batch, 300 seconds, and 4 GiB observed process RSS plus
MPS driver allocation. The caller can lower bounds; wall time is capped at
600 seconds. Guards are cooperative between batches, not OS isolation. CPU/MPS
are supported through the qualified inference component; CUDA is not enabled.
No network, source download, scaler fitting, optimizer update, automatic retry,
public prediction dump, or event threshold fit is implemented here.

Focused validation:

```sh
PYTHONPATH=src .venv-offline-check/bin/python -m unittest discover -s tests -p test_scoring.py -v
```

Tests verify physical arithmetic, support gates, undefined/missing handling,
equal-fold weighting, all-seed metric means, exact ties, coordinate/assessment
guards, default nonexecution, separate source parents, and label access only
after all fake-inference batches. They construct no model or optimizer.
