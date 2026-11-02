# Restartable development training components

`training.py` implements train-only affine transforms, explicit training sample
order, a bounded CPU/MPS update loop, inner early stopping and immutable
checkpoints. This is a prepared training component, not a frozen scientific
protocol, a complete experiment orchestrator or authorization to fit/score.
The full study still needs admitted cohort manifests, the agreed transforms,
loss/constraints/metric, CV and trial/seed budgets, and protected-data approval.

The current v2 implementation supports the full H2/W4/S2 input schema and a
matched Q-only H2/horizon1/empty-state schema, with one gauge per fit. See
`docs/q-only-training.md` for their separately bound cohort/feature identities.
It does not silently add standalone GRU/Mamba2 controls,
select a Q-only architecture, open protected data or choose a scientific winner.

## Data isolation and transformations

Before numerical admission, origin/history/target clocks must be exact UTC hours,
histories end at C−2 inclusive, every target is C+1…C+18, and all target clocks
precede protected assessment. Training and inner-stopping indices are disjoint;
every training target must remain strictly before the stopping boundary. The
caller must additionally prove each fold's source availability, interval end,
quality/cohort and role from its admitted manifest. The class cannot infer those
facts from an arbitrary source digest.

Affine standardization is an implemented candidate, not the selected scientific
transform. Statistics use only masked observed cells referenced by training
indices. Validation changes cannot alter them. Constant training channels retain
a scale of1 and an explicit flag; an entirely missing training channel fails
rather than borrowing validation data. Statistics, training case IDs and source
array digests are checkpointed. FP32 network encoding has ordinary rounding
error; raw physical arrays are unchanged.

Missing input cells use zero only at the network boundary and retain false
masks. Missing targets stay NaN with false masks; no fabricated target enters
the loss. Current mechanics use masked standardized-Q MSE and an unconstrained
linear head. Scientific transformation, loss and output constraints remain to
be frozen before comparisons.

## Sample order, epochs and stopping

Chronological order or within-training shuffling is explicit. Shuffling never
changes the chronological split. The sampler records its seed, epoch, cursor,
training membership and permutation digest; partial final batches are retained.
`fit_epochs` uses only the provided **inner stopping** subset. Outer CV scores
and assessment outcomes must never be supplied as that subset.

Maximum updates and stopping settings are explicit. A depleted update allowance
raises rather than calling an incomplete fit scientifically complete. Early
stopping saves the best model state separately while retaining the last model
and optimizer for exact continuation. `fit_epochs` does not silently swap the
last model to best weights. Evaluation must explicitly use the selected
`best_model_state` or its exported checkpoint; never assume last equals best.

## Checkpoint qualification

Checkpoint publication writes a new staged file, synchronizes it, renames it and
synchronizes the containing directory. Loading verifies its exact hash and
uses `torch.load(weights_only=True)` from CPU storage. State includes model,
optimizer, sampler, fitted transform, early stopping, best model, epoch records,
Torch CPU/MPS RNG, data/spec identities, implementation hashes, dependency
versions and runtime thread/dtype settings. A changed source, implementation or
execution setting is rejected for exact continuation.

`scripts/qualify_training_recovery.py` defaults to plan-only. Its current explicitly
enabled full/Q-only qualification is capped at twelve synthetic optimizer updates,300seconds
and4GiB observed worker RSS plus reported MPS allocation, with a2GiB MPS allocator
cap. Failed allowances remain charged. A killed/interrupted worker is reaped and
its last entered/confirmed counters are recorded. These are sampled/cooperative
resource observations, not proof that an instantaneous peak was impossible.

The original v1 receipt `reports/trainer-recovery.json` passed CPU LSTM and MPS
Mamba2 recovery in separate processes: two reference updates and one replayed
next update per model, **six synthetic updates total**. The next batch, loss,
model/optimizer, scaler, sampler, RNG and stopping state match exactly. A
completed stopping checkpoint is also restored without another optimizer step.
No real observations, protected values, network calls or scientific scores were
used. Ordinary `tests/test_training.py` performs zero optimizer updates.

The benchmark in `benchmark_real_batches.py` is separate: it uses fixed
engineering unit mappings to measure execution cost. Its timings do not select
the affine transform or establish full-study convergence.

The v2 full/Q-only extension is documented in `docs/q-only-training.md`; its new
12-update receipt is `reports/trainer-recovery-v2.json`. The old six-update record
is retained without being relabeled as a v2 execution.
