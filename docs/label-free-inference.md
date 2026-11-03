# Label-free best-checkpoint inference

`inference.py` provides a bounded development-only prediction path without
constructing a trainer or optimizer, fitting statistics, or requesting target
values or target masks. It accepts at most128 cases per call. Scientific
execution, source/cohort admission and protected assessment authorization remain
external requirements; protected or crossing forecast coordinates are rejected
regardless of hashes.

```python
from hydra_directflow.inference import BestCheckpointPredictor, input_only_view

predictor = BestCheckpointPredictor.load(
    checkpoint_path,
    expected_sha256=checkpoint_sha256,
    expected_source_identity=training_bundle.source_identity,
    expected_feature_schema=feature_schema,
    expected_training_cohort_admission_digest=training_parent_cohort_sha256,
    expected_gauge_id=gauge_id,
    gradient_fit_case_ids=tuple(training_bundle.arrays["case_id"][
        list(training_bundle.indices_for("fit"))].tolist()),
)
inputs = input_only_view(input_batch, inference_source_identity=current_input_source_sha256)
predicted = predictor.predict(
    inputs,
    expected_inference_source_identity=current_input_source_sha256,
    expected_inference_cohort_admission_digest=current_input_parent_cohort_sha256,
)
```

`input_batch` must declare its feature schema and parent cohort. It may omit
labels entirely; if labels happen to be present, the adapter never fetches them.
Known `target_time` coordinates become `forecast_time` in the input-only view.
The coordinates are validated before reading numerical inputs, then input
geometry is checked before copying. History ends at C−2 inclusive; forecast
coordinates are exactly C+1…C+18. Case IDs must agree with the single gauge and
chronological UTC origins. False input masks produce zero placeholders only
when applying the saved transform, preserving their missingness indicators.
Q-only uses the declared H1–18 tokens, saved `h/18` encoding and an empty state.

The loader verifies bounded resident checkpoint bytes and an explicit SHA256,
v2 schema, training source/cohort identities, feature schema, model code and
dependencies. Exact gradient-fit case IDs must match the saved scaler's case-ID
hash and declared gauge. This adds gauge binding without loading training
labels, since the existing v2 checkpoint does not store its gauge separately.
The saved input and target transform statistics must be valid and match the
feature schema; they are never refitted. Synthetic tests change all inner,
outer and unassigned inputs/labels/masks while proving unchanged gradient-fit
scaler parameters. Changing a gradient-fit label changes those parameters.

Only `best_model` (the serialized `best_model_state`) is loaded. Missing best
state, inconsistent stopping history, malformed keys/shapes/dtypes, nonfinite
weights or changed positional buffers fail. The recorded stopping decisions
are replayed, including strict improvement and minimum-epoch semantics. There
is no fallback to the final model state. The loader does not restore an
optimizer or training RNG; model construction preserves the caller's CPU RNG.
Inference runs in evaluation/inference mode on the checkpoint's declared local
CPU/MPS device. It does not qualify CUDA or cross-device equivalence.

The returned `prediction_q` is float64 physical discharge in m3/s, obtained by
applying the saved target affine inverse to the model's FP32 predictions. It is
not clipped. Coordinate arrays and predictions are read-only. The receipt
identifies the checkpoint, selected best epoch, architecture/seed/schema,
gradient-fit transform provenance, actual prediction-input identity and case-ID
hash. It reports zero optimizer updates, no scaler fit and no label access.
It does not calculate forecast scores.

Training and inference parent identities are deliberately separate. Future
inputs may belong to a different, explicitly supplied inference source/cohort;
reusing a training cohort hash for different cases would misstate provenance.
For current development subsets, the same verified parent may be referenced
with a distinct actual input/case digest. Structural hash checks document the
provided identities; they do not independently grant scientific authorization
or re-audit raw source data.

## Qualification state and executed bounded plan

Executed unit qualification is synthetic CPU LSTM inference in full and Q-only
modes, with saved synthetic best states and deliberately different last states.
It includes missing/poisoned labels, separate future input identities,
protected-clock rejection, saved-transform parity and malformed-best-state
rejection. There are zero optimizer constructions or updates in these tests.
The controlled stopping records test selection mechanics, not learned forecast
performance. Existing trainer-bundle and temporal tests also pass.

A deeper trainer-to-inference qualification has now been executed by the root
under the following separate bounded engineering plan:

- Four synthetic cases: LSTM/CPU and Mamba2–Transformer/MPS, each in full and
  Q-only modes. Preserve the existing qualified local backend choices; no
  silent CPU fallback for MPS.
- Use a fixed-seed,12-case synthetic L24 fixture:8 gradient-fit cases and4
  inner-stopping cases, with valid development clocks and explicit full/Q
  parent identities. Fit statistics only on the8 gradient-fit cases.
- Run exactly2 complete epochs with one8-case gradient batch per epoch and
  inner stopping after each epoch:2 updates per case, maximum8 updates total.
  Use fresh bounded processes, a900-second campaign cap,4GiB memory cap and a
  durable consumed allowance. No scientific source arrays or protected clocks.
- Save a v2 checkpoint containing the actual recorded best state. In a fresh
  process, load it through `BestCheckpointPredictor` using gradient-fit IDs and
  explicit source/cohort hashes. Predict4 separate synthetic development
  origins from a mapping with no labels. Compare against a reference forward
  using the saved best state and saved transform. Verify input identity,
  physical inverse, selected-best epoch and zero new updates during inference.
- Retain every qualification receipt and entered/confirmed update count; do not
  retry against a reset allowance. Publish only engineering pass/fail, hashes,
  counters and resource observations. Do not publish synthetic losses as
  scientific results or use this qualification to select an architecture.

`reports/best-inference-qualification.json` records four passing fresh-process
pairs and exactly eight precharged/entered/confirmed synthetic updates in
17.315 seconds. Saved transforms, best states and physical predictions match
exactly; inference performed no label access, optimizer construction or scaler
fit. The full-input CPU LSTM selected epoch 1 while its checkpoint also contained
epoch 2's last state, providing an actual best-versus-last integration check.
These results qualify the stated synthetic L24/B8 paths, not every architecture,
history length, device or scientific fit. Earlier model/backend and recovery
receipts retain their original scope. The prior twelve-update qualification was
not repeated or rewritten. Cumulative synthetic engineering updates are now 904;
real-data engineering updates remain 852. Scientific comparison fits remain zero.
