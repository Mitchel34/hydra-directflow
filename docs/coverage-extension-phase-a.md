# Phase A: input-only coverage extension

The user explicitly adopted private plan v1 (SHA256
`2a68d386dabe553bdda2e77f07c99c89e4081b8dc899296162fd203d0c81fd4a`)
as an active continuous goal. The [public protocol](../configs/coverage-extension-protocol-v1.json)
records the adopted scientific scope and finite limits. This is a separate
inference-only development sensitivity; it does not replace the frozen primary
180-fit experiment or open protected assessment.

## Products and temporal population

Use the H18-safe origin calendar `[2023-07-31T00:00Z, 2024-07-30T06:00Z)`:
8,766 origins per gauge, 26,298 gauge/origin rows. Only the 72 existing
final-development full-input L168/L24 checkpoints are eligible. All four
architectures and seeds 17/29/43 remain present. There are no fold-specific
L24 CV checkpoints or Q-only L24 models; this extension does not fabricate them.

- `PRIMARY_FROZEN`: original complete-L168-input/complete-H18-target predictions.
- `L168_INPUT_ONLY_V1`: every qualifying L168 input case, independent of target quality.
- `L24_INPUT_ONLY_V1`: every qualifying clean L24 input case, independent of targets.
- `ROUTED_CLEAN_168_24_V1`: L168 when complete, otherwise clean L24, otherwise unavailable.

The router uses one checkpoint for the entire H1–H18 vector and is shared across
architectures and seeds. Source/weather/soil failures reject both lookbacks.
Complete L168 must imply complete L24; violations require investigation.
Numerical model failures do not create new routes or disappear from denominators.

## Source and information boundaries

The new reader starts from the original resident candidate shards, not the old
common-case postprocessing arrays. It authenticates archived bytes and decodes
only input tensors, input masks and clock/source fields. It does not decode
`target`, `target_mask`, mixed `admission_mask` or target-derived eligibility
members. Mixed summary counts are not used for routing. Archive hashes still
cover stored target bytes; hashing those bytes is distinct from decoding outcomes.

Original hourly qualifiers/source IDs are joined only to exact causal-history
positions. Monthly `values.npy` is not opened in this path. Native provenance
columns outside the causal-history union are not decoded. A history observation
for a later origin may be an earlier origin's valid time; every individual origin
still uses only C−169…C−2 or C−25…C−2. Original qualifiers, numeric/presence masks,
last reading ages, gap positions and weather/soil sources remain reconstructible.
The age input remains C minus the observation timestamp; no missingness feature
or repair is introduced.

The original A/A+R rule, simulated two-hour availability, one covering HRRR cycle,
older-cycle fallback and separate soil state with maximum age eight hours remain
unchanged. Missing weather and targets are never filled. The historical revision
vintage limitation remains explicit.

## Release sequence

1. Authenticate the original ledger, all 72 checkpoint blobs and their exact
   gauge/history/seed/feature identities. Recover gradient case membership from
   existing authenticated metadata; do not refit statistics.
2. Commit the complete input-only eligibility and route maps.
3. Reproduce all 72 original clean prediction sets with original batch order,
   B64 partition, backend, saved transforms and best weights. Require byte-exact
   equality and a global clean-regression barrier before expanded forwards.
4. Generate the two input-only products. Verified clean outputs may be reused
   only after exact typed input/coordinate/checkpoint equality; count actual
   forwards separately. Assemble routed vectors without a third forward pass.
5. Authenticate and commit every requested vector and the prediction barrier.
6. Only then load original development targets/qualifiers and score each lead
   independently. Missing H3 truth does not cancel valid H18 verification.

New inference-source/cohort identities remain distinct from training identities.
A temporary reconstruction of the old input commitment is used only to verify
clean parity; the old cohort scalar is never substituted into new inference.
All saved-transform digests are recorded at clean load and bound into the global
regression barrier before expanded inference.

## Evaluation and limits

Report original common cases, additional L168 cases, L24 fallback and remaining
unavailable origins separately. Keep per-lead valid-target/evaluable counts,
raw physical errors, negative frequencies, undefined metrics and all seeds.
Equal-lead mean NSE requires all 18 leads to meet the existing 168-distinct-hour,
28-UTC-date and positive-variance floors. Mean individual-seed metrics rather
than scoring an averaged-prediction ensemble.

Natural L24 and routed forecasts have the same input-eligible population, so
those products can be compared on shared support. Scores from recovered versus
old common populations are not paired skill improvements. The missing Q-only
L24 comparator prevents a routed weather/soil attribution claim. Reuse the fixed
gradient thresholds and observed-event catalog; do not reselect events from errors
or invent new peak-timing/false-alarm acceptance definitions. No additional
uncertainty family is run by default.

The phase is limited to one worker, 1,280,000 case–checkpoint evaluations,
20,000 batches of at most 64, 72 distinct checkpoints, and at most 145 model
constructions (two possible loads per checkpoint plus one infrastructure replay).
Use original CPU/MPS assignments, deterministic FP32 and a 2 GiB MPS cap.
A separate durable ledger retains a six-hour real-execution deadline, 4 GiB
combined memory, 8 GiB logical artifact-read reservations and an 8 GiB free-space
floor. A single extra GiB is precharged inside the original aggregate 8 GiB output
allowance. Failures and replay consume allowance; no refunds or clock resets.
One diagnosed infrastructure resume is permitted. No automatic retry hides a
numerical, integrity or implementation failure.

There are zero new training fits, optimizer updates or scaler fits. Paid jobs,
new downloads, access/security changes, real short-gap repairs and protected
assessment remain excluded. Ninety-percent coverage is a research aspiration,
not a pass condition or permission to add methods until results look favorable.

After trustworthy Phase A results, prepare a Phase B decision packet if repair
research appears useful. Its policies, route order, causal endpoints, artificial-gap
controls, numerical tolerances and measured resources require separate adoption
before real repair experiments. Completion of Phase A does not complete the AGU study.
