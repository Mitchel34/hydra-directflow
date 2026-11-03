# Missingness pilot implementation and qualification

The user approved implementation and qualification only. The strict 180-fit
primary remains unchanged; the proposed six neural and three ridge fits on study
data remain held until the pilot protocol and clean-degradation/failure tolerances
are agreed. Protected assessment is separately gated. No new download, paid job
or access change was performed for this work.

## Implemented components

`gap_filling.py` reconstructs only eligible historical Q slots. Linear interior
interpolation uses approved original endpoints within the exact history window.
Trailing carry uses the last real reading. The fixed ridge alternative has six
consecutive real lags, two direct outputs, alpha1, gradient-only feature scaling
and an unpenalized intercept. All fit-pair clocks, masks, source IDs, original
qualifiers and refit bounds are checked before numerical Q reads. Ridge artifacts
bind source, gauge, fit window, parameters, policy and implementation identities.

Both maximum run length and total missing hours are capped at the chosen one/two
hours. Leading and longer gaps remain excluded from this repair. A failure leaves
all proposed fills in that window unapplied; there is no partial rescue, clipping,
recursive prediction, use of a filled endpoint, or silent carry fallback after a
ridge failure. No verification-target or weather filling interface exists.

`pilot_inputs.py` supplies a separate H4 schema: Q, coordinate age, missingness
indicator and elapsed time since a real reading. It keeps original-observation,
artificially-hidden, effective-observation and usable-numeric masks separate.
Artificially hidden magnitudes are removed before fillers, endpoints or lag
features are formed. They remain only in the caller's reference-side object.
Elapsed time never resets at a derived fill. Q/coordinate-age use the saved
original-gradient H2 statistics; new channels use fixed identity and hours/168
maps. No scaler fit or label read occurs in the input adapter.

Seventeen deterministic scenarios cover clean input; interior1/2h in early,
middle and recent bands; trailing1/2h with carry/ridge; and mixed interior/trailing
gaps with carry/ridge in each band. Masks depend on case ID, epoch and mask seed,
not Q magnitudes, targets, model errors, batch order or process-global RNG.
Carry/ridge variants share the same artificial masks. Band layout and scientific
mask-seed mapping are still proposed policy choices, not silently frozen by code.

`pilot_training.py` is a separate LSTM/CPU/L168 implementation using the existing
configurable architecture. It does not extend or monkeypatch the primary H2
trainer, model, scaler or inference path. H4 has 68,161 parameters, 256 more than
the corresponding full-input H2 LSTM. Clean and augmented arms share the original
gradient scaler and case membership; augmentation only replaces gradient inputs.
Inner stopping uses clean standardized-Q MSE. Equal opportunities do not force
identical observed stopping epochs.

Checkpoints bind the source arrays/provenance, gauge, roles, simulated refit,
fill/augmentation/ridge policies, original transform, sampler position, update
journal, optimizer state, owned CPU RNG and selected best state. Unsupported
batches fail before advancing the sampler. Fresh inference loads the saved best
model and saved transform without constructing an optimizer, fitting a scaler,
reading labels or clipping physical predictions. Cross-gauge, wrong-role and
protected clocks reject before numerical reads.

## Executed synthetic qualification

The default qualification command prints a plan. Its explicit execution path has
no study-array argument: it generates deterministic synthetic histories, weather,
soil, labels and provenance. A persistent once-only ledger precharges 16 optimizer
updates and one synthetic ridge fit for the bounded run. It permits no retry or
reset of that consumed allowance. Network is denied in workers; CPU threads are
fixed at one, with a 900-second campaign bound and 4 GiB parent/child RSS ceiling.

The executed run passed in **34.31 seconds**, with **401 MiB peak observed combined
RSS**. It used 64-case L168 batches, 128 synthetic gradient cases, 64 inner cases,
and 64 separate label-free inference cases. For each clean/augmented arm:

- A two-epoch uninterrupted run used four updates.
- A separate one-epoch prefix used two updates and saved its checkpoint.
- A fresh process restored that prefix and used two more updates.
- The uninterrupted and resumed sample order, masks, losses, optimizer/model
  state, selected best state and predictions matched exactly.
- Another fresh process loaded best weights with optimizer/scaler-fit/trainer
  construction poisoned. Physical predictions matched the independent reference.

Both actual fixtures selected epoch1 while their last weights were from epoch2,
so the best-versus-last check was nontrivial. Total entered and confirmed updates
were exactly 16. The bounded ridge stage fitted one synthetic artifact and all
subsequent phases loaded it without fitting. Ordinary NumPy unit-test fixtures
are separate from this qualification-stage fit count.

All 17 scenarios were exercised for each arm. Sixteen produced64-case predictions.
The mixed-recent/ridge scenario had56 eligible cases and8 cases with a missing
required original ridge lag; the entire requested batch was explicitly refused.
It was not silently reduced to56 or repaired with carry-forward. This is a
failure-handling qualification, not a successful64-case forecast or a scientific
paired-score result. A real paired experiment must preserve the declared support
and report failed attempts rather than omit them after seeing errors.

The integrated implementation suite ran 689 tests: 686 passed and 3 unchanged
GIS-only tests were skipped. The new focused checks comprise 13 filler tests,
14 adapter/scenario tests,13 zero-update trainer tests and7 pure qualification
controller tests. Original H2 and acquisition code remain unchanged.

## Limits before study-data execution

The two-hour delivery delay remains simulated. Source timestamps and approved
revised flags do not prove historical delivery or revision availability. Actual
pilot preparation must provide an authenticated per-slot provenance join from
the retained source cache; the H2 numeric bundle alone does not invent flags or
source identities. There is no scientific campaign CLI enabled by this package.

The tested mechanics do not choose acceptable clean-flow degradation, false-alarm
tradeoffs, failure tolerance, augmentation mixture, final scientific seed mapping,
or uncertainty policy. A synthetic mixture was used only for qualification.
Those choices and a concrete fit allowance remain required before the proposed
study-data pilot. Optional final confirmation and combined short-Q/L24 routing
remain separate proposals.

The observed H4 update times include input/provenance processing. They are larger
than the old H2 model-throughput proxy; the older runtime proposal is preserved,
and a separate preparation profile/reassessment records this material limitation.
No additional optimizer updates are needed for that forward-only profile.

The separate zero-update profile completed20 calls in each of five stages
(100 timed calls total) in 16.24 seconds, at about 287 MiB processRSS. Mean times
were86.24ms for clean preparation,76.90ms for augmented preparation,211.40ms
for full provenance normalization/digest,14.88ms for the neural forward and
335.92ms for the full clean prediction path. These are separately timed stages,
not a disjoint trace whose percentages can be summed. Warmup/order differences
do not establish that augmentation is faster. No backward call, fit or optimizer
update occurred; the qualification ledger stayed unchanged.

The old H2 rate timed pre-mapped/tensorized kernels after warmups. H4 qualification
updates include preparation, provenance hashes/copies and durable qualification
status writes; their first step also creates AdamW. The ratio is therefore not a
GPU/model speed comparison. The current neural forward is close to the older
H2 inference rate, while repeated metadata handling dominates the full path.

Using the mean of only three later total-update samples per arm and the profiled
prediction path as an approximate stopping/inference proxy gives:

| Optional scope | 20 epochs | 50 epochs |
|---|---:|---:|
| Six-neural-fit pilot | 4.22 h | 8.64 h |
| Eighteen-neural-fit confirmation | 25.28 h | 57.81 h |
| Both stages | 29.50 h | 66.45 h |

These are conditional estimates, not confidence bounds or a full-cohort
benchmark. Ridge fitting, full-scale source/provenance construction, scoring,
resampling, failures and some I/O remain unmeasured. All 17 requested stress
passes are budgeted even though an unsupported scenario cannot produce a valid
complete scored panel. The primary 180-fit estimate and $150 ceiling are unchanged.

The highest-value next engineering step is to validate and hash immutable
provenance once, reuse clean prepared inputs and retain compact source/endpoint
identities for corrupted inputs. Any optimization must preserve source, masks,
clock guards and exact restart semantics and be requalified before scientific
fits. Full-cohort constructor memory/time also needs a bounded check. No such
optimization, extra optimizer allowance or cloud release is implied by profiling.

## Evidence and safe verification

- [Executed qualification](../reports/gap-pilot-qualification-v1.json)
- [Zero-update preparation profile](../reports/gap-pilot-preparation-profile.json)
- [Runtime reassessment and arithmetic](../reports/gap-pilot-runtime-reassessment.json)
- [Coverage census and proposed decision](coverage-sensitivity-census.md)
- [Scientific sensitivity design, still pending freeze](coverage-sensitivity-design.md)

Run ordinary checks with `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`.
Print the current plan with `.venv/bin/python scripts/qualify_gap_pilot.py`.
Do not rerun its consumed execution allowance as a routine test. Previous
qualification reports, checkpoints, source identities and counters are preserved.
