# Provenance optimization: v2 qualification and performance

The optimized H4 implementation passed a separate, bounded synthetic
requalification. It preserves the v1 numerical results while reducing repeated
metadata work. **The six neural pilot fits and three ancillary ridge fits remain
held.** The primary 180-fit scope and $150 combined ceiling are unchanged; these
receipts do not authorize scientific fits or paid cloud jobs.

## What was verified

The [v2 qualification](../reports/gap-pilot-qualification-v2.json) used exactly
16 new synthetic optimizer updates: clean and augmented arms each ran a
four-update baseline, a two-update prefix, and a two-update fresh-process
continuation. It finished in 21.84 seconds. The consumed v1 allowance and all v1
artifacts were preserved. The previously qualified ridge artifact was reused,
with zero new ridge fits.

Both arms matched v1 exactly on update traces, masks, losses, model and optimizer
state, sampler order, saved H2 transforms, CPU RNG, selected best state and
physical predictions. Fresh-process restart and label-free best inference also
passed. The new checkpoints have different code fingerprints, as expected; old
checkpoints were compared as authenticated reference data, not restored under
changed code. This establishes engineering equivalence on the fixed synthetic
fixture, not forecast skill or convergence on scientific data.

The optimization removes repeated JSON roundtrips and reuses validated,
immutable original provenance. It preserves source IDs, ordered qualifiers,
original observation masks, clocks, and case mappings. It does not cache Q
values, targets, prepared fills or model outputs. Dynamic input values, masks,
gap endpoints and fill admissibility still receive their existing checks.

## Measured preparation and inference

The [v2 profile](../reports/gap-pilot-preparation-profile-v2.json) made 120 timed
calls: 20 calls in each of six stages, with zero optimizer updates, scaler fits
or ridge fits. It finished in 11.32 seconds with approximately 323 MiB observed
process RSS. CPU, one Torch thread, FP32, B64 and L168 were fixed.

| Path | v1 mean | v2 mean | Interpretation |
|---|---:|---:|---|
| Clean adapter | 86.24 ms | 50.18 ms | v2 uses the validated snapshot view used by training |
| Augmented adapter | 76.90 ms | 53.03 ms | Same snapshot path plus deterministic artificial gaps |
| Full provenance digest | 211.40 ms | 62.72 ms | Actual optimized digest helper; identical logical bytes |
| Model-only forward | 14.88 ms | 14.82 ms | Prepared tensors; no preparation or inverse transform |
| Repeated-batch full prediction | 335.92 ms | 144.09 ms | v2 includes one initial cache miss and 19 hits |
| New-origin full prediction | Not measured | 175.06 ms | Twenty distinct batches, all actual cache misses |

The model-only rate is essentially unchanged. The observed improvement is in
work surrounding that model call. These are separately timed paths, not an
additive trace that precisely assigns every millisecond. Workspace/runtime
residency and cache conditions also changed, so the table is not a controlled
ablation attributing every gain to one code edit.

The original training provenance snapshot was built once outside the adapter
timers; its measured setup was 31.68 ms for 64 cases. New-origin fixtures were
also generated outside their timers, and their output-identity checks ran after
timing. Each new batch shifted its 64 origins by another hour within the
declared outer interval. Cache diagnostics confirmed 20 additional misses and
zero hits, with at most one retained entry. The batches overlap in calendar
content; this measures metadata-cache misses, not cold disk reads or actual data
extraction.

The 19 warm-hit calls alone averaged 142.18 ms. A repeated-batch result must not
stand in for ordinary new-origin inference. Runtime planning below therefore
defaults to 175.06 ms. A batch-major scenario is shown separately and requires
one cold request followed by 16 warm requests for each outer batch; it does not
assume all 17 requests are warm or that this schedule has been adopted.

## Full training-source constructor check

The [constructor qualification](../reports/gap-pilot-constructor-qualification-v2.json)
passed at **17,869 synthetic source rows**: 15,717 gradient-fit and 2,152 inner
rows, matching the largest proposed CV1 gauge counts. It retained **zero outer
or unassigned rows**. Synthetic placement and values were used; actual scientific
case membership and source bodies were not read.

Recorded work was:

- Synthetic fixture generation: 6.48 seconds.
- Trainer construction, including one synthetic original-H2 scaler fit:
  9.01 seconds.
- One clean and one augmented 64-case batch: 114.46 and 112.65 ms.
- Total after imports: 15.73 seconds.
- Peak observed parent-plus-child RSS: **1.832 GiB**, under the 4 GiB limit;
  wall allowance: 120 seconds.
- Zero optimizer construction, neural updates, model forward calls or ridge fits.

The original NumPy payload was 97,779,508 bytes and provenance covered 3,001,992
history slots. Payload bytes, retained metadata size and total process RSS are
different quantities. This check does not isolate their individual peak costs.

The future pilot assembler must keep this fit/inner footprint and load outer
cases separately. The optional 26,425-row outer-inclusive constructor was not
run. Nor does this check qualify the optional final-development constructor,
whose largest fit/inner source would contain 42,696 cases, or full-cohort
backpropagation memory. Those larger workloads require their own qualification
or a separately reviewed streaming design.

## Updated conditional runtime

The [v2 reassessment](../reports/gap-pilot-runtime-reassessment-v2.json) retains
the per-gauge admitted counts and B64 batch ceilings. It separates each arm's
first update from its next three samples:

| Arm | First update | Mean of next three | v1 mean of next three |
|---|---:|---:|---:|
| Clean H4 | 654.89 ms | 151.16 ms | 356.46 ms |
| Augmented H4 | 545.17 ms | 155.27 ms | 324.11 ms |

These update timings include preparation, hashes, copies, model work and the
qualification worker's durable status writes. The first update also constructs
the optimizer lazily. Four samples per arm do not establish stable throughput,
convergence or a confidence interval. The JSON includes later-sample
minimum/mean/maximum scenarios without interpreting those extremes as bounds.

The following table uses the later-three means. All figures are **modeled hours,
not measured scientific-run durations**.

| Proposed work | 20 epochs, cold default | 50 epochs, cold default | 20 epochs, conditional batch-major | 50 epochs, conditional batch-major |
|---|---:|---:|---:|---:|
| Six-fit CV1 pilot | 2.03 | 4.06 | 1.91 | 3.94 |
| Optional 18-fit confirmation, excluding unknown constructor cost | 11.71 | 26.47 | 11.38 | 26.14 |
| Combined modeled portions | 13.74 | 30.53 | 13.30 | 30.09 |

The pilot estimate includes the largest-CV1 measured synthetic constructor time
once per fit as a setup proxy for all three gauges. It does not substitute the
synthetic fixture-generation time for actual source extraction. Final-stage
constructor cost is explicitly unknown and omitted, not assigned a zero cost.

For each fit, training time is the observed first update plus the remaining
update count multiplied by the selected later rate. Clean-inner time uses the
new-origin full-predictor rate as an **unmeasured surrogate** in both schedules;
it is neither a measurement of the inner-loss loop nor a proven upper bound.
Outer time is either 17 cold calls per batch, or one cold plus 16 warm calls.
The older H2 per-epoch checkpoint-write placeholder remains separately labeled;
the cost of growing H4 journals is not yet measured.

The 17 requests are evaluation opportunities, not guaranteed successful scored
panels. Mixed-recent ridge still correctly rejected its 64-case qualification
batch because only 56 cases had admissible original lag context. Failures may
not be silently dropped to improve coverage or metrics.

Scientific ridge fitting, actual data assembly, scoring/resampling, larger
constructor costs, long-run thermal effects and failed attempts remain outside
these measured costs. The six neural pilot fits plus three ridge fits, and any
optional expansion, still require their separate protocol and execution review.

The [v1 reassessment](../reports/gap-pilot-runtime-reassessment.json) and
[original proposal](../reports/coverage-sensitivity-resource-proposal.json)
remain unchanged. The primary 180-fit estimates are not revised by this optional
H4 timing work. Synthetic checks also do not certify that every development
corpus file is resident; data restoration has separate evidence and completion
criteria.
