# Development corpus admission audit

`scripts/audit_corpus_admission.py` reports candidate coverage and input-source
diagnostics from an immutable `corpus_dataset` snapshot. It does not adopt a
scientific cohort, apply a new exclusion rule, fit transforms/models, score
forecasts, reopen raw source files, or contact a network service. Root runs the
full audit after all477 planned development shards have committed; preparation
and the small partial-fixture check do not establish full-corpus readiness.

```sh
PYTHONPATH=src .venv-offline-check/bin/python scripts/audit_corpus_admission.py \
  --corpus .local/development-corpus-L168-v4 \
  --expected-plan-sha256 EXACT_APPROVED_PLAN_SHA256 \
  --report reports/development-corpus-admission-v1.json
```

Use the actual immutable plan hash. Existing reports cannot be overwritten.
The output directory must exist, and the output must be outside the source
corpus. A temporary file is fsynced, then linked into its final name without
replacement. The CLI uses a900-second deadline. Process RSS is checked before
and after the bounded snapshot scan and between batch reads, including the
process high-water mark; this is cooperative enforcement, not an operating
system memory reservation. Maximum allowed RSS is4GiB, maximum planned shards
is1024, and each shard retains the dataset reader's16MiB expansion bound. Only
one source shard is decoded at a time; gauges are streamed separately, so the
source snapshot scan and each per-gauge iteration reread derived shards. No
full-corpus dense array is built.

The default refuses an incomplete corpus. `--diagnostic-partial` permits a
report labeled `PARTIAL_DIAGNOSTIC_NOT_FINAL_COHORT`, preserving its missing
chunks and observed immutable commitments. It cannot count unstaged cases as
upstream gaps. Later appended shards require a fresh snapshot/report; they do
not expand an existing selection. Corrupt committed shards fail verification
rather than becoming missing cases. An L168 source store is required for the
L24/L168 comparison.

## Report contents

Each gauge and scientific split has explicit candidate-selection digests for:

- All published candidate cases, without dropping missing inputs or labels.
- Natural L24-complete cases and natural L168-complete cases.
- The same L168-complete cases viewed as L24 and L168; their membership digests
  must agree. These are comparison candidates, not a chosen primary cohort.

Natural L24 eligibility is computed from the last24 exact hourly history slots;
an older missing observation does not automatically exclude the shorter view.
Original composite eligibility reasons, recomputed view reasons, separate
possibly overlapping exclusion dimensions, calendar status, and per-feature
mask counts remain visible. Target information is limited to mask/support
counts, including each H1–18 lead. There are no discharge magnitude summaries,
quantiles, scaler statistics, event thresholds, predictions, or scores. The
existing dataset validator still decodes derived discharge arrays to verify
finite/nonnegative values and mask semantics; this is not a zero-value-decode
claim. Repeated history/target cells are counted as tensor occurrences, not
unique observations or independent samples.

Weather diagnostics compare each selected source initialization with the
latest *nominal* six-hour cycle at or before `C−2h`, computed as
`floor((C−2h)/6h)×6h`. They count exact-source matches, older selections, source
age, hours older than nominal, native first/last leads, and no-source cases.
Older selection is evidence of fallback relative to the nominal calendar;
it does not identify why a newer source was unavailable or re-audit global
upstream availability. Soil availability and age are counted independently,
including cases with a different weather source cycle. Outside-calendar warmup
is reported separately using the source shard's explicit calendar status.

Physical-domain checks count **mask-admitted input cells** violating
precipitation/SWE≥0, temperature>0K, specific humidity within0..1, and soil
fraction within0..1. Counts include affected cases and the subset already in
the common L168 candidate set. Violations flag `physical_domain_review_needed`;
they do not change source masks or candidate membership. Finite-value and soil
fraction/age checks already enforced by `inspect_corpus` remain hard failures;
the audit does not relax them to produce a successful report. These broad
checks are diagnostics, not a complete hydrometeorological quality assessment.

The report binds the source plan file/digest, exact observed shard commitments,
pinned source index/USGS manifest/basin-weight identities, explicit input/target
quality policies, original producer code hashes, and current audit/reader code
hashes. A content digest excludes runtime measurements. Source cache bodies,
original raw arrays, original qualifiers and upstream acquisition receipts are
not re-audited or reconstructed. No private source paths are included. A
complete audit remains a prerequisite for the separate scientific cohort and
protocol decision, not permission to fit or access protected assessment data.
