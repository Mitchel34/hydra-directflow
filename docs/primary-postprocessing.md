# Development postprocessing after the 180-fit campaign

This stage audits and analyzes the frozen campaign's saved predictions. It does
not fit models, select another checkpoint, change targets or access protected
assessment. The original protocol and training artifacts remain unchanged.

## Inputs and independent metric audit

The completed ledger supplies the exact 180 prediction/metric receipts. Before
prediction values are decoded, the new reader authenticates artifact bytes,
metadata, units, gauge/case identities and development-only H18 coordinates.
One canonical label bundle per fold/gauge is loaded through the existing guarded
loader. Its typed-array identity must agree across the full, Q-only and L24 arms.

All 180 panels are independently recomputed from raw physical predictions and
original targets: MSE, RMSE, MAE, bias, NSE, negative-prediction counts, and exact
hour/date support. Numeric comparisons use fixed relative/absolute tolerance
1e-10; count and membership checks are exact. These calculations do not reuse
the production metric functions. Missing/nonfinite requested predictions fail.

The resulting twelve private groups preserve complete hourly origin calendars
and their original admission masks. Missing calendar rows contain NaN values
and false masks; they are not imputed observations. All later comparisons retain
this common L168 support, including the matched L24 contrast.

## Event and reference analysis

The existing `development-coverage-regimes-v2.json` is the fixed catalog and
threshold authority. Its SHA256 is
`8e82f613165a5d452c74a30783b02e1c0ae5fb5d93dbe2e338d0010c215e8c0a`.
Gradient-only q10/q90/q95 thresholds are reused; they are not recomputed from
admitted training cases or evaluation targets. Every ranked catalog event remains
visible, including events without forecasts. The first three supply representative
hydrographs without using model errors to select or replace events.

The user approved materializing 33 existing cloud-only cache files, totaling
105,064 bytes, and verifying/copying 147 files (467,088 bytes) for July 2020 through
July 2024. All files passed their recorded checksums. These owned references
supply observed hydrographs through forecast gaps. Original qualifiers and source
identities are retained; the existing development-only July 2024 export gate is
required. No new USGS request or protected-data read is made.

Event and regime scores remain descriptive and conditional on available forecasts.
Counts distinguish origin/lead cells, unique target hours, catalog events and
observed peak support. The catalog's 168-hour separation is not proof of independent
floods. Missing references are not assigned inferred flow regimes. Rising-limb
and false-alarm definitions are not added to this frozen primary analysis.

The fixed display uses H1 and H18 for each first-three event at every gauge. All
four full-L168 architectures are shown. Seed-mean forecast curves and seed ranges
are display summaries, not ensemble-performance metrics or prediction intervals.
All reported performance metrics continue to average individual-seed scores.

## Paired conditional uncertainty

Use the frozen 2,000 noncircular seasonal moving-block draws at 14 days, plus the
mandatory 7- and 28-day sensitivities. Draws preserve full calendars, seasonal
segment lengths and all 18 horizons. The same draw bundle is reused across gauges,
models and final seeds. Weights are integer multiplicities on original coordinates;
duplicate draws cannot inflate distinct-hour/date support.

Before observing intervals, the implementation fixes the stream mapping:
`SeedSequence(20261102).spawn(12)`, traversed in block order 7/14/28 days and fold
order CV1/CV2/CV3/final development. Each child yields one uint64 seed for the
existing PCG64 block sampler. This gives separate fold streams while preserving
required pairing within each fold. Stream keys and exact draw hashes are saved.

Baseline scores are checked against the independent audit for every model.
For each fold and block length, the first draw and first Jefferson panel also
pass the existing scalar multiplicity-weighted reference calculation before any
family interval is released. These are fixed qualification checks, not selected
favorable replicates.

The final family contains 42 primary contrasts; CV contains 30. Each family uses
one joint-valid pool, retains all seeds/folds, and never redraws invalid replicates.
More than 100 invalid draws makes the family's intervals unavailable. Percentiles
are 2.5 and 97.5, with linear interpolation. Per-lead bands use that same family's
valid pool. All three block-length results remain visible.

Intervals are conditional within-year, pointwise exploratory estimates—not
simultaneous family-wise coverage, between-year uncertainty, random-catchment
inference or calibrated forecast prediction intervals. Fixed-lag residual
correlations are descriptive diagnostics on the original calendar; they do not
select a favorable block length or establish independence.

## Local bounds and evidence

The stage precharges **1 GiB** in the existing campaign output ledger. It retains
the original 8 GiB campaign cap and deadline, adds a persistent six-hour stage
limit, and enforces 3 GiB combined process/child memory, 8 GiB cumulative logical
read reservations and the original 8 GiB free-disk floor. Original input files
and training evidence are retained. Completed group commits support verified
reuse without silently replacing conflicting output.

Figures use an already-installed local Matplotlib runtime; no dependency download
is needed. New code, tests, execution plans, source identities and outputs receive
separate checksums. Protected assessment, additional training and paid cloud jobs
remain outside this execution.
