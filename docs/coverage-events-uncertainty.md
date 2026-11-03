# Coverage, events and uncertainty — pre-freeze specification

This specification is recorded before scientific comparison fits or protected
assessment access. Coverage diagnostics use development observations only.
The event and resampling choices below remain reviewable scientific choices;
they do not open assessment or create model-performance findings.

## Coverage denominators and reference regimes

`audit_coverage_regimes.py` binds the complete v4 snapshot and the monthly
discharge cache keys referenced by its immutable chunks. It recomputes each
monthly key from its derivation identity before opening values. July 2024 must
have the verified development-export cutoff. No raw files, network service,
cache writer or model are used.

Origin-season coverage uses UTC DJF/MAM/JJA/SON and gauge/origin cases. Input
admission, complete H18 labels, natural L24, common L168 and their exclusions
are separate. Exclusion dimensions can overlap; they are not summed as mutually
exclusive causes. The 36 target-boundary-purged origins remain separate from
data exclusions.

Flow regimes use target valid times, so their denominators are gauge/origin/lead
cells. Repeated cells are not unique hours or independent samples. For each
fold/gauge, q10, q90 and q95 are calculated once from **unique approved exact-hour
USGS references inside that fold's gradient-training calendar**, subject to the
declared simulated refit cutoff. No stopping or outer values determine thresholds.
These diagnostic thresholds are not model scalers and do not redefine training
membership. They deliberately include approved gradient-calendar references
whose forecast cases lack weather/history inputs, avoiding an additional
input-availability filter on the reference distribution.

Regimes are LOW (Q≤q10), ORDINARY (q10<Q≤q90), ELEVATED (q90<Q≤q95), and HIGH
(Q>q95), with exact ties retained. Rejected/missing references are UNKNOWN;
they are not assigned a flow class using estimated values or interpolation.
These are relative discharge regimes, not official flood stages or return periods.
The audit reports independent input/reference support and the stricter common
complete-vector comparison support.

Contiguous unusable-reference intervals retain their missing-record,
invalid-numeric, estimated, provisional or unknown-qualifier reasons. They are
not proof that a station was disabled or damaged. No peak magnitude or regime
is inferred inside an unusable interval. This limits claims about rare high-flow
outages and complete-case survivorship.

## Event rule fixed before model-error inspection

For each gauge and outer-development year, rank approved q95-exceeding reference
hours by discharge, breaking exact ties by earliest UTC time. Greedily retain
peaks at least 168 hours from previously retained peaks. This creates a diagnostic
catalog; it is not a claim of physically independent floods. Select the first
three for representative hydrographs without consulting any model error or
requiring model-input admission. Keep the remaining catalog and coverage reasons.

Use peak−48h to peak+72h windows, clipped to the declared outer interval.
Report clipping and reference gaps. The 168h separation exceeds the 120h window,
so retained windows do not overlap. Event coverage is reported by forecast lead;
an observed peak with no common forecast case remains in the catalog. Missing
peaks cannot be recovered. Later event error summaries must retain these events
and report unavailable metrics rather than substitute better-performing events.
Short-window RMSE/MAE/bias and counts are descriptive; they do not inherit the
annual-panel NSE support rule or establish event independence.

## Proposed dependence-aware uncertainty

Use paired resampling of **whole hourly-origin vectors**: retain all 18 horizons,
all model outputs, all masks and all three fixed gauges for each sampled origin.
Never compact away missing calendar rows, resample individual lead cells, or
count overlapping targets as independent observations. Apply the same draws to
all compared models and fixed seeds. Average seed metrics only after computing
each seed; report all seed results and their dispersion separately.

Propose 2,000 noncircular moving-block draws, stratified by contiguous UTC
meteorological-season segments within each outer year. Primary block length is
14 days, with mandatory 7- and 28-day sensitivities and a fixed random seed
20261102. Blocks cannot cross season-segment or fold boundaries. Sample blocks
with replacement to each segment's original length and truncate only the final
drawn block. This preserves seasonal sample lengths; noncircular sampling gives
edge hours unequal inclusion probabilities, which must be disclosed. It is a
conditional approximation, not proof that 14 days removes hydrologic dependence.

The primary 14-day choice exceeds the 187-hour endpoint span from the oldest
L168 observation at C−169h to the last target at C+18h. That is a design rationale,
not a theorem about residual independence. Inspect
development residual dependence and event durations before assessment. Report all
block sensitivities; do not choose the one producing favorable intervals. If
dependence beyond the tested lengths materially invalidates the approximation,
bring the issue back before freeze/assessment rather than silently relabeling it.

Recompute metrics through integer multiplicities on the **original unique
coordinates**, not duplicated sample IDs fed to the ordinary scorer. The
predeclared comparison mask is hash-bound separately from labels and prediction
availability. Requested predictions must exist even when their labels are
masked. Outside-comparison calendar rows remain present but contribute zero
score weight. The baseline and every draw must satisfy the primary support
checks using distinct positive-weight original target hours/dates; multiplying
one observation does not create independent support. Weighted reference means,
variances and error sums are recomputed for each draw.

Use 95% percentile intervals, explicitly the 2.5th and 97.5th percentiles. The
final-development **joint validity family** contains 42 gauge-level primary
contrasts: three gauges times four full/Q contrasts, four full-input L168/L24
contrasts, and six pairwise full-L168 architecture contrasts. CV has 30 such
contrasts because no L24 CV arm exists. Each CV contrast uses the equal mean of
its three required folds. A draw is jointly invalid if any required primary
panel or contrast in that family is unavailable. Record its cause; do not redraw.
If more than 100 of 2,000 draws are jointly invalid, the family's primary
intervals are unavailable. Otherwise all primary intervals use the same jointly
valid draw pool and report its size and conditional interpretation. The 5% rule
is a declared reporting guard, not a calibrated coverage theorem.

Joint validity does **not** imply simultaneous 95% confidence coverage. Per-lead
bands and these contrasts remain pointwise/exploratory; they are not 18 separate
independent discoveries or a family-wise confirmatory test. NWM uses a separate,
genuinely matched supported-lead analysis and cannot enter the independent
HYDRA H1–18 primary aggregate by dropping unsupported horizons.

The proposed `uncertainty.py` utilities and synthetic tests implement calendar/
season index draws and multiplicity-weighted metrics. They do not yet implement
the family-level percentile orchestration above, and no study bootstrap has run.
For development, the complete origin calendar ends at the last H18-eligible
origin (exclusive 2024-07-30T06:00Z for final development); it does not pad in
origins whose targets cross assessment. All internal unavailable rows remain.
The 28-day sensitivity has limited block diversity in short season segments;
all 7/14/28-day results and available start counts must remain visible.

**Material limitation for review:** one protected year cannot support resampling
independent hydrologic years or establish between-year uncertainty. CV fold
variation is reported directly; folds have different, overlapping training
histories and are not independent copies of one fitted model. The three gauges
are fixed sites, not a random sample of catchments. These intervals are neither
calibrated per-forecast prediction intervals nor proof of operational reliability.
Any confirmatory multiplicity claim or practical-effect threshold needs an
explicit decision before freeze; otherwise effects and conditional uncertainty
remain qualified comparisons.

Clark et al. demonstrate the sensitivity of hydrologic efficiency metrics to a
small number of influential errors and use annual blocks to retain seasonal
structure. That supports reporting uncertainty and cautions against treating
individual hours as independent; their annual-block design is not estimable from
one protected year here. [Clark et al. (2021)](https://doi.org/10.1029/2020WR029001).
Hydrologic block-bootstrap work also makes its dependence and periodicity
assumptions explicit; it does not prescribe the block length for this study.
[Ebtehaj et al. (2010)](https://doi.org/10.1029/2009WR007981).

## Interpretation controls

The selected matrix estimates the joint contribution of weather **plus soil**.
It does not isolate either source, standalone GRU/Mamba-2 components, or attention
as a causal mechanism. NWM stays an external comparison on genuinely matched
available forecasts; it never determines HYDRA admission. Persistence remains
excluded. Primary predictions remain raw and unclipped. Nonnegative admitted
observations alone do not establish that all physical flows or rejected raw
observations are nonnegative; a zero-clipping sensitivity would need its own
predeclared rule and label and would not replace the primary scores.
