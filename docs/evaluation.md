# Guarded development evaluation

`hydra_directflow.evaluation` contains pure standard-library functions. It does
not read source records, fit models, select a primary metric, bootstrap results,
or grant access to protected assessment outcomes. Its tests use invented values.

## Identities, labels and masks

Each `TargetKey` identifies one target cell with an explicit sample ID, confirmed
gauge, nominal origin, target valid time and information cutoff `C`. The origin
and cutoff must be the same exact UTC hour. The target must be `C+1h` through
`C+18h`. Duplicate IDs, duplicate coordinates, unknown gauges and mismatched
key order fail rather than being silently joined or reordered.

`LabelSeries` carries label values and its own boolean quality/missingness mask,
plus SHA256 identities for the source and label-quality policy. Each metric
result also binds the admitted labels, exact keys and mask with a digest. These
identities make the declared inputs inspectable; they do not authenticate an
external artifact by themselves.

`PredictionSeries` separates its numeric-prediction mask from original source
support. A forecast can exist in an archive while remaining unsupported at `C`.
The scoring mask intersects label admission, prediction presence, source support
and an explicitly supplied comparison cohort. Masked values may be `None` or
nonfinite and stay excluded. An admitted nonfinite value fails validation rather
than disappearing from a score. Finite negative predictions remain in the score
and receive a count; this module does not clip them.

## Metric definitions

For the same `n` admitted samples, let `e = prediction - observation`:

| Metric | Calculation | Undefined result |
|---|---|---|
| RMSE | `sqrt(sum(e**2) / n)` | `None`, `ZERO_SUPPORT` when `n=0` |
| MAE | `sum(abs(e)) / n` | `None`, `ZERO_SUPPORT` when `n=0` |
| Bias | `sum(e) / n` | `None`, `ZERO_SUPPORT` when `n=0` |
| NSE | `1 - sum(e**2) / sum((observation - mean_observation)**2)` | `None`, `ZERO_SUPPORT` when `n=0`; `None`, `CONSTANT_REFERENCE` when the denominator is zero |

Even a perfect prediction against constant observations has undefined NSE. No
replacement value is invented. Results include requested and scored counts,
the scored-sample digest, and separate exclusion counts. Exclusion counts may
overlap and are labelled accordingly. Empty support therefore remains visible.
Units must already be aligned by the caller; this module does not transform flow.
Identical floating-point references are explicitly detected, avoiding an invented
NSE caused by rounding their mean. A nonconstant reference whose squared variance
underflows to zero reports `REFERENCE_VARIANCE_NUMERIC_UNDERFLOW` instead.

## Architecture and NWM comparisons

`compare_architectures` requires exact matching keys and one common SHA256-bound
input-cohort identity. With the default `missing_prediction_policy="reject"`,
missing or unsupported predictions on admitted labels fail. An explicitly chosen
`"shared_intersection"` scores every model on the same remaining samples and
reports how many admitted labels were removed. The label series and policy are
shared across all models in either mode.

`NwmShortRangeRun` requires an explicit source identity and hash, gauge,
initialization, availability timestamp, original valid-time coordinates, numeric
prediction mask and original source-support mask. Availability must not precede
initialization; retrieval time is never substituted. Actual native valid times
must lie at source leads 1–18. Unsupported leads cannot be created by relabelling
the origin or repeating values. Unknown availability cannot be admitted as an
integer clock. Source selection and qualification must occur upstream.

`align_nwm_predictions` joins a source assigned to each exact `(gauge, C)` with
the target's original valid time. Missing runs, unavailable runs, unsupported
native valid times and original false support masks remain explicit exclusions.
Wrong gauges, future source initializations, conflicting source IDs and competing
vintages for one initialization fail and require upstream resolution.

For example, a native run initialized at 00Z and available at 02Z has source
valid times 01Z–18Z. For origin `C=02Z`, the requested H18 targets are 03Z–20Z.
Only origin-relative horizons 1–16 are supported, using source leads 3–18.
Horizons 17 and 18 remain unsupported. Native lead 18 is never extended to 20Z.

`compare_with_nwm` requires the original source-run map and aligns support
internally. A prealigned series with manually declared support is rejected by
that comparison entry point. It returns the architecture-only result and a
separately labelled matched NWM subset. NWM missingness never changes the independent architecture
result or its input-cohort digest. This API accepts no training cohort and cannot
change training eligibility. The external subset must not be presented as full
independent coverage.

The comparison also returns a digest binding original source identities, hashes,
initialization/availability/valid clocks, original masks, exact target keys and
alignment reasons. A portable summary reports source ages, genuinely supported
native leads and exclusion reasons. It contains no individual forecast values.

## Protected scoring

Every scoring and NWM alignment entry point calls `ScoringPermit.check` before
inspecting numerical values. The default permit disables protected scoring. It
uses the existing `require_scoring_allowed` gate and checks the full H18 window
for every supplied origin, including when a caller reports only selected leads.
An early development origin whose H18 crosses the protected boundary is rejected.

Protected enablement additionally requires explicit protocol and selection lock
digests under the existing guard. Those strings are structural evidence pointers,
not authentication, scientific approval, or a study freeze. A caller must perform
the same guard before loading protected source values; this pure API has no data
loader. No protected values are read or scored by the development tests.
