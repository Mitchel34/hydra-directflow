# Phase A development coverage extension — November 5, 2026

The existing models can issue substantially more forecasts when admission uses
inputs alone and a clean 24-hour model handles gaps in the older part of the
168-hour history. This expands coverage without filling observations or training
new models. It does not establish better accuracy or operational reliability.

The [complete results](../reports/coverage-extension-v1/review.md),
[machine-readable summary](../reports/coverage-extension-v1/summary.json) and
48 linked detail files retain all four architectures, three gauges, three seeds,
four forecast products, five reporting cohorts and 18 leads. These are a
separately registered **2023–24 development sensitivity**. The primary 180-fit
experiment and protected assessment are unchanged.

## Coverage and remaining gaps

| Gauge | Original common origins | Additional L168 | Clean L24 fallback | Routed availability |
|---|---:|---:|---:|---:|
| Jefferson | 8,766 | 0 | 0 | 100.00% |
| Galax | 8,125 | 56 | 432 | 98.25% |
| Sugar Grove | 7,229 | 136 | 1,027 | 95.73% |

Each gauge has 8,766 requested origins. Overall, 1,651 gauge/origin cases were
added: 192 through input-only L168 admission and 1,459 through clean L24 fallback.
These counts are not multiplied by model or seed. Forecast availability is
distinct from evaluable target coverage, and neither is a measure of accuracy.

All 153 remaining Galax and 374 Sugar Grove unavailable origins have incomplete
recent 24-hour discharge inputs. No non-Q input failures occurred in this
development calendar. The structural census reports the lengths of gaps across
L168; repeated origins can share a missing observation and are not independent
outages. Longer and multiple gaps remain excluded. No precipitation, soil state,
historical discharge or verification target was filled in Phase A.

Known-high target-cell coverage increased from 69.20% to 87.85% at Galax and from
50.00% to 96.74% at Sugar Grove; Jefferson remained 100%. The corresponding
unique high target-hour coverage with any lead is 186/193, 333/336 and 192/192.
With all 18 leads it is 156/193, 320/336 and 192/192. HIGH uses the unchanged
gradient q95 threshold, not an official flood stage. These denominators differ
from the count of observed event episodes.

Galax's largest catalog peak remains unavailable at H1–H3 but is now supported
at H4–H18. Sugar Grove's largest peak, previously absent from the primary common
forecast set, is supported at all 18 leads. All 11 Sugar Grove catalog peaks have
all-lead forecasts; this does not mean their magnitudes are accurately predicted.

## Accuracy and interpretation

On identical evaluable support, the three-seed mean routed-minus-L24 NSE is
positive in 3 of 12 gauge/model comparisons and negative in 9 of 12. Differences
range from approximately −0.0140 to +0.0116. No new uncertainty intervals or
significance claim are attached. This is evidence against assuming that choosing
L168 whenever available is universally preferable to the clean L24 model.

The fixed-event figures show substantial underprediction of Sugar Grove's
largest peak despite its recovered forecast coverage. Other events show timing
and magnitude discrepancies at longer leads. A usable forecast is not necessarily
an accurate forecast. The report retains these events rather than replacing
them with better-performing examples. Each model curve is a three-seed prediction
mean for display; metric summaries average individual seed scores, and shaded
seed ranges are not confidence intervals.

At Sugar Grove's largest event, the observed peak is 186.04 m³/s, while the
four architectures' descriptive three-seed mean H18 predictions at that time
range from 23.56 to 41.76 m³/s. At Galax's largest event, the corresponding
observed peak is 597.49 m³/s and H18 means range from 90.39 to 129.34 m³/s.
These are forecasts at the observed peak time, not an inferred forecast-peak
timing statistic. Coverage repair alone has not been shown to resolve these errors.

Galax's newly added L168 cohort spans only 3–6 dates per lead; the L24 fallback
spans 20–21 dates. Their primary NSE is correctly undefined under the fixed
28-date rule, with physical-unit descriptive errors still reported. The largest
emitted negative fraction across the natural L24 results is 216/8,392 (2.57%),
for Sugar Grove GRU–Transformer, seed 17, H17. Raw negatives remain in all scores.

Scores on newly recovered cases are descriptive results on a different event
population, not paired improvements over the old common cohort. All original
common predictions reproduced exactly, and the corresponding old metrics agree
within numerical arithmetic precision. There is no Q-only L24 comparator, so
the routed extension cannot isolate a weather/soil effect. Historical data
vintage, the simulated two-hour delay, approximate basins and Galax's longer-lag
residual dependence remain limitations. Protected assessment stays sealed.

## Figures

- [Evaluable coverage by lead](../figures/coverage-extension-v1/phase-a-valid_target_coverage.svg)
- [Known-high-flow coverage by lead](../figures/coverage-extension-v1/phase-a-high_flow_coverage.svg)
- [Paired routed versus L24 NSE](../figures/coverage-extension-v1/phase-a-paired_routed_l24.svg)
- [Fixed top-three event hydrographs at H1](../figures/coverage-extension-v1/phase-a-fixed-events-H01.svg)
- [Fixed top-three event hydrographs at H18](../figures/coverage-extension-v1/phase-a-fixed-events-H18.svg)

All five figures were visually reviewed. Public exports use SVG; raw arrays,
checkpoints, detailed per-origin provenance and PNG previews remain private.

## Execution and next decision

The local run completed all 152 scheduled tasks in 22 minutes 53 seconds at
18:22:30 Central on November 5. It authenticated 72 existing checkpoints,
reproduced 72 original prediction sets byte-exactly, performed 600,996 charged
case/checkpoint forwards in 9,420 batches, and scored 144 products with 48
three-seed aggregates. Independent verification covered 12,960 per-lead panels;
the maximum absolute metric difference was 1.46e-11. Final artifact review also
authenticated 1,095 referenced files. No execution failure or worker resume
occurred. A Codex authentication interruption stopped agent supervision at
18:10, while the already running local supervisor completed normally.

There were zero new optimizer updates, scaler fits, paid jobs, downloads or
protected-data access. The original scientific update count remains 1,151,690.
The final verification receipt records the bounded review/publication charges
separately from the report's earlier accounting snapshot.

The [Phase B decision packet](coverage-extension-phase-b-decision.md) proposes
short-gap research, with explicit policies, paired artificial gaps, provisional
tolerances and measured runtime estimates. Its structural census suggests at
most 50 additional Galax and 82 Sugar Grove origins under the proposed two-slot
cap. No repairs or new scientific fits were run. Complete the development-results
poster from the reviewed evidence, then review that optional sensitivity before
authorizing it. Phase A completion is not completion of the entire AGU study.
