# Assessment first pass: saved FULL L168 weights

All 36 saved models completed and the independent numerical/provenance audit passed.
Origins: 2024-07-31 through 2025-07-30 UTC; the final H18 target is 2025-07-31 17:00 UTC.
No training, transform refit, clipping, repair or seed selection was performed.

Matched H1-16 equal-lead NSE (HYDRA: arithmetic mean of all 3 seed metrics):

| Gauge | NWM | LSTM | Transformer | GRU-Transformer | Mamba2-Transformer |
|---|---:|---:|---:|---:|---:|
| Jefferson | 0.632 | 0.422 | 0.428 | 0.406 | 0.435 |
| Galax | 0.878 | 0.414 | 0.343 | 0.335 | 0.335 |
| Sugar Grove | 0.459 | 0.540 | 0.424 | 0.424 | 0.508 |

NWM has higher matched mean NSE than all four models at Jefferson and Galax.
At Sugar Grove, LSTM and Mamba2-Transformer have higher seed means than NWM;
the seed ranges are substantial. These are descriptive point comparisons,
not significance claims or model-selection decisions. Sugar Grove's NWM lead
profile is nonmonotonic; a higher H1-16 mean does not establish superiority
at every lead. Event-level diagnosis remains pending.

| Gauge | Forecasts available / 8,760 | Complete-H18 target origins | NWM-matched origins / lead |
|---|---:|---:|---:|
| Jefferson | 8,760 (100.00%) | 8,760 | 8,712 |
| Galax | 7,805 (89.10%) | 7,765 | 7,717 |
| Sugar Grove | 7,391 (84.37%) | 7,313 | 7,265 |

NWM is an external comparator. Exact C-2 issues support origin H1-16; H17-18
are explicitly unavailable. Independent HYDRA H1-18 primary and per-lead
valid-target secondary scores remain separate from the matched H1-16 comparison.

This is the strict assessment baseline. Prior development-pool or coverage-router
availability does not describe these assessment forecasts. No assessment Q-only,
L24/router or peak-event analysis is claimed here. The poster retains the known
development peak-underprediction limitation with its development-period label.

All three seeds 17/29/43 are retained. Min/max ranges are descriptive, not confidence
intervals. Forecast windows overlap and are not independent floods. The observation
delay is simulated using revised approved A/A+R data; it is not a reconstruction
of historical USGS publication times. Basin footprints remain approximate.

`summary.json` holds coverage, NWM and all seed aggregates. Each gauge/model JSON
retains every seed and support panel; `lead-metrics.csv` provides physical-unit
metrics. Empty CSV metrics mean undefined or structurally unsupported, never zero.
PBIAS uses 100*sum(prediction-observation)/sum(observation).
