# Confirmed scope, actual-count runtime scenarios

The user confirmed 180 fits: paired full/Q-only L168 across the three proposed CV folds and final development fits, plus a final full-input L24 contrast on the same L168 common cases. Fold-window and label-delay conventions, stopping/convergence and remaining scientific choices are still pending. Scope confirmation is not a protocol freeze or authorization supplied by this report to execute fits.

The complete development snapshot contains 159,084 candidate gauge-origin cases and 146,464 common-L168 complete cases. `reports/proposed-fold-admitted-counts.json` counts actual members within each proposed fit, inner-stopping and outer-scoring window for each gauge. `scripts/plan_admitted_workload.py` validates that report against the complete admission audit, their content/file/source/snapshot hashes, parent selection identities, exact proposed windows and calendar bounds. It reads JSON receipts only and never reopens source/model arrays.

The output is `reports/confirmed-180-fit-admitted-runtime.json`. It uses the measured full-L168, Q-only-L168 and full-L24 B64 rates with the previously selected CPU/MPS policy. Each gauge's own counts determine its batches; one gauge is not multiplied by three. There is no Q-only L24 or CUDA proxy in this 180-fit scope.

| Epoch sizing scenario | Proposed label delay | Total local hours | CV portion | Final development plus L24 contrast |
|---|---:|---:|---:|---:|
| 20 | 0h | 33.09 | 10.43 | 22.66 |
| 20 | 2h | 33.09 | 10.43 | 22.66 |
| 50 | 0h | 82.45 | 25.95 | 56.50 |
| 50 | 2h | 82.45 | 25.95 | 56.50 |

The two-hour convention removes one more inner-stopping case per gauge/fold. These particular differences do not change the number of B64 batches, so the rounded-batch runtime totals coincide. The policies and case identities are still different; equal sizing totals do not select either convention.

| Gauge | 20-epoch total | 50-epoch total |
|---|---:|---:|
| 03161000 | 11.59h | 28.90h |
| 03164000 | 10.35h | 25.79h |
| 03479000 | 11.14h | 27.76h |

These are conditional runtime calculations, not convergence forecasts or elapsed-time guarantees. They charge measured startup, training/inner-stopping/checkpoint writing each epoch and one outer prediction pass; partial batches count as full. They exclude full-cohort loading/scaling, scoring/resampling/outage studies/figures, retries, contention, sleep and operator delays. Fixed engineering timing transformations do not determine the scientific transform. No model was fit or selected by this addendum.

```sh
python3 scripts/plan_admitted_workload.py \
  --counts reports/proposed-fold-admitted-counts.json \
  --admission reports/development-corpus-admission-v1.json \
  --measurements reports/real-batch-benchmark.json \
  --controls reports/timing-controls.json \
  --output .local/new-confirmed-180-runtime.json
```

Earlier calendar/proxy reports remain unchanged. Their 180-fit estimates were 35.85/89.34 hours for 20/50 epochs with measured controls and calendar counts. The new 33.09/82.45-hour figures replace the count assumptions for this confirmed scope while retaining the original timing evidence and the unresolved protocol boundaries.
