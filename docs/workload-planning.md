# Proposed scientific workload and schedule implications

`scripts/plan_scientific_workload.py` performs date and cost arithmetic only. It
does not implement or adopt model-selection cross-validation, admit numerical
cases, freeze a protocol, fit a model or authorize protected scoring. The original
proxy-only output remains unchanged at `reports/proposed-scientific-workload.json`.
The current measured-controls scenario is
`reports/proposed-scientific-workload-measured-controls.json`.

The scenarios use the three expanding annual folds and the separate final
development check proposed in `docs/scientific-study-design.md`. Each CV fold
refits once, with its own gradient-fit, inner-stopping and outer-scoring periods.
Hourly rolling forecasts do not imply hourly refits. Preprocessing parameters
must be fit only on the gradient-fit subset of each independent fit.

Calendar counts respect H18: `C+18h < partition_end` removes18 final origins.
Under the additional proposed simulated two-hour label-availability convention,
the inner-stop labels must be usable at refit time B: `C+18h <= B-2h`. That
removes19 origins, since C=B-20h remains allowed. The earlier gradient-fit end
still purges18 origins. These are candidate simulated-availability rules, not
evidence about historical publication/revision times.
Fitting, tuning and selection elapsed time are not simulated. These counts do
not establish that a selected model was operationally ready at the first outer
forecast origin; the proposed retrospective design must not be described as an
operational replay.

All estimates use B64 and the previously selected local backend for each family.
They include measured startup/warmup excess, training plus inner-stop evaluation
and checkpoint writing every epoch, then one outer prediction pass. Partial
batches are charged as full. The three final seeds are17/29/43; CV uses one seed
and one configuration per family unless explicitly changed in the plan. No
favorable seed is selected and no validation-period refit is assumed.

| Candidate scope | Fits | 20 epochs | 50 epochs |
|---|---:|---:|---:|
| Four families, three gauges, three final seeds, full-input L168 | 36 | 9.60h | 23.94h |
| Same final fits plus matched Q-only | 72 | 19.00h | 47.37h |
| Same matched arms plus three annual CV folds | 144 | 30.24h | 75.35h |
| Both L168 and L24 with matched arms and CV; Q-only L24 remains a proxy | 288 | 54.44h | 135.66h |

Two smaller lookback-control scopes are also explicit in the report:

| Composite candidate scope | Fits | 20 epochs | 50 epochs |
|---|---:|---:|---:|
| Final full/Q-only L168 plus final full-input L24 | 108 | 24.61h | 61.36h |
| Paired L168 CV/final fits plus final full-input L24 | 180 | 35.85h | 89.34h |

The L24 contrast adds36 final fits on the same L168-admissible cases, using the
same four families, three gauges and three seeds. It tests longer history under
the full input bundle, without claiming a full history-by-input factorial or
L24 cross-validation. Natural L24 coverage is reported separately. All scopes
remain proposals; the scope preference is being clarified before protocol freeze.

The new controls receipt, `reports/timing-controls.json`, records eight passed
cases: four architectures in Q-only L168 and full-input L24 modes. It contains
200 engineering updates (40 warmup and 160 measured), 32.18 seconds of campaign
wall time and 1.68 GiB maximum observed combined memory. The original 600-update
full-input L168 receipt is unchanged. No new optimizer run is performed by this
planner.

With `--controls`, the planner checks matching sealed fixture/array and source
receipt identities, the parent benchmark plan, shared implementation hashes,
case membership and target-coordinate digests, update accounting, checkpoint
qualification and the recorded B64/device/FP32/width64 runtime geometry. Parameter
counts must match each input arm; Q-only has 640 fewer projection parameters,
so this is not an exact-capacity comparison. Runtime metadata are compared on
the fields recorded by the benchmarks, not a unique physical-machine identifier.

Each scenario part now selects its own `(architecture, input arm, history)` rate.
Full L168, Q-only L168 and full L24 use their corresponding measured rows. Q-only
L24 has no measurement and retains the original **full-L168 sizing proxy** with
explicit source-geometry and mismatch fields. No shorter-history speedup is
invented for that arm. Omitting `--controls` preserves the previous proxy
behavior, and existing output files cannot be overwritten.

```sh
python3 scripts/plan_scientific_workload.py \
  --measurements reports/real-batch-benchmark.json \
  --controls reports/timing-controls.json \
  --output .local/new-measured-workload-scenario.json
```

These are calendar-count scenarios before numerical eligibility, not runtime
bounds, forecasts of convergence or a selected experiment matrix. The control
benchmarks used the same fixed engineering unit maps, not a newly selected
scientific scaler. Their devices were held at the prior full-L168 policy rather
than retuned for each arm. No CUDA acceleration or simultaneous CPU/MPS speedup
is assumed. Full-cohort
loading/scaling, scoring/resampling/figures, guard overhead outside measured
batch calls, retries, sleep, travel and thermal contention remain additional.
Missing-input exclusions may reduce counts; they must not be selected to favor
model results. Actual fold/common-case counts will replace calendar counts after
the corpus audit.

The earlier36-fit report used full canonical training/validation counts and
per-epoch validation sizing. This proposal instead distinguishes the inner
stopping tail from the annual outer prediction pass. Both reports retain their
original assumptions and hashes. The proxy-only workload report is also retained;
the new report combines the two existing benchmark receipts without rerunning
models or changing their outcomes.

The local72-fit matrix appears plausible within the November6 target, contingent
on admission, protocol readiness and uninterrupted runtime. The144-fit proposal
leaves little room for the remaining work; the288-fit proposal exceeds the time
available if it runs near the50-epoch scenario. Early stopping cannot be assumed
to rescue a schedule. Final scope and actual admission should determine whether
the already approved conditional CUDA pilot is needed. A successful pilot must
measure real CUDA correctness, recovery and throughput before pricing the cloud
campaign within the existing combined ceiling. No cloud job is started by this
planner and no new funding is requested here.
