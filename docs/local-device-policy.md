# Fastest demonstrated local backend policy

**Qualification update:** the [primary campaign readiness receipt](primary-campaign-readiness.md)
now records successful B64 fresh-process recovery/best-inference checks for all12
planned backend/geometry cells and all27 real-data views. The earlier remaining
qualification notes below describe the state before this completed work. Final
protocol confirmation remains pending; no scientific fit has started.

The user first requested MPS for all experiments, then clarified: **“Ok go with
what’s fastest!”** The active plan uses CPU for LSTM and GRU–Transformer, and MPS
for Transformer and Mamba-2–Transformer. This follows the recorded full-L168/B64
engineering means; it does not establish an optimum for every possible feature
shape, batch size or complete scientific run.

The [active v3 seed roster](../reports/local-seed-sequence-fastest-v3.json)
preserves all180 jobs, cohorts, roles, seeds, B64, FP32 and the requested108/36/36
seed completion milestones. Its selected device agrees with the actual timing
source for all12 architecture/input/history combinations. The original v1
mixed-backend roster remains unchanged. The
[all-MPS v2 roster](../reports/local-seed-sequence-all-mps-v2.json) is retained as
the evaluated alternative, not the active execution request.

## Measured comparison

The full-L168/B64 benchmark timed20 engineering updates after5 warmups, using the
same retained data fixture and fixed unit mappings. These means include input
copying and the optimizer update:

| Architecture | CPU milliseconds | MPS milliseconds | Active backend |
|---|---:|---:|---|
| LSTM | 34.51 | 55.10 | CPU |
| Transformer | 122.47 | 91.21 | MPS |
| GRU–Transformer | 49.61 | 155.36 | CPU |
| Mamba-2–Transformer | 263.79 | 110.88 | MPS |

[Timing review](../reports/all-mps-timing-review-v1.json) binds the original
benchmark and control hashes. It introduces no new model run. MPS can behave
differently on other model sizes/workloads. No forecast quality selected a device.

## Timing attribution

The all-MPS alternative has matching engineering rates for126 jobs. Its54
unmeasured jobs are LSTM/GRU–Transformer Q-only L168 and full-input L24: their
existing controls ran on CPU. The all-MPS report leaves its complete matched
campaign estimate null. Its supplemental full-L168 MPS proxy scenario is neither
an empirical complete estimate nor an upper bound. Do not use it to claim a
precise campaign-wide speed advantage.

The active mixed policy has matching backend/geometry engineering rates for all
180 jobs. Its conditional totals remain33.09/82.45 hours at20/50 epochs and329.26
hours if every fit reaches the proposed200-epoch ceiling. These retain the
previous overhead exclusions and are not expected stopping times or complete
scientific-run measurements. No deadline guarantee is introduced.

## Execution safeguards and remaining work

Fresh MPS workers must set `PYTORCH_ENABLE_MPS_FALLBACK=0` before importing Torch.
An unavailable backend or unsupported operation fails visibly instead of
silently changing the assigned fit to CPU. CPU loading, preprocessing and metric
aggregation are distinct from the neural training/inference device. Checkpoint
backend/runtime identity stays fixed within a fit and exact resume. The current
unfused Mamba math uses the selected tensor device; it is not a CPU fallback.

Current-schema trainer recovery and best-state inference have narrower completed
qualification than the B64 throughput matrix. Actual main-study backend/geometry
integration, larger role-limited bundles and the durable campaign controller
still need qualification. Retain existing checks; never reset consumed update
allowances. The optional H4 pilot remains a separate CPU/LSTM implementation and
is not inserted into this180-fit schedule.

Fourteen metadata-only tests passed across the new comparison and active-policy
planners. They cover unchanged identities/order, timing attribution, explicit
missing MPS measurements, changed-winner rejection, no fallback and execution
gates. No study-data values, model calls, optimizer updates, scientific scores or
cloud resources were used.
