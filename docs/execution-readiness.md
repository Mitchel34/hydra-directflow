# Execution readiness and approved conditional budget

Updated November 2, 2026. Engineering measurements below are complete; scientific
fits, CV adoption, full-cohort numerical admission and protected assessment are
not. There is no need to wait for November 4 or6 if the prerequisites finish early.

## Completed implementation

- Confirmed A/A+R metadata admission audit for both development splits, all
  gauges and L24/L168. After the approved owned-file materialization,72 native
  months and the verified July export resolve all73 development months, with
  zero unknown metadata months. Earlier native/aggregate audits remain preserved.
- Corrected the over-conservative weather initialization guard. Actual targets
  must still remain before protected assessment. A separately authorized local
  July2024 export now contains only preboundary rows; protected scoring remains off.
- Train-only affine transforms, deterministic training-order sampler, explicit
  stopping state and checkpoint recovery. A six-update synthetic qualification
  reproduces the next update and sample order in fresh CPU/MPS processes.
- A real August benchmark:24 cases,600 updates (120 warmup and480 measured),
 120.15seconds,1.66GiB peak observed combined memory. All checkpoint forward
 replays pass. These benchmark weights are not selected scientific models.

## Data coverage and storage

Joint USGS metadata admission at L168 is now fully enumerated:

| Gauge | Training known eligible / unknown | Validation known eligible / unknown |
|---|---:|---:|
| 03161000 | 42,828 /0 | 8,766 /0 |
| 03164000 | 38,403 /0 | 8,125 /0 |
| 03479000 | 41,445 /0 | 7,229 /0 |

These are not final full-model cohorts: HRRR numerical basin masks remain to be
verified. The evidence is `reports/development-eligibility-owned-metadata-v2.json`;
its audit reads no discharge-value columns or raw July2024. UNKNOWN in older
reports described incomplete inspection, not an absent observation or permission
to train. The current complete-input/complete-target
count scenario also differs from a future policy allowing partial histories or
partially labelled targets.

Local original commit metadata now proves exact bundle sizes for4,805 required
slots, totaling26,566,963,200bytes (24.74GiB). Another4,036 sealed rows lack exact
bundle sizes in this inventory; three candidate warmup slots precede the acquired
calendar. If every remaining archive satisfies the16MiB reader bound, the sealed
source-size scenario is87.80GiB. This is a conditional ceiling, not an exact full
inventory. A one-month average suggested about44–46GiB. The approved streaming
stager verifies exact manifests before each archive read. The11 owned USGS files
were materialized under the new approval, totaling774,216 encoded bytes; their
FileProvider wire transfer was not observed. All copies match the original
manifest, and the original contents are unchanged.

## Measured local execution settings

At common batch64/FP32, CPU is faster for LSTM and GRU–Transformer; MPS is faster
for Transformer and Mamba2–Transformer. This choice uses throughput, not forecast
accuracy. CPU workers used one intra/inter-op thread on this M1 Pro/32GiB Mac;
no concurrent CPU+MPS speedup is assumed.

| Architecture | Mean CPU train batch | Mean MPS train batch |
|---|---:|---:|
| LSTM | 34.51ms | 55.10ms |
| Transformer | 122.47ms | 91.21ms |
| GRU–Transformer | 49.61ms | 155.36ms |
| Mamba2–Transformer | 263.79ms | 110.88ms |

For four models × three gauges × three seeds, one configuration (**36 fits**),
the mixed B64 policy gives **9.54h or10.49h at20epochs**, and **23.83h or26.20h at
50epochs**, using refined known-USGS versus post-purge calendar case scenarios.
These are distinct workload calculations, not confidence bounds. They include
measured validation and checkpoint costs each epoch but exclude full-corpus I/O,
convergence, failures, thermal/sleep delays and figure production. Using full-input
cost as a sizing proxy for matching Q-only fits roughly doubles the work; Q-only
throughput is not separately measured here. CV/lookback/component arms add work.

The main matrix therefore appears locally feasible once the cohort and protocol
are ready. The expanded investigation, travel interruptions or slow convergence
could threaten the dates. Recompute after exact admission counts and the approved
matrix are available. `scripts/estimate_training_runtime.py` reproduces the
earlier scenarios without fitting models. The separate proposed-CV workload
arithmetic in `docs/workload-planning.md` distinguishes gradient fitting, inner
stopping and outer inference and reports the additional input/lookback arms.

The subsequently executed 200-update controls replace Q-only L168 and full L24
timing proxies with measurements. At the 50-epoch cap, the 108-fit final-only
scope is now 61.36 hours and the 180-fit paired-CV plus final-lookback scope is
89.34 hours. The 288-fit factorial is 135.66 hours, still including 72 unmeasured
Q-only/L24 fits represented by an explicit full-L168 proxy. These remain calendar
count scenarios before final numerical admission and exclude full-corpus loading,
failures, interruptions and scientific review/figure preparation. See
`reports/proposed-scientific-workload-measured-controls.json`. The scope preference
is pending, and these arithmetic scenarios do not adopt CV or start fitting.

## Approved additional funding: $150 total ceiling

**User approved November 2, 2026:** “Approve $150 combined ceiling.”
`reports/execution-budget-approval.json` records the authorization and its phase
conditions. This is one incremental envelope for the existing study, including
contingency, not a predicted bill. The completed August allowance and historical
acquisition liabilities are separate. Development staging is released; cloud
funding remains reserved pending measured need and qualification.

| Allocation | Approved allowance | Conditions |
|---|---:|---|
| Development source staging | $35 | Existing generation/hash-pinned objects only; up to100GiB total response bodies, bounded2GiB raw staging; materialize only the identified owned USGS files; no new acquisition |
| CUDA qualification/timing pilot | $10 | Only if local availability/runtime threatens readiness; at most one worker and4machine-hours; no scientific selection |
| Approved scientific compute | $75 | Only after protocol/cohort/CUDA qualification; at most two G2 L4 workers using the existing project, ≤$1 per machine-hour and a cost-derived aggregate hour cap |
| Retry/setup/storage contingency | $30 | Owned-job recovery, setup, disks, output retention or transfer variance within the same total; no unapproved experiment expansion |
| **Total** | **$150** | All phases and reserves share one ledger; no automatic increase |

The displayed G2 `g2-standard-4` and `g2-standard-8` on-demand machine rates are
about$0.707 and$0.854/hour respectively. Confirm the actual regional SKU and
capacity before provisioning; CPU/MPS timings do not predict L4 speed.
[Google accelerator pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized).
The staging allowance uses conservative transfer/retrieval rates up to$0.23+
$0.05/GiB plus request charges; storage class/location must be verified.
[Google Storage pricing](https://cloud.google.com/storage/pricing).

The approval permits staged release under one ceiling: stage sources,
qualify local execution, and keep cloud allocation reserved unless needed. The
pilot must establish CUDA correctness and measured throughput before main cloud
fits. Reserve can cover retries or setup without another budget request, but
projected completion must still fit the remaining money and time. Stop rather
than overspend if it cannot. A dollar budget cannot guarantee convergence or
scientific readiness.

Owned cloud resources would need labels, checkpoint/artifact verification,
explicit lifetime/hour limits and independent shutdown/cleanup. Keep source
artifacts immutable; remove only newly created raw staging copies after verified
reductions commit. Existing source caches and unrelated resources are not cleanup
targets. Do not rely on billing alerts alone as a spending control. A read-only
21:31UTC snapshot found all22 acquisition instances terminated and zero GPU quota
usage, with a limit of one L4 in us-central1 and one GPU globally. Plan around one
GPU under existing quota; the approved two-worker ceiling is not reserved capacity.
Zonal capacity, launch permissions and runtime qualification remain unverified;
quota/access/security changes require separate review and authorization if needed.
The snapshot does not reconcile retained disks/storage or invoices. This budget approval does not authorize
protected assessment access, scoring, new model families or hosted monitoring.

Local execution remains the measured first option. The reserved cloud allowance
can be released within the approved conditions if measured need develops;
scientific choices and the assessment gate remain independent of funding.

## Staging implementation and evidence versions

The approved source plan contains 8,841 retained development HRRR cycles and
73 USGS month entries, with July 2024 routed only through the verified development
export. Three warmup cycles precede the acquired calendar and remain explicit
unavailable inputs. No upstream acquisition is repeated.

The corpus plan preserves 53,028 candidate origins after 36 H18 boundary purges,
across 477 resumable chunks. L168 and the complete-case admission count remain
engineering candidates, not frozen protocol choices. The shared source budget
precharges each GET before transport and retains failed charges across resumes.
Only owned raw staging copies may be removed, after a verified derived cache
commit. Original artifacts are retained. See `docs/development-staging.md`.

Earlier inventories, timing measurements and failed increments retain their
original implementation hashes. They are evidence of those executions, not
claims that all subsequent code was exercised by an earlier receipt. Final
cohort eligibility and the updated execution ledger supersede planning counts
only when their own checks finish.
