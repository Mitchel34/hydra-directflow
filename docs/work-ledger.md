# Completed connection-window work

Each code chunk has been tested locally and committed separately. Raw data and local dependency caches are excluded from Git.

| Commit | Chunk | Evidence |
|---|---|---|
| `c0c448b` | Repository foundation | Portable package and publication checks |
| `6b93183` | Causal protocol | 32 contract tests |
| `b9f1866` | Verified development loader | 13 loader tests and a real three-gauge H18 smoke |
| `0fd7cca` | Four-model qualification | CPU/MPS L24/L168, 72 synthetic updates |
| `502467d` | Run telemetry and input/model integration | Eight real-input forwards, zero updates; CI template |
| `25b2bf1` | Offline readiness | 37 local cache members; fresh no-index environment passed55 tests at that checkpoint |
| `6bf39c5` | Fresh-process recovery | CPU LSTM and MPS Mamba2; six synthetic updates |
| `ac64e2d` | Real-data optimizer integration | Two LSTM CPU plus two Mamba2 MPS engineering updates |
| `9970334` | Monitoring template | Terminal-state metrics and Grafana JSON template |
| `385582d` | Evaluation APIs | 15 synthetic metric/cohort/NWM tests |
| `4863559` | Current-shape local benchmark | 20 cases; 800 synthetic updates; 600 timed samples |
| `b2c2f64` | Development availability | Quality/timestamp columns only; 12 comparisons, one month |
| `cd02171` | Q-only control | 11 ablation tests and two zero-update model forwards |

The October 29 suite passed **114 tests** (`reports/release-check.json`), the first November continuation passed **154** (`reports/continuation-check.json`), cache/month preparation passed **211** (`reports/cache-month-readiness-check.json`), the real-month exercise passed **217** (`reports/month-execution-check.json`), and the quality adapter passed **233** (`reports/primary-quality-release.json`). The eligibility/boundary/trainer/real-batch sequence now passes **315 tests** (`reports/implementation-sequence-check.json`). These are local checks; hosted CI remains inactive.

Qualification accounting: 878 synthetic optimizer updates across the model, recovery and benchmark attempts; four completed real-data engineering updates on one repeated observed case. One earlier failed engineering admission precharged two updates but failed before optimizer entry; its record remains. These are not scientific campaign fits or forecast-skill results.

The current local benchmark reports synchronized copy/update timings, not whole-epoch or full-data workflow duration. CPU/MPS choices should be confirmed with representative real data; no CUDA throughput or cloud budget has been established.

## Offline and next steps

The wheel and small input caches are local-only. Verify them with `scripts/verify_offline_cache.py`; follow `docs/offline-work.md`. The full historical HRRR payload corpus remains in private cloud storage.

The November continuation implements the multi-cycle reader and bounded pipeline timing described below. Main study policies and a paid cloud execution budget remain unresolved. Do not restart acquisition or use protected assessment results for selection.

GitHub Actions activation is pending workflow-scoped credentials. The tested core workflow template is in `docs/ci/core-workflow.yml`. Grafana is a JSON template plus local telemetry output; no hosted monitoring service is running.

## November 1 continuation

| Commit or artifact | Tested chunk | Evidence |
|---|---|---|
| `227ae99` | Authenticated multi-cycle preprocessing | Five extended cycles, 20 origins, 60 cases at L24 and L168; ordinary-run H18 rejection; 136 tests at this checkpoint |
| `18b7ce7` | Local end-to-end pipeline timing | All four models on CPU/MPS; 48 confirmed engineering updates; exact checkpoint forwards; 151 tests |
| `scripts/project_runtime.py` and `reports/runtime-scenarios.json` | Runtime sensitivity and budget preparation | Independently recomputed local scenarios; three arithmetic regressions; 154 integrated tests |

The November engineering campaign completed in 39.60 seconds with peak combined
memory of 883 MiB. The cumulative record now contains **52 completed real-data
engineering updates** (four earlier plus 48 new) and the unchanged 878 synthetic
updates. The earlier failed two-update allowance remains recorded separately.
There has been no full scientific fit or protected assessment scoring.

The new private source cache contains 17 authenticated files totaling 29,279,895
bytes, including six HRRR archives/commits and five monthly USGS partitions.
The original offline dependency cache remains valid. Source files, tensors and
checkpoints remain outside Git. Paid cloud jobs remain on hold.

Next: authenticated legacy-index adapters and resumable per-cycle basin/hourly
discharge caches, followed by a representative-month timing run. See
`docs/runtime-and-budget.md` for scale estimates and current limitations.

## Legacy and cache continuation

| Commit or artifact | Tested chunk | Evidence |
|---|---|---|
| `dce3e1f` | Legacy index and payload adapter | Three actual numeric cases, one additional identity-only case; 16 dedicated tests; 171 integrated tests |
| `90b2ba5` | Resumable basin/discharge caches | Exact L24/L168 uncached/cache/resume array parity; 16 cache tests; 187 integrated tests |
| `scripts/time_month.py` and `scripts/fetch_month_inputs.py` | Complete-month runner and bounded existing-object retrieval preparation | 5 monthly/checkpoint tests and 19 mocked retrieval tests; 211 integrated tests; no month execution yet |

The caches materialize five basin cycles and five full USGS months in 718,453
bytes including commit metadata. Subsequent parity runs reused all ten entries
without source reduction or cache writes. Source qualifiers, masks and identities
remain preserved. Route-specific receipt row counts differ between full-month
stores and window-filtered readers, so array parity is reported separately.

August 2023 requires 127 six-hour cycles and 744 hourly origins, with July/August/
September USGS support. Three archives are resident; the 124 missing archives
need the separately requested storage-read approval (1 GiB and $1 estimated
read/transfer allowance). The actual full-month timing is pending that approval.
No credentials or storage downloads were requested by this preparation, no cloud
job was started, and optimizer-update totals are unchanged.

Next: complete the approved storage reads if authorized, then cold/interrupted,
resumed, warm full-month and verification-only timing. Preserve the distinction
between those timing modes and between source gaps and incomplete acquisition.

## Approved August execution — November 2

- `12b3613`: approved storage retrieval completed; 137 verified requests,
  634.52 MiB transferred, 51.46 seconds, estimated incremental fees $0.1804.
- `7b5fe0b`: added narrow admission for original census-proof files in the
  genuine soil-only HRRR gap archive. Six regressions and actual source checks
  pass; the first failed monthly attempt remains in its own receipt.
- `reports/month-timing.json`: all 744 origins and 2,232 gauge cases processed;
  2,046 complete cases, 168 missing-history and 18 missing-target cases preserved.
  Cold stop/resume 9.22 s, warm full assembly 5.12 s, verification-only 0.78 s.
  All chunk digests match; all 130 cache entries verified.

The source download allowance was used only for the approved month. The cloud
compute budget remains undecided and no VMs/cloud training jobs were started.
Optimizer totals remain 52 real-data engineering updates and 878 synthetic
updates from earlier work; this month exercise added zero. Scientific training,
protocol freeze and protected assessment remain pending.

Next: use this observed month to refine a bounded full-corpus preprocessing and
model-training proposal. Account for legacy/gap variants, full source sizes,
data-transfer location, eligibility, batching and the remaining scientific
choices before requesting any additional downloads or paid compute.

## Scientific reconciliation and quality adapter — November 2

Baseline at 11:44 Central: clean local/remote `a4d3fd1`, no newer implementation,
five acquisition authorities and all retained August/cache identities verified,
and no related local worker observed. No current cloud-resource query was made.
The old acquisition report and October pause fields do not supersede the current
repository and November checkpoint. See `reports/study-reconciliation-20261102.json`.

The confirmed A-required/optional-R policy is now implemented and tested. P, e,
unknown, empty and malformed qualifiers are rejected. Source ordering/duplicates
are preserved, input and target policies are independent, and no target is filled.
Historical default/receipts remain strict-A; new scientific callers must select
the primary policy explicitly. All arrays from the saved local L168 fixture match
under the unchanged default. Policy-specific source tests and 16 new regressions
pass; full-cohort eligibility remains to be audited. No optimizer updates or new
scientific-data downloads occurred.

Persistence is excluded from the active study following the user's account of
advisor Zoom guidance, with delayed observations and gauge outages as the stated
concerns. This is user-reported guidance, not transcript verification. The learned
Q-only control remains. No advisor message was sent.

`docs/scientific-study-design.md` is a proposal, not a protocol freeze or assessment
permission. It specifies controlled hypotheses, actual model mechanisms/counts,
component-control estimates, expanding-window CV dates, fit counts, uncertainty,
figures, milestones and a separate future edge-AI outline. The accepted abstract
still requires coauthor alignment with the current direct-forecast framing.

## Highest-value implementation sequence — November 2

| Commit/artifact | Completed work | Evidence |
|---|---|---|
| `0f3da47` | Full development metadata inventory and tri-state eligibility |62/73 native months verified,11 nonresident;25 initial tests; no flow columns decoded |
| `72240d0` | Correct source-init guard and separately gated July export |17 boundary/export tests plus4 cache/export tests; protected target guards retained |
| `55c9172` | Train-only transforms and restartable training mechanics |10 pure tests;6 synthetic updates qualify exact fresh-process next update/sample/scaler/optimizer/RNG/stopping recovery |
| `db12e93` | Real August CPU/MPS B16/32/64 benchmark |24cases/600updates,120.15s,1.66GiB;17 zero-update guard tests; exact checkpoint replay |
| `20c1684` and follow-up CLI publication | Aggregate metadata refinement and runtime/budget proposal |34 eligibility tests;29/33 unread gauge-month admission axes proven; original audit preserved |

The user separately approved the July2024 administrative export. It produced only
preboundary development rows and no protected values in logs/results. Subsequent
cache ingestion/reuse verified2,160 exact-hour observations; scoring stayed off.

Runtime scenarios now use actual B64 batches and refined case counts. The best
tested local mixture uses CPU forLSTM/GRU–Transformer andMPS forTransformer/Mamba2.
That is a throughput decision, not forecast selection. The36-fit main-matrix
scenarios are9.54/10.49h at20epochs and23.83/26.20h at50epochs. CV,controls,
convergence and complete data materialization are additional work.

Update accounting is now **652 completed real-data engineering updates** and
**884 synthetic updates**. This sequence added600 real benchmark updates and6
synthetic recovery updates, not scientific campaign fits. The earlier failed
two-update allowance remains separate. No protected scores or paid cloud jobs
were produced. Additional data staging and the proposed$150 funding envelope
remain pending authorization; scientific protocol freeze remains independent.

## Approved staging and resumable corpus — November 2, 21:15 UTC

The user approved the combined **$150 incremental ceiling**: $35 development
staging, $10 conditional CUDA pilot, $75 conditional scientific compute and $30
contingency. The approval is recorded in `reports/execution-budget-approval.json`.
Development staging is active; compute release still requires measured need and
qualification. Protected assessment scoring and IAM/security changes remain off.

The source plan binds8,841 sealed development cycles and73 USGS month entries.
The candidate L168 corpus contains53,028 origins/159,084 gauge cases in477 chunks,
after36 H18 split-boundary purges. It is not a protocol freeze. All source attempts
share a persistent precharge ledger; failed increments do not reset spending.

Three actual historical formats now pass narrow reader checks: zero-record
metadata-only gaps, original404/component-metadata recovery, and float32 weather
tiles. Nineteen dedicated regressions pass. A resident-source preflight passes196
checks across191 unique cycles, rejects two wrong-identity variants and skips35
without local authority. Old failures and executed-code hashes remain intact.

Increment04 committed20 chunks/2,304 origins/6,912 gauge cases, of which6,513 meet
the current candidate complete-case rule. Cumulative verified GETs are838 and
observed response bodies2,220,295,776bytes; the conservative estimate is$0.633458,
not a billing invoice. No native USGS months remain unresolved by the staging
resolver. Increment05 continues under the same budget and code-bound v3 plan.

The integrated suite passes406 tests with zero optimizer updates. Staging performs
no scientific fits or protected scoring. Next: finish and verify the corpus,
publish exact admission/exclusion counts, qualify its read-only training adapter,
and resolve the remaining scientific protocol decisions before fitting models.
