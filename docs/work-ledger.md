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

## Matched Q-only training and conditional CUDA preparation — November 2

Trainer/transform/checkpoint v2 now carries explicit full-input and Q-only schemas
and a shared cohort admission digest. Q-only never reads the weather/soil values
or masks and keeps the same cases, labels and training-only history/target
statistics. The known horizon uses fixed h/18 encoding. Old v1 checkpoints are
not silently migrated. Scientific transformation/loss choices remain pending.

`reports/trainer-recovery-v2.json` records12 precharged, entered and confirmed
synthetic updates across full/Q-only CPU LSTM and MPS Mamba2 paths at L24/B4.
All four fresh-process continuations reproduce the next sample and optimizer
update exactly, including model/scaler/sampler/RNG/stopping state. Qualification
took19.45s with551.23MiB sampled peak; it does not establish every model, batch
or history geometry.44 focused training/ablation/device tests pass without new
updates. Cumulative accounting is652 real-data engineering and896 synthetic
updates, with zero scientific campaign fits.

The new device helpers and CUDA plan are non-executing preparation. The trainer
still refuses CUDA; no hardware, GPU throughput or cloud execution is claimed.
The reserved pilot remains conditional on measured need and bounded runtime,
budget, correctness and recovery qualification.

## Complete USGS metadata coverage and storage interruption — November 2

The approved owned-file materialization and independent audit verify all11 copies
and original contents:774,216 encoded bytes,11 verified attempts,774,227 retained
precharged bytes. FileProvider wire/HTTP accounting remains unobserved. The new
script-only metadata adapter validates the exact source set before reads and
uses July's safe export. Its8 tests pass; the corpus-bound core is unchanged.

`reports/development-eligibility-owned-metadata-v2.json` now resolves all73 months,
with zero unknown USGS metadata windows. L168 joint eligible training counts are
42,828/38,403/41,445 for Jefferson/Galax/Sugar Grove; validation counts are
8,766/8,125/7,229. These are metadata counts, not final numerical HRRR cohorts.
The audit reads no discharge-value columns and no raw July source bytes.

Increment05 stopped at21:27:39UTC on an HTTPError for a pinned20190730_18Z commit,
after61 new chunks and81 total committed chunks/9,096 origins. The failed request
remains charged. A separately recorded bounded diagnostic verified that exact
commit with a fresh client; the original HTTP status was not retained, so the
precise transient/authentication cause is unconfirmed. No identity or generation
was substituted. Increment06 resumes the same v3 corpus and persistent budget.
Cumulative precharges after the diagnostic are7,864,343,921bytes/3,138GETs with
one failed attempt and a conservative$2.254758 estimate, not an invoice.

## Proposed workload arithmetic and cloud inventory — November 2

The non-executing workload planner passes15 tests and independent date/rate/fit
arithmetic review. It separates gradient fitting, inner stopping and outer
prediction. Strict H18 partition ends purge18 origins; the proposed simulated
two-hour label-availability rule at refit purges19 from the inner stopping tail.
The planning calculation does not establish historical model-selection readiness.

At50 epochs, calendar-count scenarios are23.94h/36 full-input final fits,
47.88h/72 matched full/Q-only final fits,76.16h/144 fits including paired annual
CV, and152.31h/288 fits for both histories with paired CV. A final-only full-input
L24 contrast adds36 fits:108 fits/71.82h without CV, or180 fits/100.10h with paired
L168 CV. Q-only/L24 timings remain explicit full-L168 proxies. The user scope
preference is pending; no CV or scientific fitting was executed.

The21:31UTC read-only Compute Engine snapshot found22 acquisition-labelled
instances, all terminated; global GPU and us-central1 L4 quotas each have limit1
and usage0. No resources or permissions were changed. One GPU is the current
planning limit, not a capacity reservation or successful CUDA qualification.
Retained disks/storage liabilities and provider invoices were not reconciled.

## Read-only dataset adapter — November 2

The new adapter verifies immutable plans/shards, numerical schemas, clocks,
masks and source ages before yielding bounded per-gauge batches. It explicitly
separates all candidates, natural L24/L168 completeness and common L168 case
membership. Full/Q-only views preserve that membership, history and labels.
Partial snapshots cannot become final cohorts by default; later appends create
a new snapshot rather than changing an existing selection.14 tests pass.

`reports/corpus-dataset-partial-check.json` records a bounded earlier snapshot
of132 committed chunks/44,136 candidate cases, with40,290 candidate L168-complete
and42,166 natural L24-complete cases. Sixteen paired full/Q-only batch cases have
exactly matching histories, labels, clocks and IDs. Inspection took1.30s and
50.6MiB peak RSS. These are provisional development counts, not a complete cohort,
source/cache re-audit, trained result or protocol freeze.

## Progress inspector and legacy-format stop — November 2

The read-only status CLI passes12 focused tests and independent review. It checks
pinned plan/commit/case-index metadata, stats but never opens numeric shards,
reconciles the SQLite budget in a read transaction, and keeps worker observation
separate from old receipts and lockfiles. It never settles pending charges.

Increment06 stopped at21:40:37UTC on legacy20200422_00Z metadata admission. The
21:47 snapshot verifies140 committed chunks/15,576 origins/46,728 gauge cases,
with42,696 recorded candidate complete cases and337 chunks still missing. The
expected worker is no longer observed. All5,266 storage attempts reconcile:
5,265 verified and one earlier failed request; none pending. Observed bodies are
13,032,041,830bytes and the conservative precharge estimate is$3.741287.

The original legacy writer and retained request log confirm compact404 records
without the later nested proof. A narrow reader extension and exact-code reuse
of already qualified basin entries are being prepared. This is an explicit
format stop, not an upstream weather gap or acquisition incompleteness. Original
v3 artifacts, failed-run receipts and budget charges remain intact.

## Qualified compatibility recovery and exact prefix parity — November 2

The compact404 branch is now tested against the original raw-v1 manifest/source
receipt and its unique settled GET404 request. Modern nested-proof checks and all
twelve component/numeric checks remain.23 reader regressions pass; the actual
20200422 source passes145 records/four recovered fields. The resident preflight
passes198 checks across193 unique cycles, with only the two known wrong-identity
variants rejected and33 absent-local-authority skips.

The exact v3→v4 basin-cache rule passed13 dedicated tests and independent review.
It returns original keys/receipts under the existing cache lock, records separate
per-use provenance and never creates a relabeled cache entry. Source fetches were
disabled for two prefix reconstructions. All140 typed model payloads then matched
v3 byte-for-byte:15,576 origins/46,728 cases,42,696 candidate complete cases and
2,596 unique original basin entries reverified. Network requests were zero;
local USGS reprocessing was permitted. Original v3 files remain intact.

The integrated suite passed482 tests. The v4 corpus plan is
`0dad073824984fd7eeba88e4c313d77b4015dcccb94e8eb27297fcf7c6b39b0f`;
increment07 resumes it under the same approved budget and source scope.

## Measured Q-only and L24 timing controls — November 2

The separate reviewed200-update benchmark passed8 B64 cases on the fixed earlier
device policy. It preserves the sealed558/168/18 August cohort and target IDs.
Q-only L168 uses known horizon tokens and no weather/soil predictors; full L24
uses the exact final24 history slots. All observed parameter counts and checkpoint
forward replays pass. Execution took32.18s and1.68GiB observed combined peak, with
40 warmup plus160 measured updates, zero retries/network/cloud jobs and no
scientific selection. The original600-update benchmark remains unchanged.

Cumulative accounting is now **852 real-data engineering updates** and
**896 synthetic updates**, with zero scientific campaign fits. Mode-specific
timing estimates will replace the corresponding sizing proxies in a new report;
the earlier proposal remains an attributable planning snapshot.

## Workload update using measured controls — November 2

The geometry-specific planner passes 21 tests and independent receipt/arithmetic
review. It binds both benchmark receipts, their common fixture/cohort/targets,
device policy and model geometry. The original proxy-only report is unchanged.
At 50 epochs, the calendar scenarios are 61.36 hours for 108 final fits and
89.34 hours for the 180-fit paired-CV plus final-L24 contrast. Both use measured
geometries throughout. The 288-fit factorial is 135.66 hours; its 72 Q-only/L24
fits still use an explicit full-L168 proxy. No CUDA speedup, final admission,
convergence, CV adoption or scientific result is inferred from these estimates.

## Typed trainer bundles with explicit role boundaries — November 2

The bounded adapter combines verified selections from one gauge/history/snapshot
into the trainer's typed mapping while retaining separate source provenance.
Caller-specified UTC windows govern both origins and all H18 targets. Missing
early observations cannot move a boundary; crossing/out-of-window rows remain
unassigned. Empty roles stay visible, and outer validation never substitutes for
an absent inner stopping set. Partial snapshots are refused by default.

Ten focused tests and independent review pass, including direct repeated use of
the adapter's concrete index tuples with the existing split validator. The v2
engineering qualification uses one copied immutable derived shard: 120 cases,
54 fit, 30 inner-stop, zero outer, and 36 retained unassigned rows. Full/Q-only
tensors, membership and role identities match. It took 0.149 seconds and peaked
at 48.2 MiB RSS, with no scaler fitting, optimizer, network or raw-source reads.
Both qualification receipts remain preserved. The optional simulated refit-label
availability convention is explicitly deferred, not inferred by this adapter.

## Bounded token refresh and next staging increment — November 2

Increment07 finished normally at 240 committed chunks and 26,760 origins.
Cumulative storage accounting is 8,030 attempted GETs: 8,029 verified and one
earlier failed request. Observed response bodies total 22,659,534,392 bytes;
the conservative precharge estimate is $6.431802, not an invoice.

The provider now permits one HTTP 401 token refresh within its existing
three-attempt limit. It clears only the process's cached token, precharges the
next attempt, then uses the unchanged credential command. Repeated 401 and all
403 responses stop. Failed charges remain, and sanitized HTTP status labels use
the existing database column. Fourteen mocked tests and independent review pass;
no new authentication, IAM or configuration action was performed by the tests.
The earlier failure's unrecorded HTTP status remains unknown.

The integrated suite passes 503 tests. Increment08 uses a new provider execution
plan while retaining the same v4 corpus plan, source identities and persistent
budget. No scientific fits or protected scores have been produced.

## Cache verification diagnostic and protocol packet — November 2

Increment08 stopped at 248 committed chunks with a `CacheCorruption` exception.
The last published entry, for `20210903_06Z`, then passed its stored checksums and
schema validation. Independent recomputation from the retained authenticated raw
archive exactly matched its means, masks, coverage and original source receipt.
Neither the cache nor raw source was changed or quarantined. The precise original
trigger was not retained and remains unconfirmed. Increment09 resumes unchanged
verification; this processing exception is separate from storage-request errors.

The status inspector now aggregates only recognized HTTP status labels, reports
legacy unqualified HTTP errors as `UNSPECIFIED`, and groups other failures without
returning arbitrary error text. Its 13 focused tests pass; the database remains
read-only. The earlier failed request remains charged.

`docs/protocol-decision-packet.md` provides a reviewed proposal for the remaining
settings and the shared 108-fit final-development core. The 108/180/288 scope
question remains open. The packet does not freeze choices, enable fitting or
assessment, launch a cloud job, or send a coauthor message.

## Full admission audit preparation and connection interruption — November 2

The admission-audit script passes 10 focused tests and independent review. Its
default requires every planned shard. It will report per-gauge/split natural and
common cohorts, masks/exclusions, weather fallback and soil ages, and broad
physical-domain diagnostics without changing admission. It emits no discharge
magnitude summaries and fits no statistics or models. The explicitly partial
one-shard check covers 360 candidates and 300 complete cases; it is not a full
corpus qualification. Full execution waits for all 477 shards.

Increment09 reached 341 committed chunks/37,872 origins, then stopped after three
bounded connection failures for the same pinned `20221106_18Z` archive. The four
cumulative failed GETs remain charged. A subsequent DNS/TCP/TLS probe passed with
certificate verification intact and no source GET or credential request.
Increment10 resumes under the same execution plan and budget. At the stop,
10,195 attempts reconciled, observed bodies were 32,587,096,060 bytes, and the
conservative precharge estimate was $9.165681, not an invoice.

## Development preparation completed — November 2, 23:21 UTC

All 477 shards completed: 53,028 retained origins after 36 H18 boundary purges,
159,084 gauge cases, and 146,464 candidate complete L168 cases. The independent
full audit passed in 18.04 seconds with 66.9 MiB peak RSS and zero physical-domain
flags. Natural L24 support is 153,216 cases; its 6,752 additional cases remain
separate from the matched L168 comparison cohort. No source masks were changed.

L168 training/validation counts are 42,715/8,766 at Jefferson, 38,297/8,125 at
Galax, and 41,332/7,229 at Sugar Grove. Common L24/L168 membership hashes match.
These are candidate admission results, not forecast-skill findings or a protocol
freeze. All 73 USGS development months are covered through native files or the
verified July export; protected assessment modeling/scoring remains gated.

The exact metadata inventory resolves all 8,841 development archives: 8,770
complete cycles and 71 documented gaps, totaling 46,942,832,640 bytes (43.72 GiB).
It preserves 2,600 v3 and 6,241 current-reader cache entries. The full private
catalog is hash-bound; original source/plan receipts remain unchanged.

Cumulative GCS accounting is 13,071 attempts, 13,067 verified and four retained
failed charges. Observed bodies total 46,218,766,666 bytes; the conservative
estimate is $12.907585 within the $35 allocation. No invoice reconciliation or
cloud compute execution is claimed. The final integrated suite passes 541 tests
in 5.074 seconds, including the independently reviewed final inventory and
actual-count planning components.

## Confirmed 180-fit scope and exact candidate workload

The user confirmed 180 fits: 72 paired L168 CV fits plus 108 final-development
fits, including the common-case full-input L24 contrast. This confirms scope,
not the remaining methodological settings or assessment permission.

Exact per-gauge fold counts now replace calendar counts in a separate addendum.
The confirmed scope sizes to 33.09 local hours at 20 epochs or 82.45 hours at
50 epochs, before the stated loading/scaling, uncertainty/figures, failures and
travel overhead. Both proposed refit-label conventions produce the same rounded
B64 batch counts, while the two-hour convention excludes one additional inner
stopping case per gauge/fold. Neither convention is silently adopted. There are
still zero scientific fits, zero protected scores and zero cloud compute jobs.

## Conditional training core: pre-freeze implementation checks — November 2

The user conditionally accepted the recommended core, explicitly requiring
spatial, temporal, scoring/inference and coverage/uncertainty checks before
protocol freeze or scientific comparison fits. The 180-fit scope remains
confirmed; protected assessment and the 2023–24 development-validation role
are unchanged. No scientific or cloud comparison fit was started.

The new refit guard tests actual last-target timestamps plus the explicit
simulated delay against refit time, separately from strict role boundaries.
It affects fit/inner-stop admission only and retains excluded rows unassigned.
The existing strict-only default is preserved. Tests also establish that
stopping/outer/unassigned data cannot change gradient-fit scaler parameters.

Best-checkpoint inference is label-free and uses the saved train-only transform.
An independently reviewed, once-only eight-update synthetic qualification passed
four fresh-process CPU/MPS full/Q paths in 17.315 seconds. Physical predictions,
saved transforms and selected best states matched exactly; one fixture selected
epoch 1 while last state was epoch 2. Inference constructed no optimizer and read
no labels. Cumulative counts are 852 real-data engineering updates and 904
synthetic updates; scientific comparison fits remain zero.

The scoring layer has 15 synthetic tests and independent review. It defines
equal-lead/equal-fold NSE, retains all final seeds, propagates missing/undefined
primary support, reports physical errors and negative frequency, and loads best
weights before any label access. Every source row's origin/H18 clocks are gated,
including unselected rows. Its 168-hour/28-UTC-date primary support rule is a
proposed reporting floor, not an independence or power guarantee. Checkpoint
selection uses inner standardized-Q MSE; reporting NSE and physical MSE remain
distinct. No scoring policy is silently adopted by this implementation.

## Spatial source validation and coverage tradeoffs — November 2

The separately approved 334,177-byte Galax source materialization hit its initial
60-second child timeout. The original subsequently became resident; an eight-second
bounded local completion verified its exact retained SHA and copied it without
another explicit hydration request. No new USGS geometry request or source-content
change occurred. FileProvider wire bytes remain unknown. The timeout, partial v2
audit and completed v3 audit are preserved separately.

All three official-source footprints and 514 positive-cell weights now reproduce
independently, with zero off-crop area and maximum absolute weight error below
3.4e-14. The accepted Galax 33.73 m outlet offset and Sugar Grove −1.03254% area
discrepancy remain unchanged. Jefferson's 5.179115 km²/0.975086% area outside
Galax is reproduced; no forced nesting or geometry repair occurred. Missing any
positive-weight cell masks the production basin mean; no renormalization occurs.
Original candidate hold/admission flags are preserved for provenance.

The coverage audit binds the complete corpus and monthly references, gates the
July export before value reads, and changes no source masks. Thirteen synthetic
tests and independent review pass. Both executed versions remain. Version 2 adds
the natural-L24/common-L168 loss by flow regime; thresholds use unique approved
gradient-calendar reference hours, not stopping/outer values or repeated cells.

In 2023–24 development validation, common L168 support retains 3,024/6,048 high-flow
target cells at Sugar Grove (50.0%) versus 5,851/6,048 under natural L24 (96.7%).
Galax retains 2,404/3,474 (69.2%) versus 2,837/3,474 (81.7%). These are repeated
origin/lead cells, not independent events. Unknown reference regimes remain
unknown; missing/rejected intervals do not establish instrument damage. The
observed-peak catalog uses no model errors and retains unsupported peaks.

Seasonal and high-flow availability SVGs were generated from the hash-bound v2
report with the existing plotting runtime and visually checked. The proposed
14-day seasonal block approach, 7/28-day sensitivities, distinct-support rules
and within-year limits are documented; 13 synthetic resampling/weighted-metric
tests pass with independent review. No study resampling or intervals were produced.

The final main suite reports 614 tests: 611 passed and 3 GIS-only tests skipped;
all 9 spatial tests separately pass in the existing GIS runtime. The numerical
and implementation checks are ready, but complete-case high-flow representativeness,
conditional uncertainty interpretation and the approximate/non-nested spatial
claim remain material items for user review before freeze. No scientific comparison
fits, protected scores or paid cloud jobs have started. The 180-fit scope and $150 ceiling are unchanged.


## Coverage sensitivity census and proposal — November 2

The user requested a development-only coverage census and missingness-aware
sensitivity design before freezing the protocol. The 90% coverage aspiration is
not an assumed result. The strict matched L168 primary, 180-fit scope, original
quality flags and target masks remain unchanged. No extra fits were authorized.

A pure mask/clock classifier now tests interior, trailing and combined one/two-hour
candidates with both run-length and total-missing caps. Exact C−169…C−2 slots and
approved endpoint clocks are required; availability remains the simulated two-hour
convention, not historical revision-delivery evidence. Thirteen tests and peer
review pass. It reads no Q values and performs no repairs.

The full census preserves 159,084 candidate cases and reconciles all prior primary
counts. Twelve tests and peer review pass. It separately reports forecast-input
support, target-evaluable fractions over all origins and over valid references,
known-high-flow support, unique hours, gap types/positions and fixed observed
peak episodes. Weather/soil/calendar gates remain separate. The initial output
attempt exceeded its public-size guard; its log/private details remain, and the
successful packaging splits event details into three hash-bound public siblings.
The completed census ran in 49.94 seconds at about 193 MiB peak RSS.

In final development, minimum-lead known-high-flow support is 90.16% at Galax
and 54.46% at Sugar Grove under the combined two-hour candidate, versus 87.56%
and 95.24% under natural L24. The prior 69.2%/50.0% figures retain their original
complete-H18 pooled denominator. No denominator was substituted silently.

A separately tested routing diagnostic independently reconstructs and exactly
matches the base/short/L24 counts, then unions short-gap support with strict L24.
Three tests and peer review pass. At the worst lead, target-evaluable/all-origin
support is 100%/98.55%/95.87% for Jefferson/Galax/Sugar Grove; all cataloged peaks
have structural support at every lead (9/6/11 defined episodes). These are not
independent flood counts, forecasts, error measurements or operational guarantees.
The route ran in 5.67 seconds at about 129 MiB RSS. Original targets remain missing
where unobserved or rejected. The static source-bound coverage figure was rendered
and visually checked; its first layout draft is retained privately.

The proposal adds a six-neural/three-ridge CV1 pilot, with clean-H4 and augmented-H4
controls, issuance-safe fillers, paired artificial gaps and clean/false-alarm/event
checks. Optional final confirmation would add eighteen neural and three ridge
fits. Both stages would add 24 neural plus 6 ancillary fits, all unapproved.
Measured H2 proxies including extra inference are 2.62/6.24 hours at20/50epochs;
H4, filler, ancillary, scoring and bootstrap costs still require qualification.
An existing-final-L24 route adds no neural fit beyond the primary matrix, but its
inference and differing information/support require separate interpretation.

The integrated suite ran 642 tests: 639 passed and three GIS-only tests skipped.
The previous separately qualified spatial implementation is unchanged. No fills,
model fits, optimizer updates, new data downloads, paid jobs, or protected
assessment access/scoring occurred. Cumulative executed engineering update counts
remain 852 real-data and 904 synthetic. Primary protocol freeze and sensitivity
expansion await review of the concrete census/method/resource packet.


## Pilot implementation and synthetic qualification — November2

The user approved pilot implementation and qualification ONLY. The separate
scientific six-neural/three-ridge pilot remains held for protocol and tolerance
review. The primary180 fits, original H2 code/data, protected gate and150USD ceiling
are unchanged. New code implements causal interior/carry/ridge input handling,
source/qualifier-preserving H4 masks/age, seventeen deterministic artificial gap
scenarios, clean/augmented training mechanics and label-free best inference.

Filler13, adapter14, pilot-trainer13 zero-update and qualification-controller7
focused tests passed with independent reviews. The integrated suite ran689 tests
in16.807s:686 passed,3 unchangedGIS skips. No optimizer work occurs in those
ordinary tests; NumPy synthetic ridge fixtures are separate ancillary mechanics.

The once-only qualification precharged and confirmed exactly16 synthetic optimizer
updates: each arm's4-update uninterrupted run versus a2-update prefix and2-update
fresh-process continuation. All sample order, masks, loss/model/optimizer state,
saved transform and best-state predictions match exactly. Both selected epoch1
while last weights were epoch2. The H4 LSTM has68161parameters. The bounded run
created one synthetic ridge artifact; later phases loaded it without fitting.
All17scenarios were exercised perarm:16 had64-case outputs; mixedrecent/ridge
had56admissible/8unsupportedcases and the requestedbatch was correctly refused,
not silently filtered or repaired with another method. The run completed34.31s
at401MiB combinedRSS. Its consumed allowance, checkpoints and source identities
remain intact; cumulative updates are852real-data engineering and920synthetic.

A separately reviewed forward-only profile used the same qualified synthetic
checkpoint and completed20calls each acrossfive stages, with100timedcalls and
no optimizer/backward/scaler/ridgefits. It completed16.24s at287MiBRSS and verified
the qualification ledger unchanged. Model-onlyforward averaged14.88ms versus
335.92ms for fullcleanprediction; separately timed provenance normalization/digest
averaged211.40ms. This is an implementation overhead finding, not a GPU or
architecture-performance comparison. Original H2 timing measured pre-mapped
kernels; current H4update times also include preparation and qualification
status writes. First-use and three-later-sample estimates are kept distinct.

New immutable runtime scenarios using later-three mean updates estimate4.22/8.64h
for the six-neural pilot at20/50epochs, with the18-neural optionalconfirmation at
25.28/57.81h. Ancillaryfits/fullscaleprovenance/scoring and other work remain
unmeasured; these are notconfidencebounds. Earlier2.62/6.24h combined-H2 proxies
remain historical. Reduce repeated immutable metadata processing and qualify
full-cohort construction before scientificrelease; no optimization or new update
allowance has been silently executed. No study-data fit, new download, paid job,
protected-access/score or securitychange occurred during this work.


## Resident execution, provenance optimization and v2 qualification — November 3

The user authorized local Git/cloud-only-file recovery, provenance optimization
and requalification, while keeping scientific fits and paid cloud jobs held.
The active source checkout was recovered at published b35e0f7 outside Documents.
The original Git metadata and unheld stale index lock were preserved; standard
linked-worktree repair synchronized its frozen HEAD/index to b35e0f7 without
rewriting working-source or research files. Active Git commands now operate on
the resident checkout and push normally. The original cloud workspace remains
an evidence/data archive; the private handoff records explicit execution paths.

A fully isolated Python3.11 runtime was installed offline from all29 exact
previously-qualified wheels (142,010,840 bytes); critical imports and all locked
versions verified outside cloud storage. Resident wheels were reused and missing
runtime bytes restored only from official package sources matching their old
SHA/size. No model weights or scientific sources were downloaded by that step.
All21 required v1 qualification artifacts and the consumed16-update marker were
copied and verified; the original plan derivation was labeled. Other consumed
markers were preserved, with an explicit fail-closed blocker for one unreadable
legacy marker rather than a new allowance.

Provenance optimization preserves exact canonical bytes and hashes, then reuses
an owned immutable source-metadata snapshot. It interns immutable primitives and
stores original clocks/masks in byte-backed arrays; no Q values, targets, fills
or model outputs are cached. Training uses ordered snapshot views. Inference
reuses one context only after actual metadata, clock, mask and identity checks.
All dynamic quality, timing, value, endpoint and fill-failure gates remain.
23 focused parity tests and14 adapter tests passed independent review.

The integrated pre-execution suite ran741 tests in14.620s:738 passed and3 GIS
skips. The separately precharged v2 run confirmed16 synthetic optimizer updates
in21.84s, with zero new ridge fits. Both baselines match v1 exactly on traces,
losses, masks, weights, optimizer/RNG/sampler state, transforms, best weights and
physical predictions. Fresh-process prefix/resume and best inference also pass;
old checkpoints were authenticated comparison references, never restored under
changed code fingerprints. Cumulative optimizer counts are852 real-data
engineering and936 synthetic; scientific fit counts remain zero.

The zero-update v2 profile completed120 timed calls across six paths. Mean full
prediction is144.09ms for one miss plus19 repeated-batch hits, and175.06ms for
20 actual new-origin misses; these are not disk-I/O rates. The v1 repeated-batch
mean was335.92ms. Provenance digest mean fell211.40→62.72ms; model-only forward
is essentially unchanged at14.82ms. Residency, workload and cache limits remain
explicit; separate stage times are not a disjoint causal trace.

The17869-row fit+inner-only synthetic constructor passed with one synthetic
scaler fit, zero neural updates/forwards and1.832GiB observed combined RSS.
Construction took9.01s after fixture generation. Outer/unassigned rows and the
larger optional final constructor were not qualified. Later-mean cold-default
six-fit proxies are2.03/4.06h at20/50epochs, with unmeasured data assembly, ridge,
scoring/resampling and journal costs. Warm batch-major scheduling remains a
separate conditional scenario. The primary180 fits and150USD ceiling are unchanged.

Development residency recovery binds the exact existing plan0dad.../snapshot
b416... and1432 immutable files/75,805,012 bytes, with no raw-source acquisition
or assessment files. The first900s-window sweep preserved five timeout failures;
three subsequently resident files were copied under a separately recorded small
supplement and adopted by exact target hashes, keeping all failures/charges.
Currently1430 files are verified outside cloud storage and two (206,468 bytes)
remain cloud-only. The first deadline expired and was not reset; no numeric
integrity pass has started. A separate resident-only finalization plan is saved.
Finder access is blocked by the locked Mac; the user was asked to unlock while
independent work continued. Scientific fits, assessment scoring and paid jobs
remain held; no false complete-corpus claim is made.


Final integrated validation after adding the prepared finalizer ran751 tests in
11.863s:748 passed and3 unchangedGIS tests skipped. The finalizer's10 focused
synthetic tests and independent review passed. It is prepared, not executed:
2MiB freshsource/1GiB cumulative source/4GiB RSS/120s, actualbothresident checks
beforeclaims, read-only oldhistory and one new integrity authority bound to the
unused priorpass. No oldclock orcharge is reset. Exact remaining-file availability
still requires the manual residency step; the complete corpus is not claimed.

## Development residency finalized — November 3, 16:25 UTC

The user unlocked the Mac and explicitly requested the two remaining owned
downloads, final integrity check and checkpoint publication, while retaining the
scientific-fit and paid-job holds. Finder did not complete a download action;
the two exact files were requested once each through Apple's documented
FileManager iCloud API. The metadata monitor confirmed both resident before the
unchanged finalizer claimed its separate allowance. Nominal file content was
206,468 bytes; network wire traffic was not measured.

The two copies passed their original byte counts and SHA checks. The one local
development integrity pass passed at 16:25:07 UTC in a 5.51-second finalization,
with 102,236,160 bytes peak observed parent-plus-child RSS. All 1,432 files,
75,805,012 bytes and 477 shards are resident outside cloud sync, matching the
existing snapshot b416... and source plan 0dad.... There are no missing chunks.
The integrity receipt retains 53,028 origins, 159,084 candidate gauge cases and
146,464 strict L168 complete cases; these are engineering counts, not forecast
skill or an approved final scientific cohort.

All prior failure receipts, source charges and expired deadlines remain intact.
The original ledger's pending integrity state is historical; the separate final
allowance records PASSED and is consumed. Cumulative source-read reservations
are 76,424,532 bytes. The completion is an exact snapshot restoration, not a new
acquisition, preprocessing run or upstream-source re-audit.

The 25 synthetic recovery/finalizer tests pass (0.273 seconds). No neural updates,
new ridge/scaler fits, assessment access or paid cloud jobs were performed. The
prior 751-test integrated result remains historical and was not rerun. Public
receipts are `development-corpus-materialization-v1.json`,
`development-corpus-residency-finalized-v1.json`,
`development-corpus-residency-integrity-v1.json` and
`resident-optimization-release-v3.json`. The partial v2 release remains unchanged.
Real per-slot provenance qualification and pilot protocol/tolerance review still
precede any separate scientific execution release.

## CV1 original provenance cache recovered — November 3

The user authorized real-data assembly and a bounded pilot protocol, with
tolerances and scientific execution to be reviewed afterward. Metadata inventory
found 19 owned cloud-only cache files (66,039 encoded bytes), separately from the
already complete numerical corpus. After explicit approval, 19 scoped Apple
iCloud requests completed. All 37 monthly entries for July 2018–July 2021 were
copied outside cloud sync: 111 files / 370,776 bytes. Original commitments were
pinned by the published coverage report and matched corpus monthly references;
all payload byte counts and SHA checks passed. Five previously unreadable
commitments were authenticated before their payload identities were trusted.

`reports/pilot-cv1-provenance-cache-recovery-v1.json` preserves the exact copied
file identities. This step copied opaque payload bytes without decoding numeric
arrays, reconstructing targets, fitting a model or acquiring new USGS records.
Original source IDs identify shared source requests; assembly must join by gauge
and exact timestamp, retaining those IDs and qualifiers verbatim. Historical
publication availability remains a simulated two-hour convention.

## CV1 real-data assembly completed — November 3, 16:59 UTC

`reports/gap-pilot-real-assembly-v1.json` records one bounded local assembly under
controller plan `6ba6aeb0…`: 57.62 seconds, 135,217,152 bytes combined peak RSS,
44,407,134 source bytes read and 57,750,943 output bytes. All source joins and
published strict CV1 role case hashes passed. The immutable output manifest is
`4795efd085a4924df9b4b6b5b96933cb9c70590ebc71ec6e46484c63a259c540`.

There are 928 committed input/label parts, separate compact source provenance,
strict paired cohorts, 8,742 all-candidate outer origins per gauge and three
gradient-only ridge series. The ridge eligibility counts are 15,809 / 15,370 /
15,604 for Jefferson / Galax / Sugar Grove; these are prepared pairs, not fits.
Original gradient mask counts support a separate review of augmentation recipes.
The source corpus, prior recovery/integrity receipts and qualified model code
remain unchanged. Twenty-eight focused synthetic tests passed in 2.749 seconds,
and independent review passed. No model/scaler fitting, neural update, forecast
scoring, protected access or paid job occurred. Prepared-loader qualification and
the bounded pilot protocol/tolerance review remain before scientific release.

## Real loaders qualified; bounded pilot proposal prepared — November 3

`reports/gap-pilot-real-loader-qualification-v1.json` records the once-only
zero-fit loader check under plan `d20f765c…`. It passed in 35.51 seconds with
612,335,616 bytes peak parent-plus-child RSS and 107,710,407 logical bytes reserved
at file open. All three complete fit+inner bundles matched the exact role case
digests, with zero outer/unassigned training rows. Original provenance and raw
array identities are recorded. Six first-returned outer/coverage batches were
checked with label opens prohibited during input loading; the subsequent label
checks were explicit and retrospective. Three ridge source stores passed their
gradient-only pair checks. Outer iterators were sampled, not exhausted.

Seven new synthetic loader-controller tests passed (0.520 seconds), bringing the
unique focused assembly/loader tests to 35. Model/Torch/scaler/ridge/optimizer,
forecast scoring, network and protected-access counters remained zero. No
consumed model-qualification or acquisition allowances were rerun.

`docs/bounded-gap-pilot-protocol.md` and
`configs/bounded-gap-pilot-proposal.json` are proposed for review, not frozen or
authorized. They bind the actual assembly/loader identities and propose six H4
neural fits, three ridge fits, 67,900 maximum entered updates and 126 inference
requests, under eight hours and four-GiB memory/output caps. The current
conservative storage worksheet estimates 3,847,987,200 bytes conditional on
enforced cadence and serialization limits; it is not a measured full-run bound.

The gradient-only gap census supports conditional empirical shape mixtures for
Galax and Sugar Grove. Jefferson has no observed short gaps in this period, so
its explicit balanced-stress recipe needs review. All use proposed 0.5 artificial
corruption, not an estimated outage rate. The practical clean/failure tolerances,
event/uncertainty interpretation and resources remain user choices. A new
natural-gap inference wrapper must keep the checkpoint's training policy fixed
while binding the separate requested fill policy; the bounded runner's journal,
checkpoint cadence and byte/update controls also need qualification. Conditional
pilot approval does not automatically release the primary campaign or assessment.

## Higher-epoch preference recorded — November 3

The user prefers a high epoch limit with early stopping. The new
`docs/high-epoch-stopping-proposal.md` proposes 200 epochs and records a
100-epoch alternative, while preserving minimum10, patience5, strict inner
standardized-Q MSE improvement and earliest-best evaluation. The numerical
ceiling remains a proposal; qualified training code and prior limits were not
changed. `reports/high-epoch-stopping-proposal-v1.json` records exact scaling of
the admitted 180-fit workload: 164.72h at100 or329.26h at200 if every fit reaches
its cap, with all prior runtime exclusions retained. At200,116 fit instances
would exceed the current100,000-update guard; the largest needs126,800 updates.
Per-fit/global update, time and storage allowances must be reviewed together.

Three pure stopping-state cases and the existing oversized-update rejection
passed with zero model/scaler/optimizer activity or source-value reads. Earlier
50-epoch receipts remain historical evidence. This direction neither releases
scientific fits nor changes the protected-assessment or paid-cloud holds.

The intervening discussion also accepted ridge removal and retained all four
deep-learning families. The user asked whether a separate missingness pilot is
needed before the primary campaign; that sequence remains unresolved. The older
six-neural/three-ridge approval question and packet must not be treated as current
execution authority. No24-fit pilot or main campaign has been released by this
discussion, and no new fit was started.

## GCP compute exploration — November 3, 19:32 UTC

The user requested exploration of more powerful GCP compute. Five read-only
metadata queries succeeded; a sixth modern quota read timed out without retry.
Current global GPU quota and us-central1 L4 quota remain1withusage0; A10040/80GB
quota there is0. G4/H100 quota remains unverified. All22 listed Compute Engine
instances are terminated; catalog listings are not capacity reservations.

`docs/gcp-compute-options.md` and
`reports/gcp-compute-options-20261103.json` compare current official whole-VM
prices, hardware restrictions, live quota evidence and the existing150USD
funding conditions. Recommendation is to prepare the CUDA integration and a
bounded one-L4 benchmark first, then compare measured cost per completed fit or
independent-fit concurrency before choosing larger hardware. CUDA is not yet
enabled/qualified in the scientific trainer. No source values, model runs,
paid jobs, resources, reservations, quotas, permissions or data objects were
created/modified/transferred during this exploration. No claimed cloud speedup,
remaining cash balance or deadline guarantee is inferred from the catalog.

## Local execution ordered by seed — November 3

The user requested staying local and completing each seed's training and
development evaluation before advancing, so earlier complete results can be
reviewed while later seeds run. CUDA benefit remains unmeasured; it was not found
ineffective. `scripts/plan_seed_sequence.py` and
`reports/local-seed-sequence-v1.json` prepare exactly180unique metadata jobs:
72CV17 +36final17, then36final29, then36final43. All four families, three gauges,
paired full/Q L168 arms and matched final L24 contrasts remain unchanged.

Seven metadata-only tests passed. The first complete seed milestone contains108
fits, not60; later seeds do not repeat CV. Its historical50-epoch modeled duration
is44.78h, followed by18.83h each, before stated additional work. Seed review does
not tune later seeds or drop poor models; a methodological change requires a new
version and consistent evidence. Protected assessment is outside these reports.
No unrequested human pause is added between unchanged released seed stages.

This chunk specifies ordering and evidence barriers; the durable execution
controller and large main-study bundle integration remain to be completed. The
specific high epoch ceiling/resource ledger and remaining protocol interpretations
must be bound before fitting. No model, scientific fit/evaluation, source-value
read, cloud action or optimizer update occurred in this scheduling chunk.

## Device comparison and latest fastest-backend direction — November 3

The user initially requested MPS for all experiments, then directed “Ok go with
what’s fastest!” after the measured differences were explained. The active
`reports/local-seed-sequence-fastest-v3.json` therefore retains CPU for LSTM/
GRU–Transformer and MPS for Transformer/Mamba-2–Transformer. All180 job identities,
cohorts, roles, seeds, B64/FP32 settings and108/36/36 milestones stay unchanged.
Every selected backend/feature geometry has a corresponding engineering timing
cell. No general optimum across untested shapes or batch sizes is claimed.

The separately generated all-MPS v2 plan and timing review remain comparison
evidence. They have126 matched and54 unmeasured jobs; CPU controls were never
relabeled MPS. A complete matched all-MPS campaign estimate stays null, and its
supplemental full-L168 proxy is explicitly not an upper bound or measured total.
The active mixed-backend timing scenarios retain their original exclusions.

Nine all-MPS planner tests and five active-policy tests passed. Independent review
passed. No model call, new optimizer update, scientific score, source-value read,
cloud action or consumed-qualification replay occurred. Current trainer B64
restart/best-state integration and larger main bundles remain qualification work;
the durable execution controller and final epoch/resource plan remain pending.

## Primary launch implementation and qualification — November 3

The user explicitly requested starting training. A final method-confirmation
question remains pending because the earlier instruction required review of
material tradeoffs before freeze: proposed200 epochs/minimum10/patience5, strict
matched L168 without repair, approximate independent basins, conditional
within-year uncertainty, raw unclipped output and sealed assessment. The new
protocol draft records these choices without pretending a reply was received.

The primary trainer now accepts a finite200,000-update hard bound; each actual
fit remains restricted by its exact logical allowance. Restore reuses its
authenticated gradient-only transform without refitting. Checkpoint publication
supports exact byte precharge and epoch callbacks reuse the existing stopping
loop. New synthetic qualification covered all12 primary architecture/input/
history cells atB64 on the selected CPU/MPS backends: exactly36 entered/confirmed
updates and12 scaler fits,94.88seconds,1,872,297,984bytes observed peak combined
RSS+MPS. Every case passed fresh-process next-update state/RNG equivalence and
best-state inference distinct from last weights. Cumulative engineering counts
are now852 real-data updates and972 synthetic updates; scientific fits remain0.

The new campaign data loader reuses the pinned full corpus integrity receipt,
selects only exact roster memberships and separates outer inputs from labels.
All27 unique real-data views passed in152.28seconds with598,081,536bytes peak RSS
and3,833,862,913 logical file bytes reserved at open. The42,696-row largest
fit+inner bundle is included. No model/scaler fit, protected access, raw source
cache read, cloud transfer or full-corpus verification replay occurred.

The durable SQLite ledger, one-job worker, supervising process and seed reporter
are implemented. Update and output charges persist across interruptions; the
latest checkpoint alone supports up to3 explicit resumes. Seed barriers require
authenticated training plus matching predictions/metrics for every job, followed
by committed JSON/HTML reports at108/144/180 completions. Later seeds may continue
unchanged while the user reviews earlier results. Event/uncertainty figures are
explicitly pending postprocessing, not falsely included in initial report claims.

The integrated suite ran911 tests in104.532seconds:908 passed,3 unchanged optional
GIS skips. The first910-test attempt had two no-Torch loader tests sharing a
process with unrelated Torch-importing tests; fresh-process test isolation fixed
that without changing production qualification guards. The failed log remains
preserved. Independent reviews passed for the loader, ledger, worker, supervisor
and seed-report publisher. `reports/primary-campaign-controller-check-v1.json`
binds the exact technical versions and both executed qualification receipts.

Current free local space exceeded20GiB. Proposed run guards are6h/job,336h total,
4GiB host+MPS (2GiB MPS),8GiB new output including2GiB bookkeeping, and8GiB free
space floor. At200 epochs, logical updates cap18,200,800; physical replay reserve
caps20,930,920. No quota is refunded, deadline reset or old qualification rerun.
No scientific release marker, campaign ledger or study fit has been created
while human method confirmation is pending.

## Primary 100-epoch protocol adoption — November 3

The user answered the pending method question with “Approved 100epoch.” The
primary protocol is now frozen for the local 180-fit roster with minimum10,
patience5 and the other reviewed methodological choices retained. The approved
ceiling is100, superseding the earlier200 proposal. Logical updates cap9,100,400;
physical entries including the unchanged30-epoch replay reserve cap11,830,520.
Largest per-job limits are63,400 logical /82,420 physical. Existing time, memory,
output, checkpoint and recovery guards are unchanged. Qualification receipts and
execution code remain unchanged; no consumed qualification is replayed.

This adoption authorizes local scientific execution, not paid cloud, new downloads,
protected assessment, ridge or gap-repair fits. The next launch records the exact
protocol/plan hashes and verifies actual optimizer updates in the durable ledger.

The adoption passed all14 focused controller tests and a metadata-only preflight
that authenticated current model/data/controller qualification receipts, all180
jobs and the exact100-epoch arithmetic. The publication scan checked379 files
with0 findings. No model call or qualification allowance was consumed by these
checks; the execution implementation is unchanged.

## Primary local training launch — 2026-11-03T23:02:52.608768+00:00

The100-epoch protocol was committed and pushed as09df7f4 before launch. The
private plan is `7723e33746e51badd5ca6cc2d8ac61283fdd0c152306f2d754f047862ced60ff` and binds protocol
`3d94d182e1a22a84f3cd55b8824670821492e86a2ec015a465b155dcb91015f4`. The supervisor launched at
2026-11-03T23:01:50.792156+00:00 with scoped idle/system-sleep prevention.
At the recorded check, CV1 Jefferson LSTM/fullL168/seed17 on CPU had completed
4 epochs; the durable ledger confirmed1198 scientific optimizer updates
(1199 entered). The initial checkpoint bytes and receipt were authenticated.
Completed scientific fits at this check: 0. These are startup observations,
not a forecast-skill conclusion. One worker runs at a time; later frozen seeds
continue after verified development reports. No paid cloud job, new download,
protected assessment access or consumed qualification replay occurred.

## Heartbeat conflict repair and explicit recovery — November 4

The supervisor stopped after30 completed fits because an atomically refreshed
heartbeat was read using immutable-corpus pathname checks. The failure was in
monitoring, not a source-shard read. The user explicitly requested resolving the
conflict and getting training back on track. The epoch10 checkpoint and30
completed receipts were authenticated;772 post-checkpoint updates will replay
with their original charges retained.

An additive heartbeat snapshot reader and separately authenticated recovery
wrapper preserve all frozen execution files, the protocol, original privateplan
and ledger. The wrapper exposes only same-job resume and keeps immutable reads,
worker commands, identity/resource checks and original deadlines.47 focused
tests passed with0failures/errors, including deterministic race reproduction and
original-supervisor malformed/foreign-heartbeat stops. Independent review passed.
No scientific update or old qualification replay was performed by these checks.
The recovery uses a separately hashed operational amendment and launch receipt;
preparation is not a claim that resumed training has started.

The qualified repair was pushed as5e5df2e. The explicit recovery launched at
2026-11-04T07:34:37.127003+00:00 (01:34CentralNov4), using operational amendment
`2ec9d376cb0d6db6ae60728760459df880ffc246043f2dce7e3890b527a01228`. At2026-11-04T07:38:25.955977+00:00,
the resumed MPS attempt32 had passed the original interrupted logical prefix of
4,572updates. Replayed epochs11/12 matched the original epoch records and stopping
state exactly. All30completed-receipt digests, original protocol/code/plan identities,
jobdeadline and prior charges were preserved. The public versioned recovery
launch receipt records the observed counters; no old fit was restarted.

## All 180 training and development fits verified — November 5

The campaign completed at 7:09 a.m. Central on November 5: 72 CV fits and 108
final-development fits, all four architectures and seeds 17/29/43. All 180 jobs
are VERIFIED_TRAIN_EVAL and all three seed reports are committed locally.
Physical entered/confirmed updates agree at 1,151,690; logical completed updates
total 1,150,918, with the 772 replay updates retained. All original job/campaign
limits were respected; no new failure followed the heartbeat repair. No active
training worker remains.

The final report regenerated byte-for-byte from 919 authenticated artifact files
(507,221,045 logical bytes), and all three report commit maps matched the read-only
ledger. Independent arithmetic reproduced 180 primary panel means, 24 CV means,
36 final-seed means/sample SDs and 648 per-lead aggregate rows. No new model call,
source-value read, optimizer update, ledger mutation, download, paid cloud job or
protected access occurred. The completion receipt and development-review note
preserve exact source hashes and distinguish descriptive findings from pending
event/uncertainty work.

The component-level PROPOSED_NOT_FROZEN scoring label was reconciled with the
external frozen protocol; numerical rules agree and immutable receipts remain
unchanged. Full inputs exceed Q-only primary means in all 12 three-seed gauge/model
pairs, but H1 is lower in 11/12 and H18 higher in 12/12. This is preliminary
development evidence, not an isolated-input effect, significance result or
protected assessment.

## Postprocessing implementation qualified — November 5

The user requested continuing after all180fits completed. New, separate modules
authenticate saved prediction values and original labels, independently recompute
all physical metrics, preserve full origin calendars, retain the fixed event
catalog, and implement the frozen paired seasonal-block analysis. The operational
plan precharges1GiB inside the existing campaign output allowance, with persistent
6h wall,3GiB memory,8GiB logicalread and8GiB free-space bounds. No training method,
checkpoint, original artifact or protected-assessment gate is changed.

The exact fixed stream split and descriptive residual lag grid are documented
before real postprocessing. All93focused tests passed and independent reviews
found no blocking issue. A realistic synthetic2000draw×36panel bootstrap used
under1second for scoring per block length and peaked at563MBprocessRSS. These are
synthetic qualification timings, not study findings. No model calls or optimizer
updates were used. The subsequent real run must still pass all actual baseline
and fixed scalar-reference checks before releasing intervals.

The user separately approved33owned cloud-only reference-cache files totaling
105,064bytes. All147required files (467,088bytes) were copied/verified resident
with their exact checksums. The July2024development-export boundary is retained.
No new USGS request, paid cloud job or protected access was made.

## Development postprocessing executed and reviewed — November 5

The qualified code was pushed as 599b358 before execution. Postprocessing plan
7aee14c8dc9f332bd16e44ae2297001b71c4f9d6e5e7bfa33b3e8f2c82ccc5f9 ran
inside the existing resource allowance. All 180 saved raw prediction panels passed
independent physical-metric/count/time checks; maximum absolute metric difference
was 4.55e-13. All 12 fold/gauge reference/event groups matched original targets and
fixed observed peaks. No threshold, source value or model selection was changed.

All 7/14/28-day bootstrap analyses completed, with 2,000/2,000 jointly valid draws
for each 30-CV/42-final contrast family. All 540 baseline checks and 12 fixed scalar
weighted-reference checks passed. Independent sorted-order interpolation then
reproduced 216 contrast intervals and 3,240 per-lead intervals exactly. Seven PNG/SVG
figure pairs were rendered and visually reviewed. Raw arrays/checkpoints remain
private; public artifacts contain aggregate results and scientific figures.

The scientific review retains material limitations: only 5/12 final joint-input
intervals exclude zero at all three lengths; high-flow cell coverage is 100%,
69.2% and 50.0% for Jefferson, Galax and Sugar Grove; major observed peaks lack
forecasts; Galax residual dependence persists at multi-week lags. The adequacy of
the seasonal bootstrap interpretation requires review before protected assessment,
without silently changing block lengths or selecting favorable results. The current
poster path is clearly labeled development results. No new training, paid cloud
job or protected access occurred.

The unchanged publication checker rejects unreviewed binary files. Public figure
exports therefore use the verified SVG versions; all seven PNG previews remain
in the private artifact directory. Only newly created duplicate PNG export copies
were removed after checksum comparison. No source/model artifact was deleted,
and no publication/security guard was relaxed.

## Adopted Phase A and input-source qualification — November 5

The user adopted the private Phase A inference-only extension plan with SHA256
2a68d386dabe553bdda2e77f07c99c89e4081b8dc899296162fd203d0c81fd4a as an active
continuous goal. The version matched exactly. Current repository f203885 and the
completed primary/postprocessing evidence were reconciled; no scientific worker
was active and the original optimizer counter remained 1,151,690. A further
1 GiB reservation fits the existing 8 GiB cap; it has not yet been charged.

The public sensitivity protocol preserves all 72 final-development full-input
checkpoints, input-only L168/L24 admission, the fixed clean router, global clean
parity and prediction-before-target barriers, all seeds, per-lead A/A+R scoring
and the finite adopted envelope. Ninety-percent coverage is not a pass condition.
No new training, real repairs, paid jobs, downloads or protected access are authorized.

Additive source reconciliation, selected-member input loading, native causal
provenance, route construction and immutable bundle storage passed 51 synthetic
tests and independent review. Target arrays, target masks and mixed admission
members are not decoded in that path. All three routes are committed before
original common-case selectors are consulted. Real source/model execution remains
pending executor qualification; these tests did not repeat a consumed model
qualification or create any model/optimizer/scaler fit.

## Phase A executor and evaluation qualification — November 5

The bounded inference executor, separate SQLite resource ledger, one-worker
supervisor, independent per-lead evaluator and publication pipeline are implemented.
The integrated suite passed 180 tests; a final independently requested clean-barrier
binding check passed two follow-up tests, bringing unique qualified methods to 181.
All tests used synthetic data/mocked models; no study forward or target read occurred.
The qualification receipt records current code/test hashes and independent reviews.

The fixed first eight clean jobs measure all four architectures and both lookbacks
before broader execution; profiling is charged as part of required clean regression.
All 72 original clean predictions must reproduce byte-exactly before expanded
forwards. All 72 clean, 72 input-only and 36 routed products must authenticate before
verification values are loaded. Original common scores, per-lead masks, empty
cohorts, all seeds, raw negatives and independent metric/count checks remain explicit.
The phase does not add uncertainty families or attribute routed gains to weather.

Metadata reads are conservatively reserved before worker launch. Failures retain
charges; exactly one diagnosed infrastructure supervision resume is allowed within
the same clock and caps, while numerical/integrity failures remain visible.
Figure rendering is local and precharged; public exports stay sanitized SVG/JSON/MD.
Real execution and final outcome evidence remain pending at this checkpoint.

## Phase A executed, audited and prepared for publication — November 5

Execution plan 09c7a35c0d38b47e3e0325a678c970851849aa48888ea1d9cdc7069748f58d01
completed all 152 tasks at 2026-11-06T00:22:30Z, in 1,372.957 seconds. All 72
existing checkpoint predictions reproduced byte-exactly before expanded inference.
All requested predictions committed before target decoding. There were 600,996
confirmed case/checkpoint forwards, 9,420 batches, 120 model constructions using
72 distinct checkpoints, and no failed attempt or infrastructure resume. New
training/scaler fitting, downloads, paid jobs and protected access remained zero.
The original 1,151,690 scientific-update counter and frozen code/protocol remain
unchanged. An agent authentication failure at 00:10:03Z interrupted supervision,
while the detached qualified process completed normally; no worker was restarted.

All 144 product reports and 48 three-seed aggregates were produced. The independent
metric/count implementation checked 12,960 per-lead panels (maximum metric
difference 1.46e-11). All 72 original-common checks passed; 1,095 referenced
artifact hashes were verified during final review. All five scientific figures
were visually checked. Final report/publication review has separately precharged
896 MiB logical reads and 64 MiB output inside the original Phase A envelope;
unused reservations are not refunded or used for scientific repairs.

The router adds 1,651 gauge/origin cases, reaching 100%, 98.25% and 95.73%
issuance availability. Galax and Sugar Grove known-high-cell coverage reaches
87.85% and 96.74%, but Galax H1–H3 at its largest peak remain unavailable and
Sugar Grove's largest peak remains substantially underestimated. Routed-minus-L24
mean NSE is positive in 3/12 and negative in 9/12 gauge/model panels on identical
support; this does not support a universal advantage for L168-first routing.
No new policy uncertainty or routed weather/soil attribution is claimed.

A no-forward structural census applied the exact proposed K1/K2 total-slot caps,
leading-gap exclusion and original admissible endpoint constraints. Seven fixtures
and six vectorized-versus-rowwise checks passed. K2 could add 50 Galax and 82
Sugar Grove origins; one additional Galax origin has no evaluable targets. The
Phase B packet now contains measured throughput, a conservative 54-minute planning
estimate and proposed finite limits. Its real scientific execution and all new
training remain unapproved. The development-results poster is the next study
deliverable; Phase A completion does not authorize protected assessment.

## Phase A publication and closure — November 5

Results commit a0077abae5e918304ee8692ac5e2d1325a1d1a9a was pushed and matched
the remote main ref. Independent aggregate review checked 144 seed reports,
12,960 support panels and 25,920 metric means; public artifact review and all
five visual checks passed. The unchanged publication checker found zero issues.
Both Phase A bookkeeping/publication reservations and the original campaign's
Phase A reservation were settled without refunding unused allowances. Final
reservations are 6,397,053,055 logical-read bytes and 578,861,442 Phase A output
bytes; the parent global reservation remains 5,140,232,046 bytes. All limits
and the original deadline remain unchanged, and the original training counter
is still 1,151,690. A reporting-helper field-name correction required no scientific
replay. All 12 Phase A completion criteria are verified in the completion receipt;
the private handoff is updated. No worker is active. The bounded Phase A goal is
complete; the poster, optional repairs and protected assessment remain distinct.

## Assessment source and timing qualification — November 6

The user requested saved-weight assessment inference and exact-cutoff NWM
comparison, and approved bounded assessment staging (16 GiB, 4,000 GCS GETs,
$5 estimated charges within the existing $150 ceiling, plus five owned USGS
files totaling 1,371,323 bytes). All13 required USGS months and the manifest are
now verified locally. All1,463 HRRR archive identities are resolved; the first
128 archives were prefetched as opaque bytes. No assessment numerical inputs,
forecasts or scores were produced by this source-preparation chunk.

The additive calendar, source transport and NWM normalization components passed
132 tests including70 new assessment fixtures. Original development guards remain
unchanged. NWM supports originH1–16 from exactC−2 issues; H17–18 stay unsupported.
The public qualification report contains code hashes and no private source paths.

## Assessment inference readiness — November 6

The additive reader/cache/inference/runtime/controller path passed the integrated
187-test assessment suite;96 tests cover this inference chunk. Independent
reader/inference, cache and runtime/controller reviews passed. A metadata-only
preflight authenticated36 exact FULL L168 best checkpoints, all1,463 source
candidates and original training memberships; no assessment numerical read or
forward had occurred at the qualification checkpoint. All224 prefetched archives
remain opaque, with629 charged GCS GETs and a conservative $0.508967 estimate.

The separate execution plan binds code and original source/checkpoint identities.
Original development guards and1,151,690 training updates remain unchanged.
Source is_core flags are retained as acquisition ownership/padding provenance,
while explicit assessment windows and native-month ownership determine returned
rows. Tests preserve the authorized finalH18 tail through2025-07-31 17Z.
The final scoring controller has a separate release before verification reads.

## Assessment point-scoring qualification — November 6

The separate scorer and controller passed128 combined tests, including16 pure
metric tests and15 controller fixtures. Independent review passed. Every saved
prediction product must match its exact checkpoint, input-source/cohort identity,
calendar, complete-input admission and chronological B64 receipts before NWM
or verification targets are opened. A durable semantic barrier binds target-role
access; the earlier structural barrier is retained. No assessment score was
computed during this qualification.

Strict complete-H18 primary, per-lead valid-target secondary and same-support
NWM H1–16 comparisons stay distinct. All three seeds are retained with descriptive
mean/min/max; these are not confidence intervals. PBIAS is explicitly defined
as100 × sum(prediction − observation) / sum(observation).
