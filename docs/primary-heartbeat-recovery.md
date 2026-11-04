# Primary campaign heartbeat recovery — November 4

The primary campaign stopped on November 3 at 20:55:47 Central after 30 verified
training/development-scoring fits. The supervisor read `heartbeat.json` while the
worker atomically replaced it. The immutable-corpus reader compared pathname
identity before and after the read and rejected the replacement. The traceback
identifies a mutable monitoring file, not a source-data shard. No source-data
corruption is established by this error.

The user explicitly requested: “Ok let’s resolve the timing conflict and get
training back on track.” This authorizes the targeted repair and same-job resume.
The latest verified checkpoint is epoch 10, logical update 3,800, for CV2
Jefferson Mamba-2–Transformer/full L168/seed 17 on MPS. The interrupted attempt
confirmed 4,572 updates. Recovery therefore replays 772 updates and retains all
original charges. The 30 completed fits are not repeated.

## Narrow operational amendment

The frozen primary supervisor, worker, trainer, data readers, protocol, private
campaign plan and original qualification receipts are unchanged. A separate
versioned recovery wrapper authenticates those original identities and applies
the new reader only to the supervisor's exact planned heartbeat paths. The
worker still runs the original script; checkpoint restoration, sample order,
optimizer, scaler, seed, stopping and scoring rules are unchanged.

The new reader opens a bounded resident regular-file descriptor. That descriptor
pins a complete old or new heartbeat during an atomic pathname replacement.
Content size and modification time must stay stable; symlinks, offloaded files,
nonregular files, oversized files and actual in-place changes remain errors.
The original supervisor still parses the JSON and checks worker/job/plan
identities, memory, disk and deadlines. All other reads retain the strict
immutable-data reader. This applies to subsequent planned heartbeat paths too.

The additive implementation has its own qualification receipt. The original
qualification does not claim to cover this amendment. The private amendment
binds the new wrapper/helper/test hashes and qualification receipt, original
plan/protocol, completed-receipt map, interrupted job, latest checkpoint,
attempts, counters and deadlines. These are rechecked immediately before an
explicit `supervise(..., resume=True)` call. A separate immutable start receipt
records the operational amendment alongside the existing run evidence.

## Preserved limits and scientific scope

The original interrupted-job deadline is November 4 at 02:41:32 Central. It is
not reset or extended. The global clock, 100-epoch ceiling, early stopping,
physical update limits, three-resume ceiling, byte/memory guards and protected
assessment gate remain unchanged. There is no fresh-campaign initialization or
automatic retry path in the recovery wrapper. If a deadline or guard prevents
completion, that remains an unfinished stop requiring review.

The new operational receipt is covered by the existing bookkeeping reservation
and counted in actual run-folder growth. No cloud job, download, source rewrite,
protected assessment access or additional scientific fit is introduced.

See the versioned qualification and resumed-run receipts in `reports/` for
executed checks and observed launch state. Protocol adoption, preparation,
launch and verified forward progress are recorded separately.

## Executed qualification

All47 focused tests passed:13 snapshot-reader tests,20 recovery-wrapper tests
and14 existing controller tests. Independent implementation review found no
blocking issue. The [qualification receipt](../reports/primary-heartbeat-recovery-qualification-v1.json)
binds the exact additive code and tests. No model call, optimizer update or
consumed qualification allowance was used. A read-only preflight authenticated
all30 completed receipts and the latest checkpoint against the original plan.
