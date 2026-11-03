# Primary local campaign readiness

The user requested starting training on the fastest demonstrated local backends,
in seed order17→29→43. The execution components are now implemented and tested.
Scientific fits remain unstarted while the explicitly requested final method
confirmation is pending in [the protocol](primary-training-protocol-v1.md).
Paid cloud jobs and protected assessment remain outside this campaign.

## Executed qualifications

The [current trainer qualification](../reports/primary-campaign-qualification-v1.json)
passed all12 primary geometries: four architectures × fullL168/Q-onlyL168/fullL24,
B64/FP32, CPU for LSTM/GRU–Transformer and MPS for Transformer/Mamba2–Transformer.
It used exactly36 synthetic optimizer updates and12 synthetic scaler fits, in
94.88seconds, with1,872,297,984bytes peak observed parent+worker RSS plus MPS driver
memory. Fresh-process next-update losses, sample order, weights, optimizer,
CPU/MPS RNG and stopping state matched exactly. Best-only predictions matched
the saved reference and differed from last-state predictions in every case.
The large synthetic minimum-improvement threshold intentionally retained epoch1
as best; this is qualification behavior, not the scientific stopping policy.

The [real-data qualification](../reports/primary-campaign-data-qualification-v1.json)
passed all27 distinct fold/gauge/feature views, including the42,696-row largest
fit+inner bundle. It loaded every declared common outer input/label array, checked
paired target/history identities and exercised the first bounded predictor input
slice without any forward call. It used3,833,862,913 logical bytes reserved at
file open in152.28seconds, peaking at598,081,536bytes combined RSS. It performed
zero Torch imports, model/scaler fits, optimizer updates or scientific scoring.
Protected rows, raw source caches and the completed full-corpus audit were not
reopened/replayed. The source corpus remains immutable.

The [integrated verification receipt](../reports/primary-campaign-controller-check-v1.json)
records911 tests:908 passed and3 unchanged optional GIS tests skipped. The first
integrated attempt exposed two test-process isolation issues: loader-only checks
shared a process with tests that imported Torch. The tests now run those checks
in fresh subprocesses. Production guards and executed qualification receipts
were unchanged; the original failed log is retained. Independent code reviews
passed for data, ledger, worker, supervisor and seed reporting.

## Execution and recovery

The trainer's finite hard update ceiling is now200,000 per fit. A campaign still
binds each job to its smaller exact logical epoch allowance; raising a class
limit does not grant updates. Saved transforms are authenticated against original
gradient membership and values without refitting on resume. Checkpoint bytes
are serialized and precharged before file creation. An optional epoch callback
lets the supervisor retain the qualified stopping loop unchanged.

The SQLite ledger uses an exclusive worker lease, WAL/FULL durability, update
precharges before optimizer steps and confirmations afterward. Failed/unconfirmed
entries are not refunded. Each worker handles exactly one planned fit in a fresh
offline process. Source, implementation, protocol, backend and cohort identities
must match. A worker exit or checkpoint file alone is not completion: both the
selected-best model and matching prediction/metric artifacts are authenticated
before the next job advances.

Initial, every10-epoch and terminal checkpoints are retained, at most21 per job.
Three explicit same-job resumes can use only the latest authenticated checkpoint;
the budget includes up to30 epochs of physical replay while logical training
still obeys the selected ceiling. At200 epochs this is18,200,800 logical updates
and20,930,920 maximum physical entries across180 fits. Clocks and charges never
reset. There is no automatic retry, backend substitution or deletion of evidence.

Proposed hard guards are6hours per job,336hours for the campaign,4GiB combined
host/MPS memory with2GiB MPS driver/allocator ceilings,8GiB new artifacts including
a2GiB ledger/reporting reserve, and8GiB minimum free disk. These are failure-stop
limits, not completion promises. Actual large-run throughput and early-stopping
epochs are still unmeasured.

## Reports and launch boundary

`scripts/run_primary_campaign.py` prepares the exact private plan and requires
the reviewed protocol plus pinned model/data/controller receipts. A global
release record prevents moving the campaign to reset its allowance. The
supervisor launches one fresh worker per job, watches memory/disk/time/log limits,
and writes a local JSON/HTML progress view. An interrupted progress write does
not prevent later same-run recovery; abandoned partial evidence is preserved.

At108/144/180 verified jobs, the seed publisher authenticates artifacts and
publishes private JSON/HTML reports before advancing. Seed17 includes the72 CV
fits and36 final-development fits. Later seeds each add36 fits. Reports retain
undefined panels and every model/seed; three-seed means and descriptive spread
appear only after all180 jobs finish. The user can review an earlier seed while
later unchanged seeds run. No new human pause is inserted between released stages.

These first packages complete training and primary development metrics. Event
figures and dependence-aware uncertainty remain explicit postprocessing tasks
under the predeclared protocol; the initial seed report is not presented as a
finished scientific poster. The2024–25 assessment remains sealed throughout.

The protocol draft remains unfrozen until the pending method reply is received.
No primary release marker, campaign ledger, scientific optimizer update or paid
job has been created by this readiness work.
