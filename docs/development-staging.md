# Approved development staging

The November 2 approval reserves $35 of a combined $150 incremental ceiling for
development staging. It covers the existing generation/hash-pinned HRRR objects
and the eleven identified owned USGS files. It does not authorize new upstream
acquisition, assessment scoring, IAM changes or unconditional cloud compute.
The exact human decision is retained in `reports/execution-budget-approval.json`.

## Source plan and controls

`scripts/plan_development_staging.py` produces a private identity-pinned plan and
a portable summary. It inventories 8,841 retained HRRR cycles, 72 native USGS
development months and one verified July 2024 development export. Three warmup
cycles precede the acquired calendar and remain unavailable. The plan records
local candidates without treating filename or residency as proof of identity.

`scripts/stage_development.py` defaults to planning. Execution requires the exact
execution-plan digest, one versioned corpus directory, a bounded chunk allowance
(at most 100 per invocation), and a wall allowance (at most 1,800 seconds).
The corpus checks its time and memory limits at source/chunk boundaries; each
network operation has its own timeout. These are bounded cooperative stops,
not a claim that the entire invocation can never overrun a wall deadline.

Every GET is precharged durably in a shared SQLite WAL/FULL ledger outside the
Documents/FileProvider tree. Failed or interrupted reservations remain charged.
All run versions share that ledger: a code change or restart cannot reset it.
The limits are 100 GiB of response content, 20,000 GET attempts, a $35 conservative
cost estimate, and a 2 GiB owned raw hotset. Twenty MiB of the content allowance
is reserved for the separately bounded owned-USGS materialization path.
Cloud billing has not been reconciled; the estimate is not an invoice.

The provider reuses verified local files, authenticates commit/bundle lineage,
then verifies generation, size and SHA-256 for downloaded objects. Transient
transport errors have at most three attempts per callback, all counted against
the persistent global allowance. Integrity and admission failures stop the
increment; they are never relabeled as upstream gaps. No cloud objects are
written and no compute resources are provisioned.

Only newly owned raw staging files can be evicted. The provider requires its
ownership record, path and checksum plus a verified durable basin-cache commit.
Original source archives and unrelated files are never cleanup candidates.
The corpus, cache, source provider and transport controller have separate writer
locks. Cache-directory creation fsyncs each new ancestor before eviction can run.

`scripts/materialize_owned_usgs.py` defaults to planning and admits only the
eleven pinned native development files. Its worker claims each precharged attempt
once, inherits the provider lock, has a 60-second read deadline, and commits an
owned copy only after size and checksum verification. The original contents stay
unchanged. FileProvider wire bytes and HTTP request counts are not observable;
the receipt distinguishes encoded source bytes from network accounting.

## Corpus and source formats

`hydra_directflow.corpus` partitions the candidate development splits into 477
chunks with 53,028 origins and 159,084 gauge cases. The 36 excluded origins are
the H18 tails at the two scientific split ends; ordinary month boundaries are
not purged. L168 and complete-case summaries are candidate preparation settings,
not a protocol freeze. Every shard retains masks, exact clocks, case/source IDs
and exclusions; source flags and full provenance remain in the authenticated
shared stores and receipts. Transport failure is distinct from missing data.

Three original acquisition representations required narrow reader extensions:

- A zero-record upstream gap can contain metadata with null numeric shapes and
  no NPY payload. Admission requires the exact authenticated empty-gap contract;
  no weather or soil values are invented.
- A recorded consolidated-metadata 404 can be followed by twelve authenticated
  component JSON records. Recovery requires the exact field root, complete unique
  coordinate/field components and matching source descriptors. The original 404
  remains part of provenance.
- Weather tiles may use either little-endian float16 or float32. Element-size
  bounds follow the recorded dtype; shape, clocks, masks, grid and exact crop
  checks remain required.

The resident-format preflight passed 196 checks across 191 unique cycles. Two
wrong local variants were correctly rejected by identity; 35 candidates lacked
local authority and were skipped. This does not qualify all 8,841 cycles.
The producer-code check corroborated these three representations; speculative
support for unobserved formats was not added.

Earlier failed increments and code-bound receipts remain intact. Reader changes
use a new corpus version rather than changing the meaning of an existing shard.
Scientific training and protected scoring remain off throughout staging.
