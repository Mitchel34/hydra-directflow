# Read-only staging snapshots

`scripts/inspect_staging_progress.py` prints portable JSON and performs no source fetch, hydration, budget settlement, model operation or background monitoring. It imports only the standard library and never opens `model.npz`.

```sh
python3 scripts/inspect_staging_progress.py \
  --corpus-directory .local/development-corpus-L168-v3 \
  --expected-plan-sha256 <trusted-corpus-plan-sha256> \
  --last-increment .local/development-source-stage-v1/LATEST_INCREMENT.json \
  --budget-database <private-source-budget.sqlite3> \
  --expected-budget-approval-sha256 <trusted-approval-sha256>
```

The inspector verifies the pinned PLAN identity, each matching COMMIT and the hashed `case-index.json`. Numeric shards receive existence/type/size checks only. Shared-store bodies and numerical arrays are not reverified, so the result is a **metadata snapshot**, not a full dataset-integrity or training-readiness certificate. Duplicate payload claims, corrupt metadata, absent members and unplanned entries cannot inflate progress. Missing planned chunks are uncommitted work, not upstream weather gaps.

Counts distinguish requested candidate origins/cases, committed metadata and admissions recorded in case-index receipts. `last_committed_origin_utc` can lie beyond a hole; `contiguous_committed_through_utc` stops at the first missing or invalid planned chunk. Dataless numeric payloads are reported without hydrating them. Current chunk counts come from immutable commitments, not a replaceable terminal summary.

The optional budget database opens with SQLite `mode=ro` and `query_only`. Its totals must reconcile with attempt rows. `PRECHARGED` attempts remain pending and retain their full charge; the inspector never resets, retries or settles them. An unreadable or inconsistent budget is unknown, not zero. The displayed cost is the existing staging formula's conservative estimate, not an invoice or a reconciliation of the entire combined funding envelope. FileProvider transport is outside this SQLite view.

The last increment is separately attributed to its creation time. Its age, progress lag, terminal status and sanitized failure are visible even while newer chunks appear. A previous stopped/failed increment does not establish current process health. Filesystem, SQLite and process observations have separate timestamps and are not one globally atomic snapshot.

Live process association is optional. Supply `--expected-pid`, the SHA256 of the **exact trusted launch command as represented by ps** with `--expected-command-sha256`, and its absolute checkout directory with `--expected-cwd`. The inspector checks ps, cwd, the staging script, execution arguments and corpus-run name. It returns no raw command or cwd. Without a strong match it reports observed identity as unverified or mismatched. A missing process is reported as not observed. Lockfile existence never implies a running worker. A verified process observation still does not claim progress, health or future liveness.

Tests use invented JSON, opaque placeholder payloads, temporary SQLite databases and mocked process observations. They cover corruption, duplicate/missing chunks, preserved pending charges, budget errors and absent/unverified workers. No user target or feature values are read.
