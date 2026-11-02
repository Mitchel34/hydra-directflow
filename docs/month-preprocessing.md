# A real month of preprocessing, with resumable checkpoints

The selected engineering month is August 2023. It contains 744 hourly forecast
origins and 2,232 three-gauge cases. Under the current simulated two-hour delay,
H18 targets need the retained six-hour runs from July 31 at 06Z through August
31 at 18Z: **127 cycles**, including 126 complete cycles and one explicit source
gap. L168 history and the H18 label tail require July, August and September USGS
partitions. All three USGS partitions are resident and authenticated.

Only three of the required HRRR archives are currently resident. The remaining
124 require reads from the existing private storage archive. Consequently,
**full-month timing has not yet run**. This is an actual month selected for an
engineering throughput exercise, not evidence of representative performance
across all seasons. No repeated-cycle substitute is used.

## Prepared bounded storage read

`scripts/fetch_month_inputs.py` defaults to planning. It binds the private object
inventory to the sealed index, then authenticates original commits and their
exact archive identities. Reads are pinned to object generations and SHA256.
Verified local files are reused. The plan allows at most **1 GiB of response
bodies**, **137 object attempts**, and **900 seconds** per invocation, with a
persistent byte/attempt ledger across resumes. Failed attempts retain their
precharge. HTTP redirects and automatic retries are disabled.

The requested incremental read/transfer allowance is **$1 estimated**, still
awaiting approval. Byte and request limits are enforced locally; this is not a
provider billing cap. No credentials are requested in plan mode or mocked tests.
No cloud compute is provisioned and no cloud object is written. The existing
acquisition spending allowance is not reused.

The private inventory has exact identities for 114 of the 127 bundles; the
remaining 13 commit reads resolve their bundle identities. Its conservative
response-body requirement is 812,032,000 bytes, within the proposed 1 GiB cap.
See `reports/month-readiness.json` for the portable plan and explicit pending
state. Neither that report nor a plan digest grants spending authority.

## Monthly processing

`month_calendar()` enumerates every hourly origin, required USGS partition and
potential single weather run. A cycle can cover H18 only if its retained 36
leads cover C+1 through C+18 and it is available at C. Under the current two-hour
lag this means a source initialization between C−18 and C−2 hours, inclusive.
All six-hour candidates in that interval are included, along with independently
selected soil. Source gaps remain masks, rather than disappearing from the
calendar. Ordinary native-18-hour forecasts cannot cover H18 after this lag.

The runner assembles at most 120 origins per chunk by default, authenticating
the index once per session. Per-cycle basin stores and monthly discharge stores
are shared between chunks. Each completed chunk atomically commits a compact
case index, source-cache keys, array digest, counts and timings. Original source
metadata remains in the shared caches. A restart verifies both chunk payloads
and dependent caches, then skips completed windows. Uncommitted staging is never
treated as completion.

```sh
PYTHONPATH=src .venv-offline-check/bin/python scripts/time_month.py \
  --request .local/month-202308-inputs/request.json \
  --month 2023-08 \
  --cache-directory .local/month-202308-derived \
  --run-directory .local/month-202308-first
```

This first call only plans. After the inputs are available, an enabled call
requires `--execute --approved-plan-sha256 <the-current-plan-digest>`. These are
software execution controls, separate from any needed user budget approval.
Resources are checked cooperatively at chunk boundaries: default 900 seconds
and 4 GiB RSS. Measurements report their sampling limitations. No model updates,
fitted scalers, protected assessment scoring or network access occur in the
month-processing function.

## Timing and recovery checks to execute once inputs are available

1. Use a new derived cache and run directory; stop after one chunk with
   `--stop-after-chunks 1`. Confirm that its completed stores and chunk persist.
2. Resume the same directory. Report both attempts' wall times; the resume alone
   is not the cold full-calendar timing.
3. Use another new run directory with the populated cache to measure warm
   full-month window assembly. Compare all chunk array digests and case counts.
4. Reuse the completed directory to measure verification-only resume. Expect
   zero rebuilt chunks; do not call this a warm preprocessing measurement.

The current calendar, missing data and cache policies are development settings.
Complete-month processing does not imply that all cases have complete inputs or
that the main scientific split, quality policy or basin geometry is frozen.
