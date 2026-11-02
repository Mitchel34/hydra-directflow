# A real month of preprocessing, with resumable checkpoints

The selected engineering month is August 2023. It contains 744 hourly forecast
origins and 2,232 three-gauge cases. Under the current simulated two-hour delay,
H18 targets need the retained six-hour runs from July 31 at 06Z through August
31 at 18Z: **127 cycles**, including 126 complete cycles and one explicit source
gap. L168 history and the H18 label tail require July, August and September USGS
partitions. All three USGS partitions are resident and authenticated.

All 127 required HRRR archives are now resident and authenticated. The approved
download fetched 124 archives and 13 previously uncached commits; three archives
and 114 commits were reused locally. **Full-month timing passed on November 2,
2026.** This is an actual engineering month, not evidence of representative
performance across all seasons. No repeated-cycle substitute was used.

## Completed bounded storage read

`scripts/fetch_month_inputs.py` defaults to planning. It binds the private object
inventory to the sealed index, then authenticates original commits and their
exact archive identities. Reads are pinned to object generations and SHA256.
Verified local files are reused. The plan allows at most **1 GiB of response
bodies**, **137 object attempts**, and **900 seconds** per invocation, with a
persistent byte/attempt ledger across resumes. Failed attempts retain their
precharge. HTTP redirects and automatic retries are disabled.

The user approved **1 GiB and $1 estimated incremental read/transfer fees**.
Execution completed 137 successful GETs in **51.46 seconds**, transferring
665,344,706 bytes (**634.52 MiB**). The ledger conservatively charged one extra
byte per request: 665,344,843 bytes total, below the cap. There were no failed or
unsettled requests, cloud compute jobs or cloud writes. The existing acquisition
spending allowance was not reused.

A conservative read/transfer estimate is **$0.1804**, using published upper
rates of $0.23/GiB transfer, $0.05/GiB retrieval and $0.05 per 1,000 Class B
operations. This is an estimate, not a provider invoice; existing storage and
other project costs are outside this incremental allowance.
[Google Cloud Storage pricing](https://cloud.google.com/storage/pricing).

Before execution, the private inventory had exact identities for 114 of the
127 bundles; the remaining 13 commit reads resolved their bundle identities.
Its conservative response-body requirement was 812,032,000 bytes, within the
approved 1 GiB cap.
`reports/month-readiness.json` preserves the historical pending plan. The later
approval and completed read are recorded in `reports/month-storage-approval.json`
and `reports/month-storage-transfer.json`. Plan mode and mocked tests still make
no credential or network requests. A plan digest alone grants no spending authority.

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

## Completed timing and recovery checks

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

| Executed mode | Process wall time | Chunk behavior |
|---|---:|---|
| Empty derived cache, deliberate stop plus resumed completion | 9.2249 s | One chunk, then six new chunks and one verified reuse |
| Populated cache, new run directory, full window assembly | 5.1235 s | All seven chunks reconstructed |
| Completed run directory, verification only | 0.7841 s | Seven chunks verified; none rebuilt |

The cold number sums two fresh process invocations, including their startup;
worker-only elapsed time was 8.8236 seconds. Source download is excluded. The OS
page cache was not flushed, so "cold" refers only to the derived cache. Maximum
observed worker RSS across boundary checks and external sampling was **849 MiB**;
sampling can miss short peaks. All seven chunk array digests matched between
completed cold, warm and verification modes. The 130 shared entries (127 basin
cycles and three USGS months) total 17,041,091 bytes including commit metadata.

There are **2,046 complete cases**: all 744 at Jefferson, all 744 at Galax and
558 at Sugar Grove. Sugar Grove's missing observation at August 14, 18Z affects
18 target windows and 168 L168 history windows. Every requested case survives
with its masks. The August 17 06Z HRRR gap stays soil-only; 18 gauge cases at
origins 08Z–13Z correctly use the older complete 00Z weather run and separate
06Z soil state. No forecast stitching or observation filling occurred.

The first attempt stopped at that gap because its archive included additional
manifest-bound census-evidence files. The reader now admits only the exact
date-matched JSON/XML pair with the original proof hashes. That failed attempt
and its three completed chunks are preserved in `reports/month-timing-attempt-1.json`.
The successful timing uses a fresh cache and new code identity; failed-attempt
time is not silently folded into the successful timing or discarded.

See `reports/month-timing.json`, `reports/month-data-quality.json` and
`reports/month-execution-check.json`. No optimizer updates, scientific model
selection or protected assessment scoring occurred during this month exercise.
