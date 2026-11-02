# Development data completion — November 2, 2026

The development corpus completed at **23:21:33 UTC / 17:21:33 Central**. The
subsequent complete audit and independent report reconciliation passed. These
are engineering and data-admission results, not forecast-skill findings.

| Item | Verified result |
|---|---:|
| Immutable typed shards | 477 |
| Requested origins | 53,064 |
| H18 boundary-purged origins | 36 |
| Retained origins | 53,028 |
| Gauge/origin cases | 159,084 |
| Candidate complete cases, L168 | 146,464 |
| Natural complete cases, L24 | 153,216 |
| Physical-domain violations in admitted inputs | 0 |

The common L168 cohort is preserved when comparing full/Q-only inputs or viewing
the same histories at L24. The 6,752 additional natural L24 cases describe
coverage; they do not silently enter a matched history comparison.

| Gauge | Training L168 | Validation L168 | Natural L24 total |
|---|---:|---:|---:|
| Jefferson, 03161000 | 42,715 | 8,766 | 52,057 |
| Galax, 03164000 | 38,297 | 8,125 | 49,867 |
| Sugar Grove, 03479000 | 41,332 | 7,229 | 51,292 |

All 73 development USGS months have known metadata: 72 native partitions and
the verified July 2024 development export. Original qualifiers and exclusions
remain preserved. The three pre-acquisition warmup cycle references remain
explicitly unavailable; none was invented or fetched outside the sealed scope.

The exact development source inventory contains **8,841 unique retained archives**:
8,770 complete cycles and 71 documented source gaps, totaling **46,942,832,640
bytes (43.72 GiB)**. This is the development subset, distinct from the already
accounted full acquisition calendar of 61,658 complete cycles plus 166 gaps.
No new upstream acquisition was performed. Original raw archives remain in their
retained storage; local training uses verified derived stores and typed shards.

Storage transport used 13,071 precharged attempts: 13,067 verified and four failed
attempts whose charges were retained. Observed GCS response bodies total
46,218,766,666 bytes; the conservative estimate is **$12.907585 within the approved
$35 staging allocation**. This is not a reconciled invoice. Owned-USGS FileProvider
traffic is separately qualified as unobserved wire transfer. The approved combined
incremental ceiling remains $150. No cloud compute jobs were started.

Local disk usage observed after completion was approximately 1.2 GiB for shared
derived stores, 75 MiB for the final typed corpus, and 23 MiB for source-provider
metadata, owned USGS copies and retained interruption evidence. Earlier versions
and failed-run receipts remain intact; no broad cleanup was performed.

## Evidence and remaining decisions

- `reports/development-staging-increment-11.json`: completed corpus and transport accounting.
- `reports/development-corpus-admission-v1.json`: complete numerical candidate audit, masks, exclusions, source ages and domains.
- `reports/development-source-final-inventory.json`: exact metadata/identity inventory and private catalog hash; no fresh cloud-existence or raw-payload audit is claimed by that inventory.
- `reports/experiment-scope-confirmation.json`: the user's confirmed 180-fit scope.
- `reports/proposed-fold-admitted-counts.json`: exact common-case counts under both still-proposed refit-label conventions.
- `reports/confirmed-180-fit-admitted-runtime.json`: measured-rate, actual-count workload scenarios.

The 180-fit scope consists of 72 paired L168 CV fits and 108 final-development
fits, including the full-input L24 contrast. Its actual-count local scenarios are
**33.09 hours at 20 epochs** and **82.45 hours at 50 epochs**, excluding full-cohort
loading/scaling, scoring/uncertainty/figures, failures and travel interruptions.
Both proposed refit-label conventions yield the same batch-rounded timing while
retaining distinct case counts. No early-stopping or CUDA speedup is assumed.

The remaining methodological settings are in `docs/protocol-decision-packet.md`.
Source preparation does not freeze them. Scientific fitting, model selection,
conditional CUDA qualification and protected assessment remain separate steps.
Protected assessment access/scoring still requires its own satisfied gates and
explicit authorization; the earlier approved July administrative export is not
assessment-scoring permission. Negative or inconclusive findings remain valid.
