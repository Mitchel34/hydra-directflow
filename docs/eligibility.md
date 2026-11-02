# Development metadata eligibility and source inventory

`scripts/audit_development_eligibility.py` audits the development interval from July 12, 2018 through the exclusive protected-assessment boundary, July 31, 2024. It accepts explicit counting splits. This audit does not select or freeze a scientific split, admit a full model cohort, train a model, or score forecasts.

The executable API is `audit_development_eligibility(qualified_index_path=..., usgs_manifest_path=..., splits=..., usgs_paths=..., local_archive_specs=...)`. Splits are a list of objects with exactly `name`, `start_utc`, and `end_utc`; endpoints are exact UTC hours and intervals are half open. Local USGS path overrides are optional. Otherwise, paths come from the authenticated original manifest. Supplied HRRR archive specifications provide local paths for inventory; absent local paths do not mean the cloud acquisition is incomplete.

A typical invocation is:

```sh
PYTHONPATH=src .venv-offline-check/bin/python -B scripts/audit_development_eligibility.py \
  --index .local/inputs/qualified-hrrr-index.jsonl \
  --source-manifest .local/inputs/usgs-source-manifest.json \
  --splits configs/development-audit-splits.json \
  --local-request .local/month-202308-inputs/request.json \
  --local-request .local/multicycle-inputs/request-L168.json \
  --report reports/development-eligibility.json \
  --private-inventory .local/development-eligibility-inventory.json
```

Outputs must be new files. For another execution, choose new output names. The private inventory must remain under this checkout's ignored `.local/`; it contains local paths and private generation-pinned cloud object references. The public report contains aggregate counts, logical months, hashes, and a hash/size binding to that private inventory. When local requests contain the same USGS month, the first request has precedence. Relative paths are interpreted relative to each request file.

## What is read

The complete sealed HRRR index and the pinned USGS source manifest are authenticated by exact SHA256. Each required USGS monthly file is inspected before opening. Missing, dataless, nonregular, size-mismatched, or SHA-failing files cannot establish known monthly coverage. The audit never hydrates files or fetches data. For resident files, whole encoded Parquet bytes are hashed, but only `site`, timestamps, `is_exact_utc_hour`, `numeric_valid`, `source_qualifiers_json`, and `is_core` are decoded. Discharge and other value columns are never selected.

July 2024 can be inventoried here because only metadata columns are decoded. Returned metadata is restricted to timestamps before July 31. Parquet filters are logical and do not guarantee that only selected rows of those metadata columns are decoded. This is separate from the numeric preprocessing/export boundary; it does not authorize reading protected discharge values or using protected labels.

The audit uses retained `numeric_valid` flags. It does not recompute discharge sign, magnitude, units, or revision history from values. Source qualifier sequences are preserved, including order and duplicate flags. The confirmed primary rule requires A and allows only additional R flags. P, e, unknown flags, and absent A remain inadmissible.

## Three-state window counts

For each supplied split, gauge, and L24/L168 lookback, the report separates input-history, target, and joint USGS metadata counts:

| State | Meaning |
|---|---|
| `ELIGIBLE` | Every required hour lies in an authenticated, decoded metadata month and has an exact-hour row with `numeric_valid=true` and admitted qualifiers. |
| `INELIGIBLE` | All required monthly metadata is known, and at least one required hour is missing or inadmissible. |
| `UNKNOWN` | At least one required monthly partition was not available for authenticated metadata decoding. It is not counted as a missing observation or as zero eligible cases. |

For joint counts, `UNKNOWN` deliberately takes precedence when either component lacks complete inspection, even if the other component is already known to be ineligible. It describes incomplete inspection; it is not evidence that every unknown case could become eligible. The separate history and target counts retain the known component result.

History contains L exact hours ending at C−2h, inclusive. Targets are C+1 through C+18. The audit purges every origin whose last target reaches or crosses its split's exclusive end. For a sufficiently long hourly split this removes its final 18 origins. Earlier observations may enter history at a validation split's start; this reflects prior information available at C. Availability remains the existing simulated two-hour-lag scenario, not reconstructed historical publication time.

The two lookbacks use identical requested origins. Their counts describe their own masks; a scientific lookback comparison must still use a shared admissible cohort. Split ranges may overlap, such as expanding training folds. Do not sum their counts as unique source observations. The supplied July 2023–July 2024 validation year has 8,784 requested hourly origins because it includes February 29; the protected 2024–25 assessment interval is a different calendar.

## HRRR inventory limits

Only six-hour initializations can provide the retained extended H18 support under the two-hour scenario. An ordinary native 18-hour run reaches at most 16 target hours after that lag. The inventory covers single-run candidates up to 18 hours old and independent soil states up to eight hours old. Its earliest warmup candidates precede the acquired calendar and remain explicit unknowns.

Available-group summaries establish structural weather/soil support, not numerical basin coverage. Weather requires all four groups in one candidate run; complementary partial runs are never stitched. An explicitly older complete candidate may establish structural support. Legacy rows without sufficient group summaries remain unknown, even when they carry a complete-cycle flag. Final selection, source lead clocks, basin-cell masks, and values still require the authenticated numerical reader.

Known bundle bytes come only from the sealed index or SHA-authenticated resident original commits. Commit object bytes are never mistaken for weather payload bytes. Archive paths are inspected with filesystem metadata only: a resident archive is labeled stat-only because its payload is neither opened nor hashed here. Exact known byte totals are lower bounds, not a full-corpus download budget. The inventory makes no cloud listing, credential request, network call, or modification to source artifacts.

`full_model_numeric_input_eligibility` therefore remains unestablished. The combined USGS-plus-HRRR structural counts are a preparation screen, not proof that those cases can enter a scientific fit. Resolve unknown metadata coverage and perform numerical source validation before freezing that cohort.
