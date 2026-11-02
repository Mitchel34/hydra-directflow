# Offline multi-cycle preprocessing

The development reader verifies the sealed acquisition index, each original cycle
commit, the archive and its members, source clocks and weather tiles before
reducing the crop with the existing candidate basin weights. USGS monthly
partitions are authenticated against the original source manifest. It performs
no network requests, downloads, source mutation, scaler fitting or training.

`build_multicycle_development` accepts up to 32 resident cycle archives and 128
hourly origins per invocation. The limits bound an offline development chunk;
they are not a full-corpus scheduler. Source layouts with 18, 36, 47 or 48 tile
slots are understood, with the original retained weather leads left intact.
Source gaps and an unavailable H18 forecast remain explicit missing cases.

For each origin and gauge, the reader selects one complete H18 weather run from
the supplied candidate set. An older extended run is allowed; forecasts are
never stitched. The most recent eligible soil state is selected independently.
The report explicitly says `OFFLINE_SUPPLIED_CANDIDATES`: a sparse local sample
does not establish which forecast would be latest across the whole archive.

The current development policy uses simulated two-hour source availability,
strict exact `A` USGS qualifiers, history ending at C−2 hours inclusive, L24 or
L168, H18 targets, and an eight-hour maximum soil age. These are engineering
settings, not a frozen main-study protocol. Original qualifiers, source IDs,
weather/soil metadata, record indices, time coordinates and missing masks remain
in the private cache. No missing basin cell is renormalized away and no missing
history value is filled. All requested origin/gauge cases survive preprocessing.

Protected assessment origins and crossing target windows are rejected before
source reads. Entire USGS months that overlap the protected assessment are also
rejected, because Parquet filters do not guarantee that only returned rows were
decoded internally. The reader therefore conservatively excludes July 2024.

## Reproduce a bounded chunk

Put local source paths in a private request JSON; relative paths resolve from
the request's directory. A minimal example is:

```json
{
  "archive_specs": [{
    "cycle_id": "20210823_00Z",
    "archive_path": "20210823_00Z.tar",
    "commit_path": "20210823_00Z.json"
  }],
  "qualified_index_path": "qualified-hrrr-index.jsonl",
  "weights_path": "weights.csv",
  "usgs_manifest_path": "usgs-source-manifest.json",
  "usgs_paths": {"2021-08": "2021-08.parquet"},
  "origins_utc": ["2021-08-23T02:00:00Z"],
  "history_hours": 168
}
```

```sh
PYTHONPATH=src .venv-offline-check/bin/python scripts/preprocess_multicycle.py \
  --request .local/multicycle-inputs/request-L168.json \
  --tensor-output .local/new-multicycle-cache.npz \
  --report .local/new-multicycle-receipt.json
```

Outputs must be new files. Tensor artifacts must remain inside this checkout's
ignored `.local/` directory. The receipt contains portable source identities,
aggregate counts, implementation hashes, stage timings and the exact cache
digest; it is suitable for review before publication. The input request, raw
data and tensor/provenance cache remain private.

## Observed integration sample

The five resident extended cycles are `20210823_00Z`, `20220317_00Z`,
`20230126_00Z`, `20230817_00Z` and `20240229_00Z`. Four origins per cycle (02Z–05Z)
produce 20 origins and 60 gauge cases. L24 has 60 complete cases. L168 has 56
complete cases: the missing Sugar Grove observation at `2023-08-14T18:00Z`
affects four history windows and is preserved as a mask. Every target, weather
and soil mask in these samples is complete.

See `reports/multicycle-L24.json` and `reports/multicycle-L168.json`. This is a
small, opportunistic resident sample spanning several years and seasons, not a
representative estimate of full-corpus missingness or scientific performance.
