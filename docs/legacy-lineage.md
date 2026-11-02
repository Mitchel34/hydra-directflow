# Reading the retained legacy archive formats

`lineage.py` normalizes supported acquisition formats without changing the
sealed index or any original archive. It accepts two authenticated routes:

- A sealed commit identity binds the original commit bytes and their bundle
  identity. When the old index omits the commit byte count, the reader still
  checks its exact sealed SHA256 under a strict size bound.
- A sealed bundle identity directly binds the archive. These rows do not need
  a separate commit file.

Missing record counts and available-group summaries are derived from the
authenticated manifest and original records. A supplied but conflicting summary
is rejected. Missing values are never converted into a claim of complete data.
Zero-record source gaps remain valid explicit gaps.

Older `HRRR-raw-cycle-acquisition/v1` payloads can lack `AVAILABILITY.json` and
`POST_FETCH_CACHE_PROVENANCE.json`. The adapter labels its availability view
`ADAPTER_DERIVED_FROM_ORIGINAL_MANIFEST_AND_RECORDS`; it does not label that view
as an original source flag. Original manifest, provenance and field-source
records remain unchanged. The pinned one-cycle loader still requires its
original modern archive layout by default.

The numerical reader continues to check record and mask hashes, source clocks,
grid identity, retained weather support and weather tile crop values. The soil
record hash and native grid are checked; this is not a fresh numerical GRIB soil
decode. Protected-period and resident-file guards remain active.

## Actual integration evidence

`reports/lineage-integration.json` records successful numerical checks for:

- `20180713_00Z`: older payload, missing index summaries/commit byte count,
  36-slot weather source tiles.
- `20210819_00Z`: direct bundle identity, no commit required, 47-slot tiles.
- `20210823_00Z`: existing modern commit-bound fixture.

The original commit for `20180726_00Z`, whose index row omits the record count,
was also authenticated. Its local tar was unavailable, so that case is labeled
identity-only and does not claim numerical validation. Synthetic tests cover
conflicting summaries, tampering, missing identities, empty gaps and unsupported
payload schemas.

An archive specification always needs `cycle_id` and `archive_path`.
`commit_path` may be omitted only when the sealed index supplies direct bundle
lineage; the loader determines this, not a caller-provided override. The
preprocessing and timing CLIs both accept that form.

Older execution receipts remain historical records bound to their listed code
hashes. They were not rerun as model-training jobs after this adapter change.
