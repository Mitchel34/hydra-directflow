# Development boundaries and the July 2024 export

The protected assessment starts at `2024-07-31T00:00:00Z`. Development origins
and all H18 target times must precede that boundary. The last admissible
development origin is July 30 at 05Z, with its last target at July 30 at 23Z.
An origin at July 30 at 06Z is rejected because H18 reaches the boundary.

HRRR **initializations** must also precede the assessment start. The previous
blanket rule that initialization plus 48 hours must precede the boundary was
unnecessarily restrictive: a retained forecast can contain later, unused leads.
Selection still requires each source to be available at the actual cutoff C and
one complete cycle to supply the requested development targets. Forecasts
initialized during the assessment remain rejected. This change does not enable
protected origins or scoring, and it does not expose observed assessment labels.

## A mixed monthly USGS file is not a development input

The original July 2024 USGS Parquet includes July 31 observations. A Parquet
predicate controls returned rows but may decode an entire row group first.
Consequently, supplying the original July file in `usgs_paths` still fails,
even if the requested window ends before July 31.

`development_export.py` implements a separate administrative preparation step
for exactly this month. Its default is plan-only:

```python
from hydra_directflow.development_export import plan_development_export

plan = plan_development_export(
    source_path="private-original-2024-07.parquet",
    source_manifest_path="private-original-usgs-manifest.json",
    output_directory=".local/development-2024-07",
)
```

Planning authenticates the original source manifest and inspects source file
metadata. A missing or dataless source is reported without opening or hydrating
it. Planning decodes no observation values and writes nothing.

Actual mixed-month execution is **not authorized by this implementation**. It
needs separate human authorization and an explicit call to
`execute_development_export` with `execute=True` and the exact
`ADMIN_DEVELOPMENT_EXPORT:<plan_sha256>` token from the current plan. The token
is a software admission control, not proof of human permission. It is unrelated
to and cannot replace a protected-scoring permit. No actual July 2024 export was
executed for this implementation; verification uses generated synthetic data.

## What a separately authorized execution would produce

The exporter verifies the entire original partition SHA256 and byte count
before Parquet decoding. It retains original core rows whose timestamps are in
`[2024-07-01T00:00:00Z, 2024-07-31T00:00:00Z)`. It preserves the selected original
columns, qualifier sequences, source wire identities, missing/invalid values,
and units. It does not interpolate, fit scalers, alter quality admission, or
fill targets.

The source decode may internally include protected values from mixed row
groups. Receipts state this explicitly; they do not claim zero internal protected
decoding. The function returns identities and row counts, never observations.
Mixed-source decode errors use a generic message rather than echoing input
fragments. Protected values are neither returned nor logged.

A successful export creates a new directory under this checkout's ignored
`.local` containing `data.parquet` and `COMMIT.json`. Files are flushed and the
staging directory is published atomically. Existing exports cannot be replaced;
an interrupted staging directory is not a valid export. The commitment records:

- Original source-manifest and partition hashes and byte count.
- Exact month, exclusive cutoff, selection rule, and column contract.
- The administrative plan identity, producer implementation hashes, and actual
  artifact creation time.
- Exported payload hash/size and source/exported/excluded row counts.
- The explicit mixed-row-group decoding scope and absence of scoring permission.

## Consumer admission

The Python preprocessing API takes a separate, explicit map:

```python
usgs_development_exports = {
    "2024-07": {
        "directory": ".local/development-2024-07",
        "commit_sha256": "<SHA256 returned by the authorized exporter>",
    }
}
```

Pass this map as `usgs_development_exports` to
`build_multicycle_development`, or as `development_exports` to
`load_usgs_windows`. A directory without its pinned commitment hash is rejected.
There is no fallback to the original mixed month if export verification fails.

`inspect_development_export(reference, source_partition)` authenticates the
commitment, its original source lineage, cutoff, payload identity, and
administrative-only scope without decoding values. It returns the admitted
start and exclusive end timestamps for downstream cache bounds. Cache identities
must include the pinned export commitment; a July hourly store must stop at
July 30 at 23Z rather than allocating protected July 31 slots.

`read_development_export` additionally checks the payload hash and decodes its
complete time-coordinate columns first. Any out-of-boundary row fails before
the discharge column is decoded. Only then are the development observations
returned to the ordinary quality/window adapter. Original raw files and sealed
manifests remain unchanged throughout.
