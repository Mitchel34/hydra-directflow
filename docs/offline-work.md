# Work while travelling

The verified cache supports this Mac's Python 3.11 environment and a small retained
development fixture. GitHub contains code and receipts; it intentionally does not
contain the wheel cache or research data. Keep the current local checkout/cache.

Verify availability before losing connectivity:

```sh
python3.11 scripts/verify_offline_cache.py
```

The cached wheel tags are macOS arm64, not a Linux/CUDA deployment bundle. A fresh
isolated environment was installed with networking disabled in pip and passed the
current unit suite. Restore it with the already installed Python 3.11:

```sh
python3.11 -m venv .venv-restored
.venv-restored/bin/python -m pip install --no-index --find-links .cache/wheels -r requirements/offline-macos-lock.txt
.venv-restored/bin/python -m pip install --no-index --no-build-isolation --no-deps -e .
.venv-restored/bin/python -m unittest discover -s tests -v
```

The fixture cache contains one verified HRRR cycle, the August 2021 USGS partition,
existing candidate weights, source metadata and the complete qualified HRRR index.
It does **not** contain the full historical weather payloads. No source downloads
are performed by the local smoke command:

```sh
.venv/bin/python scripts/local_smoke.py \
  --archive .local/inputs/hrrr-20210823_00Z.tar \
  --weights .local/inputs/weights.csv \
  --usgs .local/inputs/2021-08.parquet \
  --usgs-manifest .local/inputs/usgs-source-manifest.json \
  --report .local/repeated-source-smoke.json
```

Use `.local/` for private tensors and `runs/` for journals/checkpoints. These paths
are ignored. Do not override source hashes, fill missing observations or move the
fixture into the protected assessment period. Current bounded origins are August
23, 2021 at 02Z through 07Z with history lengths 24 or 168.

The model qualification runner performs optimizer updates and maintains an
explicit attempt ledger. Do not repeatedly invoke it as an ordinary unit test.
Its historical receipt is synthetic engineering evidence, not a new scientific fit.

CI is preserved in `docs/ci/core-workflow.yml`. The authenticated GitHub token
lacked the workflow scope required to install it under `.github/workflows/`.
Local checks are verified; no hosted CI run is claimed. Once that permission is
available, install the template without changing the scientific tests.

No service signup or paid cloud resource was created. Full data processing and
scientific training need a new bounded execution budget and the remaining study
choices. Local file residency was checked at preparation time; rerun the cache
verifier after filesystem/cloud-sync changes.
