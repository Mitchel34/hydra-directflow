# Completed connection-window work

Each code chunk has been tested locally and committed separately. Raw data and local dependency caches are excluded from Git.

| Commit | Chunk | Evidence |
|---|---|---|
| `c0c448b` | Repository foundation | Portable package and publication checks |
| `6b93183` | Causal protocol | 32 contract tests |
| `b9f1866` | Verified development loader | 13 loader tests and a real three-gauge H18 smoke |
| `0fd7cca` | Four-model qualification | CPU/MPS L24/L168, 72 synthetic updates |
| `502467d` | Run telemetry and input/model integration | Eight real-input forwards, zero updates; CI template |
| `25b2bf1` | Offline readiness | 37 local cache members; fresh no-index environment passed55 tests at that checkpoint |
| `6bf39c5` | Fresh-process recovery | CPU LSTM and MPS Mamba2; six synthetic updates |
| `ac64e2d` | Real-data optimizer integration | Two LSTM CPU plus two Mamba2 MPS engineering updates |
| `9970334` | Monitoring template | Terminal-state metrics and Grafana JSON template |
| `385582d` | Evaluation APIs | 15 synthetic metric/cohort/NWM tests |
| `4863559` | Current-shape local benchmark | 20 cases; 800 synthetic updates; 600 timed samples |
| `b2c2f64` | Development availability | Quality/timestamp columns only; 12 comparisons, one month |
| `cd02171` | Q-only control | 11 ablation tests and two zero-update model forwards |

The integrated offline-environment suite now passes **114 tests**. See `reports/release-check.json` for the tested code revision and source hashes.

Qualification accounting: 878 synthetic optimizer updates across the model, recovery and benchmark attempts; four completed real-data engineering updates on one repeated observed case. One earlier failed engineering admission precharged two updates but failed before optimizer entry; its record remains. These are not scientific campaign fits or forecast-skill results.

The current local benchmark reports synchronized copy/update timings, not whole-epoch or full-data workflow duration. CPU/MPS choices should be confirmed with representative real data; no CUDA throughput or cloud budget has been established.

## Offline and next steps

The wheel and small input caches are local-only. Verify them with `scripts/verify_offline_cache.py`; follow `docs/offline-work.md`. The full historical HRRR payload corpus remains in private cloud storage.

Next: generalize the authenticated reader beyond the pinned development fixture, freeze the main study policies, measure representative full preprocessing/loader/validation throughput, and prepare a bounded cloud-read/compute proposal. Do not restart acquisition or use protected assessment results for selection.

GitHub Actions activation is pending workflow-scoped credentials. The tested core workflow template is in `docs/ci/core-workflow.yml`. Grafana is a JSON template plus local telemetry output; no hosted monitoring service is running.
