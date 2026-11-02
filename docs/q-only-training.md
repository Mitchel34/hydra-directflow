# Versioned Q-only training compatibility

Trainer/checkpoint v2 adds an explicit feature schema while preserving the full-model default and its numerical transformation rules:

| Schema | History | Forecast-token inputs | Initial state |
|---|---|---|---|
| `FULL_HRRR_H2_W4_S2_V1` | Discharge and age | Four issued HRRR features | Soil water and soil age |
| `Q_ONLY_H2_HORIZON1_S0_V1` | The identical discharge and age | Known horizon offsets 1–18 only | Empty |

`q_only_training_bundle(full_arrays, cohort_admission_digest=...)` wraps the existing Q-only view. It copies case IDs, order, site/origin/history/target coordinates, history values/masks and target values/masks. It never fetches the original weather or soil values/masks. Full-cohort admission must already have happened upstream; its supplied digest is preserved explicitly and must match any existing full-bundle cohort tag. Already-adapted or other declared feature modes are rejected. The adapter cannot restore excluded cases, alter eligibility or invent missing targets. Outputs are read-only NumPy snapshots. Protected coordinates are rejected before numerical flow access.

Use `TrainerSpec(..., feature_schema=Q_ONLY_FEATURE_SCHEMA)` for this mode. The architecture remains an explicitly selected existing candidate; this capability does not add an architecture or choose an ablation winner. Only CPU and MPS remain admitted by the trainer. The prepared CUDA primitives are separate and do not enable this runner on CUDA.

History and target affine statistics use exactly the same training indices, observed cells, weighting, means, scales and missing-data rules in full and Q-only modes. Tests verify identical learned statistics and transformed history/labels on a matched cohort. Validation changes cannot influence them. Empty Q-only state has no scaler parameters. The known horizon channel uses the declared `h/18` encoding, recorded as `HORIZON_DIV18`, with zero fitted source cells. No physical weather or soil statistic is calculated for Q-only.

Transforms use `hydra-train-only-affine/v2`; checkpoints use `hydra-development-trainer-checkpoint/v2`. The feature schema, original source identity, comparison-cohort digest, data identity, transform, sampler, model, optimizer and stopping/RNG state are bound separately. Q-only restore requires both `expected_feature_schema=Q_ONLY_FEATURE_SCHEMA` and `expected_cohort_admission_digest=...`, in addition to the checkpoint SHA and expected source identity. Full mode remains the restore default. A mode/source/cohort mismatch fails before constructing a trainer. V1 checkpoints are not silently migrated.

The original [six-update v1 receipt](../reports/trainer-recovery.json) remains historical evidence for that exact code. The separate [v2 qualification](../reports/trainer-recovery-v2.json) passes four cases: CPU LSTM and MPS Mamba2–Transformer, each with full and Q-only features. Each case uses L24, batch four, eight synthetic training cases and four later stopping cases. Two reference updates plus one replayed next update per case give **12 synthetic updates across eight worker processes**. Model, optimizer, scaler, sampler, RNG, stopping state and next update match exactly. This qualifies those tested recovery paths, not every architecture/device/lookback combination or scientific forecast skill.

The 20 training unit tests execute no optimizer updates. This feature work does not itself authorize real scientific fitting, protected assessment, new architecture controls or cloud execution, and it does not change the previously executed fixed-unit throughput benchmark.
