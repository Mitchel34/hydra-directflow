# Matched Q-only and L24 timing controls

`scripts/benchmark_timing_controls.py` implements a separate local engineering benchmark using the original sealed August 2023 fixture and its exact gauge/origin/target cohort. The completed 600-update benchmark script, budget and report remain unchanged.

The two input arms are Q-only L168 and full-input L24. Each includes all four existing architectures at B64. The earlier full-L168 execution policy is held fixed: CPU for LSTM and GRU–Transformer, MPS for Transformer and Mamba2–Transformer. This is not a new hardware search or a claim that those devices are optimal for shorter/Q-only inputs.

Eight cases × five warmup plus 20 measured updates gives **200 total updates**: 40 warmup and 160 measured. The new persistent allowance is separate, precharged once, and never refunds failed or unstarted cases. A worker must present the exact plan/run allowance and durably claim its case before work. The 900-second/four-GiB combined supervisor/worker/MPS limits, two-GiB MPS fraction, network denial, interruption accounting and no-retry policy match the earlier benchmark's discipline.

L24 uses the exact last 24 history values, masks and UTC slots from the admitted L168 fixture. It does not regenerate clocks, admit additional cases or change labels. Q-only uses the reviewed pure adapter; original weather and soil never enter its transform/model branch. Both arms retain the same fixed engineering Q/age/target mappings as the full benchmark, without fitting a scientific scaler. Q-only forecast tokens are known horizon offsets divided by 18; state is empty. Targets remain loss-only arguments.

Every arm preserves the original 558 training, 168 validation and 18 purged cases. Digests bind all case IDs, train/validation membership, target values/masks and target coordinates. All 64 cases within each batch are distinct; repeated passes through the real cohort are explicit. The source fixture still authenticates the full comparison cohort before deriving either arm.

Parameter counts are observed from the instantiated model in an executed receipt. At the current width 64, removing W4/S2 to horizon 1/empty-state removes 640 weather-projection parameters:

| Architecture | Full L24 | Q-only L168 |
|---|---:|---:|
| LSTM | 67,905 | 67,265 |
| Transformer | 85,057 | 84,417 |
| GRU–Transformer | 76,545 | 75,905 |
| Mamba2–Transformer | 86,093 | 85,453 |

The arms are therefore **not exactly parameter matched**. These counts do not imply forecast skill; each executed worker independently verified them before updating.

Timing includes fixture verification, mode transformation, tensorization, model setup, copy/update, first and warmed validation, inference to CPU, checkpoint write/reload and exact eval-mode forward replay. It records 20 timed validation and 20 inference batches, with no optimizer updates in either. It does not qualify resumed optimizer continuation; that is a separate trainer test. Losses/predictions are discarded and checkpoint weights remain private and excluded from scientific selection.

The default command prints a plan only. Execution requires the original `--receipt`, `--fixture`, a new private `--run-directory`, a new `--report`, `--execute` and the exact `--approved-plan-sha256`. Root execution follows review. Preparing this code performs no optimizer update, cloud operation or new source download, and does not choose a scientific normalization or winning architecture.

The reviewed execution in `reports/timing-controls.json` passed all8 cases:
200 precharged/entered/confirmed updates, including40 warmup and160 measured,
in32.18seconds with1.68GiB observed combined peak. All checkpoint forward replays
and cohort/target identity checks passed. No case was unstarted or retried;
network requests and cloud jobs were zero. Scientific fits, fitted scientific
scalers, architecture selection and protected scoring remain absent. The prior
600-update receipt is preserved, and these200 updates have their own ledger.
