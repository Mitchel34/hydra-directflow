# Protocol decision packet — proposal only

**The user has confirmed the 180-fit scope. This packet does not freeze the remaining protocol settings, enable training or assessment, or launch a cloud job.** The confirmation is recorded in `reports/experiment-scope-confirmation.json`. Funding approval and engineering qualification are separate from scientific adoption. This proposal uses the existing [study design](scientific-study-design.md), [confirmed decisions](../configs/scientific-decisions.json), [trainer mechanics](training-runner.md) and [measured workload report](../reports/proposed-scientific-workload-measured-controls.json); it makes no new literature or forecast-skill claim.

## Confirmed rules and retained planning boundaries

- Forecast direct hourly discharge at H1–18 for `03161000`, `03164000` and `03479000`, retaining LSTM, Transformer, GRU–Transformer and Mamba2–Transformer.
- Primary USGS admission requires A and permits only additional R flags. Preserve their order/duplicates; exclude P, e and unknown flags. Input and target admission remain separate. Never fill targets.
- Keep the global development dates: training `[2018-07-12,2023-07-31)` and validation `[2023-07-31,2024-07-31)`, UTC. These are the retained planning intervals; scientific admission and internal partitions/CV below remain proposals.
- Protect assessment **origins** `[2024-07-31,2025-07-31)`, including their declared label tail through `2025-07-31T17:00:00Z`. No development use or scoring permission follows from this packet. July development input uses the separately authorized, verified export, never the raw mixed month.
- NWM is an external comparator only, never a predictor or training-eligibility gate. Match its retained issuance/availability and native lead support. An 18-lead run under the proposed two-hour availability lag supports origin-relative H1–16; do not manufacture H17–18 support.
- Persistence is excluded from the active study following the user's reported advisor guidance. The learned Q-only control remains.
- The approved incremental ceiling is **$150 total**: $35 staging, $10 conditional CUDA qualification, $75 conditional scientific compute and $30 contingency. It excludes existing acquisition liabilities and is not a provider invoice. No assessment or IAM changes are authorized.

## Recommended scientific settings to review and adopt

These are concrete defaults for discussion, not silently selected settings.

| Decision | Recommendation | Implementation and limitation |
|---|---|---|
| Information cutoff | C is the hourly origin/cutoff. Use the prepared simulated two-hour observation/weather/soil lag; history ends C−2 inclusive. Select one complete eligible weather run with declared older-run fallback, no stitching; soil age ≤8h. | Prepared candidate mechanics. This does not reconstruct publication times or historical revisions. Record source cycle age/leads even though they are not additional model channels. |
| Cohort and grouping | Separate models per gauge. Primary comparisons use complete L168 history, weather/soil inputs and all18 admitted labels. Use the same per-gauge cases across architectures, full/Q-only and L24/L168 contrasts. | Select `common_l168_complete` only after the full numerical corpus audit. Report natural L24 coverage and every exclusion separately. Partial-input/partial-label fits and outage augmentation stay outside the primary matrix. No common-case intersection across different gauges is required. |
| Transforms | Use existing train-only affine standardization in physical-Q space; one pooled target mean/scale per gauge/fit across its H18 training labels. Fit each input channel only on observed gradient-fit tensor cells. Keep Q-only horizon tokens at horizon/18. | Implemented candidate. No log-Q or data-dependent clipping. Constant channels use scale1; absent training support fails. Overlapping tensor-cell occurrences contribute to statistics; they are not unique independent observations. Full/Q at the same lookback must agree on history/target transforms; L24 applies the same fitting rule but may have different history statistics. |
| Loss | Masked standardized-Q MSE with equal horizon contribution; no event/lead weights in the first configuration. | Implemented. Complete H18 labels make the primary weighting unambiguous. This favors reducing squared physical-Q errors and does not establish optimal low-flow or flood performance. |
| Output | Retain the implemented linear head and exact affine inverse; primary scores use the resulting physical-Q predictions **without post-hoc clipping**. Report negative-prediction frequency by gauge/lead. | No automatic nonnegativity exists. Resolve this research-output limitation during protocol review, or choose a fixed nonnegative projection/alternative head **before** fits; any alternative needs tests and uniform application. Never choose clipping from assessment results. |
| Initial configuration | Width64, dropout0.1, FP32, B64; AdamW learning rate `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01`, `foreach=False`; seeded within-fit shuffling; no scheduler initially. | Matches prepared mechanics/current optimizer defaults; write all values explicitly into run manifests. Shared width is not matched parameter count. Q-only removes640 projection parameters. Expanded search must have predeclared, comparable opportunities. |
| Stopping/checkpoints | Cap50 epochs; complete at least10; patience5; `min_improvement=0` on inner-stopping standardized MSE. Retain the earliest checkpoint at the best strictly improving loss. | Prepared stopping mechanics; best epoch may precede epoch10 although training completes at least10. Explicitly export/use `best_model_state` for scoring: the trainer's last state is not automatically the best state. Outer/assessment outcomes never drive stopping. |
| Metrics and selection | Primary development criterion: equal-weight mean of18 lead-wise NSE values, then equal weights across declared development folds. Report each gauge separately; use equal gauge weights only for a declared study summary. Also report physical-unit RMSE, MAE, signed bias, negative predictions and support counts. | Metric primitives exist; the fold/seed aggregation and selection runner still need implementation. Undefined NSE or missing required fold/lead support must be reported and make that primary comparison unavailable, not silently omitted or replaced by zero. Meaningful-effect margin and uncertainty rules need coauthor review. |
| Seeds/final weights | CV seed17, one configuration initially; final seeds17/29/43. Retain every seed's selected inner-stopping checkpoint, report per-seed results and their mean metric; no favorable-seed selection. No primary prediction ensemble or post-validation refit. | Seed/recovery mechanics exist. Freeze the checkpoint policy before execution. An ensemble, combined-development refit, or additional configurations would be a separate declared amendment. |
| Spatial interpretation | Use the existing514-row weight artifact unchanged as approximate independent study basins, preserving the specific accepted outlet/area discrepancies. No forced nesting or conservation claim. | Numerical extraction is prepared. Whole-study spatial admission and the limitation wording still require user/coauthor approval; do not present surveyed outlets or exact catchment geometry. |

## Common implementation core across all three scopes

All choices contain the same **108-fit final-development core**: 72 full/Q-only L168 fits plus36 full-input L24 fits, across four families, three gauges and three final seeds. Preparation can share:

1. A complete, immutable numerical corpus snapshot and per-gauge common-cohort manifests, with masks, coverage and source provenance retained.
2. Explicit role bounds and typed full/Q bundles. Proposed final roles are gradient fit `[2018-07-12,2023-05-01)`, inner stopping `[2023-05-01,2023-07-31)`, and outer development validation `[2023-07-31,2024-07-31)`. Boundaries never come from the first surviving row.
3. Train-only transforms, fixed feature schemas, seeded within-training sampling, checkpoint/recovery, explicit best-state export, and per-fit update/resource ceilings.
4. Development-only metrics, support accounting, paired comparisons and figure inputs, with no assessment access.

Strict `C+18h < partition_end` purges18 final hourly origins. Recommend also testing/adopting the proposed two-hour label-availability condition at simulated refit B for fit/stopping data: `C+18h+2h <= B`, which purges19 at B, not20. The earlier gradient-fit boundary and outer scoring retain the strict18-origin rule. **The typed bundle currently implements strict windows only; the additional refit predicate is not yet implemented/adopted.** Fit/tuning/selection elapsed time is not simulated, so neither convention proves historical operational readiness.

The 180-fit choice adds72 paired-L168 CV fits to the108 core. The288-fit choice additionally includes paired L24 CV and final Q-only L24, completing the proposed history/input factorial. The annual CV dates remain those proposed in [study design](scientific-study-design.md); no CV execution or adoption is implied here.

## Confirmed scope, schedule and conditional one-GPU qualification

| Scope comparison | 20-epoch sizing | 50-epoch sizing | Timing qualification |
|---|---:|---:|---|
| 108 fits | 24.61h | 61.36h | Full L168, Q-only L168 and full L24 measured |
| 180 fits | 35.85h | 89.34h | Same measured geometries; adds L168 CV |
| 288 fits | 54.44h | 135.66h | Includes72 Q-only L24 fits still using an explicit full-L168 proxy |

These are sequential local calendar-count scenarios, before actual cohort counts, convergence, full-cohort loading/scaling, scoring/uncertainty/figures, retries and travel interruptions. They are not runtime guarantees or recommended epoch counts. Once data are resident, local training need not maintain Wi-Fi, but laptop sleep, power and availability still constrain completion. Do not assume early stopping will rescue the **November6 draft target**; retain the **December11,2026 presentation** milestone. If readiness fails, produce an explicitly preliminary/development-results poster in the existing AGU layout rather than opening assessment early.

A conditional CUDA pilot may be justified if the chosen scope and documented local availability threaten that schedule. The November2 read-only snapshot showed one L4/global GPU quota slot available, not reserved capacity. Plan **one GPU**; do not assume parallel GPUs or CUDA speedup. CUDA primitives/planning exist, but actual qualification and scientific-trainer integration remain pending. Before using the reserved$10 pilot allowance, bind the admitted data/protocol, current budget ledger, bounded machine/runtime plan and local-need evidence. Qualify numerical behavior, selected-device RNG/checkpoint recovery, memory and real throughput before pricing or releasing scientific cloud work within the existing ceiling. This packet launches nothing.

## Remaining review decisions

The 180-fit scope is confirmed; the other counts remain comparison scenarios. Methodological review is still needed for the proposed settings above, CV/refit convention, primary output acceptability, whole-study spatial admission, practical-effect/uncertainty rules and any extra controls. Standalone GRU/Mamba2, separate weather-versus-soil arms and attention-component ablations remain outside these counts. Reconcile the direct-HRRR study with the older NWM/ERA5 accepted-abstract framing; no old skill claims transfer to this study and no coauthor message is sent.

After adoption, record a versioned protocol, cohort and selection contract before scientific fits. Lock all selected checkpoints before requesting separate assessment authorization. A budget approval, hash, engineering benchmark or this proposal is not that authorization.
