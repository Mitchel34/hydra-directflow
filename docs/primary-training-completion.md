# Primary training complete — November 5, 2026

**Ready for development-results review, with scientific caveats.** All 180
planned fits completed training and development scoring by **7:09 a.m. Central on
November 5**, ahead of the November 6 draft target. Training is finished; the
scientific investigation and poster are not yet complete. Protected assessment
remains sealed and no paid cloud job is recorded.

The [completion receipt](../reports/primary-training-completion-v1.json) binds
the original plan/protocol/snapshot and all three seed reports. It records 72 CV
fits and 108 final-development fits, all four architectures and seeds 17/29/43.
All 180 primary panels and all 36 three-seed aggregates are defined; no required
lead or comparison panel was dropped.

## Verification and execution evidence

The final report was regenerated from verified artifacts and matched its existing
bytes exactly. This authenticated 919 artifact files totaling 507,221,045 logical
bytes, including each completed job’s referenced checkpoint, prediction artifact and metric panel.
Checkpoint bytes were hashed without deserialization; prediction coordinates
were checked without decoding prediction values. All three report commit maps
match the read-only ledger. No model, optimizer update, source-observation read,
protected-data access or ledger mutation was needed for verification.

Independent arithmetic from the saved per-lead metrics reproduced all 180
per-job equal-lead means, 24 CV means, 36 three-seed means/sample standard deviations
and 648 three-seed per-lead metric means within 1e-12. This check does not independently
recompute NSE from raw predictions and targets; that remains a useful additional
review step. The working arithmetic script is retained with the local evidence.

Training used 1,150,918 logical updates and 1,151,690 confirmed physical updates;
the 772 difference is the retained checkpoint replay. One monitoring interruption
and its successful recovery remain recorded. No completed fit was replaced.
Elapsed calendar time was 38.13 hours, including 4.69 hours between attempts, mostly
the recovery gap. Summed attempt spans were 33.44 hours, including data loading,
inner validation and development evaluation; these are not isolated GPU timings.

Early stopping completed fits after 10–37 epochs (median 10); 94 fits stopped at the
minimum 10, and none reached the 100-epoch cap. Selected-best checkpoints may precede
epoch 10 under the frozen protocol. All original deadlines and update allowances
were respected. The concentration at the minimum should be reported, not used
to change settings after viewing results.

## Preliminary three-seed development comparisons

These values describe the **2023–24 development validation period**, not protected
assessment. NSE is the equal-weight mean across all 18 leads, followed by the equal
mean of seeds 17/29/43. Predictions remain raw and unclipped. No best seed is selected.

All arms below use common L168-admitted support: 8,766 Jefferson, 8,125 Galax and
7,229 Sugar Grove origins per lead and seed. These counts are not independent flood
events or coverage of the entire unfiltered period. L24 uses that same support,
so this comparison does not measure its potential natural-coverage gain.

| Gauge | Model | Full L168 | Q-only L168 | Full L24 | Full−Q L168 |
|---|---|---:|---:|---:|---:|
| Jefferson | LSTM | 0.8283 | 0.6409 | 0.8284 | +0.1875 |
| Jefferson | Transformer | 0.7971 | 0.6237 | 0.8039 | +0.1734 |
| Jefferson | GRU–Transformer | 0.7917 | 0.6089 | 0.7982 | +0.1827 |
| Jefferson | Mamba-2–Transformer | 0.8007 | 0.6117 | 0.7989 | +0.1890 |
| Galax | LSTM | 0.8683 | 0.8152 | 0.8628 | +0.0531 |
| Galax | Transformer | 0.8638 | 0.8255 | 0.8782 | +0.0383 |
| Galax | GRU–Transformer | 0.8282 | 0.8220 | 0.8408 | +0.0062 |
| Galax | Mamba-2–Transformer | 0.8561 | 0.8243 | 0.8638 | +0.0318 |
| Sugar Grove | LSTM | 0.6715 | 0.5867 | 0.6786 | +0.0848 |
| Sugar Grove | Transformer | 0.6800 | 0.5647 | 0.6825 | +0.1153 |
| Sugar Grove | GRU–Transformer | 0.6503 | 0.5707 | 0.6590 | +0.0796 |
| Sugar Grove | Mamba-2–Transformer | 0.7016 | 0.5737 | 0.6698 | +0.1279 |

Full weather-plus-soil inputs have a higher primary mean than Q-only inputs in
all 12 gauge/model comparisons. This is a descriptive paired result, without a
significance claim. The lead pattern matters: full-input H1 NSE is lower in 11 of 12
comparisons, while H18 NSE is higher in all 12. The result must not be described as
improvement at every horizon or as separate evidence for weather and soil.

There is no consistent hybrid advantage across gauges and periods. For example,
LSTM has the largest full-L168 mean at Jefferson and Galax, while Mamba-2–Transformer
has the largest at Sugar Grove; these are descriptive rankings, not a model-selection
decision. CV and final-development orderings also vary. Seed dispersion, events
and dependence-aware uncertainty still need interpretation. Standalone GRU/Mamba-2
components were not tested, so the experiment cannot establish hybrid superiority
over those components.

Longer history has no uniform mean advantage. Matched full L24 is higher in 9 of 12
comparisons, usually by small amounts; Sugar Grove Mamba-2 has a larger L168 advantage.
Negative predictions remain present and must be reported by gauge/lead without
posthoc clipping. No NWM comparison is established by these primary results.

## Metadata interpretation

The nested scoring panels retain `policy_status=PROPOSED_NOT_FROZEN` and
`scientific_policy_adopted=false`. These fields are deliberately hardcoded in the
reusable scoring component, whose validation says it cannot adopt a protocol.
The [externally frozen primary protocol](../configs/primary-training-protocol-v1.json),
its recorded human approval and the hash-bound campaign plan are the adoption
authority. The numeric scoring rules match that protocol. This annotation
reconciles the metadata hierarchy; immutable run receipts are not rewritten.

## Next scientific steps

1. Audit saved predictions and target alignment, then make gauge-by-lead panels
   with common support, physical-unit errors and negative-prediction frequency.
2. Apply the predeclared event catalog and hydrograph-selection rule. Separate
   unique target hours, event episodes and peak coverage from overlapping windows.
3. Implement and benchmark the declared paired 14-day block uncertainty analysis,
   with 7/28-day sensitivity checks and all seeds retained. Report its conditional
   within-year scope and invalid-draw/support rules.
4. Assemble the development-results scientific review and existing AGU poster
   layout, including approximate-basin, revised-data availability and coverage
   limitations. Reconcile abstract framing with the advisor before final claims.
5. Review the separate protected-assessment readiness and approval gate. Training
   completion, these development scores and this receipt do not open that gate.

The November 6 scientific-review/poster draft remains a target. If assessment
readiness is incomplete, use a clearly labeled development-results poster path.
