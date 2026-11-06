# Phase A: input-only admission and clean L168→L24 routing

**Descriptive 2023–24 development sensitivity.** Existing full-input models, saved transforms and selected-best weights are reused. No new training, short-gap repair or protected assessment is included.

At origin/cutoff C, exact Q histories end at C−2: L168 covers C−169…C−2 and L24 covers C−25…C−2. The two-hour availability delay is simulated; today's revised approved observations do not prove historical delivery at that delay. One eligible issued HRRR cycle and the separately timestamped soil initial state retain their frozen rules.

Forecast availability uses one denominator of 8,766 candidate origins per gauge. It is not multiplied by architectures or seeds. Verification is independent at each lead; an invalid H3 reference does not cancel an admissible H18 reference.

## Unique-origin coverage

| Gauge | Original common | Additional L168 | Clean L24 fallback | Unavailable | Routed forecast availability |
|---|---:|---:|---:|---:|---:|
| Jefferson | 8,766 | 0 | 0 | 0 | 8,766/8,766 (100.00%) |
| Galax | 8,125 | 56 | 432 | 153 | 8,613/8,766 (98.25%) |
| Sugar Grove | 7,229 | 136 | 1,027 | 374 | 8,392/8,766 (95.73%) |

The router selects L168 for the entire H18 vector when its inputs qualify, otherwise clean L24, otherwise unavailable. Route selection never uses future observations, target-quality masks or forecast errors. Ninety percent is a research aspiration, not a pass condition.

## Known high flow and fixed peaks

| Gauge | Product | Known-high origin/lead cells covered | Unique known-high hours with any lead | Unique hours with all 18 leads | Observed catalog peaks with any / all leads |
|---|---|---:|---:|---:|---:|
| Jefferson | Original saved L168 common | 3,456/3,456 | 192/192 | 192/192 | 9/9 · 9/9 |
| Jefferson | L168 input-only | 3,456/3,456 | 192/192 | 192/192 | 9/9 · 9/9 |
| Jefferson | L24 input-only | 3,456/3,456 | 192/192 | 192/192 | 9/9 · 9/9 |
| Jefferson | Clean L168→L24 router | 3,456/3,456 | 192/192 | 192/192 | 9/9 · 9/9 |
| Galax | Original saved L168 common | 2,404/3,474 | 154/193 | 131/193 | 5/6 · 4/6 |
| Galax | L168 input-only | 2,619/3,474 | 154/193 | 137/193 | 5/6 · 4/6 |
| Galax | L24 input-only | 3,052/3,474 | 186/193 | 156/193 | 6/6 · 5/6 |
| Galax | Clean L168→L24 router | 3,052/3,474 | 186/193 | 156/193 | 6/6 · 5/6 |
| Sugar Grove | Original saved L168 common | 3,024/6,048 | 172/336 | 164/336 | 8/11 · 7/11 |
| Sugar Grove | L168 input-only | 3,024/6,048 | 172/336 | 164/336 | 8/11 · 7/11 |
| Sugar Grove | L24 input-only | 5,851/6,048 | 333/336 | 320/336 | 11/11 · 11/11 |
| Sugar Grove | Clean L168→L24 router | 5,851/6,048 | 333/336 | 320/336 | 11/11 · 11/11 |

HIGH uses the fixed gradient-only q95 threshold, not an official flood stage. Target cells, unique valid hours and catalog peaks are different denominators; catalog episodes are not proven independent floods. No missing target is imputed and no unseen peak is inferred. All originally fixed representative events remain, including unsupported cases.

## Accuracy on explicit populations

The detail files retain every gauge, architecture, seed, product, cohort and lead, including empty/unsupported panels. Equal-lead mean NSE is unavailable unless all 18 leads meet the unchanged 168-hour/28-date/positive-variance support rules. Raw negative predictions and both emitted/evaluable negative-frequency counts remain reported.

| Gauge | Model | Routed minus L24 mean NSE on identical support |
|---|---|---:|
| Jefferson | LSTM | -0.0001 |
| Jefferson | Transformer | -0.0068 |
| Jefferson | GRU–Transformer | -0.0065 |
| Jefferson | Mamba-2–Transformer | 0.0018 |
| Galax | LSTM | 0.0047 |
| Galax | Transformer | -0.0140 |
| Galax | GRU–Transformer | -0.0089 |
| Galax | Mamba-2–Transformer | -0.0129 |
| Sugar Grove | LSTM | -0.0026 |
| Sugar Grove | Transformer | -0.0009 |
| Sugar Grove | GRU–Transformer | -0.0030 |
| Sugar Grove | Mamba-2–Transformer | 0.0116 |

Only the routed-versus-L24 contrast above is labeled paired, after exact coordinate/mask equality. Errors on newly recovered cases are reported separately; a difference from original-common scores is not a paired improvement. Original-common routed/L168 scores reproduce the frozen primary. All three seed metrics are averaged, never an averaged-prediction ensemble.

There is no Q-only L24 model. A routed-full versus unrouted-Q comparison cannot isolate weather or soil. No new policy uncertainty family is claimed; the original 42/30 contrast intervals do not automatically apply to this extension. Galax residual dependence and historical-vintage/availability limitations remain relevant.

## Execution and next decision

The accompanying accounting retains charged model/forward work, failed/replayed work, resource use and the original training counter. The complete prediction barrier precedes target loading. Raw arrays, checkpoints and detailed per-origin source/route provenance remain private.

Phase B remains a decision packet, not execution authority. If the remaining gaps justify study, predeclare bounded Q-only interior/trailing policies, route order, paired artificial gaps, clean-degradation/peak/false-alarm/negative-output/failure tolerances and measured compute costs before seeking authorization. Do not treat coverage below 90% as permission to add trials.
