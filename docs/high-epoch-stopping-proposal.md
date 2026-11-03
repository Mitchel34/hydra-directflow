# Higher ceiling with inner early stopping — proposal

**Implementation update:** the [primary campaign readiness work](primary-campaign-readiness.md)
has raised and qualified the finite trainer limit to200,000 updates per fit.
The previously identified100,000-update blocker below is historical. Exact
campaign logical/replay allowances remain explicitly bound, and the numerical
epoch choice still awaits the pending protocol confirmation.

The user prefers a high epoch limit with early stopping. A numerical ceiling has
not yet been selected by the user or frozen into a scientific execution plan.
Propose **200 maximum epochs**, retaining the existing minimum of 10 epochs,
patience of 5 completed epochs and strict loss improvement (`min_improvement=0`).
The same policy applies to LSTM, Transformer, GRU–Transformer and
Mamba-2–Transformer. A 100-epoch alternative is included below because it fits
the current per-fit update guard without extending that guard.

Monitor clean **inner-stopping standardized-Q MSE**. Each fit's transformations
remain fitted only on its gradient-training data. Outer development scores and
protected assessment never determine stopping. Evaluate the earliest saved best
inner checkpoint; final weights are not automatically the best weights. A fit
may stop well before 200 epochs. A fit reaching the epoch ceiling is recorded as
cap-limited, and a fit interrupted by a time/update/storage limit remains a
resource interruption rather than being relabeled early-stopped.

The [sizing and validation receipt](../reports/high-epoch-stopping-proposal-v1.json)
extends the published admitted-count workload arithmetically. These scenarios
assume every fit reaches its ceiling, not an expected stopping epoch:

| Ceiling | Main 180-fit local proxy | Largest per-fit update allowance | Total maximum updates |
|---|---:|---:|---:|
| Historical 50 | 82.45 hours | 31,700 | 4,550,200 |
| Alternative 100 | 164.72 hours | 63,400 | 9,100,400 |
| Proposed 200 | 329.26 hours | 126,800 | 18,200,800 |

These are conditional projections using the same measured batch rates and
admitted cases. Engineering, larger-run loading/scaling, uncertainty, figures,
thermal effects, failures and interruptions remain additional. No actual
stopping-epoch distribution or GPU acceleration has been measured for this
study. A higher ceiling is not a promise that the November 6 milestone is feasible.

The main early-stopping state machine already accepts 200 epochs, but 116 fit
instances would exceed the current 100,000-update safety bound if they ran that
long. The old LSTM missingness-pilot code separately enforces 50 epochs. Neither
guard is bypassed or changed by this proposal. Revise and qualify the exact
per-fit/global update ledger and time/storage allocations before executing a
higher-cap plan. Prior 50-epoch configurations, qualification artifacts and
consumed allowances remain unchanged.

Three pure stopping-state checks passed: a synthetic loss plateau stopped at
epoch 15 while retaining epoch 10 as the best; continual improvement reached
the 200-epoch ceiling; and the minimum-epoch guard prevented an earlier stop.
The current oversized-update rejection was confirmed. These checks created no
model, fitted no scaler, consumed no optimizer updates and read no source values.
This proposal grants no scientific execution, spending or assessment access.
