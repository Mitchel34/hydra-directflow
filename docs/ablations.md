# Q-only LSTM comparison control

`hydra_directflow.ablations.q_only_view` produces a pure NumPy view of an already
admitted full-model case batch. It does not load sources, import Torch, instantiate
a model, fit transforms, run an optimizer or score outcomes.

For the weather-value comparison, the full model defines the admitted cohort.
Pass its exact SHA256 `cohort_admission_digest` to this function. The function
retains that digest and every case in its existing order; it never recomputes
eligibility or adds origins that lacked full-model weather or soil support. The
digest is an evidence pointer, not authentication. A broader independently
admitted Q-only cohort would be a separate declared experiment.

## Input and output fields

The input is a mapping containing the six full input keys (`history`,
`history_mask`, `weather`, `weather_mask`, `state`, `state_mask`) and these explicit
arrays and metadata:

| Field | Required shape or content |
|---|---|
| `history`, `history_mask` | `[B,L,2]`; existing floating values and Boolean masks |
| `history_features` | `("discharge_cms", "age_hours")` |
| `target`, `target_mask` | `[B,18]`; labels and Boolean label masks |
| `site_id` | `[B]`; indexes 0–2 in confirmed gauge order, or confirmed gauge strings |
| `origin_time` | `[B]`; exact integer UTC epoch hours |
| `target_time` | `[B,18]`; exact origin-relative hourly targets 1–18 |
| `case_ids` | Unique immutable tuple, one identifier per batch row |

The returned `QOnlyView` snapshots history, history mask, labels and case metadata
without changing values, masks, order, units, dtype or timestamps. Copied arrays
are read-only to prevent accidental in-place edits. This function does not scale
or encode history; the caller must already provide the same prepared history
used by the full model.

The issued-input tensor becomes known `horizon_hours` with shape `[B,18,1]`,
containing 1 through 18 for every case. Its Boolean mask is true because these
offsets are always known. The state tensor and mask are empty `[B,0]`. This true
calendar mask is not a claim that weather observations were present. Original
weather/soil values, their masks and cycle ages are never inspected or used as
predictors. Their removal does not change the inherited case-admission decision.

## Model interface and labels

`view.model_config_fields()` supplies pure keyword metadata for the existing
`ModelConfig`: architecture `LSTM`, history fields `discharge_cms` and `age_hours`,
issued field `horizon_hours`, empty state fields, the existing history length,
H18 and `site_mode="per_gauge"`. No NWM field is allowed in history or introduced
by the transformation.

`view.forward_args()` returns only the six positional model inputs in
`DirectFlowModel.forward` order. `view.target`, `view.target_mask`, gauge IDs,
origin/target clocks, case IDs and cohort digest remain outside those arguments.
Targets can change without changing forward inputs. Label masks remain exactly
those supplied by the full-model cohort; no missing label is filled.

The NumPy tests use invented arrays to verify weather/soil/age perturbation
invariance, target isolation, unchanged labels/case identity and configuration
metadata. They perform no model forward calls or updates. A later model-interface
smoke or scientific Q-only fitting run requires its own explicit execution and
receipt; creating this view establishes neither.
