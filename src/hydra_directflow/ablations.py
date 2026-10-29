"""A Q-only LSTM view of an already admitted full-model comparison cohort.

Only NumPy transformations live here. This module does not import Torch, create
models, change case eligibility, fit scalers, load sources, or compute scores.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .contract import GAUGE_IDS, HORIZONS_HOURS
from .reference import require


Q_HISTORY_FEATURES = ("discharge_cms", "age_hours")
Q_ISSUED_FEATURES = ("horizon_hours",)
Q_STATE_FEATURES = ()
_FULL_INPUT_KEYS = ("history", "history_mask", "weather", "weather_mask", "state", "state_mask")


def _snapshot(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _array(value, name, shape=None, kind=None):
    require(isinstance(value, np.ndarray), name + " must be an explicit NumPy array")
    if shape is not None:
        require(value.shape == shape, name + " shape differs from the unchanged cohort")
    if kind is not None:
        require(value.dtype.kind in kind, name + " dtype differs")
    return value


@dataclass(frozen=True)
class QOnlyView:
    """Copied read-only arrays; labels and case metadata stay outside forward."""

    history: np.ndarray
    history_mask: np.ndarray
    weather: np.ndarray
    weather_mask: np.ndarray
    state: np.ndarray
    state_mask: np.ndarray
    site_id: np.ndarray
    origin_time: np.ndarray
    target_time: np.ndarray
    case_ids: tuple[str, ...]
    target: np.ndarray
    target_mask: np.ndarray
    cohort_admission_digest: str

    def forward_args(self):
        """Match DirectFlowModel.forward's six positional inputs, without labels."""
        return (self.history, self.history_mask, self.weather, self.weather_mask,
                self.state, self.state_mask)

    def model_config_fields(self):
        """Pure metadata for a later ModelConfig; never construct/import a model."""
        return dict(architecture="LSTM", history_features=Q_HISTORY_FEATURES,
                    weather_features=Q_ISSUED_FEATURES, state_features=Q_STATE_FEATURES,
                    history_hours=self.history.shape[1], horizon_hours=HORIZONS_HOURS,
                    site_mode="per_gauge")


def q_only_view(batch: Mapping, *, cohort_admission_digest):
    """Remove HRRR predictors while retaining the full-model's exact case set.

    Required numeric arrays: six full input arrays, target/target_mask, site_id,
    origin_time [B], target_time [B,18]. Metadata: case_ids, history_features.
    The original weather/state values AND masks are deliberately never read.
    Their presence identifies the supplied full input batch; admission and source
    authentication must already have happened upstream. A digest is not proof by
    itself. Missing cases are never reintroduced by this view.
    """
    require(isinstance(batch, Mapping), "An explicitly admitted full-model batch is required")
    require(all(key in batch for key in _FULL_INPUT_KEYS), "Supply the full-model input batch before ablation")
    require(isinstance(cohort_admission_digest, str) and len(cohort_admission_digest) == 64
            and all(c in "0123456789abcdef" for c in cohort_admission_digest),
            "Preserve the full-model SHA256 cohort admission identity")
    require(tuple(batch["history_features"]) == Q_HISTORY_FEATURES,
            "Q-only history must contain exactly discharge and its age; no weather or NWM channels")
    history = _array(batch["history"], "history", kind="f")
    require(history.ndim == 3 and history.shape[0] > 0 and history.shape[1] > 0
            and history.shape[2] == len(Q_HISTORY_FEATURES), "History must have shape [B,L,2]")
    count, _, _ = history.shape
    horizons = len(HORIZONS_HOURS)
    history_mask = _array(batch["history_mask"], "history_mask", history.shape, "b")
    target = _array(batch["target"], "target", (count, horizons), "f")
    target_mask = _array(batch["target_mask"], "target_mask", target.shape, "b")
    site_id = _array(batch["site_id"], "site_id", (count,), "iU")
    if site_id.dtype.kind == "i":
        require(np.all((site_id >= 0) & (site_id < len(GAUGE_IDS))), "Gauge indexes must use the confirmed scope order")
    else:
        require(all(value in GAUGE_IDS for value in site_id.tolist()), "Gauge identities must use the confirmed scope")
    origin_time = _array(batch["origin_time"], "origin_time", (count,), "i")
    target_time = _array(batch["target_time"], "target_time", (count, horizons), "i")
    require(np.all(origin_time % 3600 == 0), "Origins must be exact UTC epoch hours")
    # Python integers avoid silent int64 overflow when validating source clocks.
    expected_times = [[int(origin) + h * 3600 for h in HORIZONS_HOURS] for origin in origin_time]
    require(target_time.tolist() == expected_times, "Target coordinates must preserve exact C+1..18 hours")
    case_ids = batch["case_ids"]
    require(type(case_ids) is tuple and len(case_ids) == count
            and all(isinstance(case, str) and case for case in case_ids)
            and len(set(case_ids)) == count, "Preserve unique immutable case IDs in their original order")
    require(len(set(zip(site_id.tolist(), origin_time.tolist()))) == count,
            "Duplicate gauge/origin cases require upstream resolution")
    known_horizons = np.broadcast_to(np.array(HORIZONS_HOURS, dtype=history.dtype)[None, :, None],
                                     (count, horizons, 1))
    return QOnlyView(
        history=_snapshot(history), history_mask=_snapshot(history_mask),
        weather=_snapshot(known_horizons), weather_mask=_snapshot(np.ones((count, horizons, 1), dtype=bool)),
        state=_snapshot(np.empty((count, 0), dtype=history.dtype)), state_mask=_snapshot(np.empty((count, 0), dtype=bool)),
        site_id=_snapshot(site_id), origin_time=_snapshot(origin_time), target_time=_snapshot(target_time),
        case_ids=case_ids, target=_snapshot(target), target_mask=_snapshot(target_mask),
        cohort_admission_digest=cohort_admission_digest,
    )
