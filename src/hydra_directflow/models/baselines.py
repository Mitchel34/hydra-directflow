"""Direct-flow reference forecasts from an already causal history window.

The caller supplies the availability/quality mask from scaffold.reference.
No observation is filled or changed here: these functions create predictions.
"""
import math


def _check(values, mask):
    if len(values) != len(mask) or any(type(m) is not bool for m in mask):
        raise ValueError("Aligned values and boolean admission masks required")
    if any(m and (type(v) not in (int, float) or not math.isfinite(v) or v < 0)
           for v, m in zip(values, mask)):
        raise ValueError("Admitted discharge must be finite and nonnegative")


def persistence(history_values, history_mask, *, horizons_hours):
    """Repeat the most recent admitted past observation; retain missingness."""
    _check(history_values, history_mask)
    if (not horizons_hours or any(type(h) is not int or h < 1 for h in horizons_hours)
            or tuple(sorted(set(horizons_hours))) != tuple(horizons_hours)):
        raise ValueError("Positive ordered unique horizon offsets required")
    admitted = [v for v, m in zip(history_values, history_mask) if m]
    value = float(admitted[-1]) if admitted else None
    return dict(values=(value,) * len(horizons_hours),
                mask=(bool(admitted),) * len(horizons_hours))


def seasonal_persistence(history_times, history_values, history_mask, *,
                         target_times, origin_anchor, cutoff, period_hours):
    """Use target minus a declared period, only when that past value is admitted."""
    _check(history_values, history_mask)
    if (len(history_times) != len(history_values)
            or tuple(sorted(set(history_times))) != tuple(history_times)
            or type(cutoff) is not int or type(origin_anchor) is not int
            or type(period_hours) is not int or period_hours < 1
            or any(type(t) is not int or t >= origin_anchor for t in history_times)
            or any(m and t > cutoff for t, m in zip(history_times, history_mask))
            or not target_times
            or any(type(t) is not int or t <= max(origin_anchor, cutoff) for t in target_times)):
        raise ValueError("Unique ordered causal history and future target times required")
    past = {t: float(v) for t, v, m in zip(history_times, history_values, history_mask) if m}
    keys = tuple(t - period_hours * 3600 for t in target_times)
    masks = tuple(key <= cutoff and key in past for key in keys)
    return dict(values=tuple(past[key] if m else None for key, m in zip(keys, masks)),
                mask=masks, reference_times=keys)
