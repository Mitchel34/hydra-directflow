"""Pure reference functions for causal windows, support masks and shared cases.

All times are explicit integer UTC epoch seconds. Inputs are caller-supplied
records, not data locations. Missing values stay None until the separate masked
network encoding boundary. No interpolation, collection or training occurs.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import time
from typing import Callable, Iterable, Mapping

from .contract import CORE_WEATHER_FEATURES, HORIZONS_HOURS

HOUR = 3600
ASSESSMENT_START = int(datetime(2024, 7, 31, tzinfo=timezone.utc).timestamp())
ASSESSMENT_END = int(datetime(2025, 7, 31, tzinfo=timezone.utc).timestamp())
ASSESSMENT_LAST_TARGET = ASSESSMENT_END + 17 * HOUR


def require(condition, message):
    if not condition:
        raise ValueError(message)


def epoch(value):
    require(type(value) is int, "Explicit integer UTC epoch seconds are required")
    return value


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def forecast_target_times(origin_anchor, horizons_hours):
    origin_anchor = epoch(origin_anchor)
    horizons_hours = tuple(horizons_hours)
    require(origin_anchor % HOUR == 0, "Forecast origin must be an exact UTC hour")
    require(horizons_hours and all(type(h) is int and h > 0 for h in horizons_hours)
            and tuple(horizons_hours) == tuple(sorted(set(horizons_hours))),
            "Unique increasing positive horizon offsets required")
    return tuple(origin_anchor + h * HOUR for h in horizons_hours)


@dataclass(frozen=True)
class ReplayClock:
    """Historical information/delivery clocks and actual artifact time stay distinct.

    Creating this record stamps actual wall-clock time. Historical artifacts may
    supply their recorded creation time explicitly; it is never inferred from C.
    """

    replay_cutoff: int
    replay_completed_at: int
    artifact_created_at: int = field(default_factory=lambda: int(time.time()))

    def __post_init__(self):
        require(epoch(self.replay_cutoff) % HOUR == 0, "C must be an exact UTC hour")
        require(epoch(self.replay_completed_at) >= self.replay_cutoff,
                "Replay completion cannot precede C")
        epoch(self.artifact_created_at)


@dataclass(frozen=True)
class AvailabilityPolicy:
    policy_id: str
    mode: str
    simulated_lag_seconds: int | None = None

    def __post_init__(self):
        require(bool(self.policy_id), "A named availability policy is required")
        require(self.mode in ("documented", "retrospective_simulated"), "Unknown availability mode")
        if self.mode == "retrospective_simulated":
            require(type(self.simulated_lag_seconds) is int and self.simulated_lag_seconds >= 0,
                    "Simulated availability needs an explicit nonnegative lag")
        else:
            require(self.simulated_lag_seconds is None, "Documented policy cannot hide a simulated lag")

    def available_at(self, reference, documented):
        reference = epoch(reference)
        if self.mode == "retrospective_simulated":
            return reference + self.simulated_lag_seconds
        require(documented is not None, "Historical availability is unknown; choose an explicit simulation policy")
        require(epoch(documented) >= reference, "Availability cannot precede the observed/source reference time")
        return documented


@dataclass(frozen=True)
class WeatherRun:
    run_id: str
    reference: int
    available_at: int | None
    valid_times: tuple[int, ...]
    feature_support: tuple[tuple[str, tuple[bool, ...]], ...] = ()
    source_quality_flags: tuple[str, ...] = ()

    def __post_init__(self):
        require(isinstance(self.run_id, str) and self.run_id, "An explicit source run identity is required")
        require(type(self.valid_times) is tuple, "Immutable source valid-time coordinates required")
        require(epoch(self.reference) % HOUR == 0, "HRRR initialization must be an exact UTC hour")
        require(all(epoch(t) > self.reference and (t - self.reference) % HOUR == 0 for t in self.valid_times),
                "Only exact retained forecast lead times are allowed")
        require(len(set(self.valid_times)) == len(self.valid_times), "Duplicate source valid time")
        if self.available_at is not None:
            require(epoch(self.available_at) >= self.reference, "Weather availability cannot precede initialization")
        require(type(self.feature_support) is tuple, "Immutable original feature masks required")
        names = []
        for item in self.feature_support:
            require(type(item) is tuple and len(item) == 2, "Expected (feature, mask) pairs")
            name, mask = item
            require(isinstance(name, str) and name, "Explicit weather feature identity required")
            require(type(mask) is tuple and len(mask) == len(self.valid_times)
                    and all(type(value) is bool for value in mask), "Feature mask must align with valid_times")
            names.append(name)
        require(len(names) == len(set(names)), "Duplicate weather feature support")
        require(type(self.source_quality_flags) is tuple
                and all(isinstance(flag, str) for flag in self.source_quality_flags), "Retain immutable source quality flags")


def latest_weather_run(runs: Iterable[WeatherRun], *, cutoff, origin_anchor, policy,
                       horizons_hours, required_features=CORE_WEATHER_FEATURES,
                       allow_older_extended_cycles):
    """Latest available SINGLE run complete for every requested feature/target.

    An older extended cycle is considered only with explicit permission. Its
    targets can have source leads above 18. No cross-cycle stitching occurs.
    Coordinate coverage alone does not prove complete source feature support.
    """
    cutoff, origin_anchor = epoch(cutoff), epoch(origin_anchor)
    require(cutoff == origin_anchor, "Nominal origin and information cutoff are the same C")
    target_times = forecast_target_times(origin_anchor, horizons_hours)
    required_features = tuple(required_features)
    require(required_features and len(set(required_features)) == len(required_features)
            and all(feature in CORE_WEATHER_FEATURES for feature in required_features),
            "Explicit supported core-weather features required; no NWM predictors")
    require(type(allow_older_extended_cycles) is bool, "Older extended-cycle permission must be explicit")
    candidates = []
    seen = set()
    for run in runs:
        require(run.run_id not in seen, "Duplicate source run identity")
        seen.add(run.run_id)
        reference = epoch(run.reference)
        require(all(epoch(t) > reference for t in run.valid_times), "Forecast valid times must follow initialization")
        require(len(set(run.valid_times)) == len(run.valid_times), "Duplicate forecast valid time")
        if reference <= cutoff and policy.available_at(reference, run.available_at) <= cutoff:
            candidates.append(run)
    if not candidates:
        return None
    references = [run.reference for run in candidates]
    require(len(references) == len(set(references)), "Competing vintages at one initialization need explicit resolution")
    ordered = sorted(candidates, key=lambda run: run.reference, reverse=True)
    for run in ordered if allow_older_extended_cycles else ordered[:1]:
        indexes = {valid: index for index, valid in enumerate(run.valid_times)}
        support = dict(run.feature_support)
        if all(t in indexes and all(name in support and support[name][indexes[t]]
                                   for name in required_features) for t in target_times):
            return run
    return None


def weather_support(run, *, origin_anchor, completion, horizons_hours):
    """A source lead18 cannot promise18 hours after delay or inference time."""
    origin_anchor, completion = epoch(origin_anchor), epoch(completion)
    require(origin_anchor % HOUR == 0, "Forecast origin anchor must be an exact UTC hour")
    require(completion >= origin_anchor, "Forecast completion cannot precede its origin anchor")
    target_times = forecast_target_times(origin_anchor, horizons_hours)
    return dict(target_times=target_times,
                support_scope="SOURCE_TIME_COORDINATES_ONLY_REQUIRES_ORIGINAL_FEATURE_MASKS",
                run_age_hours=(origin_anchor - run.reference) / HOUR if run else None,
                source_quality_flags=run.source_quality_flags if run else (),
                effective_lead_hours=tuple((t - completion) / HOUR for t in target_times),
                source_lead_hours=tuple((t - run.reference) / HOUR if run else None for t in target_times),
                supported=tuple(bool(run and t > completion and t in run.valid_times) for t in target_times))


@dataclass(frozen=True)
class Observation:
    site: str
    valid_at: int
    available_at: int | None
    value: float | None
    quality_flags: tuple[str, ...]
    source_id: str | None = None

    def __post_init__(self):
        require(isinstance(self.site, str) and self.site, "Explicit observation gauge required")
        epoch(self.valid_at)
        if self.available_at is not None:
            require(epoch(self.available_at) >= self.valid_at, "Observation availability cannot precede valid time")
        require(type(self.quality_flags) is tuple and all(isinstance(flag, str) for flag in self.quality_flags),
                "Original observation quality flags must remain an immutable tuple")


def history_window(observations, *, site, origin_anchor, cutoff, lookback_hours, history_lag_hours,
                   policy, admit_quality: Callable[[tuple[str, ...]], bool]):
    """L exact hours ending at C-lag, inclusive; no search back through gaps."""
    origin_anchor, cutoff = epoch(origin_anchor), epoch(cutoff)
    require(origin_anchor % HOUR == 0, "Hourly origin anchor must be on the UTC hour")
    require(origin_anchor == cutoff, "Nominal origin and information cutoff are the same C")
    require(type(lookback_hours) is int and lookback_hours > 0 and callable(admit_quality),
            "Explicit lookback and quality policy are required")
    require(type(history_lag_hours) is int and history_lag_hours >= 1,
            "Explicit positive whole-hour history lag required")
    last_history = cutoff - history_lag_hours * HOUR
    times = tuple(last_history - h * HOUR for h in range(lookback_hours - 1, -1, -1))
    rows = {}
    for row in observations:
        if row.site == site and row.valid_at in times:
            require(row.valid_at not in rows, "Duplicate exact-hour gauge record")
            rows[row.valid_at] = row
    values, masks, ages, reasons, flags, sources = [], [], [], [], [], []
    for t in times:
        row = rows.get(t)
        reason = None
        if row is None:
            reason = "MISSING_EXACT_HOUR"
        elif policy.available_at(epoch(row.valid_at), row.available_at) > cutoff:
            reason = "UNAVAILABLE_AT_CUTOFF"
        elif row.value is None or type(row.value) not in (int, float) or not math.isfinite(row.value) or row.value < 0:
            reason = "INVALID_DISCHARGE"
        elif not admit_quality(row.quality_flags):
            reason = "QUALITY_NOT_ADMITTED"
        values.append(float(row.value) if reason is None else None)
        masks.append(reason is None)
        ages.append((cutoff - t) / HOUR if reason is None else None)
        reasons.append(reason)
        flags.append(row.quality_flags if row else None)
        sources.append(row.source_id if row else None)
    return dict(times=times, values=tuple(values), mask=tuple(masks), age_hours=tuple(ages),
                reasons=tuple(reasons), quality_flags=tuple(flags), source_ids=tuple(sources))


def target_window(observations, *, site, origin_anchor, horizons_hours, admit_quality):
    """Separate retrospective labels; missing/rejected targets remain None.

    Label availability is not an input eligibility rule. The original acquired
    flags/identity are retained; this does not reconstruct historical revisions.
    """
    require(callable(admit_quality), "Explicit label quality policy required")
    times = forecast_target_times(origin_anchor, horizons_hours)
    rows = {}
    for row in observations:
        if row.site == site and row.valid_at in times:
            require(row.valid_at not in rows, "Duplicate exact-hour target record")
            rows[row.valid_at] = row
    values, masks, reasons, flags, sources = [], [], [], [], []
    for valid in times:
        row = rows.get(valid)
        reason = ("MISSING_EXACT_HOUR" if row is None else
                  "INVALID_DISCHARGE" if type(row.value) not in (int, float)
                  or not math.isfinite(row.value) or row.value < 0 else
                  "QUALITY_NOT_ADMITTED" if not admit_quality(row.quality_flags) else None)
        values.append(float(row.value) if reason is None else None)
        masks.append(reason is None)
        reasons.append(reason)
        flags.append(row.quality_flags if row else None)
        sources.append(row.source_id if row else None)
    return dict(times=times, values=tuple(values), mask=tuple(masks), reasons=tuple(reasons),
                quality_flags=tuple(flags), source_ids=tuple(sources))


@dataclass(frozen=True)
class SoilInitialState:
    source_id: str
    reference: int
    available_at: int | None
    value: float | None
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self):
        require(isinstance(self.source_id, str) and self.source_id, "Explicit soil source identity required")
        epoch(self.reference)
        if self.available_at is not None:
            require(epoch(self.available_at) >= self.reference, "Soil availability cannot precede initialization")
        require(type(self.quality_flags) is tuple and all(isinstance(flag, str) for flag in self.quality_flags),
                "Original soil quality flags must remain an immutable tuple")


def soil_state(states, *, cutoff, policy, max_age_seconds):
    """Carry one known initial state with age; never invent future soil."""
    cutoff = epoch(cutoff)
    require(type(max_age_seconds) is int and max_age_seconds >= 0, "An explicit maximum soil age is required")
    candidates = []
    for state in states:
        reference = epoch(state.reference)
        require(reference % (6 * HOUR) == 0, "SOILW initial states exist only at selected00/06/12/18 hours")
        if reference <= cutoff and policy.available_at(reference, state.available_at) <= cutoff:
            candidates.append(state)
    if not candidates:
        return dict(value=None, mask=False, age_hours=None, source_id=None, quality_flags=None,
                    reason="NO_AVAILABLE_INITIAL_STATE")
    latest_time = max(s.reference for s in candidates)
    latest = [s for s in candidates if s.reference == latest_time]
    require(len(latest) == 1, "Competing soil vintages need explicit resolution")
    state = latest[0]
    age = cutoff - state.reference
    usable = (age <= max_age_seconds and type(state.value) in (int, float)
              and math.isfinite(state.value) and 0 <= state.value <= 1)
    return dict(value=float(state.value) if usable else None, mask=usable, age_hours=age / HOUR,
                source_id=state.source_id, quality_flags=state.quality_flags,
                reason=None if usable else "STATE_AGE_OR_VALUE_NOT_ADMITTED")


@dataclass(frozen=True)
class Split:
    name: str
    start: int
    end: int

    def __post_init__(self):
        require(epoch(self.start) < epoch(self.end), "A nonempty half-open UTC split is required")


def target_window_contained(split, *, origin_anchor, target_times, embargo_seconds):
    """Development only: purge every case whose labels cross its boundary."""
    epoch(origin_anchor)
    target_times = tuple(target_times)
    require(split.end <= ASSESSMENT_START, "Development splits cannot include protected assessment labels")
    require(type(embargo_seconds) is int and embargo_seconds >= 0, "Explicit nonnegative embargo required")
    require(target_times and all(epoch(t) > origin_anchor for t in target_times), "Targets must be strictly future valid times")
    return (split.start <= origin_anchor < split.end and
            all(split.start <= t < split.end - embargo_seconds for t in target_times))


def assessment_origin_admitted(*, origin_anchor, target_times):
    """Fixed final ORIGIN interval, including its explicitly declared H18 tail.

    This checks coordinates only and grants no permission to compute scores.
    """
    expected = forecast_target_times(origin_anchor, HORIZONS_HOURS)
    require(tuple(target_times) == expected, "Protected assessment requires exactly the confirmed H18 targets")
    return (ASSESSMENT_START <= origin_anchor < ASSESSMENT_END
            and expected[-1] <= ASSESSMENT_LAST_TARGET)


def require_scoring_allowed(*, origin_anchor, target_times,
                            protected_assessment_enabled=False,
                            locked_protocol_digest=None, locked_selection_digest=None):
    """Fail closed before a scoring path sees protected values or predictions.

    Digest arguments identify explicit external lock artifacts. This structural
    check is not an authentication mechanism and does not itself freeze a study.
    """
    require(type(protected_assessment_enabled) is bool, "Scoring enablement must be an explicit boolean")
    epoch(origin_anchor)
    target_times = tuple(target_times)
    require(target_times and all(epoch(t) > origin_anchor for t in target_times),
            "Scoring requires explicit future target times")
    if origin_anchor < ASSESSMENT_START:
        require(all(t < ASSESSMENT_START for t in target_times),
                "Development scoring cannot cross into protected labels")
        return True
    require(assessment_origin_admitted(origin_anchor=origin_anchor, target_times=target_times),
            "Scoring origin lies outside the declared protected origin interval")
    require(protected_assessment_enabled, "Protected assessment scoring is disabled by default")
    for name, value in (("protocol", locked_protocol_digest), ("selection", locked_selection_digest)):
        require(isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value),
                "Protected scoring needs a SHA256-bound locked " + name + " artifact")
    return True


@dataclass(frozen=True)
class ScaleCell:
    case_id: str
    source_key: str
    value: float | None
    mask: bool


@dataclass(frozen=True)
class ScaleStatistics:
    mean: float
    scale: float
    count: int
    variance_ddof: int
    training_case_digest: str
    unique_source_digest: str


def fit_train_scaler(cells, *, case_splits: Mapping[str, str], variance_ddof, constant_scale):
    """Fit once per unique training source cell, avoiding overlap multiplicity."""
    require(type(variance_ddof) is int and variance_ddof in (0, 1), "Choose variance convention explicitly")
    require(type(constant_scale) in (int, float) and math.isfinite(constant_scale) and constant_scale > 0,
            "Choose a finite positive constant-series scale explicitly")
    training = sorted(key for key, split in case_splits.items() if split == "train")
    require(training, "Explicit training case membership required")
    unique = {}
    for cell in cells:
        require(cell.case_id in case_splits, "Unknown case identity in scaler input")
        require(type(cell.mask) is bool, "Scaler masks must be actual booleans")
        if case_splits[cell.case_id] != "train" or not cell.mask:
            continue
        require(type(cell.value) in (int, float) and math.isfinite(cell.value), "Admitted training value must be finite")
        require(bool(cell.source_key), "Unique source-cell identity is required")
        value = float(cell.value)
        require(cell.source_key not in unique or unique[cell.source_key] == value, "Conflicting duplicate source cell")
        unique[cell.source_key] = value
    require(unique, "No admitted training source cells")
    keys = sorted(unique)
    mean = math.fsum(unique[k] for k in keys) / len(keys)
    numerator = math.fsum((unique[k] - mean) ** 2 for k in keys)
    variance = numerator / (len(keys) - variance_ddof) if len(keys) > variance_ddof else 0.0
    scale = math.sqrt(variance) if variance > 0 else float(constant_scale)
    require(math.isfinite(mean) and math.isfinite(scale), "Scaler moments exceed finite numeric support")
    fitted_inputs = [dict(source_key=k, value=unique[k], mask=True) for k in keys]
    return ScaleStatistics(mean, scale, len(keys), variance_ddof, digest(training), digest(fitted_inputs))


def transform_masked(values, mask, scaler):
    require(len(values) == len(mask) and all(type(m) is bool for m in mask), "Value/mask dimensions differ")
    require(math.isfinite(scaler.mean) and math.isfinite(scaler.scale) and scaler.scale > 0,
            "A finite positive fitted scale is required")
    result = []
    for value, admitted in zip(values, mask):
        if not admitted:
            result.append(None)
        else:
            require(type(value) in (int, float) and math.isfinite(value), "Admitted transform value must be finite")
            result.append((value - scaler.mean) / scaler.scale)
    return tuple(result)


def encode_masked(values, mask):
    """Network-boundary placeholder only; missing source values remain masked."""
    require(len(values) == len(mask) and all(type(m) is bool for m in mask), "Value/mask dimensions differ")
    require(all(not m or type(v) in (int, float) and math.isfinite(v) for v, m in zip(values, mask)),
            "Every admitted network value must be finite")
    return tuple(float(v) if m else 0.0 for v, m in zip(values, mask)), tuple(mask)


def combine_masks(*masks):
    """Intersect explicitly chosen masks; no feature/quality policy is chosen."""
    require(masks and len({len(mask) for mask in masks}) == 1, "Aligned explicit masks required")
    require(all(type(value) is bool for mask in masks for value in mask), "Masks must be booleans")
    return tuple(all(values) for values in zip(*masks))


@dataclass(frozen=True)
class CohortCase:
    case_id: str
    site: str
    origin: int
    target_times: tuple[int, ...]
    target_mask: tuple[bool, ...]
    feature_admission_digest: str

    def __post_init__(self):
        require(bool(self.case_id) and bool(self.site), "Explicit case/site identity required")
        require(type(self.target_times) is tuple and type(self.target_mask) is tuple, "Immutable cohort coordinates/masks required")
        epoch(self.origin)
        require(self.target_times and all(epoch(t) > self.origin for t in self.target_times)
                and len(set(self.target_times)) == len(self.target_times), "Unique future target times required")
        require(len(self.target_times) == len(self.target_mask) and all(type(m) is bool for m in self.target_mask),
                "Typed per-target masks required")
        require(isinstance(self.feature_admission_digest, str) and len(self.feature_admission_digest) == 64,
                "Hash-bound feature admission required")


def validate_matched_cohorts(cohorts: Mapping[str, tuple[CohortCase, ...]]):
    require(cohorts, "At least one explicit model cohort is required")
    reference = next(iter(cohorts.values()))
    require(reference and len({c.case_id for c in reference}) == len(reference), "Unique nonempty case identities required")
    for case in reference:
        require(len(case.target_times) == len(case.target_mask) and all(type(m) is bool for m in case.target_mask),
                "Exact per-target masks required")
    for model, cases in cohorts.items():
        require(cases == reference, "Model changed case order, source support or target masks: " + model)
    return digest([dict(case_id=c.case_id, site=c.site, origin=c.origin, target_times=c.target_times,
                        target_mask=c.target_mask, feature_admission_digest=c.feature_admission_digest) for c in reference])
