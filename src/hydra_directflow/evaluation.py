"""Pure guarded metrics and exact matched-case comparisons; no data readers.

This module selects no primary metric and performs no bootstrap, model fitting,
or protected scoring by default. Comparators never determine training membership.
"""
from dataclasses import dataclass
from collections import Counter
import math
from numbers import Real
from typing import Mapping

from .contract import GAUGE_IDS, HORIZONS_HOURS
from .reference import digest, epoch, forecast_target_times, require, require_scoring_allowed

HOUR = 3600


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class TargetKey:
    """Identity of ONE gauge/origin/target cell, not an entire H18 forecast."""
    sample_id: str
    site: str
    origin: int
    target_time: int
    cutoff: int

    def __post_init__(self):
        require(isinstance(self.sample_id, str) and self.sample_id, "Explicit per-target sample ID required")
        require(self.site in GAUGE_IDS, "Unknown evaluation gauge")
        require(epoch(self.origin) % HOUR == 0 and epoch(self.cutoff) == self.origin,
                "Declared cutoff C must equal the exact hourly origin")
        require(epoch(self.target_time) in forecast_target_times(self.origin, HORIZONS_HOURS),
                "Target clock must be a confirmed origin-relative H1 through H18")


def _keys(keys):
    require(type(keys) is tuple and all(isinstance(k, TargetKey) for k in keys), "Immutable target keys required")
    require(len({k.sample_id for k in keys}) == len(keys), "Duplicate sample IDs")
    require(len({(k.site, k.origin, k.target_time, k.cutoff) for k in keys}) == len(keys),
            "Duplicate exact gauge/origin/target/cutoff coordinates")


def _aligned(values, masks, size):
    require(type(values) is tuple and len(values) == size, "Immutable aligned values required")
    for mask in masks:
        require(type(mask) is tuple and len(mask) == size and all(type(m) is bool for m in mask),
                "Immutable aligned boolean masks required")


@dataclass(frozen=True)
class LabelSeries:
    keys: tuple[TargetKey, ...]
    values: tuple[float | None, ...]
    label_mask: tuple[bool, ...]
    label_policy_digest: str
    source_digest: str

    def __post_init__(self):
        _keys(self.keys)
        _aligned(self.values, (self.label_mask,), len(self.keys))
        require(_sha(self.label_policy_digest) and _sha(self.source_digest),
                "Labels require explicit SHA256 identities for their quality policy and source")


@dataclass(frozen=True)
class PredictionSeries:
    name: str
    keys: tuple[TargetKey, ...]
    values: tuple[float | None, ...]
    prediction_mask: tuple[bool, ...]
    support_mask: tuple[bool, ...]
    input_cohort_digest: str | None
    role: str = "architecture"
    source_ids: tuple[str | None, ...] = ()
    support_reasons: tuple[str | None, ...] = ()

    def __post_init__(self):
        require(isinstance(self.name, str) and self.name, "Explicit model/comparator name required")
        require(self.role in ("architecture", "external_nwm"), "Unknown prediction role")
        require(_sha(self.input_cohort_digest) if self.role == "architecture" else self.input_cohort_digest is None,
                "Architectures require a SHA-bound input cohort; NWM cannot declare training membership")
        _keys(self.keys)
        _aligned(self.values, (self.prediction_mask, self.support_mask), len(self.keys))
        for metadata in (self.source_ids, self.support_reasons):
            require(type(metadata) is tuple and (not metadata or len(metadata) == len(self.keys))
                    and all(v is None or isinstance(v, str) for v in metadata), "Source metadata must align with keys")


@dataclass(frozen=True)
class ScoringPermit:
    """Structural gate only; digest strings do not authenticate or grant approval."""
    protected_assessment_enabled: bool = False
    locked_protocol_digest: str | None = None
    locked_selection_digest: str | None = None

    def check(self, keys):
        _keys(keys)
        require(type(self.protected_assessment_enabled) is bool, "Explicit boolean scoring enablement required")
        for origin in sorted({k.origin for k in keys}):
            # Guard the entire H18 origin even when reporting a subset of leads.
            require_scoring_allowed(origin_anchor=origin,
                                    target_times=forecast_target_times(origin, HORIZONS_HOURS),
                                    protected_assessment_enabled=self.protected_assessment_enabled,
                                    locked_protocol_digest=self.locked_protocol_digest,
                                    locked_selection_digest=self.locked_selection_digest)


def _finite(value, kind):
    require(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value),
            "Admitted " + kind + " must be finite numeric values")
    return float(value)


def score_metrics(labels, prediction, *, cohort_mask=None, permit=ScoringPermit()):
    """Bias means prediction minus observation; all metrics use the SAME mask.

    Exclusion counts overlap intentionally. A missing prediction is distinct
    from unsupported source time and from a missing/rejected reference label.
    """
    require(labels.keys == prediction.keys, "Prediction keys/order differ from labels")
    permit.check(labels.keys)  # Before inspecting any numerical outcomes.
    size = len(labels.keys)
    if cohort_mask is None:
        cohort_mask = (True,) * size
    _aligned(labels.values, (cohort_mask,), size)
    observed, predicted, admitted_label_values = [], [], []
    selected = []
    for index in range(size):
        # Invalid admitted source values cannot hide behind another exclusion.
        y = _finite(labels.values[index], "label") if labels.label_mask[index] else None
        p = _finite(prediction.values[index], "prediction") if prediction.prediction_mask[index] else None
        admitted_label_values.append(y)
        admitted = (labels.label_mask[index] and prediction.prediction_mask[index]
                    and prediction.support_mask[index] and cohort_mask[index])
        if admitted:
            observed.append(y)
            predicted.append(p)
            selected.append(vars(labels.keys[index]))
    n = len(observed)
    result = {"n": n, "n_requested": size, "rmse": None, "mae": None, "bias": None, "nse": None,
              "undefined": {}, "scored_sample_digest": digest(selected),
              "negative_predictions_scored": sum(p < 0 for p in predicted),
              "label_policy_digest": labels.label_policy_digest, "label_source_digest": labels.source_digest,
              "label_series_digest": digest({"keys": [vars(k) for k in labels.keys],
                                             "admitted_values": admitted_label_values,
                                             "label_mask": labels.label_mask,
                                             "quality_policy": labels.label_policy_digest,
                                             "source": labels.source_digest}),
              "exclusion_counts_overlap": {
                  "label_mask_false": sum(not x for x in labels.label_mask),
                  "prediction_mask_false": sum(not x for x in prediction.prediction_mask),
                  "source_support_false": sum(not x for x in prediction.support_mask),
                  "cohort_mask_false": sum(not x for x in cohort_mask)}}
    if n == 0:
        result["undefined"] = {name: "ZERO_SUPPORT" for name in ("rmse", "mae", "bias", "nse")}
        return result
    errors = [p - y for y, p in zip(observed, predicted)]
    sse = math.fsum(e * e for e in errors)
    constant_reference = all(y == observed[0] for y in observed)
    mean = math.fsum(y / n for y in observed)
    denominator = math.fsum((y - mean) ** 2 for y in observed)
    result.update(rmse=math.sqrt(sse / n), mae=math.fsum(abs(e) for e in errors) / n,
                  bias=math.fsum(errors) / n)
    if constant_reference:
        result["undefined"]["nse"] = "CONSTANT_REFERENCE"
    elif denominator == 0:
        result["undefined"]["nse"] = "REFERENCE_VARIANCE_NUMERIC_UNDERFLOW"
    else:
        result["nse"] = 1 - sse / denominator
    require(all(value is None or math.isfinite(value) for name, value in result.items()
                if name in ("rmse", "mae", "bias", "nse")), "Metric arithmetic exceeded finite numeric range")
    return result


def compare_architectures(labels, predictions: Mapping[str, PredictionSeries], *,
                          missing_prediction_policy="reject", permit=ScoringPermit()):
    """Require identical declared input cohorts and identical scored samples.

    Default rejects failed/missing architecture predictions on admitted labels.
    An explicitly chosen shared intersection reports the removed support count.
    """
    permit.check(labels.keys)
    require(bool(predictions), "At least one architecture series required")
    require(missing_prediction_policy in ("reject", "shared_intersection"), "Explicit missing-prediction policy required")
    cohort_ids = set()
    for name, prediction in predictions.items():
        require(name == prediction.name and name != "NWM" and prediction.role == "architecture", "Architecture identity/role differs")
        require(prediction.keys == labels.keys, "Architecture changed exact sample keys/order")
        cohort_ids.add(prediction.input_cohort_digest)
    require(len(cohort_ids) == 1, "Architecture input cohorts differ")
    common = tuple(label and all(p.prediction_mask[i] and p.support_mask[i] for p in predictions.values())
                   for i, label in enumerate(labels.label_mask))
    removed = sum(labels.label_mask) - sum(common)
    require(missing_prediction_policy != "reject" or removed == 0,
            "Architecture missingness changes the declared scoring cohort; choose shared_intersection explicitly")
    metrics = {name: score_metrics(labels, p, cohort_mask=common, permit=permit) for name, p in predictions.items()}
    require(len({m["scored_sample_digest"] for m in metrics.values()}) == 1, "Scored architecture cohorts differ")
    return {"input_cohort_digest": next(iter(cohort_ids)), "requested_samples": len(labels.keys),
            "label_supported_samples": sum(labels.label_mask), "shared_scored_samples": sum(common),
            "removed_admitted_labels": removed, "missing_prediction_policy": missing_prediction_policy,
            "scored_sample_digest": next(iter(metrics.values()))["scored_sample_digest"], "metrics": metrics}


@dataclass(frozen=True)
class NwmShortRangeRun:
    """Actual native NWM short-range coordinates, never origin-relabelled leads."""
    run_id: str
    site: str
    reference: int
    available_at: int
    valid_times: tuple[int, ...]
    values: tuple[float | None, ...]
    prediction_mask: tuple[bool, ...]
    source_support: tuple[bool, ...]
    source_digest: str

    def __post_init__(self):
        require(isinstance(self.run_id, str) and self.run_id and self.site in GAUGE_IDS, "Explicit NWM source identity/gauge required")
        require(_sha(self.source_digest), "Explicit NWM source SHA256 identity required")
        require(epoch(self.reference) % HOUR == 0 and epoch(self.available_at) >= self.reference,
                "Invalid native initialization/availability clock")
        require(type(self.valid_times) is tuple and all(type(t) is int for t in self.valid_times)
                and tuple(sorted(set(self.valid_times))) == self.valid_times
                and all(t in forecast_target_times(self.reference, HORIZONS_HOURS) for t in self.valid_times),
                "Native source valid times must be unique actual leads 1 through 18")
        _aligned(self.values, (self.prediction_mask, self.source_support), len(self.valid_times))


def align_nwm_predictions(keys, runs: Mapping[tuple[str, int], NwmShortRangeRun], *, permit=ScoringPermit()):
    """Map by (gauge, declared origin C), then exact original valid timestamp.

    A two-hour-old 18-lead run supports C+1..C+16 only. An unavailable or missing
    run affects comparator support; no independent training case is removed.
    """
    permit.check(keys)
    requested = {(k.site, k.origin) for k in keys}
    require(set(runs) <= requested, "Comparator source map contains unrequested gauge/origin keys")
    by_initialization, by_identity = {}, {}
    for (site, origin), run in runs.items():
        require(run.site == site and run.reference <= origin, "Comparator source gauge or initialization differs from declared C")
        initialization = site, run.reference
        identity = run.run_id, run.source_digest
        require(initialization not in by_initialization or by_initialization[initialization] == identity,
                "Competing NWM initialization vintages require explicit upstream resolution")
        require((site, run.run_id) not in by_identity or by_identity[site, run.run_id] == run,
                "Duplicate NWM run ID has conflicting source records")
        by_initialization[initialization] = identity
        by_identity[site, run.run_id] = run
    values, predictions, support, sources, reasons = [], [], [], [], []
    for key in keys:
        run = runs.get((key.site, key.origin))
        value, present, supported, source, reason = None, False, False, None, "SOURCE_RUN_MISSING"
        if run is not None:
            source = run.run_id + "@sha256:" + run.source_digest
            positions = {valid: i for i, valid in enumerate(run.valid_times)}
            if key.target_time in positions:
                i = positions[key.target_time]
                value, present = run.values[i], run.prediction_mask[i]
                supported = run.available_at <= key.cutoff and run.source_support[i]
                reason = ("SOURCE_NOT_AVAILABLE_AT_C" if run.available_at > key.cutoff else
                          "ORIGINAL_SOURCE_MASK_FALSE" if not run.source_support[i] else None)
            else:
                reason = "TARGET_OUTSIDE_NATIVE_SUPPORT"
        values.append(value)
        predictions.append(present)
        support.append(supported)
        sources.append(source)
        reasons.append(reason)
    return PredictionSeries("NWM", keys, tuple(values), tuple(predictions), tuple(support), None,
                            role="external_nwm", source_ids=tuple(sources), support_reasons=tuple(reasons))


def compare_with_nwm(labels, architectures, runs: Mapping[tuple[str, int], NwmShortRangeRun], *,
                     missing_prediction_policy="reject", permit=ScoringPermit()):
    """Return independent model metrics AND a separately labelled NWM subset.

    No fitting/training cohort is accepted or modified by this API. The input
    cohort digest and independent architecture result survive NWM missingness.
    """
    require(isinstance(runs, Mapping), "NWM comparison requires original run metadata, not predeclared support")
    independent = compare_architectures(labels, architectures, missing_prediction_policy=missing_prediction_policy,
                                       permit=permit)
    comparator = align_nwm_predictions(labels.keys, runs, permit=permit)
    common = tuple(label and comparator.prediction_mask[i] and comparator.support_mask[i]
                   and all(p.prediction_mask[i] and p.support_mask[i] for p in architectures.values())
                   for i, label in enumerate(labels.label_mask))
    metrics = {name: score_metrics(labels, prediction, cohort_mask=common, permit=permit)
               for name, prediction in {**architectures, "NWM": comparator}.items()}
    require(len({m["scored_sample_digest"] for m in metrics.values()}) == 1, "External matched cohorts differ")
    source_metadata = [{"assigned_site": site, "assigned_origin": origin,
                        **{name: getattr(run, name) for name in
                           ("run_id", "source_digest", "site", "reference", "available_at", "valid_times",
                            "prediction_mask", "source_support")}}
                       for (site, origin), run in sorted(runs.items())]
    ages = [(key.cutoff - runs[key.site, key.origin].reference) // HOUR for key in labels.keys
            if (key.site, key.origin) in runs]
    supported_leads = sorted({(key.target_time - runs[key.site, key.origin].reference) // HOUR
                              for key, supported in zip(labels.keys, comparator.support_mask) if supported})
    alignment = {"source_alignment_digest": digest({"keys": [vars(k) for k in labels.keys],
                                                    "source_metadata": source_metadata,
                                                    "source_ids": comparator.source_ids,
                                                    "support_reasons": comparator.support_reasons}),
                 "source_run_count": len({(run.site, run.run_id, run.source_digest) for run in runs.values()}),
                 "native_supported_samples": sum(comparator.support_mask),
                 "prediction_present_samples": sum(comparator.prediction_mask),
                 "source_age_hours_range": [min(ages), max(ages)] if ages else None,
                 "actual_source_leads_supported": supported_leads,
                 "support_reason_counts": dict(Counter(reason or "SOURCE_SUPPORTED" for reason in comparator.support_reasons))}
    return {"independent_architectures": independent,
            "nwm_alignment": alignment,
            "matched_nwm_subset": {"n": sum(common),
                                   "excluded_from_independent_scored_samples": independent["shared_scored_samples"] - sum(common),
                                   "scored_sample_digest": metrics["NWM"]["scored_sample_digest"], "metrics": metrics},
            "independent_input_cohort_unchanged": True,
            "scope": "Matched scoring on explicitly admitted origins; no primary metric freeze, bootstrap, or significance claim"}
