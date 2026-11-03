"""Proposed development scoring/aggregation and a disabled-by-default runner.

No policy adoption, fitting, protected assessment, network or event uncertainty
is implemented here. Pure functions are tested on synthetic outcomes only.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
import time

import numpy as np

from .contract import ARCHITECTURES, GAUGE_IDS
from .evaluation import LabelSeries, PredictionSeries, TargetKey, ScoringPermit, score_metrics
from .reference import ASSESSMENT_START, digest, require

LEADS = tuple(range(1, 19))
INPUT_KEYS = ("history", "history_mask", "weather", "weather_mask", "state", "state_mask")
METRICS = ("nse", "physical_mse", "rmse", "mae", "bias", "negative_frequency")


@dataclass(frozen=True)
class ScoringPolicy:
    minimum_distinct_target_hours: int = 168
    minimum_distinct_utc_dates: int = 28
    expected_cv_folds: tuple[str, ...] = ("cv_1", "cv_2", "cv_3")
    final_seeds: tuple[int, ...] = (17, 29, 43)
    status: str = "PROPOSED_NOT_FROZEN"

    def __post_init__(self):
        require(type(self.minimum_distinct_target_hours) is int and self.minimum_distinct_target_hours >= 2
                and type(self.minimum_distinct_utc_dates) is int and self.minimum_distinct_utc_dates >= 1,
                "Explicit positive primary support gates required")
        require(self.expected_cv_folds == ("cv_1", "cv_2", "cv_3") and self.final_seeds == (17, 29, 43),
                "The proposed complete three-fold/three-seed rule is explicit")
        require(self.status == "PROPOSED_NOT_FROZEN", "This component cannot adopt a scientific protocol")

    @property
    def sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class ScoreIdentity:
    architecture: str
    configuration_id: str
    gauge_id: str
    feature_schema: str
    history_hours: int
    fold_id: str
    seed: int

    def __post_init__(self):
        require(self.architecture in ARCHITECTURES and self.gauge_id in GAUGE_IDS and self.history_hours in (24, 168),
                "Prepared architecture/gauge/history identity required")
        require(self.feature_schema in ("FULL_HRRR_H2_W4_S2_V1", "Q_ONLY_H2_HORIZON1_S0_V1"), "Explicit feature schema required")
        require(type(self.configuration_id) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", self.configuration_id), "Stable configuration ID required")
        require(self.fold_id in ("cv_1", "cv_2", "cv_3", "final_development") and type(self.seed) is int and self.seed in (17, 29, 43),
                "Explicit proposed fold and seed identity required")


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _hour(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
            and stamp.minute == stamp.second == stamp.microsecond == 0, "Exact UTC hour required")
    return int(stamp.timestamp())


def validate_score_coordinates(*, case_ids, origins, forecast_time, gauge_id):
    """Coordinates and protected boundaries are checked before any outcomes."""
    require(gauge_id in GAUGE_IDS, "Confirmed gauge required")
    origin, future, ids = np.asarray(origins), np.asarray(forecast_time), np.asarray(case_ids)
    require(origin.ndim == 1 and 0 <= len(origin) <= 20000 and origin.dtype.kind in "iu"
            and future.dtype.kind in "iu" and future.shape == (len(origin), 18)
            and ids.shape == origin.shape and ids.dtype.kind in "US", "Bounded H18 score coordinates required")
    require(np.all(origin >= 0) and np.all(origin < ASSESSMENT_START) and np.all(origin % 3600 == 0)
            and (len(origin) < 2 or np.all(np.diff(origin.astype(np.int64)) > 0)), "Strict chronological development origins required")
    require(np.array_equal(future, origin[:, None] + np.arange(1, 19)[None, :] * 3600)
            and np.all(future < ASSESSMENT_START), "Protected or crossing score coordinates are forbidden")
    strings = tuple(x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in ids.tolist())
    require(strings == tuple(gauge_id + ":" + datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") for t in origin),
            "Score case IDs must preserve exact gauge/origin coordinates")
    return strings, origin, future


def score_physical_h18(*, case_ids, origins, forecast_time, labels, label_mask, predictions, identity,
                       input_cohort_digest, label_policy_digest, label_source_digest,
                       prediction_units="m3/s", label_units="m3/s", prediction_mask=None, policy=ScoringPolicy()):
    require(isinstance(identity, ScoreIdentity) and isinstance(policy, ScoringPolicy), "Explicit score identity and proposed policy required")
    ids, origin, future = validate_score_coordinates(case_ids=case_ids, origins=origins, forecast_time=forecast_time, gauge_id=identity.gauge_id)
    require(all(_sha(value) for value in (input_cohort_digest, label_policy_digest, label_source_digest)), "Cohort and label provenance required")
    require(prediction_units == label_units == "m3/s", "Physical discharge in m3/s is required; standardized losses are different")
    truth, mask, prediction = np.asarray(labels), np.asarray(label_mask), np.asarray(predictions)
    shape = (len(ids), 18)
    require(truth.shape == mask.shape == prediction.shape == shape and truth.dtype.kind == prediction.dtype.kind == "f"
            and mask.dtype == np.dtype(bool), "Physical values and label-mask geometry differ")
    present = np.ones(shape, bool) if prediction_mask is None else np.asarray(prediction_mask)
    require(present.shape == shape and present.dtype == np.dtype(bool) and present.all()
            and np.isfinite(prediction).all(), "Missing/nonfinite requested predictions fail the run; no prediction-driven intersection")
    require(np.isfinite(truth[mask]).all() and (truth[mask] >= 0).all(), "Admitted physical discharge labels must be finite and nonnegative")
    leads, reasons = [], []
    for column, lead in enumerate(LEADS):
        keys = tuple(TargetKey(f"{case}:H{lead:02d}", identity.gauge_id, int(c), int(t), int(c))
                     for case, c, t in zip(ids, origin, future[:, column]))
        label_series = LabelSeries(keys, tuple(truth[:, column].tolist()), tuple(mask[:, column].tolist()), label_policy_digest, label_source_digest)
        predicted = PredictionSeries(identity.architecture, keys, tuple(prediction[:, column].tolist()),
                                     (True,) * len(ids), (True,) * len(ids), input_cohort_digest)
        metrics = score_metrics(label_series, predicted, permit=ScoringPermit())
        selected = mask[:, column]
        target_hours = {int(value) for value in future[selected, column]}
        dates = {datetime.fromtimestamp(value, timezone.utc).date().isoformat() for value in target_hours}
        primary_reasons = []
        if len(target_hours) < policy.minimum_distinct_target_hours: primary_reasons.append("TOO_FEW_DISTINCT_TARGET_HOURS")
        if len(dates) < policy.minimum_distinct_utc_dates: primary_reasons.append("TOO_FEW_DISTINCT_UTC_DATES")
        if metrics["nse"] is None: primary_reasons.append(metrics["undefined"].get("nse", "UNDEFINED_NSE"))
        n = metrics["n"]
        mse = math.fsum((float(p) - float(y)) ** 2 for p, y in zip(prediction[selected, column], truth[selected, column])) / n if n else None
        row = dict(lead_hours=lead, **metrics, physical_mse=mse,
                   negative_frequency=metrics["negative_predictions_scored"] / n if n else None,
                   distinct_target_hours=len(target_hours), distinct_utc_dates=len(dates),
                   primary_eligible=not primary_reasons, primary_ineligibility=primary_reasons,
                   units={"rmse": "m3/s", "mae": "m3/s", "bias": "m3/s", "physical_mse": "(m3/s)^2", "nse": "dimensionless", "negative_frequency": "fraction"})
        require(mse is None or math.isfinite(mse), "Physical MSE overflowed")
        leads.append(row)
        if primary_reasons: reasons.append(dict(lead_hours=lead, reasons=primary_reasons))
    primary = math.fsum(row["nse"] / 18 for row in leads) if not reasons else None
    return dict(schema="hydra-development-h18-score/v1", status="PRIMARY_DEFINED" if primary is not None else "PRIMARY_UNAVAILABLE",
                identity=asdict(identity), policy=asdict(policy), policy_sha256=policy.sha256,
                input_cohort_digest=input_cohort_digest, label_policy_digest=label_policy_digest, label_source_digest=label_source_digest,
                case_ids_sha256=digest(list(ids)), primary_metric="EQUAL_LEAD_PHYSICAL_NSE", primary_score=primary,
                first_origin_epoch=int(origin[0]) if len(origin) else None,
                last_target_epoch=int(future[-1, -1]) if len(origin) else None,
                primary_unavailable_reasons=reasons, leads=leads, negative_predictions_clipped=False,
                missing_prediction_policy="REJECT_REQUESTED_MISSING_OR_NONFINITE", undefined_panels_dropped=False,
                support_qualification="Predeclared computational/reporting gate only; not independence, statistical power or event uncertainty",
                loss_distinction="Checkpoint/training standardized-Q MSE is not physical MSE or the primary NSE criterion",
                policy_status="PROPOSED_NOT_FROZEN", scientific_policy_adopted=False, protected_assessment_scores=0)


def _scope(identity, omit):
    return {key: value for key, value in identity.items() if key not in omit}


def _validate_panel(panel, policy):
    require(panel.get("schema") == "hydra-development-h18-score/v1" and panel.get("policy_sha256") == policy.sha256
            and panel.get("primary_metric") == "EQUAL_LEAD_PHYSICAL_NSE" and len(panel.get("leads", [])) == 18
            and tuple(row["lead_hours"] for row in panel["leads"]) == LEADS, "Exact H18 score panel and common policy required")
    ScoreIdentity(**panel["identity"])
    for row in panel["leads"]:
        require(type(row["n"]) is int and 0 <= row["n"] <= row["n_requested"]
                and row["distinct_target_hours"] == row["n"]
                and (row["distinct_utc_dates"] == 0 if row["n"] == 0 else 1 <= row["distinct_utc_dates"] <= row["n"]),
                "Score support accounting differs")
        require(all(row[name] is None or type(row[name]) in (int, float) and math.isfinite(row[name]) for name in METRICS), "Finite metric values or explicit undefined values required")
        eligible = row["n"] >= policy.minimum_distinct_target_hours and row["distinct_utc_dates"] >= policy.minimum_distinct_utc_dates and row["nse"] is not None
        require(row["primary_eligible"] is eligible, "Primary support qualification differs")
    expected = math.fsum(row["nse"] / 18 for row in panel["leads"]) if all(row["primary_eligible"] for row in panel["leads"]) else None
    require(panel["primary_score"] == expected, "Primary score differs from all18 equally weighted qualified leads")


def _metric_means(panels):
    result = []
    for lead in LEADS:
        rows = [panel["leads"][lead - 1] for panel in panels]
        result.append(dict(lead_hours=lead, **{name: math.fsum(row[name] / len(rows) for row in rows)
            if all(row[name] is not None and math.isfinite(row[name]) for row in rows) else None for name in METRICS}))
    return result


def aggregate_cv_folds(panels, *, policy=ScoringPolicy()):
    require(isinstance(panels, dict) and set(panels) == set(policy.expected_cv_folds), "All three CV folds are required; no omission")
    ordered = [panels[name] for name in policy.expected_cv_folds]
    for name, panel in zip(policy.expected_cv_folds, ordered):
        _validate_panel(panel, policy)
        require(panel["identity"]["fold_id"] == name and panel["identity"]["seed"] == 17, "CV fold identity/declared seed differs")
    common = _scope(ordered[0]["identity"], {"fold_id"})
    require(all(_scope(p["identity"], {"fold_id"}) == common for p in ordered), "CV candidate/gauge identities differ")
    require(all(a["last_target_epoch"] is None or b["first_origin_epoch"] is None or a["last_target_epoch"] < b["first_origin_epoch"]
                for a, b in zip(ordered, ordered[1:])), "CV scoring windows overlap or are out of temporal order")
    available = all(p["primary_score"] is not None for p in ordered)
    value = math.fsum(p["primary_score"] / 3 for p in ordered) if available else None
    return dict(schema="hydra-equal-fold-score/v1", identity=common, policy_sha256=policy.sha256,
                primary_metric="EQUAL_FOLD_MEAN_OF_EQUAL_LEAD_PHYSICAL_NSE", primary_score=value,
                status="PRIMARY_DEFINED" if available else "PRIMARY_UNAVAILABLE", folds=list(policy.expected_cv_folds),
                undefined_folds=[name for name in policy.expected_cv_folds if panels[name]["primary_score"] is None],
                per_lead_mean_metrics=_metric_means(ordered), aggregation="ARITHMETIC_MEAN_OF_THREE_FOLD_METRICS_NOT_POOLED_ROWS_OR_SSE",
                fold_evidence={name: dict(case_ids_sha256=panels[name]["case_ids_sha256"], input_cohort_digest=panels[name]["input_cohort_digest"],
                    label_policy_digest=panels[name]["label_policy_digest"], label_source_digest=panels[name]["label_source_digest"],
                    scored_sample_digests=[row["scored_sample_digest"] for row in panels[name]["leads"]],
                    label_series_digests=[row["label_series_digest"] for row in panels[name]["leads"]]) for name in policy.expected_cv_folds},
                fold_scores={name: panels[name]["primary_score"] for name in policy.expected_cv_folds},
                omitted_folds=False, policy_status="PROPOSED_NOT_FROZEN")


def aggregate_final_seeds(panels, *, policy=ScoringPolicy()):
    require(isinstance(panels, dict) and set(panels) == set(policy.final_seeds), "All three final seeds must be retained; no best-seed selection")
    ordered = [panels[seed] for seed in policy.final_seeds]
    for seed, panel in zip(policy.final_seeds, ordered):
        _validate_panel(panel, policy)
        require(panel["identity"]["seed"] == seed and panel["identity"]["fold_id"] == "final_development", "Final seed/fold identity differs")
    common = _scope(ordered[0]["identity"], {"seed"})
    require(all(_scope(p["identity"], {"seed"}) == common for p in ordered), "Final candidate/gauge identities differ")
    fields = ("case_ids_sha256", "input_cohort_digest", "label_policy_digest", "label_source_digest")
    require(all(all(p[key] == ordered[0][key] for key in fields) and
                [r["label_series_digest"] for r in p["leads"]] == [r["label_series_digest"] for r in ordered[0]["leads"]]
                for p in ordered), "Final seeds must score the same labels/cohort")
    available = all(p["primary_score"] is not None for p in ordered)
    return dict(schema="hydra-all-final-seeds-score/v1", identity=common, policy_sha256=policy.sha256,
                status="PRIMARY_DEFINED" if available else "PRIMARY_UNAVAILABLE",
                primary_metric="MEAN_OF_ALL_THREE_SEED_EQUAL_LEAD_PHYSICAL_NSE",
                primary_score=math.fsum(p["primary_score"] / 3 for p in ordered) if available else None,
                seeds=list(policy.final_seeds), seed_scores={str(seed): panels[seed]["primary_score"] for seed in policy.final_seeds},
                undefined_seeds=[seed for seed in policy.final_seeds if panels[seed]["primary_score"] is None],
                per_lead_mean_metrics=_metric_means(ordered), seed_reports={str(seed): panels[seed] for seed in policy.final_seeds},
                aggregation="MEAN_OF_METRICS_NOT_METRIC_OF_ENSEMBLE_PREDICTIONS", seed_selection_performed=False,
                omitted_seeds=False, policy_status="PROPOSED_NOT_FROZEN")


def select_cv_configuration(results, *, declared_configuration_order):
    require(type(declared_configuration_order) is tuple and declared_configuration_order
            and len(set(declared_configuration_order)) == len(declared_configuration_order)
            and set(results) == set(declared_configuration_order), "Exact preregistered configuration order required")
    ordered = [results[name] for name in declared_configuration_order]
    for name, result in zip(declared_configuration_order, ordered):
        require(result.get("schema") == "hydra-equal-fold-score/v1" and result.get("identity", {}).get("configuration_id") == name
                and result.get("status") == "PRIMARY_DEFINED" and type(result.get("primary_score")) in (int, float)
                and math.isfinite(result["primary_score"]), "Every candidate requires a complete defined CV primary score")
        require(result.get("folds") == ["cv_1", "cv_2", "cv_3"] and set(result.get("fold_scores", {})) == set(result["folds"])
                and all(type(v) in (int, float) and math.isfinite(v) for v in result["fold_scores"].values())
                and result["primary_score"] == math.fsum(result["fold_scores"][fold] / 3 for fold in result["folds"]),
                "Configuration primary score must retain every fold equally")
    first = ordered[0]
    require(all(_scope(r["identity"], {"configuration_id"}) == _scope(first["identity"], {"configuration_id"})
                and r["policy_sha256"] == first["policy_sha256"] and r["fold_evidence"] == first["fold_evidence"] for r in ordered),
            "Configuration selection requires the same family/gauge/mode and exact per-fold cohorts")
    best = max(r["primary_score"] for r in ordered)
    tied = [name for name in declared_configuration_order if results[name]["primary_score"] == best]
    return dict(selected_configuration=tied[0], exactly_tied_configurations=tied, primary_score=best,
                declared_configuration_order=list(declared_configuration_order), order_sha256=digest(list(declared_configuration_order)),
                tie_rule="EXACT_FLOAT_EQUALITY_THEN_PREREGISTERED_ORDER_NO_ROUNDING_OR_SECONDARY_METRICS",
                seed_selection_performed=False, additional_trials_authorized=False, policy_status="PROPOSED_NOT_FROZEN")


def _load_best_predictor(path, **kwargs):
    from .inference import BestCheckpointPredictor
    return BestCheckpointPredictor.load(path, **kwargs)


def run_development_score(*, arrays, outer_indices, outer_start_utc, outer_end_utc, checkpoint_path,
                          checkpoint_sha256, training_source_identity, training_cohort_digest, gradient_fit_case_ids,
                          inference_source_identity, inference_cohort_digest, identity, label_policy_digest, label_source_digest,
                          policy=ScoringPolicy(), execution_enabled=False, approved_plan_sha256=None,
                          batch_size=128, max_wall_seconds=300, max_combined_bytes=4 * 1024**3):
    """Plan by default; explicit development execution uses best-state inference.

    The flag/digest bind a caller's request; they do not authenticate approval or
    freeze this proposed policy. Only synthetic fake-predictor tests run here.
    """
    require(isinstance(identity, ScoreIdentity) and isinstance(policy, ScoringPolicy) and type(execution_enabled) is bool,
            "Explicit identity/policy and execution choice required")
    require(type(batch_size) is int and 1 <= batch_size <= 128 and type(max_wall_seconds) is int and 0 < max_wall_seconds <= 600
            and type(max_combined_bytes) is int and 0 < max_combined_bytes <= 4 * 1024**3, "Bounded development score resources required")
    require(all(_sha(x) for x in (checkpoint_sha256, training_source_identity, training_cohort_digest, inference_source_identity,
                                 inference_cohort_digest, label_policy_digest, label_source_digest)), "Explicit checkpoint/training/inference/label identities required")
    indices = tuple(outer_indices)
    require(indices and len(indices) <= 20000 and all(type(i) is int and i >= 0 for i in indices)
            and tuple(sorted(set(indices))) == indices, "Explicit ordered outer-role indices required")
    origin = np.asarray(arrays["origin_time"])
    all_future = np.asarray(arrays["target_time"])
    require(origin.ndim == 1 and 0 < len(origin) <= 200000 and origin.dtype.kind in "iu"
            and all_future.dtype.kind in "iu" and all_future.shape == (len(origin), 18), "Bounded development source clocks required")
    require(np.all((origin >= 0) & (origin < ASSESSMENT_START) & (origin % 3600 == 0))
            and np.array_equal(all_future, origin[:, None] + np.arange(1, 19)[None, :] * 3600)
            and np.all(all_future < ASSESSMENT_START),
            "Protected, crossing or inconsistent clocks anywhere in the source bundle are forbidden")
    require(max(indices) < len(origin), "Outer index outside source bundle")
    origin, future, case_ids = origin[list(indices)], all_future[list(indices)], np.asarray(arrays["case_id"])[list(indices)]
    validate_score_coordinates(case_ids=case_ids, origins=origin, forecast_time=future, gauge_id=identity.gauge_id)
    start, end = _hour(outer_start_utc), _hour(outer_end_utc)
    require(start < end <= ASSESSMENT_START and np.all((origin >= start) & (origin < end)) and np.all(future < end), "Outer role window differs or crosses assessment")
    require(type(gradient_fit_case_ids) is tuple and gradient_fit_case_ids and len(set(gradient_fit_case_ids)) == len(gradient_fit_case_ids),
            "Explicit gradient-fit case identities required")
    require(all(type(case) is str and case.startswith(identity.gauge_id + ":")
                and _hour(case[len(identity.gauge_id) + 1:]) + 18 * 3600 < start for case in gradient_fit_case_ids),
            "Gradient-fit labels must precede the requested outer scoring window")
    require(set(case_ids.tolist()).isdisjoint(gradient_fit_case_ids), "Outer scoring cases overlap gradient-fit cases")
    plan = dict(schema="hydra-development-score-plan/v1", identity=asdict(identity), policy=asdict(policy), policy_sha256=policy.sha256,
                checkpoint_sha256=checkpoint_sha256, selected_state_required="best_model", training_source_identity=training_source_identity,
                training_cohort_digest=training_cohort_digest, gradient_fit_case_ids_sha256=digest(list(gradient_fit_case_ids)),
                inference_source_identity=inference_source_identity, inference_cohort_digest=inference_cohort_digest,
                outer_window={"start_utc": outer_start_utc, "end_utc": outer_end_utc}, outer_case_ids_sha256=digest(case_ids.tolist()),
                label_policy_digest=label_policy_digest, label_source_digest=label_source_digest, case_count=len(indices),
                batch_size=batch_size, max_wall_seconds=max_wall_seconds, max_combined_bytes=max_combined_bytes,
                implementation_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ("scoring.py", "inference.py", "evaluation.py", "training.py")},
                training_role_admission="Caller must bind the verified training bundle's separate fit/inner-stop refit-availability proof",
                policy_status="PROPOSED_NOT_FROZEN", protected_assessment_enabled=False)
    plan["plan_sha256"] = digest(plan)
    if not execution_enabled:
        return {**plan, "status": "PLANNED_NOT_EXECUTED", "model_loads": 0, "labels_read": False}
    require(approved_plan_sha256 == plan["plan_sha256"], "Exact caller-approved development score plan required")
    import psutil
    from .inference import input_only_view
    started, process, peak = time.monotonic(), psutil.Process(), 0
    predictor = None
    def guard():
        nonlocal peak
        driver = predictor.torch.mps.driver_allocated_memory() if predictor is not None and predictor.receipt.get("device") == "mps" else 0
        peak = max(peak, process.memory_info().rss + driver)
        require(peak <= max_combined_bytes and time.monotonic() - started < max_wall_seconds, "Development scoring resource guard reached")
    guard()
    predictor = _load_best_predictor(checkpoint_path, expected_sha256=checkpoint_sha256, expected_source_identity=training_source_identity,
        expected_feature_schema=identity.feature_schema, expected_training_cohort_admission_digest=training_cohort_digest,
        expected_gauge_id=identity.gauge_id, gradient_fit_case_ids=gradient_fit_case_ids)
    required = dict(checkpoint_sha256=checkpoint_sha256, training_source_identity=training_source_identity,
                    training_cohort_admission_digest=training_cohort_digest, gauge_id=identity.gauge_id, feature_schema=identity.feature_schema,
                    architecture=identity.architecture, seed=identity.seed, history_hours=identity.history_hours, prediction_units="m3/s",
                    selected_state="best_model", optimizer_constructed=False, scaler_fitted=False, last_state_used=False)
    require(all(predictor.receipt.get(k) == v for k, v in required.items()) and predictor.receipt.get("device") in ("cpu", "mps")
            and type(predictor.receipt.get("selected_best_epoch")) is int and 0 < predictor.receipt["selected_best_epoch"] <= predictor.receipt["completed_epochs"],
            "Scoring requires the identified saved best state and physical inverse transform")
    required.update({name: predictor.receipt[name] for name in ("selected_best_epoch", "completed_epochs", "device")})
    predictions, receipts = [], []
    for offset in range(0, len(indices), batch_size):
        guard(); take = list(indices[offset:offset + batch_size])
        batch = {name: np.asarray(arrays[name])[take] for name in INPUT_KEYS + ("origin_time", "history_time", "target_time", "site_id", "case_id")}
        for name in ("feature_schema", "cohort_admission_digest"): batch[name] = arrays[name]
        inputs = input_only_view(batch, inference_source_identity=inference_source_identity)
        output = predictor.predict(inputs, expected_inference_source_identity=inference_source_identity,
                                   expected_inference_cohort_admission_digest=inference_cohort_digest)
        evidence = output["receipt"]
        require(all(evidence.get(k) == v for k, v in required.items())
                and evidence.get("inference_source_identity") == inference_source_identity
                and evidence.get("inference_cohort_admission_digest") == inference_cohort_digest
                and evidence.get("target_values_read") is False and evidence.get("target_masks_read") is False
                and evidence.get("optimizer_updates") == 0 and evidence.get("posthoc_clipping") is False,
                "Inference receipt changed state/source/labels/units")
        require(np.array_equal(output["case_id"], batch["case_id"]) and np.array_equal(output["origin_time"], batch["origin_time"])
                and np.array_equal(output["forecast_time"], batch["target_time"]) and np.array_equal(output["site_id"], batch["site_id"])
                and np.array_equal(output["history_time"], batch["history_time"]),
                "Inference changed exact scoring coordinates")
        value = np.asarray(output["prediction_q"])
        require(value.shape == (len(take), 18) and value.dtype.kind == "f" and np.isfinite(value).all(), "Missing/nonfinite requested physical predictions")
        predictions.append(value.copy()); receipts.append(evidence); guard()
    # Labels are first accessed after all prediction calls have completed.
    truth, masks = np.asarray(arrays["target"])[list(indices)], np.asarray(arrays["target_mask"])[list(indices)]
    result = score_physical_h18(case_ids=case_ids, origins=origin, forecast_time=future, labels=truth, label_mask=masks,
        predictions=np.concatenate(predictions), identity=identity, input_cohort_digest=inference_cohort_digest,
        label_policy_digest=label_policy_digest, label_source_digest=label_source_digest, policy=policy)
    guard()
    return {**result, "execution": dict(plan_sha256=plan["plan_sha256"], checkpoint_sha256=checkpoint_sha256,
            selected_state="best_model", selected_best_epoch=predictor.receipt["selected_best_epoch"],
            checkpoint_selection_loss="SAVED_INNER_STOPPING_STANDARDIZED_Q_MSE", inference_batches=len(receipts),
            inference_receipts=receipts, labels_accessed_only_after_predictions=True, optimizer_updates=0,
            elapsed_seconds=time.monotonic() - started, observed_combined_peak_bytes=peak, protected_assessment_scores=0,
            resource_scope="Cooperative between bounded batches; not OS isolation", policy_adopted=False)}
