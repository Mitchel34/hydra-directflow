"""Synthetic only: no source reads, real scores, training, or assessment access."""
from dataclasses import replace
import math
import unittest

from hydra_directflow.evaluation import (
    LabelSeries, NwmShortRangeRun, PredictionSeries, ScoringPermit, TargetKey,
    align_nwm_predictions, compare_architectures, compare_with_nwm, score_metrics,
)
from hydra_directflow.reference import ASSESSMENT_START

HOUR = 3600
C = 1629684000  # A development UTC hour; tests use invented numbers only.
SITE = "03161000"
INPUT_DIGEST = "a" * 64
LABEL_POLICY = "b" * 64
LABEL_SOURCE = "c" * 64


def keys(n=3, *, origin=C, site=SITE):
    return tuple(TargetKey(f"{site}:{origin}:h{h}", site, origin, origin + h * HOUR, origin)
                 for h in range(1, n + 1))


def labels(values=(1., 2., 3.), mask=None, *, target_keys=None):
    return LabelSeries(target_keys or keys(len(values)), tuple(values),
                       tuple(mask) if mask is not None else (True,) * len(values), LABEL_POLICY, LABEL_SOURCE)


def model(values=(2., 2., 2.), mask=None, support=None, *, target_keys=None, name="LSTM"):
    n = len(values)
    return PredictionSeries(name, target_keys or keys(n), tuple(values), tuple(mask) if mask is not None else (True,) * n,
                            tuple(support) if support is not None else (True,) * n, INPUT_DIGEST)


def nwm(*, reference=C - 2 * HOUR, available=C, site=SITE, run_id="issued-native-run", digest="d" * 64):
    return NwmShortRangeRun(run_id, site, reference, available,
                            tuple(reference + h * HOUR for h in range(1, 19)),
                            tuple(float(h) for h in range(1, 19)), (True,) * 18, (True,) * 18, digest)


class MetricTests(unittest.TestCase):
    def test_known_metric_values_and_bias_direction(self):
        result = score_metrics(labels(), model())
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["rmse"], math.sqrt(2 / 3))
        self.assertAlmostEqual(result["mae"], 2 / 3)
        self.assertEqual(result["bias"], 0)
        self.assertEqual(result["nse"], 0)
        self.assertEqual(result["undefined"], {})
        shifted = score_metrics(labels(), model((3., 4., 5.)))
        self.assertEqual(shifted["bias"], 2.)
        self.assertEqual(shifted["rmse"], 2.)

    def test_constant_reference_has_undefined_nse_even_for_perfect_prediction(self):
        result = score_metrics(labels((7., 7.)), model((7., 7.)))
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["rmse"], 0)
        self.assertIsNone(result["nse"])
        self.assertEqual(result["undefined"], {"nse": "CONSTANT_REFERENCE"})
        decimal = score_metrics(labels((.1, .1, .1)), model((.1, .1, .1)))
        self.assertIsNone(decimal["nse"])
        self.assertEqual(decimal["undefined"]["nse"], "CONSTANT_REFERENCE")
        tiny = score_metrics(labels((1e-200, 2e-200)), model((1e-200, 2e-200)))
        self.assertIsNone(tiny["nse"])
        self.assertEqual(tiny["undefined"]["nse"], "REFERENCE_VARIANCE_NUMERIC_UNDERFLOW")

    def test_zero_support_is_explicit_not_an_invented_zero_score(self):
        result = score_metrics(labels((None, None), (False, False)), model((float("nan"), None), (False, False)))
        self.assertEqual(result["n"], 0)
        for name in ("rmse", "mae", "bias", "nse"):
            self.assertIsNone(result[name])
            self.assertEqual(result["undefined"][name], "ZERO_SUPPORT")

    def test_label_prediction_and_support_masks_are_distinct(self):
        truth = labels((1., float("nan"), 3., 4.), (True, False, True, True))
        pred = model((2., 900., float("nan"), 900.), (True, True, False, True), (True, True, True, False))
        result = score_metrics(truth, pred)
        self.assertEqual((result["n"], result["rmse"], result["bias"]), (1, 1., 1.))
        self.assertEqual(result["exclusion_counts_overlap"], {"label_mask_false": 1, "prediction_mask_false": 1,
                                                            "source_support_false": 1, "cohort_mask_false": 0})
        # Reversing support changes eligibility; it never changes the label mask.
        reversed_support = replace(pred, support_mask=(False, False, False, True))
        other = score_metrics(truth, reversed_support)
        self.assertEqual((other["n"], other["bias"]), (1, 896.))

    def test_invalid_admitted_values_fail_and_negative_predictions_are_not_clipped(self):
        with self.assertRaisesRegex(ValueError, "prediction.*finite"):
            score_metrics(labels(), model((float("nan"), 2., 3.)))
        with self.assertRaisesRegex(ValueError, "label.*finite"):
            score_metrics(labels((float("inf"), 2., 3.)), model())
        result = score_metrics(labels((1., 2.)), model((-1., 2.)))
        self.assertEqual(result["negative_predictions_scored"], 1)
        self.assertEqual(result["bias"], -1.)

    def test_label_policy_and_source_bindings_are_visible(self):
        result = score_metrics(labels(), model())
        changed = score_metrics(replace(labels(), label_policy_digest="e" * 64), model())
        self.assertEqual(result["label_source_digest"], LABEL_SOURCE)
        self.assertNotEqual(result["label_series_digest"], changed["label_series_digest"])
        moved = tuple(replace(k, origin=k.origin + HOUR, cutoff=k.cutoff + HOUR,
                              target_time=k.target_time + HOUR) for k in keys())
        shifted = score_metrics(labels(target_keys=moved), model(target_keys=moved))
        self.assertNotEqual(result["scored_sample_digest"], shifted["scored_sample_digest"])

    def test_duplicate_ids_coordinates_wrong_clocks_and_untyped_masks_fail(self):
        first = keys(1)[0]
        with self.assertRaisesRegex(ValueError, "Duplicate sample"):
            LabelSeries((first, replace(first, target_time=C + 2 * HOUR)), (1., 2.), (True, True), LABEL_POLICY, LABEL_SOURCE)
        with self.assertRaisesRegex(ValueError, "Duplicate exact"):
            LabelSeries((first, replace(first, sample_id="other")), (1., 2.), (True, True), LABEL_POLICY, LABEL_SOURCE)
        for changed in ({"cutoff": C + HOUR}, {"target_time": C + 19 * HOUR}, {"origin": C + 1}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                replace(first, **changed)
        with self.assertRaisesRegex(ValueError, "boolean masks"):
            model(mask=(1, True, True))

    def test_guard_rejects_all_protected_origins_before_numerical_inspection(self):
        class NotANumber:
            def __float__(self):
                raise AssertionError("Protected numerical value was inspected")
        mixed_keys = (keys(1)[0], keys(1, origin=ASSESSMENT_START)[0])
        truth = labels((NotANumber(), NotANumber()), target_keys=mixed_keys)
        pred = model((NotANumber(), NotANumber()), target_keys=mixed_keys)
        with self.assertRaisesRegex(ValueError, "disabled by default"):
            score_metrics(truth, pred)
        with self.assertRaisesRegex(ValueError, "locked protocol"):
            score_metrics(truth, pred, permit=ScoringPermit(protected_assessment_enabled=True))

    def test_development_h18_crossing_boundary_rejects_even_if_only_h1_supplied(self):
        early = ASSESSMENT_START - 2 * HOUR
        one_key = keys(1, origin=early)
        with self.assertRaisesRegex(ValueError, "cross into protected"):
            score_metrics(labels((1.,), target_keys=one_key), model((1.,), target_keys=one_key))


class ComparisonTests(unittest.TestCase):
    def test_architectures_require_exact_keys_and_same_input_cohort(self):
        models = {"LSTM": model(), "Transformer": model(name="Transformer")}
        result = compare_architectures(labels(), models)
        self.assertEqual(result["shared_scored_samples"], 3)
        self.assertEqual(len({m["scored_sample_digest"] for m in result["metrics"].values()}), 1)
        for wrong in (replace(models["Transformer"], input_cohort_digest="e" * 64),
                      replace(models["Transformer"], keys=tuple(reversed(keys())))):
            with self.assertRaises(ValueError):
                compare_architectures(labels(), {**models, "Transformer": wrong})

    def test_missing_architecture_output_requires_explicit_shared_subset(self):
        models = {"LSTM": model(), "Transformer": model((None, 2., 3.), (False, True, True), name="Transformer")}
        with self.assertRaisesRegex(ValueError, "shared_intersection"):
            compare_architectures(labels(), models)
        result = compare_architectures(labels(), models, missing_prediction_policy="shared_intersection")
        self.assertEqual((result["label_supported_samples"], result["shared_scored_samples"], result["removed_admitted_labels"]), (3, 2, 1))
        self.assertEqual({m["n"] for m in result["metrics"].values()}, {2})

    def test_two_hour_old_nwm_native18_supports_only_horizons1_to16(self):
        target_keys = keys(18)
        prediction = align_nwm_predictions(target_keys, {(SITE, C): nwm()})
        self.assertEqual(prediction.support_mask, (True,) * 16 + (False, False))
        self.assertEqual(prediction.values[:2], (3., 4.))  # exact source leads, never relabelled h1/h2
        self.assertEqual(prediction.support_reasons[-2:], ("TARGET_OUTSIDE_NATIVE_SUPPORT",) * 2)
        self.assertEqual(prediction.prediction_mask[-2:], (False, False))

    def test_nwm_unavailable_missing_or_original_mask_false_is_separate_support(self):
        target_keys = keys(18)
        late = align_nwm_predictions(target_keys, {(SITE, C): nwm(available=C + 1)})
        self.assertFalse(any(late.support_mask))
        self.assertTrue(late.prediction_mask[0])
        self.assertEqual(late.support_reasons[0], "SOURCE_NOT_AVAILABLE_AT_C")
        missing = align_nwm_predictions(target_keys, {})
        self.assertEqual(set(missing.support_reasons), {"SOURCE_RUN_MISSING"})
        masked_source = replace(nwm(), source_support=(True, True, False) + (True,) * 15)
        masked = align_nwm_predictions(target_keys, {(SITE, C): masked_source})
        self.assertFalse(masked.support_mask[0])
        self.assertTrue(masked.prediction_mask[0])
        self.assertEqual(masked.support_reasons[0], "ORIGINAL_SOURCE_MASK_FALSE")

    def test_invalid_nwm_source_leads_clocks_gauge_and_vintages_fail(self):
        with self.assertRaisesRegex(ValueError, "leads 1 through 18"):
            replace(nwm(), valid_times=nwm().valid_times[:-1] + (C + 17 * HOUR,))
        with self.assertRaisesRegex(ValueError, "availability"):
            nwm(available=C - 3 * HOUR)
        with self.assertRaisesRegex(ValueError, "source gauge"):
            align_nwm_predictions(keys(), {(SITE, C): nwm(site="03164000")})
        with self.assertRaisesRegex(ValueError, "source gauge or initialization"):
            align_nwm_predictions(keys(), {(SITE, C): nwm(reference=C + HOUR, available=C + HOUR)})
        combined = keys(1) + keys(1, origin=C + HOUR)
        with self.assertRaisesRegex(ValueError, "Competing NWM"):
            align_nwm_predictions(combined, {(SITE, C): nwm(), (SITE, C + HOUR): nwm(run_id="competing", digest="e" * 64)})
        with self.assertRaisesRegex(ValueError, "conflicting source"):
            align_nwm_predictions(combined, {(SITE, C): nwm(),
                                             (SITE, C + HOUR): replace(nwm(), values=(999.,) + nwm().values[1:])})

    def test_nwm_missingness_never_changes_independent_architecture_result(self):
        target_keys = keys(18)
        truth = labels(tuple(float(h) for h in range(1, 19)), target_keys=target_keys)
        architectures = {"LSTM": model(truth.values, target_keys=target_keys),
                         "Transformer": model(truth.values, target_keys=target_keys, name="Transformer")}
        independent = compare_architectures(truth, architectures)
        external = align_nwm_predictions(target_keys, {(SITE, C): nwm()})
        with self.assertRaisesRegex(ValueError, "original run metadata"):
            compare_with_nwm(truth, architectures, external)
        matched = compare_with_nwm(truth, architectures, {(SITE, C): nwm()})
        self.assertEqual(matched["independent_architectures"], independent)
        self.assertEqual(matched["matched_nwm_subset"]["n"], 16)
        self.assertEqual(matched["matched_nwm_subset"]["excluded_from_independent_scored_samples"], 2)
        self.assertEqual(matched["nwm_alignment"]["actual_source_leads_supported"], list(range(3, 19)))
        self.assertEqual(matched["nwm_alignment"]["source_age_hours_range"], [2, 2])
        self.assertEqual(matched["nwm_alignment"]["support_reason_counts"],
                         {"SOURCE_SUPPORTED": 16, "TARGET_OUTSIDE_NATIVE_SUPPORT": 2})
        absent = compare_with_nwm(truth, architectures, {})
        self.assertEqual(absent["independent_architectures"], independent)
        self.assertEqual(absent["matched_nwm_subset"]["n"], 0)
        self.assertNotEqual(absent["nwm_alignment"]["source_alignment_digest"],
                            matched["nwm_alignment"]["source_alignment_digest"])
        self.assertEqual(absent["matched_nwm_subset"]["metrics"]["NWM"]["undefined"]["rmse"], "ZERO_SUPPORT")


if __name__ == "__main__":
    unittest.main()
