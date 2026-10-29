"""Pure synthetic NumPy ablations; no Torch imports, forward calls or updates."""
import copy
import unittest

import numpy as np

from hydra_directflow.ablations import (
    Q_HISTORY_FEATURES, Q_ISSUED_FEATURES, Q_STATE_FEATURES, q_only_view,
)
from hydra_directflow.contract import GAUGE_IDS, HORIZONS_HOURS


COHORT = "a" * 64
C = 1629684000


def batch(length=24):
    origins = np.array([C, C + 3600, C + 7200], dtype=np.int64)
    result = {
        "history": np.arange(3 * length * 2, dtype=np.float32).reshape(3, length, 2),
        "history_mask": np.ones((3, length, 2), dtype=bool),
        "weather": np.ones((3, 18, 4), dtype=np.float32),
        "weather_mask": np.ones((3, 18, 4), dtype=bool),
        "state": np.ones((3, 2), dtype=np.float32),
        "state_mask": np.ones((3, 2), dtype=bool),
        "site_id": np.array([2, 0, 1], dtype=np.int64),
        "origin_time": origins,
        "target_time": origins[:, None] + np.arange(1, 19, dtype=np.int64)[None, :] * 3600,
        "case_ids": ("case-c", "case-a", "case-b"),
        "history_features": Q_HISTORY_FEATURES,
        "target": np.arange(54, dtype=np.float32).reshape(3, 18),
        "target_mask": np.ones((3, 18), dtype=bool),
        "run_age_hours": np.array([2., 3., 4.]),
    }
    result["history_mask"][0, 0, 0] = False
    result["history"][0, 0, 0] = 0.
    result["target_mask"][1, 5] = False
    result["target"][1, 5] = np.nan
    return result


def assert_forward_equal(test, left, right):
    for a, b in zip(left.forward_args(), right.forward_args()):
        np.testing.assert_array_equal(a, b)


class AblationTests(unittest.TestCase):
    def test_only_history_known_horizon_and_empty_state_are_forward_inputs(self):
        original = batch()
        view = q_only_view(original, cohort_admission_digest=COHORT)
        np.testing.assert_array_equal(view.history, original["history"])
        np.testing.assert_array_equal(view.history_mask, original["history_mask"])
        self.assertEqual(view.weather.shape, (3, 18, 1))
        for row in view.weather:
            np.testing.assert_array_equal(row[:, 0], HORIZONS_HOURS)
        self.assertTrue(view.weather_mask.all())
        self.assertEqual(view.state.shape, (3, 0))
        self.assertEqual(view.state_mask.shape, (3, 0))
        self.assertEqual(len(view.forward_args()), 6)
        self.assertTrue(all(value is not view.target and value is not view.target_mask for value in view.forward_args()))

    def test_weather_soil_masks_and_cycle_ages_cannot_change_q_only_view(self):
        original = batch()
        changed = copy.deepcopy(original)
        for name in ("weather", "state", "run_age_hours"):
            changed[name][...] = np.nan
        changed["weather_mask"][...] = False
        changed["state_mask"][...] = False
        expected = q_only_view(original, cohort_admission_digest=COHORT)
        actual = q_only_view(changed, cohort_admission_digest=COHORT)
        assert_forward_equal(self, expected, actual)
        self.assertEqual(expected.cohort_admission_digest, actual.cohort_admission_digest)
        self.assertEqual(expected.case_ids, actual.case_ids)
        np.testing.assert_array_equal(expected.target_mask, actual.target_mask)

    def test_ignored_weather_and_soil_are_never_coerced_or_inspected(self):
        class Unreadable:
            def __array__(self, *args, **kwargs):
                raise AssertionError("Ignored HRRR predictor was inspected")
        original = batch()
        expected = q_only_view(original, cohort_admission_digest=COHORT)
        for name in ("weather", "weather_mask", "state", "state_mask", "run_age_hours"):
            original[name] = Unreadable()
        assert_forward_equal(self, expected, q_only_view(original, cohort_admission_digest=COHORT))

    def test_target_values_and_masks_do_not_change_forward_inputs(self):
        original = batch()
        expected = q_only_view(original, cohort_admission_digest=COHORT)
        original["target"][...] = np.nan
        original["target_mask"][...] = False
        actual = q_only_view(original, cohort_admission_digest=COHORT)
        assert_forward_equal(self, expected, actual)
        self.assertTrue(np.isnan(actual.target).all())
        self.assertFalse(actual.target_mask.any())

    def test_metadata_and_label_masks_preserve_original_case_order(self):
        original = batch()
        view = q_only_view(original, cohort_admission_digest=COHORT)
        for name in ("site_id", "origin_time", "target_time", "target", "target_mask"):
            np.testing.assert_array_equal(getattr(view, name), original[name])
        self.assertEqual(view.case_ids, ("case-c", "case-a", "case-b"))
        self.assertEqual(view.cohort_admission_digest, COHORT)

    def test_known_calendar_mask_does_not_borrow_weather_support(self):
        original = batch()
        original["weather_mask"][...] = False
        original["state_mask"][...] = False
        view = q_only_view(original, cohort_admission_digest=COHORT)
        self.assertTrue(view.weather_mask.all())
        self.assertEqual(view.case_ids, original["case_ids"])
        self.assertEqual(view.cohort_admission_digest, COHORT)
        np.testing.assert_array_equal(view.history_mask, original["history_mask"])

    def test_snapshot_does_not_modify_source_and_cannot_be_mutated_accidentally(self):
        original = batch()
        before = copy.deepcopy(original)
        view = q_only_view(original, cohort_admission_digest=COHORT)
        for name, value in original.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, before[name])
        with self.assertRaises(ValueError):
            view.history[0, 0, 0] = 777
        original["history"][0, 1, 0] = 999
        self.assertNotEqual(view.history[0, 1, 0], 999)

    def test_model_field_description_is_lstm_with_no_hrrr_or_nwm_predictors(self):
        for length in (24, 168):
            view = q_only_view(batch(length), cohort_admission_digest=COHORT)
            fields = view.model_config_fields()
            self.assertEqual(fields["architecture"], "LSTM")
            self.assertEqual(fields["history_features"], Q_HISTORY_FEATURES)
            self.assertEqual(fields["weather_features"], Q_ISSUED_FEATURES)
            self.assertEqual(fields["state_features"], Q_STATE_FEATURES)
            self.assertEqual(fields["history_hours"], length)
            self.assertEqual(fields["horizon_hours"], HORIZONS_HOURS)
            self.assertFalse(any("nwm" in name for group in (fields["history_features"], fields["weather_features"], fields["state_features"]) for name in group))

    def test_unknown_or_mislabeled_history_and_unbound_cohort_are_rejected(self):
        for names in (("discharge_cms", "nwm_forecast"), ("discharge_cms", "precip_mm"), ("discharge_cms",)):
            original = batch()
            original["history_features"] = names
            with self.subTest(names=names), self.assertRaises(ValueError):
                q_only_view(original, cohort_admission_digest=COHORT)
        for identity in (None, "", "not-a-hash", "G" * 64):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                q_only_view(batch(), cohort_admission_digest=identity)

    def test_no_silent_case_reordering_filtering_or_time_relabeling(self):
        for field in ("target_time", "history_mask", "case_ids", "site_id"):
            original = batch()
            if field == "target_time":
                original[field][0, -1] += 3600
            elif field == "history_mask":
                original[field] = original[field].astype(np.float32)
            elif field == "case_ids":
                original[field] = ("same", "same", "same")
            else:
                original[field][0] = 3
            with self.subTest(field=field), self.assertRaises(ValueError):
                q_only_view(original, cohort_admission_digest=COHORT)

    def test_string_gauge_identity_is_preserved_without_reencoding(self):
        original = batch()
        original["site_id"] = np.array([GAUGE_IDS[2], GAUGE_IDS[0], GAUGE_IDS[1]])
        view = q_only_view(original, cohort_admission_digest=COHORT)
        np.testing.assert_array_equal(view.site_id, original["site_id"])


if __name__ == "__main__":
    unittest.main()
