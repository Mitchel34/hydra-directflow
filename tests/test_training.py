"""Train-only statistics, temporal admission and sample-order tests: zero updates."""
from dataclasses import asdict
import copy
import unittest
from unittest.mock import patch
import hashlib
import numpy as np

from hydra_directflow.training import (TrainOnlyTransform, EpochSampler, EarlyStopping,
    TrainerSpec, validate_coordinates, validate_split, array_identity, validate_development_bundle,
    FULL_FEATURE_SCHEMA, Q_ONLY_FEATURE_SCHEMA, CHECKPOINT_SCHEMA, feature_schema_fields,
    q_only_training_bundle, validate_checkpoint_mode, DevelopmentTrainer)
from hydra_directflow.local_data import utc_epoch


def fixture():
    n, length = 4, 24
    origins = np.array([utc_epoch("2021-01-01T00:00:00Z") + i * 86400 for i in range(n)], dtype=np.int64)
    arrays = dict(origin_time=origins, history_time=origins[:, None] - np.arange(length + 1, 1, -1) * 3600,
                  target_time=origins[:, None] + np.arange(1, 19) * 3600, site_id=np.zeros(n, np.int64), case_id=np.array([str(i) for i in range(n)]))
    for name, shape in (("history", (n, length, 2)), ("weather", (n, 18, 4)), ("state", (n, 2)), ("target", (n, 18))):
        arrays[name] = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1
        arrays[name + "_mask"] = np.ones(shape, dtype=bool)
    return arrays


class TrainingTests(unittest.TestCase):
    def test_validation_values_cannot_change_train_statistics(self):
        arrays = fixture()
        original = TrainOnlyTransform.fit(arrays, [0, 1])
        for name in ("history", "weather", "state", "target"):
            arrays[name][2:] = 1e12
        changed = TrainOnlyTransform.fit(arrays, [0, 1])
        self.assertEqual(asdict(original), asdict(changed))

    def test_missing_labels_remain_nan_and_input_placeholders_have_false_masks(self):
        arrays = fixture()
        arrays["target_mask"][0, 0] = False
        arrays["target"][0, 0] = np.nan
        arrays["history_mask"][0, 0] = False
        arrays["history"][0, 0] = np.nan
        before = array_identity(arrays)
        transformed = TrainOnlyTransform.fit(arrays, [0, 1]).transform(arrays)
        self.assertTrue(np.isnan(transformed["target"][0, 0]))
        self.assertTrue(np.all(transformed["history"][0, 0] == 0))
        self.assertFalse(transformed["history_mask"][0, 0].any())
        self.assertEqual(before, array_identity(arrays))

    def test_constant_channel_and_inverse_transform(self):
        arrays = fixture(); arrays["state"][:] = 5
        scaler = TrainOnlyTransform.fit(arrays, [0, 1])
        self.assertTrue(all(p["constant"] and p["scale"] == 1 for p in scaler.parameters["state"]))
        mapped = scaler.transform(arrays)
        # Standardized model values are FP32; inverse cancellation near zero
        # should be assessed at single-precision accuracy, not bit identity.
        np.testing.assert_allclose(scaler.inverse_target(mapped["target"]), arrays["target"], rtol=1e-6, atol=3e-6)
        self.assertEqual(asdict(scaler), asdict(TrainOnlyTransform.from_dict(asdict(scaler))))

    def test_empty_training_channel_cannot_borrow_validation_data(self):
        arrays = fixture(); arrays["weather_mask"][:2, :, 0] = False
        with self.assertRaisesRegex(ValueError, "No observed training"):
            TrainOnlyTransform.fit(arrays, [0, 1])

    def test_protected_coordinates_rejected_before_value_access(self):
        arrays = fixture()
        shift = utc_epoch("2024-08-01T00:00:00Z") - arrays["origin_time"][0]
        for name in ("origin_time", "history_time", "target_time"):
            arrays[name] += shift
        for name in ("history", "weather", "state", "target"):
            del arrays[name]
        with self.assertRaisesRegex(ValueError, "Protected"):
            validate_coordinates(arrays, 24)

    def test_split_purges_crossing_labels_and_never_overlaps(self):
        arrays = fixture(); boundary = int(arrays["origin_time"][2])
        self.assertEqual(validate_split(arrays, [0, 1], [2, 3], boundary_epoch=boundary), ((0, 1), (2, 3)))
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split(arrays, [0, 1], [1, 2], boundary_epoch=boundary)
        arrays["target_time"][1, -1] = boundary
        with self.assertRaisesRegex(ValueError, "cross"):
            validate_split(arrays, [0, 1], [2, 3], boundary_epoch=boundary)

    def test_exact_sample_order_across_mid_epoch_and_boundary_resume(self):
        sampler = EpochSampler(list(range(11)), seed=17, shuffle=True)
        sampler.next(4)
        resumed = EpochSampler.restore(sampler.state())
        for _ in range(8):
            self.assertEqual(sampler.next(4), resumed.next(4))
        corrupted = copy.deepcopy(sampler.state()); corrupted["order_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "order changed"):
            EpochSampler.restore(corrupted)

    def test_sampler_does_not_change_membership_or_drop_partial_batch(self):
        sampler = EpochSampler([3, 5, 9], seed=29, shuffle=False)
        self.assertEqual(sampler.next(2), [3, 5]); self.assertEqual(sampler.next(2), [9])
        self.assertEqual(sampler.next(2), [3, 5]); self.assertEqual(sampler.epoch, 1)

    def test_early_stopping_state_survives_restart(self):
        stopping = EarlyStopping(10, 3, 2, .01)
        stopping.observe(1.); stopping.observe(.5)
        resumed = EarlyStopping(**asdict(stopping))
        for loss in (.51, .52):
            self.assertEqual(stopping.observe(loss), resumed.observe(loss))
        self.assertTrue(resumed.finished); self.assertEqual(resumed.best_loss, .5)

    def test_update_allowance_and_explicit_local_device(self):
        base = dict(architecture="LSTM", device="cpu", history_hours=24, batch_size=4, learning_rate=.001, seed=17, shuffle_training=True, max_updates=0)
        self.assertEqual(TrainerSpec(**base).max_updates, 0)
        self.assertEqual(TrainerSpec(**base).feature_schema, FULL_FEATURE_SCHEMA)
        for change in (dict(device="cuda"), dict(max_updates=True), dict(batch_size=0)):
            with self.assertRaises(ValueError):
                TrainerSpec(**{**base, **change})

    def test_q_only_adapter_never_fetches_original_weather_or_soil(self):
        original = fixture()
        before = array_identity(original)
        forbidden = {"weather", "weather_mask", "state", "state_mask"}
        class PoisonWeather(dict):
            def __getitem__(self, key):
                if key in forbidden:
                    raise AssertionError("Original weather/soil was read: " + key)
                return super().__getitem__(key)
        adapted = q_only_training_bundle(PoisonWeather(original), cohort_admission_digest="b" * 64)
        for name in ("history", "history_mask", "target", "target_mask", "history_time", "target_time", "origin_time", "case_id", "site_id"):
            np.testing.assert_equal(adapted[name], original[name])
            self.assertFalse(adapted[name].flags.writeable)
            self.assertFalse(np.shares_memory(adapted[name], original[name]))
        self.assertEqual(adapted["cohort_admission_digest"].item(), "b" * 64)
        self.assertEqual(adapted["feature_schema"].item(), Q_ONLY_FEATURE_SCHEMA)
        self.assertEqual(adapted["weather"].shape, (4, 18, 1))
        self.assertEqual(adapted["state"].shape, (4, 0))
        np.testing.assert_equal(adapted["weather"][0, :, 0], np.arange(1, 19))
        self.assertEqual(array_identity(original), before)

    def test_q_only_protected_coordinates_reject_before_any_flow_access(self):
        original = fixture()
        shift = utc_epoch("2024-08-01T00:00:00Z") - original["origin_time"][0]
        for name in ("origin_time", "history_time", "target_time"):
            original[name] += shift
        class PoisonValues(dict):
            def __getitem__(self, key):
                if key in ("history", "target", "weather", "state"):
                    raise AssertionError("Numerical values were accessed before the protected guard")
                return super().__getitem__(key)
        with self.assertRaisesRegex(ValueError, "Protected"):
            q_only_training_bundle(PoisonValues(original), cohort_admission_digest="b" * 64)

    def test_q_only_and_full_share_identical_history_target_scaling(self):
        arrays = fixture()
        arrays["history_mask"][0, 0, 0] = False
        arrays["history"][0, 0, 0] = np.nan
        arrays["target_mask"][0, 0] = False
        arrays["target"][0, 0] = np.nan
        q = q_only_training_bundle(arrays, cohort_admission_digest="b" * 64)
        full_scaler = TrainOnlyTransform.fit(arrays, [0, 1])
        q_scaler = TrainOnlyTransform.fit(q, [0, 1], feature_schema=Q_ONLY_FEATURE_SCHEMA)
        self.assertEqual(full_scaler.parameters["history"], q_scaler.parameters["history"])
        self.assertEqual(full_scaler.parameters["target"], q_scaler.parameters["target"])
        self.assertEqual(full_scaler.training_case_ids_sha256, q_scaler.training_case_ids_sha256)
        full_values, q_values = full_scaler.transform(arrays), q_scaler.transform(q)
        for name in ("history", "history_mask", "target", "target_mask"):
            np.testing.assert_equal(full_values[name], q_values[name])
        self.assertEqual(q_scaler.parameters["state"], [])
        self.assertEqual(q_scaler.parameters["weather"][0]["declared"], "HORIZON_DIV18")
        self.assertEqual(q_scaler.parameters["weather"][0]["cells"], 0)
        np.testing.assert_allclose(q_values["weather"][0, :, 0], np.arange(1, 19) / 18, rtol=1e-7)
        self.assertEqual(q_values["state"].shape, (4, 0))
        self.assertEqual(asdict(q_scaler), asdict(TrainOnlyTransform.from_dict(asdict(q_scaler))))

    def test_q_only_validation_values_cannot_change_training_statistics(self):
        q = q_only_training_bundle(fixture(), cohort_admission_digest="b" * 64)
        before = TrainOnlyTransform.fit(q, [0, 1], feature_schema=Q_ONLY_FEATURE_SCHEMA)
        changed = {name: np.array(value, copy=True) for name, value in q.items()}
        for name in ("history", "target"):
            changed[name][2:] = 1e12
        after = TrainOnlyTransform.fit(changed, [0, 1], feature_schema=Q_ONLY_FEATURE_SCHEMA)
        self.assertEqual(asdict(before), asdict(after))

    def test_schema_and_declared_horizon_cannot_be_substituted(self):
        q = q_only_training_bundle(fixture(), cohort_admission_digest="b" * 64)
        validate_development_bundle(q, 24, feature_schema=Q_ONLY_FEATURE_SCHEMA)
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_development_bundle(q, 24)
        with self.assertRaisesRegex(ValueError, "schema"):
            TrainOnlyTransform.fit(q, [0, 1])
        with self.assertRaises(ValueError):
            feature_schema_fields("implicit_q_only")
        changed = {name: np.array(value, copy=True) for name, value in q.items()}
        changed["weather"][0, 0, 0] = 999
        with self.assertRaisesRegex(ValueError, "known"):
            validate_development_bundle(changed, 24, feature_schema=Q_ONLY_FEATURE_SCHEMA)
        with self.assertRaisesRegex(ValueError, "known"):
            TrainOnlyTransform.fit(changed, [0, 1], feature_schema=Q_ONLY_FEATURE_SCHEMA)
        state = asdict(TrainOnlyTransform.fit(q, [0, 1], feature_schema=Q_ONLY_FEATURE_SCHEMA))
        state["parameters"]["weather"][0]["scale"] = 1.
        with self.assertRaisesRegex(ValueError, "horizon"):
            TrainOnlyTransform.from_dict(state)

    def test_q_only_requires_explicit_cohort_binding(self):
        for value in (None, "", "invalid"):
            with self.assertRaises(ValueError):
                q_only_training_bundle(fixture(), cohort_admission_digest=value)
        q = q_only_training_bundle(fixture(), cohort_admission_digest="b" * 64)
        changed = dict(q)
        del changed["cohort_admission_digest"]
        with self.assertRaisesRegex(ValueError, "cohort"):
            validate_development_bundle(changed, 24, feature_schema=Q_ONLY_FEATURE_SCHEMA)

    def test_q_only_cannot_relabel_an_existing_full_cohort(self):
        arrays = fixture()
        arrays["feature_schema"] = np.array(FULL_FEATURE_SCHEMA)
        arrays["cohort_admission_digest"] = np.array("a" * 64)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            q_only_training_bundle(arrays, cohort_admission_digest="b" * 64)
        q = q_only_training_bundle(arrays, cohort_admission_digest="a" * 64)
        self.assertEqual(q["cohort_admission_digest"].item(), "a" * 64)

    def test_q_only_cannot_be_adapted_again_or_accept_a_nonfull_mode(self):
        q = q_only_training_bundle(fixture(), cohort_admission_digest="a" * 64)
        with self.assertRaisesRegex(ValueError, "schema"):
            q_only_training_bundle(q, cohort_admission_digest="a" * 64)
        original = fixture()
        original["feature_schema"] = np.array("UNRECOGNIZED_INPUT_MODE")
        with self.assertRaisesRegex(ValueError, "schema"):
            q_only_training_bundle(original, cohort_admission_digest="a" * 64)

    def test_checkpoint_mode_source_and_cohort_are_separate_bindings(self):
        checkpoint = dict(schema=CHECKPOINT_SCHEMA, source_identity="a" * 64, feature_schema=Q_ONLY_FEATURE_SCHEMA,
                          spec={"feature_schema": Q_ONLY_FEATURE_SCHEMA}, cohort_admission_digest="b" * 64)
        validate_checkpoint_mode(checkpoint, expected_source_identity="a" * 64, expected_feature_schema=Q_ONLY_FEATURE_SCHEMA,
                                 expected_cohort_admission_digest="b" * 64)
        with self.assertRaisesRegex(ValueError, "mode"):
            validate_checkpoint_mode(checkpoint, expected_source_identity="a" * 64)
        with self.assertRaisesRegex(ValueError, "expected"):
            validate_checkpoint_mode(checkpoint, expected_source_identity="a" * 64, expected_feature_schema=Q_ONLY_FEATURE_SCHEMA)
        with self.assertRaisesRegex(ValueError, "cohort"):
            validate_checkpoint_mode(checkpoint, expected_source_identity="a" * 64, expected_feature_schema=Q_ONLY_FEATURE_SCHEMA,
                                     expected_cohort_admission_digest="c" * 64)
        with self.assertRaisesRegex(ValueError, "source"):
            validate_checkpoint_mode(checkpoint, expected_source_identity="d" * 64, expected_feature_schema=Q_ONLY_FEATURE_SCHEMA,
                                     expected_cohort_admission_digest="b" * 64)
        with self.assertRaisesRegex(ValueError, "legacy"):
            validate_checkpoint_mode({**checkpoint, "schema": "hydra-development-trainer-checkpoint/v1"},
                                     expected_source_identity="a" * 64, expected_feature_schema=Q_ONLY_FEATURE_SCHEMA,
                                     expected_cohort_admission_digest="b" * 64)

    def test_restore_mode_mismatch_fails_before_constructing_model_or_optimizer(self):
        raw = b"synthetic-checkpoint-byte-identity"
        checkpoint = dict(schema=CHECKPOINT_SCHEMA, source_identity="a" * 64, feature_schema=Q_ONLY_FEATURE_SCHEMA,
                          spec={"feature_schema": Q_ONLY_FEATURE_SCHEMA}, cohort_admission_digest="b" * 64)
        with patch("hydra_directflow.engineering._local_bytes", return_value=raw), patch("torch.load", return_value=checkpoint), \
             patch.object(DevelopmentTrainer, "__init__", side_effect=AssertionError("Trainer/model construction should not occur")):
            with self.assertRaisesRegex(ValueError, "mode"):
                DevelopmentTrainer.restore("unused", fixture(), expected_sha256=hashlib.sha256(raw).hexdigest(), expected_source_identity="a" * 64)


if __name__ == "__main__":
    unittest.main()
