"""Train-only statistics, temporal admission and sample-order tests: zero updates."""
from dataclasses import asdict
import copy
import unittest
import numpy as np

from hydra_directflow.training import (TrainOnlyTransform, EpochSampler, EarlyStopping,
    TrainerSpec, validate_coordinates, validate_split, array_identity)
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
        for change in (dict(device="cuda"), dict(max_updates=True), dict(batch_size=0)):
            with self.assertRaises(ValueError):
                TrainerSpec(**{**base, **change})


if __name__ == "__main__":
    unittest.main()
