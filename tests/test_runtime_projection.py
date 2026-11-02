"""Arithmetic regressions for one-time startup sensitivity; no model execution."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("project_runtime", Path(__file__).resolve().parents[1] / "scripts/project_runtime.py")
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


def fixture():
    return dict(status="COMPLETED_ENGINEERING_TIMING", spec=dict(architecture="LSTM", device="cpu"),
                training_batches=[dict(epoch=e, cases=8, copy_seconds=.01, update_seconds=.29 if e == 1 else .09)
                                  for e in (1, 2, 3)],
                validation_batches=[dict(epoch=e, cases=4, copy_seconds=.01, forward_loss_seconds=.14 if e == 1 else .04)
                                    for e in (1, 2, 3)],
                stage_seconds=dict(imports=1., cache_read_verified=.2, tensorization=.1, model_setup=.3,
                                   checkpoint_write=.4, preprocess_first_pass=2.))


class RuntimeProjection(unittest.TestCase):
    def scenario(self):
        return dict(train_cases=8, validation_cases=4, epochs=10, trials=2, seeds=3, gauges=3)

    def test_first_use_excess_is_charged_once_per_fit_not_every_epoch(self):
        result = projection.later_epoch_projection(fixture(), self.scenario())
        self.assertAlmostEqual(result["first_use_excess_seconds_per_fit"], .3)
        self.assertAlmostEqual(result["per_fit_seconds"], 7.4)
        self.assertAlmostEqual(result["total_seconds"], 133.2)
        scenario = self.scenario()
        scenario["epochs"] = 20
        doubled = projection.later_epoch_projection(fixture(), scenario)
        self.assertAlmostEqual(doubled["per_fit_seconds"] - result["per_fit_seconds"], 5.5)

    def test_original_measurement_is_preserved(self):
        source = fixture()
        before = copy.deepcopy(source)
        projection.later_epoch_projection(source, self.scenario())
        self.assertEqual(source, before)

    def test_unequal_case_counts_or_incomplete_epochs_rejected(self):
        for mutation in ("missing_epoch", "changed_cases"):
            source = fixture()
            if mutation == "missing_epoch":
                source["training_batches"].pop()
            else:
                source["validation_batches"][-1]["cases"] = 3
            with self.assertRaises(ValueError):
                projection.later_epoch_projection(source, self.scenario())


if __name__ == "__main__":
    unittest.main()
