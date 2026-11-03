"""Metadata-only roster checks; no model imports or study values."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/plan_seed_sequence.py"
SPEC = importlib.util.spec_from_file_location("seed_sequence_tests", PATH)
schedule = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(schedule)


class SeedSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = schedule.source_report()

    def test_exact_roster_and_seed_completion_milestones(self):
        result = schedule.build_plan(self.report)
        self.assertEqual(len(result["jobs"]), 180)
        self.assertEqual(len({j["job_id"] for j in result["jobs"]}), 180)
        self.assertEqual([p["fits"] for p in result["phases"]], [24,24,24,36,36,36])
        self.assertEqual([m["fits"] for m in result["seed_milestones"]], [108,36,36])
        self.assertEqual([m["cumulative_completed_fits"] for m in result["seed_milestones"]], [108,144,180])
        self.assertEqual([j["seed"] for j in result["jobs"]], [17]*108+[29]*36+[43]*36)
        self.assertFalse(result["cv_repeated_for_later_seeds"])

    def test_no_silent_new_arms_or_cloud_backends(self):
        jobs = schedule.build_plan(self.report)["jobs"]
        self.assertEqual(sum(j["device"] == "cpu" for j in jobs), 90)
        self.assertEqual(sum(j["device"] == "mps" for j in jobs), 90)
        self.assertFalse(any(j["input_arm"] == "q_only" and j["history_hours"] == 24 for j in jobs))
        self.assertFalse(any(j["fold"] != "final_development" and j["seed"] != 17 for j in jobs))

    def test_source_order_does_not_change_schedule(self):
        changed = deepcopy(self.report)
        changed["scenarios"].reverse()
        for row in changed["scenarios"]: row["parts"].reverse()
        self.assertEqual(schedule.build_plan(self.report), schedule.build_plan(changed))

    def test_missing_duplicate_or_new_matrix_cell_rejects(self):
        for mutation in ("missing", "duplicate", "new_arm"):
            changed = deepcopy(self.report)
            scenario = next(s for s in changed["scenarios"] if s["epochs"] == 20 and s["simulated_label_delay_hours"] == 2)
            if mutation == "missing": scenario["parts"].pop()
            elif mutation == "duplicate": scenario["parts"][0] = deepcopy(scenario["parts"][1])
            else: scenario["parts"][0]["history_hours"] = 24
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "matrix"):
                schedule.build_plan(changed)

    def test_unplanned_seed_or_changed_cohort_rejects(self):
        changed = deepcopy(self.report)
        scenario = next(s for s in changed["scenarios"] if s["epochs"] == 20 and s["simulated_label_delay_hours"] == 2)
        scenario["parts"][0]["seeds"] = [17,29]
        with self.assertRaisesRegex(ValueError, "seed count"):
            schedule.build_plan(changed)
        scenario["parts"][0]["seeds"] = [17]
        scenario["parts"][0]["role_case_ids_sha256"]["outer_scoring"] = "0"*64
        with self.assertRaisesRegex(ValueError, "cohort membership"):
            schedule.build_plan(changed)

    def test_runtime_uses_per_fit_not_seed_multiplication_twice(self):
        result = schedule.build_plan(self.report)
        for epochs in (20,50):
            source = next(s for s in self.report["scenarios"] if s["epochs"] == epochs and s["simulated_label_delay_hours"] == 2)
            actual = sum(m["conditional_local_hours_by_epochs"][str(epochs)] for m in result["seed_milestones"])
            self.assertAlmostEqual(actual, source["total_hours"], places=10)
        jobs = result["jobs"]
        self.assertEqual(sum(j["update_ceiling_scenarios"]["200"] > 100000 for j in jobs),116)
        self.assertEqual(max(j["update_ceiling_scenarios"]["200"] for j in jobs),126800)

    def test_report_barriers_and_no_extra_human_pause_are_explicit(self):
        result = schedule.build_plan(self.report)
        for i, phase in enumerate(result["phases"]):
            self.assertEqual(phase["predecessor_complete_phase"], None if i==0 else result["phases"][i-1]["phase_id"])
            self.assertIn("TRAINING_AND_MATCHING_DEVELOPMENT_EVALUATION", phase["completion_requires"])
        self.assertTrue(all(m["review_may_overlap_next_seed"] for m in result["seed_milestones"]))
        self.assertFalse(any(m["extra_human_approval_pause_between_unchanged_seeds"] for m in result["seed_milestones"]))
        self.assertFalse(result["protected_assessment_access"])
        self.assertEqual(result["scientific_fits_started"],0)
        self.assertIsNone(result["scientific_epoch_ceiling_selected"])


if __name__ == "__main__":
    unittest.main()
