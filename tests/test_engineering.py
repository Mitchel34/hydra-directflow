"""Zero-update contract tests; no Torch import, fixture decode or optimizer."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra_directflow.contract import GAUGE_IDS, HORIZONS_HOURS
from hydra_directflow.engineering import (
    CYCLE, FEATURES, FIXTURE_BYTES, FIXTURE_SHA256, HISTORY_TIMES, HRRR_SHA256, ORIGIN,
    ORIGIN_EPOCH, TARGET_TIMES, UNIT_MAPPING, USGS_SHA256, WEATHER_GROUPS, WEIGHTS_SHA256,
    ProbeSpec, _verify_coordinates, _verify_selected_provenance,
    plan_probe, run_engineering_probe, validate_fixture_receipt,
)


def fixture_receipt():
    # Synthetic metadata only. No private dataset is consumed by these tests.
    return {
        "schema": "hydra-directflow-local-smoke/v1", "status": "PASSED_DEVELOPMENT_SMOKE",
        "origin_utc": ORIGIN, "source_cycle": CYCLE, "gauge_ids": list(GAUGE_IDS),
        "history_hours": 24, "horizons_hours": list(HORIZONS_HOURS),
        "history_start_utc": "2021-08-22T01:00:00Z", "history_end_utc": "2021-08-23T00:00:00Z",
        "target_start_utc": "2021-08-23T03:00:00Z", "target_end_utc": "2021-08-23T20:00:00Z",
        "sources": {"hrrr": {"sha256": HRRR_SHA256}, "usgs": {"sha256": USGS_SHA256}},
        "geometry": {"weights_sha256": WEIGHTS_SHA256, "scientific_admission": False,
                     "original_candidate_status_unchanged": True},
        "availability": {"mode": "retrospective_simulated", "lag_seconds": 7200,
                         "retrieval_time_used_as_availability": False},
        "features": {**{name: list(values) for name, values in FEATURES.items()},
                     "target": "discharge_cms", "transform": "PHYSICAL_UNITS_UNSCALED",
                     "quality_policy": "STRICT_EXACT_A_ONLY"},
        "execution": {"assessment_rows_read": 0, "training_steps": 0},
        "private_tensor_artifact_written": True,
        "private_tensor_artifact": {"format": "npz", "sha256": FIXTURE_SHA256, "bytes": FIXTURE_BYTES},
    }


class EngineeringAdmissionTests(unittest.TestCase):
    def test_zero_update_default_and_only_confirmed_candidates(self):
        for gauge in GAUGE_IDS:
            spec = ProbeSpec(gauge_id=gauge)
            self.assertEqual(spec.max_optimizer_steps, 0)
            self.assertEqual(spec.device, "cpu")
        self.assertEqual(ProbeSpec(GAUGE_IDS[0], architecture="Mamba2_Transformer").architecture,
                         "Mamba2_Transformer")

    def test_invalid_or_unbounded_execution_specs_rejected(self):
        invalid = (
            {"max_optimizer_steps": -1}, {"max_optimizer_steps": 17}, {"max_optimizer_steps": True},
            {"max_wall_seconds": 301}, {"max_wall_seconds": 0}, {"max_wall_seconds": float("nan")},
            {"max_combined_bytes": 4 * 1024**3 + 1}, {"max_combined_bytes": 0},
            {"learning_rate": float("nan")}, {"learning_rate": .1},
            {"device": "cuda"}, {"architecture": "unknown"}, {"seed": -1},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ProbeSpec(GAUGE_IDS[0], **kwargs)
        with self.assertRaises(ValueError):
            ProbeSpec("unconfirmed-gauge")

    def test_pinned_development_fixture_admitted_without_execution(self):
        admitted = validate_fixture_receipt(fixture_receipt())
        self.assertEqual(admitted["origin_utc"], ORIGIN)
        self.assertEqual(admitted["npz_sha256"], FIXTURE_SHA256)

    def test_arbitrary_npz_hash_or_byte_count_is_not_admitted(self):
        for key, value in (("sha256", "b" * 64), ("bytes", FIXTURE_BYTES + 1)):
            receipt = fixture_receipt()
            receipt["private_tensor_artifact"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "provenance-v2"):
                validate_fixture_receipt(receipt)

    def test_embedded_coordinates_are_checked_independently_of_receipt(self):
        _verify_coordinates([0, 1, 2], list(HISTORY_TIMES), list(TARGET_TIMES))
        for sites, histories, targets in (([1, 0, 2], HISTORY_TIMES, TARGET_TIMES),
                                         ([0, 1, 2], HISTORY_TIMES[1:] + (ORIGIN_EPOCH,), TARGET_TIMES),
                                         ([0, 1, 2], HISTORY_TIMES, tuple(t + 3600 for t in TARGET_TIMES)),
                                         ([False, 1, 2], HISTORY_TIMES, TARGET_TIMES)):
            with self.subTest(sites=sites, targets=targets), self.assertRaises(ValueError):
                _verify_coordinates(sites, histories, targets)

    def test_protected_or_other_development_dates_rejected(self):
        for origin in ("2024-07-31T00:00:00Z", "2025-07-31T00:00:00Z", "2021-08-23T03:00:00Z"):
            receipt = fixture_receipt()
            receipt["origin_utc"] = origin
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "August 2021"):
                validate_fixture_receipt(receipt)

    def test_source_hash_changes_rejected(self):
        for source in ("hrrr", "usgs"):
            receipt = fixture_receipt()
            receipt["sources"][source]["sha256"] = "b" * 64
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "identities"):
                validate_fixture_receipt(receipt)
        receipt = fixture_receipt()
        receipt["geometry"]["weights_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            validate_fixture_receipt(receipt)

    def test_target_order_horizon_and_source_claim_changes_rejected(self):
        variants = []
        for field, value in (("horizons_hours", [1, 2]), ("history_hours", 168),
                             ("target_end_utc", "2024-08-01T20:00:00Z"),
                             ("status", "FROZEN_SCIENCE")):
            receipt = fixture_receipt()
            receipt[field] = value
            variants.append(receipt)
        receipt = fixture_receipt()
        receipt["geometry"]["scientific_admission"] = True
        variants.append(receipt)
        receipt = fixture_receipt()
        receipt["features"]["weather"] = list(reversed(receipt["features"]["weather"]))
        variants.append(receipt)
        receipt = fixture_receipt()
        receipt["execution"]["assessment_rows_read"] = 1
        variants.append(receipt)
        for receipt in variants:
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                validate_fixture_receipt(receipt)

    def test_unapproved_quality_or_availability_change_rejected(self):
        for group, key, value in (("features", "quality_policy", "ALL"),
                                  ("features", "transform", "ALREADY_FITTED"),
                                  ("availability", "mode", "documented"),
                                  ("availability", "lag_seconds", 0),
                                  ("availability", "retrieval_time_used_as_availability", True)):
            receipt = fixture_receipt()
            receipt[group][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_fixture_receipt(receipt)

    def test_fixed_engineering_mapping_has_separate_target_role(self):
        self.assertEqual(UNIT_MAPPING["target"], ((0., 100.),))
        self.assertNotIn("target", FEATURES)
        self.assertEqual(FEATURES["weather"], ("precip_mm", "temp_k", "specific_humidity", "swe_kg_m2"))


class EngineeringPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.receipt_path = self.root / "receipt.json"
        self.receipt_path.write_text(json.dumps(fixture_receipt()))

    def tearDown(self):
        self.temp.cleanup()

    def test_disabled_execution_does_not_import_backend_read_npz_or_create_directory(self):
        destination = self.root / "must-not-exist"
        with patch("hydra_directflow.engineering.subprocess.Popen", side_effect=AssertionError("No worker")), \
             patch("hydra_directflow.engineering._load_case", side_effect=AssertionError("No source decode")), \
             patch("hydra_directflow.engineering.RunJournal", side_effect=AssertionError("No journal creation")):
            plan = run_engineering_probe(receipt_path=self.receipt_path, fixture_path=self.root / "absent.npz",
                                        spec=ProbeSpec(GAUGE_IDS[0], max_optimizer_steps=2), run_directory=destination)
        self.assertEqual(plan["status"], "PLANNED_NOT_EXECUTED")
        self.assertEqual(plan["target_role"], "LOSS_ONLY_NEVER_FORWARD")
        self.assertFalse(destination.exists())

    def test_zero_steps_or_missing_review_digest_reject_before_worker(self):
        for steps in (0, 2):
            with self.subTest(steps=steps), patch("hydra_directflow.engineering.subprocess.Popen") as worker:
                with self.assertRaises(ValueError):
                    run_engineering_probe(receipt_path=self.receipt_path, fixture_path="absent.npz",
                        spec=ProbeSpec(GAUGE_IDS[0], max_optimizer_steps=steps),
                        run_directory=self.root / "new", execution_enabled=True)
                worker.assert_not_called()

    def test_digest_changes_with_gauge_budget_architecture_or_source_receipt(self):
        original = plan_probe(self.receipt_path, ProbeSpec(GAUGE_IDS[0], max_optimizer_steps=2))
        for spec in (ProbeSpec(GAUGE_IDS[1], max_optimizer_steps=2),
                     ProbeSpec(GAUGE_IDS[0], max_optimizer_steps=3),
                     ProbeSpec(GAUGE_IDS[0], architecture="Transformer", max_optimizer_steps=2)):
            self.assertNotEqual(original["spec_sha256"], plan_probe(self.receipt_path, spec)["spec_sha256"])
        receipt = fixture_receipt()
        receipt["additional_metadata"] = "Changed receipt requires a new reviewed digest"
        self.receipt_path.write_text(json.dumps(receipt))
        changed = plan_probe(self.receipt_path, ProbeSpec(GAUGE_IDS[0], max_optimizer_steps=2))
        self.assertNotEqual(original["source_digest"], changed["source_digest"])
        with self.assertRaisesRegex(ValueError, "reviewed"):
            run_engineering_probe(receipt_path=self.receipt_path, fixture_path="absent.npz",
                spec=ProbeSpec(GAUGE_IDS[0], max_optimizer_steps=2), run_directory=self.root / "new",
                execution_enabled=True, approved_spec_sha256=original["spec_sha256"])

    def test_actual_plan_timestamp_is_not_replay_origin(self):
        plan = plan_probe(self.receipt_path, ProbeSpec(GAUGE_IDS[0]))
        self.assertNotEqual(plan["artifact_created_at_utc"], ORIGIN)
        self.assertEqual(plan["source"]["origin_utc"], ORIGIN)

    def test_symlink_receipt_rejected_without_source_decode(self):
        link = self.root / "link.json"
        link.symlink_to(self.receipt_path)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            plan_probe(link, ProbeSpec(GAUGE_IDS[0]))

    def test_supervisor_kills_fake_worker_on_combined_memory_limit(self):
        self._fake_limit_case(rss_bytes=3 * 1024**3, fake_clock=None, expected_error="memory limit")

    def test_supervisor_kills_fake_worker_on_overall_wall_limit(self):
        self._fake_limit_case(rss_bytes=1024, fake_clock=(0., 301.), expected_error="wall limit")

    def _fake_limit_case(self, *, rss_bytes, fake_clock, expected_error):
        # No subprocess, model, fixture decoder or optimizer is ever invoked.
        class FakeChild:
            pid = 12345
            returncode = None
            killed = False

            def poll(self):
                return self.returncode

            def kill(self):
                self.killed = True
                self.returncode = -9

            def wait(self, timeout):
                return self.returncode

        class FakeProcess:
            def __init__(self, pid=None):
                pass

            def memory_info(self):
                return SimpleNamespace(rss=rss_bytes)

            def children(self, recursive):
                return []

        fake_psutil = SimpleNamespace(Process=FakeProcess, NoSuchProcess=ProcessLookupError)
        child = FakeChild()
        spec = ProbeSpec(GAUGE_IDS[0], max_optimizer_steps=2)
        plan = plan_probe(self.receipt_path, spec)
        clock_patch = (patch("hydra_directflow.engineering.time.monotonic", side_effect=fake_clock)
                       if fake_clock else patch("hydra_directflow.engineering.time.monotonic", return_value=0.))
        with patch.dict("sys.modules", {"psutil": fake_psutil}), \
             patch("hydra_directflow.engineering.subprocess.Popen", return_value=child), \
             patch("hydra_directflow.engineering._load_case", side_effect=AssertionError("No decode")), clock_patch:
            result = run_engineering_probe(receipt_path=self.receipt_path, fixture_path=self.root / "absent.npz",
                spec=spec, run_directory=self.root / "fake-attempt", execution_enabled=True,
                approved_spec_sha256=plan["spec_sha256"])
        self.assertTrue(child.killed)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["optimizer_steps_precharged"], 2)
        self.assertEqual(result["confirmed_optimizer_steps"], 0)
        self.assertFalse(result["optimizer_step_count_exact"])
        self.assertIn(expected_error, result["error"])
        self.assertEqual(result["journal_status"], "FAILED")


class EmbeddedProvenanceTests(unittest.TestCase):
    def metadata(self):
        def record(group, valid, index):
            return dict(group=group, source_reference_UTC="2021-08-23T00:00:00Z",
                source_valid_UTC=datetime.fromtimestamp(valid, timezone.utc).isoformat().replace("+00:00", "Z"),
                lead=(valid - ORIGIN_EPOCH + 7200) // 3600, record_index=index,
                group_availability={"status": "AVAILABLE_VERIFIED", "source_key": "original-key"},
                source_values_sha256_float64_le="a" * 64, missing_bits_sha256="b" * 64,
                source_identity={"key": "original-key", "sha256": "c" * 64, "bytes": 500})

        weather, ids, indices = [], [], []
        for horizon, valid in enumerate(TARGET_TIMES):
            row, row_ids, row_indices = [], [], []
            for column, group in enumerate(WEATHER_GROUPS):
                source = record(group, valid, horizon * 4 + column)
                source.update(field_source_id="original-field-" + group,
                              units=("kg m-2", "K", "kg kg-1", "kg m-2")[column],
                              field_availability={"historical_available_at_UTC": None},
                              APCP_interval_hours=[source["lead"] - 1, source["lead"]] if column == 0 else None)
                row.append(json.dumps(source))
                row_ids.append(source["field_source_id"])
                row_indices.append(source["record_index"])
            weather.append(row)
            ids.append(row_ids)
            indices.append(row_indices)
        soil = record("soil_moisture", ORIGIN_EPOCH - 7200, 144)
        soil.update(historical_available_at_UTC=None, native_units="Proportion")
        return dict(history_quality_flags=['["A"]'] * 24, target_quality_flags=['["A"]'] * 18,
                    history_source_ids=["usgs-wire-sha256:" + "d" * 64] * 24,
                    target_source_ids=["usgs-wire-sha256:" + "e" * 64] * 18,
                    weather_source_metadata=weather, weather_source_ids=ids, weather_record_indices=indices,
                    soil_source_metadata=json.dumps(soil), soil_source_ids="original-key", soil_record_indices=144)

    def test_original_selected_provenance_is_admitted(self):
        _verify_selected_provenance(self.metadata())

    def test_equivalent_explicit_utc_spellings_preserve_original_metadata(self):
        metadata = self.metadata()
        for row in metadata["weather_source_metadata"]:
            for index, raw in enumerate(row):
                source = json.loads(raw)
                for key in ("source_reference_UTC", "source_valid_UTC"):
                    source[key] = source[key].replace("Z", "+00:00")
                row[index] = json.dumps(source)
        soil = json.loads(metadata["soil_source_metadata"])
        for key in ("source_reference_UTC", "source_valid_UTC"):
            soil[key] = soil[key].replace("Z", "+00:00")
        metadata["soil_source_metadata"] = json.dumps(soil)
        original = copy.deepcopy(metadata)
        _verify_selected_provenance(metadata)
        self.assertEqual(metadata, original)

    def test_naive_non_utc_fractional_and_changed_source_times_rejected(self):
        for value in ("2021-08-23T03:00:00", "2021-08-23T04:00:00+01:00",
                      "2021-08-23T03:00:00.000001Z", "2021-08-23T04:00:00+00:00"):
            metadata = self.metadata()
            source = json.loads(metadata["weather_source_metadata"][0][0])
            source["source_valid_UTC"] = value
            metadata["weather_source_metadata"][0][0] = json.dumps(source)
            with self.subTest(value=value), self.assertRaises(ValueError):
                _verify_selected_provenance(metadata)

    def test_rejected_quality_missing_identity_and_wrong_field_clock(self):
        original = self.metadata()
        variants = []
        changed = copy.deepcopy(original)
        changed["target_quality_flags"][0] = '["P"]'
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["history_source_ids"][0] = ""
        variants.append(changed)
        changed = copy.deepcopy(original)
        field = json.loads(changed["weather_source_metadata"][0][0])
        field["source_valid_UTC"] = "2024-07-31T00:00:00Z"
        changed["weather_source_metadata"][0][0] = json.dumps(field)
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["soil_record_indices"] = 143
        variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                _verify_selected_provenance(changed)


if __name__ == "__main__":
    unittest.main()
