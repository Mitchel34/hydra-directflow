"""Plan-only retained-source calendar, identity, and boundary tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plan_development_staging", ROOT / "scripts/plan_development_staging.py")
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)
CYCLE = "20230801_00Z"


def obj(body=b"tar", kind="cycles", cycle=CYCLE):
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            "durable_uri": f"gs://example-bucket/{kind}/{cycle}." + ("tar" if kind == "cycles" else "json"), "generation": "123"}


def approval():
    return {"schema": "hydra-execution-budget-approval/v1", "approved_total_incremental_usd": 150,
            "allocations_usd": dict(planner.ALLOCATIONS), "phase_release": {
                "development_staging": "AUTHORIZED_WITH_DECLARED_LIMITS",
                "cuda_pilot": "CONDITIONAL_ON_MEASURED_LOCAL_NEED_AND_BOUNDED_QUALIFICATION",
                "scientific_compute": "CONDITIONAL_ON_PROTOCOL_COHORT_AND_CUDA_READINESS"},
            "data_response_body_cap_bytes": planner.BODY_CAP, "maximum_get_attempts": 20000,
            "raw_staging_cap_bytes": planner.HOTSET_CAP, "protected_assessment_access_authorized": False,
            "protected_scoring_authorized": False, "iam_or_security_changes_authorized": False,
            "automatic_budget_increase": False}


def inventory(row, bundle=None):
    return {"cycle_id": row["cycle"], "source_locations": row["source_locations"],
            "data_complete": row["data_complete"], "local_candidates": [], "index_status": "PRESENT"}


class StagingPlanTests(unittest.TestCase):
    def test_approved_envelope_preserves_conditional_cloud_and_protected_boundaries(self):
        planner.validate_approval(approval())
        for change in ({"approved_total_incremental_usd": 151}, {"protected_scoring_authorized": True},
                       {"maximum_get_attempts": 20001}, {"raw_staging_cap_bytes": 3 * 1024**3},
                       {"automatic_budget_increase": True}, {"iam_or_security_changes_authorized": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                planner.validate_approval({**approval(), **change})
        changed = approval()
        changed["phase_release"]["cuda_pilot"] = "START_NOW"
        with self.assertRaises(ValueError):
            planner.validate_approval(changed)

    def test_object_identity_requires_generation_hash_and_bounded_bytes(self):
        self.assertEqual(planner.object_identity(obj(), maximum=16), obj())
        for change in ({"generation": ""}, {"durable_uri": "https://example.invalid/file"}, {"bytes": 17}, {"sha256": "bad"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                planner.object_identity({**obj(), **change}, maximum=16)
        identity = planner.object_identity({**obj(), "bytes": None}, maximum=16, optional_bytes=True)
        self.assertIsNone(identity["bytes"])

    def test_absent_warmup_is_reported_and_cannot_invent_a_fetch_target(self):
        prior = {"cycle_id": "20180711_06Z", "index_status": "ABSENT_WARMUP_OR_CALENDAR_UNKNOWN"}
        proof = {"cycle_id": prior["cycle_id"], "bundle_identity": None}
        cycles, excluded = planner.build_cycle_manifest({}, [prior["cycle_id"]], [prior], [proof])
        self.assertEqual(cycles, [])
        self.assertFalse(excluded[0]["fetch_authorized"])
        with self.assertRaises(ValueError):
            planner.build_cycle_manifest({}, [prior["cycle_id"]], [prior], [{**proof, "bundle_identity": obj()}])

    def test_direct_sealed_bundle_can_be_planned_without_opening_weather_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "cycle.tar"
            archive.write_bytes(b"tar")
            row = {"cycle": CYCLE, "data_complete": True, "location_kind": "bundle", "source_locations": {"bundle": obj()}}
            prior = inventory(row)
            prior["local_candidates"] = [{"archive_path": str(archive)}]
            proof = {"cycle_id": CYCLE, "bundle_identity": obj(), "identity_source": "SEALED_INDEX"}
            with patch.object(planner, "verified_local_bytes") as read:
                cycles, _ = planner.build_cycle_manifest({CYCLE: row}, [CYCLE], [prior], [proof])
                read.assert_not_called()
            self.assertEqual(cycles[0]["bundle_proof_route"], "SEALED_INDEX_BUNDLE")
            self.assertEqual(cycles[0]["local_archive_candidates"][0]["status"], "RESIDENT_UNVERIFIED_SHA")
            self.assertFalse(cycles[0]["archive_payloads_read_by_planner"])

    def test_current_original_commit_reverification_binds_bundle_and_missing_commit_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commit.json"
            raw = json.dumps({"cycle": CYCLE, "bundle": obj()}).encode()
            path.write_bytes(raw)
            commit = obj(raw, "commits")
            del commit["bytes"]
            row = {"cycle": CYCLE, "data_complete": True, "location_kind": "commit", "source_locations": {"commit": commit}}
            proof = {"cycle_id": CYCLE, "bundle_identity": obj(), "identity_source": "AUTHENTICATED_RESIDENT_ORIGINAL_COMMIT",
                     "commit_path": str(path), "commit_sha256": commit["sha256"]}
            cycles, _ = planner.build_cycle_manifest({CYCLE: row}, [CYCLE], [inventory(row)], [proof])
            self.assertEqual(cycles[0]["commit_identity"]["bytes"], len(raw))
            self.assertEqual(cycles[0]["bundle_proof_route"], "CURRENT_AUTHENTICATED_LOCAL_COMMIT")
            self.assertFalse(cycles[0]["commit_verification_required_before_bundle_use"])

    def test_unresolved_bundle_remains_unknown_with_bounded_commit_fetch(self):
        commit = {**obj(b"commit", "commits"), "bytes": None}
        row = {"cycle": CYCLE, "data_complete": False, "location_kind": "commit", "source_locations": {"commit": commit}}
        proof = {"cycle_id": CYCLE, "bundle_identity": None, "identity_source": "UNKNOWN"}
        cycles, _ = planner.build_cycle_manifest({CYCLE: row}, [CYCLE], [inventory(row)], [proof])
        self.assertIsNone(cycles[0]["bundle_identity"])
        self.assertIsNone(cycles[0]["commit_identity"]["bytes"])
        self.assertEqual(cycles[0]["commit_identity"]["max_response_body_bytes"], 16384)
        self.assertTrue(cycles[0]["commit_verification_required_before_bundle_use"])
        self.assertFalse(cycles[0]["data_complete"])

    def test_inventory_and_recorded_proof_cannot_override_sealed_identities(self):
        row = {"cycle": CYCLE, "data_complete": True, "location_kind": "bundle", "source_locations": {"bundle": obj()}}
        with self.assertRaises(ValueError):
            planner.build_cycle_manifest({CYCLE: row}, [CYCLE], [inventory(row)], [{"cycle_id": CYCLE, "bundle_identity": {**obj(), "generation": "999"}}])
        prior = inventory(row)
        prior["data_complete"] = False
        with self.assertRaises(ValueError):
            planner.build_cycle_manifest({CYCLE: row}, [CYCLE], [prior], [{"cycle_id": CYCLE, "bundle_identity": obj()}])

    def test_recorded_commit_proof_stays_explicitly_revalidation_required_when_nonresident(self):
        commit = obj(b"commit", "commits")
        row = {"cycle": CYCLE, "data_complete": True, "location_kind": "commit", "source_locations": {"commit": commit}}
        proof = {"cycle_id": CYCLE, "bundle_identity": obj(), "identity_source": "AUTHENTICATED_RESIDENT_ORIGINAL_COMMIT",
                 "commit_path": "not-present", "commit_sha256": commit["sha256"]}
        cycles, _ = planner.build_cycle_manifest({CYCLE: row}, [CYCLE], [inventory(row)], [proof])
        self.assertEqual(cycles[0]["bundle_proof_route"], "HASH_BOUND_RECORDED_AUTHENTICATED_COMMIT")
        self.assertTrue(cycles[0]["commit_verification_required_before_bundle_use"])

    def test_july_stages_only_verified_export_never_original_mixed_path(self):
        with tempfile.TemporaryDirectory() as directory:
            export = Path(directory) / "export"
            export.mkdir()
            raw = b"synthetic-development-export"
            (export / "data.parquet").write_bytes(raw)
            original_path = str(Path(directory) / "do-not-read" / "2024-07.parquet")
            original = {"path": original_path, "sha256": "a" * 64, "bytes": 999}
            payload = {"name": "data.parquet", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            reference = {"2024-07": {"directory": str(export), "commit_sha256": "b" * 64}}
            with patch.object(planner, "months_between", return_value=["2024-07"]), \
                 patch.object(planner, "inspect_development_export", return_value={"payload": payload, "cutoff_exclusive_epoch": planner.DEVELOPMENT_END}), \
                 patch.object(planner, "verified_local_bytes", wraps=planner.verified_local_bytes) as read:
                result = planner.build_usgs_manifest({"native_partitions": [original]}, [], reference)
            self.assertNotIn(original_path, json.dumps(result))
            self.assertEqual(result[0]["stage_kind"], "VERIFIED_DEVELOPMENT_EXPORT_ONLY")
            self.assertTrue(result[0]["raw_mixed_month_staging_forbidden"])
            self.assertEqual(Path(read.call_args.args[0]), export / "data.parquet")
            self.assertEqual(read.call_count, 1)

    def test_source_bounds_include_unknown_archives_commits_and_usgs(self):
        cycles = [{"bundle_identity": obj(), "commit_identity": {"bytes": None}, "local_archive_candidates": [], "local_commit_candidates": []},
                  {"bundle_identity": None, "commit_identity": {"bytes": 10}, "local_archive_candidates": [], "local_commit_candidates": []}]
        usgs = [{"payload_identity": {"bytes": 7}, "stage_kind": "test", "local_candidates": []}]
        result = planner.source_bound_summary(cycles, usgs)
        self.assertEqual(result["all_sources_get_upper_bound"], 5)
        self.assertEqual(result["all_sources_body_upper_bound_bytes"], 3 + planner.ARCHIVE_BOUND + planner.COMMIT_BOUND + 10 + 7 + 5)
        self.assertEqual(result["unresolved_bundle_identities"], 1)
        with patch.object(planner, "BODY_CAP", 100), self.assertRaisesRegex(ValueError, "approved"):
            planner.source_bound_summary(cycles, usgs)

    def test_duplicate_calendars_rejected(self):
        prior = {"cycle_id": CYCLE, "index_status": "ABSENT_WARMUP_OR_CALENDAR_UNKNOWN"}
        proof = {"cycle_id": CYCLE, "bundle_identity": None}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            planner.build_cycle_manifest({}, [CYCLE], [prior, prior], [proof])

    def test_plan_input_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real.json"
            path.write_text('{}')
            link = Path(directory) / "link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "resident regular"):
                planner.read_json(link)


if __name__ == "__main__":
    unittest.main()
