"""Synthetic metadata/SQLite tests; no user payloads, provider or model calls."""
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/inspect_staging_progress.py"
SPEC = importlib.util.spec_from_file_location("inspect_staging_progress_tests", PATH)
INSPECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECT)


@contextmanager
def test_database(path):
    connection = sqlite3.connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def write_json(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    Path(path).write_bytes(raw)
    return raw


def fixture(directory, completed=(0,)):
    root = Path(directory) / "corpus"
    root.mkdir()
    chunks = [dict(index=i, first_origin_utc=f"2021-01-01T{2*i:02d}:00:00Z", last_origin_utc=f"2021-01-01T{2*i+1:02d}:00:00Z",
                   origin_count=2, cycle_ids=["20201231_18Z" if i == 0 else "20210101_00Z"], calendar_unavailable_cycle_ids=[]) for i in range(2)]
    unit = dict(id="candidate_training__2021-01__L24", split="candidate_training", month="2021-01", history_hours=24,
                requested_origins=22, candidate_origins=4, purged_origins=18,
                split_end_exclusive_utc="2021-01-01T22:00:00Z", chunks=chunks, required_usgs_months=["2020-12", "2021-01"])
    plan = dict(schema=INSPECT.SCHEMA, splits=[dict(name="candidate_training", start_utc="2021-01-01T00:00:00Z", end_utc="2021-01-01T22:00:00Z")],
                history_hours=24, chunk_hours=2, input_quality_policy="PRIMARY_APPROVED_A_OPTIONAL_R_V1", target_quality_policy="PRIMARY_APPROVED_A_OPTIONAL_R_V1",
                qualified_index_sha256="a" * 64, usgs_manifest_sha256="b" * 64, candidate_weights_sha256="c" * 64,
                units=[unit], required_cycle_ids=["20201231_18Z", "20210101_00Z"], calendar_unavailable_cycle_ids=[], implementation_sha256={"test": "d" * 64},
                requested_origins=22, candidate_origins=4, purged_origins=18)
    plan["plan_sha256"] = INSPECT.digest({key: plan[key] for key in INSPECT.PLAN_KEYS})
    write_json(root / "PLAN.json", plan)
    for index in completed:
        path = root / "chunks" / unit["id"] / f"{index:03d}"
        path.mkdir(parents=True)
        chunk = chunks[index]
        summary = dict(unit_id=unit["id"], chunk_index=index, origin_count=2, case_count=6, admitted_cases=5,
                       first_origin_utc=chunk["first_origin_utc"], last_origin_utc=chunk["last_origin_utc"], calendar_unavailable_cycle_ids=[],
                       cycle_keys={name: "e" * 64 for name in chunk["cycle_ids"]}, month_keys={"2020-12": "f" * 64, "2021-01": "a" * 64},
                       case_source_digest="b" * 64, input_quality_policy=plan["input_quality_policy"], target_quality_policy=plan["target_quality_policy"],
                       eligibility_counts={"COMPLETE": 5, "HISTORY_INCOMPLETE": 1}, source_calendar_counts={"RESOLVED_WITHIN_ACQUIRED_CALENDAR": 6})
        raw = write_json(path / "case-index.json", summary)
        model = ("opaque-model-payload-" + str(index)).encode()
        (path / "model.npz").write_bytes(model)
        write_json(path / "COMMIT.json", dict(schema=INSPECT.SCHEMA, plan_sha256=plan["plan_sha256"], unit_id=unit["id"], chunk_index=index,
                   payloads={"case-index.json": {"sha256": INSPECT.sha(raw), "bytes": len(raw)}, "model.npz": {"sha256": INSPECT.sha(model), "bytes": len(model)}}, arrays={}))
    return root, plan


def budget(directory):
    path = Path(directory) / "source-budget.sqlite3"
    with test_database(path) as db:
        db.execute("CREATE TABLE budget(id INTEGER,approval TEXT,body_cap INTEGER,get_cap INTEGER,cost_cap_micro INTEGER,charged_body INTEGER,attempts INTEGER)")
        db.execute("CREATE TABLE attempts(id INTEGER,reserved INTEGER,observed INTEGER,status TEXT,created_utc TEXT)")
        db.execute("INSERT INTO budget VALUES(1,?,1000,10,1000000,25,2)", ("a" * 64,))
        db.executemany("INSERT INTO attempts VALUES(?,?,?,?,?)", [(1, 10, 9, "VERIFIED", "2021-01-01T00:00:00Z"), (2, 15, 0, "PRECHARGED", "2021-01-01T00:00:01Z")])
    return path


class ProgressTests(unittest.TestCase):
    def test_partial_progress_is_verified_without_opening_numeric_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = fixture(directory)
            real_open = INSPECT.os.open
            def no_models(path, *args, **kwargs):
                if str(path).endswith("model.npz"):
                    raise AssertionError("Numeric payload was opened")
                return real_open(path, *args, **kwargs)
            with patch.object(INSPECT.os, "open", side_effect=no_models):
                result = INSPECT.inspect_progress(corpus_directory=root, expected_plan_sha256=plan["plan_sha256"])
            p = result["progress"]
            self.assertEqual((p["committed_chunks"], p["committed_origins"], p["committed_cases"], p["admitted_cases_from_case_indices"]), (1, 2, 6, 5))
            self.assertEqual(p["missing_chunks"], 1)
            self.assertEqual(p["contiguous_committed_through_utc"], "2021-01-01T01:00:00Z")
            self.assertEqual(p["model_payload_bytes_read"], 0)
            self.assertNotIn(str(root), json.dumps(result))

    def test_complete_metadata_does_not_claim_payload_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = fixture(directory, (0, 1))
            p = INSPECT.inspect_progress(corpus_directory=root, expected_plan_sha256=plan["plan_sha256"])["progress"]
            self.assertEqual(p["status"], "ALL_PLANNED_CHUNK_METADATA_COMMITTED")
            self.assertEqual((p["committed_origins"], p["committed_cases"]), (4, 12))
            self.assertIn("no payload hash", p["verification_scope"])

    def test_corrupt_case_index_is_excluded_and_breaks_contiguous_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = fixture(directory, (0, 1))
            path = root / "chunks" / plan["units"][0]["id"] / "000/case-index.json"
            path.write_text('{}')
            p = INSPECT.inspect_progress(corpus_directory=root, expected_plan_sha256=plan["plan_sha256"])["progress"]
            self.assertEqual((p["invalid_chunks"], p["committed_chunks"], p["contiguous_committed_chunks"]), (1, 1, 0))
            self.assertEqual(p["last_committed_origin_utc"], "2021-01-01T03:00:00Z")
            self.assertEqual(p["issues"][0]["code"], "CASE_INDEX_HASH_MISMATCH")

    def test_duplicate_payload_claims_cannot_double_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = fixture(directory, (0, 1))
            unit = root / "chunks" / plan["units"][0]["id"]
            first = json.loads((unit / "000/COMMIT.json").read_text())
            second = json.loads((unit / "001/COMMIT.json").read_text())
            second["payloads"]["model.npz"] = first["payloads"]["model.npz"]
            write_json(unit / "001/COMMIT.json", second)
            p = INSPECT.inspect_progress(corpus_directory=root, expected_plan_sha256=plan["plan_sha256"])["progress"]
            self.assertEqual((p["invalid_chunks"], p["committed_origins"]), (2, 0))
            self.assertTrue(all(i["code"] == "DUPLICATE_MODEL_PAYLOAD_CLAIM" for i in p["issues"]))

    def test_missing_model_member_is_not_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = fixture(directory)
            (root / "chunks" / plan["units"][0]["id"] / "000/model.npz").unlink()
            result = INSPECT.inspect_progress(corpus_directory=root, expected_plan_sha256=plan["plan_sha256"])
            self.assertEqual(result["progress"]["committed_chunks"], 0)
            self.assertEqual(result["progress"]["invalid_chunks"], 1)

    def test_wrong_plan_and_duplicate_unit_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = fixture(directory)
            result = INSPECT.inspect_progress(corpus_directory=root, expected_plan_sha256="f" * 64)
            self.assertIsNone(result["progress"])
            plan["units"].append(copy.deepcopy(plan["units"][0]))
            plan["plan_sha256"] = INSPECT.digest({key: plan[key] for key in INSPECT.PLAN_KEYS})
            write_json(root / "PLAN.json", plan)
            result = INSPECT.inspect_progress(corpus_directory=root, expected_plan_sha256=plan["plan_sha256"])
            self.assertEqual(result["error_code"], "DUPLICATE_OR_UNSAFE_PLAN_UNIT")

    def test_missing_plan_is_unknown_not_zero_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = INSPECT.inspect_progress(corpus_directory=directory, expected_plan_sha256="a" * 64)
            self.assertIsNone(result["progress"])
            self.assertEqual(result["status"], "PLAN_OR_PROGRESS_UNAVAILABLE")

    def test_read_only_budget_keeps_precharged_attempt_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            path = budget(directory)
            before = INSPECT.sha(path.read_bytes())
            result = INSPECT.read_budget(path, "a" * 64)
            self.assertEqual(result["status"], "RECONCILED_READ_ONLY")
            self.assertEqual(result["attempt_states"], {"PRECHARGED": 1, "VERIFIED": 1})
            self.assertEqual(result["pending_precharged_bytes"], 15)
            self.assertEqual(result["settlements_performed"], 0)
            self.assertEqual(INSPECT.sha(path.read_bytes()), before)
            with test_database(path) as db:
                self.assertEqual(db.execute("SELECT status FROM attempts WHERE id=2").fetchone()[0], "PRECHARGED")

    def test_budget_errors_do_not_reset_create_or_expose_raw_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = budget(directory)
            self.assertIsNone(INSPECT.read_budget(path, "b" * 64)["totals"])
            with test_database(path) as db:
                db.execute("UPDATE budget SET charged_body=26")
            self.assertEqual(INSPECT.read_budget(path)["error_code"], "BUDGET_TOTALS_DO_NOT_RECONCILE")
            missing = Path(directory) / "missing.sqlite3"
            self.assertIsNone(INSPECT.read_budget(missing)["totals"])
            self.assertFalse(missing.exists())

    def test_failure_snapshot_is_attributed_stale_and_redacts_unapproved_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "increment.json"
            value = dict(created_at_utc="2001-01-01T00:00:00Z", corpus=dict(schema=INSPECT.SCHEMA, plan_sha256="a" * 64,
                         created_at_utc="2001-01-01T00:00:00Z", completed_origins=0, status="STOPPED_WITH_EXPLICIT_FAILURE",
                         failure=dict(error_type="SourceUnavailable", source_status="SOURCE_TRANSPORT_FAILED", cause_type="HTTPError",
                                      cycle_id="20190730_18Z", error_message="private-secret-text")))
            write_json(path, value)
            result = INSPECT.last_increment(path, "a" * 64, {"committed_origins": 2})
            self.assertTrue(result["stale_by_age"])
            self.assertTrue(result["committed_progress_after_snapshot"])
            self.assertEqual(result["reported_failure"]["cause_type"], "HTTPError")
            self.assertNotIn("private-secret-text", json.dumps(result))
            self.assertEqual(INSPECT.last_increment(path, "b" * 64, {})["status"], "INCREMENT_UNAVAILABLE_OR_INVALID")

    def test_no_worker_is_inferred_and_absent_process_is_reported(self):
        with patch.object(INSPECT.subprocess, "run") as run:
            self.assertEqual(INSPECT.observe_worker()["status"], "UNVERIFIED_NOT_REQUESTED")
            run.assert_not_called()
        with patch.object(INSPECT.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")):
            self.assertEqual(INSPECT.observe_worker(expected_pid=123)["status"], "PROCESS_NOT_OBSERVED")

    def test_process_requires_command_checkout_and_run_association(self):
        command = "/python -B scripts/stage_development.py --staging-plan .local/plan.json --execute --approved-execution-plan-sha256 " + "a" * 64 + " --corpus-run-name corpus"
        with tempfile.TemporaryDirectory() as directory, patch.object(INSPECT.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=command)):
            self.assertEqual(INSPECT.observe_worker(expected_pid=123)["status"], "PROCESS_OBSERVED_IDENTITY_UNVERIFIED")
            kwargs = dict(expected_pid=123, expected_command_sha256=INSPECT.sha(command.encode()), expected_cwd=directory, corpus_run_name="corpus")
            with patch.object(INSPECT, "process_cwd", return_value=directory):
                result = INSPECT.observe_worker(**kwargs)
                self.assertEqual(result["status"], "VERIFIED_CALLER_PROCESS_AT_SNAPSHOT")
                self.assertNotIn(command, json.dumps(result))
            with patch.object(INSPECT, "process_cwd", return_value=directory + "/other"):
                self.assertEqual(INSPECT.observe_worker(**kwargs)["status"], "PROCESS_IDENTITY_MISMATCH")
            with patch.object(INSPECT, "process_cwd", side_effect=PermissionError):
                self.assertEqual(INSPECT.observe_worker(**kwargs)["status"], "PROCESS_OBSERVED_IDENTITY_UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
