"""Owned-file materialization tests use tiny synthetic files, never iCloud data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("materialize_owned_usgs", ROOT / "scripts/materialize_owned_usgs.py")
materialize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materialize)


def fixture(root, body=b"synthetic"):
    source = root / "source.parquet"
    source.write_bytes(body)
    output = root / "provider" / "usgs"
    output.mkdir(parents=True)
    return {"month": "2018-12", "source_path": str(source), "destination": str(output / "2018-12.parquet"),
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}, output


class MaterializationTests(unittest.TestCase):
    def test_primitive_requires_execution_and_refuses_boundary_month(self):
        with tempfile.TemporaryDirectory() as directory:
            item, _ = fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "explicit"):
                materialize.copy_exact_owned_source(item)
            with self.assertRaisesRegex(ValueError, "approved native month"):
                materialize.copy_exact_owned_source({**item, "month": "2024-07"}, execution_enabled=True)
            self.assertFalse(Path(item["destination"]).exists())

    def test_verified_copy_preserves_original_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            item, _ = fixture(Path(directory))
            before = Path(item["source_path"]).read_bytes()
            result = materialize.copy_exact_owned_source(item, execution_enabled=True)
            self.assertEqual(Path(item["source_path"]).read_bytes(), before)
            self.assertEqual(Path(item["destination"]).read_bytes(), before)
            self.assertEqual(result["encoded_bytes_read"], len(before))
            self.assertFalse(result["flow_columns_decoded"])
            with self.assertRaisesRegex(ValueError, "must be new"):
                materialize.copy_exact_owned_source(item, execution_enabled=True)

    def test_wrong_hash_and_symlink_source_never_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item, _ = fixture(root)
            with self.assertRaisesRegex(ValueError, "pinned"):
                materialize.copy_exact_owned_source({**item, "sha256": "0" * 64}, execution_enabled=True)
            link = root / "link.parquet"
            link.symlink_to(item["source_path"])
            with self.assertRaises(ValueError):
                materialize.copy_exact_owned_source({**item, "source_path": str(link)}, execution_enabled=True)
            self.assertFalse(Path(item["destination"]).exists())

    def test_shared_hotset_guard_precedes_source_open(self):
        with tempfile.TemporaryDirectory() as directory:
            item, output = fixture(Path(directory))
            (output / "existing.parquet").write_bytes(b"12345")
            with patch.object(materialize, "RAW_HOTSET", 5), patch.object(materialize.os, "open", wraps=materialize.os.open) as opened:
                with self.assertRaisesRegex(ValueError, "hotset"):
                    materialize.copy_exact_owned_source(item, execution_enabled=True)
                opened.assert_not_called()

    def test_precharge_is_durable_before_child_and_verified_resume_has_no_child(self):
        with tempfile.TemporaryDirectory() as directory:
            item, output = fixture(Path(directory))
            fake = ({"files": [item]}, {"plan_sha256": "a" * 64, "status": "PLAN_ONLY_NOT_EXECUTED"})
            ledger_path = output.parent / "usgs-materialization-control/ledger.json"
            def child(*args, **kwargs):
                ledger = json.loads(ledger_path.read_text())
                self.assertEqual(ledger["attempts"][-1]["status"], "PRECHARGED")
                self.assertEqual(ledger["reserved_encoded_bytes"], item["bytes"] + 1)
                request = json.loads(Path(args[0][-1]).read_text())
                self.assertEqual(kwargs["pass_fds"], (request["provider_lock_fd"],))
                materialize.claim_attempt(ledger_path.parent, "a" * 64, request["attempt_id"], item)
                result = {**materialize.copy_exact_owned_source(item, execution_enabled=True), "attempt_id": request["attempt_id"]}
                return SimpleNamespace(returncode=0, stdout=json.dumps(result))
            with patch.object(materialize, "plan_materialization", return_value=fake), patch.object(materialize.subprocess, "run", side_effect=child) as run:
                first = materialize.execute_materialization("plan", "approval", output, approved_plan_sha256="a" * 64)
                again = materialize.execute_materialization("plan", "approval", output, approved_plan_sha256="a" * 64)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(first["status"], "VERIFIED_OWNED_USGS_COPIES_READY")
            self.assertEqual(again["reused_files"], 1)
            self.assertEqual(again["reserved_encoded_bytes"], item["bytes"] + 1)
            self.assertIsNone(first["fileprovider_http_requests"])

    def test_timeout_keeps_charge_and_explicit_attempts_are_bounded_across_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            item, output = fixture(Path(directory))
            fake = ({"files": [item]}, {"plan_sha256": "b" * 64, "status": "PLAN_ONLY_NOT_EXECUTED"})
            with patch.object(materialize, "plan_materialization", return_value=fake), \
                 patch.object(materialize.subprocess, "run", side_effect=subprocess.TimeoutExpired("synthetic", 60)) as run:
                for number in range(1, 4):
                    result = materialize.execute_materialization("plan", "approval", output, approved_plan_sha256="b" * 64)
                    self.assertEqual(result["status"], "STOPPED_WITH_RETAINED_CHARGES")
                    self.assertEqual(result["reserved_encoded_bytes"], number * (item["bytes"] + 1))
                with self.assertRaisesRegex(ValueError, "allowance exhausted"):
                    materialize.execute_materialization("plan", "approval", output, approved_plan_sha256="b" * 64)
            self.assertEqual(run.call_count, 3)
            self.assertFalse(Path(item["destination"]).exists())

    def test_shared_provider_lock_prevents_concurrent_raw_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            item, output = fixture(Path(directory))
            fake = ({"files": [item]}, {"plan_sha256": "c" * 64, "status": "PLAN_ONLY_NOT_EXECUTED"})
            with (output.parent / "provider.lock").open("a+") as provider:
                materialize.fcntl.flock(provider.fileno(), materialize.fcntl.LOCK_EX | materialize.fcntl.LOCK_NB)
                with patch.object(materialize, "plan_materialization", return_value=fake), patch.object(materialize.subprocess, "run") as run:
                    with self.assertRaises(BlockingIOError):
                        materialize.execute_materialization("plan", "approval", output, approved_plan_sha256="c" * 64)
                    run.assert_not_called()
                materialize.fcntl.flock(provider.fileno(), materialize.fcntl.LOCK_UN)

    def test_changed_plan_cannot_reset_materialization_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            item, output = fixture(Path(directory))
            with patch.object(materialize, "plan_materialization", return_value=({"files": [item]}, {"plan_sha256": "a" * 64})), \
                 patch.object(materialize.subprocess, "run", side_effect=subprocess.TimeoutExpired("synthetic", 60)):
                materialize.execute_materialization("plan", "approval", output, approved_plan_sha256="a" * 64)
            with patch.object(materialize, "plan_materialization", return_value=({"files": [item]}, {"plan_sha256": "b" * 64})):
                with self.assertRaisesRegex(ValueError, "cannot reset"):
                    materialize.execute_materialization("plan", "approval", output, approved_plan_sha256="b" * 64)

    def test_worker_rejects_unreserved_settled_and_replayed_requests_before_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            item, output = fixture(Path(directory))
            control = output.parent / "usgs-materialization-control"
            control.mkdir()
            ledger_path = control / "ledger.json"
            ledger = {"plan_sha256": "d" * 64, "reserved_encoded_bytes": 0, "attempts": []}
            materialize.atomic_json(ledger_path, ledger)
            fake = ({"files": [item]}, {"plan_sha256": "d" * 64})
            with (output.parent / "provider.lock").open("a+") as lock:
                materialize.fcntl.flock(lock.fileno(), materialize.fcntl.LOCK_EX | materialize.fcntl.LOCK_NB)
                request = {"staging_plan_path": "synthetic-plan", "approval_path": "synthetic-approval", "output_directory": str(output),
                           "month": item["month"], "plan_sha256": "d" * 64, "provider_lock_fd": lock.fileno(), "attempt_id": "unreserved"}
                path = control / "request.json"
                materialize.atomic_json(path, request)
                with patch.object(materialize, "plan_materialization", return_value=fake), patch.object(materialize, "copy_exact_owned_source") as copy:
                    with self.assertRaisesRegex(ValueError, "precharged"):
                        materialize.worker(path)
                    copy.assert_not_called()
                    index = materialize.reserve_attempt(ledger, ledger_path, item)
                    request["attempt_id"] = ledger["attempts"][index]["attempt_id"]
                    ledger["attempts"][index]["status"] = "FAILED_OR_UNSETTLED"
                    materialize.atomic_json(ledger_path, ledger)
                    materialize.atomic_json(path, request)
                    with self.assertRaisesRegex(ValueError, "settled"):
                        materialize.worker(path)
                    copy.assert_not_called()
                    index = materialize.reserve_attempt(ledger, ledger_path, item)
                    request["attempt_id"] = ledger["attempts"][index]["attempt_id"]
                    materialize.atomic_json(path, request)
                    copy.return_value = {"status": "VERIFIED_OWNED_COPY", "month": item["month"]}
                    materialize.worker(path)
                    self.assertEqual(copy.call_count, 1)
                    claimed = json.loads(ledger_path.read_text())["attempts"][index]
                    self.assertEqual(claimed["status"], "WORKER_CLAIMED")
                    with self.assertRaisesRegex(ValueError, "claimed"):
                        materialize.worker(path)
                    self.assertEqual(copy.call_count, 1)
                materialize.fcntl.flock(lock.fileno(), materialize.fcntl.LOCK_UN)

    def test_default_cli_only_plans(self):
        argv = ["materialize", "--staging-plan", "private.json", "--approval", "approval.json", "--output-directory", ".local/stage/usgs"]
        with patch.object(materialize.sys, "argv", argv), \
             patch.object(materialize, "plan_materialization", return_value=({}, {"status": "PLAN_ONLY_NOT_EXECUTED"})) as plan, \
             patch.object(materialize, "execute_materialization") as execute, patch("builtins.print"):
            materialize.main()
        self.assertEqual(plan.call_count, 1)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
