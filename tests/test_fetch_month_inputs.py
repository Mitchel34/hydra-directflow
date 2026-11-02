"""Storage-transfer safety/integrity tests with in-memory responses; no network."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_month_inputs", ROOT / "scripts/fetch_month_inputs.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def identity(body=b"test"):
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            "durable_uri": "gs://example-test-bucket/cycles/20230801_00Z.tar", "generation": "12345"}


class Response(io.BytesIO):
    def __init__(self, body, obj, generation=None):
        super().__init__(body)
        self.status = 200
        self.headers = {"Content-Length": str(obj["bytes"]), "x-goog-generation": generation or obj["generation"]}


class Client:
    def __init__(self, body=b"test", generation=None, fail=False):
        self.calls, self.body, self.generation, self.fail = 0, body, generation, fail

    def open(self, obj):
        self.calls += 1
        if self.fail:
            raise OSError("synthetic transport failure")
        return Response(self.body, obj, self.generation)


class FetchTests(unittest.TestCase):
    def test_url_is_only_google_storage_and_generation_pinned(self):
        obj = identity()
        url = fetch.media_url(fetch.object_identity(obj, 100))
        self.assertEqual(url, "https://storage.googleapis.com/example-test-bucket/cycles/20230801_00Z.tar?generation=12345")
        for change in ({"durable_uri": "https://elsewhere.invalid/data"}, {"generation": ""},
                       {"generation": "1&redirect=evil"}, {"bytes": True}, {"sha256": "bad"}):
            with self.assertRaises(ValueError):
                fetch.object_identity({**obj, **change}, 100)

    def test_redirect_is_rejected_before_any_followup(self):
        with self.assertRaisesRegex(ValueError, "redirect"):
            fetch.DenyRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.invalid")

    def test_verified_transfer_is_atomic_and_reuse_has_no_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = fetch.Budget(directory, "a" * 64)
            target, client = Path(directory) / "object.tar", Client()
            self.assertTrue(fetch.transfer_object(identity(), target, budget, client))
            self.assertEqual(target.read_bytes(), b"test")
            self.assertFalse(target.with_name("object.tar.partial").exists())
            self.assertEqual(budget.value["charged_body_bytes"], 5)
            self.assertEqual(budget.value["attempts"][0]["status"], "VERIFIED")
            self.assertFalse(fetch.transfer_object(identity(), target, budget, client))
            self.assertEqual(client.calls, 1)
            self.assertEqual(budget.value["charged_body_bytes"], 5)

    def test_durable_precharge_precedes_get_and_durable_object_precedes_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            target = Path(directory) / "object.tar"
            ledger = Path(directory) / "transfer-ledger.json"
            actual_sync = fetch.sync_directory
            def record_sync(parent):
                actual_sync(parent)
                value = json.loads(ledger.read_text())
                latest = value["attempts"][-1]["status"] if value["attempts"] else "EMPTY"
                events.append(("directory_synced", latest, target.exists()))
            class OrderedClient(Client):
                def open(self, obj):
                    self_test.assertEqual(events[-1], ("directory_synced", "PRECHARGED_UNSETTLED", False))
                    events.append(("GET",))
                    return super().open(obj)
            self_test = self
            with patch.object(fetch, "sync_directory", side_effect=record_sync):
                budget = fetch.Budget(directory, "a" * 64)
                fetch.transfer_object(identity(), target, budget, OrderedClient())
            self.assertEqual(events, [
                ("directory_synced", "EMPTY", False),
                ("directory_synced", "PRECHARGED_UNSETTLED", False),
                ("GET",),
                ("directory_synced", "PRECHARGED_UNSETTLED", True),
                ("directory_synced", "VERIFIED", True),
            ])

    def test_precharge_directory_sync_failure_prohibits_network(self):
        with tempfile.TemporaryDirectory() as directory:
            budget, client = fetch.Budget(directory, "b" * 64), Client()
            with patch.object(fetch, "sync_directory", side_effect=OSError("Synthetic directory sync failure")):
                with self.assertRaises(OSError):
                    fetch.transfer_object(identity(), Path(directory) / "object.tar", budget, client)
            self.assertEqual(client.calls, 0)
            self.assertEqual(budget.value["charged_body_bytes"], 5)

    def test_failed_transport_keeps_precharge_across_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "object.tar"
            budget = fetch.Budget(directory, "b" * 64)
            with self.assertRaises(OSError):
                fetch.transfer_object(identity(), target, budget, Client(fail=True))
            resumed = fetch.Budget(directory, "b" * 64)
            self.assertEqual(resumed.value["charged_body_bytes"], 5)
            self.assertEqual(resumed.value["attempts"][0]["status"], "FAILED_PRECHARGE_RETAINED")
            fetch.transfer_object(identity(), target, resumed, Client())
            self.assertEqual(resumed.value["charged_body_bytes"], 10)
            self.assertEqual(len(resumed.value["attempts"]), 2)

    def test_wrong_generation_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = fetch.Budget(directory, "c" * 64)
            target = Path(directory) / "object.tar"
            with self.assertRaisesRegex(ValueError, "generation"):
                fetch.transfer_object(identity(), target, budget, Client(generation="999"))
            self.assertFalse(target.exists())
            self.assertEqual(budget.value["attempts"][0]["observed_body_bytes"], 0)
            self.assertEqual(budget.value["charged_body_bytes"], 5)

    def test_wrong_content_and_oversized_body_retain_charge_and_remove_owned_partial(self):
        for body in (b"FAIL", b"test-extra"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                budget = fetch.Budget(directory, "d" * 64)
                target = Path(directory) / "object.tar"
                with self.assertRaises(ValueError):
                    fetch.transfer_object(identity(), target, budget, Client(body))
                self.assertFalse(target.exists())
                self.assertFalse(target.with_name("object.tar.partial").exists())
                self.assertEqual(budget.value["charged_body_bytes"], 5)
                self.assertLessEqual(budget.value["attempts"][0]["observed_body_bytes"], 5)

    def test_budget_prevents_get_before_byte_or_attempt_overrun(self):
        for caps in (dict(BYTE_CAP=4), dict(OBJECT_CAP=0)):
            with self.subTest(caps=caps), tempfile.TemporaryDirectory() as directory, patch.multiple(fetch, **caps):
                budget, client = fetch.Budget(directory, "e" * 64), Client()
                with self.assertRaisesRegex(ValueError, "budget"):
                    fetch.transfer_object(identity(), Path(directory) / "object.tar", budget, client)
                self.assertEqual(client.calls, 0)
                self.assertEqual(budget.value["charged_body_bytes"], 0)

    def test_wall_limit_rejects_before_get(self):
        with tempfile.TemporaryDirectory() as directory:
            stamps = iter([0., 901.])
            budget = fetch.Budget(directory, "f" * 64, clock=lambda: next(stamps))
            client = Client()
            with self.assertRaisesRegex(ValueError, "wall"):
                fetch.transfer_object(identity(), Path(directory) / "object.tar", budget, client)
            self.assertEqual(client.calls, 0)

    def test_strict_wall_alarm_interrupts_blocking_work_and_restores_handler(self):
        import time
        previous = fetch.signal.getsignal(fetch.signal.SIGALRM)
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            with fetch.wall_alarm(.02):
                time.sleep(1.)
        self.assertEqual(fetch.signal.getitimer(fetch.signal.ITIMER_REAL)[0], 0.)
        self.assertEqual(fetch.signal.getsignal(fetch.signal.SIGALRM), previous)

    def test_resume_cannot_change_plan_or_reset_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            fetch.Budget(directory, "a" * 64)
            with self.assertRaisesRegex(ValueError, "another plan"):
                fetch.Budget(directory, "b" * 64)

    def test_existing_bad_cache_and_symlinks_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "object.tar"
            target.write_bytes(b"bad")
            client, budget = Client(), fetch.Budget(directory, "a" * 64)
            with self.assertRaisesRegex(ValueError, "failed authentication"):
                fetch.transfer_object(identity(), target, budget, client)
            self.assertEqual(target.read_bytes(), b"bad")
            link = Path(directory) / "link.tar"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                fetch.transfer_object(identity(), link, budget, client)
            self.assertEqual(client.calls, 0)

    def test_cache_requires_ignored_dedicated_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(fetch, "PRIVATE_ROOT", Path(directory) / ".local"):
            self.assertEqual(fetch.private_destination(Path(directory) / ".local" / "month"), Path(directory) / ".local" / "month")
            for path in (Path(directory) / "public", Path(directory) / ".local"):
                with self.assertRaises(ValueError):
                    fetch.private_destination(path)

    def test_authenticated_local_commit_binds_bundle_and_never_calls_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "commits").mkdir()
            bundle = identity()
            commit = {"schema": "hrrr-verified-cloud-cycle/v1", "cycle": "20230801_00Z", "bundle": {**bundle, "verified": True},
                      "worker_manifest": {"cycle": "20230801_00Z", "bundle_sha256": bundle["sha256"], "bundle_bytes": bundle["bytes"]}}
            raw = json.dumps(commit).encode()
            source = root / "original.json"
            source.write_bytes(raw)
            obj = {**identity(raw), "durable_uri": "gs://example-test-bucket/commits/20230801_00Z.json"}
            budget, client = fetch.Budget(root, "a" * 64), Client()
            chosen, actual, got = fetch.authenticated_commit({"cycle_id": "20230801_00Z", "commit_path": str(source)},
                {"source_locations": {"commit": obj}}, root, budget, client)
            self.assertEqual(chosen, source)
            self.assertEqual(actual, bundle)
            self.assertFalse(got)
            self.assertEqual(client.calls, 0)

    def test_nested_cache_directories_are_durable_before_ledger_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "new-parent" / "month" / "commits"
            actual_sync, synced = fetch.sync_directory, []
            def record(parent):
                actual_sync(parent)
                synced.append(Path(parent))
            with patch.object(fetch, "sync_directory", side_effect=record):
                fetch.durable_directory(target)
            self.assertEqual(synced, [root, root / "new-parent", root / "new-parent" / "month"])
            self.assertTrue(target.is_dir())

    def test_inventory_cannot_replace_sealed_object_identity(self):
        from datetime import datetime, timedelta, timezone
        start = datetime(2023, 7, 31, 6, tzinfo=timezone.utc)
        rows, sealed = [], {}
        for offset in range(127):
            cycle = (start + timedelta(hours=6 * offset)).strftime("%Y%m%d_%HZ")
            locations = {"commit": identity()}
            row = {"cycle_id": cycle, "source_locations": locations, "location_kind": "commit", "data_complete": True}
            rows.append(row)
            sealed[cycle] = {"source_locations": {"commit": {**identity(), "generation": "999"}},
                             "location_kind": "commit", "data_complete": True}
        inventory = {"schema": "hydra-private-month-retrieval-inventory/v1", "month": "2023-08",
                     "qualified_index_sha256": fetch.INDEX_SHA256, "cycles": rows}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            path.write_text(json.dumps(inventory))
            with patch.object(fetch, "qualified_cycle_rows", return_value=sealed):
                with self.assertRaisesRegex(ValueError, "sealed index"):
                    fetch.load_inventory(path, "unused")

    def test_concurrent_execution_cannot_share_or_reset_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "transfer.lock").open("a") as existing:
                fetch.fcntl.flock(existing, fetch.fcntl.LOCK_EX | fetch.fcntl.LOCK_NB)
                with patch.object(fetch, "make_plan", return_value=({"plan_sha256": "a" * 64}, {}, {}, root)), \
                     patch.object(fetch, "_execute_locked") as transfer:
                    with self.assertRaisesRegex(ValueError, "Another transfer"):
                        fetch.execute("inventory", "index", root, weights_path="weights", usgs_manifest_path="manifest",
                                      approved_plan_sha256="a" * 64)
                    transfer.assert_not_called()
                fetch.fcntl.flock(existing, fetch.fcntl.LOCK_UN)

    def test_cli_default_does_not_execute_or_request_credentials(self):
        with patch.object(sys := fetch.sys, "argv", ["fetch", "--inventory", "private.json", "--index", "index.jsonl",
                    "--cache-directory", ".local/month", "--weights", "weights.csv", "--usgs-manifest", "usgs.json"]), \
             patch.object(fetch, "make_plan", return_value=({"status": "PLANNED_NOT_EXECUTED"}, {}, {}, Path("unused"))), \
             patch.object(fetch, "execute") as execute, patch.object(fetch.subprocess, "run") as auth, \
             patch("builtins.print"):
            fetch.main()
            execute.assert_not_called()
            auth.assert_not_called()


if __name__ == "__main__":
    unittest.main()
