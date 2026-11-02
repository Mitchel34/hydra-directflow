"""Persistent read-budget and owned-staging tests; all HTTP clients are mocked."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra_directflow.staging import (SourceBudget, DevelopmentSourceProvider, object_identity,
                                     conservative_cost_microusd, MAX_BODY, MAX_GETS)


def identity(raw, *, size=True):
    return dict(durable_uri="gs://test-bucket/cycles/20230801_00Z.tar", generation="123",
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw) if size else None)


class Response(io.BytesIO):
    status = 200
    def __init__(self, raw):
        super().__init__(raw)
        self.headers = {"x-goog-generation": "123", "Content-Length": str(len(raw))}


class Client:
    def __init__(self, raw, callback=None):
        self.raw, self.calls, self.callback = raw, 0, callback
    def open(self, source):
        self.calls += 1
        if self.callback: self.callback()
        return Response(self.raw)


class StagingTests(unittest.TestCase):
    def test_failed_and_unsettled_precharges_survive_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = SourceBudget(folder, "a" * 64, body_cap=20, get_cap=2)
            one = budget.reserve(identity(b"one"), 4, "archive")
            budget.settle(one, "FAILED", 1)
            budget.reserve(identity(b"two"), 4, "archive")
            budget.close()
            resumed = SourceBudget(folder, "a" * 64, body_cap=20, get_cap=2)
            self.assertEqual(resumed.snapshot()["precharged_body_bytes"], 8)
            self.assertEqual(resumed.snapshot()["attempt_states"], {"FAILED": 1, "PRECHARGED": 1})
            with self.assertRaisesRegex(ValueError, "exhausted"):
                resumed.reserve(identity(b"three"), 6, "archive")
            self.assertEqual(resumed.snapshot()["attempted_gets"], 2)
            resumed.close()

    def test_budget_identity_or_caps_cannot_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            SourceBudget(folder, "a" * 64, body_cap=20, get_cap=2).close()
            with self.assertRaisesRegex(ValueError, "reset"):
                SourceBudget(folder, "a" * 64, body_cap=21, get_cap=2)

    def test_symlinked_budget_is_rejected_before_connection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); original = root / "original"; original.write_bytes(b"keep")
            (root / "source-budget.sqlite3").symlink_to(original)
            with self.assertRaisesRegex(ValueError, "never links"):
                SourceBudget(root, "a" * 64)
            self.assertEqual(original.read_bytes(), b"keep")

    def test_conservative_maximum_is_inside_approved_staging_dollars(self):
        self.assertEqual(conservative_cost_microusd(MAX_BODY, MAX_GETS), 29300000)
        self.assertLess(conservative_cost_microusd(MAX_BODY, MAX_GETS), 35000000)

    def test_uri_generation_and_size_guards(self):
        valid = identity(b"abc", size=False)
        self.assertIsNone(object_identity(valid, 16, bytes_required=False, allowed_buckets={"test-bucket"})["bytes"])
        with self.assertRaises(ValueError):
            object_identity(valid, 16, bytes_required=True, allowed_buckets={"test-bucket"})
        with self.assertRaises(ValueError):
            object_identity(identity(b"abc"), 16, bytes_required=True, allowed_buckets={"different"})
        with self.assertRaises(ValueError):
            object_identity({**valid, "generation": ""}, 16, bytes_required=False, allowed_buckets={"test-bucket"})

    def provider(self, folder, raw, callback=None):
        root = Path(folder)
        fake_file = root / "src/hydra_directflow/staging.py"
        plan = dict(schema="hydra-development-staging-plan/v1", sources={"budget_approval": {"sha256": "a" * 64}, "qualified_index": {"sha256": "b" * 64}},
                    cycles=[dict(cycle_id="20230801_00Z", bundle_identity=identity(raw), local_archive_candidates=[])])
        budget = SourceBudget(root / "budget", "a" * 64, body_cap=100, get_cap=3)
        with patch("hydra_directflow.staging.__file__", str(fake_file)):
            provider = DevelopmentSourceProvider(plan=plan, directory=root / ".local/stage", budget=budget, client=Client(raw, callback))
        return provider, budget

    def test_durable_precharge_exists_before_mock_http_and_verified_hit_has_no_get(self):
        with tempfile.TemporaryDirectory() as folder:
            provider, budget = self.provider(folder, b"abc")
            def before_get():
                observer = SourceBudget(Path(folder) / "budget", "a" * 64, body_cap=100, get_cap=3)
                self.assertEqual(observer.snapshot()["precharged_body_bytes"], 4)
                observer.close()
            provider.client.callback = before_get
            with provider:
                target = provider.directory / "raw/20230801_00Z.tar"
                provider._transfer(identity(b"abc"), target, 16, "archive")
                provider._transfer(identity(b"abc"), target, 16, "archive")
                self.assertEqual(provider.client.calls, 1)
                self.assertEqual(budget.snapshot()["attempt_states"], {"VERIFIED": 1})
                self.assertTrue(target.exists())
            budget.close()

    def test_bad_sha_consumes_charge_and_does_not_publish_or_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            provider, budget = self.provider(folder, b"bad")
            with provider:
                target = provider.directory / "raw/20230801_00Z.tar"
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    provider._transfer(identity(b"abc"), target, 16, "archive")
                self.assertFalse(target.exists())
                self.assertEqual(provider.client.calls, 1)
                self.assertEqual(budget.snapshot()["precharged_body_bytes"], 4)
                self.assertEqual(budget.snapshot()["attempt_states"], {"FAILED": 1})
            budget.close()

    def test_unknown_commit_size_uses_full_bounded_precharge(self):
        with tempfile.TemporaryDirectory() as folder:
            provider, budget = self.provider(folder, b"abc")
            with provider:
                provider._transfer(identity(b"abc", size=False), provider.directory / "commits/id.json", 16, "commit")
                self.assertEqual(budget.snapshot()["precharged_body_bytes"], 17)
                self.assertEqual(budget.snapshot()["observed_body_bytes"], 3)
            budget.close()

    def test_only_matching_owned_copy_can_be_evicted_after_callback(self):
        with tempfile.TemporaryDirectory() as folder:
            provider, budget = self.provider(folder, b"abc")
            with provider:
                target = provider.directory / "raw/20230801_00Z.tar"
                provider._transfer(identity(b"abc"), target, 16, "archive")
                ownership = dict(owned_archive=True, archive_path=str(target), owner_run_id=provider.owner,
                                 archive_sha256=identity(b"abc")["sha256"])
                with self.assertRaises(ValueError):
                    provider.on_cache_committed({"archive_sha256": "f" * 64}, {"ownership": ownership})
                self.assertTrue(target.exists())
                provider.on_cache_committed({"archive_sha256": ownership["archive_sha256"]}, {"ownership": {**ownership, "owned_archive": False}})
                self.assertTrue(target.exists())
                provider.on_cache_committed({"archive_sha256": ownership["archive_sha256"], "cache_payload_verified": True,
                                             "cache_key": "c" * 64, "cache_commit_sha256": "d" * 64}, {"ownership": ownership})
                self.assertFalse(target.exists())
            budget.close()


if __name__ == "__main__":
    unittest.main()
