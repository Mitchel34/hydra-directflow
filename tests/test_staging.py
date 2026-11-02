"""Persistent read-budget and owned-staging tests; all HTTP clients are mocked."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import urllib.error
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace

from hydra_directflow.staging import (SourceBudget, DevelopmentSourceProvider, object_identity,
                                     conservative_cost_microusd, MAX_BODY, MAX_GETS, GCSClient)


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

    def provider(self, folder, raw, callback=None, *, get_cap=3, max_attempts=3):
        root = Path(folder)
        fake_file = root / "src/hydra_directflow/staging.py"
        plan = dict(schema="hydra-development-staging-plan/v1", sources={"budget_approval": {"sha256": "a" * 64}, "qualified_index": {"sha256": "b" * 64}},
                    cycles=[dict(cycle_id="20230801_00Z", bundle_identity=identity(raw), local_archive_candidates=[])])
        budget = SourceBudget(root / "budget", "a" * 64, body_cap=100, get_cap=get_cap)
        with patch("hydra_directflow.staging.__file__", str(fake_file)):
            provider = DevelopmentSourceProvider(plan=plan, directory=root / ".local/stage", budget=budget, client=Client(raw, callback),
                                                  max_attempts_per_object=max_attempts)
        return provider, budget

    def token_client(self, budget, statuses):
        client = GCSClient()
        old, new = "OLD_FAKE_PRIVATE_TOKEN_123456789", "NEW_FAKE_PRIVATE_TOKEN_123456789"
        client.token, client.token_time = old, time.monotonic()
        requests, sequence = [], iter(statuses)
        def open_response(request, timeout):
            requests.append(request)
            # Every HTTP attempt, including one after credential refresh, has
            # its own committed reservation before reaching this fake opener.
            self.assertEqual(budget.snapshot()["attempted_gets"], len(requests))
            self.assertEqual(budget.snapshot()["precharged_body_bytes"], 4 * len(requests))
            self.assertEqual(budget.snapshot()["attempt_states"].get("PRECHARGED"), 1)
            code = next(sequence)
            if code != 200:
                raise urllib.error.HTTPError("https://example.invalid/FAKE_PRIVATE_PATH", code,
                    "FAKE_PRIVATE_ERROR_BODY_TOKEN", {"private": old}, io.BytesIO(b"FAKE_PRIVATE_BODY"))
            return Response(b"abc")
        client.opener = SimpleNamespace(open=open_response)
        def credential_command(command, **kwargs):
            self.assertEqual(command, ["/mock/gcloud", "auth", "print-access-token"])
            self.assertEqual(kwargs, {"capture_output": True, "text": True, "timeout": 20})
            self.assertEqual(budget.snapshot()["attempt_states"].get("PRECHARGED"), 1)
            return SimpleNamespace(returncode=0, stdout=new, stderr="FAKE_PRIVATE_STDERR")
        return client, requests, old, new, credential_command

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

    def test_401_refreshes_once_through_same_cli_after_new_precharge_and_keeps_failed_charge(self):
        with tempfile.TemporaryDirectory() as folder:
            provider, budget = self.provider(folder, b"abc")
            client, requests, old, new, credential = self.token_client(budget, [401, 200])
            provider.client = client
            output, errors = io.StringIO(), io.StringIO()
            with provider, patch("hydra_directflow.staging.shutil.which", return_value="/mock/gcloud"), \
                 patch("hydra_directflow.staging.subprocess.run", side_effect=credential) as command, \
                 patch.object(client, "invalidate_token", wraps=client.invalidate_token) as invalidated, \
                 redirect_stdout(output), redirect_stderr(errors):
                target = provider.directory / "raw/20230801_00Z.tar"
                provider._transfer(identity(b"abc"), target, 16, "archive")
                self.assertEqual(target.read_bytes(), b"abc")
            self.assertEqual((client.network_requests, client.credential_requests, invalidated.call_count, command.call_count), (2, 1, 1, 1))
            self.assertEqual(requests[0].get_header("Authorization"), "Bearer " + old)
            self.assertEqual(requests[1].get_header("Authorization"), "Bearer " + new)
            self.assertEqual(budget.snapshot()["precharged_body_bytes"], 8)
            self.assertEqual(budget.snapshot()["attempt_states"], {"FAILED": 1, "VERIFIED": 1})
            rows = budget.connection.execute("SELECT status,error_type,observed FROM attempts ORDER BY id").fetchall()
            self.assertEqual(rows, [("FAILED", "HTTPError_401", 0), ("VERIFIED", None, 3)])
            public = json.dumps({"budget": budget.snapshot(), "errors": rows}) + output.getvalue() + errors.getvalue()
            for secret in (old, new, "FAKE_PRIVATE_PATH", "FAKE_PRIVATE_ERROR_BODY_TOKEN", "FAKE_PRIVATE_BODY", "FAKE_PRIVATE_STDERR"):
                self.assertNotIn(secret, public)
            budget.close()

    def test_repeated401_stops_after_single_refresh_and_two_charged_gets(self):
        with tempfile.TemporaryDirectory() as folder:
            provider, budget = self.provider(folder, b"abc")
            client, _, _, _, credential = self.token_client(budget, [401, 401, 200])
            provider.client = client
            with provider, patch("hydra_directflow.staging.shutil.which", return_value="/mock/gcloud"), \
                 patch("hydra_directflow.staging.subprocess.run", side_effect=credential) as command, \
                 patch.object(client, "invalidate_token", wraps=client.invalidate_token) as invalidated:
                target = provider.directory / "raw/20230801_00Z.tar"
                with self.assertRaises(urllib.error.HTTPError) as error:
                    provider._transfer(identity(b"abc"), target, 16, "archive")
                self.assertEqual(error.exception.code, 401)
                self.assertFalse(target.exists())
            self.assertEqual((client.network_requests, command.call_count, invalidated.call_count), (2, 1, 1))
            self.assertEqual(budget.snapshot()["precharged_body_bytes"], 8)
            self.assertEqual(budget.snapshot()["attempt_states"], {"FAILED": 2})
            self.assertEqual(budget.connection.execute("SELECT error_type FROM attempts ORDER BY id").fetchall(), [("HTTPError_401",), ("HTTPError_401",)])
            budget.close()

    def test_403_never_refreshes_or_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            provider, budget = self.provider(folder, b"abc")
            client, _, old, _, credential = self.token_client(budget, [403, 200])
            provider.client = client
            with provider, patch("hydra_directflow.staging.subprocess.run", side_effect=credential) as command, \
                 patch.object(client, "invalidate_token", wraps=client.invalidate_token) as invalidated:
                with self.assertRaises(urllib.error.HTTPError) as error:
                    provider._transfer(identity(b"abc"), provider.directory / "raw/20230801_00Z.tar", 16, "archive")
                self.assertEqual(error.exception.code, 403)
            command.assert_not_called(); invalidated.assert_not_called()
            self.assertEqual(client.token, old)
            self.assertEqual(budget.snapshot()["attempted_gets"], 1)
            self.assertEqual(budget.connection.execute("SELECT error_type FROM attempts").fetchone()[0], "HTTPError_403")
            budget.close()

    def test_auth_refresh_and_transient_retry_share_the_existing_three_attempt_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            provider, budget = self.provider(folder, b"abc")
            client, _, _, _, credential = self.token_client(budget, [503, 401, 200])
            provider.client = client
            with provider, patch("hydra_directflow.staging.shutil.which", return_value="/mock/gcloud"), \
                 patch("hydra_directflow.staging.subprocess.run", side_effect=credential) as command, \
                 patch("hydra_directflow.staging.time.sleep") as sleep, patch.object(client, "invalidate_token", wraps=client.invalidate_token) as invalidated:
                provider._transfer(identity(b"abc"), provider.directory / "raw/20230801_00Z.tar", 16, "archive")
            self.assertEqual((client.network_requests, command.call_count, invalidated.call_count), (3, 1, 1))
            self.assertEqual(budget.snapshot()["precharged_body_bytes"], 12)
            self.assertEqual(budget.snapshot()["attempt_states"], {"FAILED": 2, "VERIFIED": 1})
            sleep.assert_called_once_with(1)
            budget.close()

    def test_existing_attempt_or_budget_limits_prevent_an_extra_get_without_refund(self):
        for max_attempts, get_cap, expected_error in ((1, 3, urllib.error.HTTPError), (3, 1, ValueError)):
            with self.subTest(max_attempts=max_attempts, get_cap=get_cap), tempfile.TemporaryDirectory() as folder:
                provider, budget = self.provider(folder, b"abc", get_cap=get_cap, max_attempts=max_attempts)
                client, _, _, _, credential = self.token_client(budget, [401, 200])
                provider.client = client
                with provider, patch("hydra_directflow.staging.subprocess.run", side_effect=credential) as command:
                    with self.assertRaises(expected_error):
                        provider._transfer(identity(b"abc"), provider.directory / "raw/20230801_00Z.tar", 16, "archive")
                command.assert_not_called()
                self.assertEqual(client.network_requests, 1)
                self.assertEqual(budget.snapshot()["precharged_body_bytes"], 4)
                self.assertEqual(budget.snapshot()["attempt_states"], {"FAILED": 1})
                budget.close()


if __name__ == "__main__":
    unittest.main()
