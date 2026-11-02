"""Synthetic corpus scheduling/cache/shard tests; no network or model updates."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow import corpus
from hydra_directflow import cache as cache_module
from hydra_directflow.cache import DerivedCache
from hydra_directflow.local_data import GAUGES, LocalCycle, iso, utc_epoch
from hydra_directflow.reference import digest

H = 3600
CYCLE = "20210823_00Z"


def small_plan():
    units = corpus.partition_split({"name": "synthetic", "start_utc": "2021-08-23T02:00:00Z",
                                    "end_utc": "2021-08-24T02:00:00Z"}, history_hours=24, chunk_hours=2)
    cycles = sorted({cycle for unit in units for chunk in unit["chunks"] for cycle in chunk["cycle_ids"]})
    return dict(plan_sha256="a" * 64, units=units, required_cycle_ids=cycles,
                calendar_unavailable_cycle_ids=[], requested_origins=24, candidate_origins=6, purged_origins=18)


def synthetic_arrays(origins, history_hours, cycles):
    times = np.repeat(np.array(origins, np.int64), 3)
    n = len(times)
    data = dict(origin_time=times, site_id=np.tile(np.arange(3, dtype=np.int64), len(origins)),
                history_time=times[:, None] - np.arange(history_hours + 1, 1, -1) * H,
                target_time=times[:, None] + np.arange(1, 19) * H,
                case_id=np.array([site + ":" + iso(origin) for origin in origins for site in GAUGES]),
                eligibility_reason=np.array(["COMPLETE"] * n))
    for name, shape in (("history", (n, history_hours, 2)), ("weather", (n, 18, 4)), ("state", (n, 2)), ("target", (n, 18))):
        data[name], data[name + "_mask"] = np.ones(shape, np.float64), np.ones(shape, bool)
    selected = [max(cycle for cycle in cycles if corpus._epoch_cycle(cycle) <= origin - 2 * H) for origin in times]
    data["source_cycle"] = np.array(selected)
    data["soil_source_cycle"] = np.array(selected)
    data["weather_source_leads"] = np.array([(data["target_time"][i] - corpus._epoch_cycle(cycle)) // H for i, cycle in enumerate(selected)])
    data["source_metadata_json"] = np.array(["Synthetic large metadata should not be repeated in model shards"])
    return data


class PlanTests(unittest.TestCase):
    def test_exact_candidate_calendar_and_scientific_boundary_purges(self):
        plan = corpus.plan_corpus()
        self.assertEqual((plan["requested_origins"], plan["candidate_origins"], plan["purged_origins"]), (53064, 53028, 36))
        self.assertEqual(len(plan["units"]), 74)  # July 2023 is split into two scientific pieces.
        self.assertEqual(len(plan["required_usgs_months"]), 73)
        self.assertEqual(len(plan["required_cycle_ids"]), 8841)
        self.assertEqual(plan["calendar_unavailable_cycle_ids"], ["20180711_06Z", "20180711_12Z", "20180711_18Z"])
        august = next(unit for unit in plan["units"] if unit["month"] == "2023-08")
        self.assertEqual((august["candidate_origins"], august["purged_origins"]), (744, 0))

    def test_first_last_partial_months_and_validation_history_tail(self):
        plan = corpus.plan_corpus()
        self.assertEqual(plan["units"][0]["chunks"][0]["first_origin_utc"], "2018-07-12T00:00:00Z")
        self.assertEqual(plan["units"][0]["candidate_origins"], 480)
        last = plan["units"][-1]
        self.assertEqual((last["requested_origins"], last["candidate_origins"]), (720, 702))
        self.assertEqual(last["chunks"][-1]["last_origin_utc"], "2024-07-30T05:00:00Z")
        val = next(unit for unit in plan["units"] if unit["split"] == "candidate_validation")
        self.assertEqual(val["chunks"][0]["first_origin_utc"], "2023-07-31T00:00:00Z")
        self.assertEqual(val["candidate_origins"], 24)
        self.assertIn("2023-08", val["required_usgs_months"])

    def test_model_shards_drop_repeated_provenance_but_preserve_all_cases(self):
        unit = small_plan()["units"][0]; chunk = unit["chunks"][0]
        data = synthetic_arrays(corpus._origins(chunk), 24, chunk["cycle_ids"])
        shard = corpus.compact_shard(data, chunk)
        self.assertNotIn("source_metadata_json", shard)
        self.assertTrue(shard["admission_mask"].all())
        self.assertEqual(len(shard["case_id"]), 6)
        np.testing.assert_equal(shard["target"], data["target"])

    def test_preacquisition_calendar_is_not_reported_as_an_upstream_gap(self):
        unit = corpus.partition_split(corpus.DEFAULT_SPLITS[0], history_hours=24, chunk_hours=2)[0]
        chunk = unit["chunks"][0]
        self.assertEqual(chunk["cycle_ids"], [])
        n = 6
        data = synthetic_arrays([corpus.ACQUISITION_START + 2 * H, corpus.ACQUISITION_START + 3 * H], 24,
                                ["20180712_00Z"])
        data["origin_time"] -= 2 * H
        data["source_cycle"] = np.array([""] * n)
        data["soil_source_cycle"] = np.array([""] * n)
        data["eligibility_reason"] = np.array(["NO_COMPLETE_SINGLE_WEATHER_RUN|SOIL_UNAVAILABLE_OR_STALE"] * n)
        data["weather"][:] = np.nan; data["weather_mask"][:] = False
        data["state"][:] = np.nan; data["state_mask"][:] = False
        result = corpus.compact_shard(data, chunk)
        self.assertFalse(result["admission_mask"].any())
        self.assertTrue(all(value.startswith("PRE_ACQUISITION_WARMUP_UNRESOLVED") for value in result["source_calendar_status"]))
        self.assertFalse(any("UPSTREAM_GAP" in value for value in result["source_calendar_status"]))

    def test_arbitrary_snapshot_maps_are_rejected_and_factory_subsets_are_copies(self):
        with self.assertRaisesRegex(ValueError, "factory\|from_file|from_file"):
            corpus.AuthenticatedIndexSnapshot("index", {CYCLE: {}})
        original = {CYCLE: {"cycle": CYCLE, "data_complete": True}}
        with patch.object(corpus, "read_required_index", return_value=original) as reader:
            snapshot = corpus.AuthenticatedIndexSnapshot.from_file("index", [CYCLE])
        reader.assert_called_once()
        selected = snapshot.subset("index", [CYCLE]); selected[CYCLE]["data_complete"] = False
        self.assertTrue(snapshot.subset("index", [CYCLE])[CYCLE]["data_complete"])
        with self.assertRaises(ValueError):
            snapshot.subset("another-index", [CYCLE])


class LazySourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.archive = self.root / "owned.tar"; self.archive.write_bytes(b"synthetic archive")
        self.commit = self.root / "original.json"; self.commit.write_text("{}")
        self.receipt = {"cycle_id": CYCLE, "archive_sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
                        "archive_bytes": self.archive.stat().st_size, "data_complete": False, "record_count": 1}
        cycle = LocalCycle(utc_epoch("2021-08-23T00:00:00Z"), np.array([[.2, .3, .4]]), np.ones((1, 3), bool),
                           ({"group": "soil_moisture", "lead": 0},), 1, {}, {"groups": []})
        self.rows = {CYCLE: {"cycle": CYCLE, "data_complete": False}}
        self.stack = ExitStack()
        self.stack.enter_context(patch("hydra_directflow.cache.implementation_identity", return_value={"fixed": "b" * 64}))
        self.stack.enter_context(patch("hydra_directflow.cache.qualified_cycle_rows", return_value=self.rows))
        self.stack.enter_context(patch("hydra_directflow.cache.load_candidate_weights", return_value=np.eye(3)))
        self.stack.enter_context(patch("hydra_directflow.cache.verified_local_bytes", return_value=b'{"native_partitions":[]}'))
        self.loader = self.stack.enter_context(patch.object(corpus, "load_authenticated_cycle", return_value=(cycle, self.receipt)))
        self.args = dict(archive_specs=[{"cycle_id": CYCLE, "archive_path": "provider-not-entered"}],
                         qualified_index_path="index", weights_path="weights", usgs_paths={},
                         usgs_manifest_path="manifest", cache_directory=self.root / "cache")

    def tearDown(self):
        self.stack.close(); self.temp.cleanup()

    def provider(self, cycle, row):
        return {"status": "READY", "spec": {"cycle_id": cycle, "archive_path": str(self.archive), "commit_path": str(self.commit)},
                "ownership": {"owned_archive": True, "archive_path": str(self.archive), "owner_run_id": "synthetic-test"}}

    def test_provider_only_on_miss_and_finalization_after_verified_cache_commit(self):
        calls = []
        def finalize(receipt, result):
            calls.append(receipt)
            self.assertTrue(receipt["cache_payload_verified"])
            commit = self.root / "cache/entries" / receipt["cache_key"] / "COMMIT.json"
            self.assertEqual(hashlib.sha256(commit.read_bytes()).hexdigest(), receipt["cache_commit_sha256"])
            self.assertTrue(result["ownership"]["owned_archive"])
            self.archive.unlink()  # Only this synthetic callback deletes its owned copy.
        with corpus.SourceBackedSession(**self.args, source_provider=self.provider, on_cache_committed=finalize) as session:
            first = session._cycle(CYCLE)
            self.assertFalse(first[2]["data_complete"])
            self.assertEqual(session.provider_calls, 1)
        self.assertFalse(self.archive.exists()); self.assertTrue(self.commit.exists())
        with corpus.SourceBackedSession(**self.args, source_provider=lambda *args: self.fail("Fetched on hit"),
                                       on_cache_committed=lambda *args: self.fail("Finalized again")) as resumed:
            again = resumed._cycle(CYCLE)
            self.assertEqual(again[3], first[3])
            self.assertEqual(resumed.provider_calls, 0)
        self.assertEqual(len(calls), 1)

    def test_new_cache_parent_links_and_entry_are_durable_before_raw_finalization(self):
        nested = self.root / "nested" / "cache"
        args = {**self.args, "cache_directory": nested}
        synced = []
        actual = cache_module._sync_directory
        def record(path):
            actual(path)
            synced.append(Path(path))
        def finalize(receipt, result):
            self.assertIn(self.root, synced)
            self.assertIn(nested.parent, synced)
            self.assertIn(nested, synced)
            self.assertEqual(synced[-1], nested / "entries")
            self.assertTrue(receipt["cache_payload_verified"])
        with patch.object(cache_module, "_sync_directory", side_effect=record):
            with corpus.SourceBackedSession(**args, source_provider=self.provider, on_cache_committed=finalize) as session:
                session._cycle(CYCLE)
                self.assertEqual(session.finalization_calls, 1)

    def test_transport_failure_is_not_promoted_to_upstream_gap(self):
        for returned in (None, {"status": "UNKNOWN"}, {"status": "UPSTREAM_GAP"}):
            with self.subTest(returned=returned):
                with corpus.SourceBackedSession(**self.args, source_provider=lambda *args: returned) as session:
                    with self.assertRaises(corpus.SourceUnavailable) as caught:
                        session._cycle(CYCLE)
                    self.assertEqual(caught.exception.status, "SOURCE_TRANSPORT_NOT_READY")
        self.assertEqual(list((self.root / "cache/entries").iterdir()), [])
        self.loader.assert_not_called()

    def test_admission_failure_preserves_raw_and_never_calls_finalizer(self):
        self.loader.side_effect = ValueError("Synthetic archive mismatch")
        with corpus.SourceBackedSession(**self.args, source_provider=self.provider,
                                       on_cache_committed=lambda *args: self.fail("Finalized failed source")) as session:
            with self.assertRaises(corpus.SourceUnavailable) as caught:
                session._cycle(CYCLE)
        self.assertEqual(caught.exception.status, "SOURCE_ADMISSION_FAILED")
        self.assertTrue(self.archive.exists())
        self.assertEqual(list((self.root / "cache/entries").iterdir()), [])

    def test_cleanup_failure_keeps_verified_cache_and_is_separate_failure(self):
        def failed(*unused):
            raise OSError("Synthetic cleanup failure")
        with corpus.SourceBackedSession(**self.args, source_provider=self.provider, on_cache_committed=failed) as session:
            with self.assertRaises(corpus.SourceUnavailable) as caught:
                session._cycle(CYCLE)
        self.assertEqual(caught.exception.status, "OWNED_STAGE_FINALIZATION_FAILED_AFTER_CACHE_COMMIT")
        self.assertEqual(len(list((self.root / "cache/entries").iterdir())), 1)
        self.assertTrue(self.archive.exists())

    def test_provider_cannot_mutate_the_authenticated_row_or_misidentify_owned_path(self):
        def mutate(cycle, row):
            row["data_complete"] = True
            return self.provider(cycle, row)
        with corpus.SourceBackedSession(**self.args, source_provider=mutate) as session:
            session._cycle(CYCLE)
        self.assertFalse(self.loader.call_args.args[1]["data_complete"])
        self.assertFalse(self.rows[CYCLE]["data_complete"])


class FakeSession:
    builds = 0
    fail_on_build = None
    def __init__(self, **kwargs):
        self.provider_calls = self.finalization_calls = 0
        self.qualified_cache_reuse = {}
        self.exports = kwargs.get("usgs_development_exports") or {}
        self.specs = kwargs["archive_specs"]
    def __enter__(self): return self
    def __exit__(self, *unused): return False
    def _cycle(self, cycle): return None, None, {}, digest(cycle)
    def _month(self, month): return [], {}, digest(month)
    def build(self, *, origins_utc, history_hours, archive_cycle_ids, **unused):
        FakeSession.builds += 1
        if FakeSession.fail_on_build == FakeSession.builds:
            raise corpus.SourceUnavailable(archive_cycle_ids[0], "SOURCE_TRANSPORT_FAILED")
        origins = [utc_epoch(value) for value in origins_utc]
        arrays = synthetic_arrays(origins, history_hours, archive_cycle_ids)
        receipt = {"case_source_digest": digest(origins), "cache": {"cycle_keys": {cycle: digest(cycle) for cycle in archive_cycle_ids},
                   "month_keys": {"2021-08": digest("2021-08")}}, "stage_seconds": {"total": .01}}
        return arrays, receipt


class CorpusResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.plan = small_plan()
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(corpus, "__file__", str(self.root / "src/hydra_directflow/corpus.py")))
        self.stack.enter_context(patch.object(corpus, "plan_corpus", return_value=self.plan))
        self.index_read = self.stack.enter_context(patch.object(corpus, "read_required_index", return_value={c: {} for c in self.plan["required_cycle_ids"]}))
        self.stack.enter_context(patch.object(corpus, "SourceBackedSession", FakeSession))
        FakeSession.builds, FakeSession.fail_on_build = 0, None
        self.args = dict(qualified_index_path="index", weights_path="weights", usgs_paths={}, usgs_manifest_path="manifest",
                         cache_directory=self.root / ".local/cache", run_directory=self.root / ".local/run", history_hours=24,
                         execution_enabled=True, approved_plan_sha256=self.plan["plan_sha256"])

    def tearDown(self):
        self.stack.close(); self.temp.cleanup()

    def test_default_plan_never_opens_sources_or_provider(self):
        args = {**self.args, "execution_enabled": False, "source_provider": lambda *args: self.fail("Network callback during plan")}
        self.assertEqual(corpus.run_corpus(**args), self.plan)
        self.index_read.assert_not_called()
        self.assertFalse(Path(self.args["run_directory"]).exists())

    def test_stop_and_resume_reuse_immutable_shards_and_authenticate_index_once(self):
        first = corpus.run_corpus(**self.args, stop_after_chunks=1)
        self.assertEqual(first["status"], "STOPPED_AT_CHUNK_BOUNDARY")
        self.assertEqual(first["chunks_built"], 1)
        resumed = corpus.run_corpus(**self.args)
        self.assertEqual(resumed["status"], "COMPLETED_DEVELOPMENT_CORPUS")
        self.assertEqual((resumed["chunks_reused"], resumed["chunks_built"]), (1, 2))
        self.assertEqual(resumed["completed_cases"], 18)
        self.assertEqual(FakeSession.builds, 3)
        self.assertEqual(resumed["sealed_index_authentication_passes"], 1)
        self.assertEqual(self.index_read.call_count, 2)  # Once per invocation, not per monthly session.

    def test_failed_transport_keeps_prior_shards_and_resumes_pending_chunk(self):
        FakeSession.fail_on_build = 2
        first = corpus.run_corpus(**self.args)
        self.assertEqual(first["failure"]["source_status"], "SOURCE_TRANSPORT_FAILED")
        self.assertEqual(first["completed_cases"], 6)
        FakeSession.fail_on_build = None
        resumed = corpus.run_corpus(**self.args)
        self.assertEqual(resumed["status"], "COMPLETED_DEVELOPMENT_CORPUS")
        self.assertEqual(resumed["chunks_reused"], 1)

    def test_shard_tamper_stops_without_rebuilding_or_fetching(self):
        corpus.run_corpus(**self.args, stop_after_chunks=1)
        path = next(Path(self.args["run_directory"]).glob("chunks/*/*/model.npz"))
        path.write_bytes(b"damaged")
        before = FakeSession.builds
        result = corpus.run_corpus(**self.args)
        self.assertEqual(result["status"], "STOPPED_WITH_EXPLICIT_FAILURE")
        self.assertEqual(FakeSession.builds, before)
        self.assertEqual(path.read_bytes(), b"damaged")

    def test_foreign_plan_cannot_overwrite_existing_result(self):
        corpus.run_corpus(**self.args, stop_after_chunks=1)
        path = Path(self.args["run_directory"]) / "RESULT.json"; original = path.read_bytes()
        self.plan["plan_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "another plan"):
            corpus.run_corpus(**{**self.args, "approved_plan_sha256": "b" * 64})
        self.assertEqual(path.read_bytes(), original)

    def test_concurrent_run_cannot_overwrite_active_result(self):
        corpus.run_corpus(**self.args, stop_after_chunks=1)
        root = Path(self.args["run_directory"])
        original = (root / "RESULT.json").read_bytes()
        with (root / ".writer.lock").open("a") as locked:
            corpus.fcntl.flock(locked, corpus.fcntl.LOCK_EX | corpus.fcntl.LOCK_NB)
            try:
                with self.assertRaises(BlockingIOError):
                    corpus.run_corpus(**self.args)
            finally:
                corpus.fcntl.flock(locked, corpus.fcntl.LOCK_UN)
        self.assertEqual((root / "RESULT.json").read_bytes(), original)

    def test_invalid_model_clock_never_publishes_a_shard(self):
        unit, chunk = self.plan["units"][0], self.plan["units"][0]["chunks"][0]
        arrays = synthetic_arrays(corpus._origins(chunk), 24, chunk["cycle_ids"])
        arrays["target_time"][0, -1] += H
        root = self.root / ".local/test-shard"; (root / "chunks").mkdir(parents=True); (root / "staging").mkdir()
        with self.assertRaisesRegex(ValueError, "history/targets"):
            corpus._shard_write(root, "a" * 64, unit, chunk, arrays, {})
        self.assertFalse((root / "chunks" / unit["id"] / "000").exists())


if __name__ == "__main__":
    unittest.main()
