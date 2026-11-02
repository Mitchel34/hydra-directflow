"""Offline cache crash/reuse/integrity tests; no optimizer or network calls."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow.cache import (CacheCorruption, CachedDevelopmentSession, DerivedCache, _month_bounds, _write_durable,
                                    pack_cycle, pack_month, unpack_cycle, unpack_month)
from hydra_directflow.local_data import LocalCycle
from hydra_directflow.reference import Observation


def payload():
    return {"means": np.array([[1., np.nan, 3.]]), "masks": np.array([[True, False, True]])}, {"source": "fixed"}


def tiny_cycle():
    return LocalCycle(1609459200, np.array([[1., 2., 3.]]), np.ones((1, 3), bool),
                      ({"record_index": 0, "group": "temperature", "lead": 1},), 1, {}, {"groups": []})


class CacheTests(unittest.TestCase):
    def test_committed_entry_reuses_verified_payload_without_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = DerivedCache(directory)
            with cache.locked():
                first = cache.get_or_build("basin_cycle", {"source_sha256": "a" * 64}, payload)
                def forbidden():
                    raise AssertionError("Cache hit re-entered builder")
                second = cache.get_or_build("basin_cycle", {"source_sha256": "a" * 64}, forbidden)
                self.assertEqual(first[2], second[2])
                self.assertEqual((first[3], second[3]), ("built", "hit"))
                np.testing.assert_equal(first[0]["means"], second[0]["means"])
                self.assertEqual(cache.counts["hits"], 1)
                self.assertEqual(cache.counts["misses"], 1)

    def test_partial_build_is_not_committed_and_resume_preserves_completed_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            with DerivedCache(directory).locked() as cache:
                good = cache.get_or_build("basin_cycle", {"cycle": "good"}, payload)
                def interrupted():
                    raise InterruptedError("Simulated process interruption")
                with self.assertRaises(InterruptedError):
                    cache.get_or_build("basin_cycle", {"cycle": "next"}, interrupted)
                self.assertEqual(len(list((Path(directory) / "entries").iterdir())), 1)
                self.assertEqual(len(list((Path(directory) / "staging").iterdir())), 1)
            with DerivedCache(directory).locked() as cache:
                again = cache.get_or_build("basin_cycle", {"cycle": "good"}, lambda: self.fail("Rebuilt complete entry"))
                next_entry = cache.get_or_build("basin_cycle", {"cycle": "next"}, payload)
                self.assertEqual(again[2], good[2])
                self.assertEqual(next_entry[3], "built")

    def test_payload_tamper_fails_closed_then_explicit_repair_quarantines(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = {"source": "identity"}
            with DerivedCache(directory).locked() as cache:
                _, _, key, _ = cache.get_or_build("basin_cycle", identity, payload)
                entry = Path(directory) / "entries" / key
                commit = json.loads((entry / "COMMIT.json").read_text())
                damaged = entry / commit["payloads"]["arrays"]["name"]
                damaged.write_bytes(b"broken")
                with self.assertRaises(CacheCorruption):
                    cache.get_or_build("basin_cycle", identity, lambda: self.fail("Silent repair"))
                self.assertEqual(damaged.read_bytes(), b"broken")
            with DerivedCache(directory, repair_corrupt=True).locked() as cache:
                arrays, _, same_key, status = cache.get_or_build("basin_cycle", identity, payload)
                self.assertEqual(key, same_key)
                self.assertEqual(status, "built")
                self.assertEqual(cache.counts["repairs"], 1)
                quarantine = list((Path(directory) / "quarantine").iterdir())
                self.assertEqual(len(quarantine), 1)
                self.assertEqual((quarantine[0] / damaged.name).read_bytes(), b"broken")
                np.testing.assert_equal(arrays["means"], payload()[0]["means"])

    def test_interruption_after_payload_write_cannot_publish_partial_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            with DerivedCache(directory).locked() as cache:
                def interrupted(path, raw):
                    _write_durable(path, raw)
                    if Path(path).suffix == ".npz":
                        raise InterruptedError("Interrupted after durable array write")
                with patch("hydra_directflow.cache._write_durable", side_effect=interrupted):
                    with self.assertRaises(InterruptedError):
                        cache.get_or_build("basin_cycle", {}, payload)
                self.assertEqual(list((Path(directory) / "entries").iterdir()), [])
                stages = list((Path(directory) / "staging").iterdir())
                self.assertEqual(len(stages), 1)
                self.assertEqual(len(list(stages[0].glob("*.npz"))), 1)
                _, _, _, status = cache.get_or_build("basin_cycle", {}, payload)
                self.assertEqual(status, "built")
                self.assertEqual(len(list((Path(directory) / "entries").iterdir())), 1)

    def test_different_source_code_or_policy_identity_cannot_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            with DerivedCache(directory).locked() as cache:
                keys = [cache.get_or_build("basin_cycle", {"source": source, "code": code, "policy": policy}, payload)[2]
                        for source, code, policy in (("a", "x", "strict"), ("b", "x", "strict"), ("a", "y", "strict"), ("a", "x", "other"))]
                self.assertEqual(len(set(keys)), 4)
                self.assertEqual(cache.counts["misses"], 4)

    def test_lock_blocks_another_process(self):
        with tempfile.TemporaryDirectory() as directory:
            with DerivedCache(directory).locked():
                script = "import fcntl,sys; f=open(sys.argv[1],'a+');\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(0)\nsys.exit(7)"
                result = subprocess.run([sys.executable, "-c", script, str(Path(directory) / ".writer.lock")], timeout=5, capture_output=True)
                self.assertEqual(result.returncode, 0)

    def test_requires_lock_and_refuses_noncache_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = DerivedCache(directory)
            with self.assertRaisesRegex(ValueError, "lock"):
                cache.get_or_build("basin_cycle", {}, payload)
            (Path(directory) / "unrelated.txt").write_text("Preserve me")
            with self.assertRaisesRegex(ValueError, "unrelated"):
                with cache.locked():
                    pass
            self.assertEqual((Path(directory) / "unrelated.txt").read_text(), "Preserve me")

    def test_symlink_payload_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with DerivedCache(directory).locked() as cache:
                _, _, key, _ = cache.get_or_build("basin_cycle", {}, payload)
                entry = Path(directory) / "entries" / key
                commit = entry / "COMMIT.json"
                moved = Path(directory) / "moved.json"
                commit.rename(moved)
                commit.symlink_to(moved)
                with self.assertRaises(CacheCorruption):
                    cache.get_or_build("basin_cycle", {}, payload)

    def test_invalid_builder_never_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            with DerivedCache(directory).locked() as cache:
                with self.assertRaises(ValueError):
                    cache.get_or_build("basin_cycle", {}, lambda: ({"bad": np.array([object()])}, {}))
                self.assertEqual(list((Path(directory) / "entries").iterdir()), [])

    def test_semantic_validator_applies_on_disk_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            with DerivedCache(directory).locked() as cache:
                cache.get_or_build("basin_cycle", {}, payload)
                def reject(arrays, metadata):
                    raise ValueError("Invalid semantics")
                with self.assertRaises(CacheCorruption):
                    cache.get_or_build("basin_cycle", {}, payload, validator=reject)

    def test_month_store_preserves_missing_invalid_and_quality_separately(self):
        start, _ = _month_bounds("2024-02")
        rows = [Observation("03161000", start, None, 3., ("A",), "wire-a"),
                Observation("03161000", start + 3600, None, None, ("P", "e"), "wire-a"),
                Observation("03164000", start, None, 4., ("A", "R"), "wire-b")]
        arrays, metadata = pack_month("2024-02", rows, {"sha256": "a" * 64})
        self.assertEqual(arrays["values"].shape, (3, 696))
        self.assertTrue(arrays["row_present"][0, 1])
        self.assertFalse(arrays["numeric_valid"][0, 1])
        self.assertFalse(arrays["row_present"][0, 2])
        self.assertEqual(unpack_month(arrays, metadata), rows)
        self.assertEqual(len(metadata["sources"]), 2)

    def test_month_store_rejects_duplicate_or_protected_data(self):
        start, _ = _month_bounds("2024-02")
        row = Observation("03161000", start, None, 3., ("A",), "wire-a")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pack_month("2024-02", [row, row], {})
        with self.assertRaisesRegex(ValueError, "protected"):
            _month_bounds("2024-07")

    def test_month_missing_mask_cannot_gain_value_or_source(self):
        arrays, metadata = pack_month("2024-02", [], {})
        arrays["values"][0, 0] = 1.
        with self.assertRaisesRegex(ValueError, "masks"):
            unpack_month(arrays, metadata)
        arrays["values"][0, 0] = np.nan
        arrays["source_index"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "dictionary index"):
            unpack_month(arrays, metadata)

    def test_basin_store_drops_raw_grid_and_preserves_missing_coverage(self):
        means = np.array([[1., np.nan, 3.]])
        mask = np.array([[True, False, True]])
        coverage = np.array([[1., .8, 1.]])
        arrays, metadata = pack_cycle(tiny_cycle(), (means, mask, coverage), {"record_count": 1})
        cycle, reduced = unpack_cycle(arrays, metadata)
        self.assertEqual(cycle.values.shape, (1, 0))
        self.assertEqual(cycle.records, tiny_cycle().records)
        np.testing.assert_equal(reduced[0], means)
        np.testing.assert_equal(reduced[1], mask)
        np.testing.assert_equal(reduced[2], coverage)
        arrays["means"][0, 1] = 0.
        with self.assertRaisesRegex(ValueError, "masks"):
            unpack_cycle(arrays, metadata)

    def test_authentic_zero_record_gap_survives_store_and_reuse(self):
        gap = LocalCycle(1609459200, np.empty((0, 957)), np.empty((0, 957), bool), (), 7, {},
                         {"groups": [], "original_gap_reason": "SOURCE_OBJECTS_NOT_AVAILABLE"})
        reduced = (np.empty((0, 3)), np.empty((0, 3), bool), np.empty((0, 3)))
        receipt = {"record_count": 0, "data_complete": False, "groups_available": []}
        with tempfile.TemporaryDirectory() as directory:
            with DerivedCache(directory).locked() as cache:
                arrays, metadata, _, _ = cache.get_or_build("basin_cycle", {"gap": "sealed"},
                    lambda: pack_cycle(gap, reduced, receipt), validator=unpack_cycle)
                restored, means = unpack_cycle(arrays, metadata)
                self.assertEqual(restored.records, ())
                self.assertEqual(restored.availability, gap.availability)
                self.assertEqual(means[0].shape, (0, 3))
                self.assertEqual(metadata["source_receipt"], receipt)
                _, _, _, status = cache.get_or_build("basin_cycle", {"gap": "sealed"},
                    lambda: self.fail("Gap cache was rebuilt"), validator=unpack_cycle)
                self.assertEqual(status, "hit")

    def test_session_authenticates_once_and_reuses_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"native_partitions": [{"path": "2021-01.parquet", "sha256": "a" * 64, "bytes": 100}]}
            specs = [{"cycle_id": "20210101_00Z", "archive_path": "unused", "commit_path": "unused"}]
            with patch("hydra_directflow.cache.implementation_identity", return_value={"code": "fixed"}), \
                 patch("hydra_directflow.cache.qualified_cycle_rows", return_value={"20210101_00Z": {"source": "fixed"}}) as index, \
                 patch("hydra_directflow.cache.load_candidate_weights", return_value=np.eye(3)), \
                 patch("hydra_directflow.cache.verified_local_bytes", return_value=json.dumps(manifest).encode()), \
                 patch("hydra_directflow.cache.load_authenticated_cycle", return_value=(tiny_cycle(), {"record_count": 1})) as loader:
                with CachedDevelopmentSession(archive_specs=specs, qualified_index_path="unused", weights_path="unused", usgs_paths={},
                                              usgs_manifest_path="unused", cache_directory=directory) as session:
                    first, second = session._cycle("20210101_00Z"), session._cycle("20210101_00Z")
                    self.assertIs(first, second)
                    self.assertEqual(session.cache.counts["memory_hits"], 1)
                self.assertEqual(index.call_count, 1)
                self.assertEqual(loader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
