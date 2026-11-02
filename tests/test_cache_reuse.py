"""Exact producer reuse; invented source arrays, no network/model operations."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow.cache import DerivedCache, CacheCorruption, pack_cycle, unpack_cycle
from hydra_directflow.cache_reuse import qualified_basin_reuse, PRODUCER_CODE, CONSUMER_CODE, POLICY
from hydra_directflow.corpus import SourceBackedSession
from hydra_directflow.local_data import LocalCycle, WEIGHTS_SHA256, utc_epoch
from hydra_directflow.multicycle import INDEX_SHA256
from hydra_directflow.reference import digest


def identity(code=CONSUMER_CODE, row=None):
    return dict(cycle_id="20200101_00Z", qualified_index_sha256=INDEX_SHA256,
                sealed_row_sha256=digest(row or {"synthetic": True}), candidate_weights_sha256=WEIGHTS_SHA256,
                implementation_sha256=dict(code), policy=dict(POLICY))


def built(value=.2, receipt_cycle="20200101_00Z"):
    cycle = LocalCycle(utc_epoch("2020-01-01T00:00:00Z"), np.empty((1, 0)), np.empty((1, 0), bool),
                       ({"group": "soil_moisture", "lead_hour": 0},), 1, {}, {})
    return pack_cycle(cycle, (np.full((1, 3), value), np.ones((1, 3), bool), np.ones((1, 3))),
                      {"cycle_id": receipt_cycle, "archive_sha256": "a" * 64, "record_count": 1})


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = DerivedCache(Path(self.temp.name) / "cache")
        self.lock = self.cache.locked(); self.lock.__enter__()
        self.addCleanup(self.lock.__exit__, None, None, None)

    def old(self, **kwargs):
        return self.cache.get_or_build("basin_cycle", identity(PRODUCER_CODE), lambda: built(**kwargs), validator=unpack_cycle)

    def test_original_bytes_key_receipt_and_arrays_remain_unchanged(self):
        arrays, metadata, key, _ = self.old()
        path = self.cache.path / "entries" / key
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
        source = deepcopy(metadata["source_receipt"])
        found, original, result_key, proof = qualified_basin_reuse(self.cache, identity())
        self.assertEqual(key, result_key)
        self.assertEqual(original["source_receipt"], source)
        for name in arrays: np.testing.assert_array_equal(found[name], arrays[name])
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()})
        self.assertEqual(proof["original_commit_sha256"], before["COMMIT.json"])
        self.assertEqual(proof["original_source_receipt_sha256"], digest(source))
        self.assertFalse(proof["raw_source_redecoded"])
        self.assertFalse((self.cache.path / "entries" / self.cache.key("basin_cycle", identity())).exists())

    def test_exact_original_producer_is_preferred_even_if_current_entry_exists(self):
        _, _, key, _ = self.old(value=.2)
        self.cache.get_or_build("basin_cycle", identity(), lambda: built(value=.9), validator=unpack_cycle)
        arrays, _, found, _ = qualified_basin_reuse(self.cache, identity())
        self.assertEqual(found, key)
        self.assertTrue(np.all(arrays["means"] == .2))

    def test_corrupt_chosen_original_fails_instead_of_using_current_entry(self):
        _, _, key, _ = self.old()
        self.cache.get_or_build("basin_cycle", identity(), built, validator=unpack_cycle)
        payload = next((self.cache.path / "entries" / key).glob("*.npz"))
        payload.write_bytes(b"corrupt")
        with self.assertRaises(CacheCorruption): qualified_basin_reuse(self.cache, identity())

    def test_other_consumer_code_cannot_enter_this_compatibility_rule(self):
        self.old()
        for name in CONSUMER_CODE:
            value = identity(); value["implementation_sha256"][name] = "f" * 64
            with patch.object(self.cache, "_load", side_effect=AssertionError("read under unqualified code")):
                self.assertIsNone(qualified_basin_reuse(self.cache, value))

    def test_other_producer_version_is_not_adopted(self):
        other = identity(PRODUCER_CODE); other["implementation_sha256"]["local_data.py"] = "f" * 64
        self.cache.get_or_build("basin_cycle", other, built, validator=unpack_cycle)
        self.assertIsNone(qualified_basin_reuse(self.cache, identity()))

    def test_changed_source_weights_or_policy_cannot_be_reused(self):
        self.old()
        for key in ("qualified_index_sha256", "candidate_weights_sha256", "policy"):
            value = identity(); value[key] = {} if key == "policy" else "f" * 64
            with self.assertRaises(ValueError): qualified_basin_reuse(self.cache, value)
        value = identity(); value["sealed_row_sha256"] = "f" * 64
        self.assertIsNone(qualified_basin_reuse(self.cache, value))

    def test_protected_or_unacquired_reference_is_rejected_before_entry_read(self):
        for cycle in ("20240731_00Z", "20180711_00Z", "20200101_01Z"):
            value = identity(); value["cycle_id"] = cycle
            with self.assertRaises(ValueError): qualified_basin_reuse(self.cache, value)

    def test_absence_creates_no_cache_or_staging_entry(self):
        self.assertIsNone(qualified_basin_reuse(self.cache, identity()))
        self.assertEqual(list((self.cache.path / "entries").iterdir()), [])
        self.assertEqual(list((self.cache.path / "staging").iterdir()), [])

    def test_lock_is_required(self):
        with self.assertRaises(ValueError): qualified_basin_reuse(DerivedCache("unused"), identity())

    def test_selected_directory_link_is_rejected_before_any_payload_read(self):
        _, _, key, _ = self.old()
        original = self.cache.path / "entries" / key
        retained = self.cache.path / "retained-original"
        original.rename(retained); original.symlink_to(retained, target_is_directory=True)
        with self.assertRaises(ValueError): qualified_basin_reuse(self.cache, identity())

    def test_original_receipt_cycle_is_checked(self):
        self.old(receipt_cycle="20200102_00Z")
        with self.assertRaisesRegex(ValueError, "reference/source receipt"):
            qualified_basin_reuse(self.cache, identity())

    def test_proof_is_stable_across_reloads(self):
        self.old()
        a = qualified_basin_reuse(self.cache, identity())[3]
        b = qualified_basin_reuse(self.cache, identity())[3]
        self.assertEqual(a, b)
        self.assertEqual(self.cache.counts["qualified_producer_hits"], 2)

    def test_corpus_reuse_calls_no_provider_or_finalizer_and_preserves_receipt(self):
        arrays, metadata, old_key, _ = self.old()
        session = object.__new__(SourceBackedSession)
        session.cache, session.code, session.policy = self.cache, dict(CONSUMER_CODE), dict(POLICY)
        session.cycles, session.qualified_cache_reuse = {}, {}
        session.index = {"20200101_00Z": {"synthetic": True}}
        session.guard = lambda: None
        session.source_provider = lambda *_: self.fail("Provider entered on qualified reuse")
        session.on_cache_committed = lambda *_: self.fail("Finalizer entered on original cache reuse")
        session.provider_calls = session.finalization_calls = 0
        result = session._cycle("20200101_00Z")
        self.assertEqual(result[2], metadata["source_receipt"])
        self.assertEqual(result[3], old_key)
        self.assertEqual(session.qualified_cache_reuse["20200101_00Z"]["original_cache_key"], old_key)
        self.assertIs(session._cycle("20200101_00Z"), result)
        self.assertEqual((session.provider_calls, session.finalization_calls), (0, 0))


if __name__ == "__main__":
    unittest.main()
