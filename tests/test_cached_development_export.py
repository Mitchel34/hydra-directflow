"""Synthetic boundary-cache tests; no real July 2024 values are read."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from hydra_directflow.cache import CachedDevelopmentSession, _month_bounds, pack_month, unpack_month
from hydra_directflow.local_data import utc_epoch
from hydra_directflow.reference import ASSESSMENT_START, Observation
from hydra_directflow.monthly import month_calendar


class CachedExportTests(unittest.TestCase):
    def test_only_exact_july_cutoff_is_supported(self):
        with self.assertRaises(ValueError):
            _month_bounds("2024-07")
        start, end = _month_bounds("2024-07", development_cutoff=ASSESSMENT_START)
        self.assertEqual((end - start) // 3600 + 1, 720)
        self.assertEqual(end, ASSESSMENT_START - 3600)
        with self.assertRaises(ValueError):
            _month_bounds("2024-08", development_cutoff=ASSESSMENT_START)

    def test_boundary_month_roundtrip_never_materializes_protected_hour(self):
        cutoff = ASSESSMENT_START
        receipt = {"development_export": {"cutoff_exclusive_epoch": cutoff}}
        row = Observation("03161000", cutoff - 3600, None, 7., ("A", "R"), "original")
        arrays, metadata = pack_month("2024-07", [row], receipt, development_cutoff=cutoff)
        self.assertEqual(arrays["values"].shape, (3, 720))
        self.assertTrue(np.all(arrays["times"] < cutoff))
        self.assertEqual(unpack_month(arrays, metadata), [row])
        metadata["source_receipt"] = {}
        with self.assertRaisesRegex(ValueError, "commitment"):
            unpack_month(arrays, metadata)

    def test_export_commitment_changes_cache_key_and_does_not_open_raw_july(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            keys = []
            for fingerprint in ("a" * 64, "b" * 64):
                session = CachedDevelopmentSession(archive_specs=[{"cycle_id": "20240601_00Z"}], qualified_index_path="unused", weights_path="unused",
                    usgs_paths={}, usgs_manifest_path="unused", cache_directory=Path(folder) / "cache",
                    usgs_development_exports={"2024-07": {"directory": "synthetic-export", "commit_sha256": fingerprint}})
                session.partitions = {"2024-07": {"sha256": "c" * 64, "bytes": 1}}
                session.code, session.policy = {"synthetic": "fixed"}, {"synthetic": "fixed"}
                identity = {"commit_sha256": fingerprint, "cutoff_exclusive_epoch": ASSESSMENT_START}
                receipt = {"month": "2024-07", "development_export": identity}
                def loader(raw_paths, manifest, windows, *, development_exports):
                    self.assertEqual(raw_paths, {})
                    self.assertEqual(windows[-1][-1], ASSESSMENT_START - 3600)
                    paths.append(development_exports)
                    return [], [receipt]
                with session.cache.locked(), patch("hydra_directflow.cache.inspect_development_export", return_value=identity), \
                     patch("hydra_directflow.cache.load_usgs_windows", side_effect=loader):
                    _, _, key = session._month("2024-07")
                    keys.append(key)
            self.assertNotEqual(*keys)
            self.assertEqual(len(paths), 2)

    def test_june_month_plan_needs_explicit_export_and_july_targets_stay_blocked(self):
        with self.assertRaises(ValueError):
            month_calendar("2024-06")
        plan = month_calendar("2024-06", development_export_months=("2024-07",))
        self.assertEqual(plan["origin_count"], 720)
        with self.assertRaises(ValueError):
            month_calendar("2024-07", development_export_months=("2024-07",))


if __name__ == "__main__":
    unittest.main()
