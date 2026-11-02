"""Synthetic regressions for original zero-record, metadata-only source gaps."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

from hydra_directflow.local_data import GRID_SHA256, read_cycle_tar, sha256, basin_reduce
from hydra_directflow.multicycle import load_authenticated_cycle
from hydra_directflow.cache import pack_cycle, unpack_cycle

CYCLE = "20180712_00Z"
GROUPS = ("temperature", "humidity", "precipitation", "snow_water_equivalent", "soil_moisture")


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


class MetadataOnlyGapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, *, mutate_manifest=None, fields=None, records=None, mutate_availability=None, arrays=None):
        availability = {"cycle": CYCLE, "data_complete": False, "coverage_status": "CONFIRMED_GAPS",
                        "expected_groups": list(GROUPS), "groups": [
                            {"group": group, "status": "SYNTHETIC_CONFIRMED_GAP", "record_count": 0} for group in GROUPS]}
        if mutate_availability:
            mutate_availability(availability)
        members = {"AVAILABILITY.json": encoded(availability), "FIELD_SOURCES.json": encoded({} if fields is None else fields),
                   "RAW_PROVENANCE.json": encoded([] if records is None else records), "POST_FETCH_CACHE_PROVENANCE.json": b"{}"}
        members.update(arrays or {})
        manifest = {"schema": "HRRR-gap-aware-raw-cycle-acquisition/v1", "cycle": CYCLE,
                    "record_count": 0, "groups": [], "expected_groups": list(GROUPS), "data_complete": False,
                    "coverage_status": "CONFIRMED_GAPS", "raw_shape": None, "mask_shape": None,
                    "availability": availability, "assets": {"grid_sha256": GRID_SHA256}}
        if mutate_manifest:
            mutate_manifest(manifest)
        manifest["files"] = {name: {"sha256": sha256(data), "bytes": len(data)} for name, data in members.items()}
        members["CYCLE_MANIFEST.json"] = encoded(manifest)
        path = self.root / "source.tar"
        with tarfile.open(path, "w:") as archive:
            for name, data in members.items():
                entry = tarfile.TarInfo(CYCLE + "/" + name); entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
        raw = path.read_bytes()
        bundle = {"sha256": sha256(raw), "bytes": len(raw), "verified": True}
        commit = {"schema": "hrrr-verified-cloud-cycle/v1", "cycle": CYCLE, "bundle": bundle,
                  "worker_manifest": {"cycle": CYCLE, "bundle_sha256": bundle["sha256"], "bundle_bytes": bundle["bytes"],
                                      "manifest_sha256": sha256(members["CYCLE_MANIFEST.json"]), "record_count": manifest["record_count"]}}
        commit_path = self.root / "commit.json"; commit_path.write_bytes(encoded(commit))
        row = {"cycle": CYCLE, "record_count": manifest["record_count"], "location_kind": "commit",
               "data_complete": manifest["data_complete"], "coverage_status": "CONFIRMED_GAPS", "groups_available": [],
               "groups_missing": {group: "SYNTHETIC_CONFIRMED_GAP" for group in GROUPS}, "original_quality_flags_rewritten": False,
               "source_locations": {"commit": {"sha256": sha256(commit_path.read_bytes()), "bytes": commit_path.stat().st_size}}}
        return path, bundle["sha256"], members, manifest, {"cycle_id": CYCLE, "archive_path": path, "commit_path": commit_path}, row

    def test_exact_metadata_only_source_gap_preserves_original_bytes_and_null_shapes(self):
        path, digest, original, manifest, _, _ = self.fixture()
        members, actual = read_cycle_tar(path, digest, cycle_id=CYCLE, allow_legacy=True)
        self.assertEqual(members, original); self.assertEqual(actual, manifest)
        self.assertNotIn("raw_values.npy", members); self.assertNotIn("raw_missing_bits.npy", members)
        self.assertIsNone(actual["raw_shape"]); self.assertIsNone(actual["mask_shape"])

    def test_empty_inventory_survives_reduction_and_cache_interface_without_records(self):
        _, _, _, _, spec, row = self.fixture()
        cycle, receipt = load_authenticated_cycle(spec, row)
        self.assertEqual(cycle.values.shape, (0, 957)); self.assertEqual(cycle.records, ())
        self.assertFalse(receipt["raw_payload_present"]); self.assertEqual(receipt["numeric_records_decoded"], 0)
        self.assertTrue(receipt["in_memory_empty_inventory_only"]); self.assertFalse(receipt["data_complete"])
        weights = np.zeros((3, 957)); weights[:, 0] = 1
        reduced = basin_reduce(cycle.values, cycle.present, weights)
        self.assertEqual([value.shape for value in reduced], [(0, 3)] * 3)
        arrays, metadata = pack_cycle(cycle, reduced, receipt)
        restored, _ = unpack_cycle(arrays, metadata)
        self.assertEqual(restored.records, ()); self.assertEqual(restored.values.shape, (0, 0))

    def test_partial_raw_pair_is_rejected_even_for_a_gap(self):
        for name in ("raw_values.npy", "raw_missing_bits.npy"):
            path, digest, *_ = self.fixture(arrays={name: b"synthetic-incomplete-pair"})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "present or absent together"):
                read_cycle_tar(path, digest, cycle_id=CYCLE, allow_legacy=True)

    def test_nonzero_complete_or_nonnull_missing_array_shapes_are_rejected(self):
        for changes in ({"record_count": 1}, {"record_count": False}, {"data_complete": True},
                        {"raw_shape": [0, 957]}, {"mask_shape": [0, 120]}, {"groups": ["temperature"]}):
            path, digest, *_ = self.fixture(mutate_manifest=lambda m: m.update(changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "zero-record"):
                read_cycle_tar(path, digest, cycle_id=CYCLE, allow_legacy=True)

    def test_acquired_fields_or_records_cannot_hide_behind_zero_record_summary(self):
        for kwargs in ({"fields": {"temperature": {}}}, {"records": [{"group": "temperature", "lead": 1}]}):
            path, digest, *_ = self.fixture(**kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "acquired records or fields"):
                read_cycle_tar(path, digest, cycle_id=CYCLE, allow_legacy=True)

    def test_missing_expected_groups_available_status_counts_or_leads_rejected(self):
        changes = [lambda a: a["groups"].pop(), lambda a: a["expected_groups"].pop(),
                   lambda a: a["groups"][0].update(status="AVAILABLE_VERIFIED"),
                   lambda a: a["groups"][0].update(record_count=1),
                   lambda a: a["groups"][0].update(record_count=False),
                   lambda a: a["groups"][0].update(leads=[1])]
        for change in changes:
            path, digest, *_ = self.fixture(mutate_availability=change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                read_cycle_tar(path, digest, cycle_id=CYCLE, allow_legacy=True)

    def test_pinned_fixture_reader_remains_strict_by_default(self):
        path, digest, *_ = self.fixture()
        with self.assertRaisesRegex(ValueError, "Required archive members"):
            read_cycle_tar(path, digest, cycle_id=CYCLE)


if __name__ == "__main__":
    unittest.main()
