"""Offline safety and missingness regressions; no private source data required."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

from hydra_directflow.local_data import (
    CYCLE_ID, DEVELOPMENT_QUALIFICATION, FEATURE_GROUPS, LocalCycle,
    basin_reduce, bounded_npy, build_development_smoke, hrrr_provenance_arrays, load_candidate_weights,
    load_usgs_window, read_cycle_tar, sha256, utc_epoch, verified_local_bytes,
)


class LocalReaderSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_tar(self, members):
        p = self.root / "cycle.tar"
        with tarfile.open(p, "w:") as archive:
            for name, data, kind in members:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.size = len(data) if kind == tarfile.REGTYPE else 0
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    info.linkname = "outside"
                archive.addfile(info, io.BytesIO(data) if info.size else None)
        return p, sha256(p.read_bytes())

    def test_hash_and_byte_limits_precede_decoding(self):
        p = self.root / "data"
        p.write_bytes(b"content")
        self.assertEqual(verified_local_bytes(p, sha256(b"content")), b"content")
        for kwargs in ({"max_bytes": 3}, {"expected_bytes": 8}):
            with self.assertRaises(ValueError):
                verified_local_bytes(p, sha256(b"content"), **kwargs)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verified_local_bytes(p, "0" * 64)

    def test_symlink_input_is_rejected(self):
        original = self.root / "original"
        original.write_bytes(b"content")
        link = self.root / "link"
        link.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "regular local file"):
            verified_local_bytes(link, sha256(b"content"))

    def test_archive_traversal_and_unknown_members_are_rejected(self):
        for name in (CYCLE_ID + "/../escape", "/absolute", CYCLE_ID + "/untrusted.pkl"):
            with self.subTest(name=name):
                p, digest = self.make_tar([(name, b"x", tarfile.REGTYPE)])
                with self.assertRaises(ValueError):
                    read_cycle_tar(p, digest)

    def test_archive_symlinks_and_hardlinks_are_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            p, digest = self.make_tar([(CYCLE_ID + "/raw_values.npy", b"", kind)])
            with self.assertRaisesRegex(ValueError, "regular nonsparse"):
                read_cycle_tar(p, digest)

    def test_duplicate_archive_member_is_rejected(self):
        member = (CYCLE_ID + "/CYCLE_MANIFEST.json", b"{}", tarfile.REGTYPE)
        p, digest = self.make_tar([member, member])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_cycle_tar(p, digest)

    def test_npy_object_pickle_is_rejected_before_loading(self):
        raw = io.BytesIO()
        np.save(raw, np.array([{"payload": "not admitted"}], dtype=object))
        with self.assertRaisesRegex(ValueError, "dtype differs"):
            bounded_npy(raw.getvalue(), shape=(1,), dtype="<f8")

    def test_npy_shape_order_and_trailing_bytes_are_rejected(self):
        raw = io.BytesIO()
        np.save(raw, np.zeros((2, 3), dtype="<f8"), allow_pickle=False)
        actual = bounded_npy(raw.getvalue(), shape=(2, 3), dtype="<f8")
        self.assertEqual(actual.shape, (2, 3))
        with self.assertRaises(ValueError):
            bounded_npy(raw.getvalue(), shape=(6,), dtype="<f8")
        with self.assertRaisesRegex(ValueError, "byte count"):
            bounded_npy(raw.getvalue() + b"trailing", shape=(2, 3), dtype="<f8")

    def test_candidate_geometry_requires_explicit_development_label(self):
        with self.assertRaisesRegex(ValueError, "development geometry"):
            load_candidate_weights(self.root / "does-not-exist", qualification="FROZEN_SCIENCE")

    def test_assessment_window_rejected_before_any_path_open(self):
        with self.assertRaisesRegex(ValueError, "limited to August 2021"):
            load_usgs_window(self.root / "no-data", self.root / "no-manifest",
                             start=utc_epoch("2024-08-01T00:00:00Z"),
                             end=utc_epoch("2024-08-02T00:00:00Z"))

    def test_unavailable_or_other_origin_rejected_before_source_access(self):
        paths = dict(archive_path="absent", weights_path="absent", usgs_path="absent", usgs_manifest_path="absent")
        for origin in ("2021-08-23T01:00:00Z", "2021-08-23T08:00:00Z", "2024-08-01T02:00:00Z"):
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "origins 02Z through 07Z"):
                build_development_smoke(**paths, origin_utc=origin)

    def test_missing_weighted_cell_masks_mean_without_renormalization(self):
        values = np.array([[10., 100., np.nan]])
        present = np.array([[True, False, False]])
        weights = np.array([[.25, .75, 0.], [1., 0., 0.]])
        result, mask, coverage = basin_reduce(values, present, weights)
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertFalse(mask[0, 0])
        self.assertEqual(coverage[0, 0], .25)
        self.assertEqual(result[0, 1], 10.)
        self.assertTrue(mask[0, 1])

    def test_complete_basin_mean_uses_original_weights(self):
        values = np.array([[10., 30.]])
        result, mask, coverage = basin_reduce(values, np.ones_like(values, dtype=bool),
                                               np.array([[.25, .75]]))
        self.assertEqual(result.tolist(), [[25.]])
        self.assertTrue(mask.all())
        self.assertEqual(coverage.tolist(), [[1.]])

    def test_hrrr_original_provenance_aligns_by_feature_and_lead_without_development_labels(self):
        records, fields, groups = [], {}, []
        for feature, group in FEATURE_GROUPS:
            field_id = "source-field-" + feature
            fields[field_id] = {
                "availability_status": "ORIGINAL_PUBLICATION_" + feature,
                "historical_available_at_UTC": None,
                "source": {"key": "source-key-" + group, "sha256": "a" * 64, "bytes": 123},
            }
            groups.append({"group": group, "status": "ORIGINAL_GROUP_" + group,
                           "source_key": group, "record_count": 2, "leads": [3, 4]})
            for lead in (3, 4):
                records.append(dict(group=group, lead=lead, record_index=len(records), source_route="original-route",
                                    source_dtype="float16", source_reference_UTC="2021-08-23T00:00:00Z",
                                    source_valid_UTC=f"2021-08-23T{lead:02}:00:00Z",
                                    source_values_sha256_float64_le=sha256(f"{group}-{lead}".encode()),
                                    missing_bits_sha256="b" * 64, missing_cells=lead,
                                    field_source_id=field_id, storage_dtype="original-storage",
                                    units="original-units", APCP_interval_hours=[lead - 1, lead] if group == "precipitation" else None))
        groups.append(dict(group="soil_moisture", status="ORIGINAL_SOIL_GROUP", source_key="soil-group",
                           record_count=1, leads=[0]))
        records.append(dict(group="soil_moisture", lead=0, record_index=8, source_route="native-route",
                            source_dtype="float64", source_reference_UTC="2021-08-23T00:00:00Z",
                            source_valid_UTC="2021-08-23T00:00:00Z", source_values_sha256_float64_le="c" * 64,
                            missing_bits_sha256="d" * 64, missing_cells=0,
                            availability_status="ORIGINAL_SOIL_PUBLICATION", historical_available_at_UTC=None,
                            soil_definition="original definition", native_metadata={"units": "Proportion"},
                            source={"key": "original-native-soil-source", "sha256": "e" * 64, "bytes": 456}))
        cycle = LocalCycle(0, np.empty((0, 0)), np.empty((0, 0), dtype=bool), tuple(records), 0,
                           fields, {"groups": groups})
        arrays = hrrr_provenance_arrays(cycle, (4, 3), site_count=2)
        self.assertEqual(arrays["weather_source_metadata"].shape, (2, 2, 4))
        precipitation = json.loads(arrays["weather_source_metadata"][1, 0, 0])
        self.assertEqual(precipitation["lead"], 4)
        self.assertEqual(precipitation["APCP_interval_hours"], [3, 4])
        self.assertEqual(precipitation["source_values_sha256_float64_le"], sha256(b"precipitation-4"))
        self.assertEqual(precipitation["field_availability"]["availability_status"], "ORIGINAL_PUBLICATION_precip_mm")
        self.assertIsNone(precipitation["field_availability"]["historical_available_at_UTC"])
        self.assertEqual(precipitation["group_availability"]["status"], "ORIGINAL_GROUP_precipitation")
        self.assertNotIn(DEVELOPMENT_QUALIFICATION, arrays["weather_source_metadata"][1, 0, 0])
        self.assertIn(DEVELOPMENT_QUALIFICATION, arrays["weather_development_labels"])
        self.assertEqual(arrays["soil_source_ids"].tolist(), ["original-native-soil-source"] * 2)
        self.assertEqual(json.loads(arrays["soil_source_metadata"][0])["source_identity"]["sha256"], "e" * 64)
        self.assertNotIn("weather_quality_flags", arrays)
        # Metadata remains available through the same no-pickle artifact boundary.
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        with np.load(io.BytesIO(buffer.getvalue()), allow_pickle=False) as restored:
            self.assertEqual(restored["weather_source_metadata"].tolist(), arrays["weather_source_metadata"].tolist())


if __name__ == "__main__":
    unittest.main()
