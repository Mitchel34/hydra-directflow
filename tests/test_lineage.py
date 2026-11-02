"""Synthetic signed-byte fixtures exercise old index summaries and exact binding."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from hydra_directflow.lineage import (authenticate_bundle_identity, availability_view,
    normalize_manifest_summary, read_authenticated_bundle)
from hydra_directflow.local_data import read_cycle_tar, sha256

CYCLE = "20210823_01Z"
GROUPS = ("temperature", "humidity", "precipitation", "snow_water_equivalent")


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, *, legacy_payload=False, omitted=(), missing_commit_bytes=False, bundle_kind=False):
        records = [dict(record_index=index, group=group, lead=lead)
                   for index, (group, lead) in enumerate((g, h) for g in GROUPS for h in range(1, 19))]
        availability = {"schema": "hrrr-group-availability/v1", "cycle": CYCLE, "data_complete": True,
                        "coverage_status": "ALL_REQUESTED_GROUPS_AVAILABLE", "expected_groups": list(GROUPS),
                        "groups": [{"group": g, "status": "AVAILABLE_VERIFIED", "record_count": 18,
                                    "leads": list(range(1, 19))} for g in GROUPS]}
        members = {"FIELD_SOURCES.json": b"{}", "RAW_PROVENANCE.json": encode(records),
                   "raw_values.npy": b"synthetic-not-numerically-admitted", "raw_missing_bits.npy": b"synthetic"}
        manifest = {"schema": "HRRR-raw-cycle-acquisition/v1" if legacy_payload else "HRRR-gap-aware-raw-cycle-acquisition/v1",
                    "cycle": CYCLE, "record_count": 72, "groups": list(GROUPS), "raw_shape": [72, 957], "mask_shape": [72, 120]}
        if not legacy_payload:
            members.update({"AVAILABILITY.json": encode(availability), "POST_FETCH_CACHE_PROVENANCE.json": b"{}"})
            manifest.update(availability=availability, data_complete=True, coverage_status="ALL_REQUESTED_GROUPS_AVAILABLE")
        manifest["files"] = {name: {"bytes": len(data), "sha256": sha256(data)} for name, data in members.items()}
        members["CYCLE_MANIFEST.json"] = encode(manifest)
        archive = self.root / ("legacy.tar" if legacy_payload else "modern.tar")
        with tarfile.open(archive, "w:") as stream:
            for name, content in members.items():
                info = tarfile.TarInfo(CYCLE + "/" + name)
                info.size = len(content)
                stream.addfile(info, io.BytesIO(content))
        raw = archive.read_bytes()
        bundle = {"sha256": sha256(raw), "bytes": len(raw), "verified": True}
        worker = {"cycle": CYCLE, "bundle_sha256": bundle["sha256"], "bundle_bytes": bundle["bytes"],
                  "manifest_sha256": sha256(members["CYCLE_MANIFEST.json"]), "record_count": 72}
        commit = {"schema": "hrrr-verified-cloud-cycle/v1", "cycle": CYCLE, "bundle": bundle, "worker_manifest": worker}
        commit_path = self.root / ("legacy.json" if legacy_payload else "modern.json")
        commit_path.write_bytes(encode(commit))
        identity = {"sha256": sha256(commit_path.read_bytes()), "bytes": commit_path.stat().st_size}
        if missing_commit_bytes:
            identity.pop("bytes")
        row = {"cycle": CYCLE, "data_complete": True, "original_quality_flags_rewritten": False,
               "location_kind": "bundle" if bundle_kind else "commit", "record_count": 72,
               "coverage_status": "ALL_REQUESTED_GROUPS_AVAILABLE", "groups_available": list(GROUPS),
               "groups_missing": {}, "source_locations": {"commit": identity, "bundle": bundle}}
        for key in omitted:
            row.pop(key)
        spec = {"cycle_id": CYCLE, "archive_path": archive, "commit_path": commit_path}
        if bundle_kind:
            spec.pop("commit_path")
        return spec, row, members, manifest, commit

    def test_missing_summary_keys_derive_exact_same_values_without_mutating_index(self):
        spec, row, *_ = self.fixture()
        _, _, modern = read_authenticated_bundle(spec, row)
        old = deepcopy(row)
        for key in ("record_count", "groups_available", "coverage_status"):
            old.pop(key)
        before = deepcopy(old)
        _, _, legacy = read_authenticated_bundle(spec, old)
        for key in ("record_count", "groups_available", "coverage_status", "groups_missing", "data_complete"):
            self.assertEqual(modern["normalized_summary"][key], legacy["normalized_summary"][key])
        self.assertEqual(legacy["normalized_summary"]["derived_summary_keys"],
                         ["coverage_status", "groups_available", "record_count"])
        self.assertEqual(old, before)
        self.assertFalse(legacy["source_summaries_rewritten"])

    def test_missing_commit_size_retains_full_sha256_verification(self):
        spec, row, *_ = self.fixture(missing_commit_bytes=True)
        _, _, receipt = read_authenticated_bundle(spec, row)
        self.assertEqual(receipt["commit"]["byte_count_source"], "AUTHENTICATED_LOCAL_BYTES")
        spec["commit_path"].write_bytes(spec["commit_path"].read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            read_authenticated_bundle(spec, row)

    def test_bundle_kind_uses_exact_direct_hash_without_opening_missing_commit(self):
        spec, row, *_ = self.fixture(bundle_kind=True)
        with patch("hydra_directflow.lineage.verified_local_bytes") as commit_reader:
            _, _, receipt = read_authenticated_bundle(spec, row)
            commit_reader.assert_not_called()
        self.assertIsNone(receipt["commit"])
        self.assertEqual(receipt["binding_route"], "SEALED_INDEX_BUNDLE")

    def test_bundle_kind_supplied_commit_is_checked_and_not_silently_ignored(self):
        spec, row, *_ = self.fixture(bundle_kind=True)
        bad = self.root / "bad.json"
        bad.write_bytes(b"{}")
        spec["commit_path"] = bad
        with self.assertRaises(ValueError):
            read_authenticated_bundle(spec, row)

    def test_conflicting_index_and_commit_bundle_is_rejected(self):
        spec, row, *_ = self.fixture()
        row["source_locations"]["bundle"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bundle identities conflict"):
            read_authenticated_bundle(spec, row)

    def test_supplied_summary_disagreements_are_not_filled_over(self):
        spec, row, *_ = self.fixture()
        for key, bad in (("record_count", 71), ("groups_available", list(GROUPS[:-1])),
                         ("coverage_status", "UNSUPPORTED"), ("groups_missing", {"temperature": "MISSING"})):
            changed = deepcopy(row)
            changed[key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "conflicts"):
                read_authenticated_bundle(spec, changed)

    def test_completeness_disagreement_is_rejected(self):
        spec, row, *_ = self.fixture()
        row["data_complete"] = False
        with self.assertRaisesRegex(ValueError, "completeness conflict"):
            read_authenticated_bundle(spec, row)

    def test_legacy_payload_retains_original_members_and_labels_derived_view(self):
        spec, row, original_members, manifest, _ = self.fixture(legacy_payload=True, omitted=("groups_available",))
        members, returned_manifest, receipt = read_authenticated_bundle(spec, row)
        self.assertEqual(members, original_members)
        self.assertEqual(returned_manifest, manifest)
        self.assertNotIn("AVAILABILITY.json", members)
        self.assertEqual(receipt["availability_metadata_origin"], "ADAPTER_DERIVED_FROM_ORIGINAL_MANIFEST_AND_RECORDS")
        view = availability_view(row, manifest, members)
        self.assertTrue(all(g["status"] == "ADAPTER_DERIVED_COMPLETE_GROUP" for g in view["groups"]))
        self.assertTrue(all(g["metadata_origin"].startswith("ADAPTER_DERIVED") for g in view["groups"]))

    def test_legacy_archive_is_not_accepted_by_default_pinned_reader(self):
        spec, row, *_ = self.fixture(legacy_payload=True)
        with self.assertRaisesRegex(ValueError, "Required archive members"):
            read_cycle_tar(spec["archive_path"], row["source_locations"]["bundle"]["sha256"], cycle_id=CYCLE)

    def test_legacy_view_cannot_invent_missing_or_duplicate_leads(self):
        _, row, members, manifest, _ = self.fixture(legacy_payload=True)
        records = json.loads(members["RAW_PROVENANCE.json"])
        for changed in (records[:-1], [records[0], *records[:-1]]):
            altered = dict(members, **{"RAW_PROVENANCE.json": encode(changed)})
            with self.assertRaises(ValueError):
                availability_view(row, manifest, altered)

    def test_legacy_gap_is_not_promoted_to_complete(self):
        _, row, members, manifest, _ = self.fixture(legacy_payload=True)
        row["data_complete"] = False
        with self.assertRaisesRegex(ValueError, "sealed complete"):
            availability_view(row, manifest, members)

    def test_commit_kind_never_falls_back_to_bundle_when_commit_path_missing(self):
        spec, row, *_ = self.fixture()
        spec.pop("commit_path")
        with self.assertRaisesRegex(ValueError, "original local commit"):
            read_authenticated_bundle(spec, row)

    def test_archive_hash_failure_precedes_manifest_interpretation(self):
        spec, row, *_ = self.fixture(bundle_kind=True)
        raw = bytearray(spec["archive_path"].read_bytes())
        raw[-1] ^= 1
        spec["archive_path"].write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            read_authenticated_bundle(spec, row)

    def test_protected_cycle_rejected_before_any_source_open(self):
        with patch("hydra_directflow.lineage.verified_local_bytes") as reader:
            with self.assertRaisesRegex(ValueError, "Protected"):
                authenticate_bundle_identity({"cycle_id": "20240801_00Z", "commit_path": "absent"}, {})
            reader.assert_not_called()

    def test_missing_hash_and_oversized_or_boolean_bytes_are_rejected(self):
        spec, row, *_ = self.fixture()
        for key, value in (("sha256", ""), ("bytes", True), ("bytes", 16385)):
            changed = deepcopy(row)
            changed["source_locations"]["commit"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                read_authenticated_bundle(spec, changed)

    def test_worker_manifest_hash_conflict_is_rejected(self):
        spec, row, _, _, commit = self.fixture()
        commit["worker_manifest"]["manifest_sha256"] = "0" * 64
        raw = encode(commit)
        spec["commit_path"].write_bytes(raw)
        row["source_locations"]["commit"] = {"sha256": sha256(raw), "bytes": len(raw)}
        with self.assertRaisesRegex(ValueError, "Commit manifest binding"):
            read_authenticated_bundle(spec, row)


if __name__ == "__main__":
    unittest.main()
