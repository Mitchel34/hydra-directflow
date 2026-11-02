"""Synthetic archive regressions for retained forecast-root absence evidence."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from hydra_directflow.local_data import read_cycle_tar, sha256

CYCLE = "20230817_06Z"
DAY = "20230817"


def encode(value):
    return json.dumps(value, sort_keys=True).encode()


class CensusArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "cycle.tar"

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, *, evidence=None, bad_proof=False, complete=False, link_name=None):
        evidence = ({f"census_evidence/{DAY}.json": b'{"synthetic":true}',
                     f"census_evidence/{DAY}.xml": b"<SyntheticEmptyListing/>"}
                    if evidence is None else evidence)
        proof = {"receipt_sha256": sha256(evidence.get(f"census_evidence/{DAY}.json", b"")),
                 "raw_sha256": sha256(evidence.get(f"census_evidence/{DAY}.xml", b""))}
        if bad_proof:
            proof["raw_sha256"] = "0" * 64
        availability = {"cycle": CYCLE, "data_complete": False, "groups": [
            {"group": "temperature", "status": "MISSING_FORECAST_ROOT_CENSUS", "proof": proof}]}
        members = {"AVAILABILITY.json": encode(availability), "FIELD_SOURCES.json": b"{}",
                   "RAW_PROVENANCE.json": b"[]", "POST_FETCH_CACHE_PROVENANCE.json": b"{}",
                   "raw_values.npy": b"synthetic", "raw_missing_bits.npy": b"synthetic", **evidence}
        manifest = {"cycle": CYCLE, "schema": "HRRR-gap-aware-raw-cycle-acquisition/v1",
                    "data_complete": complete, "availability": availability,
                    "files": {name: {"bytes": len(data), "sha256": sha256(data)} for name, data in members.items()}}
        members["CYCLE_MANIFEST.json"] = encode(manifest)
        with tarfile.open(self.path, "w:") as archive:
            directory = tarfile.TarInfo(CYCLE + "/census_evidence")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            for name, data in members.items():
                item = tarfile.TarInfo(CYCLE + "/" + name)
                if name == link_name:
                    item.type, item.linkname = tarfile.SYMTYPE, "outside"
                    archive.addfile(item)
                else:
                    item.size = len(data)
                    archive.addfile(item, io.BytesIO(data))
        return sha256(self.path.read_bytes()), members, manifest

    def test_exact_original_census_pair_is_preserved_and_authenticated(self):
        digest, original, manifest = self.fixture()
        members, actual_manifest = read_cycle_tar(self.path, digest, cycle_id=CYCLE)
        self.assertEqual(members, original)
        self.assertEqual(actual_manifest, manifest)
        self.assertFalse(actual_manifest["data_complete"])
        self.assertEqual(json.loads(members["AVAILABILITY.json"])["groups"][0]["status"],
                         "MISSING_FORECAST_ROOT_CENSUS")

    def test_different_day_extension_or_nested_member_is_rejected(self):
        for name in ("census_evidence/20230818.json", f"census_evidence/{DAY}.bin",
                     f"census_evidence/nested/{DAY}.xml"):
            with self.subTest(name=name):
                digest, _, _ = self.fixture(evidence={name: b"synthetic"})
                with self.assertRaisesRegex(ValueError, "outside whitelist"):
                    read_cycle_tar(self.path, digest, cycle_id=CYCLE)

    def test_directory_only_or_partial_pair_is_rejected(self):
        for evidence in ({}, {f"census_evidence/{DAY}.json": b"{}"}):
            with self.subTest(evidence=evidence):
                digest, _, _ = self.fixture(evidence=evidence)
                with self.assertRaisesRegex(ValueError, "exact current-day JSON/XML pair"):
                    read_cycle_tar(self.path, digest, cycle_id=CYCLE)

    def test_matching_archive_hashes_do_not_bypass_original_proof_hash(self):
        digest, _, _ = self.fixture(bad_proof=True)
        with self.assertRaisesRegex(ValueError, "source-gap proof"):
            read_cycle_tar(self.path, digest, cycle_id=CYCLE)

    def test_evidence_cannot_relabel_a_complete_manifest_as_a_source_gap(self):
        digest, _, _ = self.fixture(complete=True)
        with self.assertRaisesRegex(ValueError, "original source-gap manifest"):
            read_cycle_tar(self.path, digest, cycle_id=CYCLE)

    def test_whitelisted_census_path_cannot_be_a_symlink(self):
        digest, _, _ = self.fixture(link_name=f"census_evidence/{DAY}.xml")
        with self.assertRaisesRegex(ValueError, "regular nonsparse"):
            read_cycle_tar(self.path, digest, cycle_id=CYCLE)


if __name__ == "__main__":
    unittest.main()
