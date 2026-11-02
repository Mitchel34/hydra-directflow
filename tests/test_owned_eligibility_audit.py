"""Synthetic metadata-only tests for the owned-source audit adapter."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_owned_development_eligibility", ROOT / "scripts/audit_owned_development_eligibility.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class OwnedEligibilityTests(unittest.TestCase):
    def fixture(self, directory, *, protected=False):
        root = Path(directory)
        forbidden = root / "forbidden" / "2024-07.parquet"
        forbidden.parent.mkdir()
        forbidden.write_bytes(b"do-not-open-raw-mixed-month")
        source = {"path": str(forbidden), "bytes": forbidden.stat().st_size, "sha256": "a" * 64}
        manifest = root / "source-manifest.json"
        manifest.write_text(json.dumps({"native_partitions": [source]}))
        exported = root / "safe-export"
        exported.mkdir()
        stamp = "2024-07-31T00:00:00Z" if protected else "2024-07-01T00:00:00Z"
        row = {"site": audit.GAUGES[0], "sample_epoch": audit.core.utc_hour(stamp), "sample_time_utc": stamp,
               "is_exact_utc_hour": True, "numeric_valid": True, "source_qualifiers_json": '["A", "R"]', "is_core": True,
               "discharge_cfs": 987654321.}  # Synthetic sentinel, never selected.
        payload = exported / "data.parquet"
        pq.write_table(pa.Table.from_pylist([row]), payload)
        identity = {"payload": {"name": "data.parquet", "bytes": payload.stat().st_size,
                                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}, "exported_rows": 1}
        return forbidden, manifest, exported, identity

    def native_result(self):
        return {site: {} for site in audit.GAUGES}, {"2024-06"}, [
            {"month": "2024-06", "metadata_known": True, "status": "VERIFIED_METADATA_READ"},
            {"month": "2024-07", "metadata_known": False, "status": "NOT_SUPPLIED", "path": "None"}]

    def receipt_fixture(self):
        quotient, remainder = divmod(774216, 11)
        entries = [{"month": month, "sha256": "a" * 64, "bytes": quotient + int(index < remainder)}
                   for index, month in enumerate(audit.OWNED_MONTHS)]
        manifest = {"native_partitions": [{"path": item["month"] + ".parquet", "sha256": item["sha256"], "bytes": item["bytes"]} for item in entries]}
        return {"files": entries}, manifest

    def test_owned_receipt_rejects_duplicate_unexpected_traversal_and_mixed_july(self):
        receipt, manifest = self.receipt_fixture()
        self.assertEqual(len(audit.validate_owned_entries(receipt, manifest)), 11)
        for month in ("../2024-07", "2024-07", "2020-01", receipt["files"][1]["month"]):
            changed = {"files": [dict(item) for item in receipt["files"]]}
            changed["files"][0]["month"] = month
            with self.subTest(month=month), self.assertRaisesRegex(ValueError, "exact unique eleven"):
                audit.validate_owned_entries(changed, manifest)

    def test_all_owned_sizes_and_hashes_are_manifest_bound_before_any_file_read(self):
        receipt, manifest = self.receipt_fixture()
        for change in ({"sha256": "b" * 64}, {"bytes": receipt["files"][0]["bytes"] + 1}, {"bytes": True}):
            changed = {"files": [dict(item) for item in receipt["files"]]}
            changed["files"][0].update(change)
            with patch.object(audit, "verified_local_bytes") as read:
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "per-file identity"):
                    audit.validate_owned_entries(changed, manifest)
                read.assert_not_called()

    def test_partial_metadata_coverage_never_claims_all_months_known(self):
        months = audit.core.months_between(audit.core.DEVELOPMENT_START - 169 * 3600, audit.core.DEVELOPMENT_END)
        inventory = [{"month": month} for month in months]
        complete, text = audit.coverage_statement(set(months[:-1]), inventory)
        self.assertFalse(complete)
        self.assertIn("72 of 73", text)
        self.assertIn("UNKNOWN", text)
        self.assertNotIn("All development", text)
        self.assertTrue(audit.coverage_statement(set(months), inventory)[0])

    def test_raw_july_is_withheld_before_native_loader_and_never_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            forbidden, manifest, exported, identity = self.fixture(directory)
            original_hash = hashlib.sha256(Path(audit.core.__file__).read_bytes()).hexdigest()
            actual_reader = pq.read_table
            with patch.object(audit, "USGS_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()), \
                 patch.object(audit.core, "inventory_usgs", return_value=self.native_result()) as native, \
                 patch.object(audit, "inspect_development_export", return_value=identity), \
                 patch.object(audit, "verified_local_bytes", wraps=audit.verified_local_bytes) as read, \
                 patch("pyarrow.parquet.read_table", wraps=actual_reader) as parquet:
                exact, known, inventory = audit.metadata_with_export(manifest, {"2024-07": str(forbidden)},
                    {"2024-07": {"directory": str(exported), "commit_sha256": "b" * 64}})
            self.assertIsNone(native.call_args.args[1]["2024-07"])
            self.assertNotIn(forbidden, [Path(call.args[0]) for call in read.call_args_list])
            self.assertEqual(parquet.call_args.kwargs["columns"], list(audit.core.METADATA_COLUMNS))
            self.assertNotIn("discharge_cfs", parquet.call_args.kwargs["columns"])
            self.assertEqual(known, {"2024-06", "2024-07"})
            self.assertEqual(inventory[-1]["status"], "VERIFIED_DEVELOPMENT_EXPORT_METADATA_READ")
            self.assertEqual(inventory[-1]["original_source_identity"]["sha256"], "a" * 64)
            self.assertEqual(next(iter(exact[audit.GAUGES[0]].values()))["qualifiers"], ("A", "R"))
            self.assertEqual(hashlib.sha256(Path(audit.core.__file__).read_bytes()).hexdigest(), original_hash)

    def test_export_protected_coordinates_reject_without_any_flow_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest, exported, identity = self.fixture(directory, protected=True)
            actual_reader = pq.read_table
            with patch.object(audit, "USGS_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()), \
                 patch.object(audit.core, "inventory_usgs", return_value=self.native_result()), \
                 patch.object(audit, "inspect_development_export", return_value=identity), \
                 patch("pyarrow.parquet.read_table", wraps=actual_reader) as reader:
                with self.assertRaisesRegex(ValueError, "nondevelopment coordinate"):
                    audit.metadata_with_export(manifest, {}, {"2024-07": {"directory": str(exported), "commit_sha256": "b" * 64}})
            self.assertEqual(reader.call_count, 1)
            self.assertNotIn("discharge_cfs", reader.call_args.kwargs["columns"])

    def test_export_hash_failure_precedes_metadata_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest, exported, identity = self.fixture(directory)
            identity["payload"]["sha256"] = "0" * 64
            with patch.object(audit, "USGS_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()), \
                 patch.object(audit.core, "inventory_usgs", return_value=self.native_result()), \
                 patch.object(audit, "inspect_development_export", return_value=identity), patch("pyarrow.parquet.read_table") as reader:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    audit.metadata_with_export(manifest, {}, {"2024-07": {"directory": str(exported), "commit_sha256": "b" * 64}})
                reader.assert_not_called()

    def test_source_months_outside_explicit_export_scope_reject_before_source_read(self):
        with patch.object(audit.core, "inventory_usgs") as native:
            for value in ({}, {"2024-08": {}}, {"2024-07": {}, "2024-08": {}}):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Exactly one"):
                    audit.metadata_with_export("absent", {}, value)
            native.assert_not_called()

    def test_native_route_cannot_silently_read_july(self):
        exact, known, inventory = self.native_result()
        known.add("2024-07")
        with patch.object(audit.core, "inventory_usgs", return_value=(exact, known, inventory)), \
             patch.object(audit, "verified_local_bytes") as read:
            with self.assertRaisesRegex(ValueError, "withheld"):
                audit.metadata_with_export("absent", {}, {"2024-07": {}})
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
