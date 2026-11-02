"""Boundary/export tests use generated synthetic Parquet, never real observations."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from hydra_directflow import development_export as export
from hydra_directflow import multicycle
from hydra_directflow.contract import CORE_WEATHER_FEATURES
from hydra_directflow.lineage import _cycle_guard
from hydra_directflow.local_data import CFS_TO_CMS, iso, utc_epoch
from hydra_directflow.reference import AvailabilityPolicy, WeatherRun, latest_weather_run

SECRET_VALUE = 987654321.125  # Synthetic sentinel; never an observed protected value.


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def row(stamp, value, *, site="03161000", core=True, flags=("A",)):
    instant = utc_epoch(stamp)
    return {"site": site, "sample_epoch": instant, "sample_time_utc": stamp, "source_units": "ft3/s",
            "discharge_cfs": value, "source_qualifiers_json": json.dumps(flags), "numeric_valid": value is not None,
            "is_exact_utc_hour": instant % 3600 == 0, "is_core": core, "source_wire_sha256": "a" * 64,
            "historical_available_at_utc": None}


class BoundaryRules(unittest.TestCase):
    def test_preboundary_initializations_are_not_rejected_for_unused_future_forecast_leads(self):
        for cycle in ("20240729_00Z", "20240730_00Z", "20240730_23Z"):
            with self.subTest(cycle=cycle):
                self.assertLess(multicycle.cycle_reference(cycle), export.CUTOFF)
                _cycle_guard(cycle)
        for cycle in ("20240731_00Z", "20240801_00Z"):
            for check in (multicycle.cycle_reference, _cycle_guard):
                with self.subTest(cycle=cycle), self.assertRaisesRegex(ValueError, "Protected"):
                    check(cycle)

    def test_last_development_origin_passes_but_boundary_target_and_assessment_origins_fail(self):
        specs = [{"cycle_id": "20240730_00Z"}]
        _, origins = multicycle.validate_request(specs, ["2024-07-30T05:00:00Z"], 168)
        self.assertEqual(origins[0] + 18 * 3600, export.CUTOFF - 3600)
        for origin in ("2024-07-30T06:00:00Z", "2024-07-31T00:00:00Z"):
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "boundary-crossing"):
                multicycle.validate_request(specs, [origin], 168)

    def test_issued_forecast_can_cross_boundary_but_selected_targets_and_availability_cannot(self):
        reference, origin = utc_epoch("2024-07-30T00:00:00Z"), utc_epoch("2024-07-30T05:00:00Z")
        valid = tuple(reference + h * 3600 for h in range(1, 37))
        support = tuple((name, (True,) * 36) for name in CORE_WEATHER_FEATURES)
        earlier = WeatherRun("20240730_00Z", reference, None, valid, support)
        future = WeatherRun("20240730_06Z", reference + 6 * 3600, None,
                            tuple(t + 6 * 3600 for t in valid), support)
        selected = latest_weather_run([earlier, future], cutoff=origin, origin_anchor=origin,
            policy=AvailabilityPolicy("test-lag", "retrospective_simulated", 7200),
            horizons_hours=tuple(range(1, 19)), allow_older_extended_cycles=True)
        self.assertEqual(selected.run_id, earlier.run_id)
        self.assertGreater(selected.valid_times[-1], export.CUTOFF)
        self.assertLess(origin + 18 * 3600, export.CUTOFF)

    def test_raw_mixed_month_stays_rejected_before_source_access(self):
        with patch("hydra_directflow.multicycle.verified_local_bytes") as reader:
            with self.assertRaisesRegex(ValueError, "verified development export"):
                multicycle.load_usgs_windows({"2024-07": "absent"}, "absent",
                    [(utc_epoch("2024-07-30T00:00:00Z"), utc_epoch("2024-07-30T23:00:00Z"))])
            reader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Protected USGS"):
            multicycle._months_for_windows([(export.CUTOFF - 3600, export.CUTOFF)], development_export_months=("2024-07",))


class AdministrativeExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "synthetic-2024-07.parquet"
        self.output = self.root / ".local" / "administrative-export"
        self.rows = [row("2024-06-30T23:45:00Z", 1., core=False),
                     row("2024-07-01T00:00:00Z", 20., site="03164000"),
                     row("2024-07-30T22:00:00Z", 10., flags=("R", "A", "R")),
                     row("2024-07-30T23:00:00Z", None),
                     row("2024-07-31T00:00:00Z", SECRET_VALUE),
                     row("2024-08-01T00:00:00Z", SECRET_VALUE, core=False)]
        pq.write_table(pa.Table.from_pylist(self.rows), self.source)
        raw = self.source.read_bytes()
        self.partition = {"month": "2024-07", "bytes": len(raw), "sha256": sha(raw)}
        self.manifest = self.root / "synthetic-manifest.json"
        self.manifest.write_text(json.dumps({"native_partitions": [{"path": "2024-07.parquet", **self.partition}]}))
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(export, "PRIVATE_ROOT", self.root / ".local"))
        manifest_sha = sha(self.manifest.read_bytes())
        self.stack.enter_context(patch.object(export, "USGS_MANIFEST_SHA256", manifest_sha))
        self.stack.enter_context(patch.object(multicycle, "USGS_MANIFEST_SHA256", manifest_sha))
        self.args = dict(source_path=self.source, source_manifest_path=self.manifest, output_directory=self.output)

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def make_export(self):
        plan = export.plan_development_export(**self.args)
        receipt = export.execute_development_export(**self.args, execute=True,
                                                     administrative_plan_token=plan["administrative_plan_token"])
        reference = {"directory": str(self.output), "commit_sha256": receipt["commit_sha256"]}
        return receipt, reference

    def replace_commit(self, reference, mutate):
        path = self.output / "COMMIT.json"
        commit = json.loads(path.read_bytes())
        mutate(commit)
        raw = json.dumps(commit, sort_keys=True).encode()
        path.write_bytes(raw)
        return {**reference, "commit_sha256": sha(raw)}

    def test_plan_only_and_default_execution_never_decode_or_write_source_values(self):
        original = export.verified_local_bytes
        with patch("pyarrow.parquet.read_table", side_effect=AssertionError("Plan decoded values")) as decoder, \
             patch.object(export, "verified_local_bytes", wraps=original) as reader:
            plan = export.execute_development_export(**self.args)
            decoder.assert_not_called()
            self.assertEqual([Path(call.args[0]) for call in reader.call_args_list], [self.manifest])
        self.assertEqual(plan["status"], "PLAN_ONLY_NOT_EXECUTED")
        self.assertFalse(plan["source_values_decoded"])
        self.assertFalse(plan["token_is_human_authorization"])
        self.assertFalse(self.output.exists())

    def test_missing_and_nonresident_sources_can_be_planned_without_hydration(self):
        self.source.unlink()
        plan = export.plan_development_export(**self.args)
        self.assertEqual(plan["source_residency"], "MISSING")
        self.assertFalse(plan["ready_for_separately_authorized_execution"])
        information = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_flags=0x40000000, st_size=self.partition["bytes"])
        with patch.object(Path, "lstat", return_value=information), \
             patch.object(Path, "open", side_effect=AssertionError("Dataless file was opened")):
            self.assertEqual(export._source_residency(self.source, self.partition["bytes"]), "NONRESIDENT_NO_HYDRATION")

    def test_wrong_or_scoring_token_fails_before_source_decode(self):
        for token in (None, "", "SCORING_PERMIT:" + "a" * 64):
            with self.subTest(token=token), patch("pyarrow.parquet.read_table") as decoder:
                with self.assertRaisesRegex(ValueError, "administrative export token"):
                    export.execute_development_export(**self.args, execute=True, administrative_plan_token=token)
                decoder.assert_not_called()

    def test_full_original_source_hash_is_checked_before_numeric_decode(self):
        plan = export.plan_development_export(**self.args)
        raw = bytearray(self.source.read_bytes()); raw[-1] ^= 1; self.source.write_bytes(raw)
        with patch("pyarrow.parquet.read_table") as decoder:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                export.execute_development_export(**self.args, execute=True,
                                                  administrative_plan_token=plan["administrative_plan_token"])
            decoder.assert_not_called()

    def test_execution_returns_only_identities_counts_and_preboundary_artifact(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            receipt, reference = self.make_export()
        self.assertEqual(captured.getvalue(), "")
        self.assertNotIn(str(SECRET_VALUE), json.dumps(receipt))
        self.assertEqual(receipt["exported_rows"], 3)
        self.assertEqual(receipt["excluded_rows"], 3)
        self.assertFalse(receipt["protected_scoring_permitted"])
        self.assertEqual(receipt["physical_decode_scope"], export.DECODE_SCOPE)
        rows, identity = export.read_development_export(reference, self.partition)
        self.assertEqual(rows, self.rows[1:4])
        self.assertEqual(json.loads(rows[1]["source_qualifiers_json"]), ["R", "A", "R"])
        self.assertIsNone(rows[-1]["discharge_cfs"])
        self.assertFalse(rows[-1]["numeric_valid"])
        self.assertEqual(identity["source_partition"], self.partition)

    def test_export_is_immutable_and_existing_target_cannot_be_replaced(self):
        _, reference = self.make_export()
        original = (self.output / "COMMIT.json").read_bytes()
        plan = export.plan_development_export(**self.args)
        with patch("pyarrow.parquet.read_table") as decoder:
            with self.assertRaisesRegex(ValueError, "not ready"):
                export.execute_development_export(**self.args, execute=True,
                                                  administrative_plan_token=plan["administrative_plan_token"])
            decoder.assert_not_called()
        self.assertEqual((self.output / "COMMIT.json").read_bytes(), original)
        self.assertEqual(sha(original), reference["commit_sha256"])

    def test_wrong_commit_or_payload_hash_rejected_before_any_value_decode(self):
        _, reference = self.make_export()
        with patch("pyarrow.parquet.read_table") as decoder:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                export.read_development_export({**reference, "commit_sha256": "0" * 64}, self.partition)
            payload = self.output / "data.parquet"
            raw = bytearray(payload.read_bytes()); raw[-1] ^= 1; payload.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                export.read_development_export(reference, self.partition)
            decoder.assert_not_called()

    def test_cutoff_source_or_scoring_scope_tamper_is_rejected(self):
        _, reference = self.make_export()
        original = (self.output / "COMMIT.json").read_bytes()
        for key, value in (("cutoff_exclusive_utc", "2024-08-01T00:00:00Z"),
                           ("protected_scoring_permitted", True), ("source_manifest_sha256", "0" * 64)):
            (self.output / "COMMIT.json").write_bytes(original)
            changed = self.replace_commit(reference, lambda commit: commit.update({key: value}))
            with self.subTest(key=key), patch("pyarrow.parquet.read_table") as decoder:
                with self.assertRaises(ValueError):
                    export.read_development_export(changed, self.partition)
                decoder.assert_not_called()

    def test_export_with_protected_coordinates_fails_before_discharge_column_decode(self):
        _, reference = self.make_export()
        payload_path = self.output / "data.parquet"
        pq.write_table(pa.Table.from_pylist([self.rows[4]]), payload_path)
        raw = payload_path.read_bytes()
        def change(commit):
            commit.update(exported_rows=1, excluded_rows=commit["source_rows"] - 1,
                          payload={"name": "data.parquet", "sha256": sha(raw), "bytes": len(raw)})
        altered = self.replace_commit(reference, change)
        actual_read = pq.read_table
        with patch("pyarrow.parquet.read_table", wraps=actual_read) as decoder:
            with self.assertRaisesRegex(ValueError, "nondevelopment coordinate"):
                export.read_development_export(altered, self.partition)
            self.assertEqual(decoder.call_count, 1)
            self.assertNotIn("discharge_cfs", decoder.call_args.kwargs["columns"])

    def test_loader_uses_verified_export_without_opening_raw_mixed_partition(self):
        _, reference = self.make_export()
        self.source.unlink()  # Admission cannot silently fall back to original mixed bytes.
        start, end = utc_epoch("2024-07-30T22:00:00Z"), utc_epoch("2024-07-30T23:00:00Z")
        rows, receipts = multicycle.load_usgs_windows({"2024-07": self.source}, self.manifest,
                        [(start, end)], development_exports={"2024-07": reference})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].quality_flags, ("R", "A", "R"))
        self.assertAlmostEqual(rows[0].value, 10 * CFS_TO_CMS)
        self.assertIsNone(rows[1].value)
        self.assertEqual(receipts[0]["development_export"]["commit_sha256"], reference["commit_sha256"])
        self.assertEqual(receipts[0]["sha256"], self.partition["sha256"])

    def test_unpinned_export_cannot_enable_raw_mixed_month_reads(self):
        with patch("pyarrow.parquet.read_table") as decoder:
            with self.assertRaisesRegex(ValueError, "Pinned development export"):
                multicycle.load_usgs_windows({"2024-07": self.source}, self.manifest,
                    [(export.MONTH_START, export.MONTH_START + 3600)], development_exports={"2024-07": {"directory": self.output}})
            decoder.assert_not_called()

    def test_mixed_source_decode_error_does_not_echo_input_fragments(self):
        plan = export.plan_development_export(**self.args)
        with patch("pyarrow.parquet.read_table", side_effect=ValueError("private " + str(SECRET_VALUE))):
            with self.assertRaisesRegex(ValueError, "mixed-month decoding") as caught:
                export.execute_development_export(**self.args, execute=True,
                                                  administrative_plan_token=plan["administrative_plan_token"])
        self.assertNotIn(str(SECRET_VALUE), str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_export_directory_must_be_private_and_payload_cannot_be_symlink(self):
        with self.assertRaisesRegex(ValueError, "ignored .local"):
            export.plan_development_export(**{**self.args, "output_directory": self.root / "public"})
        _, reference = self.make_export()
        payload = self.output / "data.parquet"
        saved = self.root / "saved.parquet"
        payload.rename(saved); payload.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, "regular local file"):
            export.read_development_export(reference, self.partition)


if __name__ == "__main__":
    unittest.main()
