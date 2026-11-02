"""Synthetic metadata tests; no acquired discharge or weather payload reads."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra_directflow import eligibility as audit
from hydra_directflow.local_data import GAUGES

H = 3600
BASE = audit.utc_hour("2023-08-01T00:00:00Z")


def exact_month():
    return {site: {BASE + hour * H: {"numeric_valid": True, "qualifiers": ("A",)} for hour in range(744)} for site in GAUGES}


def native_row(stamp=BASE, flags=("A",), **changes):
    return {"site": GAUGES[0], "sample_epoch": stamp, "sample_time_utc": datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
            "is_exact_utc_hour": stamp % H == 0, "numeric_valid": True,
            "source_qualifiers_json": json.dumps(list(flags)), "is_core": True, **changes}


def structural_inventory(start, end):
    return [{"reference_epoch": stamp, "weather_group_support": audit.ELIGIBLE, "soil_group_support": audit.ELIGIBLE}
            for stamp in range(start - 24 * H, end, 6 * H)]


def fixture_partition(directory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(directory) / "2023-08.parquet"
    # The extra sentinel column must not be selected by the production reader.
    table = pa.Table.from_pylist([{**native_row(flags=("R", "A", "R")), "discharge_cfs": 123456789.}])
    pq.write_table(table, path)
    raw = path.read_bytes()
    manifest = {"native_partitions": [{"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]}
    source = Path(directory) / "manifest.json"
    source.write_text(json.dumps(manifest))
    return path, source, hashlib.sha256(source.read_bytes()).hexdigest()


class EligibilityWindows(unittest.TestCase):
    def test_exact_h18_boundary_purge_and_complete_windows(self):
        split = {"name": "synthetic", "start": BASE + 9 * 24 * H, "end": BASE + 11 * 24 * H}
        for lookback in (24, 168):
            rows = audit.count_split(split, exact_month(), {"2023-08"}, structural_inventory(split["start"], split["end"]), history_hours=lookback)
            for row in rows:
                self.assertEqual(row["requested_origins"], 48)
                self.assertEqual(row["purged_h18_boundary_origins"], 18)
                self.assertEqual(row["candidate_origins_after_purge"], 30)
                self.assertEqual(row["joint_usgs_metadata"], {audit.ELIGIBLE: 30, audit.INELIGIBLE: 0, audit.UNKNOWN: 0})
                self.assertEqual(row["usgs_and_hrrr_structure"][audit.ELIGIBLE], 30)

    def test_one_known_missing_observation_affects_history_and_targets_separately(self):
        split = {"name": "synthetic", "start": BASE + 9 * 24 * H, "end": BASE + 11 * 24 * H}
        exact = exact_month()
        del exact[GAUGES[0]][split["start"] + 18 * H]
        row = audit.count_split(split, exact, {"2023-08"}, [], history_hours=24)[0]
        self.assertEqual(row["target_metadata"][audit.INELIGIBLE], 18)
        self.assertEqual(row["input_history_metadata"][audit.INELIGIBLE], 10)
        self.assertEqual(row["joint_usgs_metadata"][audit.ELIGIBLE], 2)
        self.assertEqual(row["joint_usgs_metadata"][audit.UNKNOWN], 0)
        self.assertEqual(row["usgs_and_hrrr_structure"][audit.UNKNOWN], 30)

    def test_unknown_prior_month_is_unknown_not_missing_or_zero_eligibility(self):
        split = {"name": "synthetic", "start": BASE, "end": BASE + 48 * H}
        row = audit.count_split(split, exact_month(), {"2023-08"}, [], history_hours=24)[0]
        self.assertEqual(row["input_history_metadata"], {audit.ELIGIBLE: 5, audit.INELIGIBLE: 0, audit.UNKNOWN: 25})
        self.assertEqual(row["target_metadata"][audit.ELIGIBLE], 30)
        long = audit.count_split(split, exact_month(), {"2023-08"}, [], history_hours=168)[0]
        self.assertEqual(long["input_history_metadata"], {audit.ELIGIBLE: 0, audit.INELIGIBLE: 0, audit.UNKNOWN: 30})
        self.assertEqual(long["joint_usgs_metadata"][audit.UNKNOWN], 30)

    def test_unknown_target_month_does_not_erase_known_input_history(self):
        split = {"name": "synthetic", "start": BASE + 30 * 24 * H, "end": BASE + 32 * 24 * H}
        row = audit.count_split(split, exact_month(), {"2023-08"}, [], history_hours=24)[0]
        self.assertEqual(row["input_history_metadata"][audit.ELIGIBLE], 26)
        self.assertEqual(row["target_metadata"][audit.ELIGIBLE], 6)
        self.assertEqual(row["target_metadata"][audit.UNKNOWN], 24)
        self.assertEqual(row["target_metadata"][audit.INELIGIBLE], 0)

    def test_unknown_dominates_joint_fully_known_window_classification(self):
        self.assertEqual(audit.combine_status(audit.INELIGIBLE, audit.UNKNOWN), audit.UNKNOWN)
        self.assertEqual(audit.combine_status(audit.ELIGIBLE, audit.INELIGIBLE), audit.INELIGIBLE)

    def test_primary_duplicates_admitted_but_numeric_invalid_and_unknown_flags_reject(self):
        exact = exact_month()
        split = {"name": "synthetic", "start": BASE + 9 * 24 * H, "end": BASE + 11 * 24 * H}
        stamp = split["start"] + 18 * H
        exact[GAUGES[0]][stamp]["qualifiers"] = ("R", "A", "R")
        okay = audit.count_split(split, exact, {"2023-08"}, [], history_hours=24)[0]
        self.assertEqual(okay["joint_usgs_metadata"][audit.ELIGIBLE], 30)
        for change in ({"numeric_valid": False}, {"qualifiers": ("A", "e")}, {"qualifiers": ("A", "P")}, {"qualifiers": ("A", "X")}):
            with self.subTest(change=change):
                sample = exact_month()
                sample[GAUGES[0]][stamp].update(change)
                row = audit.count_split(split, sample, {"2023-08"}, [], history_hours=24)[0]
                self.assertEqual(row["target_metadata"][audit.INELIGIBLE], 18)

    def test_short_split_can_have_zero_candidates_without_division_or_false_coverage(self):
        split = {"name": "short", "start": BASE, "end": BASE + 18 * H}
        row = audit.count_split(split, exact_month(), {"2023-08"}, [], history_hours=168)[0]
        self.assertEqual(row["candidate_origins_after_purge"], 0)
        self.assertEqual(row["purged_h18_boundary_origins"], 18)
        self.assertEqual(sum(row["joint_usgs_metadata"].values()), 0)

    def test_split_validation_rejects_protected_naive_and_duplicate_ranges(self):
        valid = {"name": "fold", "start_utc": "2023-08-01T00:00:00Z", "end_utc": "2024-07-31T00:00:00Z"}
        self.assertEqual(audit.validate_splits([valid])[0]["end"], audit.DEVELOPMENT_END)
        for splits in ([valid, valid], [{**valid, "end_utc": "2024-08-01T00:00:00Z"}],
                       [{**valid, "start_utc": "2023-08-01T00:00:00"}], [{**valid, "name": "../unsafe"}]):
            with self.subTest(splits=splits), self.assertRaises(ValueError):
                audit.validate_splits(splits)


class MetadataIntegrity(unittest.TestCase):
    def test_flags_preserved_with_duplicates_without_a_flow_column(self):
        row = native_row(flags=("R", "A", "R"))
        exact, profiles = audit.metadata_rows([row], "2023-08")
        self.assertEqual(exact[GAUGES[0]][BASE]["qualifiers"], ("R", "A", "R"))
        self.assertEqual(profiles[GAUGES[0]]["exact_qualifier_counts_verbatim"], {row["source_qualifiers_json"]: 1})
        self.assertEqual(profiles[GAUGES[0]]["missing_exact_core_hours"], 743)

    def test_duplicate_keys_bad_clock_and_accidental_value_column_rejected(self):
        for rows in ([native_row(), native_row()], [native_row(sample_time_utc="2023-08-01T01:00:00Z")],
                     [{**native_row(), "discharge_cfs": 1.}], [native_row(numeric_valid=1)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                audit.metadata_rows(rows, "2023-08")

    def test_padding_and_quarter_hour_rows_do_not_replace_missing_exact_hours(self):
        exact, profile = audit.metadata_rows([native_row(BASE + 900), native_row(BASE - H, is_core=False)], "2023-08")
        self.assertEqual(exact[GAUGES[0]], {})
        self.assertEqual(profile[GAUGES[0]]["missing_exact_core_hours"], 744)

    def test_july_metadata_axis_excludes_protected_july31(self):
        start, _ = audit.month_bounds("2024-07")
        row = native_row(start)
        protected = native_row(audit.DEVELOPMENT_END)
        exact, profile = audit.metadata_rows([row, protected], "2024-07")
        self.assertNotIn(audit.DEVELOPMENT_END, exact[GAUGES[0]])
        self.assertEqual(profile[GAUGES[0]]["expected_exact_core_hours_before_assessment"], 720)

    def test_real_parquet_projection_never_selects_discharge_values(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            path, manifest, digest = fixture_partition(directory)
            actual = pq.read_table
            with patch.object(audit, "USGS_MANIFEST_SHA256", digest), patch("pyarrow.parquet.read_table", wraps=actual) as reader:
                exact, known, inventory = audit.inventory_usgs(manifest)
            self.assertEqual(known, {"2023-08"})
            self.assertEqual(len(inventory), 73)
            self.assertEqual(reader.call_args.kwargs["columns"], list(audit.METADATA_COLUMNS))
            self.assertNotIn("discharge_cfs", reader.call_args.kwargs["columns"])
            self.assertEqual(exact[GAUGES[0]][BASE]["qualifiers"], ("R", "A", "R"))
            self.assertEqual(sum(item["metadata_known"] for item in inventory), 1)

    def test_dataless_partition_is_never_opened_or_decoded(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest, digest = fixture_partition(directory)
            with patch.object(audit, "USGS_MANIFEST_SHA256", digest), \
                 patch.object(audit, "inspect_local", return_value={"status": "DATALESS_NOT_READ", "bytes": path.stat().st_size}), \
                 patch("pyarrow.parquet.read_table") as decode, \
                 patch.object(audit, "verified_local_bytes", wraps=audit.verified_local_bytes) as read:
                exact, known, inventory = audit.inventory_usgs(manifest)
            self.assertEqual(read.call_count, 1)  # Manifest only.
            decode.assert_not_called()
            self.assertEqual(known, set())
            self.assertEqual(next(item for item in inventory if item["month"] == "2023-08")["status"], "DATALESS_NOT_READ")

    def test_identity_failure_is_unknown_without_parquet_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest, _ = fixture_partition(directory)
            content = json.loads(manifest.read_text())
            content["native_partitions"][0]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(content))
            with patch.object(audit, "USGS_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()), \
                 patch("pyarrow.parquet.read_table") as decode:
                exact, known, inventory = audit.inventory_usgs(manifest)
            decode.assert_not_called()
            self.assertEqual(known, set())
            entry = next(item for item in inventory if item["month"] == "2023-08")
            self.assertFalse(entry["metadata_known"])
            self.assertEqual(entry["status"], "RESIDENT_IDENTITY_CHECK_FAILED")

    def test_local_presence_distinguishes_missing_symlink_and_size_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partition"
            self.assertEqual(audit.inspect_local(None)["status"], "NOT_SUPPLIED")
            self.assertEqual(audit.inspect_local(path)["status"], "MISSING")
            path.write_bytes(b"abc")
            self.assertEqual(audit.inspect_local(path, expected_bytes=4)["status"], "RESIDENT_SIZE_MISMATCH")
            link = Path(directory) / "link"
            link.symlink_to(path)
            self.assertEqual(audit.inspect_local(link)["status"], "NONREGULAR_NOT_ADMITTED")


class HrrrInventory(unittest.TestCase):
    def test_required_cycles_include_causal_warmup_and_purge_end(self):
        cycles = audit.required_six_hour_cycles(BASE, BASE + 48 * H)
        self.assertEqual(cycles[0], "20230731_06Z")
        self.assertEqual(cycles[-1], "20230802_00Z")
        self.assertEqual(len(cycles), 8)
        self.assertEqual(audit.required_six_hour_cycles(BASE, BASE + 18 * H), [])

    def test_legacy_and_absent_summary_are_unknown_not_source_gaps(self):
        self.assertEqual(audit.group_support(None, audit.WEATHER_GROUPS), audit.UNKNOWN)
        self.assertEqual(audit.group_support({"data_complete": True}, audit.WEATHER_GROUPS), audit.UNKNOWN)
        self.assertEqual(audit.group_support({"groups_missing": {"humidity": "confirmed_missing"}}, audit.WEATHER_GROUPS), audit.INELIGIBLE)

    def test_structural_weather_uses_one_complete_older_run_not_stitched_groups(self):
        complete = {"groups_available": list(audit.WEATHER_GROUPS)}
        partial_a = {"groups_available": ["temperature", "precipitation"]}
        partial_b = {"groups_available": ["humidity", "snow_water_equivalent"]}
        by_ref = {BASE + k * H: {"weather_group_support": audit.group_support(row, audit.WEATHER_GROUPS),
                                "soil_group_support": audit.ELIGIBLE}
                  for k, row in ((-6, partial_a), (0, partial_b), (6, partial_a))}
        weather, _ = audit.hrrr_structural_status(BASE + 8 * H, by_ref)
        self.assertEqual(weather, audit.INELIGIBLE)
        by_ref[BASE]["weather_group_support"] = audit.group_support(complete, audit.WEATHER_GROUPS)
        self.assertEqual(audit.hrrr_structural_status(BASE + 8 * H, by_ref), (audit.ELIGIBLE, audit.ELIGIBLE))

    def test_weather_and_soil_have_different_age_limits(self):
        by_ref = {BASE: {"weather_group_support": audit.ELIGIBLE, "soil_group_support": audit.ELIGIBLE},
                  BASE + 6 * H: {"weather_group_support": audit.INELIGIBLE, "soil_group_support": audit.INELIGIBLE}}
        self.assertEqual(audit.hrrr_structural_status(BASE + 9 * H, by_ref), (audit.ELIGIBLE, audit.INELIGIBLE))

    def test_archive_presence_is_only_stat_and_unmatched_commit_cannot_supply_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.tar"
            path.write_bytes(b"unread synthetic payload")
            commit = Path(directory) / "commit.json"
            commit.write_text('{}')
            cycle = "20230801_00Z"
            row = {"cycle": cycle, "groups_available": list(audit.WEATHER_GROUPS), "data_complete": True,
                   "source_locations": {"commit": {"sha256": "0" * 64, "bytes": 2}}}
            items = audit.inventory_hrrr({cycle: row}, [cycle], [{"cycle_id": cycle, "archive_path": str(path), "commit_path": str(commit)}])
            self.assertIsNone(items[0]["bundle_identity"])
            self.assertEqual(items[0]["local_archive_status"], "RESIDENT_STAT_ONLY_PAYLOAD_NOT_READ")
            self.assertEqual(items[0]["local_candidates"][0]["commit"]["status"], "RESIDENT_IDENTITY_CHECK_FAILED")

    def test_authenticated_commit_adds_exact_bundle_bytes_without_reading_tar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.tar"
            path.write_bytes(b"unread")
            bundle = {"bytes": 6, "sha256": "a" * 64, "durable_uri": "gs://example-bucket/cycles/20230801_00Z.tar", "generation": "123"}
            commit = Path(directory) / "commit.json"
            commit.write_text(json.dumps({"cycle": "20230801_00Z", "bundle": bundle}))
            identity = {"sha256": hashlib.sha256(commit.read_bytes()).hexdigest(), "bytes": commit.stat().st_size}
            row = {"data_complete": True, "groups_available": list(audit.WEATHER_GROUPS), "source_locations": {"commit": identity}}
            with patch.object(audit, "verified_local_bytes", wraps=audit.verified_local_bytes) as read:
                item = audit.inventory_hrrr({"20230801_00Z": row}, ["20230801_00Z"],
                    [{"cycle_id": "20230801_00Z", "archive_path": str(path), "commit_path": str(commit)}])[0]
            self.assertEqual(read.call_count, 1)
            self.assertEqual(Path(read.call_args.args[0]), commit)
            self.assertEqual(item["bundle_identity"]["bytes"], 6)
            self.assertEqual(item["bundle_bytes_source"], "AUTHENTICATED_LOCAL_COMMIT")

    def test_wrong_size_index_rejected_before_scanning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index"
            path.write_text('{}\n')
            with self.assertRaisesRegex(ValueError, "exact resident"):
                audit.read_required_index(path, ["20230801_00Z"])


class EligibilityCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("audit_development_eligibility", Path(__file__).resolve().parents[1] / "scripts/audit_development_eligibility.py")
        cls.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cli)

    def test_private_inventory_cannot_be_published_or_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(self.cli, "ROOT", Path(directory)):
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "ignored .local"):
                self.cli.validate_outputs(root / "report.json", root / "public.json", [])
            with self.assertRaisesRegex(ValueError, "source"):
                self.cli.validate_outputs(root / "source.json", root / ".local/private.json", [root / "source.json"])
            self.cli.validate_outputs(root / "report.json", root / ".local/private.json", [])

    def test_local_requests_resolve_paths_and_prefer_first_usgs_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for number in (1, 2):
                path = root / f"request-{number}.json"
                path.write_text(json.dumps({"usgs_paths": {"2023-08": f"native-{number}.parquet"},
                                            "archive_specs": [{"cycle_id": "20230801_00Z", "archive_path": "cycle.tar", "commit_path": "commit.json"}]}))
                paths.append(path)
            usgs, archives = self.cli.request_sources(paths)
            self.assertEqual(usgs["2023-08"], str(root / "native-1.parquet"))
            self.assertEqual(archives[0]["archive_path"], str(root / "cycle.tar"))


if __name__ == "__main__":
    unittest.main()
