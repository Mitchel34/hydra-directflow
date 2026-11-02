"""Causal multi-cycle regressions on explicitly synthetic source objects."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow.contract import DevelopmentProtocol
from hydra_directflow.local_data import GAUGES, FEATURE_GROUPS, LocalCycle, utc_epoch
from hydra_directflow.multicycle import (
    assemble_cases, build_multicycle_development, cycle_reference,
    load_authenticated_cycle, load_usgs_windows, qualified_cycle_rows,
    validate_request, _months_for_windows,
)
from hydra_directflow.reference import ASSESSMENT_START, Observation


BASE = utc_epoch("2021-08-23T00:00:00Z")


def fixture_cycle(hour, leads=36, soil=True, missing=None):
    """Synthetic rows never masquerade as sealed acquisition artifacts."""
    records, fields, groups = [], {}, []
    for feature, group in FEATURE_GROUPS:
        if not leads:
            continue
        fields[feature] = {"synthetic": True}
        for lead in range(1, leads + 1):
            records.append({"group": group, "lead": lead, "field_source_id": feature})
        groups.append({"group": group, "status": "SYNTHETIC"})
    if soil:
        records.append({"group": "soil_moisture", "lead": 0,
                        "source": {"key": f"synthetic-soil-{hour}"}, "availability_status": "SYNTHETIC"})
        groups.append({"group": "soil_moisture", "status": "SYNTHETIC"})
    means = np.full((len(records), 3), float(hour + 1))
    if soil:
        means[-1] = .2 + hour * .01
    masks = np.ones_like(means, dtype=bool)
    if missing:
        group, lead, site = missing
        index = next(i for i, record in enumerate(records) if record["group"] == group and record["lead"] == lead)
        means[index, site], masks[index, site] = np.nan, False
    cycle = LocalCycle(BASE + hour * 3600, np.empty((0, 957)), np.empty((0, 957), dtype=bool),
                       tuple(records), 0, fields, {"groups": groups})
    return cycle, (means, masks, masks.astype(float))


def fixture_observations(start=-180, end=60, missing=None):
    return [Observation(site, BASE + hour * 3600, None, float(hour + 200), ("A",), "synthetic-observation")
            for site in GAUGES for hour in range(start, end + 1) if (site, hour) != missing]


def assemble(specs, origins=(2,), lookback=24, observations=None):
    cycles, reduced = {}, {}
    for name, args in specs.items():
        cycles[name], reduced[name] = fixture_cycle(**args)
    return assemble_cases(cycles, reduced, fixture_observations() if observations is None else observations,
                          [BASE + h * 3600 for h in origins], DevelopmentProtocol(history_hours=lookback))


class MultiCycleCasesTests(unittest.TestCase):
    def test_custom_protocol_cannot_silently_change_hardcoded_development_lags(self):
        with self.assertRaisesRegex(ValueError, "fixed development"):
            assemble_cases({}, {}, [], [BASE + 7200], DevelopmentProtocol(weather_lag_hours=3))

    def test_latest_complete_run_excludes_newer_ordinary_h18_at_two_hour_lag(self):
        data = assemble({"extended00": dict(hour=0), "ordinary01": dict(hour=1, leads=18, soil=False)}, origins=(3,))
        self.assertEqual(data["source_cycle"].tolist(), ["extended00"] * 3)
        self.assertEqual(data["weather_source_leads"][0].tolist(), list(range(4, 22)))
        self.assertTrue(data["weather_mask"].all())

    def test_ordinary_only_retains_all_cases_with_unavailable_weather(self):
        data = assemble({"ordinary01": dict(hour=1, leads=18, soil=False)}, origins=(3,))
        self.assertEqual(data["weather"].shape, (3, 18, 4))
        self.assertTrue(np.isnan(data["weather"]).all())
        self.assertFalse(data["weather_mask"].any())
        self.assertTrue((data["weather_metadata_index"] == -1).all())
        self.assertIn("NO_COMPLETE_SINGLE_WEATHER_RUN", data["eligibility_reason"][0])

    def test_newer_complete_extended_run_selected_when_available(self):
        data = assemble({"cycle00": dict(hour=0), "cycle06": dict(hour=6)}, origins=(7, 8))
        self.assertEqual(data["source_cycle"].tolist(), ["cycle00"] * 3 + ["cycle06"] * 3)
        self.assertTrue((data["weather"][:3] == 1).all())
        self.assertTrue((data["weather"][3:] == 7).all())

    def test_incomplete_new_cycle_falls_back_as_whole_run_for_affected_gauge(self):
        data = assemble({"cycle00": dict(hour=0), "cycle06": dict(hour=6, missing=("humidity", 4, 1))}, origins=(8,))
        self.assertEqual(data["source_cycle"].tolist(), ["cycle06", "cycle00", "cycle06"])
        self.assertTrue((data["weather"][1] == 1).all())  # No cross-run stitching.

    def test_latest_eligible_soil_selected_independently_from_weather(self):
        data = assemble({"cycle00": dict(hour=0), "soil06": dict(hour=6, leads=0)}, origins=(8,))
        self.assertEqual(data["source_cycle"].tolist(), ["cycle00"] * 3)
        self.assertEqual(data["soil_source_cycle"].tolist(), ["soil06"] * 3)
        self.assertEqual(data["state"].tolist(), [[.26, 2.]] * 3)
        original = json.loads(data["source_metadata_json"][data["soil_metadata_index"][0]])
        self.assertEqual(original["cycle_id"], "soil06")

    def test_stale_soil_is_masked_without_searching_back(self):
        data = assemble({"cycle00": dict(hour=0)}, origins=(9,))
        self.assertTrue(data["weather_mask"].all())
        self.assertFalse(data["state_mask"].any())
        self.assertTrue(np.isnan(data["state"]).all())
        self.assertEqual(data["soil_source_cycle"].tolist(), ["cycle00"] * 3)

    def test_missing_history_retained_for_l168_but_not_l24(self):
        obs = fixture_observations(missing=(GAUGES[2], -72))
        long = assemble({"cycle00": dict(hour=0)}, lookback=168, observations=obs)
        short = assemble({"cycle00": dict(hour=0)}, lookback=24, observations=obs)
        self.assertEqual(long["history"].shape, (3, 168, 2))
        self.assertEqual(int((~long["history_mask"]).sum()), 2)
        self.assertIn("HISTORY_INCOMPLETE", long["eligibility_reason"][2])
        self.assertTrue(short["history_mask"].all())
        self.assertEqual(short["case_id"].tolist(), long["case_id"].tolist())

    def test_history_and_targets_have_per_case_exact_times(self):
        data = assemble({"cycle00": dict(hour=0)}, origins=(2, 3))
        self.assertEqual(data["origin_time"].tolist(), [BASE + 7200] * 3 + [BASE + 10800] * 3)
        self.assertEqual(data["history_time"][0, -1], BASE)
        self.assertEqual(data["history_time"][3, -1], BASE + 3600)
        self.assertEqual(data["target_time"][0].tolist(), [BASE + h * 3600 for h in range(3, 21)])
        self.assertEqual(data["site_id"].tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(data["history_source_ids"][0, 0], "synthetic-observation")
        self.assertEqual(data["target_quality_flags"][0, 0], '["A"]')

    def test_weather_provenance_dictionary_aligns_to_original_group_and_lead(self):
        data = assemble({"cycle00": dict(hour=0)}, origins=(2,))
        index = data["weather_metadata_index"][2, 0, 0]
        original = json.loads(data["source_metadata_json"][index])
        self.assertEqual(original["record"]["group"], "precipitation")
        self.assertEqual(original["record"]["lead"], 3)
        self.assertEqual(original["cycle_id"], "cycle00")
        self.assertNotIn("DEVELOPMENT", data["source_metadata_json"][index])


class MultiCycleAdmissionTests(unittest.TestCase):
    def test_duplicate_unsorted_origins_and_cycles_rejected(self):
        specs = [{"cycle_id": "20210823_00Z"}]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_request(specs * 2, ["2021-08-23T02:00:00Z"], 24)
        for origins in (["2021-08-23T02:00:00Z"] * 2,
                        ["2021-08-23T03:00:00Z", "2021-08-23T02:00:00Z"]):
            with self.assertRaisesRegex(ValueError, "unique, increasing"):
                validate_request(specs, origins, 24)

    def test_protected_targets_rejected_before_source_reads(self):
        args = dict(archive_specs=[{"cycle_id": "20210823_00Z"}], qualified_index_path="absent",
                    weights_path="absent", usgs_paths={}, usgs_manifest_path="absent",
                    origins_utc=["2024-07-30T06:00:00Z"])
        with patch("hydra_directflow.multicycle.qualified_cycle_rows") as reader:
            with self.assertRaisesRegex(ValueError, "boundary-crossing"):
                build_multicycle_development(**args)
            reader.assert_not_called()

    def test_protected_and_malformed_cycles_rejected(self):
        for value in ("20240801_00Z", "20210823_25Z", "../cycle", "20210230_00Z"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cycle_reference(value)

    def test_partition_windows_handle_leap_day_and_month_boundary(self):
        windows = [(utc_epoch("2024-02-29T23:00:00Z"), utc_epoch("2024-03-01T01:00:00Z"))]
        self.assertEqual(_months_for_windows(windows), {"2024-02", "2024-03"})
        with self.assertRaisesRegex(ValueError, "Required local USGS month"):
            load_usgs_windows({"2024-02": "absent"}, "absent", windows)

    def test_pre_boundary_window_in_protected_month_rejected_before_manifest_read(self):
        with self.assertRaisesRegex(ValueError, "Monthly partition overlaps"):
            load_usgs_windows({"2024-07": "absent"}, "absent",
                              [(utc_epoch("2024-07-01T00:00:00Z"), utc_epoch("2024-07-02T00:00:00Z"))])

    def test_protected_usgs_window_rejected_before_manifest_read(self):
        with self.assertRaisesRegex(ValueError, "Protected"):
            load_usgs_windows({}, "absent", [(ASSESSMENT_START - 3600, ASSESSMENT_START)])

    def test_index_wrong_size_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "index"
            p.write_text('{}\n')
            with self.assertRaisesRegex(ValueError, "byte count"):
                qualified_cycle_rows(p, ["20210823_00Z"])
            link = Path(tmp) / "link"
            link.symlink_to(p)
            with self.assertRaisesRegex(ValueError, "resident regular"):
                qualified_cycle_rows(link, ["20210823_00Z"])

    def test_wrong_commit_hash_cannot_admit_tar(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "commit"
            p.write_text('{}')
            row = {"cycle": "20210823_00Z", "location_kind": "commit", "original_quality_flags_rewritten": False,
                   "source_locations": {"commit": {"sha256": "0" * 64, "bytes": 2}}}
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_authenticated_cycle({"cycle_id": row["cycle"], "commit_path": p, "archive_path": "absent"}, row)


if __name__ == "__main__":
    unittest.main()
