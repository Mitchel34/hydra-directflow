"""Calendar completeness and crash-committed monthly chunk checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from collections import Counter
import numpy as np

import hydra_directflow.monthly as monthly
from hydra_directflow.monthly import month_calendar, _chunk_write, _chunk_read
from hydra_directflow.local_data import utc_epoch


class MonthlyTests(unittest.TestCase):
    def test_august_requires_all_hourly_origins_and_warmup_candidates(self):
        plan = month_calendar("2023-08")
        self.assertEqual((plan["origin_count"], plan["gauge_case_count"]), (744, 2232))
        self.assertEqual(len(plan["required_cycle_ids"]), 127)
        self.assertEqual(plan["required_cycle_ids"][0], "20230731_06Z")
        self.assertEqual(plan["required_cycle_ids"][-1], "20230831_18Z")
        self.assertEqual(plan["required_usgs_months"], ["2023-07", "2023-08", "2023-09"])
        self.assertEqual([origin for chunk in plan["chunks"] for origin in chunk["origins_utc"]], plan["origins_utc"])
        self.assertTrue(all(len(chunk["origins_utc"]) <= 128 and len(chunk["cycle_ids"]) <= 32 for chunk in plan["chunks"]))

    def test_every_potential_single_run_is_included_without_future_runs(self):
        plan = month_calendar("2023-08", chunk_hours=17)
        for chunk in plan["chunks"]:
            refs = [utc_epoch(c[:4] + "-" + c[4:6] + "-" + c[6:8] + "T" + c[9:11] + ":00:00Z") for c in chunk["cycle_ids"]]
            for origin in chunk["origins_utc"]:
                c = utc_epoch(origin)
                candidates = [r for r in refs if c - 18 * 3600 <= r <= c - 2 * 3600]
                self.assertTrue(candidates)
                self.assertTrue(all(1 <= (c + 3600 - r) // 3600 and (c + 18 * 3600 - r) // 3600 <= 36 for r in candidates))

    def test_leap_year_and_protected_partition_boundary(self):
        self.assertEqual(month_calendar("2024-02")["origin_count"], 696)
        for month in ("2024-06", "2024-07", "2024-08"):
            with self.assertRaisesRegex(ValueError, "[Pp]rotected"):
                month_calendar(month)

    def test_committed_chunk_reuses_exact_payload_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "chunks").mkdir()
            (root / "staging").mkdir()
            chunk = dict(index=0, origins_utc=["2023-08-01T00:00:00Z"], cycle_ids=["20230731_18Z"])
            payload = dict(index=0, origins_utc=chunk["origins_utc"], case_count=3,
                           eligibility_counts={"COMPLETE": 3}, cache={"cycle_keys": {"20230731_18Z": "key"}})
            _chunk_write(root, "plan", chunk, payload)
            target = root / "chunks/000"
            self.assertEqual(_chunk_read(target, "plan", chunk), payload)
            with self.assertRaises(ValueError):
                _chunk_read(target, "other-plan", chunk)
            with self.assertRaises(ValueError):
                _chunk_write(root, "plan", chunk, payload)
            commit = json.loads((target / "COMMIT.json").read_text())
            (target / (commit["payload_sha256"] + ".json")).write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                _chunk_read(target, "plan", chunk)

    def test_stopped_chunk_run_resumes_without_rebuilding_completed_windows(self):
        chunks = [dict(index=i, origins_utc=[f"2023-08-01T{i:02d}:00:00Z"], cycle_ids=["20230731_18Z"]) for i in range(3)]
        plan = dict(status="PLANNED_NOT_EXECUTED", plan_sha256="bound", implementation_sha256={},
                    calendar=dict(chunks=chunks, origin_count=3, gauge_case_count=9, required_cycle_ids=["20230731_18Z"], required_usgs_months=["2023-08"]))
        request = dict(history_hours=24, archive_specs=[], qualified_index_path="index", weights_path="weights",
                       usgs_paths={}, usgs_manifest_path="manifest")
        builds = []

        class FakeSession:
            def __init__(self, **unused):
                self.cache = SimpleNamespace(counts=Counter())
                self.initialization_seconds = 0.0
            def __enter__(self):
                return self
            def __exit__(self, *unused):
                pass
            def _cycle(self, unused):
                return (None, None, None, "cycle-key")
            def _month(self, unused):
                return (None, None, "month-key")
            def build(self, *, origins_utc, **unused):
                builds.append(origins_utc)
                arrays = dict(case_id=np.array([str(i) + origins_utc[0] for i in range(3)]),
                              source_cycle=np.array(["20230731_18Z"] * 3), soil_source_cycle=np.array(["20230731_18Z"] * 3),
                              eligibility_reason=np.array(["COMPLETE"] * 3))
                receipt = dict(case_count=3, complete_case_count=3, eligibility_counts={"COMPLETE": 3},
                               stage_seconds=dict(basin_store=1., discharge_store=1., case_assembly=1., total=3.),
                               cache=dict(cycle_keys={"20230731_18Z": "cycle-key"}, month_keys={"2023-08": "month-key"}))
                return arrays, receipt

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kwargs = dict(request_path="unused", month="2023-08", cache_directory=root / ".local/cache",
                          run_directory=root / ".local/run", execution_enabled=True, approved_plan_sha256="bound")
            with patch.object(monthly, "plan_month", return_value=(plan, request)), patch.object(monthly, "CachedDevelopmentSession", FakeSession), \
                    patch.object(monthly, "__file__", str(root / "src/hydra_directflow/monthly.py")):
                stopped = monthly.run_month(**kwargs, stop_after_chunks=1)
                self.assertEqual(stopped["status"], "STOPPED_AT_EXPLICIT_CHUNK_BOUNDARY")
                resumed = monthly.run_month(**kwargs)
                self.assertEqual((resumed["chunks_built"], resumed["chunks_reused"]), (2, 1))
                self.assertEqual(resumed["completed_cases"], 9)
                self.assertEqual(resumed["stage_seconds_new_chunks"]["total"], 6.)
                reused = monthly.run_month(**kwargs)
                self.assertEqual((reused["chunks_built"], reused["chunks_reused"]), (0, 3))
                self.assertEqual(len(builds), 3)
                process_state = SimpleNamespace(rss=1024)
                process = SimpleNamespace(memory_info=lambda: process_state)
                def over_budget(unused_session, unused_month):
                    process_state.rss = 5 * 1024**3
                    return (None, None, "month-key")
                with patch("psutil.Process", return_value=process), patch.object(FakeSession, "_month", over_budget):
                    with self.assertRaisesRegex(ValueError, "resource bound"):
                        monthly.run_month(**kwargs)


if __name__ == "__main__":
    unittest.main()
