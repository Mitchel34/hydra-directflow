"""Confirmed qualifier policy tests on synthetic values; no source/model runs."""
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow import quality
from hydra_directflow.cache import CachedDevelopmentSession, build_cached_development
from hydra_directflow.contract import DevelopmentProtocol
from hydra_directflow.local_data import GAUGES, LocalCycle, utc_epoch
from hydra_directflow.multicycle import assemble_cases, build_multicycle_development, load_usgs_windows
from hydra_directflow.quality import PRIMARY_APPROVED, STRICT_A, admit_usgs_quality, quality_admitter, quality_policy_binding
from hydra_directflow.reference import Observation

BASE = utc_epoch("2021-08-23T00:00:00Z")
ORIGIN = "2021-08-23T02:00:00Z"
CYCLE = "20210823_00Z"


def observations():
    values = []
    target_flags = {3: ("A", "R"), 4: ("A", "e"), 6: ("A", "X"),
                    7: ("A", "P"), 8: ("R",), 9: ()}
    for site in GAUGES:
        for hour in range(-23, 21):
            if site == GAUGES[0] and hour == 5:
                continue  # A genuine missing fixture row, not a value to impute.
            flags = ("A",)
            if site == GAUGES[0]:
                flags = ("R", "A", "R") if hour == 0 else target_flags.get(hour, flags)
            value = None if site == GAUGES[0] and hour == 10 else float(100 + hour)
            values.append(Observation(site, BASE + hour * 3600, None, value, flags, "synthetic-usgs-source"))
    return values


def soil_cycle():
    record = {"record_index": 0, "group": "soil_moisture", "lead": 0,
              "source": {"key": "synthetic-soil"}, "availability_status": "SYNTHETIC_FIXTURE"}
    return LocalCycle(BASE, np.array([[.2, .3, .4]]), np.ones((1, 3), bool), (record,), 1, {},
                      {"groups": [{"group": "soil_moisture", "status": "SYNTHETIC_FIXTURE"}]})


def builder_args():
    return dict(archive_specs=[{"cycle_id": CYCLE, "archive_path": "synthetic-unused", "commit_path": "synthetic-unused"}],
                qualified_index_path="synthetic-unused", weights_path="synthetic-unused", usgs_paths={"2021-08": "synthetic-unused"},
                usgs_manifest_path="synthetic-unused", origins_utc=[ORIGIN], history_hours=24)


def pure_cases(**policies):
    return assemble_cases({}, {}, observations(), [BASE + 7200], DevelopmentProtocol(), **policies)


class QualityPolicies(unittest.TestCase):
    def test_primary_requires_a_and_permits_r_in_either_order(self):
        for flags in (("A",), ("A", "R"), ("R", "A"), ["A", "R"]):
            with self.subTest(flags=flags):
                self.assertTrue(admit_usgs_quality(flags, policy_id=PRIMARY_APPROVED))
        self.assertFalse(admit_usgs_quality(("R",), policy_id=PRIMARY_APPROVED))

    def test_primary_duplicate_flags_are_idempotent_and_source_sequence_is_untouched(self):
        for original in (("A", "A"), ("R", "A", "R"), ["R", "A", "A", "R"]):
            before = original.copy() if isinstance(original, list) else original
            self.assertTrue(admit_usgs_quality(original, policy_id=PRIMARY_APPROVED))
            self.assertEqual(original, before)

    def test_primary_excludes_provisional_estimated_unknown_and_empty_flags(self):
        for flags in (("A", "P"), ("P", "A"), ("A", "e"), ("e", "R", "A"),
                      ("A", "X"), ("A", "r"), ("a", "R"), ("A", ""), (), []):
            with self.subTest(flags=flags):
                self.assertFalse(admit_usgs_quality(flags, policy_id=PRIMARY_APPROVED))

    def test_malformed_flag_containers_or_nonstring_members_are_rejected(self):
        for flags in ("A", "AR", "A,R", None, 1, True, {"A"}, {"A": True},
                      ("A", None), ("A", 1), ("A", True), ("A", ["R"]), (b"A",)):
            with self.subTest(flags=flags):
                self.assertFalse(admit_usgs_quality(flags, policy_id=PRIMARY_APPROVED))

    def test_legacy_default_remains_exact_single_a(self):
        self.assertTrue(admit_usgs_quality(("A",)))
        for flags in (("A", "R"), ("R", "A"), ("A", "A"), ["A"], "A", (), None):
            with self.subTest(flags=flags):
                self.assertFalse(admit_usgs_quality(flags))

    def test_unknown_policy_never_falls_back(self):
        for identifier in (None, True, "", "primary", [], {"policy": STRICT_A}):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                quality_admitter(identifier)

    def test_receipt_binding_names_both_policies_and_exact_adapter_code(self):
        bound = quality_policy_binding(input_quality_policy=PRIMARY_APPROVED, target_quality_policy=STRICT_A)
        self.assertEqual(bound["input_quality_policy"], PRIMARY_APPROVED)
        self.assertEqual(bound["target_quality_policy"], STRICT_A)
        self.assertEqual(bound["quality_policy"], "SEPARATE_INPUT_AND_TARGET_QUALITY_POLICIES")
        self.assertEqual(bound["quality_implementation_sha256"]["quality.py"], sha256(Path(quality.__file__).read_bytes()).hexdigest())


class QualityAssembly(unittest.TestCase):
    def test_default_and_explicit_legacy_outputs_are_identical(self):
        implicit = pure_cases()
        explicit = pure_cases(input_quality_policy=STRICT_A, target_quality_policy=STRICT_A)
        self.assertEqual(set(implicit), set(explicit))
        for key in implicit:
            np.testing.assert_equal(implicit[key], explicit[key], err_msg=key)

    def test_input_and_target_policies_are_independent(self):
        input_primary = pure_cases(input_quality_policy=PRIMARY_APPROVED, target_quality_policy=STRICT_A)
        target_primary = pure_cases(input_quality_policy=STRICT_A, target_quality_policy=PRIMARY_APPROVED)
        self.assertTrue(input_primary["history_mask"][0, -1].all())
        self.assertFalse(input_primary["target_mask"][0, 0])
        self.assertFalse(target_primary["history_mask"][0, -1].any())
        self.assertTrue(target_primary["target_mask"][0, 0])
        for key in ("case_id", "origin_time", "history_time", "target_time", "source_cycle", "soil_source_cycle"):
            np.testing.assert_equal(input_primary[key], target_primary[key], err_msg=key)

    def test_primary_keeps_original_flags_in_order_including_duplicates(self):
        values = pure_cases(input_quality_policy=PRIMARY_APPROVED, target_quality_policy=PRIMARY_APPROVED)
        self.assertEqual(json.loads(values["history_quality_flags"][0, -1]), ["R", "A", "R"])
        self.assertEqual(json.loads(values["target_quality_flags"][0, 0]), ["A", "R"])
        self.assertEqual(values["history_source_ids"][0, -1], "synthetic-usgs-source")

    def test_primary_does_not_fill_missing_invalid_or_rejected_targets(self):
        values = pure_cases(input_quality_policy=PRIMARY_APPROVED, target_quality_policy=PRIMARY_APPROVED)
        self.assertTrue(values["target_mask"][0, 0])
        for horizon_index in range(1, 8):  # e, missing, unknown, P, R-only, empty and invalid numeric.
            self.assertFalse(values["target_mask"][0, horizon_index])
            self.assertTrue(np.isnan(values["target"][0, horizon_index]))
        self.assertEqual(json.loads(values["target_quality_flags"][0, 1]), ["A", "e"])
        self.assertIsNone(json.loads(values["target_quality_flags"][0, 2]))
        self.assertIn("TARGET_INCOMPLETE", values["eligibility_reason"][0])

    def test_invalid_policy_rejected_before_multicycle_source_reads(self):
        with patch("hydra_directflow.multicycle.qualified_cycle_rows") as source:
            with self.assertRaises(ValueError):
                build_multicycle_development(**builder_args(), input_quality_policy="unknown")
            source.assert_not_called()

    def test_multicycle_receipt_and_case_digest_bind_admission_policies(self):
        cycle = soil_cycle()
        with patch("hydra_directflow.multicycle.qualified_cycle_rows", return_value={CYCLE: {}}), \
             patch("hydra_directflow.multicycle.load_candidate_weights", return_value=np.eye(3)), \
             patch("hydra_directflow.multicycle.load_authenticated_cycle", return_value=(cycle, {"fixture": True})), \
             patch("hydra_directflow.multicycle.load_usgs_windows", return_value=(observations(), [{"month": "2021-08"}])):
            strict, before = build_multicycle_development(**builder_args())
            primary, after = build_multicycle_development(**builder_args(), input_quality_policy=PRIMARY_APPROVED,
                                                         target_quality_policy=PRIMARY_APPROVED)
        self.assertEqual(before["input_quality_policy"], STRICT_A)
        self.assertEqual(after["target_quality_policy"], PRIMARY_APPROVED)
        self.assertIn("quality.py", after["quality_implementation_sha256"])
        self.assertNotEqual(before["case_source_digest"], after["case_source_digest"])
        self.assertFalse(strict["history_mask"][0, -1].any())
        self.assertTrue(primary["history_mask"][0, -1].all())

    def test_native_qualifier_json_must_be_an_array_and_preserves_valid_sequence(self):
        row = dict(site=GAUGES[0], sample_epoch=BASE, sample_time_utc="2021-08-23T00:00:00Z", source_units="ft3/s",
                   discharge_cfs=1., numeric_valid=True, is_exact_utc_hour=True, is_core=True,
                   source_wire_sha256="a" * 64, historical_available_at_utc=None)
        manifest = {"native_partitions": [{"path": "2021-08.parquet", "sha256": "a" * 64, "bytes": 3}]}
        for flags in ("A", {"A": True}, ["A", 1], ["A", None], ["R", "A", "R"]):
            with self.subTest(flags=flags), \
                 patch("hydra_directflow.multicycle.verified_local_bytes", side_effect=[json.dumps(manifest).encode(), b"raw"]), \
                 patch("pyarrow.parquet.read_table", return_value=SimpleNamespace(to_pylist=lambda: [{**row, "source_qualifiers_json": json.dumps(flags)}])):
                if flags == ["R", "A", "R"]:
                    actual, _ = load_usgs_windows({"2021-08": "synthetic-unused"}, "synthetic-unused", [(BASE, BASE)])
                    self.assertEqual(actual[0].quality_flags, ("R", "A", "R"))
                else:
                    with self.assertRaisesRegex(ValueError, "JSON array of strings"):
                        load_usgs_windows({"2021-08": "synthetic-unused"}, "synthetic-unused", [(BASE, BASE)])


class QualityCache(unittest.TestCase):
    def test_invalid_cached_builder_policy_fails_before_session_creation(self):
        with patch("hydra_directflow.cache.CachedDevelopmentSession") as session:
            with self.assertRaises(ValueError):
                build_cached_development(**builder_args(), cache_directory="synthetic-unused", target_quality_policy="unknown")
            session.assert_not_called()

    def test_cache_retains_native_flags_and_reuses_same_stores_for_different_policies(self):
        args = builder_args()
        session_args = {key: value for key, value in args.items() if key not in ("origins_utc", "history_hours")}
        manifest = {"native_partitions": [{"path": "2021-08.parquet", "sha256": "a" * 64, "bytes": 3}]}
        original = observations()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch("hydra_directflow.cache.qualified_cycle_rows", return_value={CYCLE: {"fixture": True}}))
            stack.enter_context(patch("hydra_directflow.cache.load_candidate_weights", return_value=np.eye(3)))
            stack.enter_context(patch("hydra_directflow.cache.verified_local_bytes", return_value=json.dumps(manifest).encode()))
            loader = stack.enter_context(patch("hydra_directflow.cache.load_authenticated_cycle", return_value=(soil_cycle(), {"fixture": True})))
            month = stack.enter_context(patch("hydra_directflow.cache.load_usgs_windows", return_value=(original, [{"month": "2021-08"}])))
            with CachedDevelopmentSession(**session_args, cache_directory=directory) as session:
                strict, strict_receipt = session.build(origins_utc=[ORIGIN])
                primary, primary_receipt = session.build(origins_utc=[ORIGIN], input_quality_policy=PRIMARY_APPROVED,
                                                        target_quality_policy=PRIMARY_APPROVED)
                self.assertEqual(session.cache.counts["misses"], 2)
                self.assertEqual(session.cache.counts["memory_hits"], 2)
                self.assertEqual(strict_receipt["cache"]["cycle_keys"], primary_receipt["cache"]["cycle_keys"])
                self.assertEqual(strict_receipt["cache"]["month_keys"], primary_receipt["cache"]["month_keys"])
                self.assertIn("quality.py", primary_receipt["implementation_sha256"])
                self.assertNotEqual(strict_receipt["case_source_digest"], primary_receipt["case_source_digest"])
                rows = session._month("2021-08")[0]
                revised = next(row for row in rows if row.site == GAUGES[0] and row.valid_at == BASE)
                self.assertEqual(revised.quality_flags, ("R", "A", "R"))
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(month.call_count, 1)
            loader.side_effect = AssertionError("Verified cache must not reread source archive")
            month.side_effect = AssertionError("Verified cache must not reread USGS partition")
            with CachedDevelopmentSession(**session_args, cache_directory=directory) as resumed:
                disk, receipt = resumed.build(origins_utc=[ORIGIN], input_quality_policy=PRIMARY_APPROVED,
                                             target_quality_policy=PRIMARY_APPROVED)
                self.assertEqual(resumed.cache.counts["hits"], 2)
                self.assertEqual(resumed.cache.counts["misses"], 0)
                self.assertEqual(receipt["input_quality_policy"], PRIMARY_APPROVED)
                for key in primary:
                    np.testing.assert_equal(disk[key], primary[key], err_msg=key)
            self.assertFalse(strict["history_mask"][0, -1].any())
            self.assertTrue(primary["history_mask"][0, -1].all())
            self.assertEqual(original, observations())


if __name__ == "__main__":
    unittest.main()
