"""Synthetic timestamp/quality metadata only; no source reads or numeric flow."""
import importlib.util
import json
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_development_availability.py"
spec = importlib.util.spec_from_file_location("development_availability", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
C = audit.ORIGIN_START
HOUR = 3600
SITE = "03161000"


def row(t, *, flags=("A",), valid=True, estimated=False, core=True):
    return dict(site=SITE, sample_epoch=t, sample_time_utc=audit.iso(t),
                is_exact_utc_hour=t % HOUR == 0, numeric_valid=valid,
                source_qualifiers_json=json.dumps(list(flags)), is_estimated=estimated, is_core=core)


def complete_rows():
    return [row(t) for t in range(C - 169 * HOUR, C + 19 * HOUR, HOUR)]


def indexed(rows):
    return audit.index_metadata_rows(rows)[0][SITE]


class AvailabilityTests(unittest.TestCase):
    def test_common_window_has_556_origins_and_both_lengths_stay_inside_august(self):
        origins = audit.origin_axis()
        self.assertEqual(len(origins), 556)
        self.assertEqual(audit.iso(origins[0] - 169 * HOUR), "2021-08-01T01:00:00Z")
        self.assertEqual(audit.iso(origins[-1] + 18 * HOUR), "2021-08-31T23:00:00Z")

    def test_policies_handle_R_e_P_unknown_and_estimated_flag_explicitly(self):
        for flags, strict, approved in ((("A",), True, True), (("A", "R"), False, True),
                                         (("R", "A"), False, True), (("R",), False, False),
                                         (("A", "e"), False, False), (("A", "P"), False, False),
                                         (("A", "unknown"), False, False)):
            with self.subTest(flags=flags):
                record = indexed([row(C, flags=flags)])[C]
                self.assertEqual(not audit.exclusion_reasons(record, audit.POLICIES[0]), strict)
                self.assertEqual(not audit.exclusion_reasons(record, audit.POLICIES[1]), approved)
        estimated_without_e = indexed([row(C, estimated=True)])[C]
        for policy in audit.POLICIES:
            self.assertEqual(audit.exclusion_reasons(estimated_without_e, policy), ("estimated_flag_true",))

    def test_L168_gap_does_not_change_L24_and_quarter_hour_cannot_substitute(self):
        gap = C - 100 * HOUR
        rows = [r for r in complete_rows() if r["sample_epoch"] != gap] + [row(gap + 900)]
        exact = indexed(rows)
        for length, complete in ((24, 1), (168, 0)):
            count, _ = audit.analyze_gauge(exact, (C,), history_hours=length, policy=audit.POLICIES[0])
            self.assertEqual(count["history_complete"], complete)
            self.assertEqual(count["targets_complete"], 1)
            self.assertEqual(count["joint_USGS_complete"], complete)
            self.assertEqual(audit.prefix_check(exact, (C,), history_hours=length, policy=audit.POLICIES[0]),
                             {k: count[k] for k in ("history_complete", "targets_complete", "joint_USGS_complete")})

    def test_lag_and_target_endpoints_and_overlapping_exclusions(self):
        # C-1 and C are not in the lagged history or future targets.
        rows = [r for r in complete_rows() if r["sample_epoch"] not in (C - HOUR, C)]
        exact = indexed(rows)
        count, _ = audit.analyze_gauge(exact, (C,), history_hours=24, policy=audit.POLICIES[0])
        self.assertEqual(count["joint_USGS_complete"], 1)
        rows = [r for r in rows if r["sample_epoch"] not in (C - 2 * HOUR, C + 18 * HOUR)]
        rows.append(row(C - 2 * HOUR, flags=("A", "e"), valid=False, estimated=True))
        count, _ = audit.analyze_gauge(indexed(rows), (C,), history_hours=24, policy=audit.POLICIES[0])
        self.assertEqual((count["history_incomplete"], count["targets_incomplete"], count["joint_USGS_complete"]), (1, 1, 0))
        self.assertEqual(count["origin_exclusions_by_reason_overlap"]["history"],
                         {"missing_exact_hour": 0, "numeric_valid_false": 1,
                          "estimated_flag_true": 1, "qualifiers_not_admitted": 1})
        self.assertEqual(count["origin_exclusions_by_reason_overlap"]["targets"]["missing_exact_hour"], 1)

    def test_duplicate_keys_malformed_flags_wrong_clocks_and_types_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate native"):
            indexed([row(C), row(C)])
        with self.assertRaisesRegex(ValueError, "duplicate qualifier"):
            indexed([row(C, flags=("A", "A"))])
        for change in ({"is_exact_utc_hour": False}, {"numeric_valid": 1},
                       {"sample_time_utc": audit.iso(C + HOUR)}, {"sample_time_utc": "2021-08-08T03:00:00+01:00"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                indexed([{**row(C), **change}])

    def test_padding_excluded_and_qualifier_strings_preserved(self):
        original = row(C, flags=("R", "A"))
        original["source_qualifiers_json"] = '[ "R", "A" ]'
        exact, profile = audit.index_metadata_rows([original, row(audit.MONTH_END, core=False)])
        self.assertEqual(exact[SITE][C]["qualifiers_verbatim"], '[ "R", "A" ]')
        self.assertEqual(profile[SITE]["padding_native_rows"], 1)
        self.assertEqual(profile[SITE]["exact_core_rows"], 1)
        self.assertNotIn(audit.MONTH_END, exact[SITE])


if __name__ == "__main__":
    unittest.main()
