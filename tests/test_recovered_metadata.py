"""Synthetic retained-component metadata recovery; no network or model work."""
from copy import deepcopy
import json
import unittest

from hydra_directflow.local_data import _field_metadata, sha256

CYCLE = "20180807_18Z"
PREFIX = "20180807/18/precipitation/"
BASE = "https://example.invalid/surface/APCP_1hr_acc_fcst/"
FIELD_PATH = "surface/APCP_1hr_acc_fcst"


def fixture():
    components = {name + "/" + suffix: {"synthetic_coordinate": name, "kind": suffix}
                  for name in (FIELD_PATH, "forecast_period", "time", "forecast_reference_time",
                               "projection_x_coordinate", "projection_y_coordinate") for suffix in (".zarray", ".zattrs")}
    spec, attrs = {"shape": [36, 1059, 1799], "chunks": [36, 150, 150], "dtype": "<f2"}, {"units": "kg m-2"}
    components[FIELD_PATH + "/.zarray"], components[FIELD_PATH + "/.zattrs"] = spec, attrs
    sources = [{"key": PREFIX + "zmetadata", "url": BASE + ".zmetadata", "missing": True, "http_status": 404,
                "missing_proof": {"http_status": 404, "outcome": "MISSING_404"}}]
    members = {}
    for name, value in components.items():
        body = json.dumps(value, sort_keys=True).encode()
        member = "source_bytes/" + sha256(name.encode()) + ".bin"
        members[member] = body
        sources.append({"key": PREFIX + name, "url": BASE + name,
                        "relative_path": "cycles/" + CYCLE + "/" + member,
                        "bytes": len(body), "sha256": sha256(body)})
    field = {"field_source_id": "20180807_18_precipitation", "metadata_sources": sources,
             "source": {"url": BASE + FIELD_PATH + "/0.3.9"}, "spec": deepcopy(spec), "attrs": deepcopy(attrs)}
    return members, field, components


class RecoveredMetadataTests(unittest.TestCase):
    def test_exact_twelve_components_rebuild_view_without_rewriting_receipts(self):
        members, field, expected = fixture()
        before_members, before_field = deepcopy(members), deepcopy(field)
        actual = _field_metadata(members, field, cycle_id=CYCLE)
        self.assertEqual(actual, expected)
        self.assertEqual(members, before_members)
        self.assertEqual(field, before_field)
        self.assertTrue(field["metadata_sources"][0]["missing"])
        self.assertNotIn("relative_path", field["metadata_sources"][0])

    def test_consolidated_path_has_identical_dictionary_result(self):
        members, field, expected = fixture()
        raw = json.dumps({"metadata": expected}).encode()
        members["source_bytes/" + "a" * 64 + ".bin"] = raw
        field["metadata_sources"] = [{"relative_path": "cycles/" + CYCLE + "/source_bytes/" + "a" * 64 + ".bin",
                                       "bytes": len(raw), "sha256": sha256(raw)}]
        self.assertEqual(_field_metadata(members, field, cycle_id=CYCLE), expected)

    def test_missing_or_wrong_original_404_proof_is_not_inferred(self):
        for change in ({"missing": False}, {"http_status": 403}, {"missing_proof": {"http_status": 404, "outcome": "UNKNOWN"}}):
            members, field, _ = fixture(); field["metadata_sources"][0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "404 proof"):
                _field_metadata(members, field, cycle_id=CYCLE)

    def test_missing_duplicate_or_unapproved_component_is_rejected(self):
        members, field, _ = fixture()
        field["metadata_sources"].pop()
        with self.assertRaisesRegex(ValueError, "twelve"):
            _field_metadata(members, field, cycle_id=CYCLE)
        for replacement in ("duplicate", "unapproved"):
            members, field, _ = fixture()
            if replacement == "duplicate":
                field["metadata_sources"][-1] = deepcopy(field["metadata_sources"][1])
            else:
                field["metadata_sources"][-1]["key"] = PREFIX + "unapproved/.zattrs"
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "component"):
                _field_metadata(members, field, cycle_id=CYCLE)

    def test_component_namespace_or_url_cannot_change_roots(self):
        for key, value in (("key", "20180808/18/precipitation/time/.zattrs"),
                           ("url", "https://example.invalid/other-root/time/.zattrs")):
            members, field, _ = fixture(); field["metadata_sources"][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _field_metadata(members, field, cycle_id=CYCLE)
        members, field, _ = fixture(); field["source"]["url"] = BASE + "other/0.3.9"
        with self.assertRaisesRegex(ValueError, "root"):
            _field_metadata(members, field, cycle_id=CYCLE)

    def test_retained_field_spec_and_attributes_must_equal_component_bytes(self):
        for key in ("spec", "attrs"):
            members, field, _ = fixture(); field[key]["changed"] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "specification/attributes"):
                _field_metadata(members, field, cycle_id=CYCLE)

    def test_component_hash_and_cross_cycle_member_binding_are_enforced(self):
        members, field, _ = fixture(); field["metadata_sources"][1]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Source hash"):
            _field_metadata(members, field, cycle_id=CYCLE)
        members, field, _ = fixture()
        field["metadata_sources"][1]["relative_path"] = field["metadata_sources"][1]["relative_path"].replace(CYCLE, "20180808_18Z")
        with self.assertRaisesRegex(ValueError, "different cycle"):
            _field_metadata(members, field, cycle_id=CYCLE)


if __name__ == "__main__":
    unittest.main()
