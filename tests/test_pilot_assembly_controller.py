"""Controller authority/resource tests; no real data or model runtime access."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/assemble_gap_pilot.py"
SPEC = importlib.util.spec_from_file_location("pilot_assembly_controller_tests", PATH)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


class AssemblyControllerTests(unittest.TestCase):
    def plan(self):
        value = dict(schema="hydra-cv1-assembly-controller-plan/v1", implementation={"test": "fixed"},
            authorities=controller.AUTHORITIES, maximum_seconds=900,
            maximum_parent_child_rss_bytes=4 * 1024**3, maximum_source_read_bytes=1024**3,
            maximum_output_bytes=1024**3, model_fits=0, scaler_fits=0, optimizer_updates=0,
            downloads=0, protected_assessment_access=False, scientific_execution_authorized=False)
        value["plan_sha256"] = controller.sha(controller.canonical(value))
        return value

    def validate(self, value):
        with patch.object(controller, "implementation", return_value={"test": "fixed"}):
            controller.validate_plan(value, value["plan_sha256"])

    def test_scope_is_fixed_and_requires_exact_plan_hash(self):
        value = self.plan(); self.validate(value)
        with patch.object(controller, "implementation", return_value={"test": "fixed"}):
            with self.assertRaisesRegex(ValueError, "Exact assembly plan"):
                controller.validate_plan(value, "0" * 64)
        value["maximum_seconds"] += 1
        with self.assertRaisesRegex(ValueError, "Exact assembly plan"):
            self.validate(value)

    def test_rehashed_scope_expansion_still_rejected(self):
        for key, change in (("maximum_seconds", 901), ("maximum_parent_child_rss_bytes", 8 * 1024**3),
                            ("maximum_source_read_bytes", 2 * 1024**3), ("maximum_output_bytes", 2 * 1024**3),
                            ("optimizer_updates", 1), ("scaler_fits", 1), ("model_fits", 1), ("downloads", 1),
                            ("scientific_execution_authorized", True), ("protected_assessment_access", True)):
            with self.subTest(key=key):
                value = self.plan(); value[key] = change; value.pop("plan_sha256")
                value["plan_sha256"] = controller.sha(controller.canonical(value))
                with self.assertRaisesRegex(ValueError, "scope or limits"):
                    self.validate(value)

    def test_changed_implementation_and_parent_authority_rejected(self):
        value = self.plan()
        with patch.object(controller, "implementation", return_value={"test": "different"}):
            with self.assertRaisesRegex(ValueError, "implementation or authority"):
                controller.validate_plan(value, value["plan_sha256"])
        value["authorities"] = {}; value.pop("plan_sha256")
        value["plan_sha256"] = controller.sha(controller.canonical(value))
        with self.assertRaisesRegex(ValueError, "implementation or authority"):
            self.validate(value)

    def test_network_and_model_imports_fail_closed(self):
        for event, args in (("socket.connect", ()), ("socket.getaddrinfo", ()),
                            ("socket.bind", ()), ("import", ("torch",)),
                            ("import", ("transformers.models",))):
            with self.subTest(event=event, args=args), self.assertRaises(RuntimeError):
                controller.deny_network_and_models(event, args)
        controller.deny_network_and_models("import", ("numpy",))
        controller.deny_network_and_models("open", ("resident-file", "r"))

    def test_receipts_never_overwrite_prior_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "receipt.json"
            controller.write_new(path, {"status": "PRESERVE"})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                controller.write_new(path, {"status": "RESET"})
            self.assertEqual(path.read_bytes(), original)

    def test_private_artifacts_reject_public_paths_and_symlink_escape(self):
        with self.assertRaisesRegex(ValueError, "tracked repository"):
            controller.private_path(controller.ROOT / "reports" / "private-plan.json")
        with self.assertRaisesRegex(ValueError, "stable local storage"):
            controller.private_path(Path("/tmp/private-plan.json"))
        self.assertEqual(controller.private_path(controller.ROOT / ".local" / "plan.json"),
                         controller.ROOT / ".local" / "plan.json")
        with tempfile.TemporaryDirectory(dir=controller.ROOT / ".local") as d:
            link = Path(d) / "escape"; link.symlink_to("/tmp")
            with self.assertRaisesRegex(ValueError, "real path"):
                controller.private_path(link / "plan.json")

    def test_private_receipt_parent_is_synced(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "claim.json"
            with patch.object(controller, "sync_directory") as sync:
                controller.write_new(path, {"claimed": True})
                sync.assert_called_once_with(path.parent)

    def test_public_report_path_is_separate_from_private_plan(self):
        self.assertEqual(controller.public_report_path(controller.ROOT / "reports" / "test.json"),
                         controller.ROOT / "reports" / "test.json")
        for path in (controller.ROOT / ".local" / "test.json", controller.ROOT / "reports" / "nested" / "test.json"):
            with self.assertRaisesRegex(ValueError, "reports JSON"):
                controller.public_report_path(path)

    def test_artifact_identity_is_manifest_hash_not_worker_wrapper_hash(self):
        manifest = {"status": "synthetic", "content_sha256": "a" * 64}
        raw = controller.canonical(manifest)
        wrapper = dict(manifest=manifest, manifest_sha256=controller.sha(raw), read_bytes=1, write_bytes=2)
        self.assertNotEqual(controller.sha(controller.canonical(wrapper)), wrapper["manifest_sha256"])
        self.assertEqual(controller.validated_manifest(wrapper, raw), manifest)
        wrapper["manifest_sha256"] = controller.sha(controller.canonical(wrapper))
        with self.assertRaisesRegex(ValueError, "actual immutable manifest"):
            controller.validated_manifest(wrapper, raw)

    def test_authority_selects_exact_delay2_roles_and_keeps_monthly_source_ids(self):
        sources, bounds, refit, counts = controller.authority(Path("/synthetic/cache"))
        self.assertEqual(len(sources), 37)
        self.assertEqual(min(sources), "2018-07"); self.assertEqual(max(sources), "2021-07")
        self.assertEqual(refit, "2020-07-31T00:00:00Z")
        self.assertEqual(bounds["fit"]["end_utc"], "2020-05-01T00:00:00Z")
        self.assertEqual(bounds["outer"]["end_utc"], "2021-07-31T00:00:00Z")
        self.assertEqual(counts["03161000"]["inner_stop"]["cases"], 2152)
        self.assertEqual(counts["03164000"]["fit"]["cases"], 13024)
        self.assertEqual(counts["03479000"]["outer"]["cases"], 8516)
        self.assertEqual(len(counts["03161000"]["outer"]["case_ids_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
