"""CLI path and admission guards; these tests do not open source data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preprocess_multicycle", ROOT / "scripts/preprocess_multicycle.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class PreprocessCLI(unittest.TestCase):
    def request(self):
        return dict(archive_specs=[dict(cycle_id="20210823_00Z", archive_path="one.tar", commit_path="one.json")],
                    qualified_index_path="index.jsonl", weights_path="weights.csv",
                    usgs_manifest_path="manifest.json", usgs_paths={"2021-08": "month.parquet"},
                    origins_utc=["2021-08-23T02:00:00Z"], history_hours=24)

    def test_relative_inputs_follow_request_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "request.json"
            path.write_text(json.dumps(self.request()))
            loaded = cli.load_request(path)
            self.assertEqual(loaded["archive_specs"][0]["archive_path"], str(Path(folder) / "one.tar"))
            self.assertEqual(loaded["usgs_paths"]["2021-08"], str(Path(folder) / "month.parquet"))

    def test_unknown_request_keys_and_unbound_archives_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "request.json"
            for change in ("extra", "commit"):
                request = self.request()
                if change == "extra":
                    request["allow_protected"] = True
                else:
                    del request["archive_specs"][0]["archive_path"]
                path.write_text(json.dumps(request))
                with self.assertRaises(ValueError):
                    cli.load_request(path)

    def test_bundle_kind_request_can_omit_commit_path_for_later_index_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "request.json"
            request = self.request()
            del request["archive_specs"][0]["commit_path"]
            path.write_text(json.dumps(request))
            self.assertNotIn("commit_path", cli.load_request(path)["archive_specs"][0])

    def test_symlinked_request_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "request.json"
            path.write_text(json.dumps(self.request()))
            link = Path(folder) / "link.json"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                cli.load_request(link)

    def test_output_must_be_private_and_not_replace_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".local").mkdir()
            tensor, report = root / ".local/output.npz", root / "report.json"
            with patch.object(cli, "__file__", str(root / "scripts/preprocess_multicycle.py")):
                cli.validate_outputs(self.request(), tensor, report)
                with self.assertRaisesRegex(ValueError, "ignored .local"):
                    cli.validate_outputs(self.request(), root / "reports/data.npz", report)
                tensor.write_bytes(b"original")
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    cli.validate_outputs(self.request(), tensor, report)
                self.assertEqual(tensor.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
