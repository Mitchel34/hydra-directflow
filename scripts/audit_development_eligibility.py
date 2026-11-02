#!/usr/bin/env python3
"""Audit development metadata and local residency without reading flow columns."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra_directflow.eligibility import audit_development_eligibility


def load_json(path, maximum=512 * 1024):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum or getattr(info, "st_flags", 0) & 0x40000000:
        raise ValueError("Configuration must be a bounded resident regular JSON file")
    return json.loads(path.read_text())


def request_sources(paths):
    usgs, archives = {}, []
    for path in paths:
        path = Path(path)
        request = load_json(path)
        def local(value):
            if type(value) is not str or not value:
                raise ValueError("Local source paths must be nonempty strings")
            item = Path(value)
            return str(item if item.is_absolute() else path.absolute().parent / item)
        for month, source in request["usgs_paths"].items():
            usgs.setdefault(month, local(source))  # First requested source has explicit precedence.
        for source in request["archive_specs"]:
            item = {"cycle_id": source["cycle_id"], "archive_path": local(source["archive_path"])}
            if source.get("commit_path"):
                item["commit_path"] = local(source["commit_path"])
            archives.append(item)
    return usgs, archives


def validate_outputs(report_path, private_path, source_paths):
    report, private = Path(report_path), Path(private_path)
    if report.suffix != ".json" or private.suffix != ".json":
        raise ValueError("Use JSON report and private inventory outputs")
    try:
        private.resolve().relative_to((ROOT / ".local").resolve())
    except ValueError as error:
        raise ValueError("Private inventory must stay in this checkout's ignored .local directory") from error
    protected = {Path(path).resolve() for path in source_paths}
    if report.resolve() == private.resolve() or any(path.resolve() in protected for path in (report, private)):
        raise ValueError("Outputs must be distinct from all source/configuration inputs")
    if any(path.exists() or path.is_symlink() for path in (report, private)):
        raise ValueError("Refusing to overwrite an existing report, inventory, or source")
    return report, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True, help="JSON array of name/start_utc/end_utc split objects")
    parser.add_argument("--local-request", type=Path, action="append", default=[], help="Existing private builder request; first source wins on duplicate USGS months")
    parser.add_argument("--aggregate-admission-proofs", action="store_true", help="Explicitly use pinned exhaustive aggregate counts plus exact-hour gaps for unread native months")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--private-inventory", type=Path, required=True)
    args = parser.parse_args()
    splits = load_json(args.splits)
    usgs, archives = request_sources(args.local_request)
    source_paths = [args.index, args.source_manifest, args.splits, *args.local_request, *usgs.values()]
    source_paths += [spec[key] for spec in archives for key in ("archive_path", "commit_path") if key in spec]
    report_path, private_path = validate_outputs(args.report, args.private_inventory, source_paths)
    report, private = audit_development_eligibility(qualified_index_path=args.index, usgs_manifest_path=args.source_manifest,
                                                   splits=splits, usgs_paths=usgs, local_archive_specs=archives,
                                                   use_aggregate_admission_proofs=args.aggregate_admission_proofs)
    report["implementation_sha256"]["scripts/audit_development_eligibility.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    private_raw = (json.dumps(private, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    report["private_inventory_identity"] = {"sha256": hashlib.sha256(private_raw).hexdigest(), "bytes": len(private_raw)}
    private_path.parent.mkdir(parents=True, exist_ok=True)
    with private_path.open("xb") as stream:
        stream.write(private_raw)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "metadata_months_verified": report["usgs_inventory"]["verified_metadata_months"],
                      "metadata_months_unknown": report["usgs_inventory"]["unknown_metadata_months"],
                      "comparison_rows": len(report["results"]), "flow_value_columns_decoded": False}))


if __name__ == "__main__":
    main()
