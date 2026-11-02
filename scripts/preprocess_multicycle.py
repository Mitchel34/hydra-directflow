#!/usr/bin/env python3
"""Build an authenticated offline development cache from a private request JSON."""
import argparse
import hashlib
import json
from pathlib import Path
import stat


def load_request(path):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_size > 256 * 1024
            or getattr(info, "st_flags", 0) & 0x40000000):
        raise ValueError("Request must be a resident regular JSON file under 256 KiB")
    request = json.loads(path.read_text())
    required = {"archive_specs", "qualified_index_path", "weights_path",
                "usgs_manifest_path", "usgs_paths", "origins_utc", "history_hours"}
    if not isinstance(request, dict) or not required <= set(request) or not set(request) <= required | {"usgs_development_exports"}:
        raise ValueError("Request must contain exactly the documented builder keys")
    if not isinstance(request["archive_specs"], list) or not request["archive_specs"]:
        raise ValueError("At least one archive specification is required")
    for spec in request["archive_specs"]:
        if not isinstance(spec, dict) or set(spec) not in ({"cycle_id", "archive_path"}, {"cycle_id", "archive_path", "commit_path"}):
            raise ValueError("Archive specs require cycle_id and archive_path; commit_path is required for commit lineage")
    if "usgs_paths" in request and (not isinstance(request["usgs_paths"], dict)
                                   or not request["usgs_paths"]):
        raise ValueError("usgs_paths must map month names to local paths")
    # Paths are interpreted relative to the request's location, never the shell cwd.
    def local(value):
        if not isinstance(value, str) or not value:
            raise ValueError("Source paths must be nonempty strings")
        item = Path(value)
        return str(item if item.is_absolute() else path.absolute().parent / item)
    for key in ("qualified_index_path", "weights_path", "usgs_manifest_path"):
        if key in request:
            request[key] = local(request[key])
    if "usgs_paths" in request:
        request["usgs_paths"] = {month: local(value) for month, value in request["usgs_paths"].items()}
    if "usgs_development_exports" in request:
        if not isinstance(request["usgs_development_exports"], dict):
            raise ValueError("Export map required")
        for reference in request["usgs_development_exports"].values():
            if not isinstance(reference, dict) or set(reference) != {"directory", "commit_sha256"}:
                raise ValueError("Pinned export reference required")
            reference["directory"] = local(reference["directory"])
    for spec in request["archive_specs"]:
        for key in ("archive_path", "commit_path"):
            if key in spec:
                spec[key] = local(spec[key])
    return request


def validate_outputs(request, tensor_path, report_path):
    tensor, report = Path(tensor_path), Path(report_path)
    private_root = Path(__file__).resolve().parents[1] / ".local"
    try:
        tensor.resolve().relative_to(private_root.resolve())
    except ValueError as error:
        raise ValueError("Tensor output must be inside this checkout's ignored .local directory") from error
    if tensor.suffix != ".npz" or report.suffix != ".json":
        raise ValueError("Use .npz for tensors and .json for the portable receipt")
    source_paths = [request[key] for key in
                    ("qualified_index_path", "weights_path", "usgs_manifest_path")
                    if key in request]
    source_paths += list(request.get("usgs_paths", {}).values())
    source_paths += [spec[key] for spec in request["archive_specs"] for key in ("archive_path", "commit_path") if key in spec]
    sources = {Path(source).resolve() for source in source_paths}
    if tensor.resolve() == report.resolve() or any(path.resolve() in sources for path in (tensor, report)):
        raise ValueError("Output paths must be distinct from each other and all sources")
    if any(path.exists() or path.is_symlink() for path in (tensor, report)):
        raise ValueError("Refusing to overwrite an existing tensor or receipt")
    return tensor, report


def main():
    from hydra_directflow.quality import STRICT_A, PRIMARY_APPROVED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input-quality-policy", choices=(STRICT_A, PRIMARY_APPROVED), default=STRICT_A)
    parser.add_argument("--target-quality-policy", choices=(STRICT_A, PRIMARY_APPROVED), default=STRICT_A)
    args = parser.parse_args()
    request = load_request(args.request)
    tensor, receipt = validate_outputs(request, args.tensor_output, args.report)
    from hydra_directflow.multicycle import build_multicycle_development
    import numpy as np
    arrays, report = build_multicycle_development(**request, input_quality_policy=args.input_quality_policy,
                                                 target_quality_policy=args.target_quality_policy)
    tensor.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with tensor.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    content = tensor.read_bytes()
    report["private_tensor_artifact"] = {"format": "npz", "bytes": len(content),
                                          "sha256": hashlib.sha256(content).hexdigest()}
    report["implementation_sha256"] = {
        str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(),
                     Path(__file__).resolve().parents[1] / "src/hydra_directflow/multicycle.py",
                     Path(__file__).resolve().parents[1] / "src/hydra_directflow/local_data.py",
                     Path(__file__).resolve().parents[1] / "src/hydra_directflow/quality.py",
                     Path(__file__).resolve().parents[1] / "src/hydra_directflow/development_export.py",
                     Path(__file__).resolve().parents[1] / "src/hydra_directflow/lineage.py",
                     Path(__file__).resolve().parents[1] / "src/hydra_directflow/contract.py",
                     Path(__file__).resolve().parents[1] / "src/hydra_directflow/reference.py")}
    with receipt.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "cases": len(arrays["site_id"]),
                      "history_hours": request["history_hours"], "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
