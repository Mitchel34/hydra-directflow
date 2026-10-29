#!/usr/bin/env python3
"""Run the pinned offline development chunk; all private paths are CLI inputs."""
import argparse
import hashlib
import json
from pathlib import Path

from hydra_directflow.local_data import build_development_smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--usgs", type=Path, required=True)
    parser.add_argument("--usgs-manifest", type=Path, required=True)
    parser.add_argument("--origin", default="2021-08-23T02:00:00Z", help="Bounded August 23 origin: 02Z through 07Z")
    parser.add_argument("--history-hours", type=int, choices=(24, 168), default=24)
    parser.add_argument("--report", type=Path, default=Path("reports/local-smoke.json"))
    parser.add_argument("--tensor-output", type=Path,
                        help="Optional private NPZ; must be under an existing .local directory")
    args = parser.parse_args()
    source_paths = {p.resolve() for p in (args.archive, args.weights, args.usgs, args.usgs_manifest)}
    if args.report.resolve() in source_paths or args.report.is_symlink():
        parser.error("Report output cannot overwrite or link to a source file")
    if args.tensor_output is not None:
        if args.tensor_output.resolve() in source_paths:
            parser.error("Tensor output cannot overwrite a source file")
        if ".local" not in args.tensor_output.absolute().parts:
            parser.error("Private tensor output must live under .local")
        if args.tensor_output.exists():
            parser.error("Refusing to overwrite an existing private tensor artifact")
        if args.tensor_output.suffix != ".npz":
            parser.error("Private tensor output must use .npz")
    arrays, report = build_development_smoke(archive_path=args.archive, weights_path=args.weights,
                                            usgs_path=args.usgs, usgs_manifest_path=args.usgs_manifest,
                                            origin_utc=args.origin, history_hours=args.history_hours)
    if args.tensor_output is not None:
        import numpy as np
        args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
        with args.tensor_output.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report["private_tensor_artifact_written"] = True
        artifact = args.tensor_output.read_bytes()
        report["private_tensor_artifact"] = {"format": "npz", "bytes": len(artifact),
                                             "sha256": hashlib.sha256(artifact).hexdigest()}
    else:
        report["private_tensor_artifact_written"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "origins": 1, "gauges": 3,
                      "history_hours": args.history_hours, "horizons": 18, "training_steps": 0}))


if __name__ == "__main__":
    main()
