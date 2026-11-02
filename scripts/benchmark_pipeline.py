#!/usr/bin/env python3
"""Plan or explicitly run one bounded local multi-cycle pipeline timing case."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra_directflow.pipeline_timing import PipelineTimingSpec, run_pipeline_timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="Private JSON arguments to build_multicycle_development")
    parser.add_argument("--architecture", choices=("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer"), default="LSTM")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--gauge", choices=("03161000", "03164000", "03479000"), default="03161000")
    parser.add_argument("--history-hours", type=int, choices=(24, 168), default=168)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1, help="Engineering timing passes, at most three")
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--validation-origin", required=True, help="UTC start of chronological validation origins")
    parser.add_argument("--max-wall-seconds", type=int, default=300)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--report", type=Path, help="Optional NEW portable JSON report; source arrays/checkpoints stay private")
    args = parser.parse_args()
    spec = PipelineTimingSpec(architecture=args.architecture, device=args.device, gauge_id=args.gauge,
                              history_hours=args.history_hours, batch_size=args.batch_size,
                              engineering_epochs=args.epochs, max_optimizer_steps=args.max_updates,
                              validation_origin_utc=args.validation_origin, max_wall_seconds=args.max_wall_seconds)
    if args.report is not None and (args.report.exists() or args.report.is_symlink()):
        parser.error("Report output must be new")
    if args.execute and (args.run_directory is None or args.approved_plan_sha256 is None):
        parser.error("Execution requires a fresh private run directory and the exact prior plan digest")
    result = run_pipeline_timing(request_path=args.request, spec=spec, run_directory=args.run_directory,
                                 execution_enabled=args.execute, approved_plan_sha256=args.approved_plan_sha256)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    if not args.execute:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(json.dumps({key: result.get(key) for key in ("status", "optimizer_updates_precharged", "optimizer_updates_confirmed",
                                                          "supervised_wall_seconds", "combined_observed_peak_bytes")}))
        if result["status"] != "COMPLETED_ENGINEERING_TIMING":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
