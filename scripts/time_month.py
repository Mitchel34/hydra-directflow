#!/usr/bin/env python3
"""Plan, time or resume a genuine development month; no models or cloud jobs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra_directflow.monthly import run_month


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--chunk-hours", type=int, default=120)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--stop-after-chunks", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is not None and (args.report.exists() or args.report.is_symlink()):
        parser.error("Portable report must be new")
    result = run_month(request_path=args.request, month=args.month, cache_directory=args.cache_directory,
                       run_directory=args.run_directory, chunk_hours=args.chunk_hours,
                       execution_enabled=args.execute, approved_plan_sha256=args.approved_plan_sha256,
                       stop_after_chunks=args.stop_after_chunks)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: result.get(key) for key in ("status", "plan_sha256", "completed_origins", "completed_cases", "elapsed_seconds", "chunks_built", "chunks_reused")}))


if __name__ == "__main__":
    main()
