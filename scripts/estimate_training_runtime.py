#!/usr/bin/env python3
"""Recompute conditional local fit times from measured batches and audited counts."""
import argparse
import importlib.util
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measured_real_batches", ROOT / "scripts/benchmark_real_batches.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def estimate(measurements, audit, audit_sha):
    if measurements.get("status") != "PASS" or len(measurements.get("cases", [])) != 24:
        raise ValueError("Complete measured matrix required")
    by_key = {(c["architecture"], c["device"], c["batch_size"]): c for c in measurements["cases"]}
    counts = {(r["gauge"], r["split"]): r for r in audit["results"] if r["history_hours"] == 168}
    policies = {"CPU": {a: "cpu" for a in benchmark.ARCHITECTURES},
                "MPS": {a: "mps" for a in benchmark.ARCHITECTURES},
                "MIXED_THROUGHPUT": {"LSTM": "cpu", "GRU_Transformer": "cpu", "Transformer": "mps", "Mamba2_Transformer": "mps"}}
    results = []
    for scenario in ("known_joint_usgs", "post_purge_calendar"):
        for epochs in (20, 50):
            for label, policy in policies.items():
                parts = []
                for gauge in ("03161000", "03164000", "03479000"):
                    train, validation = counts[gauge, "candidate_training"], counts[gauge, "candidate_validation"]
                    ntrain = train["joint_usgs_metadata"]["ELIGIBLE"] if scenario == "known_joint_usgs" else train["candidate_origins_after_purge"]
                    nval = validation["joint_usgs_metadata"]["ELIGIBLE"] if scenario == "known_joint_usgs" else validation["candidate_origins_after_purge"]
                    for architecture, device in policy.items():
                        projection = benchmark.project_runtime(by_key[architecture, device, 64], train_cases=ntrain, validation_cases=nval,
                            epochs=epochs, trials=1, seeds=3, count_audit_sha256=audit_sha,
                            count_qualification=scenario + ": not final HRRR numeric eligibility or convergence")
                        parts.append(dict(gauge=gauge, **projection))
                total = sum(part["total_hours"] for part in parts)
                results.append(dict(count_scenario=scenario, epochs=epochs, device_policy=label, batch_size=64,
                                    base_fit_count=36, total_hours=total, plus_matching_q_only_sizing_hours=2 * total,
                                    q_only_sizing_qualification="Uses full-input cost as a transparent sizing proxy, not measured Q-only throughput",
                                    per_model_gauge=parts))
    return dict(schema="hydra-measured-local-training-scenarios/v1", status="CONDITIONAL_SCENARIOS_NOT_FINAL_STUDY_BUDGET",
                scenarios=results, count_audit_sha256=audit_sha, hardware=dict(cpu="Apple M1 Pro", ram_gib=32, cpu_cores=8, cpu_threads_per_worker=1),
                scientific_scope="Four current architectures x three gauges x three seeds, one configuration; no CV or additional architectures",
                qualifications=["Known-count and calendar calculations are scenarios, not confidence or runtime bounds",
                                "Full-model numerical eligibility, transforms, losses, seeds and trial budgets remain unfrozen",
                                "Full-cohort loading/materialization, convergence, failures, thermal/sleep delays and plotting excluded",
                                "Sequential CPU/MPS execution assumed; no concurrent speedup or CUDA extrapolation",
                                "CV, lookback/input controls and extra architectures add work and require their declared budgets"],
                optimizer_updates=0, cloud_jobs=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=ROOT / "reports/real-batch-benchmark.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/development-eligibility-refined.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink(): parser.error("Output must be new")
    raw, audit_raw = args.measurements.read_bytes(), args.audit.read_bytes()
    result = estimate(json.loads(raw), json.loads(audit_raw), hashlib.sha256(audit_raw).hexdigest())
    result["measurements_sha256"] = hashlib.sha256(raw).hexdigest()
    result["implementation_sha256"] = {"scripts/estimate_training_runtime.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                         "scripts/benchmark_real_batches.py": hashlib.sha256((ROOT / "scripts/benchmark_real_batches.py").read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    for row in result["scenarios"]:
        if row["device_policy"] == "MIXED_THROUGHPUT":
            print(row["count_scenario"], row["epochs"], round(row["total_hours"], 2))


if __name__ == "__main__":
    main()
