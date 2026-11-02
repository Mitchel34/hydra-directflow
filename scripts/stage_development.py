#!/usr/bin/env python3
"""Plan or execute an approved, budgeted development-corpus staging increment."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra_directflow.cache import _read
from hydra_directflow.corpus import plan_corpus, run_corpus
from hydra_directflow.development_export import inspect_development_export
from hydra_directflow.local_data import verified_local_bytes, WEIGHTS_SHA256, USGS_MANIFEST_SHA256
from hydra_directflow.multicycle import INDEX_SHA256
from hydra_directflow.reference import digest, require
from hydra_directflow.staging import SourceBudget, DevelopmentSourceProvider, atomic_json, durable_directory, MAX_BODY, MAX_GETS

OWNED_USGS_RESERVE = 20 * 1024**2


def load_execution_plan(staging_plan_path, weights_path, *, corpus_run_name="development-corpus-L168-v1"):
    require(re.fullmatch(r"development-corpus-L168-v[1-9][0-9]*", corpus_run_name), "Explicit versioned corpus run name required")
    raw = _read(staging_plan_path, 32 * 1024**2)
    source = json.loads(raw)
    expected = digest({key: value for key, value in source.items() if key not in ("plan_sha256", "created_at_utc")})
    require(source["schema"] == "hydra-development-staging-plan/v1" and source["plan_sha256"] == expected, "Private staging plan identity differs")
    require(source["limits"]["download_body_bytes"] == MAX_BODY and source["limits"]["get_attempts"] == MAX_GETS
            and source["limits"]["raw_hotset_bytes"] == 2 * 1024**3, "Staging plan limits differ from approved envelope")
    identities = source["sources"]
    approval_ref = identities["budget_approval"]
    approval = json.loads(verified_local_bytes(approval_ref["path"], approval_ref["sha256"], expected_bytes=approval_ref["bytes"], max_bytes=65536))
    require(approval["user_response"] == "Approve $150 combined ceiling" and approval["approved_total_incremental_usd"] == 150
            and approval["allocations_usd"]["development_staging"] == 35
            and approval["phase_release"]["development_staging"] == "AUTHORIZED_WITH_DECLARED_LIMITS"
            and approval["data_response_body_cap_bytes"] == MAX_BODY and approval["maximum_get_attempts"] == MAX_GETS,
            "Exact approved staging scope required")
    require(identities["qualified_index"]["sha256"] == INDEX_SHA256 and identities["usgs_manifest"]["sha256"] == USGS_MANIFEST_SHA256,
            "Source authority differs")
    verified_local_bytes(weights_path, WEIGHTS_SHA256, max_bytes=128 * 1024)
    verified_local_bytes(identities["usgs_manifest"]["path"], USGS_MANIFEST_SHA256, max_bytes=2 * 1024**2)
    corpus = plan_corpus()
    require([row["cycle_id"] for row in source["cycles"]] == corpus["required_cycle_ids"], "Source/corpus cycle scope differs")
    require([row["month"] for row in source["usgs_partitions"]] == corpus["required_usgs_months"], "Source/corpus USGS scope differs")
    provider_dir = ROOT / ".local/development-source-stage-v1"
    cache_dir = ROOT / ".local/development-corpus-derived-v1"
    run_dir = ROOT / ".local" / corpus_run_name
    # Billing control is kept outside the Documents/FileProvider tree.
    budget_dir = Path.home() / ".codex/hydra-directflow-runtime" / approval_ref["sha256"]
    implementation = {name: hashlib.sha256(_read(ROOT / name, 256 * 1024)).hexdigest()
                      for name in ("scripts/stage_development.py", "src/hydra_directflow/staging.py", "src/hydra_directflow/corpus.py")}
    identity = dict(staging_plan_sha256=expected, staging_plan_file_sha256=hashlib.sha256(raw).hexdigest(),
                    approval_sha256=approval_ref["sha256"], corpus_plan_sha256=corpus["plan_sha256"],
                    implementation_sha256=implementation, source_limits=source["limits"],
                    gcs_body_cap=MAX_BODY - OWNED_USGS_RESERVE, owned_usgs_reserved_bytes=OWNED_USGS_RESERVE,
                    provider_location_sha256=digest(str(provider_dir.resolve())), budget_location_sha256=digest(str(budget_dir.resolve())),
                    corpus_run_location_sha256=digest(str(run_dir.resolve())))
    public = dict(schema="hydra-approved-staging-execution-plan/v1", status="PLANNED_NOT_EXECUTED", **identity,
                  execution_plan_sha256=digest(identity), corpus_chunks=sum(len(u["chunks"]) for u in corpus["units"]),
                  candidate_origins=corpus["candidate_origins"], source_cycle_count=len(source["cycles"]),
                  cloud_jobs=0, cloud_writes=0, protected_scoring=False, approval_is_separate_human_record=True)
    private = dict(source=source, corpus=corpus, provider_dir=provider_dir, cache_dir=cache_dir, run_dir=run_dir,
                   budget_dir=budget_dir, weights_path=Path(weights_path))
    return public, private


def source_usgs_paths(private):
    paths, exports, unresolved = {}, {}, []
    for row in private["source"]["usgs_partitions"]:
        month = row["month"]
        if row["stage_kind"] == "VERIFIED_DEVELOPMENT_EXPORT_ONLY":
            require(month == "2024-07" and row["raw_mixed_month_staging_forbidden"] is True, "Only the declared July export is admitted")
            inspect_development_export(row["development_export_reference"], row["original_source_identity"])
            exports[month] = row["development_export_reference"]
            continue
        require(row["stage_kind"] == "RETAINED_NATIVE_DEVELOPMENT_ONLY" and month < "2024-07", "Raw protected/mixed-month materialization prohibited")
        source = row["payload_identity"]
        candidates = [private["provider_dir"] / "usgs" / (month + ".parquet")] + [Path(item["path"]) for item in row["local_candidates"]]
        selected = None
        for candidate in candidates:
            try:
                verified_local_bytes(candidate, source["sha256"], expected_bytes=source["bytes"], max_bytes=512 * 1024)
                selected = candidate; break
            except (OSError, ValueError):
                pass
        if selected is not None:
            paths[month] = str(selected)
        else:
            unresolved.append(month)
    return paths, exports, unresolved


def execute(staging_plan_path, weights_path, *, approved_execution_plan_sha256, stop_after_chunks=1, max_wall_seconds=900,
            corpus_run_name="development-corpus-L168-v1"):
    plan, private = load_execution_plan(staging_plan_path, weights_path, corpus_run_name=corpus_run_name)
    require(plan["execution_plan_sha256"] == approved_execution_plan_sha256, "Exact reviewed execution plan required")
    require(type(stop_after_chunks) is int and 1 <= stop_after_chunks <= 100, "Each invocation must have a bounded chunk allowance")
    require(type(max_wall_seconds) is int and 1 <= max_wall_seconds <= 1800, "Bounded invocation wall limit required")
    paths, exports, unresolved = source_usgs_paths(private)
    # Corpus checks each needed monthly source before calling transport. Thus an
    # initial increment can use verified resident months without hydrating others.
    durable_directory(private["budget_dir"])
    lock_path = private["budget_dir"] / "source-controller.lock"
    require(not lock_path.is_symlink(), "Controller lock cannot be a symlink")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = SourceBudget(private["budget_dir"], plan["approval_sha256"], body_cap=plan["gcs_body_cap"])
        try:
            with DevelopmentSourceProvider(plan=private["source"], directory=private["provider_dir"], budget=budget) as provider:
                result = run_corpus(qualified_index_path=private["source"]["sources"]["qualified_index"]["path"],
                    weights_path=private["weights_path"], usgs_paths=paths,
                    usgs_manifest_path=private["source"]["sources"]["usgs_manifest"]["path"],
                    cache_directory=private["cache_dir"], run_directory=private["run_dir"],
                    source_provider=provider, on_cache_committed=provider.on_cache_committed,
                    usgs_development_exports=exports, execution_enabled=True,
                    approved_plan_sha256=private["corpus"]["plan_sha256"], max_wall_seconds=max_wall_seconds,
                    stop_after_chunks=stop_after_chunks)
                summary = dict(schema="hydra-development-staging-increment/v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                               execution_plan_sha256=plan["execution_plan_sha256"], corpus=result, transport=provider.summary(),
                               unresolved_native_usgs_months=unresolved, owned_usgs_reserved_bytes=OWNED_USGS_RESERVE,
                               scientific_training=False, protected_scoring=False, cloud_jobs=0)
                atomic_json(private["provider_dir"] / "LATEST_INCREMENT.json", summary)
                return summary
        finally:
            budget.close()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-plan", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-execution-plan-sha256")
    parser.add_argument("--stop-after-chunks", type=int, default=1)
    parser.add_argument("--max-wall-seconds", type=int, default=900)
    parser.add_argument("--corpus-run-name", default="development-corpus-L168-v1")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is not None:
        require(not args.report.exists() and not args.report.is_symlink(), "Increment report must be new")
    if not args.execute:
        result = load_execution_plan(args.staging_plan, args.weights, corpus_run_name=args.corpus_run_name)[0]
    else:
        result = execute(args.staging_plan, args.weights, approved_execution_plan_sha256=args.approved_execution_plan_sha256,
                         stop_after_chunks=args.stop_after_chunks, max_wall_seconds=args.max_wall_seconds,
                         corpus_run_name=args.corpus_run_name)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True); atomic_json(args.report, result)
    if args.execute:
        print(json.dumps({"status": result["corpus"]["status"], "completed_origins": result["corpus"]["completed_origins"],
                          "chunks_built": result["corpus"]["chunks_built"], "transport": result["transport"]}))
        if result["corpus"]["failure"] is not None: raise SystemExit(1)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
