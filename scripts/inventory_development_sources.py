#!/usr/bin/env python3
"""Reconcile committed development source identities using metadata only.

No cache/model NPZ or raw weather archive is opened. The catalog records the
authenticated original archive sizes, not download charges or resident bytes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tempfile
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from inspect_staging_progress import (digest, sha, require, read_json, resident_directory,
                                     load_plan, read_chunk, scan_progress, is_digest)
from hydra_directflow.cache_reuse import PRODUCER_CODE, CONSUMER_CODE, POLICY, RULE_ID
from hydra_directflow.multicycle import qualified_cycle_rows, INDEX_SHA256, INDEX_BYTES
from hydra_directflow.lineage import authenticate_bundle_identity

SCHEMA = "hydra-development-source-final-inventory/v1"
MAX_CYCLES = 8841


def pinned_json(path, expected, maximum):
    require(is_digest(expected), "PINNED_METADATA_HASH_REQUIRED")
    value, raw, _ = read_json(path, maximum)
    require(sha(raw) == expected, "PINNED_METADATA_HASH_MISMATCH")
    return value


def object_identity(value):
    uri = value.get("durable_uri")
    generation = value.get("generation")
    if value.get("uri_generation"):
        alternate_uri, alternate_generation = value["uri_generation"].rsplit("#", 1)
        require(uri in (None, alternate_uri) and generation in (None, alternate_generation, int(alternate_generation)),
                "OBJECT_LOCATION_CONFLICT")
        uri, generation = alternate_uri, alternate_generation
    parsed = urlsplit(uri or "")
    require(parsed.scheme == "gs" and parsed.netloc and parsed.path.startswith("/")
            and len(parsed.path) > 1 and not parsed.query and not parsed.fragment, "INVALID_SOURCE_URI")
    require(re.fullmatch(r"[1-9][0-9]{0,24}", str(generation or "")) and is_digest(value.get("sha256")),
            "INVALID_SOURCE_GENERATION_OR_HASH")
    size = value.get("bytes")
    require(type(size) is int and 0 < size <= 16 * 1024**2, "INVALID_ARCHIVE_SIZE")
    return dict(durable_uri=uri, generation=str(generation), sha256=value["sha256"], bytes=size)


def merge_cycle_reference(keys, proofs, cycle, key, proof):
    require(cycle not in keys or keys[cycle] == key, "COMPETING_CACHE_KEYS_FOR_CYCLE")
    require(cycle not in proofs or proofs[cycle] == proof, "COMPETING_REUSE_PROOFS_FOR_CYCLE")
    keys[cycle], proofs[cycle] = key, proof


def collect_references(corpus, plan_sha, admission):
    plan, descriptors, info = load_plan(corpus, plan_sha)
    progress = scan_progress(corpus, plan, descriptors)
    require(not progress["issues"] and not progress["missing_chunks"] and not progress["invalid_chunks"],
            "COMPLETE_COMMITTED_METADATA_REQUIRED")
    observed, keys, proofs = [], {}, {}
    for unit_id, index, unit, chunk in descriptors:
        directory = Path(corpus) / "chunks" / unit_id / f"{index:03d}"
        checked = read_chunk(directory, plan, unit, chunk)
        summary, summary_raw, _ = read_json(directory / "case-index.json", 256 * 1024)
        commit, commit_raw, _ = read_json(directory / "COMMIT.json", 65536)
        require(sha(commit_raw) == checked["commit_sha256"]
                and sha(summary_raw) == commit["payloads"]["case-index.json"]["sha256"], "CHUNK_CHANGED_DURING_INVENTORY")
        reuse = summary.get("qualified_cache_reuse", {})
        require(set(reuse) <= set(summary["cycle_keys"]), "UNREFERENCED_REUSE_PROOF")
        for cycle, key in summary["cycle_keys"].items():
            merge_cycle_reference(keys, proofs, cycle, key, reuse.get(cycle))
        observed.append(dict(chunk=f"{unit_id}/{index:03d}", commit_sha256=checked["commit_sha256"]))
    require(set(keys) == set(plan["required_cycle_ids"]) and len(keys) == len(plan["required_cycle_ids"])
            and not (set(keys) & set(plan["calendar_unavailable_cycle_ids"])), "REQUIRED_CYCLE_SET_DIFFERS")
    snapshot = dict(plan_sha256=plan_sha, plan_file_sha256=info["file_sha256"], observed=observed, missing=[])
    prior = admission["snapshot"]
    require(prior["status"] == "COMPLETE_COMMITTED_CORPUS_SNAPSHOT" and prior["missing_chunks"] == []
            and prior["plan_sha256"] == plan_sha and prior["plan_file_sha256"] == info["file_sha256"]
            and prior["observed_commits"] == observed and prior["snapshot_sha256"] == digest(snapshot),
            "ADMISSION_SNAPSHOT_DIFFERS")
    return plan, info, keys, proofs, digest(snapshot), observed


def inspect_cache_entry(cache_root, cycle, key, row, plan, proof):
    require(is_digest(key), "INVALID_CACHE_KEY")
    directory = Path(cache_root) / "entries" / key
    resident_directory(directory)
    commit, raw, _ = read_json(directory / "COMMIT.json", 128 * 1024)
    identity = commit["identity"]
    require(commit.get("schema") == "hydra-derived-cache/v1" and commit.get("kind") == "basin_cycle"
            and commit.get("key") == key == digest(dict(schema=commit["schema"], kind="basin_cycle", identity=identity)),
            "CACHE_KEY_OR_SCHEMA_DIFFERS")
    code = identity["implementation_sha256"]
    require(code in (PRODUCER_CODE, CONSUMER_CODE), "UNQUALIFIED_CACHE_PRODUCER")
    require(identity == dict(cycle_id=cycle, qualified_index_sha256=plan["qualified_index_sha256"],
            sealed_row_sha256=digest(row), candidate_weights_sha256=plan["candidate_weights_sha256"],
            implementation_sha256=code, policy=POLICY), "CACHE_SOURCE_POLICY_IDENTITY_DIFFERS")
    require(all(plan["implementation_sha256"].get(name) == value for name, value in CONSUMER_CODE.items()),
            "CORPUS_CONSUMER_CODE_DIFFERS")
    payloads = commit["payloads"]
    require(set(payloads) == {"arrays", "metadata"}, "CACHE_PAYLOAD_INVENTORY_DIFFERS")
    for label, suffix, maximum in (("arrays", ".npz", 16 * 1024**2), ("metadata", ".json", 4 * 1024**2)):
        item = payloads[label]
        require(is_digest(item.get("sha256")) and item.get("name") == item["sha256"] + suffix
                and type(item.get("bytes")) is int and 0 < item["bytes"] <= maximum, "INVALID_CACHE_PAYLOAD_IDENTITY")
    require({p.name for p in directory.iterdir()} == {"COMMIT.json", *(v["name"] for v in payloads.values())},
            "CACHE_DIRECTORY_INVENTORY_DIFFERS")
    # Stat the numerical payload only. Its previous decoding is evidenced by
    # the completed corpus/admission receipts, not repeated by this inventory.
    arrays_stat = (directory / payloads["arrays"]["name"]).lstat()
    require(stat.S_ISREG(arrays_stat.st_mode) and arrays_stat.st_size == payloads["arrays"]["bytes"],
            "CACHE_ARRAY_STAT_DIFFERS")
    metadata, body, _ = read_json(directory / payloads["metadata"]["name"], 4 * 1024**2)
    require(len(body) == payloads["metadata"]["bytes"] and sha(body) == payloads["metadata"]["sha256"],
            "CACHE_METADATA_HASH_DIFFERS")
    receipt = metadata["source_receipt"]
    expected_reference = int(datetime.strptime(cycle, "%Y%m%d_%HZ").replace(tzinfo=timezone.utc).timestamp())
    require(metadata["reference"] == expected_reference and receipt["cycle_id"] == cycle
            and expected_reference < int(datetime(2024, 7, 31, tzinfo=timezone.utc).timestamp()), "CACHE_SOURCE_CLOCK_DIFFERS")
    lineage, normalized = receipt["lineage"], receipt["lineage"]["normalized_summary"]
    require(lineage["cycle_id"] == cycle and lineage["index_location_kind"] == row["location_kind"]
            and lineage["source_summaries_rewritten"] is False and row["original_quality_flags_rewritten"] is False,
            "LINEAGE_SOURCE_IDENTITY_DIFFERS")
    for name in ("archive_sha256", "archive_bytes", "manifest_sha256"):
        require(receipt[name] == lineage[name], "RECEIPT_LINEAGE_IDENTITY_DIFFERS")
    require(type(receipt["data_complete"]) is bool and receipt["data_complete"] is normalized["data_complete"]
            and receipt["data_complete"] is row["data_complete"] and receipt["record_count"] == normalized["record_count"]
            and receipt["groups_available"] == normalized["groups_available"], "RECEIPT_COMPLETENESS_DIFFERS")
    for name in ("record_count", "groups_available", "groups_missing", "coverage_status"):
        if name in row:
            supplied = sorted(row[name]) if name == "groups_available" else row[name]
            require(supplied == normalized[name], "SEALED_SUMMARY_DIFFERS")
    require(is_digest(receipt["archive_sha256"]) and is_digest(receipt["manifest_sha256"])
            and type(receipt["archive_bytes"]) is int and 0 < receipt["archive_bytes"] <= 16 * 1024**2,
            "INVALID_RECEIPT_ARCHIVE_IDENTITY")
    if code == PRODUCER_CODE:
        consumer_identity = {**identity, "implementation_sha256": CONSUMER_CODE}
        expected = dict(schema="hydra-qualified-basin-cache-reuse/v1", rule_id=RULE_ID,
            rule_sha256=plan["implementation_sha256"]["cache_reuse.py"], original_cache_key=key,
            original_commit_sha256=sha(raw), producer_identity_sha256=digest(identity),
            consumer_identity_sha256=digest(consumer_identity), producer_implementation_sha256=PRODUCER_CODE,
            consumer_implementation_sha256=CONSUMER_CODE, original_source_receipt_sha256=digest(receipt),
            payload_and_schema_reverified=True, original_entry_or_receipt_modified=False,
            raw_source_redecoded=False, provider_calls=0)
        require(proof == expected, "ORIGINAL_PRODUCER_REUSE_PROOF_DIFFERS")
    else:
        require(proof is None, "UNEXPECTED_CURRENT_PRODUCER_REUSE_PROOF")
    return receipt, dict(cache_key=key, cache_commit_sha256=sha(raw), metadata_sha256=sha(body),
                         source_receipt_sha256=digest(receipt), producer_implementation_sha256=code,
                         numerical_payload_read=False)


def resolve_source(cycle, row, planned, receipt, commit_root):
    require(planned["cycle_id"] == cycle and planned["sealed_row_sha256"] == digest(row), "STAGING_ROW_DIFFERS")
    spec = dict(cycle_id=cycle)
    if row["location_kind"] == "commit":
        sealed = row["source_locations"]["commit"]
        candidates = [Path(commit_root) / (cycle + ".json")]
        candidates += [Path(item["path"] if isinstance(item, dict) else item) for item in planned["local_commit_candidates"]]
        for path in candidates:
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or getattr(info, "st_flags", 0) & 0x40000000 or not 0 < info.st_size <= 16384:
                    continue
                require(path.suffix == ".json", "ORIGINAL_COMMIT_JSON_REQUIRED")
                _, raw, _ = read_json(path, 16384)
                if sha(raw) == sealed["sha256"]:
                    spec["commit_path"] = str(path); break
            except FileNotFoundError:
                continue
        require("commit_path" in spec, "RESIDENT_AUTHENTICATED_ORIGINAL_COMMIT_REQUIRED")
    bundle, worker, authenticated = authenticate_bundle_identity(spec, row)
    archive = object_identity(bundle)
    require(archive["sha256"] == receipt["archive_sha256"] and archive["bytes"] == receipt["archive_bytes"],
            "AUTHENTICATED_ARCHIVE_DIFFERS_FROM_CACHE_RECEIPT")
    require(authenticated["binding_route"] == receipt["lineage"]["binding_route"]
            and authenticated["commit"] == receipt["lineage"]["commit"], "ORIGINAL_COMMIT_LINEAGE_DIFFERS")
    require(receipt.get("commit_sha256") == (authenticated["commit"]["sha256"] if authenticated["commit"] else None),
            "RECEIPT_COMMIT_HASH_DIFFERS")
    if worker is not None:
        require(worker["manifest_sha256"] == receipt["manifest_sha256"] and worker["record_count"] == receipt["record_count"],
                "ORIGINAL_WORKER_MANIFEST_DIFFERS")
    commit = None
    if authenticated["commit"] is not None:
        commit = object_identity({**row["source_locations"]["commit"], "bytes": authenticated["commit"]["bytes"]})
    return dict(archive=archive, source_commit=commit, original_commit_path=spec.get("commit_path"),
                binding_route=authenticated["binding_route"])


def summarize(catalog):
    unique, objects = {}, {}
    for row in catalog:
        archive = row["archive"]
        require(archive["sha256"] not in unique or unique[archive["sha256"]] == archive["bytes"], "ARCHIVE_SHA_SIZE_CONFLICT")
        key = (archive["durable_uri"], archive["generation"])
        require(key not in objects or objects[key] == (archive["sha256"], archive["bytes"]), "ARCHIVE_OBJECT_CONFLICT")
        unique[archive["sha256"]] = archive["bytes"]
        objects[key] = (archive["sha256"], archive["bytes"])
    return dict(required_cycles=len(catalog), resolved_archive_identities=len(catalog), unresolved_archive_identities=0,
                unique_archive_sha256_count=len(unique), unique_archive_bytes=sum(unique.values()),
                unique_archive_object_generations=len(objects), unique_object_archive_bytes=sum(v[1] for v in objects.values()),
                per_cycle_archive_bytes=sum(row["archive"]["bytes"] for row in catalog),
                complete_cycle_count=sum(row["data_complete"] for row in catalog),
                confirmed_gap_cycle_count=sum(not row["data_complete"] for row in catalog),
                producer_local_data_sha256_counts=dict(Counter(row["cache"]["producer_implementation_sha256"]["local_data.py"] for row in catalog)),
                producer_identity_counts=dict(Counter(digest(row["cache"]["producer_implementation_sha256"]) for row in catalog)),
                binding_route_counts=dict(Counter(row["binding_route"] for row in catalog)),
                archive_payload_schema_counts=dict(Counter(row["lineage"]["payload_schema"] for row in catalog)))


def write_new(value, path):
    path = Path(path)
    require(path.suffix == ".json" and not path.exists() and not path.is_symlink(), "NEW_JSON_OUTPUT_REQUIRED")
    resident_directory(path.parent)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= 64 * 1024**2, "CATALOG_OUTPUT_BOUND_EXCEEDED")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    finally:
        temporary.unlink()
    return dict(sha256=sha(raw), bytes=len(raw))


def validate_private_output(path):
    path, local = Path(path), ROOT / ".local"
    resident_directory(local)
    require(not path.is_symlink() and path.resolve().is_relative_to(local.resolve()),
            "PRIVATE_CATALOG_MUST_STAY_LOCAL")


def build_inventory(*, corpus, cache, qualified_index, commit_root, staging_plan, staging_plan_sha256,
                    admission_path, admission_sha256, plan_sha256):
    started = time.monotonic()
    code_paths = (Path(__file__).resolve(), ROOT / "scripts/inspect_staging_progress.py",
                  *(ROOT / "src/hydra_directflow" / name for name in ("lineage.py", "multicycle.py", "cache_reuse.py")))
    code_hashes = {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in code_paths}
    admission = pinned_json(admission_path, admission_sha256, 4 * 1024**2)
    require(admission.get("schema") == "hydra-corpus-admission-audit/v1"
            and admission.get("audit_content_sha256") == digest({k: v for k, v in admission.items() if k not in ("audit_content_sha256", "runtime")}),
            "ADMISSION_AUTHORITY_DIFFERS")
    source_plan = pinned_json(staging_plan, staging_plan_sha256, 16 * 1024**2)
    require(source_plan.get("plan_sha256") == digest({k: v for k, v in source_plan.items() if k not in ("plan_sha256", "created_at_utc")}), "STAGING_PLAN_DIGEST_DIFFERS")
    plan, info, keys, proofs, snapshot_sha, commits = collect_references(corpus, plan_sha256, admission)
    require(all(code_hashes["src/hydra_directflow/" + name] == plan["implementation_sha256"][name]
                for name in ("lineage.py", "multicycle.py", "cache_reuse.py")), "INVENTORY_SOURCE_HELPER_CODE_DIFFERS")
    require(len(keys) == MAX_CYCLES and plan["qualified_index_sha256"] == INDEX_SHA256, "FULL_DEVELOPMENT_CALENDAR_REQUIRED")
    rows = qualified_cycle_rows(qualified_index, sorted(keys))
    planned = {item["cycle_id"]: item for item in source_plan["cycles"]}
    require(len(planned) == len(source_plan["cycles"]) == MAX_CYCLES and set(planned) == set(keys), "STAGING_CYCLE_SET_DIFFERS")
    resident_directory(cache); resident_directory(Path(cache) / "entries")
    catalog = []
    for cycle in sorted(keys):
        receipt, cache_identity = inspect_cache_entry(cache, cycle, keys[cycle], rows[cycle], plan, proofs[cycle])
        source = resolve_source(cycle, rows[cycle], planned[cycle], receipt, commit_root)
        catalog.append(dict(cycle_id=cycle, sealed_row_sha256=digest(rows[cycle]), cache=cache_identity,
                            **source, manifest_sha256=receipt["manifest_sha256"], data_complete=receipt["data_complete"],
                            record_count=receipt["record_count"], groups_available=receipt["groups_available"], lineage=receipt["lineage"]))
    require(sha(read_json(Path(corpus) / "PLAN.json", 4 * 1024**2)[1]) == info["file_sha256"], "CORPUS_PLAN_CHANGED")
    require(code_hashes == {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in code_paths}, "INVENTORY_CODE_CHANGED")
    identities = dict(corpus_plan_sha256=plan_sha256, corpus_plan_file_sha256=info["file_sha256"], corpus_snapshot_sha256=snapshot_sha,
                      admission_report_sha256=admission_sha256, admission_content_sha256=admission["audit_content_sha256"],
                      staging_plan_file_sha256=staging_plan_sha256, staging_plan_sha256=source_plan["plan_sha256"],
                      qualified_index_sha256=INDEX_SHA256, qualified_index_bytes=INDEX_BYTES,
                      candidate_weights_sha256=plan["candidate_weights_sha256"], usgs_manifest_sha256=plan["usgs_manifest_sha256"],
                      input_quality_policy=plan["input_quality_policy"], target_quality_policy=plan["target_quality_policy"],
                      corpus_implementation_sha256=plan["implementation_sha256"],
                      inventory_implementation_sha256=code_hashes)
    private = dict(schema=SCHEMA, status="COMPLETE_METADATA_SOURCE_CATALOG", created_at_utc=datetime.now(timezone.utc).isoformat(),
                   identities=identities, source_cycles=catalog, corpus_observed_commits=commits,
                   excluded_outside_calendar_cycle_ids=plan["calendar_unavailable_cycle_ids"])
    public = dict(schema=SCHEMA, status="COMPLETE_METADATA_SOURCE_INVENTORY", created_at_utc=private["created_at_utc"],
                  identities=identities, corpus_shards=len(commits), candidate_origins=plan["candidate_origins"],
                  candidate_cases=plan["candidate_origins"] * 3, **summarize(catalog),
                  excluded_outside_calendar_cycle_ids=plan["calendar_unavailable_cycle_ids"],
                  execution=dict(metadata_only=True, raw_archive_bytes_read=0, numerical_array_bytes_read=0,
                                 network_requests=0, cloud_jobs=0, optimizer_updates=0, protected_assessment_access=False,
                                 original_metadata_modified=False, elapsed_seconds=time.monotonic() - started),
                  qualification="Exact retained development archive identities from committed lineage, not additional acquisition, current cloud existence, billing reconciliation or scientific admission; numerical payloads were not reverified in this metadata inventory.")
    return private, public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("corpus", "cache", "qualified-index", "commit-root", "staging-plan", "admission-path", "private-catalog", "report"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("staging-plan-sha256", "admission-sha256", "plan-sha256"):
        parser.add_argument("--" + flag, required=True)
    args = vars(parser.parse_args())
    private_path, report_path = args.pop("private_catalog"), args.pop("report")
    validate_private_output(private_path)
    require(not private_path.exists() and not report_path.exists(), "OUTPUT_ALREADY_EXISTS")
    def deadline(*unused): raise TimeoutError("Metadata inventory exceeded 900 seconds")
    previous = signal.signal(signal.SIGALRM, deadline); signal.alarm(900)
    try:
        private, public = build_inventory(**args)
        public["private_catalog"] = write_new(private, private_path)
        public["inventory_content_sha256"] = digest(public)
        write_new(public, report_path)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
    print(json.dumps({key: public[key] for key in ("status", "required_cycles", "unique_archive_bytes", "complete_cycle_count", "confirmed_gap_cycle_count")}))


if __name__ == "__main__":
    main()
