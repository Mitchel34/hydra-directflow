"""One exact, reviewed producer-to-consumer compatibility rule.

The only reader change admits additional authenticated legacy compact404
metadata. Previously accepted source reductions are unchanged. Original cache
entries, keys and source receipts stay intact; per-use provenance is separate.
This is not a generic migration, a raw-source re-audit or a permission grant.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re

from .cache import _read, _directory, unpack_cycle
from .local_data import DEVELOPMENT_QUALIFICATION, WEIGHTS_SHA256
from .multicycle import INDEX_SHA256
from .reference import ASSESSMENT_START, digest, require

RULE_ID = "V3_BASIN_TO_LEGACY_COMPACT404_CONSUMER_V1"
PRODUCER_CODE = {
    "cache.py": "780cedd3d23a3c69d6468979f5f4adfd316dbdea19714118303aa2b38d2308f5",
    "contract.py": "e3f1a5e988e02a11da19cc2aec80cfd860e4867d68c14d12bdaba706a85513e8",
    "development_export.py": "472149f464d5634298ca9ad6d42b17f24270d9d56634226b7b4ab2c58efb0ab5",
    "lineage.py": "d0c8a5e0f68c1b0315336f00e93fd1a7469d192d7e78a263eefc481852979a66",
    "local_data.py": "ed999095fde7e0825426ab499b8a876a25bb04ed752b94d09cc0fc9634a47084",
    "multicycle.py": "6028dbd8374e28a5cfbaea0108ebbdf770e1da66473a111b78f29c5a5a5e1879",
    "quality.py": "d978db5fa454e040b92544bb9dbbabe332cb575c56d150a2e616e172c33fec31",
    "reference.py": "7c21deee40c04bc72bc5cfcb190a39792c94ab65115c16ca03da551296575135",
    "telemetry.py": "50e28d1fcbbde444c6199324155a4114cae77ac2461e4f6b331933a2748e2791",
}
CONSUMER_CODE = {**PRODUCER_CODE,
    "local_data.py": "adb2fe589fe92e1598e7e21f9955af9c9e53fa6eb5266e52dd10f2d12c0d5045"}
POLICY = {"availability": "UNKNOWN_HISTORICAL_PUBLICATION_SIMULATED_LAG_2H",
          "geometry": DEVELOPMENT_QUALIFICATION,
          "quality": "PRESERVE_ALL_NATIVE_FLAGS_ADMISSION_AT_CASE_ASSEMBLY", "units": "PHYSICAL_UNSCALED"}


def qualified_basin_reuse(cache, consumer_identity):
    """Return an original verified entry or None; invalid chosen entries fail.

    The caller must prefer this deterministic original key before current-version
    lookup. A committed corpus shard continues to enforce its recorded key on
    resume. No old entry is rewritten under a new producer identity.
    """
    require(cache._held, "Qualified reuse requires the existing cache writer lock")
    if consumer_identity.get("implementation_sha256") != CONSUMER_CODE:
        return None
    require(set(consumer_identity) == {"cycle_id", "qualified_index_sha256", "sealed_row_sha256",
            "candidate_weights_sha256", "implementation_sha256", "policy"}, "Unexpected basin derivation identity")
    require(consumer_identity["qualified_index_sha256"] == INDEX_SHA256
            and consumer_identity["candidate_weights_sha256"] == WEIGHTS_SHA256
            and consumer_identity["policy"] == POLICY
            and isinstance(consumer_identity["sealed_row_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", consumer_identity["sealed_row_sha256"]),
            "Reuse cannot change source authority, weights or policy")
    cycle_id = consumer_identity["cycle_id"]
    require(isinstance(cycle_id, str) and re.fullmatch(r"\d{8}_\d{2}Z", cycle_id), "Exact cycle ID required")
    reference = int(datetime.strptime(cycle_id, "%Y%m%d_%HZ").replace(tzinfo=timezone.utc).timestamp())
    require(int(datetime(2018, 7, 12, tzinfo=timezone.utc).timestamp()) <= reference < ASSESSMENT_START
            and reference % (6 * 3600) == 0, "Reuse is restricted to acquired development cycles")
    producer_identity = deepcopy(consumer_identity)
    producer_identity["implementation_sha256"] = dict(PRODUCER_CODE)
    key = cache.key("basin_cycle", producer_identity)
    entry = cache.path / "entries" / key
    if not entry.exists() and not entry.is_symlink():
        return None
    _directory(entry)
    before = _read(entry / "COMMIT.json", 128 * 1024)
    arrays, metadata = cache._load(entry, "basin_cycle", producer_identity, key)
    unpack_cycle(arrays, metadata)
    require(metadata["reference"] == reference and metadata["source_receipt"]["cycle_id"] == cycle_id,
            "Original cache reference/source receipt differs from cycle identity")
    require(_read(entry / "COMMIT.json", 128 * 1024) == before, "Original cache commitment changed during reuse")
    proof = dict(schema="hydra-qualified-basin-cache-reuse/v1", rule_id=RULE_ID,
        rule_sha256=hashlib.sha256(_read(Path(__file__), 256 * 1024)).hexdigest(),
        original_cache_key=key, original_commit_sha256=hashlib.sha256(before).hexdigest(),
        producer_identity_sha256=digest(producer_identity), consumer_identity_sha256=digest(consumer_identity),
        producer_implementation_sha256=dict(PRODUCER_CODE), consumer_implementation_sha256=dict(CONSUMER_CODE),
        original_source_receipt_sha256=digest(metadata["source_receipt"]),
        payload_and_schema_reverified=True, original_entry_or_receipt_modified=False,
        raw_source_redecoded=False, provider_calls=0)
    cache.counts["hits"] += 1
    cache.counts["qualified_producer_hits"] += 1
    return arrays, metadata, key, proof
