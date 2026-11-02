"""Synthetic metadata-only fixtures; no archive, model, network or user inputs."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/inventory_development_sources.py"
SPEC = importlib.util.spec_from_file_location("source_inventory_tests", PATH)
INV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INV)


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def fixture(directory, *, old=False):
    cycle = "20210101_00Z"
    row = dict(cycle=cycle, data_complete=True, original_quality_flags_rewritten=False, location_kind="bundle",
               groups_available=["temperature", "humidity"],
               source_locations={"bundle": dict(durable_uri="gs://synthetic/cycle.tar", generation="123", sha256="a" * 64,
                                                 bytes=10240, verified=True)})
    code = INV.PRODUCER_CODE if old else INV.CONSUMER_CODE
    plan = dict(qualified_index_sha256=INV.INDEX_SHA256, candidate_weights_sha256="b" * 64,
                implementation_sha256={**INV.CONSUMER_CODE, "cache_reuse.py": "c" * 64})
    identity = dict(cycle_id=cycle, qualified_index_sha256=plan["qualified_index_sha256"],
                    candidate_weights_sha256=plan["candidate_weights_sha256"], sealed_row_sha256=INV.digest(row),
                    implementation_sha256=code, policy=INV.POLICY)
    key = INV.digest(dict(schema="hydra-derived-cache/v1", kind="basin_cycle", identity=identity))
    normalized = dict(data_complete=True, record_count=145, groups_available=["humidity", "temperature"], groups_missing={}, coverage_status="ALL_REQUESTED_GROUPS_AVAILABLE")
    lineage = dict(cycle_id=cycle, index_location_kind="bundle", source_summaries_rewritten=False, archive_sha256="a"*64,
                   archive_bytes=10240, manifest_sha256="d"*64, normalized_summary=normalized, commit=None,
                   binding_route="SEALED_INDEX_BUNDLE", payload_schema="HRRR-raw-cycle-acquisition/v1")
    receipt = dict(cycle_id=cycle, lineage=lineage, archive_sha256="a"*64, archive_bytes=10240, manifest_sha256="d"*64,
                   data_complete=True, record_count=145, groups_available=["humidity", "temperature"])
    metadata = dict(reference=int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()), source_receipt=receipt)
    payloads = {"metadata": raw(metadata), "arrays": b"opaque synthetic arrays never opened"}
    target = Path(directory) / "entries" / key; target.mkdir(parents=True)
    encodings = {}
    for kind, content in payloads.items():
        name = INV.sha(content) + (".json" if kind == "metadata" else ".npz")
        (target / name).write_bytes(content)
        encodings[kind] = dict(name=name, sha256=INV.sha(content), bytes=len(content))
    commit = dict(schema="hydra-derived-cache/v1", kind="basin_cycle", key=key, identity=identity, payloads=encodings)
    body = raw(commit); (target / "COMMIT.json").write_bytes(body)
    proof = None
    if old:
        proof = dict(schema="hydra-qualified-basin-cache-reuse/v1", rule_id=INV.RULE_ID, rule_sha256="c"*64,
                     original_cache_key=key, original_commit_sha256=INV.sha(body), producer_identity_sha256=INV.digest(identity),
                     consumer_identity_sha256=INV.digest({**identity,"implementation_sha256":INV.CONSUMER_CODE}),
                     producer_implementation_sha256=INV.PRODUCER_CODE, consumer_implementation_sha256=INV.CONSUMER_CODE,
                     original_source_receipt_sha256=INV.digest(receipt), payload_and_schema_reverified=True,
                     original_entry_or_receipt_modified=False, raw_source_redecoded=False, provider_calls=0)
    return cycle, key, row, plan, proof, target, receipt


class InventoryTests(unittest.TestCase):
    def test_complete_shard_snapshot_bound_and_missing_chunk_refused(self):
        helper_spec=importlib.util.spec_from_file_location('progress_inventory_fixture',Path(__file__).with_name('test_inspect_staging_progress.py'))
        helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as directory:
            root,plan=helper.fixture(directory,completed=(0,1))
            observed=[]
            for index in (0,1):
                child=root/'chunks'/plan['units'][0]['id']/f'{index:03d}'
                observed.append(dict(chunk=f"{plan['units'][0]['id']}/{index:03d}",commit_sha256=INV.sha((child/'COMMIT.json').read_bytes())))
            identity=dict(plan_sha256=plan['plan_sha256'],plan_file_sha256=INV.sha((root/'PLAN.json').read_bytes()),observed=observed,missing=[])
            admission={'snapshot':dict(status='COMPLETE_COMMITTED_CORPUS_SNAPSHOT',missing_chunks=[],observed_commits=observed,
                plan_sha256=identity['plan_sha256'],plan_file_sha256=identity['plan_file_sha256'],snapshot_sha256=INV.digest(identity))}
            got=INV.collect_references(root,plan['plan_sha256'],admission)
            self.assertEqual(set(got[2]),set(plan['required_cycle_ids']))
            with self.assertRaisesRegex(ValueError,'SNAPSHOT_DIFFERS'):
                INV.collect_references(root,plan['plan_sha256'],{'snapshot':{**admission['snapshot'],'snapshot_sha256':'f'*64}})
            (root/'chunks'/plan['units'][0]['id']/'001'/'COMMIT.json').unlink()
            with self.assertRaisesRegex(ValueError,'COMPLETE_COMMITTED'):
                INV.collect_references(root,plan['plan_sha256'],admission)

    def test_current_cache_metadata_is_verified_without_arrays_read(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle,key,row,plan,proof,target,receipt = fixture(directory)
            real_open = INV.os.open
            def only_metadata(path, *args, **kwargs):
                if str(path).endswith(('.npz', '.tar')): raise AssertionError("Numerical payload opened")
                return real_open(path,*args,**kwargs)
            with patch.object(INV.os, 'open', side_effect=only_metadata):
                result, identity = INV.inspect_cache_entry(directory,cycle,key,row,plan,proof)
            self.assertEqual(result,receipt)
            self.assertFalse(identity['numerical_payload_read'])

    def test_old_cache_requires_exact_original_commit_and_receipt_reuse_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle,key,row,plan,proof,target,receipt = fixture(directory,old=True)
            INV.inspect_cache_entry(directory,cycle,key,row,plan,proof)
            for field in ('original_commit_sha256','original_source_receipt_sha256','consumer_identity_sha256'):
                bad = {**proof,field:'f'*64}
                with self.assertRaisesRegex(ValueError,'REUSE_PROOF'):
                    INV.inspect_cache_entry(directory,cycle,key,row,plan,bad)

    def test_source_row_or_weight_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle,key,row,plan,proof,target,receipt = fixture(directory)
            with self.assertRaisesRegex(ValueError,'SOURCE_POLICY'):
                INV.inspect_cache_entry(directory,cycle,key,{**row,'data_complete':False},plan,proof)
            with self.assertRaisesRegex(ValueError,'SOURCE_POLICY'):
                INV.inspect_cache_entry(directory,cycle,key,row,{**plan,'candidate_weights_sha256':'f'*64},proof)

    def test_modified_metadata_and_unsafe_payload_name_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle,key,row,plan,proof,target,receipt = fixture(directory)
            commit=json.loads((target/'COMMIT.json').read_bytes())
            (target/commit['payloads']['metadata']['name']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'METADATA_HASH'):
                INV.inspect_cache_entry(directory,cycle,key,row,plan,proof)
            commit['payloads']['metadata']['name']='../../source.parquet'
            (target/'COMMIT.json').write_bytes(raw(commit))
            with self.assertRaisesRegex(ValueError,'PAYLOAD_IDENTITY'):
                INV.inspect_cache_entry(directory,cycle,key,row,plan,proof)

    def test_competing_cycle_keys_or_reuse_proofs_rejected(self):
        keys, proofs = {},{}
        INV.merge_cycle_reference(keys,proofs,'20210101_00Z','a'*64,None)
        INV.merge_cycle_reference(keys,proofs,'20210101_00Z','a'*64,None)
        with self.assertRaisesRegex(ValueError,'COMPETING_CACHE_KEYS'):
            INV.merge_cycle_reference(keys,proofs,'20210101_00Z','b'*64,None)
        with self.assertRaisesRegex(ValueError,'COMPETING_REUSE_PROOFS'):
            INV.merge_cycle_reference(keys,proofs,'20210101_00Z','a'*64,{})

    def test_sealed_bundle_route_needs_no_original_source_open(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle,key,row,plan,proof,target,receipt = fixture(directory)
            planned=dict(cycle_id=cycle,sealed_row_sha256=INV.digest(row))
            result=INV.resolve_source(cycle,row,planned,receipt,Path(directory)/'absent')
            self.assertEqual(result['archive']['bytes'],10240)
            self.assertIsNone(result['original_commit_path'])
            with self.assertRaisesRegex(ValueError,'ARCHIVE_DIFFERS'):
                INV.resolve_source(cycle,row,planned,{**receipt,'archive_bytes':20480},Path(directory)/'absent')

    def test_commit_route_authenticates_original_json_and_exact_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle,key,row,plan,proof,target,receipt = fixture(directory)
            bundle=row['source_locations']['bundle']
            commit=dict(schema='hrrr-verified-cloud-cycle/v1',cycle=cycle,bundle=bundle,
                        worker_manifest=dict(cycle=cycle,bundle_bytes=10240,bundle_sha256='a'*64,manifest_sha256='d'*64,record_count=145))
            body=raw(commit); path=Path(directory)/(cycle+'.json');path.write_bytes(body)
            sealed=dict(durable_uri='gs://synthetic/commit.json',generation='456',sha256=INV.sha(body),bytes=len(body))
            row={**row,'location_kind':'commit','source_locations':{'commit':sealed}}
            receipt=copy.deepcopy(receipt)
            receipt['lineage'].update(index_location_kind='commit',binding_route='SEALED_INDEX_COMMIT_BUNDLE',
                commit=dict(sha256=INV.sha(body),bytes=len(body),byte_count_source='SEALED_INDEX'))
            receipt['commit_sha256']=INV.sha(body)
            planned=dict(cycle_id=cycle,sealed_row_sha256=INV.digest(row),local_commit_candidates=[])
            result=INV.resolve_source(cycle,row,planned,receipt,directory)
            self.assertEqual(result['source_commit']['sha256'],INV.sha(body))
            path.write_bytes(body+b' ')
            with self.assertRaisesRegex(ValueError,'RESIDENT_AUTHENTICATED'):
                INV.resolve_source(cycle,row,planned,receipt,directory)

    def test_unique_bytes_do_not_double_count_same_archive_and_conflicts_fail(self):
        archive=dict(durable_uri='gs://synthetic/a',generation='1',sha256='a'*64,bytes=123)
        entry=dict(archive=archive,data_complete=True,cache={'producer_implementation_sha256':INV.CONSUMER_CODE},
                   binding_route='SEALED_INDEX_BUNDLE',lineage={'payload_schema':'synthetic'})
        report=INV.summarize([entry,entry])
        self.assertEqual((report['unique_archive_bytes'],report['per_cycle_archive_bytes']),(123,246))
        with self.assertRaisesRegex(ValueError,'SHA_SIZE_CONFLICT'):
            INV.summarize([entry,{**entry,'archive':{**archive,'bytes':124}}])

    def test_object_generation_required_and_conflicts_rejected(self):
        value=dict(durable_uri='gs://synthetic/a',generation='123',sha256='a'*64,bytes=100)
        self.assertEqual(INV.object_identity(value)['generation'],'123')
        with self.assertRaisesRegex(ValueError,'GENERATION'):
            INV.object_identity({**value,'generation':None})
        with self.assertRaisesRegex(ValueError,'LOCATION_CONFLICT'):
            INV.object_identity({**value,'uri_generation':'gs://synthetic/b#123'})

    def test_output_is_immutable_and_portable_summary_has_no_source_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.json'; identity=INV.write_new({'metadata_only':True},path)
            self.assertEqual(identity['sha256'],INV.sha(path.read_bytes()))
            with self.assertRaisesRegex(ValueError,'NEW_JSON'):
                INV.write_new({},path)

    def test_private_catalog_rejects_dotdot_and_symlink_escapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); local=root/'.local'; local.mkdir(); public=root/'reports'; public.mkdir()
            with patch.object(INV,'ROOT',root):
                INV.validate_private_output(local/'catalog.json')
                with self.assertRaisesRegex(ValueError,'MUST_STAY_LOCAL'):
                    INV.validate_private_output(local/'..'/'reports'/'catalog.json')
                (local/'escape').symlink_to(public,target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'MUST_STAY_LOCAL'):
                    INV.validate_private_output(local/'escape'/'catalog.json')


if __name__ == '__main__': unittest.main()
