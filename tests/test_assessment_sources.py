"""Synthetic assessment transport tests. No credentials, network or study reads."""
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra_directflow import assessment_sources as a
from hydra_directflow.reference import digest
from hydra_directflow.staging import SourceBudget
from hydra_directflow.lineage import authenticate_bundle_identity


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True)+'\n').encode()


class Response(io.BytesIO):
    status=200
    def __init__(self,raw,generation='123'):
        super().__init__(raw)
        self.headers={'x-goog-generation':generation,'Content-Length':str(len(raw))}


class FakeClient:
    def __init__(self,objects):self.objects=objects;self.calls=[];self.failures=[];self.after_open=None
    def open(self,identity):
        self.calls.append(copy.deepcopy(identity))
        if self.failures:raise self.failures.pop(0)
        if self.after_open:self.after_open()
        return Response(self.objects[identity['sha256']],identity['generation'])


class AssessmentSourcesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.root_patch=patch.object(a,'ROOT',self.root);self.root_patch.start()
        self.free=100*a.GIB
        self.disk_patch=patch('hydra_directflow.assessment_sources.shutil.disk_usage',
                              side_effect=lambda p:shutil._ntuple_diskusage(200*a.GIB,100*a.GIB,self.free))
        self.disk_patch.start();self.budgets=[]
        self.stage=self.root/'.local/assessment-v1/source-staging'

    def tearDown(self):
        for budget in self.budgets:budget.close()
        self.disk_patch.stop();self.root_patch.stop();self.tmp.cleanup()

    def fixture(self,*,kind='bundle',unknown_bundle=False,local_archive=False):
        raw=b'FAKE_OPAQUE_ARCHIVE_NO_WEATHER_VALUES';objects={sha(raw):raw}
        rows=[];planned=[];source=[];archive_candidates=[]
        for cycle in a.assessment_cycle_ids():
            bundle=dict(durable_uri='gs://test-bucket/assessment/cycles/'+cycle+'.tar',
                        generation='123',sha256=sha(raw),bytes=len(raw),verified=True)
            locations={'bundle':bundle};commit_identity=None
            if kind=='commit':
                body=encoded(dict(schema='hrrr-verified-cloud-cycle/v1',cycle=cycle,bundle=bundle,
                                  worker_manifest=dict(cycle=cycle,bundle_bytes=len(raw),bundle_sha256=sha(raw),
                                                       manifest_sha256='c'*64,record_count=145)))
                commit_identity=dict(durable_uri='gs://test-bucket/assessment/commits/'+cycle+'.json',generation='123',sha256=sha(body))
                objects[sha(body)]=body;locations['commit']=commit_identity
                if unknown_bundle:locations.pop('bundle')
            row=dict(cycle=cycle,location_kind=kind,data_complete=True,original_quality_flags_rewritten=False,
                     source_locations=locations);rows.append(row)
            known=None if unknown_bundle else bundle
            source.append(dict(cycle_id=cycle,sealed_row_digest=digest(row),commit_identity=commit_identity,
                               known_bundle_identity=known,bundle_proof=None,commit_candidates=[],cloud_only_commit_metadata_paths=[]))
            planned.append(dict(cycle_id=cycle,sealed_row_sha256=digest(row),commit_identity=commit_identity,
                                bundle_identity=known,local_commit_candidates=[],local_archive_candidates=[]))
        if local_archive:
            path=self.root/'retained-owned.tar';path.write_bytes(raw)
            archive_candidates.append(dict(cycle_id=rows[3]['cycle'],path=str(path)))
            planned[3]['local_archive_candidates']=[{'path':str(path)}]
        inventory=dict(schema=a.INVENTORY_SCHEMA,hrrr=dict(cycles=source,archive_candidates=archive_candidates,
                       qualified_index=dict(sha256=a.INDEX_SHA256,bytes=a.INDEX_BYTES,rows=61824)))
        inventory['inventory_content_sha256']=digest(inventory)
        path=self.root/'.local/assessment-preparation-v1/SOURCE_INVENTORY.json';path.parent.mkdir(parents=True,exist_ok=True)
        body=encoded(inventory);path.write_bytes(body)
        plan=dict(schema=a.SCHEMA,status='APPROVED_BOUNDED_STAGING',limits=dict(a.LIMITS),
                  approved_staging_ceiling_usd=5,combined_ceiling_usd=150,
                  sources=dict(budget_approval={'sha256':'a'*64},source_inventory=dict(path=str(path),bytes=len(body),sha256=sha(body)),
                               qualified_index=dict(sha256=a.INDEX_SHA256,bytes=a.INDEX_BYTES)),cycles=planned)
        plan['plan_sha256']=digest(plan)
        return plan,{r['cycle']:r for r in rows},FakeClient(objects)

    def budget(self,**kwargs):
        b=SourceBudget(kwargs.pop('path',self.stage/'budget'),kwargs.pop('approval','a'*64),
                       body_cap=kwargs.pop('body_cap',16*a.GIB),get_cap=kwargs.pop('get_cap',4000),usd_cap=kwargs.pop('usd_cap',5))
        self.budgets.append(b);return b

    def provider(self,plan,client,**kwargs):
        budget=kwargs.pop('budget',None) or self.budget()
        return a.AssessmentSourceProvider(plan=plan,directory=kwargs.pop('directory',self.stage),budget=budget,
                    expected_plan_sha256=kwargs.pop('expected',plan['plan_sha256']),client=client,**kwargs)

    def rehash(self,plan):plan['plan_sha256']=digest({k:v for k,v in plan.items() if k!='plan_sha256'})

    def test_calendar_includes_all_assessment_tail_cycles(self):
        cycles=a.assessment_cycle_ids();self.assertEqual((len(cycles),cycles[0],cycles[-1]),(1463,'20240730_06Z','20250730_18Z'))

    def test_unapproved_status_and_wrong_hash_rejected_before_client_creation(self):
        p,_,_=self.fixture();budget=self.budget()
        with patch.object(a,'GCSClient') as client:
            bad=copy.deepcopy(p);bad['status']='PLAN_ONLY_NOT_EXECUTED';self.rehash(bad)
            with self.assertRaisesRegex(ValueError,'Approved assessment'):
                self.provider(bad,None,budget=budget)
            with self.assertRaisesRegex(ValueError,'digest'):
                self.provider(p,None,budget=budget,expected='f'*64)
            client.assert_not_called()

    def test_declared_caps_cannot_change(self):
        p,_,c=self.fixture();p['limits']['download_body_bytes']+=1;self.rehash(p)
        with self.assertRaisesRegex(ValueError,'caps'):self.provider(p,c)
        self.assertFalse(c.calls)

    def test_actual_budget_caps_and_approval_must_match(self):
        p,_,c=self.fixture();b=self.budget(body_cap=15*a.GIB)
        with self.assertRaisesRegex(ValueError,'budget identity or caps'):self.provider(p,c,budget=b)
        self.assertFalse(c.calls)

    def test_old_budget_namespace_rejected(self):
        p,_,c=self.fixture();b=self.budget(path=self.root/'old-budget')
        with self.assertRaisesRegex(ValueError,'namespace'):self.provider(p,c,budget=b)
        self.assertFalse(c.calls)

    def test_different_stage_directory_rejected(self):
        p,_,c=self.fixture()
        with self.assertRaisesRegex(ValueError,'fixed new assessment'):self.provider(p,c,directory=self.stage.parent/'stage-v2')
        self.assertFalse(c.calls)

    def test_source_inventory_file_change_rejected_before_client(self):
        p,_,_=self.fixture();path=Path(p['sources']['source_inventory']['path']);path.write_bytes(path.read_bytes()+b' ')
        with patch.object(a,'GCSClient') as client:
            with self.assertRaisesRegex(ValueError,'SHA256|byte count'):self.provider(p,None)
            client.assert_not_called()

    def test_source_inventory_self_digest_is_checked(self):
        p,_,c=self.fixture();path=Path(p['sources']['source_inventory']['path']);d=json.loads(path.read_text());d['extra']='changed'
        body=encoded(d);path.write_bytes(body);p['sources']['source_inventory'].update(bytes=len(body),sha256=sha(body));self.rehash(p)
        with self.assertRaisesRegex(ValueError,'content binding'):self.provider(p,c)
        self.assertFalse(c.calls)

    def test_complete_ordered_roster_required(self):
        p,_,c=self.fixture();p['cycles'].pop();self.rehash(p)
        with self.assertRaisesRegex(ValueError,'1463'):self.provider(p,c)
        self.assertFalse(c.calls)

    def test_plan_sealed_row_or_object_change_rejected(self):
        p,_,c=self.fixture();p['cycles'][3]['sealed_row_sha256']='f'*64;self.rehash(p)
        with self.assertRaisesRegex(ValueError,'sealed-row'):self.provider(p,c)
        self.assertFalse(c.calls)

    def test_generation_and_bucket_are_not_replaceable(self):
        p,_,c=self.fixture();p['cycles'][3]['bundle_identity']=dict(p['cycles'][3]['bundle_identity'],generation='456');self.rehash(p)
        with self.assertRaisesRegex(ValueError,'source object differs'):self.provider(p,c)
        p['cycles'][3]['bundle_identity']['durable_uri']='gs://other-bucket/cycles/'+p['cycles'][3]['cycle_id']+'.tar';self.rehash(p)
        with self.assertRaisesRegex(ValueError,'approved sealed buckets'):self.provider(p,c)
        self.assertFalse(c.calls)

    def test_uninventoried_local_path_is_rejected(self):
        p,_,c=self.fixture();p['cycles'][3]['local_archive_candidates']=[str(self.root/'unknown.tar')];self.rehash(p)
        with self.assertRaisesRegex(ValueError,'Uninventoried'):self.provider(p,c)

    def test_equivalent_generation_uri_alias_is_supported(self):
        p,rows,c=self.fixture(kind='commit');cycle=a.assessment_cycle_ids()[3]
        old=p['cycles'][3]['bundle_identity'];p['cycles'][3]['bundle_identity']=dict(
            uri_generation=old['durable_uri']+'#'+old['generation'],sha256=old['sha256'],bytes=old['bytes'])
        self.rehash(p);provider=self.provider(p,c)
        with provider:self.assertEqual(provider(cycle,rows[cycle])['status'],'READY')

    def test_conflicting_generation_uri_alias_is_rejected(self):
        p,_,c=self.fixture();old=copy.deepcopy(p['cycles'][3]['bundle_identity'])
        old['uri_generation']=old['durable_uri']+'#456';p['cycles'][3]['bundle_identity']=old;self.rehash(p)
        with self.assertRaisesRegex(ValueError,'Conflicting URI/generation'):self.provider(p,c)
        self.assertFalse(c.calls)

    def test_protected_cycle_opaque_transport_and_old_lineage_guard(self):
        p,rows,c=self.fixture(kind='commit');cycle=a.assessment_cycle_ids()[3];provider=self.provider(p,c)
        with provider:
            result=provider(cycle,rows[cycle]);self.assertEqual(result['status'],'READY')
            self.assertEqual(len(c.calls),2);self.assertEqual(provider.budget.snapshot()['attempt_states'],{'VERIFIED':2})
            with self.assertRaisesRegex(ValueError,'Protected assessment'):
                authenticate_bundle_identity(result['spec'],rows[cycle])
            b,w,r=a.authenticate_assessment_bundle_identity(result['spec'],rows[cycle],expected_sealed_row_sha256=digest(rows[cycle]))
            self.assertEqual(w['cycle'],cycle);self.assertFalse(r['numeric_values_decoded'])

    def test_unknown_bundle_is_learned_only_from_authenticated_commit(self):
        p,rows,c=self.fixture(kind='commit',unknown_bundle=True);cycle=a.assessment_cycle_ids()[3];provider=self.provider(p,c)
        with provider:
            result=provider(cycle,rows[cycle]);self.assertTrue(result['ownership']['owned_archive']);self.assertEqual(len(c.calls),2)
            self.assertIsNone(provider.plan['cycles'][3]['bundle_identity'])

    def test_changed_sealed_row_rejected_before_fetch(self):
        p,rows,c=self.fixture();cycle=a.assessment_cycle_ids()[3];provider=self.provider(p,c)
        with provider:
            changed=copy.deepcopy(rows[cycle]);changed['data_complete']=False
            with self.assertRaisesRegex(ValueError,'sealed index'):provider(cycle,changed)
        self.assertFalse(c.calls)

    def test_caller_plan_mutation_does_not_change_captured_plan(self):
        p,rows,c=self.fixture();cycle=a.assessment_cycle_ids()[3];provider=self.provider(p,c)
        p['cycles'][3]['sealed_row_sha256']='f'*64
        with provider:self.assertEqual(provider(cycle,rows[cycle])['status'],'READY')

    def test_cloud_only_candidate_is_not_opened_or_hydrated(self):
        p,rows,c=self.fixture(local_archive=True);cycle=a.assessment_cycle_ids()[3];provider=self.provider(p,c)
        cloud=self.root/'retained-owned.tar';original_lstat=Path.lstat;original_open=Path.open
        def inspect(path,*args,**kwargs):
            st=original_lstat(path,*args,**kwargs)
            if path==cloud:return SimpleNamespace(st_mode=st.st_mode,st_size=st.st_size,st_flags=0x40000000)
            return st
        def opening(path,*args,**kwargs):
            if path==cloud:raise AssertionError('Cloud-only payload was opened')
            return original_open(path,*args,**kwargs)
        with provider,patch.object(Path,'lstat',inspect),patch.object(Path,'open',opening):
            result=provider(cycle,rows[cycle]);self.assertTrue(result['ownership']['owned_archive'])
        self.assertEqual(len(c.calls),1)

    def test_one_writer_and_binding_prevent_concurrent_or_changed_runs(self):
        p,_,c=self.fixture();b=self.budget();one=self.provider(p,c,budget=b);two=self.provider(p,c,budget=b)
        with one:
            with self.assertRaises(BlockingIOError):two.__enter__()
        changed=copy.deepcopy(p);changed['revision_note']='different';self.rehash(changed);three=self.provider(changed,c,budget=b)
        with self.assertRaisesRegex(ValueError,'binding differs'):three.__enter__()
        self.assertIsNone(three.lock);self.assertFalse(c.calls)

    def test_three_transient_attempts_are_precharged_and_retained(self):
        p,rows,c=self.fixture();cycle=a.assessment_cycle_ids()[3];c.failures=[ConnectionError('one'),ConnectionError('two')];provider=self.provider(p,c)
        with provider,patch('hydra_directflow.staging.time.sleep'):
            provider(cycle,rows[cycle])
        state=provider.budget.snapshot();self.assertEqual(state['attempted_gets'],3);self.assertEqual(state['attempt_states'],{'FAILED':2,'VERIFIED':1})
        self.assertEqual(state['precharged_body_bytes'],3*(len(next(iter(c.objects.values())))+1))

    def test_bad_source_hash_is_not_retried(self):
        p,rows,c=self.fixture();cycle=a.assessment_cycle_ids()[3];key=next(iter(c.objects));c.objects[key]=b'bad';provider=self.provider(p,c)
        with provider:
            with self.assertRaisesRegex(ValueError,'size differs|SHA256'):provider(cycle,rows[cycle])
        self.assertEqual(len(c.calls),1);self.assertEqual(provider.budget.snapshot()['attempt_states'],{'FAILED':1})

    def test_disk_floor_before_client_and_after_transport(self):
        p,rows,c=self.fixture();self.free=8*a.GIB-1
        with self.assertRaisesRegex(ValueError,'free-disk'):self.provider(p,c)
        self.assertFalse(c.calls);self.free=100*a.GIB
        provider=self.provider(p,c,budget=self.budgets[0]);cycle=a.assessment_cycle_ids()[3]
        with provider:
            c.after_open=lambda:setattr(self,'free',8*a.GIB-1)
            with self.assertRaisesRegex(ValueError,'free-disk'):provider(cycle,rows[cycle])
        self.assertEqual(provider.budget.snapshot()['attempted_gets'],1)

    def test_hotset_bound_stops_before_transfer(self):
        p,rows,c=self.fixture();provider=self.provider(p,c);cycle=a.assessment_cycle_ids()[3]
        with provider:
            with (provider.directory/'raw/held.tar').open('wb') as f:f.truncate(2*a.GIB)
            with self.assertRaisesRegex(ValueError,'hotset'):provider(cycle,rows[cycle])
        self.assertFalse(c.calls)

    def test_direct_transfer_cannot_fetch_an_unapproved_object(self):
        p,_,c=self.fixture();provider=self.provider(p,c)
        with provider:
            item=dict(durable_uri='gs://test-bucket/other/cycles/20240731_00Z.tar',generation='123',sha256='f'*64,bytes=3)
            with self.assertRaisesRegex(ValueError,'approved source identities'):
                provider._transfer(item,provider.directory/'raw/20240731_00Z.tar',a.MAX_ARCHIVE,'archive')
        self.assertFalse(c.calls)

    def test_only_owned_archive_is_evicted_after_verified_cache(self):
        p,rows,c=self.fixture();provider=self.provider(p,c);cycle=a.assessment_cycle_ids()[3]
        with provider:
            result=provider(cycle,rows[cycle]);path=Path(result['spec']['archive_path'])
            with self.assertRaisesRegex(ValueError,'Verified durable cache'):
                provider.on_cache_committed({},result)
            self.assertTrue(path.exists())
            provider.on_cache_committed(dict(cache_payload_verified=True,cache_key='c'*64,cache_commit_sha256='d'*64,
                                            archive_sha256=result['ownership']['archive_sha256']),result)
            self.assertFalse(path.exists());self.assertEqual(provider.evictions,1)

    def test_lineage_rejects_wrong_calendar_and_worker_binding(self):
        p,rows,c=self.fixture(kind='commit');cycle=a.assessment_cycle_ids()[3]
        with self.assertRaisesRegex(ValueError,'calendar'):
            a.authenticate_assessment_bundle_identity({'cycle_id':'20260731_00Z'},rows[cycle])
        body=json.loads(c.objects[rows[cycle]['source_locations']['commit']['sha256']]);body['worker_manifest']['cycle']='WRONG'
        raw=encoded(body);path=self.root/'bad-commit.json';path.write_bytes(raw);row=copy.deepcopy(rows[cycle]);row['source_locations']['commit']['sha256']=sha(raw)
        with self.assertRaisesRegex(ValueError,'worker manifest'):
            a.authenticate_assessment_bundle_identity({'cycle_id':cycle,'commit_path':str(path)},row)


if __name__=='__main__':unittest.main()
