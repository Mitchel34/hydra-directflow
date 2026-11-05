"""Synthetic opaque-checkpoint and provenance reconciliation; no models/data."""
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra_directflow import coverage_extension_manifest as m
from hydra_directflow.contract import ARCHITECTURES, GAUGE_IDS
from hydra_directflow.reference import digest


def encode(value): return (json.dumps(value,sort_keys=True)+'\n').encode()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def iso(stamp): return datetime.fromtimestamp(stamp,timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def membership(gauge='03161000'):
    first=int(datetime(2018,7,12,tzinfo=timezone.utc).timestamp());gi=GAUGE_IDS.index(gauge)
    chunks={'train/000':dict(first=first,origins=30,commit_sha256='c'*64)}
    ids=tuple(gauge+':'+iso(first+i*3600) for i in range(6))
    manifest=dict(role_time_bounds={'fit':dict(start_utc=iso(first),end_utc=iso(first+24*3600))},
        refit_at_utc='2023-07-31T00:00:00Z',members=[dict(source_chunk='train/000',commit_sha256='c'*64,
            rows=[gi+i*3 for i in range(30)])],roles={'fit':dict(cases=6,case_ids_sha256=digest(list(ids)))})
    return manifest,chunks,ids


def common_membership(gauge='03161000',history=168):
    start=int(datetime(2023,7,31,tzinfo=timezone.utc).timestamp());gi=GAUGE_IDS.index(gauge)
    hours=(0,2,50,8765);ids=tuple(gauge+':'+iso(start+i*3600) for i in hours)
    chunks={'outer/000':dict(first=start,origins=8766,commit_sha256='d'*64)}
    manifest=dict(operation='inputs',fold='final_development',input_arm='full',history_hours=history,gauge_id=gauge,
        source_plan_sha256=m.SOURCE_PLAN,snapshot_sha256=m.SNAPSHOT,
        role_time_bounds={'outer':dict(start_utc='2023-07-31T00:00:00Z',end_utc='2024-07-31T00:00:00Z')},
        members=[dict(source_chunk='outer/000',commit_sha256='d'*64,rows=[i*3+gi for i in hours])],
        roles={'outer':dict(cases=len(ids),case_ids_sha256=digest(list(ids)))},case_count=len(ids))
    return manifest,chunks,ids


class GradientMembershipTests(unittest.TestCase):
    def test_exact_gauge_order_and_h18_tail_reconstruct_saved_ids(self):
        for gauge in GAUGE_IDS:
            manifest,chunks,ids=membership(gauge)
            self.assertEqual(m.gradient_membership(manifest,gauge,chunks,digest(list(ids))),ids)

    def test_wrong_gauge_duplicate_out_of_range_and_unordered_rows_fail(self):
        for change,message in [('gauge','Wrong gauge'),('duplicate','Unique ordered'),('range','Wrong gauge'),('order','Unique ordered')]:
            manifest,chunks,ids=membership();rows=manifest['members'][0]['rows']
            if change=='gauge':rows[0]=1
            elif change=='duplicate':rows.insert(0,0)
            elif change=='range':rows.append(90)
            else:rows.reverse()
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,message):
                m.gradient_membership(manifest,'03161000',chunks,digest(list(ids)))

    def test_unknown_chunk_or_changed_commit_cannot_recover_membership(self):
        for key,value,message in [('source_chunk','unknown/000','Unknown gradient'),('commit_sha256','d'*64,'chunk identity')]:
            manifest,chunks,ids=membership();manifest['members'][0][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,message):
                m.gradient_membership(manifest,'03161000',chunks,digest(list(ids)))

    def test_expected_and_manifest_case_hashes_both_bind_ids(self):
        manifest,chunks,ids=membership()
        with self.assertRaisesRegex(ValueError,'case identity'):
            m.gradient_membership(manifest,'03161000',chunks,'0'*64)
        manifest['roles']['fit']['case_ids_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'case identity'):
            m.gradient_membership(manifest,'03161000',chunks,digest(list(ids)))

    def test_refit_availability_uses_timestamps_in_addition_to_partition_end(self):
        refit=int(datetime(2023,7,31,tzinfo=timezone.utc).timestamp());first=refit-22*3600
        chunks={'train/000':dict(first=first,origins=4,commit_sha256='c'*64)}
        ids=tuple('03161000:'+iso(first+i*3600) for i in range(3))
        manifest=dict(role_time_bounds={'fit':dict(start_utc=iso(first),end_utc=iso(refit))},refit_at_utc=iso(refit),
            members=[dict(source_chunk='train/000',commit_sha256='c'*64,rows=[0,3,6,9])],
            roles={'fit':dict(cases=3,case_ids_sha256=digest(list(ids)))})
        self.assertEqual(m.gradient_membership(manifest,'03161000',chunks,digest(list(ids))),ids)
        bad=tuple('03161000:'+iso(first+i*3600) for i in range(4));manifest['roles']['fit']=dict(cases=4,case_ids_sha256=digest(list(bad)))
        with self.assertRaisesRegex(ValueError,'case identity'):
            m.gradient_membership(manifest,'03161000',chunks,digest(list(bad)))

    def test_invalid_empty_window_and_duplicate_members_are_rejected(self):
        manifest,chunks,ids=membership();manifest['role_time_bounds']['fit']['end_utc']=manifest['role_time_bounds']['fit']['start_utc']
        manifest['roles']['fit']=dict(cases=0,case_ids_sha256=digest([]))
        with self.assertRaises(ValueError):m.gradient_membership(manifest,'03161000',chunks,digest([]))
        manifest,chunks,ids=membership();member=manifest['members'][0];manifest['members']=[{**member,'rows':member['rows'][:3]}, {**member,'rows':member['rows'][3:]}]
        with self.assertRaises(ValueError):m.gradient_membership(manifest,'03161000',chunks,digest(list(ids)))


class SourceAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.corpus=self.root/'corpus';self.corpus.mkdir()
        self.plan_raw=encode(dict(synthetic_metadata_only=True));(self.corpus/'PLAN.json').write_bytes(self.plan_raw)
        self.chunks=[];observed=[]
        for index in range(2):
            unit={'id':'synthetic'};chunk=dict(index=index,first_origin_utc=f'2023-06-0{index+1}T00:00:00Z',origin_count=24)
            self.chunks.append((unit,chunk));name=f'synthetic/{index:03d}';directory=self.corpus/'chunks'/name;directory.mkdir(parents=True)
            for filename in ('COMMIT.json','case-index.json','model.npz'):(directory/filename).write_bytes(b'OPAQUE NEVER DECODED '+filename.encode())
            observed.append(dict(chunk=name,commit_sha256=str(index+1)*64))
        self.snapshot=digest(dict(plan_sha256=m.SOURCE_PLAN,plan_file_sha256=sha(self.plan_raw),observed=observed,missing=[]))
        self.receipt=dict(status='COMPLETE_COMMITTED_CORPUS_SNAPSHOT',snapshot_sha256=self.snapshot,plan_sha256=m.SOURCE_PLAN,
                          plan_file_sha256=sha(self.plan_raw),missing_chunks=[],numerical_payloads_checked=True,observed_commits=observed)
        self.receipt_path=self.root/'integrity.json';self.receipt_path.write_bytes(encode(self.receipt))

    def tearDown(self):self.temp.cleanup()

    @contextmanager
    def mocks(self):
        with patch.object(m.ds,'_plan',return_value=({'synthetic':True},self.chunks,sha(self.plan_raw))), \
                patch.object(m,'SNAPSHOT',self.snapshot),patch.object(m,'INTEGRITY_PATH',self.receipt_path), \
                patch.object(m,'INTEGRITY_SHA256',sha(self.receipt_path.read_bytes())):
            yield

    def test_source_inventory_uses_prior_integrity_receipt_without_numerical_reads(self):
        original=m.ds._read;opened=[];charges=[]
        def read(path,maximum):
            opened.append(Path(path));self.assertEqual(Path(path),self.receipt_path)
            return original(path,maximum)
        with self.mocks(),patch.object(m.ds,'_read',side_effect=read):
            chunks,source=m.source_authority(self.corpus,read_budget=charges.append)
        self.assertEqual(set(chunks),{'synthetic/000','synthetic/001'});self.assertEqual(source['resident_shard_files'],6)
        self.assertFalse(source['full_integrity_audit_replayed']);self.assertEqual(opened,[self.receipt_path])
        self.assertEqual(charges,[len(self.plan_raw),self.receipt_path.stat().st_size])

    def test_unexpected_or_missing_source_file_is_not_silently_accepted(self):
        target=self.corpus/'chunks/synthetic/000/extra';target.write_bytes(b'extra')
        with self.mocks(),self.assertRaisesRegex(ValueError,'Unexpected source shard'):
            m.source_authority(self.corpus)
        target.unlink();(self.corpus/'chunks/synthetic/000/model.npz').unlink()
        with self.mocks(),self.assertRaisesRegex(ValueError,'Unexpected source shard'):
            m.source_authority(self.corpus)

    def test_cloud_only_flag_fails_before_opening_body(self):
        target=self.corpus/'chunks/synthetic/000/model.npz';original=Path.lstat
        def stat(path,*args,**kwargs):
            value=original(path,*args,**kwargs)
            if path==target:return SimpleNamespace(st_size=value.st_size,st_flags=0x40000000)
            return value
        with self.mocks(),patch.object(Path,'lstat',stat),self.assertRaisesRegex(ValueError,'resident'):
            m.source_authority(self.corpus)

    def test_incomplete_prior_receipt_cannot_stand_in_for_verified_snapshot(self):
        self.receipt['numerical_payloads_checked']=False;self.receipt_path.write_bytes(encode(self.receipt))
        with self.mocks(),self.assertRaisesRegex(ValueError,'Previously verified complete'):
            m.source_authority(self.corpus)


class OriginalCommonMembershipTests(unittest.TestCase):
    def test_sparse_exact_original_cases_and_last_h18_safe_origin(self):
        for gauge in GAUGE_IDS:
            for history in (168,24):
                manifest,chunks,ids=common_membership(gauge,history)
                self.assertEqual(m.original_common_ids(manifest,gauge,chunks,digest(list(ids))),ids)
                self.assertTrue(ids[-1].endswith('2024-07-30T05:00:00Z'))

    def test_wrong_original_manifest_scope_and_window_fail(self):
        for field,value in [('gauge_id','03479000'),('fold','cv_3'),('input_arm','q_only'),('operation','labels'),
                            ('history_hours',48),('source_plan_sha256','0'*64),('snapshot_sha256','0'*64)]:
            manifest,chunks,ids=common_membership();manifest[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                m.original_common_ids(manifest,'03161000',chunks,digest(list(ids)))
        manifest,chunks,ids=common_membership();manifest['role_time_bounds']['outer']['end_utc']='2025-07-31T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'window differs'):
            m.original_common_ids(manifest,'03161000',chunks,digest(list(ids)))

    def test_unknown_duplicate_wrong_gauge_and_cross_boundary_rows_fail(self):
        for kind in ('unknown','duplicate_chunk','duplicate_row','gauge','boundary'):
            manifest,chunks,ids=common_membership();member=manifest['members'][0]
            if kind=='unknown':member['source_chunk']='unknown/000'
            elif kind=='duplicate_chunk':manifest['members'].append(deepcopy(member))
            elif kind=='duplicate_row':member['rows'].insert(0,0)
            elif kind=='gauge':member['rows'][0]=1
            else:chunks['outer/000']['origins']=8767;member['rows'].append(8766*3)
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                m.original_common_ids(manifest,'03161000',chunks,digest(list(ids)))

    def test_manifest_and_expected_case_identity_are_both_required(self):
        manifest,chunks,ids=common_membership()
        with self.assertRaisesRegex(ValueError,'membership differs'):
            m.original_common_ids(manifest,'03161000',chunks,'0'*64)
        manifest['roles']['outer']['cases']+=1
        with self.assertRaisesRegex(ValueError,'membership differs'):
            m.original_common_ids(manifest,'03161000',chunks,digest(list(ids)))

    def test_empty_original_population_is_not_a_valid_clean_regression(self):
        manifest,chunks,_=common_membership();manifest['members']=[];manifest['roles']['outer']=dict(cases=0,case_ids_sha256=digest([]));manifest['case_count']=0
        with self.assertRaises(ValueError):m.original_common_ids(manifest,'03161000',chunks,digest([]))


class ModelManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.run=Path(self.temp.name).resolve();self.bodies={};self.reads=[]
        self.plan=dict(plan_sha256='a'*64,protocol_sha256='b'*64,source_plan_sha256=m.SOURCE_PLAN,
                       snapshot_sha256=m.SNAPSHOT,run_directory=str(self.run),jobs=[])
        self.records=[];self.chunks=None
        for gauge in GAUGE_IDS:
            for architecture in ARCHITECTURES:
                for hours in (168,24):
                    for seed in (17,29,43):self.add(gauge,architecture,hours,seed)

    def tearDown(self):self.temp.cleanup()

    def put(self,path,raw):
        self.bodies[path]=raw
        return dict(path=path,bytes=len(raw),sha256=sha(raw))

    def add(self,gauge,architecture,hours,seed):
        index=len(self.records);job_id=f'final__{gauge}__{architecture}__L{hours}__seed{seed}';base='jobs/'+job_id+'/attempt-000001'
        job=dict(order=index,job_id=job_id,fold='final_development',input_arm='full',gauge_id=gauge,
                 architecture=architecture,history_hours=hours,seed=seed,
                 device='cpu' if architecture in ('LSTM','GRU_Transformer') else 'mps')
        self.plan['jobs'].append(deepcopy(job))
        fit,chunks,ids=membership(gauge)
        inputs,outer_chunks,common_ids=common_membership(gauge,hours);self.chunks={**chunks,**outer_chunks}
        fit.update(fold='final_development',gauge_id=gauge,history_hours=hours,input_arm='full',
                   cohort_admission_digest=digest([gauge,'common-training']),source_plan_sha256=m.SOURCE_PLAN,snapshot_sha256=m.SNAPSHOT)
        fit['source_identity']=digest(fit);self.put(base+'/fit-source.json',encode(fit))
        cp=self.put(base+'/checkpoint.pt',b'SYNTHETIC OPAQUE NON-TORCH CHECKPOINT '+job_id.encode())
        pred=self.put(base+'/predictions.npz',b'SYNTHETIC OPAQUE NON-NUMPY PREDICTIONS '+job_id.encode())
        inputs.update(target_values_decoded=False,numeric_value_members_decoded=['history','weather','state'])
        inputs['source_identity']=digest(inputs);self.put(base+'/outer-input-source.json',encode(inputs))
        last=dict(checkpoint_sha256=cp['sha256'],selected_state='best_model',selected_best_epoch=7,
            architecture=architecture,gauge_id=gauge,history_hours=hours,seed=seed,device=job['device'],
            feature_schema='FULL_HRRR_H2_W4_S2_V1',optimizer_constructed=False,scaler_fitted=False,
            training_case_ids_sha256=digest(list(ids)),training_values_sha256='e'*64)
        self.put(base+'/inference.json',encode(dict(checkpoint_sha256=cp['sha256'],last_batch_inference_receipt=last)))
        receipt=dict(status='VERIFIED_TRAIN_EVAL',protected_assessment_access=False,plan_sha256=self.plan['plan_sha256'],
            protocol_sha256=self.plan['protocol_sha256'],source_plan_sha256=m.SOURCE_PLAN,snapshot_sha256=m.SNAPSHOT,
            job_sha256=digest(job),receipt_path=base+'/completion.json',artifacts={'checkpoint':cp,'predictions':pred},
            hashes={'checkpoint':cp['sha256'],'predictions':pred['sha256']},training_source_identity=fit['source_identity'],
            training_cohort_admission_digest=fit['cohort_admission_digest'],inference_source_identity=inputs['source_identity'],
            selected_best_epoch=7,feature_schema='FULL_HRRR_H2_W4_S2_V1',outer_case_ids_sha256=digest(list(common_ids)),scored_cases=len(common_ids))
        self.records.append(dict(job=job,receipt=receipt,receipt_sha256=digest(receipt)))

    @contextmanager
    def mocks(self):
        def read(path,maximum,*,read_budget=lambda n:None,expected=None):
            relative=str(Path(path).relative_to(self.run));self.reads.append(relative);raw=self.bodies[relative]
            if len(raw)>maximum:raise ValueError('Synthetic byte bound')
            if expected is not None and sha(raw)!=expected:raise ValueError('Manifest input hash differs')
            read_budget(len(raw));return raw
        with patch.object(m,'source_authority',return_value=(self.chunks,dict(synthetic=True))),patch.object(m,'read',side_effect=read):yield

    def reconcile(self):
        with self.mocks():return m.reconcile_models(self.plan,self.records,self.run/'never-opened-corpus',self.run)

    def test_all72_opaque_models_reconciled_without_deserialization_or_values(self):
        result=self.reconcile();self.assertEqual(len(result['models']),72);self.assertEqual(len(result['gradient_memberships']),3)
        self.assertEqual(result['model_constructions'],0);self.assertEqual(result['checkpoint_deserializations'],0)
        self.assertFalse(result['target_values_decoded']);self.assertEqual(len(self.reads),360)
        self.assertEqual({r['history_hours'] for r in result['models']},{168,24})
        self.assertEqual(set(result['original_common_case_ids_by_gauge']),set(GAUGE_IDS))
        self.assertTrue(all(len(ids)==4 for ids in result['original_common_case_ids_by_gauge'].values()))

    def test_saved_common_count_must_match_recovered_input_membership(self):
        self.records[0]['receipt']['scored_cases']+=1
        with self.assertRaises(ValueError):self.reconcile()

    def test_missing_duplicate_wrong_fold_and_wrong_gauge_inventory_fail(self):
        for change in ('missing','duplicate','fold','gauge'):
            old=deepcopy(self.records)
            if change=='missing':self.records.pop()
            elif change=='duplicate':self.records.append(deepcopy(self.records[0]))
            elif change=='fold':self.records[0]['job']['fold']='cv_1'
            else:self.records[0]['job']['gauge_id']='not-a-gauge'
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'Exact72'):
                self.reconcile()
            self.records=old

    def test_opaque_checkpoint_corruption_and_descriptor_size_fail(self):
        desc=self.records[0]['receipt']['artifacts']['checkpoint'];old=self.bodies[desc['path']]
        self.bodies[desc['path']]=b'changed'
        with self.assertRaisesRegex(ValueError,'hash differs'):self.reconcile()
        self.bodies[desc['path']]=old;desc['bytes']+=1
        with self.assertRaisesRegex(ValueError,'Checkpoint size differs'):self.reconcile()

    def test_wrong_inference_gauge_or_last_weight_selection_fails(self):
        path=str(Path(self.records[0]['receipt']['receipt_path']).parent/'inference.json');old=self.bodies[path]
        for key,value in [('gauge_id','03164000'),('selected_state','model'),('seed',29),('history_hours',24),('scaler_fitted',True)]:
            data=json.loads(old);data['last_batch_inference_receipt'][key]=value;self.bodies[path]=encode(data)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'inference/model identity'):
                self.reconcile()
        self.bodies[path]=old

    def test_changed_training_manifest_or_label_decoding_marker_fails(self):
        base=Path(self.records[0]['receipt']['receipt_path']).parent
        path=str(base/'fit-source.json');data=json.loads(self.bodies[path]);data['history_hours']=24;self.bodies[path]=encode(data)
        with self.assertRaisesRegex(ValueError,'training-source manifest'):self.reconcile()

    def test_original_plan_protocol_source_snapshot_and_feature_bindings(self):
        for field in ('protocol_sha256','source_plan_sha256','snapshot_sha256','feature_schema'):
            old=deepcopy(self.records[0]['receipt']);self.records[0]['receipt'][field]='changed'
            with self.subTest(field=field),self.assertRaises(ValueError):self.reconcile()
            self.records[0]['receipt']=old
        self.records[0]['job']['device']='mps';self.records[0]['receipt']['job_sha256']=digest(self.records[0]['job'])
        with self.assertRaises(ValueError):self.reconcile()

    def test_self_consistent_job_cannot_replace_the_frozen_plan_job(self):
        job=self.records[0]['job'];job['unreviewed_field']='not in immutable plan';self.records[0]['receipt']['job_sha256']=digest(job)
        with self.assertRaises(ValueError):self.reconcile()

    def test_receipt_hash_map_must_match_descriptor_and_last_feature_schema(self):
        old=deepcopy(self.records[0]['receipt']);self.records[0]['receipt']['hashes']['checkpoint']='0'*64
        with self.assertRaises(ValueError):self.reconcile()
        self.records[0]['receipt']=old
        path=str(Path(old['receipt_path']).parent/'inference.json');data=json.loads(self.bodies[path]);data['last_batch_inference_receipt']['feature_schema']='Q_ONLY_H2_HORIZON1_S0_V1';self.bodies[path]=encode(data)
        with self.assertRaises(ValueError):self.reconcile()


if __name__=='__main__':unittest.main()
