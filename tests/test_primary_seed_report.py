"""Synthetic complete-matrix evidence; no models or user observations."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from hydra_directflow.campaign_ledger import CampaignLedger,canonical,digest
from hydra_directflow.local_data import iso
from hydra_directflow.scoring import ScoreIdentity,score_physical_h18
from hydra_directflow.reference import ASSESSMENT_START
from hydra_directflow.quality import PRIMARY_APPROVED
from hydra_directflow.trainer_bundle import _array_digest
import test_campaign_ledger as ledger_fixture

PATH=Path(__file__).resolve().parents[1]/'scripts/report_primary_seed.py'
SPEC=importlib.util.spec_from_file_location('primary_seed_report_tests',PATH)
reporter=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(reporter)


class SeedReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Shared immutable synthetic artifact baseline; each test gets its own
        # complete metadata/byte copies, never a scientific ledger.
        cls.data={}
        for fi,fold in enumerate(('cv_1','cv_2','cv_3','final_development')):
            origin=1596240000+fi*365*86400+np.arange(168,dtype=np.int64)*4*3600
            future=origin[:,None]+np.arange(1,19)*3600
            for gi,gauge in enumerate(ledger_fixture.GAUGES):
                ids=np.array([gauge+':'+iso(int(t)) for t in origin],dtype='<U29')
                truth=np.broadcast_to((10+np.arange(168,dtype=float)%30)[:,None],(168,18)).copy()
                cls.data[fold,gauge]=(ids,origin,future,np.full(168,gi,np.int64),truth)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.run=self.root/'.local/run';self.run.mkdir(parents=True);self.ledger_dir=self.root/'.local/ledger'
        (self.root/'fixture.py').write_bytes(b'original synthetic implementation')
        self.plan=ledger_fixture.plan(logical=200,physical=230,global_updates=41400,output_bytes=8*1024**3)
        self.plan.update(run_directory=str(self.run),ledger_directory=str(self.ledger_dir),protocol_frozen=True,
                         scientific_training_authorized=True,implementation_sha256={'fixture.py':hashlib.sha256((self.root/'fixture.py').read_bytes()).hexdigest()})
        for job in self.plan['jobs']:
            ids=self.data[job['fold'],job['gauge_id']][0]
            job['role_case_ids_sha256']['outer_scoring']=digest(ids.tolist())
            job['data_job']={'actual_common_cases':dict(gradient_fit=1,inner_stopping=1,outer_scoring=168)}
        self.plan=ledger_fixture.repin(self.plan);self.ledger=CampaignLedger(self.ledger_dir)
        self.ledger.initialize(self.plan);self.receipts=[]
        with self.ledger.writer():
            self.ledger.reserve_output(2*1024**3,kind='BOOKKEEPING')
            with self.ledger._transaction():
                for job in self.plan['jobs']:
                    position=job['order'];receipt=self.make_artifacts(job)
                    self.receipts.append(receipt)
                    self.ledger.connection.execute("UPDATE jobs SET state='VERIFIED_TRAIN_EVAL',entered=10,confirmed=10,started=?,deadline=?,completion_json=? WHERE position=?",
                        (1000+position*100,1000+position*100+21600,canonical(receipt),position))
                    attempt=self.ledger.connection.execute("INSERT INTO attempts(job_position,lease_id,status,started,finished,resume_logical_updates,entered,confirmed) VALUES(?,?,'VERIFIED_TRAIN_EVAL',?,?,0,10,10)",
                        (position,'fixture',1000+position*100,1090+position*100)).lastrowid
                    self.ledger.connection.execute("INSERT INTO artifacts(job_position,attempt_id,kind,receipt_sha,receipt_json,logical_updates,created) VALUES(?,?,'COMPLETE',?,?,10,?)",
                        (position,attempt,digest(receipt),canonical(receipt),1090+position*100))
                self.ledger.connection.execute('UPDATE campaign SET entered=1800,confirmed=1800,started=1000,deadline=1200000 WHERE id=1')
        self.rootpatch=patch.object(reporter,'ROOT',self.root);self.rootpatch.start()

    def tearDown(self):self.rootpatch.stop();self.tmp.cleanup()

    def bindings(self,job):
        return dict(plan_sha256=self.plan['plan_sha256'],protocol_sha256=self.plan['protocol_sha256'],source_plan_sha256=self.plan['source_plan_sha256'],
            snapshot_sha256=self.plan['snapshot_sha256'],implementation_sha256=self.plan['implementation_sha256'],job_id=job['job_id'],job_sha256=digest(job))

    def put(self,directory,name,raw):
        path=directory/name;path.write_bytes(raw)
        return dict(path=str(path.relative_to(self.run)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

    def make_artifacts(self,job):
        directory=self.run/'jobs'/job['job_id'];directory.mkdir(parents=True)
        ids,origins,future,sites,truth=self.data[job['fold'],job['gauge_id']]
        schema='Q_ONLY_H2_HORIZON1_S0_V1' if job['input_arm']=='q_only' else 'FULL_HRRR_H2_W4_S2_V1'
        cohort=digest([job['fold'],job['gauge_id']])
        label_arrays_sha=_array_digest(dict(origin_time=origins,target_time=future,site_id=sites,case_id=ids,target=truth,target_mask=np.ones_like(truth,dtype=bool)))
        label_core=dict(schema='hydra-primary-campaign-data/v1',operation='labels',source_plan_sha256=self.plan['source_plan_sha256'],snapshot_sha256=self.plan['snapshot_sha256'],
            gauge_id=job['gauge_id'],fold=job['fold'],input_arm=job['input_arm'],history_hours=job['history_hours'],quality_policy=PRIMARY_APPROVED,
            roles={'outer':dict(cases=168,case_ids_sha256=digest(ids.tolist()))},case_count=168,cohort_admission_digest=cohort,raw_arrays_sha256=label_arrays_sha)
        label_source=digest(label_core)
        self.put(directory,'outer-label-source.json',reporter._bytes(dict(**label_core,source_identity=label_source,returned_arrays_sha256=label_arrays_sha,construction={})))
        cp=self.put(directory,'checkpoint.pt',b'OPAQUE_SYNTHETIC_CHECKPOINT_NO_PICKLE')
        receipt=dict(**self.bindings(job),status='VERIFIED_TRAIN_EVAL',logical_updates=10,completed_epochs=10,selected_best_epoch=5,
            training_source_identity='1'*64,training_cohort_admission_digest='2'*64,training_arrays_sha256='3'*64,
            feature_schema=schema,trainer_spec={'synthetic':True},inference_source_identity=digest([job['fold'],job['gauge_id'],'inputs']),
            inference_cohort_admission_digest=cohort,label_source_identity=label_source,outer_case_ids_sha256=digest(ids.tolist()),
            scored_cases=168,protected_assessment_access=False)
        meta=dict(**self.bindings(job),**{k:receipt[k] for k in ('training_source_identity','training_cohort_admission_digest',
            'inference_source_identity','inference_cohort_admission_digest','selected_best_epoch')},checkpoint_sha256=cp['sha256'],
            case_ids_sha256=digest(ids.tolist()),case_count=168,label_values_opened=False,posthoc_clipping=False,best_only=True)
        predictions=truth+(job['seed']%7)/10
        buffer=io.BytesIO();np.savez_compressed(buffer,prediction_q=predictions,case_id=ids,origin_time=origins,forecast_time=future,
            site_id=sites,metadata_json=np.array(reporter._bytes(meta).decode()))
        prediction=self.put(directory,'predictions.npz',buffer.getvalue())
        identity=ScoreIdentity(job['architecture'],f'{job["architecture"]}-{job["input_arm"]}-L{job["history_hours"]}',job['gauge_id'],schema,
            job['history_hours'],job['fold'],job['seed'])
        panel=score_physical_h18(case_ids=tuple(ids.tolist()),origins=origins,forecast_time=future,labels=truth,
            label_mask=np.ones_like(truth,dtype=bool),predictions=predictions,identity=identity,input_cohort_digest=cohort,
            label_policy_digest=digest({'policy':PRIMARY_APPROVED}),label_source_digest=label_source)
        metrics=dict(**self.bindings(job),checkpoint_sha256=cp['sha256'],predictions_sha256=prediction['sha256'],
            outer_case_ids_sha256=digest(ids.tolist()),panel=panel)
        metric=self.put(directory,'metrics.json',reporter._bytes(metrics));receipt['score_status']=panel['status']
        receipt.update(artifacts=dict(checkpoint=cp,predictions=prediction,metrics=metric),hashes=dict(checkpoint=cp['sha256'],predictions=prediction['sha256'],metrics=metric['sha256']),
            receipt_path=str((directory/'completion.json').relative_to(self.run)))
        (directory/'completion.json').write_bytes(reporter._bytes(receipt));return receipt

    def publish(self,seed=17):return reporter.publish_seed_report(self.plan,self.run,self.ledger_dir,seed)

    def summary(self,seed=17):return json.loads((self.run/'reports'/f'seed-{seed}'/'summary.json').read_bytes())

    def test_first_seed_exact_complete_inventory_and_private_publication(self):
        before=self.ledger.snapshot();result=self.publish();summary=self.summary()
        self.assertEqual(result['status'],'PUBLISHED');self.assertEqual(before,self.ledger.snapshot())
        self.assertEqual(summary['current_seed_jobs'],108);self.assertEqual(summary['verified_cumulative_jobs'],108)
        self.assertEqual(len(summary['cv_aggregates']),24);self.assertEqual(summary['final_seed_aggregates'],[])
        self.assertEqual(summary['evidence_status'],'SINGLE_SEED_DEVELOPMENT')
        self.assertFalse(summary['scientific_poster_ready']);self.assertFalse(summary['protected_assessment_access'])
        page=(self.run/'reports/seed-17/index.html').read_text();self.assertIn('Event analysis',page);self.assertIn('RMSE',page)
        self.assertEqual(summary['execution']['prediction_value_members_decoded'],0)

    def test_second_seed_is_not_three_seed_aggregate_and_third_keeps_all_seeds(self):
        self.publish(29);two=self.summary(29)
        self.assertEqual((two['current_seed_jobs'],two['verified_cumulative_jobs']),(36,144))
        self.assertEqual(two['final_seed_aggregates'],[]);self.assertEqual(two['evidence_status'],'SINGLE_SEED_DEVELOPMENT')
        self.publish(43);three=self.summary(43)
        self.assertEqual((three['current_seed_jobs'],three['verified_cumulative_jobs']),(36,180))
        self.assertEqual(len(three['final_seed_aggregates']),36)
        for row in three['final_seed_aggregates']:
            self.assertEqual(row['score']['seeds'],[17,29,43]);self.assertFalse(row['selected_favorable_seed'])
            self.assertFalse(row['spread_is_confidence_interval']);self.assertGreater(row['descriptive_seed_spread']['sample_standard_deviation'],0)

    def test_report_replay_is_identical_no_overwrite(self):
        first=self.publish();before=(self.run/'reports/seed-17/summary.json').read_bytes();second=self.publish()
        self.assertEqual(second['status'],'VERIFIED_EXISTING_REPORT');self.assertEqual(first['summary_sha256'],second['summary_sha256'])
        self.assertEqual(before,(self.run/'reports/seed-17/summary.json').read_bytes())

    def test_missing_required_metric_artifact_blocks_report(self):
        (self.run/self.receipts[107]['artifacts']['metrics']['path']).unlink()
        with self.assertRaises(FileNotFoundError):self.publish()
        self.assertFalse((self.run/'reports/seed-17').exists())

    def test_unfinished_expected_job_blocks_even_if_other_jobs_are_done(self):
        with sqlite3.connect(self.ledger.database) as con:con.execute("UPDATE jobs SET state='ACTIVE' WHERE position=107")
        with self.assertRaisesRegex(ValueError,'Every required job'):self.publish()
        self.assertFalse((self.run/'reports/seed-17').exists())

    def test_stale_receipt_or_artifact_sha_cannot_be_published(self):
        path=self.run/self.receipts[0]['receipt_path'];path.write_bytes(b'{"status":"VERIFIED_TRAIN_EVAL"}\n')
        with self.assertRaisesRegex(ValueError,'receipt differs'):self.publish()
        path.write_bytes(reporter._bytes(self.receipts[0]));(self.run/self.receipts[0]['artifacts']['checkpoint']['path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'size differs|SHA differs'):self.publish()

    def test_prediction_values_are_never_decoded_and_wrong_gauge_blocks(self):
        original=reporter.bounded_npy;seen=[]
        def tracked(raw,**kwargs):seen.append(kwargs['shape']);return original(raw,**kwargs)
        with patch.object(reporter,'bounded_npy',side_effect=tracked):self.publish()
        self.assertEqual(len(seen),108*5)
        receipt=deepcopy(self.receipts[0]);job=deepcopy(self.plan['jobs'][0]);job['gauge_id']='03164000'
        raw=(self.run/receipt['artifacts']['predictions']['path']).read_bytes()
        with self.assertRaisesRegex(ValueError,'case IDs|membership'):reporter._prediction_coordinates(raw,receipt,job)

    def test_undefined_primary_remains_in_cv_and_three_seed_results(self):
        rows=[]
        for i,job in enumerate(self.plan['jobs']):
            metrics=json.loads((self.run/self.receipts[i]['artifacts']['metrics']['path']).read_bytes());panel=metrics['panel']
            if job['architecture']=='LSTM' and job['input_arm']=='full' and job['history_hours']==168 and job['gauge_id']=='03161000':
                for lead in panel['leads']:lead['nse']=None;lead['primary_eligible']=False;lead['primary_ineligibility']=['CONSTANT_OBSERVED_SERIES']
                panel['primary_score']=None;panel['status']='PRIMARY_UNAVAILABLE'
            sidecar=json.loads((self.run/Path(self.receipts[i]['receipt_path']).parent/'outer-label-source.json').read_bytes())
            rows.append(dict(gauge_id=job['gauge_id'],architecture=job['architecture'],input_arm=job['input_arm'],history_hours=job['history_hours'],fold=job['fold'],seed=job['seed'],panel=panel,label_arrays_sha256=sidecar['raw_arrays_sha256']))
        cv,final=reporter._aggregates(rows,43)
        self.assertEqual(len(cv),24);self.assertEqual(len(final),36)
        missing=[r for r in final if r['score']['primary_score'] is None];self.assertEqual(len(missing),1)
        self.assertIsNone(missing[0]['descriptive_seed_spread']);self.assertEqual(missing[0]['score']['undefined_seeds'],[17,29,43])

    def test_implementation_or_private_path_change_rejects(self):
        (self.root/'fixture.py').write_bytes(b'changed implementation')
        with self.assertRaisesRegex(ValueError,'SHA differs'):self.publish()
        with self.assertRaisesRegex(ValueError,'private run/ledger'):
            reporter.publish_seed_report(self.plan,self.root/'different',self.ledger_dir,17)


class MatchedIdentityTests(unittest.TestCase):
    def rows(self):
        result=[]
        for a in ledger_fixture.ARCHITECTURES:
            for arm in ('full','q_only'):
                result.append(dict(gauge_id='03161000',fold='cv_1',seed=17,architecture=a,input_arm=arm,history_hours=168,
                    label_arrays_sha256='a'*64,panel=dict(case_ids_sha256='b'*64,input_cohort_digest='c'*64,
                        label_policy_digest=digest({'policy':PRIMARY_APPROVED}),label_source_digest=digest(arm),
                        leads=[dict(lead_hours=i,n=168,n_requested=168,scored_sample_digest=digest(i),label_series_digest=digest([arm,i])) for i in range(1,19)])))
        return result

    def test_view_specific_sources_permitted_but_changed_labels_or_support_reject(self):
        rows=self.rows();self.assertTrue(reporter._matched_views(rows)['matched_typed_labels_masks_coordinates'])
        rows[1]['label_arrays_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'different typed labels'):reporter._matched_views(rows)
        rows=self.rows();rows[1]['panel']['leads'][0]['scored_sample_digest']='e'*64
        with self.assertRaisesRegex(ValueError,'scored support'):reporter._matched_views(rows)
        rows=self.rows();rows[0]['panel']['leads'][0]['label_series_digest']='f'*64
        with self.assertRaisesRegex(ValueError,'Same-view'):reporter._matched_views(rows)

    def test_schema_derived_from_arm_and_alternate_quality_refused(self):
        self.assertEqual(reporter._feature_schema({'input_arm':'q_only'}),'Q_ONLY_H2_HORIZON1_S0_V1')
        self.assertEqual(reporter._feature_schema({'input_arm':'full'}),'FULL_HRRR_H2_W4_S2_V1')
        rows=self.rows()
        for row in rows:row['panel']['label_policy_digest']='1'*64
        with self.assertRaisesRegex(ValueError,'quality policy'):reporter._matched_views(rows)

    def test_self_consistent_wrong_schema_rejects_before_any_artifact_read(self):
        job={'job_id':'synthetic-q-only','input_arm':'q_only'}
        plan=dict(plan_sha256='a'*64,protocol_sha256='b'*64,source_plan_sha256='c'*64,snapshot_sha256='d'*64,
                  implementation_sha256={'fixture':'e'*64})
        receipt=dict(**plan,job_id=job['job_id'],job_sha256=digest(job),status='VERIFIED_TRAIN_EVAL',
                     protected_assessment_access=False,feature_schema='FULL_HRRR_H2_W4_S2_V1')
        with self.assertRaisesRegex(ValueError,'declared input arm'):
            reporter._verified_panel(plan,Path('/synthetic-not-read'),dict(job=job,receipt=receipt),None)


if __name__=='__main__':unittest.main()
