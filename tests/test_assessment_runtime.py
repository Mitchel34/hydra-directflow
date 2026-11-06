"""Temporary synthetic runtime fixtures. No source data, models or network."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hydra_directflow.assessment_runtime as ar


class Clock:
    def __init__(self):self.now=1000.
    def __call__(self):return self.now


class AssessmentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.clock=Clock();self.jobs=sorted(ar.MODEL_IDS)
        self.measure=dict(combined_rss_bytes=128*1024**2,mps_bytes=0,free_bytes=16*ar.GIB)
    def tearDown(self):self.temp.cleanup()
    def plan(self,name='run',**extra):
        p=dict(run_directory=str(self.root/name),runtime_limits=dict(ar.LIMITS),
               checkpoint_sha256_by_job={j:ar.sha(j.encode()) for j in self.jobs},**extra)
        p['plan_sha256']=ar.digest(p);return p
    def runtime(self,plan=None):
        return ar.AssessmentRuntime(plan or self.plan(),clock=self.clock,resource_probe=lambda:dict(self.measure))
    def model(self,r,j=None):
        j=j or self.jobs[0];token=r.precharge_model(j,checkpoint_sha256=r.plan['checkpoint_sha256_by_job'][j]);r.confirm_model(token);return j
    def finish(self,r,j,*,name=None):
        d=r.write((name or j)+'.bin',b'synthetic-result',kind='SYNTHETIC')
        return r.commit(j,{'result':d},kind='model',metadata={'fixture':True})
    def diagnosis(self,r):
        f=r.snapshot()['failure'];return dict(classification='INFRASTRUCTURE',
            failure_id=f['id'],plan_sha256=r.plan['plan_sha256'],reason='Synthetic interrupted process',evidence_sha256=['d'*64])

    def test_fixed_limits_and_complete_model_roster(self):
        self.assertEqual(len(ar.MODEL_IDS),36)
        self.assertEqual(ar.LIMITS,dict(max_case_checkpoint_evaluations=350000,max_forward_batches=6000,
            max_batch_size=64,max_model_constructions=36,max_model_constructions_with_resume=40,
            max_numeric_seconds=21600,max_combined_rss_mps_bytes=4*ar.GIB,max_mps_bytes=2*ar.GIB,
            max_read_bytes=64*ar.GIB,max_output_bytes=2*ar.GIB,minimum_free_disk_bytes=8*ar.GIB))
        p=self.plan();p['runtime_limits']['max_forward_batches']+=1;p['plan_sha256']=ar.digest({k:v for k,v in p.items() if k!='plan_sha256'})
        with self.assertRaises(ValueError):self.runtime(p)
        p=self.plan();p['checkpoint_sha256_by_job'].pop(self.jobs[0]);p['plan_sha256']=ar.digest({k:v for k,v in p.items() if k!='plan_sha256'})
        with self.assertRaises(ValueError):self.runtime(p)

    def test_metadata_preparation_does_not_start_numeric_clock(self):
        r=self.runtime()
        with r.writer():
            r.reserve_read(5,numeric=False);d=r.write('metadata.json',b'{}',numeric=False)
            r.commit('metadata',[d]);self.assertIsNone(r.snapshot()['numeric_started_at'])
            self.clock.now+=100000
            r.reuse('metadata');self.assertIsNone(r.snapshot()['deadline'])
            r.start_numeric();s=r.snapshot();self.assertEqual(s['numeric_started_at'],self.clock.now)
            self.assertEqual(s['deadline'],self.clock.now+21600)

    def test_default_numeric_read_starts_once_and_survives_restart(self):
        p=self.plan();r=self.runtime(p)
        with r.writer():r.reserve_read(7);start=r.snapshot()['numeric_started_at']
        self.clock.now+=60
        with self.runtime(p).writer() as r:
            r.reserve_read(3);s=r.snapshot();self.assertEqual(s['numeric_started_at'],start)
            self.assertEqual(s['read_reserved'],10);self.assertEqual(s['deadline'],start+21600)

    def test_one_writer_and_changed_plan_namespace_are_rejected(self):
        p=self.plan()
        with self.runtime(p).writer():
            with self.assertRaisesRegex(ValueError,'writer'):
                with self.runtime(p).writer():pass
        wrong=dict(p,run_directory=str(self.root/'new-run'))
        with self.assertRaises(ValueError):self.runtime(wrong)
        changed=dict(p,extra='different');changed['plan_sha256']=ar.digest({k:v for k,v in changed.items() if k!='plan_sha256'})
        with self.assertRaisesRegex(ValueError,'Original namespace'):
            with self.runtime(changed).writer():pass

    def test_lost_ledger_cannot_reset_an_existing_run(self):
        p=self.plan();r=self.runtime(p)
        with r.writer():r.start_numeric()
        r.database.unlink()
        with self.assertRaisesRegex(ValueError,'lost ledger'):
            with self.runtime(p).writer():pass

    def test_precharges_precede_confirmation_and_keep_exact_tokens(self):
        r=self.runtime()
        with r.writer():
            model=r.precharge_model(self.jobs[0]);self.assertEqual(r.snapshot()['entered_models'],1)
            self.assertEqual(r.snapshot()['confirmed_models'],0)
            with self.assertRaises(ValueError):r.confirm_model(True)
            with self.assertRaises(ValueError):r.precharge_forward(case_count=1,case_ids_sha256='a'*64)
            r.confirm_model(model)
            token=r.precharge_forward(case_count=64,case_ids_sha256='a'*64)
            s=r.snapshot();self.assertEqual((s['entered_cases'],s['entered_batches'],s['confirmed_cases']),(64,1,0))
            r.confirm_forward(token)
            with self.assertRaises(ValueError):r.confirm_forward(token)
            self.finish(r,self.jobs[0])

    def test_partial_batch_allowed_but_b65_and_multi_batch_calls_rejected(self):
        r=self.runtime()
        with r.writer():
            j=self.model(r)
            for kwargs in [dict(case_count=65),dict(case_count=0),dict(case_count=True),dict(case_count=1,batch_count=2)]:
                with self.assertRaises(ValueError):r.precharge_forward(**kwargs,case_ids_sha256='a'*64)
            t=r.precharge_forward(case_count=7,case_ids_sha256='b'*64);r.confirm_forward(t)
            self.assertEqual(r.snapshot()['entered_cases'],7);self.finish(r,j)

    def test_case_and_batch_caps_fail_without_refunding_prior_work(self):
        with patch.dict(ar.LIMITS,max_case_checkpoint_evaluations=2,max_forward_batches=1):
            r=self.runtime()
            with r.writer():
                j=self.model(r);t=r.precharge_forward(case_count=2,case_ids_sha256='a'*64);r.confirm_forward(t)
                with self.assertRaises(ar.AssessmentLimitError):r.precharge_forward(case_count=1,case_ids_sha256='b'*64)
                s=r.snapshot();self.assertEqual((s['entered_cases'],s['confirmed_cases'],s['entered_batches']),(2,2,1))
                self.assertEqual(s['failure']['kind'],'RESOURCE')

    def test_wrong_checkpoint_or_foreign_model_rejected(self):
        r=self.runtime()
        with r.writer():
            for j,h in [(self.jobs[0],'0'*64),(self.jobs[0].replace('L168','L24'),'a'*64)]:
                with self.assertRaises(ValueError):r.precharge_model(j,checkpoint_sha256=h)
            self.assertEqual(r.snapshot()['entered_models'],0)

    def test_failed_work_retains_charges_and_needs_explicit_diagnosis(self):
        p=self.plan();r=self.runtime(p)
        with self.assertRaises(OSError):
            with r.writer():
                self.model(r);r.precharge_forward(case_count=11,case_ids_sha256='a'*64)
                raise OSError('Synthetic disconnect')
        self.clock.now+=30
        with self.runtime(p).writer() as r:
            original=r.snapshot();self.assertEqual(original['failure']['kind'],'INFRASTRUCTURE')
            with self.assertRaises(ValueError):r.precharge_model(self.jobs[0])
            r.authorize_resume(self.diagnosis(r));self.assertEqual(r.snapshot()['abandoned_calls'],1)
            with self.assertRaises(ValueError):r.precharge_model(self.jobs[1])
            self.model(r);t=r.precharge_forward(case_count=11,case_ids_sha256='a'*64);r.confirm_forward(t)
            s=r.snapshot();self.assertEqual((s['entered_cases'],s['confirmed_cases']),(22,11))
            self.assertEqual(s['numeric_started_at'],original['numeric_started_at']);self.finish(r,self.jobs[0])

    def test_numerical_integrity_and_resource_failures_cannot_resume(self):
        for i,kind in enumerate(['NUMERICAL','INTEGRITY','RESOURCE','IMPLEMENTATION']):
            r=self.runtime(self.plan(str(i)))
            with r.writer():
                r.record_failure(kind,'Synthetic deliberate stop')
                with self.assertRaises(ValueError):r.authorize_resume(self.diagnosis(r))
                self.assertEqual(r.snapshot()['resume_id'],0)

    def test_changed_failure_or_plan_cannot_authorize_resume(self):
        r=self.runtime()
        with r.writer():
            r.record_failure('INFRASTRUCTURE','Synthetic disconnect')
            for field,value in [('failure_id',99),('plan_sha256','0'*64),('classification','NUMERICAL'),('evidence_sha256',[])]:
                d=self.diagnosis(r);d[field]=value
                with self.assertRaises(ValueError):r.authorize_resume(d)
            self.assertIsNone(r.snapshot()['numeric_started_at'])

    def test_model_caps_change_only_after_diagnosed_resume(self):
        with patch.dict(ar.LIMITS,max_model_constructions=1,max_model_constructions_with_resume=2):
            r=self.runtime()
            with r.writer():
                j=self.model(r);r.record_failure('INFRASTRUCTURE','Synthetic disconnect')
                r.authorize_resume(self.diagnosis(r));self.model(r,j);self.finish(r,j)
                with self.assertRaises(ar.AssessmentLimitError):r.precharge_model(self.jobs[1])
                self.assertEqual(r.snapshot()['entered_models'],2)

    def test_deadline_persists_and_cannot_be_extended_by_resume(self):
        r=self.runtime()
        with r.writer():
            r.start_numeric();deadline=r.snapshot()['deadline'];r.record_failure('INFRASTRUCTURE','Synthetic disconnect')
            self.clock.now=deadline
            with self.assertRaises(ar.AssessmentLimitError):r.authorize_resume(self.diagnosis(r))
            self.assertEqual(r.snapshot()['deadline'],deadline)

    def test_confirmation_records_completed_work_even_when_time_expires(self):
        r=self.runtime()
        with r.writer():
            self.model(r);t=r.precharge_forward(case_count=3,case_ids_sha256='a'*64)
            self.clock.now=r.snapshot()['deadline']+1
            with self.assertRaises(ar.AssessmentLimitError):r.confirm_forward(t)
            self.assertEqual(r.snapshot()['confirmed_cases'],3)

    def test_resource_observation_retains_the_limit_breach(self):
        for i,(field,value) in enumerate([('combined_rss_bytes',4*ar.GIB+1),('mps_bytes',2*ar.GIB+1),('free_bytes',8*ar.GIB-1)]):
            r=self.runtime(self.plan(str(i)))
            with r.writer():
                m=dict(self.measure);m[field]=value
                with self.assertRaises(ar.AssessmentLimitError):r.check_resources(**m)
                s=r.snapshot();self.assertEqual(s['failure']['kind'],'RESOURCE')
                if field=='mps_bytes':self.assertEqual(s['peak_mps_bytes'],value)
                if field=='free_bytes':self.assertEqual(s['minimum_free_bytes'],value)

    def test_read_and_output_caps_precede_file_operations(self):
        with patch.dict(ar.LIMITS,max_read_bytes=2,max_output_bytes=ar.BOOKKEEPING_BYTES+3):
            r=self.runtime()
            with r.writer():
                with self.assertRaises(ar.AssessmentLimitError):r.write('too-big.bin',b'abcd')
                self.assertFalse((r.directory/'artifacts/too-big.bin').exists())
            r=self.runtime(self.plan('reads'))
            with r.writer():
                r.reserve_read(2,numeric=False)
                with self.assertRaises(ar.AssessmentLimitError):r.reserve_read(1,numeric=False)
                self.assertEqual(r.snapshot()['read_reserved'],2)

    def test_atomic_artifact_hash_commit_and_reuse(self):
        r=self.runtime()
        with r.writer():
            d=r.write('input/chunk.bin',b'synthetic',numeric=False)
            self.assertEqual(set(d),{'path','bytes','sha256'})
            self.assertEqual(r.read(d,numeric=False),b'synthetic')
            c=r.commit('chunk-1',{'data':d},metadata={'source':'fixture'})
            self.assertEqual(r.reuse('chunk-1',expected_metadata={'source':'fixture'}),c)
            with self.assertRaises(ValueError):r.write('input/chunk.bin',b'replacement',numeric=False)
            self.assertIsNone(r.snapshot()['numeric_started_at'])

    def test_completed_model_is_not_reconstructed(self):
        r=self.runtime()
        with r.writer():
            j=self.model(r);c=self.finish(r,j)
            self.assertEqual(r.reuse(j),c)
            with self.assertRaises(ValueError):r.precharge_model(j)
            self.assertEqual(r.snapshot()['entered_models'],1)

    def test_changed_or_missing_artifact_stops_reuse(self):
        for i,action in enumerate(['corrupt','remove']):
            r=self.runtime(self.plan(str(i)))
            with r.writer():
                d=r.write('saved.bin',b'synthetic');r.commit('saved',[d]);p=r.directory/d['path']
                if action=='corrupt':p.write_bytes(b'corrupted')
                else:p.unlink()
                with self.assertRaises(ar.AssessmentIntegrityError):r.reuse('saved')
                self.assertEqual(r.snapshot()['failure']['kind'],'INTEGRITY')

    def test_commit_identity_metadata_conflict_is_terminal(self):
        r=self.runtime()
        with r.writer():
            d=r.write('meta.bin',b'fixture');r.commit('saved',[d],metadata={'input':'first'})
            with self.assertRaises(ar.AssessmentIntegrityError):r.reuse('saved',expected_metadata={'input':'other'})
            self.assertEqual(r.snapshot()['failure']['kind'],'INTEGRITY')

    def test_failed_write_charge_and_partial_file_are_preserved(self):
        r=self.runtime()
        with r.writer():
            before=r.snapshot()['output_reserved']
            with patch.object(ar.os,'link',side_effect=OSError('Synthetic filesystem interruption')):
                with self.assertRaises(OSError):r.write('partial.bin',b'abc')
            self.assertEqual(r.snapshot()['output_reserved'],before+3)
            self.assertTrue((r.directory/'artifacts/partial.bin.writing').exists())
            self.assertEqual(r.con.execute('SELECT status FROM outputs').fetchone()[0],'FAILED')
            with self.assertRaises(ValueError):r.write('partial.bin',b'abc')

    def test_prediction_barrier_requires_all_models_and_comparator(self):
        r=self.runtime(self.plan(required_prediction_chunks=['NWM_PREDICTIONS']))
        with r.writer():
            with self.assertRaises(ValueError):r.prediction_barrier()
            for j in self.jobs:self.model(r,j);self.finish(r,j)
            with self.assertRaisesRegex(ValueError,'comparator'):r.prediction_barrier()
            d=r.write('nwm.bin',b'synthetic-comparator');r.commit('NWM_PREDICTIONS',[d])
            barrier=r.prediction_barrier();self.assertEqual(len(barrier['metadata']['model_commit_sha256_by_job']),36)
            before=r.snapshot()['entered_models'];self.assertEqual(r.prediction_barrier(),barrier)
            self.assertEqual(r.snapshot()['entered_models'],before)

    def test_nonboolean_phase_flag_cannot_hide_a_numeric_write(self):
        r=self.runtime()
        with r.writer():
            with self.assertRaises(ValueError):r.write('bad.bin',b'bad',numeric=0)
            self.assertFalse((r.directory/'artifacts/bad.bin').exists())
            self.assertIsNone(r.snapshot()['numeric_started_at'])


if __name__=='__main__':unittest.main()
