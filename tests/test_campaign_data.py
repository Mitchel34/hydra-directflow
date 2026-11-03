"""Primary loader tests use only synthetic sealed shards; zero models/fits."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow import campaign_data as campaign, corpus, corpus_dataset as ds
from hydra_directflow.ablations import q_only_view,Q_HISTORY_FEATURES
from hydra_directflow.contract import ARCHITECTURES
from hydra_directflow.local_data import GAUGES,iso
from hydra_directflow.reference import digest,ASSESSMENT_START
from hydra_directflow.training import validate_development_bundle
import test_pilot_assembly as fixtures
from test_corpus_dataset import arrays_for


class CampaignDataTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack(); self.fixture=fixtures.PilotAssemblyTests('test_metadata_plan_never_loads_numeric_payloads'); self.fixture.setUp()
        self.root=self.fixture.root; self.corpus=self.fixture.corpus
        self.roster_path=self.root/'roster.json'
        self.roster=dict(schema='hydra-local-seed-sequence/v3',fits=180,source_plan_sha256=self.fixture.sourceplan['plan_sha256'],
            snapshot_sha256=self.fixture.kwargs['expected_snapshot_sha256'],protected_assessment_access=False,role_windows={},jobs=[])
        for fold in ('cv_1','cv_2','cv_3','final_development'):
            windows={}
            for short,long in campaign.ROLE_NAMES.items():
                windows[long]=dict(start_utc=self.fixture.roles[short]['start_utc'],end_exclusive_utc=self.fixture.roles[short]['end_utc'],
                    simulated_label_delay_hours=2,refit_at_utc=None if short=='outer' else self.fixture.refit)
            self.roster['role_windows'][fold]=windows
            for seed in ((17,29,43) if fold=='final_development' else (17,)):
                for gauge in GAUGES:
                    for architecture in ARCHITECTURES:
                        for arm,length in ([('full',168),('q_only',168),('full',24)] if fold=='final_development' else [('full',168),('q_only',168)]):
                            self.roster['jobs'].append(dict(job_id=f'{fold}__{gauge}__{architecture}__{arm}_L{length}__seed{seed}',fold=fold,gauge_id=gauge,
                                architecture=architecture,input_arm=arm,history_hours=length,seed=seed,batch_size=64,
                                actual_common_cases={long:self.fixture.expected[gauge][short]['cases'] for short,long in campaign.ROLE_NAMES.items()},
                                role_case_ids_sha256={long:self.fixture.expected[gauge][short]['case_ids_sha256'] for short,long in campaign.ROLE_NAMES.items()}))
        self.stack.enter_context(patch.object(campaign,'SOURCE_PLAN',self.fixture.sourceplan['plan_sha256']))
        self.stack.enter_context(patch.object(campaign,'INTEGRITY_PATH',self.fixture.receipt))
        self.stack.enter_context(patch.object(campaign,'ROSTER_PATH',self.roster_path))
        self.repin()

    def tearDown(self): self.stack.close(); self.fixture.tearDown()

    def repin(self):
        self.roster_path.write_text(json.dumps(self.roster))
        self.integrity_hash=hashlib.sha256(self.fixture.receipt.read_bytes()).hexdigest()
        self.stack.enter_context(patch.object(campaign,'INTEGRITY_SHA256',self.integrity_hash))
        self.stack.enter_context(patch.object(campaign,'SNAPSHOT',self.roster['snapshot_sha256']))
        self.stack.enter_context(patch.object(campaign,'ROSTER_SHA256',hashlib.sha256(self.roster_path.read_bytes()).hexdigest()))

    def job(self,fold='cv_1',arm='full',length=168,gauge=GAUGES[0],architecture='LSTM',seed=17):
        return next(x for x in self.roster['jobs'] if (x['fold'],x['input_arm'],x['history_hours'],x['gauge_id'],x['architecture'],x['seed'])
                    ==(fold,arm,length,gauge,architecture,seed))

    def load(self,job=None,**changes):
        return campaign.load_fit_bundle(self.corpus,job or self.job(),expected_integrity_hash=self.integrity_hash,**changes)

    def outer(self,job=None):
        return campaign.load_outer_inputs(self.corpus,job or self.job(),expected_integrity_hash=self.integrity_hash)

    def test_fit_inner_only_exact_roles_values_and_readonly(self):
        bundle=self.load(); arrays=bundle['arrays']
        self.assertEqual(len(arrays['case_id']),11); self.assertEqual(bundle['train_indices'],tuple(range(6)))
        self.assertEqual(bundle['inner_indices'],tuple(range(6,11))); self.assertEqual(len(bundle['gradient_case_ids']),6)
        self.assertTrue(np.all(arrays['target_time'][list(bundle['inner_indices']),-1]+7200<=ds._hour(self.fixture.refit)))
        self.assertTrue(np.all(arrays['origin_time']<ds._hour(self.fixture.roles['outer']['start_utc'])))
        np.testing.assert_equal(arrays['history'][0,:,0],[fixtures.value(t) for t in arrays['history_time'][0]])
        self.assertTrue(all(not x.flags.writeable for x in arrays.values()))
        self.assertEqual(validate_development_bundle(arrays,168),11)
        self.assertFalse(bundle['manifest']['full_corpus_inspection_replayed'])
        self.assertEqual(bundle['manifest']['unassigned_cases'],0)

    def test_q_only_never_decodes_weather_or_soil_and_matches_existing_ablation(self):
        full=self.load(); original=ds._decode
        def guard(archive,commit,name,shape):
            if name in ('weather','state'): raise AssertionError('HRRR numeric member decoded')
            return original(archive,commit,name,shape)
        with patch.object(ds,'_decode',side_effect=guard): q=self.load(self.job(arm='q_only'))
        self.assertEqual(full['cohort_admission_digest'],q['cohort_admission_digest'])
        a=full['arrays']; view=q_only_view({**a,'case_ids':tuple(a['case_id'].tolist()),'history_features':Q_HISTORY_FEATURES},
                                        cohort_admission_digest=full['cohort_admission_digest'])
        for field in ('history','history_mask','weather','weather_mask','state','state_mask','target','target_mask'):
            np.testing.assert_equal(q['arrays'][field],getattr(view,field))
        self.assertEqual(q['arrays']['feature_schema'].item(),ds.Q_ONLY_FEATURE_SCHEMA)
        self.assertNotEqual(full['source_identity'],q['source_identity'])

    def test_all_four_folds_and_final_l24_common_membership(self):
        for fold in ('cv_1','cv_2','cv_3','final_development'):
            bundle=self.load(self.job(fold=fold)); self.assertEqual(len(bundle['arrays']['case_id']),11)
        long=self.load(self.job(fold='final_development')); short=self.load(self.job(fold='final_development',length=24))
        self.assertEqual(short['cohort_admission_digest'],long['cohort_admission_digest'])
        for name in ('history','history_mask','history_time'): np.testing.assert_equal(short['arrays'][name],long['arrays'][name][:,-24:])
        for name in ('target','target_mask','case_id'): np.testing.assert_equal(short['arrays'][name],long['arrays'][name])
        self.assertNotEqual(long['source_identity'],short['source_identity'])

    def test_preflight_only_decodes_coordinates_masks_not_numeric_values(self):
        original=ds._decode
        def guard(archive,commit,name,shape):
            if name in ('history','weather','state','target'): raise AssertionError('numeric member decoded')
            return original(archive,commit,name,shape)
        with patch.object(ds,'_decode',side_effect=guard),patch.object(ds,'inspect_corpus',side_effect=AssertionError('full inspection')):
            report=campaign.preflight_plan(self.corpus,self.job(),expected_integrity_hash=self.integrity_hash)
        self.assertEqual(report['case_count'],11); self.assertEqual(report['numeric_value_members_decoded'],[])
        self.assertEqual(report['cohort_admission_digest'],self.load()['cohort_admission_digest'])

    def test_outer_inputs_never_decode_targets_and_separate_labels_align(self):
        original=ds._decode
        def no_labels(archive,commit,name,shape):
            if name=='target': raise AssertionError('label value decoded')
            return original(archive,commit,name,shape)
        with patch.object(ds,'_decode',side_effect=no_labels): outer=self.outer()
        self.assertNotIn('target',outer['arrays']); self.assertNotIn('target_mask',outer['arrays']); self.assertIn('forecast_time',outer['arrays'])
        self.assertFalse(outer['manifest']['target_values_decoded'])
        def only_labels(archive,commit,name,shape):
            if name in ('history','weather','state'): raise AssertionError('input value decoded')
            return original(archive,commit,name,shape)
        with patch.object(ds,'_decode',side_effect=only_labels):
            labels=campaign.load_outer_labels(self.corpus,self.job(),expected_outer_cohort_digest=outer['cohort_admission_digest'],
                expected_case_ids_sha256=outer['case_ids_sha256'],expected_integrity_hash=self.integrity_hash)
        self.assertEqual(outer['cohort_admission_digest'],labels['cohort_admission_digest'])
        np.testing.assert_equal(outer['arrays']['forecast_time'],labels['arrays']['target_time'])
        batches=list(campaign.iter_outer_batches(outer,batch_size=2)); self.assertEqual([len(x['case_id']) for x in batches],[2,2,2])
        self.assertTrue(all(x['cohort_admission_digest'].item()==outer['cohort_admission_digest'] for x in batches))

    def test_q_only_outer_parity_without_placeholder_labels(self):
        full=self.outer(); q=self.outer(self.job(arm='q_only'))
        labels=campaign.load_outer_labels(self.corpus,self.job(),expected_outer_cohort_digest=full['cohort_admission_digest'],
            expected_case_ids_sha256=full['case_ids_sha256'],expected_integrity_hash=self.integrity_hash)
        a=full['arrays']; view=q_only_view({**a,'target_time':a['forecast_time'],'target':labels['arrays']['target'],
            'target_mask':labels['arrays']['target_mask'],'case_ids':tuple(a['case_id'].tolist()),'history_features':Q_HISTORY_FEATURES},
            cohort_admission_digest=full['cohort_admission_digest'])
        for field in ('history','history_mask','weather','weather_mask','state','state_mask'):
            np.testing.assert_equal(q['arrays'][field],getattr(view,field))
        self.assertEqual(q['cohort_admission_digest'],full['cohort_admission_digest']); self.assertNotIn('target',q['arrays'])

    def test_wrong_outer_identity_rejects_before_label_values(self):
        outer=self.outer(); original=ds._decode
        def no_labels(archive,commit,name,shape):
            if name=='target': raise AssertionError('labels accessed before expected identity gate')
            return original(archive,commit,name,shape)
        with patch.object(ds,'_decode',side_effect=no_labels):
            with self.assertRaisesRegex(ValueError,'before target values'):
                campaign.load_outer_labels(self.corpus,self.job(),expected_outer_cohort_digest='0'*64,
                    expected_case_ids_sha256=outer['case_ids_sha256'],expected_integrity_hash=self.integrity_hash)

    def test_wrong_job_or_integrity_hash_rejects_before_shard_read(self):
        wrong=deepcopy(self.job()); wrong['actual_common_cases']['gradient_fit']+=1
        with patch.object(campaign,'_read_shard',side_effect=AssertionError('shard read')):
            with self.assertRaisesRegex(ValueError,'roster entry'): self.load(wrong)
            with self.assertRaisesRegex(ValueError,'integrity'):
                campaign.load_fit_bundle(self.corpus,self.job(),expected_integrity_hash='0'*64)

    def test_wrong_expected_role_count_blocks_before_numeric_values(self):
        self.job()['actual_common_cases']['gradient_fit']+=1; self.repin(); original=ds._decode
        def no_values(archive,commit,name,shape):
            if name in ('history','weather','state','target'): raise AssertionError('numeric value before count gate')
            return original(archive,commit,name,shape)
        with patch.object(ds,'_decode',side_effect=no_values):
            with self.assertRaisesRegex(ValueError,'before numeric values'): self.load()

    def test_boundary_expansion_or_protected_window_rejected(self):
        self.roster['role_windows']['cv_1']['outer_scoring']['end_exclusive_utc']='2024-08-01T00:00:00Z'; self.repin()
        with self.assertRaisesRegex(ValueError,'protected'): self.load()

    def test_source_corruption_and_duplicate_commit_inventory_reject(self):
        unit=self.fixture.sourceplan['units'][0]; target=self.corpus/'chunks'/unit['id']/'000'/'model.npz'
        target.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'size differs'): self.load()
        receipt=json.loads(self.fixture.receipt.read_text()); receipt['observed_commits'][1]=receipt['observed_commits'][0]
        self.fixture.receipt.write_text(json.dumps(receipt)); self.repin()
        with self.assertRaisesRegex(ValueError,'inventory/order'): self.load()

    def test_bounds_and_external_stop_before_source_values(self):
        with patch.object(campaign,'_read_shard',side_effect=AssertionError('source read')):
            with self.assertRaisesRegex(ValueError,'before allocation'): self.load(max_bytes=1)
        with self.assertRaisesRegex(ValueError,'Metadata'): self.load(max_read_bytes=1)
        with self.assertRaisesRegex(RuntimeError,'stop'):
            self.load(check=lambda:(_ for _ in ()).throw(RuntimeError('stop')))

    def test_l24_does_not_resurrect_case_missing_only_older_history(self):
        unit=self.fixture.sourceplan['units'][0]; chunk=unit['chunks'][0]; arrays=arrays_for(chunk)
        arrays['history'][...,0]=np.array([[fixtures.value(t) for t in row] for row in arrays['history_time']])
        arrays['target'][:]=np.array([[fixtures.value(t) for t in row] for row in arrays['target_time']])
        arrays['history'][0,0,:]=np.nan; arrays['history_mask'][0,0,:]=False; arrays['eligibility_reason'][0]='HISTORY_INCOMPLETE'
        directory=self.corpus/'chunks'/unit['id']/'000'; shutil.rmtree(directory)
        corpus._shard_write(self.corpus,self.fixture.sourceplan['plan_sha256'],unit,chunk,arrays,
            dict(case_source_digest=digest('old_missing'),cache={'cycle_keys':{c:digest(c) for c in chunk['cycle_ids']},
                'month_keys':{'2021-08':self.fixture.key}},stage_seconds={'total':.1}))
        snapshot=ds.inspect_corpus(self.corpus,expected_plan_sha256=self.fixture.sourceplan['plan_sha256'])
        self.fixture.receipt.write_text(json.dumps(snapshot.report())); self.roster['snapshot_sha256']=snapshot.snapshot_sha256
        first=ds._hour(chunk['first_origin_utc']); ids=[GAUGES[0]+':'+iso(t) for t in range(first+3600,first+6*3600,3600)]
        for job in self.roster['jobs']:
            if job['gauge_id']==GAUGES[0]: job['actual_common_cases']['gradient_fit']=5; job['role_case_ids_sha256']['gradient_fit']=digest(ids)
        self.repin(); short=self.load(self.job(fold='final_development',length=24)); long=self.load(self.job(fold='final_development'))
        self.assertEqual(len(short['train_indices']),5); self.assertEqual(short['cohort_admission_digest'],long['cohort_admission_digest'])
        self.assertTrue(short['arrays']['history_mask'].all())


if __name__=='__main__': unittest.main()
