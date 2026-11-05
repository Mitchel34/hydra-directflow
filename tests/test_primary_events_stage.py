"""Event orchestration on synthetic local artifacts; no owned values or models."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow.cache import DerivedCache
from hydra_directflow.local_data import USGS_MANIFEST_SHA256
from hydra_directflow.reference import Observation, ASSESSMENT_START, digest
from hydra_directflow.trainer_bundle import _array_digest
from test_post_events import fixture, iso

PATH = Path(__file__).resolve().parents[1] / 'scripts/postprocess_primary_events.py'
SPEC = importlib.util.spec_from_file_location('tested_primary_events_stage', PATH)
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


class FakeGuard:
    def __init__(self, run):
        self.run = run
        self.charges = []
        self.writes = []
        self.checks = 0

    def check(self):
        self.checks += 1

    def save(self):
        pass

    def read_budget(self, size):
        self.charges.append(size)

    def read(self, path, maximum):
        raw = Path(path).read_bytes()
        if len(raw) > maximum:
            raise ValueError('synthetic read bound')
        self.read_budget(len(raw))
        return raw

    def write(self, relative, raw):
        path = self.run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == raw
        else:
            path.write_bytes(raw)
        self.writes.append(relative)
        return dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class EventStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / 'postprocessing-v1'
        self.run.mkdir()
        self.guard = FakeGuard(self.run)
        self.plan = dict(run_directory=str(self.run), plan_sha256='a'*64,
                         original_plan_sha256='b'*64, protocol_sha256='c'*64,
                         protected_assessment_access=False)

    def tearDown(self):
        self.temp.cleanup()

    def cache_fixture(self, month='2023-01'):
        root = self.root / 'resident-reference'
        root.mkdir(exist_ok=True)
        (root/'entries').mkdir(exist_ok=True)
        identity = dict(month=month, source_manifest_sha256=USGS_MANIFEST_SHA256)
        key = DerivedCache.key('usgs_month', identity)
        directory = root/'entries'/key
        directory.mkdir()
        payloads = {}
        for kind, suffix, raw in [('arrays','.npz',b'synthetic arrays never decoded'),
                                  ('metadata','.json',b'{"synthetic":true}')]:
            sha = hashlib.sha256(raw).hexdigest()
            name = sha+suffix
            (directory/name).write_bytes(raw)
            payloads[kind] = dict(name=name, bytes=len(raw), sha256=sha)
        commit = dict(schema='hydra-derived-cache/v1', kind='usgs_month', key=key,
                      identity=identity, payloads=payloads)
        raw = json.dumps(commit).encode()
        (directory/'COMMIT.json').write_bytes(raw)
        item = dict(month=month, cache_key=key, commit_sha256=hashlib.sha256(raw).hexdigest())
        return DerivedCache(root), item, directory, commit

    def group_fixture(self, *, fold='final_development', gauge='03161000'):
        row, args, reference = fixture()
        row['fold'], row['gauge_id'] = fold, gauge
        start = int(reference['times'][0])
        n = len(reference['times'])-18
        origins = start+np.arange(n,dtype=np.int64)*3600
        future = origins[:,None]+np.arange(1,19,dtype=np.int64)*3600
        common = np.zeros(n,bool)
        indices = ((args['origins']-start)//3600).astype(np.int64)
        common[indices] = True
        labels = np.full((n,18),np.nan)
        labels[indices] = args['labels']
        predicted = np.full((1,n,18),np.nan)
        predicted[0,indices] = args['predictions']['model-seed17']
        job_id = f'{fold}__{gauge}__LSTM__full_L168__seed17'
        arrays = dict(origins=origins, forecast_time=future, comparison_mask=common,
                      panel_ids=np.array([job_id]), labels=labels,
                      label_mask=np.broadcast_to(common[:,None],labels.shape).copy(),predictions=predicted)
        panels = [dict(job_id=job_id,architecture='LSTM',input_arm='full',history_hours=168,seed=17,device='cpu')]
        manifest = dict(schema='hydra-postprocessing-development-group/v1',fold=fold,gauge_id=gauge,
                         plan_sha256=self.plan['original_plan_sha256'],protocol_sha256=self.plan['protocol_sha256'],
                         calendar_hours=n,calendar_start_utc=iso(start),calendar_end_exclusive_utc=iso(start+n*3600),
                         panels=panels,comparison_mask_sha256=digest(common.tolist()),
                         group_arrays_sha256=_array_digest(arrays),admitted_origins=int(common.sum()),
                         protected_assessment_access=False,target_imputation=False,negative_predictions_clipped=False,
                         quality_policy='PRIMARY_APPROVED_A_OPTIONAL_R_V1')
        manifest['content_sha256']=digest(manifest)
        name=f'{fold}__{gauge}'
        stream=io.BytesIO();np.savez_compressed(stream,**arrays)
        descriptors=dict(arrays=self.guard.write(f'groups/{name}/arrays.npz',stream.getvalue()),
                         manifest=self.guard.write(f'groups/{name}/manifest.json',stage._json(manifest)),
                         audit=self.guard.write(f'groups/{name}/audit.json',stage._json({'status':'ALL_RAW_PREDICTIONS_AND_PHYSICAL_METRICS_MATCH','protected_assessment_access':False})))
        record=dict(group=name,fold=fold,gauge_id=gauge,plan_sha256=self.plan['plan_sha256'],artifacts=descriptors)
        self.guard.write(f'groups/{name}/COMMIT.json',stage._json(record))
        return row,arrays,reference,record,manifest

    def test_reference_preflight_authenticates_files_and_precharges_helper_reads(self):
        cache,item,directory,commit=self.cache_fixture()
        with patch.object(stage,'reference_month',return_value=([],item)) as load:
            rows,receipt=stage.checked_reference_month(cache,item,self.guard)
        self.assertEqual(rows,[]);load.assert_called_once_with(cache,item['month'],item['cache_key'])
        size=(directory/'COMMIT.json').stat().st_size
        expected=3*size+sum(x['bytes'] for x in commit['payloads'].values())
        self.assertEqual(receipt['helper_read_bytes_reserved'],expected)
        self.assertEqual(self.guard.charges,[size,expected])

    def test_reference_preflight_rejects_outside_scope_bad_sha_and_symlink_before_values(self):
        cache,item,directory,commit=self.cache_fixture()
        for change in ('month','sha','symlink'):
            changed=dict(item)
            if change=='month':changed['month']='2018-07'
            elif change=='sha':changed['commit_sha256']='0'*64
            else:
                path=directory/commit['payloads']['arrays']['name'];raw=path.read_bytes();path.unlink()
                target=self.root/'outside.npz';target.write_bytes(raw);path.symlink_to(target)
            with self.subTest(change=change),patch.object(stage,'reference_month',side_effect=AssertionError('Values opened')):
                with self.assertRaises(ValueError):stage.checked_reference_month(cache,changed,self.guard)

    def test_actual_july_guard_rejects_missing_development_export_before_payload_load(self):
        cache,item,directory,commit=self.cache_fixture('2024-07')
        with patch.object(cache,'_load',side_effect=AssertionError('Numeric payload opened')):
            with self.assertRaisesRegex(ValueError,'July requires'):
                stage.checked_reference_month(cache,item,self.guard)

    def test_original_qualifiers_sources_and_rejections_are_preserved(self):
        start=1672531200
        source=[Observation('03161000',start,None,2.,('R','A','A'),'source-0'),
                Observation('03161000',start+3600,None,3.,('P',),'source-1'),
                Observation('03161000',start+7200,None,4.,('A','e'),'source-2'),
                Observation('03161000',start+10800,None,None,('A',),'source-3')]
        out=stage.build_reference_calendars(source,start=start,end=start+6*3600)['03161000']
        self.assertEqual(out['admitted'].tolist(),[True,False,False,False,False,False])
        self.assertEqual(out['quality_flags_json'][0],'["R","A","A"]')
        self.assertEqual(out['quality_flags_json'][4],'null')
        self.assertEqual(out['source_id'][1],'source-1')
        self.assertTrue(np.isnan(out['values'][1:]).all())
        self.assertEqual(out['statuses'][1],'PROVISIONAL_REFERENCE_REJECTED')
        self.assertEqual(out['statuses'][2],'ESTIMATED_REFERENCE_REJECTED')
        self.assertEqual(out['statuses'][4],'NO_EXACT_HOUR_RECORD')
        self.assertFalse(out['row_present'][4])
        self.assertTrue(all(not v.flags.writeable for v in out.values()))

    def test_reference_calendar_rejects_duplicates_and_protected_rows(self):
        start=1672531200
        row=Observation('03161000',start,None,2.,('A',),'s')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            stage.build_reference_calendars([row,row],start=start,end=start+24*3600)
        bad=Observation('03161000',ASSESSMENT_START,None,2.,('A',),'s')
        with self.assertRaisesRegex(ValueError,'Protected reference'):
            stage.build_reference_calendars([bad],start=start,end=start+24*3600)
        foreign=Observation('99999999',start,None,2.,('A',),'s')
        with self.assertRaisesRegex(ValueError,'gauge scope'):
            stage.build_reference_calendars([foreign],start=start,end=start+24*3600)

    def test_group_load_reproduces_hashed_full_calendar_and_rejects_bytes_changed(self):
        row,arrays,reference,record,manifest=self.group_fixture()
        returned,meta=stage.load_group(self.plan,self.guard,record)
        self.assertEqual(meta,manifest)
        for key in arrays:np.testing.assert_array_equal(returned[key],arrays[key])
        path=self.run/record['artifacts']['arrays']['path'];raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'artifact bytes changed'):
            stage.load_group(self.plan,self.guard,record)

    def test_reference_join_checks_all_targets_peaks_and_reference_counts(self):
        row,arrays,reference,record,manifest=self.group_fixture()
        result=stage.corroborate_reference(row,arrays,reference)
        self.assertTrue(result['exact_reference_values'])
        self.assertEqual(result['all_catalog_peaks'],len(row['observed_high_flow_events']))
        with self.assertRaisesRegex(ValueError,'forecast calendar differs'):
            stage.corroborate_reference(row,{**arrays,'origins':arrays['origins'][1:]},reference)
        for change in ('target','peak','mask'):
            altered={k:v.copy() for k,v in reference.items()}
            if change=='target':
                where=int((arrays['forecast_time'][arrays['comparison_mask']][0,0]-reference['times'][0])//3600)
                altered['values'][where]+=1
            elif change=='peak':
                # Largest peak is intentionally absent from matched predictions.
                where=int((row['observed_high_flow_events'][0]['peak_epoch']-reference['times'][0])//3600)
                altered['values'][where]+=1
            else:
                altered['admitted'][20]=False;altered['values'][20]=np.nan
            with self.subTest(change=change),self.assertRaises(ValueError):
                stage.corroborate_reference(row,arrays,altered)

    def test_full_stage_saves_fixed_catalog_and_complete_observed_reference_calendar(self):
        records=[];rows=[];references={}
        for fold in stage.FOLDS:
            for gauge in stage.GAUGES:
                row,arrays,reference,record,manifest=self.group_fixture(fold=fold,gauge=gauge)
                records.append(record);rows.append(row)
                reference['statuses']=np.where(reference['admitted'],'APPROVED_REFERENCE','NO_EXACT_HOUR_RECORD').astype('<U48')
                reference['quality_flags_json']=np.where(reference['admitted'],'["A"]','null')
                reference['source_id']=np.where(reference['admitted'],'synthetic-source','')
                reference['row_present']=reference['admitted'].copy()
                references[gauge]=reference
        audit=dict(plan_sha256=self.plan['plan_sha256'],status='ALL180_PREDICTIONS_INDEPENDENTLY_AUDITED',groups=records)
        self.guard.write('AUDIT.json',stage._json(audit))
        catalog=dict(fold_regime_and_event_diagnostics=rows,quality_policy='PRIMARY_APPROVED_A_OPTIONAL_R_V1')
        identity=dict(month='2023-01',cache_key='d'*64,commit_sha256='e'*64)
        receipts=[dict(identity=identity,helper_read_bytes_reserved=100,preflight_commit_bytes=10,source_files=3)]
        with patch.object(stage,'load_catalog',return_value=catalog),patch.object(stage,'load_references',return_value=(references,receipts)):
            result=stage.run_events(self.plan,self.guard)
        self.assertEqual(len(result['groups']),12)
        self.assertFalse(result['thresholds_recomputed'])
        self.assertEqual(result['new_downloads'],0)
        first=result['groups'][0]
        events=json.loads((self.run/first['artifacts']['events']['path']).read_text())
        self.assertEqual(len(events['representative_event_ids']),3)
        self.assertFalse(events['events'][0]['observed_peak_supported_any_lead'])
        with np.load(self.run/first['artifacts']['reference']['path'],allow_pickle=False) as saved:
            self.assertEqual(len(saved['times']),700)
            self.assertTrue(np.isnan(saved['values'][250:256]).all())
            # Observed largest peak is retained even when no forecast exists.
            self.assertEqual(saved['values'][300],20.)
        self.assertTrue((self.run/'EVENTS.json').exists())

    def test_stage_requires_complete_audit_before_reference_loads(self):
        self.guard.write('AUDIT.json',stage._json(dict(plan_sha256=self.plan['plan_sha256'],status='INCOMPLETE',groups=[])))
        with patch.object(stage,'load_references',side_effect=AssertionError('References opened')):
            with self.assertRaisesRegex(ValueError,'All declared prediction groups'):
                stage.run_events(self.plan,self.guard)


if __name__=='__main__':
    unittest.main()
