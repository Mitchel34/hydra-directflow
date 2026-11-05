"""Synthetic input-only routing/provenance qualification; no real values/models."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from hydra_directflow import coverage_extension_data as ext, corpus, corpus_dataset as ds
from hydra_directflow.cache import DerivedCache, pack_month
from hydra_directflow.inference import input_only_view
from hydra_directflow.local_data import GAUGES, USGS_MANIFEST_SHA256, iso
from hydra_directflow.reference import Observation, digest, ASSESSMENT_START
from hydra_directflow.training import array_identity
from test_corpus_dataset import arrays_for


class ExtensionDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.stack=ExitStack();self.count=3;self.start=ext.START
        self.stack.enter_context(patch.object(ext,'END',self.start+self.count*3600))
        self.stack.enter_context(patch.object(ext,'CANDIDATE_ORIGINS',self.count))
        self.stack.enter_context(patch.object(ext,'SOURCE_PLAN','a'*64))
        self.stack.enter_context(patch.object(ext,'SNAPSHOT','b'*64))
        self.corpus=self.root/'corpus';self.cache=self.root/'cache'
        (self.corpus/'chunks').mkdir(parents=True);(self.corpus/'staging').mkdir()
        cycles,outside=ds._candidate_cycles(self.start,self.start+(self.count-1)*3600)
        self.chunk=dict(index=0,origin_count=self.count,first_origin_utc=iso(self.start),
                        last_origin_utc=iso(self.start+(self.count-1)*3600),cycle_ids=cycles,calendar_unavailable_cycle_ids=outside)
        self.unit=dict(id='synthetic_validation__2023-07__L168',history_hours=168,
                       split_end_exclusive_utc=iso(ext.END+18*3600))
        self.native=arrays_for(self.chunk)
        self.month='2023-07';self.bad_old=self.start-100*3600;self.bad_recent=self.start-2*3600
        self.month_spec=self.write_month()
        self.write_shard()

    def tearDown(self):self.stack.close();self.temp.cleanup()

    def write_month(self):
        start=int(datetime(2023,7,1,tzinfo=timezone.utc).timestamp());end=int(datetime(2023,8,1,tzinfo=timezone.utc).timestamp())
        observations=[]
        for stamp in range(start,end,3600):
            for gi,gauge in enumerate(GAUGES):
                flags=('P',) if gi==1 and stamp==self.bad_old else ('unknown',) if gi==2 and stamp==self.bad_recent else ('R','A','A') if gi==0 and stamp==self.bad_recent else ('A',)
                observations.append(Observation(gauge,stamp,None,100.,flags,'SYNTHETIC_SOURCE_'+gauge))
        identity=dict(month=self.month,source_manifest_sha256=USGS_MANIFEST_SHA256,fixture='SYNTHETIC')
        cache=DerivedCache(self.cache)
        with cache.locked():
            arrays,metadata,key,_=cache.get_or_build('usgs_month',identity,lambda:pack_month(self.month,observations,{}))
        commit=(self.cache/'entries'/key/'COMMIT.json').read_bytes()
        return dict(month=self.month,cache_key=key,commit_sha256=hashlib.sha256(commit).hexdigest())

    def write_shard(self):
        a=self.native
        for gi,stamp in ((1,self.bad_old),(2,self.bad_recent)):
            rows,slots=np.where((a['site_id'][:,None]==gi)&(a['history_time']==stamp))
            a['history'][rows,slots]=np.nan;a['history_mask'][rows,slots]=False
        complete=np.ones(len(a['origin_time']),bool)
        for field in ('history','weather','state','target'):complete&=a[field+'_mask'].reshape(len(complete),-1).all(axis=1)
        a['eligibility_reason']=np.where(complete,'COMPLETE','SYNTHETIC_MISSING_HISTORY')
        receipt=dict(case_source_digest=digest('SYNTHETIC'),cache=dict(cycle_keys={c:digest(c) for c in self.chunk['cycle_ids']},
                     month_keys={self.month:self.month_spec['cache_key']}),stage_seconds={})
        corpus._shard_write(self.corpus,ext.SOURCE_PLAN,self.unit,self.chunk,a,receipt)
        self.shard=self.corpus/'chunks'/self.unit['id']/'000';self.refresh_commit()

    def refresh_commit(self):
        self.observed=dict(chunk=self.unit['id']+'/000',commit_sha256=hashlib.sha256((self.shard/'COMMIT.json').read_bytes()).hexdigest())

    def authority(self):
        return dict(root=self.corpus,descriptors=[(self.unit,self.chunk,self.observed)],plan_file_sha256='c'*64)

    def load(self,**kwargs):
        with patch.object(ext,'_authority',return_value=self.authority()),patch.object(ds,'inspect_corpus',side_effect=AssertionError('Full audit replay')):
            return ext.load_candidates(self.corpus,reference_cache_root=self.cache,reference_month_identities=[self.month_spec],**kwargs)

    def mutate_shard(self, change, summary_change=None):
        with np.load(self.shard/'model.npz',allow_pickle=False) as source:a={k:source[k].copy() for k in source.files}
        change(a);stream=io.BytesIO();np.savez_compressed(stream,**a);raw=stream.getvalue();(self.shard/'model.npz').write_bytes(raw)
        commit=json.loads((self.shard/'COMMIT.json').read_text());commit['arrays']={k:dict(shape=list(v.shape),dtype=v.dtype.str) for k,v in a.items()}
        commit['payloads']['model.npz']=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if summary_change:
            summary=json.loads((self.shard/'case-index.json').read_text());summary_change(summary)
            body=json.dumps(summary).encode();(self.shard/'case-index.json').write_bytes(body)
            commit['payloads']['case-index.json']=dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
        (self.shard/'COMMIT.json').write_text(json.dumps(commit));self.refresh_commit()

    def test_input_only_complete_calendar_fixed_routes_and_provenance(self):
        candidate=self.load();a=candidate['arrays'];route=candidate['routing']
        self.assertEqual(len(a['origin_time']),9)
        self.assertEqual(route['route'].tolist(),['L168','L24','UNAVAILABLE']*3)
        self.assertTrue(np.all(~route['eligible168']|route['eligible24']))
        self.assertFalse(any(k in a for k in ext.FORBIDDEN_MEMBERS))
        self.assertFalse(candidate['manifest']['target_quality_members_decoded'])
        self.assertFalse(candidate['manifest']['monthly_source_values_decoded'])
        self.assertIn(['R','A','A'],candidate['manifest']['provenance_dictionaries']['qualities'])
        self.assertEqual(a['last_original_admissible_time'][2],self.start-3*3600)
        self.assertEqual(a['last_original_present_time'][2],self.start-2*3600)
        self.assertEqual(route['L168_longest_missing_run'][1],1)
        self.assertEqual(route['L168_longest_gap_oldest_age_hours'][1],100)
        self.assertEqual(route['L168_longest_gap_first_time'][1],self.bad_old)
        self.assertEqual(route['L24_missing_hours'][1],0)
        self.assertTrue(all(not value.flags.writeable for value in a.values()))

    def test_forbidden_archive_members_never_opened_and_values_member_never_decoded(self):
        old_open=zipfile.ZipFile.open;seen=[]
        def open_checked(archive,name,*args,**kwargs):
            name=name.filename if isinstance(name,zipfile.ZipInfo) else name;seen.append(name)
            if name in {x+'.npy' for x in ext.FORBIDDEN_MEMBERS}|{'values.npy','source_calendar_status.npy'}:
                raise AssertionError('Forbidden member opened: '+name)
            return old_open(archive,name,*args,**kwargs)
        with patch.object(zipfile.ZipFile,'open',new=open_checked):self.load()
        self.assertIn('history.npy',seen);self.assertIn('quality_index.npy',seen)

    def test_poison_targets_masks_mixed_flags_change_identity_not_route_or_mock_prediction(self):
        clean=self.load()
        def prediction(candidate):
            b=ext.input_bundle(candidate,GAUGES[1],24)['arrays']
            return b['history'][:,-1,0,None]+b['weather'].mean(axis=2)+b['state'][:,0,None]
        before=prediction(clean)
        def change(a):
            a['target'][:]=np.nan;a['target_mask'][:]=False;a['admission_mask'][:]=False;a['eligibility_reason'][:]='POISON_OUTCOME'
            a['source_calendar_status'][:]='POISON_MIXED_STATUS'
        self.mutate_shard(change,lambda s:s.update(admitted_cases=-999,eligibility_counts={'POISON':999},source_calendar_counts={}))
        poisoned=self.load()
        np.testing.assert_array_equal(clean['routing']['route'],poisoned['routing']['route'])
        np.testing.assert_array_equal(before,prediction(poisoned))
        self.assertNotEqual(clean['candidate_source_identity'],poisoned['candidate_source_identity'])

    def test_exact_L24_view_and_new_cohort_metadata_are_predictor_compatible(self):
        c=self.load();bundle=ext.input_bundle(c,GAUGES[1],24);a=bundle['arrays']
        self.assertEqual(a['history'].shape,(3,24,2));self.assertTrue(a['history_mask'].all())
        np.testing.assert_array_equal(a['history_time'],a['origin_time'][:,None]-np.arange(25,1,-1)*3600)
        copied=input_only_view(a,inference_source_identity=bundle['source_identity'])
        np.testing.assert_array_equal(copied['history'],a['history'])
        self.assertEqual(bundle['manifest']['input_values_sha256'],array_identity({k:a[k] for k in ext.INFERENCE_KEYS}))
        subset=ext.input_bundle(c,GAUGES[1],24,case_ids=(str(a['case_id'][0]),))
        self.assertEqual(subset['candidate_source_identity'],bundle['candidate_source_identity'])
        self.assertNotEqual(subset['cohort_admission_digest'],bundle['cohort_admission_digest'])
        self.assertNotEqual(subset['source_identity'],bundle['source_identity'])
        np.testing.assert_array_equal(subset['arrays']['history'],a['history'][:1])

    def test_regression_subset_cannot_override_eligibility_gauge_or_order(self):
        c=self.load()
        with self.assertRaisesRegex(ValueError,'not independently input eligible'):
            ext.input_bundle(c,GAUGES[1],168,case_ids=(GAUGES[1]+':'+iso(self.start),))
        with self.assertRaises(ValueError):ext.input_bundle(c,GAUGES[0],24,case_ids=(GAUGES[1]+':'+iso(self.start),))
        ids=ext.input_bundle(c,GAUGES[0],24)['arrays']['case_id'].tolist()
        with self.assertRaises(ValueError):ext.input_bundle(c,GAUGES[0],24,case_ids=tuple(reversed(ids)))

    def test_weather_or_soil_failure_prevents_both_routes_without_target_consultation(self):
        c=self.load();a={k:v.copy() for k,v in c['arrays'].items()}
        a['weather_mask'][0,0,0]=False;a['weather'][0,0,0]=np.nan
        a['state_mask'][3]=False;a['state'][3]=np.nan
        a['soil_source_cycle'][6]=iso(int(a['origin_time'][6])-12*3600)[:10].replace('-','')+'_'+iso(int(a['origin_time'][6])-12*3600)[11:13]+'Z'
        a['state'][6,1]=12.
        r=ext.classify_inputs(a)
        self.assertTrue(all(r['route'][i]=='UNAVAILABLE' for i in (0,3,6)))
        self.assertFalse(r['eligible24'][[0,3,6]].any())

    def test_missing_weather_never_becomes_zero_precipitation(self):
        c=self.load();a={k:v.copy() for k,v in c['arrays'].items()}
        a['weather_mask'][0,0,0]=False;a['weather'][0,0,0]=0.
        with self.assertRaisesRegex(ValueError,'value/mask semantics'):ext.classify_inputs(a)

    def test_observed_features_without_source_identity_are_not_hidden_as_unavailability(self):
        c=self.load()
        for field in ('source_cycle','soil_source_cycle'):
            a={k:v.copy() for k,v in c['arrays'].items()};a[field][0]=''
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'lacks an original source'):
                ext.classify_inputs(a)

    def test_future_input_cycle_and_bad_clock_fail_before_numeric_inputs(self):
        c=self.load();a={k:v.copy() for k,v in c['arrays'].items()}
        a['history_time'][0,0]+=3600;a['history']=object()
        with self.assertRaisesRegex(ValueError,'Exact causal history'):ext.classify_inputs(a)
        a={k:v.copy() for k,v in c['arrays'].items()};a['source_cycle'][0]='20230731_00Z'
        with self.assertRaisesRegex(ValueError,'Future/unavailable weather'):ext.classify_inputs(a)

    def test_protected_coordinate_and_gauge_change_rejected_before_input_values(self):
        def change(a):a['site_id'][0]=2
        self.mutate_shard(change)
        with patch.object(ds,'_decode',wraps=ds._decode) as decode:
            with self.assertRaisesRegex(ValueError,'gauge/origin order'):self.load()
            self.assertFalse(any(call.args[2] in ext.INPUT_FIELDS for call in decode.call_args_list))
        c=arrays_for(self.chunk);c['source_calendar_status']=np.array(['RESOLVED_WITHIN_ACQUIRED_CALENDAR']*9)
        c['origin_time'][:]=ASSESSMENT_START-3600
        c['history_time']=c['origin_time'][:,None]-np.arange(169,1,-1)*3600
        c['target_time']=c['origin_time'][:,None]+np.arange(1,19)*3600
        # One gauge/case per origin for this boundary probe.
        c={k:v[:3] for k,v in c.items()};c['case_id']=np.array([g+':'+iso(ASSESSMENT_START-3600) for g in GAUGES]);c['history']=object()
        with patch.object(ext,'END',ASSESSMENT_START):
            with self.assertRaisesRegex(ValueError,'safe H18'):ext.classify_inputs(c)

    def test_history_slot_quality_policy_keeps_A_R_and_rejects_unknown(self):
        c=self.load();dictionary=c['manifest']['provenance_dictionaries']['qualities']
        index=c['arrays']['history_quality_index'][0,-1]
        self.assertEqual(dictionary[index],['R','A','A'])
        self.assertTrue(c['arrays']['history_mask'][0,-1,0])
        self.assertFalse(c['arrays']['history_mask'][2,-1,0])

    def test_qualifier_or_source_checksum_mismatch_is_technical_failure(self):
        changed=dict(self.month_spec,commit_sha256='0'*64)
        with patch.object(ext,'_authority',return_value=self.authority()):
            with self.assertRaisesRegex(ValueError,'source SHA differs'):
                ext.load_candidates(self.corpus,reference_cache_root=self.cache,reference_month_identities=[changed])
        self.mutate_shard(lambda a:(a['history_mask'].__setitem__((1,69),True),a['history'].__setitem__((1,69),[100.,100.])))
        with self.assertRaises(ValueError):self.load()

    def test_read_precharge_precedes_open_and_no_full_audit_or_month_value_reader(self):
        def refuse(_):raise RuntimeError('SYNTHETIC_BUDGET_EXHAUSTED')
        with patch.object(ds,'_read',side_effect=AssertionError('Body opened before allowance')):
            with self.assertRaisesRegex(RuntimeError,'BUDGET'):self.load(read_budget=refuse)
        with patch.object(DerivedCache,'_load',side_effect=AssertionError('All-member monthly values reader forbidden')):
            self.load()

    def test_partial_source_columns_do_not_read_unused_quality_values(self):
        values=np.arange(3*24,dtype=np.int64).reshape(3,24);stream=io.BytesIO();np.savez_compressed(stream,quality_index=values)
        with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
            selected=ext._selected_npy(archive,'quality_index',dict(shape=[3,24],dtype='<i8'),(5,9))
        np.testing.assert_array_equal(selected,values[:,5:9]);self.assertEqual(selected.size,12)

    def test_route_bundle_has_truthful_shared_origin_identity_and_separate_reasons(self):
        c=self.load();view=ext.route_bundle(c,GAUGES[1]);a=view['arrays']
        self.assertEqual(a['route'].tolist(),['L24']*3);self.assertEqual(a['primary_reason'].tolist(),['OLDER_Q_HISTORY_GAP_USE_L24']*3)
        self.assertTrue(a['non_q_eligible'].all());self.assertFalse(a['eligible168'].any());self.assertTrue(a['eligible24'].all())
        self.assertEqual(a['forecast_time'].shape,(3,18));self.assertEqual(len(view['route_map_sha256']),64)
        self.assertTrue(all(not v.flags.writeable for v in a.values()))

    def test_restored_candidates_reject_changed_new_only_input_or_route(self):
        c=self.load();self.assertEqual(ext.validate_candidates(c)['status'],'VERIFIED_RESTORED_INPUT_ONLY_CANDIDATES')
        for kind in ('input','route','manifest'):
            copy=dict(c,arrays={k:v.copy() for k,v in c['arrays'].items()},routing={k:v.copy() for k,v in c['routing'].items()},manifest=deepcopy(c['manifest']))
            if kind=='input':copy['arrays']['history'][1,-1,0]+=1 # Older-gap L24-only row, not original common input.
            elif kind=='route':copy['routing']['route'][1]='L168'
            else:copy['manifest']['source_root_identity']='0'*64
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'Restored candidate'):
                ext.validate_candidates(copy)

    def test_restored_coordinate_gate_precedes_hashing_numeric_members(self):
        c=self.load();copy=dict(c,arrays={k:v.copy() for k,v in c['arrays'].items()})
        copy['arrays']['target_time'][0,-1]=ASSESSMENT_START
        copy['arrays']['history']=object()
        with self.assertRaisesRegex(ValueError,'calendar/history/H18'):
            ext.validate_candidates(copy)


if __name__=='__main__':unittest.main()
