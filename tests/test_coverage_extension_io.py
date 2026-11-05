"""Synthetic durable-preparation and restore checks; no source/model execution."""
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from hydra_directflow import coverage_extension_data as data
from hydra_directflow import coverage_extension_io as storage
from hydra_directflow.inference import input_only_view
from hydra_directflow.local_data import GAUGES,iso
from hydra_directflow.reference import ASSESSMENT_START,digest
from hydra_directflow.trainer_bundle import _array_digest
from test_corpus_dataset import arrays_for


class MemoryStore:
    def __init__(self):self.payloads={};self.events=[];self.fail_kind=None;self.reads=[]
    def write(self,relative,raw,*,kind):
        self.events.append(kind)
        if kind==self.fail_kind:raise RuntimeError('SYNTHETIC_DURABILITY_FAILURE')
        path='attempt-000004/'+relative
        if path in self.payloads:assert self.payloads[path]==raw
        self.payloads[path]=raw
        return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def read(self,descriptor,maximum):
        self.reads.append(descriptor['path']);return self.payloads[descriptor['path']]
    def replace(self,descriptor,raw):
        self.payloads[descriptor['path']]=raw
        return dict(path=descriptor['path'],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


class IOTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.start=data.START;self.n=3
        for module in (data,storage):
            self.stack.enter_context(patch.object(module,'END',self.start+self.n*3600))
            self.stack.enter_context(patch.object(module,'CANDIDATE_ORIGINS',self.n))
        cycles,absent=data.ds._candidate_cycles(self.start,self.start+2*3600)
        chunk=dict(index=0,origin_count=self.n,first_origin_utc=iso(self.start),last_origin_utc=iso(self.start+2*3600),
                   cycle_ids=cycles,calendar_unavailable_cycle_ids=absent)
        a=arrays_for(chunk)
        for name in ('target','target_mask','eligibility_reason'):a.pop(name)
        a['source_calendar_status']=data.ds._calendar_status(a,chunk)
        a['weather_cache_key']=np.full(9,'a'*64);a['soil_cache_key']=np.full(9,'b'*64)
        for name in ('last_original_admissible_time','last_original_present_time'):a[name]=a['origin_time']-7200
        for name in ('last_original_admissible_age_hours','last_original_present_age_hours'):a[name]=np.full(9,2,np.int64)
        a['source_chunk_index']=np.zeros(9,np.int32);a['source_row_index']=np.arange(9,dtype=np.int32)
        source='c'*64
        self.candidates=dict(arrays=a,routing=data.classify_inputs(a),manifest=dict(source_members=[dict(source_chunk='synthetic/000',commit_sha256='d'*64,rows=list(range(9)))]),candidate_source_identity=source,source_root_identity=source)
        self.plan=dict(plan_sha256='e'*64,corpus_root='SYNTHETIC_NOT_OPENED',reference_cache_root='SYNTHETIC_NOT_OPENED',protected_assessment_access=False)
        self.common={g:[g+':'+iso(self.start+i*3600) for i in range(2)] for g in GAUGES}
        self.store=MemoryStore()

    def tearDown(self):self.stack.close()

    def prepare(self,reconciliation=None):
        reconciliation=reconciliation or dict(reference_month_identities=[],original_common_case_ids_by_gauge=self.common)
        with patch.object(storage,'load_candidates',return_value=self.candidates) as load, \
             patch.object(storage,'validate_candidates',return_value={'status':'SYNTHETIC_VALIDATED_ONCE'}) as validate:
            result=storage.prepare_inputs(self.plan,reconciliation,self.store)
        load.assert_called_once();validate.assert_called_once_with(self.candidates)
        return result

    def rewrite_input_arrays(self,descriptor,change,*,geometry_change=False):
        d=deepcopy(descriptor)
        with np.load(io.BytesIO(self.store.payloads[d['arrays']['path']]),allow_pickle=False) as source:
            arrays={k:source[k].copy() for k in source.files}
        change(arrays);buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
        d['arrays']=self.store.replace(d['arrays'],buffer.getvalue())
        wrapper=json.loads(self.store.payloads[d['manifest']['path']]);wrapper['arrays']=d['arrays']
        if geometry_change:wrapper['geometry']=storage._geometry(arrays)
        d['manifest']=self.store.replace(d['manifest'],storage._json(wrapper));return d

    def test_durable_route_barrier_precedes_common_selector_and_every_bundle(self):
        owner=self
        class Reconciliation(dict):
            def __getitem__(self,key):
                if key=='original_common_case_ids_by_gauge':
                    owner.assertIn('INPUT_ONLY_ROUTE_BARRIER',owner.store.events)
                return super().__getitem__(key)
        result=self.prepare(Reconciliation(reference_month_identities=[],original_common_case_ids_by_gauge=self.common))
        i=self.store.events.index('INPUT_ONLY_ROUTE_BARRIER')
        self.assertEqual(self.store.events[:i].count('INPUT_ONLY_ROUTE_ARRAYS'),3)
        self.assertEqual(self.store.events[:i].count('INPUT_ONLY_ROUTE_MANIFEST'),3)
        self.assertNotIn('INPUT_BUNDLE_ARRAYS',self.store.events[:i])
        barrier=json.loads(self.store.payloads[result['route_barrier']['path']])
        self.assertFalse(barrier['original_common_case_ids_consulted'])
        self.assertEqual(digest(barrier),result['route_barrier_content_sha256'])
        self.assertEqual(len(result['bundles']),3)
        self.assertEqual(self.store.events.count('INPUT_BUNDLE_ARRAYS'),12)
        self.assertTrue(all(value['arrays']['path'].startswith('attempt-000004/') for g in result['bundles'].values() for h in g.values() for value in h.values()))

    def test_failed_barrier_write_prevents_regression_selection(self):
        self.store.fail_kind='INPUT_ONLY_ROUTE_BARRIER'
        with patch.object(storage,'load_candidates',return_value=self.candidates), \
             patch.object(storage,'validate_candidates',return_value={}),patch.object(storage,'input_bundle') as views:
            with self.assertRaisesRegex(RuntimeError,'DURABILITY'):
                storage.prepare_inputs(self.plan,dict(reference_month_identities=[]),self.store)
        views.assert_not_called()

    def test_six_expanded_and_six_clean_views_restore_with_new_truthful_identities(self):
        result=self.prepare()
        for gauge in GAUGES:
            for length in (168,24):
                for kind,n in (('expanded',3),('clean',2)):
                    d=result['bundles'][gauge][str(length)][kind];b=storage.read_input_bundle(self.store,d)
                    self.assertEqual(len(b['arrays']['origin_time']),n)
                    self.assertEqual(b['arrays']['history'].shape,(n,length,2))
                    self.assertEqual(b['source_identity'],d['source_identity'])
                    self.assertEqual(b['candidate_source_identity'],self.candidates['candidate_source_identity'])
                    mapped=input_only_view(b['arrays'],inference_source_identity=b['source_identity'])
                    np.testing.assert_array_equal(mapped['history'],b['arrays']['history'])
                    self.assertTrue(all(not x.flags.writeable for x in b['arrays'].values()))
                self.assertNotEqual(result['bundles'][gauge][str(length)]['expanded']['cohort_admission_digest'],result['bundles'][gauge][str(length)]['clean']['cohort_admission_digest'])

    def test_restored_routes_preserve_full_calendar_and_hierarchy(self):
        result=self.prepare()
        for gauge in GAUGES:
            d=result['routes'][gauge];restored=storage.read_route(self.store,d)
            self.assertEqual(restored['arrays']['route'].tolist(),['L168']*3)
            self.assertEqual(restored['arrays']['forecast_time'].shape,(3,18))
            self.assertEqual(restored['route_map_sha256'],d['route_map_sha256'])
            self.assertTrue(all(not v.flags.writeable for v in restored['arrays'].values()))

    def test_protected_input_coordinates_fail_before_numeric_members_open(self):
        d=self.prepare()['bundles'][GAUGES[0]]['168']['expanded']
        d=self.rewrite_input_arrays(d,lambda a:a['forecast_time'].__setitem__((0,17),ASSESSMENT_START))
        real_open=zipfile.ZipFile.open;seen=[]
        def check(z,name,*args,**kwargs):
            name=name.filename if isinstance(name,zipfile.ZipInfo) else name;seen.append(name)
            if name in ('history.npy','weather.npy','state.npy'):raise AssertionError('Input values opened before boundary check')
            return real_open(z,name,*args,**kwargs)
        with patch.object(zipfile.ZipFile,'open',new=check):
            with self.assertRaisesRegex(ValueError,'clocks differ|Protected'):
                storage.read_input_bundle(self.store,d)
        self.assertNotIn('history.npy',seen)

    def test_target_members_or_wrong_numeric_dtype_refused(self):
        for change,geometry in ((lambda a:a.update(target=np.zeros((3,18))),False),
                                (lambda a:a.update(history=a['history'].astype(np.float32)),True)):
            with self.subTest(geometry=geometry):
                self.store=MemoryStore();d=self.prepare()['bundles'][GAUGES[0]]['168']['expanded']
                d=self.rewrite_input_arrays(d,change,geometry_change=geometry)
                with self.assertRaisesRegex(ValueError,'inventory|geometry'):
                    storage.read_input_bundle(self.store,d)

    def test_modified_new_only_input_is_detected_even_with_rehashed_archive(self):
        d=self.prepare()['bundles'][GAUGES[0]]['168']['expanded']
        d=self.rewrite_input_arrays(d,lambda a:a['history'].__setitem__((2,-1,0),105.))
        with self.assertRaisesRegex(ValueError,'typed input values changed'):storage.read_input_bundle(self.store,d)

    def test_descriptor_hash_source_scope_and_view_identity_are_checked(self):
        result=self.prepare();base=result['bundles'][GAUGES[0]]['168']['expanded']
        for key,value in (('gauge_id',GAUGES[1]),('history_hours',24),('phase_plan_sha256','f'*64),('source_identity','0'*64)):
            changed=deepcopy(base);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):storage.read_input_bundle(self.store,changed)
        changed=deepcopy(base);changed['arrays']['sha256']='0'*64
        with self.assertRaises(ValueError):storage.read_input_bundle(self.store,changed)

    def test_rehashed_route_cannot_override_fixed_hierarchy(self):
        d=deepcopy(self.prepare()['routes'][GAUGES[0]])
        with np.load(io.BytesIO(self.store.payloads[d['arrays']['path']]),allow_pickle=False) as archive:a={k:archive[k].copy() for k in archive.files}
        a['route'][0]='L24';raw=storage._npz(a);d['arrays']=self.store.replace(d['arrays'],raw)
        m=json.loads(self.store.payloads[d['manifest']['path']]);m['arrays']=d['arrays'];m['arrays_sha256']=_array_digest(a)
        d['route_map_sha256']=digest(dict(candidate_source_identity=d['candidate_source_identity'],gauge_id=d['gauge_id'],routing_arrays_sha256=m['arrays_sha256']))
        m['route_map_sha256']=d['route_map_sha256'];m['content_sha256']=digest({k:v for k,v in m.items() if k!='content_sha256'})
        d['manifest']=self.store.replace(d['manifest'],storage._json(m));d['route_counts']={'L168':2,'L24':1,'UNAVAILABLE':0}
        with self.assertRaisesRegex(ValueError,'fixed route hierarchy'):storage.read_route(self.store,d)

    def test_route_scope_change_or_unbound_array_body_fails(self):
        base=self.prepare()['routes'][GAUGES[0]]
        changed=deepcopy(base);changed['phase_plan_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'source/scope'):storage.read_route(self.store,changed)
        path=base['arrays']['path'];self.store.payloads[path]+=b'bad'
        with self.assertRaisesRegex(ValueError,'bytes differ'):storage.read_route(self.store,base)


if __name__=='__main__':unittest.main()
