import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import nullcontext, ExitStack
from types import SimpleNamespace
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('assessment_score_controller_tested',ROOT/'scripts/run_assessment_scores.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class OwnedReaderTests(unittest.TestCase):
    def setUp(self):
        self.source=m.NwmSourceQualification(source_id='synthetic',snapshot_sha256='a'*64,parity_receipt_sha256='b'*64,
                                            units=m.UNITS,representation=m.DECODED_REPRESENTATION)
    def row(self,origin,lead=3,flow='1.5'):
        return dict(feature_id='6892192',reference_time=origin-7200,valid_time=origin+(lead-2)*3600,
                    lead_hours=str(lead),ensemble='0',streamflow=flow,source_forecast_offset='opaque original')
    def test_first_last_shift_and_unsupported(self):
        first,last=m.ASSESSMENT_START,m.ASSESSMENT_END-3600
        rows=[self.row(first),self.row(last,18,'-2.25')]
        a,r=m.normalize_owned_rows(rows,self.source);q=a['03161000']
        self.assertEqual(q['prediction_q'][0,0],1.5);self.assertEqual(q['prediction_q'][-1,15],-2.25)
        self.assertEqual(q['source_row'][-1,15],2)
        self.assertFalse(q['source_valid'][:,16:].any());self.assertTrue(np.isnan(q['prediction_q'][:,16:]).all())
        self.assertEqual(q['source_forecast_offset'][0,0],'opaque original');self.assertEqual(r['source_rows'],2)
    def test_ignored_scope_never_accesses_flow(self):
        class Guard(dict):
            def __getitem__(self,k):
                if k=='streamflow':raise AssertionError('Out-of-scope flow read')
                return super().__getitem__(k)
        rows=[Guard(self.row(m.ASSESSMENT_START,lead=1)),Guard(self.row(m.ASSESSMENT_END))]
        a,r=m.normalize_owned_rows(rows,self.source)
        self.assertEqual(r['dispositions']['OUTSIDE_MATCHED_ORIGIN_NATIVE_LEAD_SCOPE'],2)
        self.assertFalse(any(x['source_valid'].any() for x in a.values()))
    def test_duplicate_rejected_without_deduplication(self):
        row=self.row(m.ASSESSMENT_START)
        with self.assertRaisesRegex(ValueError,'Duplicate owned'):m.normalize_owned_rows([row,row],self.source)
    def test_null_nonfinite_and_invalid_preserved(self):
        rows=[self.row(m.ASSESSMENT_START+i*3600,flow=x) for i,x in enumerate(('', 'nan', 'bad'))]
        a,_=m.normalize_owned_rows(rows,self.source);q=a['03161000']
        self.assertFalse(q['source_valid'].any())
        self.assertEqual(q['raw_streamflow'][2,0],'bad');self.assertTrue((q['source_row'][:3,0]>0).all())
    def test_shifted_valid_time_fails(self):
        row=self.row(m.ASSESSMENT_START);row['valid_time']+=3600
        with self.assertRaises(ValueError):m.normalize_owned_rows([row],self.source)



class MemoryRuntime:
    """Synthetic hash-committed byte descriptors; no filesystem or source reads."""
    def __init__(self,plan):self.plan=plan;self.blobs={};self.arrays={};self.commits={};self.events=[]
    def descriptor(self,path,raw):
        self.blobs[path]=raw;return dict(path=path,bytes=len(raw),sha256=m.first.sha(raw))
    def store_array(self,path,array):
        raw=path.encode();self.arrays[raw]=array;return self.descriptor(path,raw)
    def store_json(self,path,value):return self.descriptor(path,m.first.encoded(value))
    def read(self,d,**kw):
        self.events.append(('read',d['path']));return self.blobs[d['path']]
    def reuse(self,key,expected_metadata=None):
        c=self.commits.get(key)
        if c is not None and expected_metadata is not None:
            if c['metadata']!=expected_metadata:raise ValueError('Completed metadata changed')
        return c
    def write(self,path,raw,**kw):return self.descriptor(path,raw)
    def commit(self,key,artifacts,metadata=None,kind=None,job_id=None):
        c=dict(key=key,artifacts=artifacts,metadata=metadata,kind=kind or 'chunk',job_id=job_id)
        self.commits[key]=c;return c
    def check(self):pass
    def writer(self):return nullcontext(self)
    def reserve_read(self,*args,**kw):self.events.append(('reserve',kw['purpose']))
    def prediction_barrier(self):return self.structural


def semantic_fixture():
    from hydra_directflow.assessment_data import AssessmentReadAuthority
    jobs=[]
    for gauge in m.GAUGES:
        for arch in m.ARCHITECTURES:
            for seed in m.SEEDS:
                jid=f'final_development__{gauge}__{arch}__full_L168__seed{seed}'
                jobs.append(dict(job_id=jid,gauge_id=gauge,architecture=arch,seed=seed,input_arm='full',history_hours=168,
                    feature_schema=m.FULL_FEATURE_SCHEMA,device='cpu' if arch in ('LSTM','GRU_Transformer') else 'mps',
                    selected_weights='best_model',selected_best_epoch=4,
                    checkpoint=dict(path=jid+'.pt',bytes=100,sha256='a'*64),training_source_identity='5'*64,
                    training_cohort_admission_digest='6'*64,gradient_case_ids_sha256='7'*64,original_training_receipt_sha256='8'*64))
    p=dict(run_directory=str(m.first.RUN),input_source_identity='9'*64,
           checkpoint_sha256_by_job={j['job_id']:j['checkpoint']['sha256'] for j in jobs},
           json_sources={'protocol':{'sha256':'2'*64},'selection':{'sha256':'3'*64}})
    p['plan_sha256']=m.digest(p)
    a=AssessmentReadAuthority(p['plan_sha256'],'2'*64,'3'*64,'4'*64,inputs_enabled=True)
    runtime=MemoryRuntime(p);calendar=np.asarray(m.origins(),dtype=np.int64)
    future=calendar[:,None]+np.arange(1,19)*3600;historytime=calendar[:,None]-np.arange(169,1,-1)*3600
    history=np.ones((8760,168,2),dtype=np.float64);history_mask=np.ones_like(history,dtype=bool)
    history[-1,0,0]=np.nan;history_mask[-1,0,0]=False
    weather=np.ones((8760,18,4),dtype=np.float64);weather_mask=np.ones_like(weather,dtype=bool)
    state=np.ones((8760,2),dtype=np.float64);state_mask=np.ones_like(state,dtype=bool)
    admitted=np.ones(8760,dtype=bool);admitted[-1]=False
    bundles={};prediction_arrays={};receipts={}
    for si,g in enumerate(m.GAUGES):
        arrays=dict(origin_time=calendar,history_time=historytime,forecast_time=future,
            case_id=np.asarray([g+':'+m.iso(int(t)) for t in calendar]),site_id=np.full(8760,si,dtype=np.int64),
            input_admitted=admitted,history=history,history_mask=history_mask,weather=weather,weather_mask=weather_mask,
            state=state,state_mask=state_mask,feature_schema=np.asarray(m.FULL_FEATURE_SCHEMA),
            cohort_admission_digest=np.asarray('b'*64),inference_source_identity=np.asarray(p['input_source_identity']))
        typed=m.array_identity(arrays)
        manifest=dict(arrays_sha256=typed,gauge_id=g,source_identity=p['input_source_identity'],
            binding=dict(execution_plan_sha256=p['plan_sha256']),cohort_admission_digest='b'*64,
            feature_schema=m.FULL_FEATURE_SCHEMA,requested_origins=8760,admitted_origins=8759)
        bundle=dict(arrays=runtime.store_array(g+'.inputs',arrays),manifest=runtime.store_json(g+'.manifest',manifest))
        bundles[g]=bundle
        prediction=np.ones((8760,18),dtype=np.float64);prediction[-1]=np.nan
        z={k:arrays[k] for k in ('origin_time','case_id','forecast_time','site_id','input_admitted')};z['prediction_q']=prediction
        hashes=[m.digest(arrays['case_id'][i:min(i+64,8759)].tolist()) for i in range(0,8759,64)]
        for j in [x for x in jobs if x['gauge_id']==g]:
            jid=j['job_id'];batches=[]
            for bi,start in enumerate(range(0,8759,64)):
                batches.append(dict(job_id=jid,checkpoint_sha256='a'*64,gauge_id=g,architecture=j['architecture'],seed=j['seed'],
                    selected_state='best_model',selected_best_epoch=4,execution_plan_sha256=p['plan_sha256'],selection_sha256=a.selection_sha256,
                    inference_source_identity=p['input_source_identity'],inference_parent_cohort_admission_digest='b'*64,
                    input_case_ids_sha256=hashes[bi],case_count=min(64,8759-start),target_values_read=False,target_masks_read=False,
                    optimizer_updates=0,scaler_fits=0))
            receipt=dict(job=j,predictions_array_sha256=m.array_identity(z),input_arrays_sha256=typed,
                input_artifact=bundle['arrays'],forecast_origins=8759,candidate_origins=8760,target_values_assembled=False,
                optimizer_updates=0,scaler_fits=0,checkpoint_sha256='a'*64,batches=batches)
            artifacts=dict(predictions=runtime.store_array(jid+'.predictions',z),receipt=runtime.store_json(jid+'.receipt',receipt))
            runtime.commit(jid,artifacts,kind='model',job_id=jid)
            prediction_arrays[jid]=z;receipts[jid]=receipt
    inventory=dict(input_source_identity=p['input_source_identity'],gauge_bundles=bundles)
    runtime.commit('INPUTS',dict(inventory=runtime.store_json('INPUTS',inventory)),metadata=dict(source_identity=p['input_source_identity']))
    structural=dict(artifacts=dict(barrier=dict(path='STRUCTURAL',bytes=1,sha256='c'*64)),
        metadata=dict(model_commit_sha256_by_job={j['job_id']:m.digest(runtime.commits[j['job_id']]) for j in jobs}))
    runtime.structural=structural
    return runtime,p,dict(selection=dict(records=jobs)),a,structural,prediction_arrays,receipts


class SemanticBarrierTests(unittest.TestCase):
    def test_all36_semantics_persist_separate_barrier(self):
        runtime,p,sources,authority,structural,_,_=semantic_fixture()
        with patch.object(m.first,'arrays_from_bytes',side_effect=lambda raw:runtime.arrays[raw]):
            barrier=m.semantic_prediction_barrier(runtime,p,sources,authority,structural)
            again=m.semantic_prediction_barrier(runtime,p,sources,authority,structural)
        self.assertEqual(again,barrier)
        self.assertEqual(len(barrier['metadata']['models']),36)
        self.assertEqual(barrier['metadata']['structural_prediction_barrier_sha256'],'c'*64)
        self.assertFalse(barrier['metadata']['verification_values_read'])
        self.assertEqual(barrier['key'],'SEMANTIC_PREDICTION_BARRIER')
        self.assertNotEqual(barrier['artifacts']['barrier']['sha256'],'c'*64)

    def assert_preflight_blocks_role_open(self,mutate):
        runtime,p,sources,authority,structural,predictions,receipts=semantic_fixture()
        mutate(runtime,p,sources,authority,structural,predictions,receipts)
        def authenticate(*args):
            self.assertIn(('reserve','BOUNDED_SCORING_AUTHENTICATION'),runtime.events)
            runtime.events.append(('authenticate',));return p,sources,authority
        with ExitStack() as stack:
            stack.enter_context(patch.object(m.first,'resident',return_value=m.first.encoded(p)))
            stack.enter_context(patch.object(m.first,'authenticate',side_effect=authenticate))
            stack.enter_context(patch.object(m,'AssessmentRuntime',return_value=runtime))
            stack.enter_context(patch.object(m,'release',return_value={}))
            stack.enter_context(patch.object(m.first,'arrays_from_bytes',side_effect=lambda raw:runtime.arrays[raw]))
            nwm=stack.enter_context(patch.object(m,'get_nwm',side_effect=AssertionError('NWM opened before semantic barrier')))
            targets=stack.enter_context(patch.object(m,'get_targets',side_effect=AssertionError('Targets opened before semantic barrier')))
            with self.assertRaises(ValueError):m.run('/synthetic/plan',p['plan_sha256'])
            nwm.assert_not_called();targets.assert_not_called()

    def test_missing_prediction_rejects_before_any_outcome_reader(self):
        def mutate(runtime,p,sources,a,barrier,predictions,receipts):
            key=sources['selection']['records'][0]['job_id'];predictions[key]['prediction_q'][0,0]=np.nan
        self.assert_preflight_blocks_role_open(mutate)

    def test_changed_coordinates_reject_before_any_outcome_reader(self):
        def mutate(runtime,p,sources,a,barrier,predictions,receipts):
            key=sources['selection']['records'][0]['job_id'];z=predictions[key]
            bad=dict(z,forecast_time=z['forecast_time']+3600)
            runtime.arrays[runtime.blobs[runtime.commits[key]['artifacts']['predictions']['path']]]=bad
        self.assert_preflight_blocks_role_open(mutate)

    def test_missing_family_seed_rejects_before_any_outcome_reader(self):
        def mutate(runtime,p,sources,a,barrier,predictions,receipts):sources['selection']['records'].pop()
        self.assert_preflight_blocks_role_open(mutate)

    def test_unadmitted_finite_predictions_reject_before_any_outcome_reader(self):
        def mutate(runtime,p,sources,a,barrier,predictions,receipts):
            key=sources['selection']['records'][0]['job_id'];predictions[key]['prediction_q'][-1,0]=0.
        self.assert_preflight_blocks_role_open(mutate)

    def test_changed_selected_job_receipt_rejects_before_any_outcome_reader(self):
        def mutate(runtime,p,sources,a,barrier,predictions,receipts):
            key=sources['selection']['records'][0]['job_id'];r=receipts[key]
            r['job']=dict(r['job'],selected_best_epoch=5)
            runtime.blobs[runtime.commits[key]['artifacts']['receipt']['path']]=m.first.encoded(r)
        self.assert_preflight_blocks_role_open(mutate)

    def test_changed_input_source_rejects_before_any_outcome_reader(self):
        def mutate(runtime,p,sources,a,barrier,predictions,receipts):
            arrays=runtime.arrays[(m.GAUGES[0]+'.inputs').encode()]
            arrays['inference_source_identity']=np.asarray('d'*64)
        self.assert_preflight_blocks_role_open(mutate)

    def test_reordered_forward_receipt_rejects_before_any_outcome_reader(self):
        def mutate(runtime,p,sources,a,barrier,predictions,receipts):
            key=sources['selection']['records'][0]['job_id'];r=receipts[key]
            r['batches'][0],r['batches'][1]=r['batches'][1],r['batches'][0]
            runtime.blobs[runtime.commits[key]['artifacts']['receipt']['path']]=m.first.encoded(r)
        self.assert_preflight_blocks_role_open(mutate)

    def test_successful_role_transition_uses_semantic_barrier_identity(self):
        runtime,p,sources,authority,structural,_,_=semantic_fixture()
        def authenticate(*args):
            self.assertIn(('reserve','BOUNDED_SCORING_AUTHENTICATION'),runtime.events)
            return p,sources,authority
        def nwm(*args):
            self.assertIsNotNone(runtime.reuse('SEMANTIC_PREDICTION_BARRIER'))
            return {}
        def targets(actual_runtime,actual_plan,actual_authority,barrier_sha):
            semantic=runtime.reuse('SEMANTIC_PREDICTION_BARRIER')
            self.assertEqual(barrier_sha,semantic['artifacts']['barrier']['sha256'])
            self.assertNotEqual(barrier_sha,structural['artifacts']['barrier']['sha256'])
            raise RuntimeError('Qualified role transition reached without source reads')
        with patch.object(m.first,'resident',return_value=m.first.encoded(p)), \
             patch.object(m.first,'authenticate',side_effect=authenticate), \
             patch.object(m,'AssessmentRuntime',return_value=runtime), \
             patch.object(m,'release',return_value={}), \
             patch.object(m.first,'arrays_from_bytes',side_effect=lambda raw:runtime.arrays[raw]), \
             patch.object(m,'get_nwm',side_effect=nwm),patch.object(m,'get_targets',side_effect=targets):
            with self.assertRaisesRegex(RuntimeError,'Qualified role transition'):m.run('/synthetic/plan',p['plan_sha256'])

    def test_raw_zero_offset_is_preserved(self):
        source=m.NwmSourceQualification(source_id='synthetic',snapshot_sha256='a'*64,parity_receipt_sha256='b'*64,
            units=m.UNITS,representation=m.DECODED_REPRESENTATION)
        c=m.ASSESSMENT_START
        row=dict(feature_id='6892192',reference_time=c-7200,valid_time=c+3600,lead_hours='3',
                 ensemble='0',streamflow='0',source_forecast_offset=0)
        arrays,_=m.normalize_owned_rows([row],source)
        self.assertEqual(arrays['03161000']['source_forecast_offset'][0,0],'0')
        self.assertEqual(arrays['03161000']['prediction_q'][0,0],0.)


if __name__=='__main__':unittest.main()
