"""Synthetic-only independent result-audit qualification; no real artifacts."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_assessment_audit',ROOT/'scripts/audit_assessment_results.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    # Targets share one observed hourly series, including a missing target hour.
    hours=np.arange(m.N+17)
    series=12+4*np.sin(hours/27)+hours/10000
    lookup=np.arange(m.N)[:,None]+np.arange(18)
    y=series[lookup].copy();valid=lookup!=120;y[~valid]=np.nan
    admitted=np.ones(m.N,bool);admitted[:5]=False
    predictions={}
    for i,family in enumerate(m.FAMILIES):
        for seed in m.SEEDS:
            p=series[lookup]+(seed-17)/12+i*.1;p[:5]=np.nan;p[5,0]=-1
            predictions[family,seed]=p.astype(np.float64)
    nw=series[lookup]+.5;available=np.ones((m.N,18),bool)
    available[:,16:]=False;available[1000,0]=False;nw[~available]=np.nan
    return [m.GAUGES[0],y,valid,admitted,predictions,nw,available]


class IndependentMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Production scorer is only used on synthetic fixtures as an oracle
        # to detect schema drift. The audit implementation never imports it.
        from hydra_directflow.assessment_scoring import score_gauge
        cls.args=fixture()
        g,y,v,a,p,n,nv=cls.args
        cls.report=score_gauge(g,m.CALENDAR,y,v,a,p,n,nv)
        cls.recomputed=m.audit_gauge(*cls.args,cls.report)

    def test_independent_full_gauge_recomputation_and_exact_support(self):
        result=self.recomputed
        self.assertEqual(result['coverage']['input_admitted_origins'],8755)
        self.assertEqual(result['coverage']['primary_complete_target_origins'],8737)
        self.assertEqual(result['coverage']['per_lead'][0]['evaluable_origins'],8754)
        self.assertEqual(result['coverage']['per_lead'][0]['nwm_matched_origins'],8736)
        self.assertEqual(result['coverage']['per_lead'][17]['nwm_matched_origins'],0)
        self.assertEqual(result['models']['LSTM']['seeds']['17']['primary']['leads'][0]['negative_predictions_scored'],1)
        self.assertEqual(result['last_target_epoch'],m.END+17*m.HOUR)
        self.assertEqual(result['models']['LSTM']['seed_aggregation']['primary']['seeds'],[17,29,43])

    def test_analytic_signed_error_and_no_clipping(self):
        y=np.array([1.,2.,3.]);p=np.array([-1.,2.,5.]);mask=np.ones(3,bool)
        row=m.metric_row(y,p,np.arange(3)*3600,mask,1)
        self.assertAlmostEqual(row['physical_mse'],8/3)
        self.assertEqual(row['mae'],4/3)
        self.assertEqual(row['bias'],0.)
        self.assertEqual(row['nse'],-3.)
        self.assertEqual(row['negative_frequency'],1/3)
        self.assertEqual(row['pbias'],0.)

    def test_constant_zero_and_singleton_undefined(self):
        v=np.ones(3,bool);t=np.arange(3)*3600
        row=m.metric_row(np.zeros(3),np.ones(3),t,v,1)
        self.assertIsNone(row['nse']);self.assertIsNone(row['pbias']);self.assertIsNone(row['pearson_r'])
        self.assertEqual(row['undefined']['nse'],'CONSTANT_OBSERVED_TARGETS')
        row=m.metric_row(np.array([1.]),np.array([2.]),np.array([0]),np.ones(1,bool),1)
        self.assertEqual(row['undefined']['nse'],'FEWER_THAN_TWO_TARGETS')

    def test_undefined_seed_never_dropped(self):
        self.assertEqual(m.describe([1.,None,3.]),dict(mean=None,min=None,max=None,defined_seeds=2,total_seeds=3,undefined_seeds_omitted=False))
        self.assertEqual(m.describe([1.,2.,3.])['mean'],2.)

    def test_metric_tolerance_is_absolute_plus_relative(self):
        m.compare(1.+.9e-10,1.)
        with self.assertRaisesRegex(ValueError,'float differs'):m.compare(1.+2e-10,1.)
        m.compare(1e6+0.9e-6,1e6)
        with self.assertRaises(ValueError):m.compare(float('nan'),1.)
        with self.assertRaises(ValueError):m.compare(True,1.)

    def test_wrong_aggregate_and_support_rejected(self):
        report=copy.deepcopy(self.report)
        report['models']['LSTM']['seed_aggregation']['primary']['equal_lead_nse']['mean']+=.01
        with self.assertRaisesRegex(ValueError,'float differs'):m.compare(report,self.recomputed)
        report=copy.deepcopy(self.report);report['coverage']['input_admitted_origins']-=1
        with self.assertRaisesRegex(ValueError,'value differs'):m.compare(report,self.recomputed)

    def test_missing_admitted_prediction_fails_without_cohort_shrink(self):
        args=copy.deepcopy(self.args);args[4]['LSTM',17][100,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Missing admitted'):m.audit_gauge(*args,self.report)

    def test_filled_target_fails_even_with_false_quality_mask(self):
        args=copy.deepcopy(self.args);args[1][~args[2]]=0.
        with self.assertRaisesRegex(ValueError,'unfilled NaN'):m.audit_gauge(*args,self.report)

    def test_disagreement_in_repeated_target_hours_fails(self):
        args=copy.deepcopy(self.args);args[1][300,0]+=1
        with self.assertRaisesRegex(ValueError,'Repeated verification'):m.audit_gauge(*args,self.report)

    def test_missing_seed_or_extra_family_fails(self):
        args=copy.deepcopy(self.args);del args[4]['Transformer',29]
        with self.assertRaisesRegex(ValueError,'all three seeds'):m.audit_gauge(*args,self.report)
        report=copy.deepcopy(self.report);report['models']['Other']={}
        with self.assertRaisesRegex(ValueError,'roster'):m.audit_gauge(*self.args,report)

    def test_nwm_unsupported_lead_or_invalid_value_fails(self):
        args=copy.deepcopy(self.args);args[6][0,16]=True;args[5][0,16]=0.
        with self.assertRaisesRegex(ValueError,'NWM source'):m.audit_gauge(*args,self.report)
        args=copy.deepcopy(self.args);args[5][1000,0]=0.
        with self.assertRaisesRegex(ValueError,'NWM source'):m.audit_gauge(*args,self.report)

    def test_public_projection_contains_aggregates_not_source_rows(self):
        expected={g:copy.deepcopy(self.recomputed) for g in m.GAUGES}
        for result in expected.values():
            result['provenance']={'path':'/private/SECRET','uri':'gs://secret'}
            result['raw_target']=[100.,200.]
        public=m.public_projection(expected,{'plan_sha256':'a'*64},{'new_optimizer_updates':0})
        raw=json.dumps(public)
        self.assertNotIn('SECRET',raw);self.assertNotIn('gs://',raw);self.assertNotIn('raw_target',raw)
        self.assertEqual(public['models'],36);self.assertEqual(len(public['gauges']),3)
        self.assertFalse(public['gauges'][m.GAUGES[0]]['models']['LSTM']['seed_aggregation']['primary']['formal_confidence_intervals'])


class SourceProvenanceTests(unittest.TestCase):
    def targets(self):
        shape=(m.N,18);mask=np.zeros(shape,bool);mask[0,0]=True
        return dict(target_time=m.FUTURE.copy(),target_mask=mask,
            target_quality_flags=np.full(shape,'["A", "R"]',dtype='U32'),
            target_source_ids=np.full(shape,'original-source',dtype='U32'),
            target_exclusion_reason=np.where(mask,'','QUALITY_NOT_ADMITTED'))

    def test_original_approved_qualifiers_and_exclusions(self):
        a=self.targets();m.target_provenance(a)
        a['target_quality_flags'][0,0]='["A", "e"]'
        with self.assertRaisesRegex(ValueError,'unapproved'):m.target_provenance(a)
        a=self.targets();a['target_source_ids'][0,0]=''
        with self.assertRaisesRegex(ValueError,'identity'):m.target_provenance(a)
        a=self.targets();a['target_exclusion_reason'][1,0]=''
        with self.assertRaisesRegex(ValueError,'exclusion'):m.target_provenance(a)

    def nwm(self):
        shape=(m.N,18);v=np.zeros(shape,bool);v[0,0]=True
        status=np.full(shape,'ISSUE_OR_LEAD_ABSENT',dtype='U64');status[:,16:]='NATIVE_HORIZON_UNAVAILABLE';status[0,0]='NUMERIC_VALID'
        q=np.full(shape,np.nan);q[0,0]=-1.25
        rows=np.full(shape,-1);rows[0,0]=7
        raw=np.full(shape,'',dtype='U64');raw[0,0]='-1.25'
        return dict(prediction_q=q,source_valid=v,source_row=rows,numeric_status=status,raw_streamflow=raw,
                    source_forecast_offset=np.full(shape,'',dtype='U32'))

    def test_nwm_decoded_negative_retained_no_double_packing(self):
        a=self.nwm();m.nwm_provenance(a)
        a['prediction_q'][0,0]*=.01
        with self.assertRaisesRegex(ValueError,'rescaled'):m.nwm_provenance(a)

    def test_nwm_status_and_row_identity_required(self):
        a=self.nwm();a['source_row'][0,0]=-1
        with self.assertRaisesRegex(ValueError,'status'):m.nwm_provenance(a)
        a=self.nwm();a['numeric_status'][0,1]='NUMERIC_VALID'
        with self.assertRaisesRegex(ValueError,'status'):m.nwm_provenance(a)

    def test_source_budget_independently_reconciles_readonly(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();folder=root/'source-staging/budget';folder.mkdir(parents=True)
            db=folder/'source-budget.sqlite3';c=sqlite3.connect(db)
            c.execute('CREATE TABLE budget(id,approval,body_cap,get_cap,cost_cap_micro,charged_body,attempts)')
            c.execute('CREATE TABLE attempts(reserved,observed,status)')
            c.execute('INSERT INTO budget VALUES(1,?, ?,4000,5000000,1000,1)',('a'*64,16*1024**3))
            c.execute("INSERT INTO attempts VALUES(1000,900,'VERIFIED')");c.commit();c.close()
            class Runtime:
                directory=root
                charges=[]
                def reserve_read(self,n,**kwargs):self.charges.append(n)
            recorded=dict(body_cap=16*1024**3,get_cap=4000,cost_estimate_cap_usd=5.,precharged_body_bytes=1000,
                attempted_gets=1,observed_body_bytes=900,conservative_precharge_estimate_usd=66/1e6,
                attempt_states={'VERIFIED':1},billing_invoice_reconciled=False)
            before=db.read_bytes();m.source_accounting(Runtime(),'a'*64,recorded)
            self.assertEqual(db.read_bytes(),before);self.assertTrue(Runtime.charges)
            with self.assertRaisesRegex(ValueError,'reconcile'):m.source_accounting(Runtime(),'b'*64,recorded)
            c=sqlite3.connect(db);c.execute('UPDATE budget SET charged_body=999');c.commit();c.close()
            with self.assertRaisesRegex(ValueError,'reconcile'):m.source_accounting(Runtime(),'a'*64,recorded)

    def test_npz_refuses_object_arrays(self):
        data=io.BytesIO();np.savez(data,x=np.array([object()],dtype=object))
        with self.assertRaises(ValueError):m.decode_arrays(data.getvalue())


class BarrierTests(unittest.TestCase):
    def test_zero_origin_cohort_still_requires_all36_model_confirmations(self):
        receipts={j:dict(forecast_origins=0) for j in m.MODEL_IDS}
        counters=dict(confirmed_cases=0,confirmed_batches=0,confirmed_models=36,resume_id=0)
        self.assertEqual(m.reconcile_confirmations(counters,receipts),
                         dict(confirmed_cases=0,confirmed_batches=0,confirmed_models=36))
        counters['confirmed_models']=0
        with self.assertRaisesRegex(ValueError,'confirmations'):m.reconcile_confirmations(counters,receipts)
        counters['confirmed_models']=36
        receipts[next(iter(receipts))]['forecast_origins']=65
        with self.assertRaisesRegex(ValueError,'confirmations'):m.reconcile_confirmations(counters,receipts)
        counters.update(confirmed_cases=65,confirmed_batches=2)
        self.assertEqual(m.reconcile_confirmations(counters,receipts)['confirmed_batches'],2)

    def test_scorer_release_authenticates_frozen_code_and_no_learning(self):
        raw=b'synthetic qualified code'
        p=dict(plan_sha256='p',json_sources=dict(protocol=dict(sha256='r'),selection=dict(sha256='s')))
        release=dict(schema='hydra-assessment-point-scoring-release/v1',plan_sha256='p',protocol_sha256='r',selection_sha256='s',
            no_inference=True,no_training=True,no_post_outcome_method_selection=True,formal_confidence_intervals=False,
            implementation={name:m.sha(raw) for name in ('scripts/run_assessment_scores.py','src/hydra_directflow/assessment_scoring.py')})
        class Runtime:
            def reuse(self,key):return dict(metadata=release,artifacts=dict(release=dict(sha256='bound-release')))
            def read(self,*args,**kwargs):return m.encoded(release)
        with patch.object(m,'resident_metadata',return_value=raw):
            self.assertEqual(m.scoring_release(Runtime(),p),'bound-release')
            release['no_training']=False
            with self.assertRaisesRegex(ValueError,'no-training'):m.scoring_release(Runtime(),p)
            release['no_training']=True
        with patch.object(m,'resident_metadata',return_value=b'changed'):
            with self.assertRaisesRegex(ValueError,'code changed'):m.scoring_release(Runtime(),p)

    @staticmethod
    def complete_barrier_fixture():
        class Runtime:
            def __init__(self):self.blobs={};self.commits={};self.read_paths=[]
            def store(self,path,raw):
                self.blobs[path]=raw
                return dict(path=path,bytes=len(raw),sha256=m.sha(raw))
            def jstore(self,path,value):return self.store(path,m.encoded(value))
            def astore(self,path,arrays):
                out=io.BytesIO();np.savez_compressed(out,**arrays)
                return self.store(path,out.getvalue())
            def read(self,d,**kwargs):
                self.read_paths.append(d['path']);raw=self.blobs[d['path']]
                if len(raw)!=d['bytes'] or m.sha(raw)!=d['sha256']:raise ValueError('Synthetic hash mismatch')
                return raw
            def reuse(self,key):return self.commits.get(key)
            def check(self):pass
        runtime=Runtime();p=dict(plan_sha256='a'*64,input_source_identity='b'*64,
            checkpoint_sha256_by_job={j:'c'*64 for j in m.MODEL_IDS})
        products={};gauges={};bundles={};records=[];model_hashes={}
        for si,g in enumerate(m.GAUGES):
            admitted=np.zeros(m.N,bool);admitted[0]=True
            a=dict(origin_time=m.CALENDAR.copy(),forecast_time=m.FUTURE.copy(),
                history_time=m.CALENDAR[:,None]-np.arange(169,1,-1)*m.HOUR,
                case_id=np.array([g+':'+m.datetime.fromtimestamp(int(t),m.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') for t in m.CALENDAR]),
                site_id=np.full(m.N,si),input_admitted=admitted,inference_source_identity=np.array(p['input_source_identity']),
                cohort_admission_digest=np.array('d'*64),feature_schema=np.array('synthetic'))
            for name,shape in (('history',(m.N,168,2)),('weather',(m.N,18,4)),('state',(m.N,2))):
                a[name]=np.zeros(shape);a[name+'_mask']=np.zeros(shape,bool);a[name+'_mask'][0]=True
            typed=m.typed_sha(a)
            manifest=dict(arrays_sha256=typed,source_identity=p['input_source_identity'],gauge_id=g,
                binding=dict(execution_plan_sha256=p['plan_sha256']),cohort_admission_digest='d'*64,feature_schema='synthetic')
            bundle=dict(arrays=runtime.astore(g+'/inputs.npz',a),manifest=runtime.jstore(g+'/manifest.json',manifest))
            bundles[g]=bundle;gauges[g]=dict(input_bundle=bundle,input_arrays_sha256=typed,admitted_origins=1)
            for family in m.FAMILIES:
                for seed in m.SEEDS:
                    job=f'final_development__{g}__{family}__full_L168__seed{seed}'
                    selected=dict(job_id=job,gauge_id=g,architecture=family,seed=seed,checkpoint=dict(sha256='c'*64));records.append(selected)
                    q=np.full((m.N,18),np.nan);q[0]=1.
                    z={k:a[k] for k in ('origin_time','case_id','forecast_time','site_id','input_admitted')};z['prediction_q']=q
                    batch=dict(job_id=job,checkpoint_sha256='c'*64,case_count=1,input_case_ids_sha256=m.digest(a['case_id'][:1].tolist()),
                        target_values_read=False,target_masks_read=False,optimizer_updates=0,scaler_fits=0)
                    receipt=dict(predictions_array_sha256=m.typed_sha(z),input_arrays_sha256=typed,job=selected,
                        input_artifact=bundle['arrays'],checkpoint_sha256='c'*64,forecast_origins=1,candidate_origins=m.N,
                        target_values_assembled=False,optimizer_updates=0,scaler_fits=0,batches=[batch])
                    artifacts=dict(predictions=runtime.astore(job+'.npz',z),receipt=runtime.jstore(job+'.json',receipt))
                    commit=dict(kind='model',job_id=job,artifacts=artifacts)
                    runtime.commits[job]=commit;model_hashes[job]=m.digest(commit)
                    products[job]=dict(artifacts=artifacts,model_commit_sha256=m.digest(commit),checkpoint_sha256='c'*64,
                        predictions_array_sha256=receipt['predictions_array_sha256'],input_arrays_sha256=typed,forecast_origins=1)
        selection=dict(records=records);selection['selection_sha256']=m.digest(selection)
        inventory=dict(input_source_identity=p['input_source_identity'],gauge_bundles=bundles)
        inventory_desc=runtime.jstore('inventory.json',inventory)
        runtime.commits['INPUTS']=dict(artifacts=dict(inventory=inventory_desc))
        structural=runtime.jstore('structural.json',{'synthetic':True})
        runtime.commits['PREDICTION_BARRIER']=dict(artifacts=dict(barrier=structural),metadata=dict(model_commit_sha256_by_job=model_hashes))
        body=dict(schema='hydra-assessment-semantic-prediction-barrier/v1',
            status='ALL36_PREDICTION_SEMANTICS_VERIFIED_BEFORE_VERIFICATION_ROLE_READ',plan_sha256=p['plan_sha256'],
            selection_sha256=selection['selection_sha256'],structural_prediction_barrier_sha256=structural['sha256'],
            verification_values_read=False,nwm_values_read=False,models=products,gauges=gauges,input_inventory=inventory_desc)
        runtime.commits['SEMANTIC_PREDICTION_BARRIER']=dict(artifacts=dict(barrier=runtime.jstore('semantic.json',body)),metadata=body)
        return runtime,p,selection

    def test_complete_all36_semantics_and_missing_product_rejected(self):
        runtime,p,selection=self.complete_barrier_fixture()
        barrier,structural,inventory,predictions,cohorts,receipts=m.authenticate_barrier(runtime,p,selection)
        self.assertEqual(len(receipts),36)
        self.assertTrue(all(len(predictions[g])==12 and cohorts[g].sum()==1 for g in m.GAUGES))
        self.assertFalse(any('target' in path or 'verification' in path for path in runtime.read_paths))
        del runtime.commits[next(iter(m.MODEL_IDS))]
        with self.assertRaisesRegex(ValueError,'completed commit is absent'):
            m.authenticate_barrier(runtime,p,selection)

    def test_missing_semantic_barrier_never_reads_verification(self):
        class Runtime:
            calls=[]
            def reuse(self,key):
                self.calls.append(key)
                return {} if key=='PREDICTION_BARRIER' else None
        runtime=Runtime()
        with self.assertRaisesRegex(ValueError,'SEMANTIC_PREDICTION_BARRIER'):
            m.authenticate_barrier(runtime,{}, {})
        self.assertEqual(runtime.calls,['PREDICTION_BARRIER','SEMANTIC_PREDICTION_BARRIER'])

    def test_semantic_barrier_incomplete_roster_fails_before_array_decode(self):
        body=dict(schema='hydra-assessment-semantic-prediction-barrier/v1',
            status='ALL36_PREDICTION_SEMANTICS_VERIFIED_BEFORE_VERIFICATION_ROLE_READ',
            plan_sha256='p',selection_sha256='s',structural_prediction_barrier_sha256='h',
            verification_values_read=False,nwm_values_read=False,models={},gauges={g:{} for g in m.GAUGES})
        class Runtime:
            def reuse(self,key):
                if key=='PREDICTION_BARRIER':return dict(artifacts={'barrier':{'sha256':'h'}})
                if key=='SEMANTIC_PREDICTION_BARRIER':return dict(artifacts={'barrier':{}},metadata=body)
                raise AssertionError('No downstream artifact read permitted')
            def read(self,*args,**kwargs):return m.encoded(body)
        with patch.object(m,'decode_arrays',side_effect=AssertionError('Numerical decode prohibited')):
            with self.assertRaisesRegex(ValueError,'all36 semantic'):
                m.authenticate_barrier(Runtime(),{'plan_sha256':'p'},{'selection_sha256':'s'})

    def test_run_does_not_create_missing_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();run=root/'absent';plan=dict(run_directory=str(run),runtime_limits=m.LIMITS,
                checkpoint_sha256_by_job={j:'a'*64 for j in m.MODEL_IDS},original_training_updates=m.ORIGINAL_UPDATES,new_training=False)
            plan['plan_sha256']=m.digest(plan);path=root/'plan.json';path.write_bytes(m.encoded(plan))
            with self.assertRaisesRegex(ValueError,'already exist'):m.run(path,plan['plan_sha256'],'b'*64)
            self.assertFalse(run.exists())

    def test_no_production_scoring_model_network_imports(self):
        import ast
        tree=ast.parse((ROOT/'scripts/audit_assessment_results.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(s and any(x in s for x in ('assessment_scoring','inference','torch','urllib.request','staging')) for s in imports))


if __name__=='__main__':unittest.main()
