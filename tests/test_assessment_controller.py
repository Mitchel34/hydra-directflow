"""No real sources, models or assessment observations are opened here."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch
from types import SimpleNamespace
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('assessment_controller_tested',ROOT/'scripts/run_assessment_first_pass.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ControllerTests(unittest.TestCase):
    def test_batch_keeps_scalars_and_never_reads_target_keys(self):
        from hydra_directflow.assessment_inference import COORDINATES,METADATA,INPUT_ARRAYS
        class Guard(dict):
            def __getitem__(self,k):
                if k.startswith('target'):raise AssertionError('Verification data accessed')
                return super().__getitem__(k)
        a=Guard({k:np.arange(12) for k in COORDINATES+INPUT_ARRAYS})
        a.update({k:np.array('fixed') for k in METADATA});a['input_admitted']=np.ones(12,bool)
        a['target_q']=object();a['target_mask']=object()
        b=m.batch_view(a,np.array([1,4,9]))
        self.assertEqual(set(b),set(COORDINATES+METADATA+INPUT_ARRAYS)|{'input_admitted'})
        for k in METADATA:self.assertEqual(b[k].shape,())
        self.assertEqual(b['origin_time'].tolist(),[1,4,9])
        b['origin_time'][0]=999;self.assertEqual(a['origin_time'][1],1)

    def test_archive_roundtrip_and_no_pickle(self):
        a={'x':np.array([1.,np.nan]),'mask':np.array([True,False]),'scalar':np.array('scope')}
        b=m.arrays_from_bytes(m.npz(a))
        for k in a:np.testing.assert_equal(a[k],b[k])
        with self.assertRaises(ValueError):m.arrays_from_bytes(m.npz({'x':np.array([object()],object)}))

    def test_expansion_bound_before_loading(self):
        raw=m.npz({'x':np.zeros(10000)})
        with self.assertRaisesRegex(ValueError,'Bounded unique'):m.arrays_from_bytes(raw,maximum=1000)

    def test_duplicate_members_rejected(self):
        stream=io.BytesIO()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with zipfile.ZipFile(stream,'w') as z:z.writestr('x.npy',b'x');z.writestr('x.npy',b'y')
        with self.assertRaisesRegex(ValueError,'Bounded unique'):m.arrays_from_bytes(stream.getvalue())

    def test_resident_identity_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'input';p.write_bytes(b'only synthetic bytes')
            d=m.descriptor(p);charges=[]
            self.assertEqual(m.resident(p,1024,d,charges.append),p.read_bytes())
            self.assertEqual(charges,[20])
            p.write_bytes(b'different');
            with self.assertRaises(ValueError):m.resident(p,1024,d)
            alias=p.with_name('alias');alias.symlink_to(p)
            with self.assertRaises(ValueError):m.resident(alias,1024)

    def exercise_model(self, nonfinite=False):
        """Controller integration through one fake model, full calendar, sparse inputs."""
        from hydra_directflow.assessment_inference import COORDINATES,METADATA,INPUT_ARRAYS
        class Store:
            def __init__(self):self.files={};self.committed=[];self.loads=0;self.confirmed=0
            def read(self,d):return self.files[d['path']]
            def write(self,p,raw,**kwargs):
                self.files[p]=raw;return {'path':p,'bytes':len(raw),'sha256':m.sha(raw)}
            def reuse(self,key):return {'artifacts':{'inventory':{'path':'inputs'}}} if key=='INPUTS' else None
            def reserve_read(self,*args,**kwargs):pass
            def precharge_model(self,*args,**kwargs):self.loads+=1;return self.loads
            def confirm_model(self,token):self.confirmed+=1
            def precharge_forward(self,**kwargs):return 1
            def confirm_forward(self,token):pass
            def check(self):pass
            def commit(self,key,artifacts,**kwargs):self.committed.append((key,artifacts))
        r=Store();n=8760;c=np.arange(n,dtype=np.int64)*3600+1722384000
        a={k:np.zeros((n,1)) for k in INPUT_ARRAYS}
        a.update(origin_time=c,forecast_time=c[:,None]+np.arange(1,19)*3600,history_time=c[:,None]-7200,
                 case_id=np.array(['synthetic:'+str(x) for x in c]),site_id=np.zeros(n,dtype=np.int64),input_admitted=np.zeros(n,bool))
        a['input_admitted'][[0,70,8759]]=True
        a.update({k:np.array('b'*64) for k in METADATA});identity=m.array_identity(a)
        r.files.update(inputs=m.encoded({'gauge_bundles':{'03161000':{'arrays':{'path':'a'},'manifest':{'path':'manifest'}}}}),
                       a=m.npz(a),manifest=m.encoded({'arrays_sha256':identity}))
        seen=[]
        def forward(predictor,view,authority,**kwargs):
            seen.extend(view['origin_time'].tolist())
            return dict(case_id=view['case_id'],prediction_q=np.full((3,18),np.nan if nonfinite else -2.),receipt={'synthetic':True})
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve();cp=base/'cp';cp.write_bytes(b'synthetic checkpoint');gd=base/'gradient.json'
            ids=['synthetic-development-case'];gd.write_bytes(m.encoded({'case_ids':ids}));key=m.digest(ids)
            job=dict(job_id='synthetic-model',gauge_id='03161000',device='cpu',gradient_case_ids_sha256=key,
                checkpoint=dict(m.descriptor(cp),path='cp'),training_source_identity='a'*64,feature_schema='unused',training_cohort_admission_digest='b'*64)
            with patch.object(m,'OLD',base),patch.object(m,'progress'),\
                 patch('hydra_directflow.coverage_extension_inference.configure_runtime',return_value=SimpleNamespace()),\
                 patch('hydra_directflow.coverage_extension_inference.no_learning',return_value=nullcontext()),\
                 patch('hydra_directflow.inference.BestCheckpointPredictor.load',return_value=object()),\
                 patch('hydra_directflow.assessment_inference.validate_selected_predictor'),\
                 patch('hydra_directflow.assessment_inference.predict_assessment_batch',side_effect=forward):
                m.model(r,{'gradient_memberships':{key:m.descriptor(gd)}},{'selection':{'records':[job]}},object(),'synthetic-model')
        self.assertEqual(seen,c[[0,70,8759]].tolist());self.assertEqual((r.loads,r.confirmed),(1,1))
        self.assertEqual(len(r.committed),1)
        y=m.arrays_from_bytes(r.read(r.committed[0][1]['predictions']))['prediction_q']
        self.assertTrue(np.isnan(y[~a['input_admitted']]).all());self.assertTrue((y[a['input_admitted']]==-2).all())

    def test_one_model_records_only_admitted_inputs_and_preserves_raw_negative_outputs(self):self.exercise_model()

    def test_nonfinite_admitted_prediction_cannot_commit(self):
        with self.assertRaisesRegex(ValueError,'Prediction admission changed'):self.exercise_model(nonfinite=True)

if __name__=='__main__':unittest.main()
