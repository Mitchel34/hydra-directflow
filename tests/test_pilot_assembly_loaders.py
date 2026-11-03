"""Synthetic loader/controller checks; no study data, Torch, fits or network."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow import pilot_assembly as assembly
from hydra_directflow.training import array_identity
import test_pilot_assembly as fixtures

PATH=Path(__file__).resolve().parents[1]/'scripts/qualify_pilot_assembly_loaders.py'
SPEC=importlib.util.spec_from_file_location('pilot_loader_qualifier_tests',PATH)
qualifier=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(qualifier)


class LoaderQualificationTests(unittest.TestCase):
    def test_stream_hash_is_exact_array_identity(self):
        arrays={'float':np.arange(36,dtype=np.float64).reshape(6,6)[:,::2],
                'text':np.array(['A','R','A']), 'scalar':np.array('source'), 'mask':np.array([True,False])}
        self.assertEqual(qualifier.stream_array_identity(arrays),array_identity(arrays))

    def test_read_guard_denies_models_network_labels_and_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'labels.npz'; path.write_bytes(b'abc'); guard=qualifier.ReadGuard(root,maximum=3)
            for event,args in [('import',('torch.nn',)),('import',('transformers',)),('socket.connect',()),('socket.sendto',())]:
                with self.assertRaises(RuntimeError): guard.audit(event,args)
            guard.labels_forbidden=True
            with self.assertRaisesRegex(ValueError,'label access'): guard.audit('open',(str(path),'rb',0))
            self.assertEqual(guard.read_bytes,0)
            guard.labels_forbidden=False; guard.audit('open',(str(path),'rb',0)); self.assertEqual(guard.read_bytes,3)
            with self.assertRaisesRegex(ValueError,'before open'): guard.audit('open',(str(path),'rb',0))
            with self.assertRaisesRegex(ValueError,'immutable'): guard.audit('open',(str(path),'wb',0))
            self.assertEqual(guard.report()['file_opens'],1)

    def test_claim_is_global_once_only_and_bound_to_control(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); allowances=root/'allowances'; allowances.mkdir(); control=root/'run'; control.mkdir()
            plan={'plan_sha256':'a'*64}; grant=dict(plan_sha256='a'*64,control_directory=str(control),deadline_epoch=time.time()+30)
            (allowances/('a'*64+'.json')).write_text(json.dumps(grant))
            with patch.object(qualifier,'ALLOWANCES',allowances):
                qualifier.claim_worker(plan,grant,control)
                with self.assertRaises(FileExistsError): qualifier.claim_worker(plan,grant,control)
                other=root/'other'; other.mkdir()
                with self.assertRaisesRegex(ValueError,'worker/control'): qualifier.claim_worker(plan,grant,other)
            self.assertTrue((control/'claimed.json').is_file())

    def test_metadata_plan_reads_no_numeric_or_lookup_payload(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'MANIFEST.json').write_text('{}')
            expected={g:{r:{'cases':1,'case_ids_sha256':'b'*64} for r in ('fit','inner_stop','outer')} for g in fixtures.GAUGES}
            bounds={r:{'start_utc':'2020-01-01T00:00:00Z','end_utc':'2020-02-01T00:00:00Z'} for r in ('fit','inner_stop','outer')}
            manifest=dict(source_plan_sha256=qualifier.SOURCE_PLAN,snapshot_sha256=qualifier.SNAPSHOT,roles=expected,
                role_time_bounds=bounds,refit_at_utc='2020-02-01T00:00:00Z',execution=dict(scaler_fits=0,ridge_fits=0,optimizer_updates=0,network_requests=0,protected_access=False),
                scientific_fit_authorization=False,content_sha256='c'*64,
                coverage_outer={g:{'cases':8742,'cohort_mode':'all_outer_candidates'} for g in expected})
            with patch.object(qualifier,'private_path',side_effect=Path),patch.object(assembly,'_manifest',return_value=(root,manifest)),\
                    patch.object(qualifier,'authority',return_value=({},bounds,manifest['refit_at_utc'],expected)),\
                    patch.object(qualifier,'implementation',return_value={'test':'fixed'}),\
                    patch.object(assembly,'_load_npz',side_effect=AssertionError('numeric load')):
                plan=qualifier.prepare(root,'d'*64)
            self.assertEqual(plan['scope']['maximum_parent_child_rss_bytes'],4*1024**3)
            self.assertEqual(plan['scope']['maximum_bundle_estimated_bytes'],1024**3)
            self.assertEqual(plan['scope']['maximum_assembly_logical_read_bytes'],1024**3)
            self.assertEqual(plan['scope']['torch_imports'],0)
            changed=deepcopy(plan); changed['scope']['scaler_fits']=1; changed['plan_sha256']=qualifier.sha(qualifier.canonical({k:v for k,v in changed.items() if k!='plan_sha256'}))
            with patch.object(qualifier,'prepare',return_value=plan):
                qualifier.validate_plan(plan,plan['plan_sha256'])
                with self.assertRaisesRegex(ValueError,'source/code/scope'): qualifier.validate_plan(changed,changed['plan_sha256'])

    def test_full_synthetic_three_gauge_loader_qualification_without_torch(self):
        fixture=fixtures.PilotAssemblyTests('test_metadata_plan_never_loads_numeric_payloads'); fixture.setUp()
        try:
            result=fixture.run_assembly()
            plan=dict(assembly_directory=str(fixture.output),assembly_manifest_sha256=result['manifest_sha256'],
                role_time_bounds=fixture.roles,refit_at_utc=fixture.refit,roles=fixture.expected,
                scope={'training_gauges':list(fixtures.GAUGES)})
            guard=qualifier.ReadGuard(fixture.output); original=Path.open
            def guarded(path,mode='r',*args,**kwargs):
                guard.audit('open',(str(path),mode,0)); return original(path,mode,*args,**kwargs)
            with patch.object(Path,'open',guarded): report=qualifier.qualify(plan,guard,lambda:None)
            self.assertEqual(len(report['training']),3); self.assertEqual(len(report['outer_batch_checks']),6); self.assertEqual(len(report['ridge']),3)
            self.assertTrue(all(x['cases']==11 and x['outer_cases']==0 for x in report['training']))
            self.assertTrue(all(x['unique_gradient_hours']==24 and x['eligible_pairs']==17 for x in report['ridge']))
            self.assertGreater(report['read_telemetry']['logical_bytes_reserved_at_open'],0)
            self.assertTrue(all(x['input_phase_label_opens']==0 and not x['label_values_scored'] for x in report['outer_batch_checks']))
            self.assertNotIn('torch',sys.modules)
            for key in ('model_calls','scaler_fits','ridge_fits','optimizer_updates','scientific_scores','network_requests'): self.assertEqual(report[key],0)
        finally: fixture.tearDown()

    def test_wrong_role_hash_rejected_without_transform(self):
        fixture=fixtures.PilotAssemblyTests('test_metadata_plan_never_loads_numeric_payloads'); fixture.setUp()
        try:
            result=fixture.run_assembly(); bundle=assembly.load_training_bundle(fixture.output,expected_manifest_sha256=result['manifest_sha256'],gauge_id=fixtures.GAUGES[0])
            expected=deepcopy(fixture.expected[fixtures.GAUGES[0]]); expected['fit']['case_ids_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'membership'): qualifier.validate_role_membership(bundle,expected)
        finally: fixture.tearDown()

    def test_precharge_receipt_cannot_be_reset(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'receipt.json'; qualifier.write_new(path,{'status':'consumed'})
            with self.assertRaises(FileExistsError): qualifier.write_new(path,{'status':'reset'})
            self.assertEqual(json.loads(path.read_text()),{'status':'consumed'})


if __name__=='__main__': unittest.main()
