"""Pure v2 scope/claim/source gates; no numerical fixtures, models or updates."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import qualify_gap_pilot_v2 as q
import profile_gap_pilot_v2 as p
import qualify_gap_pilot_constructor_v2 as c


def mock_legacy():
    return {'plan':{'spec':{'implementation':{'code':{},'dependencies':{}}}}}


class ControllerTests(unittest.TestCase):
    def test_v2_scope_is_separate16_updates_and_reuses_ridge_without_fitting(self):
        with (patch.object(q,'implementation',return_value={'code':{},'dependencies':{}}),
              patch.object(q,'legacy_report',side_effect=mock_legacy),patch.object(q,'legacy_binding',return_value={'old_consumed_optimizer_updates':16})):
            value=q.plan()
        self.assertEqual(sum(w['updates'] for w in value['spec']['workers']),16)
        self.assertEqual(sum(w['ridge_fits'] for w in value['spec']['workers']),0)
        self.assertEqual(value['spec']['synthetic_ridge_fits'],0)
        for mode in ('clean','augmented'):
            self.assertEqual([w['updates'] for w in q.workers() if w['mode']==mode],[4,2,2,0])
        self.assertNotIn('budget-v1',str(q.LEDGER));self.assertTrue(value['spec']['resident_v1_ridge_reused'])

    def test_v2_claim_cannot_replay_or_change_v1_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);old=directory/'budget-v1.json';old.write_text('{"consumed":16}\n');oldbytes=old.read_bytes()
            prepared={'plan_sha256':'a'*64}
            with (patch.object(q,'ROOT',directory),patch.object(q,'LEDGER',directory/'budget-v2.json'),patch.object(q,'LOCK',directory/'v2.lock'),
                  patch.object(q,'plan',return_value=prepared)):
                q.precharge(prepared,directory/'run')
                with self.assertRaises(ValueError):q.precharge(prepared,directory/'another')
                q.claim(directory/'run',q.workers()[0],prepared)
                with self.assertRaises(ValueError):q.claim(directory/'run',q.workers()[0],prepared)
                self.assertEqual(json.loads(q.LEDGER.read_text())['precharged_synthetic_ridge_fits'],0)
            self.assertEqual(old.read_bytes(),oldbytes)

    def test_legacy_reference_paths_cannot_point_outside_stable_checkout(self):
        with self.assertRaisesRegex(ValueError,'stable checkout'):q.resident_artifact(Path('/tmp/unrelated-v1.pt'),100)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, tempfile.TemporaryDirectory() as elsewhere:
            outside=Path(elsewhere)/'data';outside.write_text('x')
            linked=Path(tmp)/'linked';linked.symlink_to(outside)
            with self.assertRaisesRegex(ValueError,'stable checkout'):q.resident_artifact(linked,100)

    def test_unreviewed_other_helper_or_runtime_changes_reject_plan(self):
        previous={'plan':{'spec':{'implementation':{'code':{'models.py':'old'},'dependencies':{'torch':'old'}}}}}
        for current in ({'code':{'models.py':'new'},'dependencies':{'torch':'old'}},
                        {'code':{'models.py':'old'},'dependencies':{'torch':'new'}}):
            with patch.object(q,'implementation',return_value=current),patch.object(q,'legacy_report',return_value=previous):
                with self.assertRaises(ValueError):q.plan()

    def test_legacy_numeric_trace_mismatch_rejects_before_checkpoint_tensor_read(self):
        with (patch.object(q,'legacy_result',return_value={'trace':[{'loss':1.}]}),
              patch.object(q,'resident_artifact',side_effect=AssertionError('Must reject before reading checkpoint'))):
            with self.assertRaisesRegex(ValueError,'differs from v1: trace'):
                q.compare_legacy_baseline(ROOT/'.local/not-executed','clean',{'trace':[{'loss':2.}]})

    def test_profile_initial_claim_is_exclusive_and_no_wrong_plan_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'profile.json';p.start_report(target,{'owner':'first'})
            with self.assertRaises(FileExistsError):p.start_report(target,{'owner':'second'})
            self.assertEqual(json.loads(target.read_text()),{'owner':'first'})
        prepared={'plan_sha256':'a'*64}
        with patch.object(p,'plan',return_value=prepared),patch.object(p,'start_report',side_effect=AssertionError('Claimed wrong plan')):
            with self.assertRaises(ValueError):p.execute(prepared,'b'*64)
        self.assertEqual(p.CALLS,20);self.assertEqual(len(p.STAGES),6);self.assertEqual(p.MAX_SECONDS,120)
        self.assertNotEqual(p.REPORT,ROOT/'reports/gap-pilot-preparation-profile.json')

    def test_profile_uses_actual_optimized_provenance_digest_entrypoint(self):
        import hydra_directflow.pilot_training as pilot
        with patch.object(pilot,'_canonical_digest',return_value='a'*64) as helper:
            self.assertEqual(p.provenance_digest({'synthetic':True}),'a'*64)
        helper.assert_called_once_with({'synthetic':True})

    def test_profile_adapter_receives_validated_snapshot_view_not_raw_metadata(self):
        view=object();snapshot=SimpleNamespace(case_count=2,storage_bytes=16,provenance_sha256='a'*64,select=Mock(return_value=view))
        factory=Mock(return_value=snapshot);module=SimpleNamespace(admit_history_provenance=factory)
        batch={'origin_time':[1,2]};raw={'synthetic':'metadata'}
        with patch.dict(sys.modules,{'hydra_directflow.pilot_provenance':module}):
            actual,receipt=p.snapshot_for_profile(batch,raw,'b'*64,'c'*64)
        self.assertIs(actual,view);self.assertEqual(receipt['setup_calls'],1)
        factory.assert_called_once_with(batch,raw,source_identity='b'*64,cohort_digest='c'*64)
        snapshot.select.assert_called_once_with((0,1))

    def test_profile_requires_measured_bounded_cache_counters(self):
        predictor=SimpleNamespace(provenance_cache_info=lambda:{'hits':19,'misses':21,'entries':1,'admissions':21,'explicit_views':0})
        self.assertEqual(p.cache_observation(predictor),{'hits':19,'misses':21,'entries':1,'admissions':21,'explicit_views':0})
        for counters in ({'hits':True,'misses':0,'entries':1,'admissions':0,'explicit_views':0},
                         {'hits':0,'misses':1,'entries':2,'admissions':1,'explicit_views':0}):
            with self.subTest(counters=counters),self.assertRaises(ValueError):
                p.cache_observation(SimpleNamespace(provenance_cache_info=lambda:counters))

    def test_constructor_counts_are_authoritative_and_outer_is_explicit(self):
        base=c.authority();outer=c.authority(True)
        self.assertEqual(base['total_source_rows'],17869);self.assertEqual(outer['total_source_rows'],26425)
        self.assertEqual(base['counts'],{'fit':15717,'inner':2152,'outer':0})
        self.assertEqual(outer['counts']['outer'],8556)
        self.assertEqual(base['report_sha256'],outer['report_sha256'])
        self.assertEqual(base['unassigned_rows_in_this_fixture'],0)
        with self.assertRaises(ValueError):c.authority(1)

    def test_constructor_has_one_zero_update_allowance_for_both_source_extent_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            prepared={'plan_sha256':'a'*64,'spec':{'include_outer':False,'authority':{'total_source_rows':17869}}}
            with (patch.object(c,'ROOT',directory),patch.object(c,'LEDGER',directory/'constructor-v2.json'),patch.object(c,'LOCK',directory/'constructor-v2.lock'),
                  patch.object(c,'plan',return_value=prepared)):
                value=c.precharge(prepared)
                self.assertEqual(value['precharged_optimizer_updates'],0)
                self.assertEqual(value['precharged_synthetic_scaler_fits'],1)
                with self.assertRaises(ValueError):c.precharge(prepared)
                c.claim(prepared)
                with self.assertRaises(ValueError):c.claim(prepared)

    def test_constructor_plan_keeps_actual_row_extent_and_no_model_work(self):
        with patch.object(c,'implementation',return_value={'code':{},'dependencies':{}}):value=c.plan()
        self.assertEqual(value['spec']['authority']['total_source_rows'],17869)
        self.assertEqual(value['spec']['maximum_optimizer_updates'],0)
        self.assertEqual(value['spec']['maximum_model_forward_calls'],0)
        self.assertEqual(value['spec']['maximum_synthetic_scaler_fits'],1)
        self.assertFalse(value['spec']['automatic_size_reduction'])
        with patch.object(c,'plan',return_value=value),patch.object(c,'precharge',side_effect=AssertionError('Wrong plan consumed')):
            with self.assertRaises(ValueError):c.execute(value,'b'*64)

    def test_small_synthetic_constructor_fixture_preserves_roles_without_observed_source_data(self):
        value=c.authority(True)
        value.update(counts={'fit':4,'inner':2,'outer':1},total_source_rows=7)
        arrays,provenance,source,cohort=c.synthetic_fixture(value)
        self.assertEqual(arrays['history'].shape,(7,168,2));self.assertEqual(arrays['target'].shape,(7,18))
        self.assertTrue(arrays['history_mask'].all());self.assertTrue(arrays['target_mask'].all())
        self.assertEqual(len(source),64);self.assertEqual(len(cohort),64)
        self.assertTrue(all(item.startswith('synthetic-usgs/') for row in provenance['source_ids'] for item in row))
        self.assertEqual(provenance['observation_times'],arrays['history_time'].tolist())
        self.assertLess(int(arrays['target_time'][3,-1]),int(arrays['origin_time'][4]))
        self.assertLess(int(arrays['target_time'][5,-1]),int(arrays['origin_time'][6]))

    def test_old_v1_controller_and_profile_bytes_match_published_receipts(self):
        qualified=json.loads((ROOT/'reports/gap-pilot-qualification-v1.json').read_text())
        profile=json.loads((ROOT/'reports/gap-pilot-preparation-profile.json').read_text())
        for name,report in [('scripts/qualify_gap_pilot.py',qualified),('scripts/profile_gap_pilot.py',profile)]:
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),report['plan']['spec']['implementation']['code'][name])


if __name__=='__main__':unittest.main()
