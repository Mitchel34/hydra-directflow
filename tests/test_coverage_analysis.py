import unittest
import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hydra_directflow.coverage_analysis import (reference_arrays,gradient_thresholds,regime_codes,
    unusable_intervals,select_observed_events,coverage_by_regime,event_coverage,reference_status,season,HOUR)
from hydra_directflow.local_data import utc_epoch
from hydra_directflow.cache import DerivedCache

S=importlib.util.spec_from_file_location('coverage_audit',Path(__file__).resolve().parents[1]/'scripts/audit_coverage_regimes.py')
audit=importlib.util.module_from_spec(S);S.loader.exec_module(audit)


class CoverageTests(unittest.TestCase):
    def setUp(self):self.start=utc_epoch('2020-01-01T00:00:00Z')
    def reference(self,n=1000):
        return reference_arrays([(self.start+i*HOUR,float(i),('A',)) for i in range(n)],start=self.start,end=self.start+n*HOUR)
    def test_status_does_not_admit_estimated_provisional_unknown_or_missing(self):
        for flags in (('A','e'),('P',),('A','unknown'),()):self.assertNotEqual(reference_status(2.,flags),'APPROVED_REFERENCE')
        self.assertEqual(reference_status(2.,('A','R')),'APPROVED_REFERENCE')
        self.assertNotEqual(reference_status(None,('A',)),'APPROVED_REFERENCE')
    def test_reference_missingness_is_not_filled_or_called_station_damage(self):
        r=reference_arrays([(self.start,2.,('A',)),(self.start+2*HOUR,200.,('e',))],start=self.start,end=self.start+4*HOUR)
        self.assertTrue(np.isnan(r['values'][1:]).all());out=unusable_intervals(r)
        self.assertEqual(out[0]['hours'],3);self.assertFalse(out[0]['hidden_peak_magnitude_known'])
        self.assertIn('NOT_PROOF',out[0]['cause'])
    def test_duplicate_reference_rejected(self):
        with self.assertRaises(ValueError):reference_arrays([(self.start,1.,('A',))]*2,start=self.start,end=self.start+HOUR)
    def test_stopping_and_outer_values_cannot_change_gradient_thresholds(self):
        r=self.reference();a=gradient_thresholds(r,start=self.start,end=self.start+200*HOUR,refit_at=self.start+400*HOUR)
        r['values'][200:]=1e12;b=gradient_thresholds(r,start=self.start,end=self.start+200*HOUR,refit_at=self.start+400*HOUR)
        self.assertEqual(a,b)
    def test_thresholds_use_unique_reference_hours_not_overlapping_targets(self):
        r=self.reference();a=gradient_thresholds(r,start=self.start,end=self.start+200*HOUR,refit_at=self.start+400*HOUR)
        self.assertEqual(a['reference_hours'],200);self.assertAlmostEqual(a['q95_cms'],189.05)
    def test_unknown_regime_stays_unknown_even_if_a_value_was_present(self):
        codes=regime_codes(np.array([0.,5.,9.,10.,99.]),np.array([1,1,1,1,0],bool),dict(q10_cms=1.,q90_cms=8.,q95_cms=9.))
        np.testing.assert_array_equal(codes,[0,1,2,3,4])
    def test_event_ranking_ties_separation_and_boundary_truncation(self):
        r=self.reference(800);r['values'][:]=0;r['values'][[5,6,200,400]]=100
        events=select_observed_events(r,dict(q95_cms=50.),start=self.start,end=self.start+800*HOUR)
        self.assertEqual([e['peak_epoch'] for e in events],[self.start+5*HOUR,self.start+200*HOUR,self.start+400*HOUR])
        self.assertTrue(events[0]['left_truncated']);self.assertTrue(all(not e['selection_uses_predictions'] for e in events))
    def test_missing_peak_not_selected_or_interpolated(self):
        r=self.reference();r['values'][:]=0;r['values'][500]=np.nan;r['admitted'][500]=False
        self.assertEqual(select_observed_events(r,dict(q95_cms=10.),start=self.start,end=self.start+1000*HOUR),[])
    def test_regime_denominators_and_independent_input_label_support(self):
        r=self.reference(100);origin=np.array([self.start,self.start+HOUR]);mask=np.ones((2,18),bool)
        rows=coverage_by_regime(origin,np.array([1,0],bool),np.array([1,0],bool),mask,r,dict(q10_cms=5.,q90_cms=15.,q95_cms=17.),start=self.start,end=self.start+100*HOUR)
        self.assertEqual(sum(x['requested_target_cells'] for x in rows),36)
        self.assertEqual(sum(x['common_comparison_cells'] for x in rows),18)
        expanded=coverage_by_regime(origin,np.array([1,0],bool),np.array([1,0],bool),mask,r,dict(q10_cms=5.,q90_cms=15.,q95_cms=17.),start=self.start,end=self.start+100*HOUR,natural_l24=np.ones(2,bool))
        self.assertEqual(sum(x['natural_L24_cells_lost_to_common_L168'] for x in expanded),18)
        with self.assertRaises(ValueError):coverage_by_regime(origin,np.ones(2,bool),np.ones(2,bool),mask,r,dict(q10_cms=5.,q90_cms=15.,q95_cms=17.),start=self.start,end=self.start+100*HOUR,natural_l24=np.array([1,0],bool))
        mask[0,0]=False
        with self.assertRaises(ValueError):coverage_by_regime(origin,np.ones(2,bool),np.ones(2,bool),mask,r,dict(q10_cms=5.,q90_cms=15.,q95_cms=17.),start=self.start,end=self.start+100*HOUR)
    def test_event_peak_without_model_input_stays_in_catalog_with_zero_support(self):
        origin=np.array([self.start]);event=dict(start_epoch=self.start,end_epoch=self.start+30*HOUR,peak_epoch=self.start+HOUR)
        rows=event_coverage(event,origin,np.array([False]),np.array([False]),np.ones((1,18),bool),outer_start=self.start,outer_end=self.start+40*HOUR)
        self.assertFalse(rows[0]['observed_peak_has_common_prediction_case']);self.assertEqual(rows[0]['approved_target_cells'],1)
    def test_seasons_use_utc_calendar(self):
        self.assertEqual([season(utc_epoch(t)) for t in ('2020-02-29T23:00:00Z','2020-03-01T00:00:00Z','2020-12-01T00:00:00Z')],['DJF','MAM','DJF'])
    def test_wrong_month_identity_key_rejected_before_value_load(self):
        with tempfile.TemporaryDirectory() as folder:
            cache=DerivedCache(folder);key='a'*64;entry=Path(folder)/'entries'/key;entry.mkdir(parents=True)
            identity={'month':'2024-07','source_manifest_sha256':audit.USGS_MANIFEST_SHA256,
                      'development_export':{'cutoff_exclusive_epoch':audit.ASSESSMENT_START}}
            (entry/'COMMIT.json').write_text(json.dumps({'kind':'usgs_month','key':key,'identity':identity}))
            with patch.object(cache,'_load',side_effect=AssertionError('numeric read')) as load:
                with self.assertRaisesRegex(ValueError,'identity/key'):audit.reference_month(cache,'2024-07',key)
                load.assert_not_called()
            self.assertFalse((Path(folder)/'CACHE_ROOT.json').exists())
            self.assertFalse((Path(folder)/'.writer.lock').exists())
    def test_raw_mixed_july_rejected_before_value_load_even_with_consistent_key(self):
        with tempfile.TemporaryDirectory() as folder:
            cache=DerivedCache(folder);identity={'month':'2024-07','source_manifest_sha256':audit.USGS_MANIFEST_SHA256}
            key=cache.key('usgs_month',identity);entry=Path(folder)/'entries'/key;entry.mkdir(parents=True)
            (entry/'COMMIT.json').write_text(json.dumps({'kind':'usgs_month','key':key,'identity':identity}))
            with patch.object(cache,'_load',side_effect=AssertionError('numeric read')) as load:
                with self.assertRaisesRegex(ValueError,'development-only'):audit.reference_month(cache,'2024-07',key)
                load.assert_not_called()


if __name__=='__main__':unittest.main()
