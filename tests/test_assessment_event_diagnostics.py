import copy
import json
import unittest

import numpy as np

from hydra_directflow import assessment_event_diagnostics as m
from hydra_directflow.contract import ARCHITECTURES


class ReferenceAndSelectionTests(unittest.TestCase):
    def test_reference_deduplicates_consistent_values_and_preserves_rejected_hours(self):
        times=np.arange(1,21)*3600
        future=np.array([times[:18],times[1:19],times[2:20]])
        q=np.arange(1,21,dtype=float);mask=np.ones(20,bool);mask[5]=False;q[5]=np.nan
        y=np.array([q[:18],q[1:19],q[2:20]]);v=np.array([mask[:18],mask[1:19],mask[2:20]])
        t,a,b=m.unique_reference(y,v,future)
        np.testing.assert_equal(t,times);np.testing.assert_equal(a,q);np.testing.assert_equal(b,mask)

    def test_conflicting_duplicate_values_or_quality_fail(self):
        t=np.arange(20)*3600;f=np.array([t[:18],t[1:19]])
        q=np.ones((2,18));v=np.ones((2,18),bool)
        q[1,0]=2
        with self.assertRaisesRegex(ValueError,'Repeated approved'):m.unique_reference(q,v,f)
        q[:]=1;v[1,0]=False
        with self.assertRaisesRegex(ValueError,'quality masks'):m.unique_reference(q,v,f)

    def test_selection_ties_separation_top3_and_half_open_window(self):
        t=np.arange(1000)*3600;q=np.zeros(1000);v=np.ones(1000,bool)
        q[200]=100;q[201]=100;q[368]=90;q[535]=80;q[536]=70;q[800]=60
        result=m.select_events(t,q,v,10.)
        self.assertEqual([r['peak_epoch']//3600 for r in result],[200,368,536,800])
        self.assertEqual([r['representative_top3'] for r in result],[True,True,True,False])
        self.assertEqual((result[0]['start_epoch'],result[0]['end_exclusive_epoch']),(152*3600,272*3600))
        self.assertEqual(result[0]['reference_requested_hours'],120)
        self.assertFalse(result[2]['selected_peak_is_window_max'])  # retained shoulder; not relabelled

    def test_unapproved_large_values_not_ranked_and_not_filled(self):
        t=np.arange(1000)*3600;q=np.zeros(1000);v=np.ones(1000,bool)
        q[100]=50;q[500]=np.nan;v[500]=False
        self.assertEqual(len(m.select_events(t,q,v,10.)),1)
        q[500]=9999
        with self.assertRaisesRegex(ValueError,'never imputed'):m.select_events(t,q,v,10.)

    def test_boundary_clipping_and_q95_ties(self):
        t=np.arange(400)*3600;q=np.zeros(400);v=np.ones(400,bool)
        q[0]=40;q[-1]=30;q[200]=10
        events=m.select_events(t,q,v,10.)
        self.assertTrue(events[0]['left_truncated']);self.assertTrue(events[1]['right_truncated'])
        self.assertEqual(events[1]['end_exclusive_epoch'],400*3600)
        self.assertEqual(len(events),2)

    def test_selection_does_not_accept_policy_changes(self):
        with self.assertRaises(ValueError):m.select_events(np.arange(2)*3600,np.ones(2),np.ones(2,bool),0.,min_separation_hours=24)


class EventAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.origins=np.arange(m.ASSESSMENT_START,m.ASSESSMENT_END,3600,dtype=np.int64)
        cls.times=np.arange(m.REFERENCE_START,m.REFERENCE_END,3600,dtype=np.int64)
        q=10.+np.sin(np.arange(len(cls.times))/20)
        q[500]=100;q[1500]=90;q[2500]=80;q[3500]=70
        cls.q=q
        cls.future=cls.origins[:,None]+np.arange(1,19)*3600
        cls.offsets=(cls.future-cls.times[0])//3600

    def fixture(self):
        y=self.q[self.offsets].copy();v=np.ones(y.shape,bool);admitted=np.ones(8760,bool)
        nw=y+1.;nw[:,16:]=np.nan;nm=np.ones(y.shape,bool);nm[:,16:]=False
        predictions={(a,s):y+float(i+1) for a in ARCHITECTURES for i,s in enumerate(m.SEEDS)}
        h=np.ones((8760,168,2));h[...,0]=7.;h[...,1]=np.arange(169,1,-1)
        cat=m.select_events(self.times,self.q,np.ones(len(self.times),bool),50.)
        return dict(origins=self.origins,truth=y,valid=v,input_admitted=admitted,predictions=predictions,
                    nwm=nw,nwm_mask=nm,history=h,q95=50.,catalog=cat)

    def test_event_metrics_peak_context_h18_and_signed_gap_reconcile(self):
        f=self.fixture();r=m.analyze_gauge(**f);summary=r['summary'];event=summary['events'][0]
        self.assertEqual(summary['representative_event_count'],3)
        self.assertEqual(len(event['nwm_paired']),16)
        self.assertEqual(event['nwm_paired'][0]['n'],120)
        self.assertAlmostEqual(event['nwm_paired'][0]['rmse'],1.)
        seed=event['models']['LSTM']['seeds']['29']['paired'][0]
        self.assertAlmostEqual(seed['rmse'],2.);self.assertAlmostEqual(seed['sse'],480.)
        self.assertEqual(seed['at_observed_peak']['prediction_q_cms'],102.)
        self.assertEqual(seed['fixed_lead_trace_maximum']['timing_from_selected_observed_peak_hours'],0.)
        context=event['coverage_by_lead'][0]['latest_observation_context']
        self.assertEqual(context['valid_epoch'],event['event']['peak_epoch']-3*3600)
        self.assertEqual(context['q_cms'],7.)
        self.assertEqual(event['models']['LSTM']['seeds']['17']['hydra_h18']['n'],120)
        d=summary['decomposition']['LSTM'];self.assertAlmostEqual(sum(v['mean'] for v in d['contributions'].values()),d['seed_mean_and_range']['mean'])
        expected=[]
        for j in range(16):
            y=f['truth'][:,j];sst=np.sum((y-y.mean())**2)
            expected.append(((1+4+9)/3-1)*len(y)/sst)
        self.assertAlmostEqual(d['seed_mean_and_range']['mean'],sum(expected)/16)
        json.dumps(summary,allow_nan=False)

    def test_mean_of_seed_errors_is_not_error_of_mean_forecast(self):
        r=m.analyze_gauge(**self.fixture())['summary']['events'][0]['models']['LSTM']['seed_mean_and_range'][0]
        self.assertAlmostEqual(r['physical_mse']['mean'],14/3)
        self.assertNotEqual(r['physical_mse']['mean'],4.)

    def test_unavailable_largest_peak_retained_not_replaced(self):
        f=self.fixture();peak=f['catalog'][0]['peak_epoch'];bad=np.any((self.future>=peak-48*3600)&(self.future<peak+72*3600),axis=1)
        f['input_admitted'][bad]=False
        for a in f['predictions'].values():a[bad]=np.nan
        r=m.analyze_gauge(**f)['summary'];first=r['events'][0]
        self.assertEqual(first['event']['peak_epoch'],peak);self.assertEqual(r['representative_event_count'],3)
        self.assertEqual(first['nwm_paired'][0]['status'],'UNAVAILABLE_NO_MATCHED_TARGETS')
        self.assertIsNone(first['models']['LSTM']['seed_mean_and_range'][0]['rmse']['mean'])
        self.assertFalse(r['unsupported_events_replaced'])

    def test_original_strict_all18_target_mask_preserved(self):
        f=self.fixture();badtime=self.times[490];invalid=self.future==badtime;f['valid'][invalid]=False;f['truth'][invalid]=np.nan
        times,q,mask=m.unique_reference(f['truth'],f['valid'],self.future);f['catalog']=m.select_events(times,q,mask,50.)
        r=m.analyze_gauge(**f);expected=f['input_admitted']&f['valid'].all(axis=1)
        np.testing.assert_equal(r['plot']['primary_mask'],expected)
        self.assertEqual(int((~expected).sum()),18)
        self.assertFalse(r['plot']['matched_mask'][:,16:].any())

    def test_unknown_peak_trace_and_constant_denominator_remain_explicit(self):
        f=self.fixture();f['truth'][:]=10.;f['nwm'][:,:16]=10.
        for p in f['predictions'].values():p[:]=10.
        f['catalog']=[];f['q95']=20.
        r=m.analyze_gauge(**f)['summary']
        self.assertEqual(r['events'],[]);self.assertIsNone(r['decomposition']['LSTM']['seed_mean_and_range']['mean'])
        json.dumps(r,allow_nan=False)

    def test_missing_admitted_prediction_and_extra_family_fail(self):
        f=self.fixture();f['predictions']['LSTM',17][0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite on admitted'):m.analyze_gauge(**f)
        f=self.fixture();f['predictions']['new_model',17]=f['truth']
        with self.assertRaisesRegex(ValueError,'four models'):m.analyze_gauge(**f)

    def test_catalog_change_or_nwm_unsupported_lead_rejected(self):
        f=self.fixture();f['catalog'][0]['peak_epoch']+=3600
        with self.assertRaisesRegex(ValueError,'Catalog differs'):m.analyze_gauge(**f)
        f=self.fixture();f['nwm_mask'][:,16]=True;f['nwm'][:,16]=1.
        with self.assertRaisesRegex(ValueError,'NWM must retain'):m.analyze_gauge(**f)

    def test_fixed_trace_maximum_distinct_from_at_observed_peak(self):
        f=self.fixture();event=f['catalog'][0];peak=event['peak_epoch'];j=0
        i=int((peak+6*3600-self.origins[0]-3600)//3600)
        f['predictions']['LSTM',17][i,j]=500.
        row=m.analyze_gauge(**f)['summary']['events'][0]['models']['LSTM']['seeds']['17']['paired'][0]
        self.assertEqual(row['at_observed_peak']['prediction_q_cms'],101.)
        self.assertEqual(row['fixed_lead_trace_maximum']['q_cms'],500.)
        self.assertEqual(row['fixed_lead_trace_maximum']['timing_from_selected_observed_peak_hours'],6.)

    def test_trace_maximum_retained_but_timing_unavailable_without_matched_observed_peak(self):
        f=self.fixture();event=f['catalog'][0];peak=event['peak_epoch'];j=0
        peak_origin=int((peak-self.origins[0]-3600)//3600)
        maximum_origin=peak_origin+6
        f['nwm_mask'][peak_origin,j]=False;f['nwm'][peak_origin,j]=np.nan
        f['predictions']['LSTM',17][maximum_origin,j]=500.
        result=m.analyze_gauge(**f)['summary']['events'][0]
        row=result['models']['LSTM']['seeds']['17']['paired'][0]
        self.assertEqual(row['n'],119)
        self.assertEqual(row['at_observed_peak']['status'],'UNAVAILABLE_AT_OBSERVED_PEAK')
        trace=row['fixed_lead_trace_maximum']
        self.assertEqual(trace['q_cms'],500.)
        self.assertEqual(trace['valid_epoch'],peak+6*3600)
        self.assertIsNone(trace['timing_from_selected_observed_peak_hours'])
        self.assertEqual(trace['timing_status'],'UNAVAILABLE_OBSERVED_PEAK_NOT_MATCHED')
        self.assertFalse(trace['coverage_complete'])
        self.assertIsNone(result['nwm_paired'][0]['fixed_lead_trace_maximum']['timing_from_selected_observed_peak_hours'])

    def test_decomposition_can_be_negative_for_an_event(self):
        f=self.fixture();e=f['catalog'][0];inside=(self.future>=e['start_epoch'])&(self.future<e['end_exclusive_epoch'])
        for p in f['predictions'].values():p[inside]=f['truth'][inside]
        d=m.analyze_gauge(**f)['summary']['decomposition']['LSTM']
        self.assertLess(d['contributions'][e['event_id']]['mean'],0)
        self.assertGreater(d['seed_mean_and_range']['mean'],0)

    def test_zero_gap_has_no_manufactured_contribution_fraction(self):
        f=self.fixture()
        for p in f['predictions'].values():p[:]=f['truth']+1.
        d=m.analyze_gauge(**f)['summary']['decomposition']['LSTM']
        self.assertEqual(d['seed_mean_and_range']['mean'],0.)
        self.assertTrue(all(v is None for v in d['signed_fraction_of_gap'].values()))

    def test_raw_negative_predictions_are_not_clipped(self):
        f=self.fixture();f['predictions']['LSTM',17][:]=-1.
        r=m.analyze_gauge(**f)['summary']['events'][0]['models']['LSTM']['seeds']['17']['paired'][0]
        self.assertEqual(r['at_observed_peak']['prediction_q_cms'],-1.)
        self.assertEqual(r['at_observed_peak']['error_cms'],-101.)


if __name__=='__main__':unittest.main()
