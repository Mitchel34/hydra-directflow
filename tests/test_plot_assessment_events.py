"""Synthetic figures and preserved-gap checks; no actual assessment values."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hydra_directflow.assessment_event_diagnostics import analyze_gauge,select_events,REFERENCE_START,REFERENCE_END
from hydra_directflow.reference import ASSESSMENT_START,ASSESSMENT_END,HOUR

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('event_plot_test',ROOT/'scripts/plot_assessment_events.py')
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)


def fixture(q95=20.):
    origin=np.arange(ASSESSMENT_START,ASSESSMENT_END,HOUR,dtype=np.int64)
    t=np.arange(REFERENCE_START,REFERENCE_END,HOUR,dtype=np.int64);x=np.arange(len(t));q=np.full(len(t),5.)
    for center,height in [(500,100),(1800,80),(3500,60)]:q+=height*np.exp(-((x-center)/12.)**2)
    valid=np.ones(len(t),bool);valid[510]=False;observed=q.copy();observed[~valid]=np.nan
    future=origin[:,None]+np.arange(1,19)*HOUR;ix=(future-REFERENCE_START)//HOUR
    raw=q[ix];truth=observed[ix];mask=valid[ix];admitted=np.ones(8760,bool);admitted[494:497]=False
    nwm=raw*.93;nwm[:,16:]=np.nan;nmask=np.ones((8760,18),bool);nmask[:,16:]=False
    nmask[499:501,:16]=False;nwm[~nmask]=np.nan
    predictions={}
    for a in plot.ARCHITECTURES:
        for seed in (17,29,43):
            p=raw*(.72+(seed-17)*.001);p[~admitted]=np.nan;predictions[a,seed]=p
    history=np.ones((8760,168,2));history[:,:,-1]=np.arange(169,1,-1)
    catalog=select_events(t,observed,valid,q95)
    result=analyze_gauge(origin,truth,mask,admitted,predictions,nwm,nmask,history,q95,catalog)
    arrays={k:v for k,v in result['plot'].items() if not k.endswith('_by_arch')}
    for kind in ('mean','min','max'):
        for a,v in result['plot']['prediction_'+kind+'_by_arch'].items():arrays['prediction_'+kind+'__'+a]=v
    return result['summary'],arrays


class PlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.summary,cls.arrays=fixture()

    def test_full_calendar_and_common_support(self):
        events=plot.validate(self.summary,self.arrays,'03161000');self.assertEqual(len(events),3)

    def test_missing_hour_does_not_bridge_fixed_lead_trace(self):
        item=self.summary['events'][0];times,obs=plot._window(item,self.arrays)
        mask,curves=plot.fixed_lead_trace(self.arrays,times,1)
        self.assertEqual(len(times),120);self.assertTrue(np.isnan(obs).any())
        self.assertTrue((~mask).any())
        for series in curves.values():
            self.assertTrue(np.isnan(series['mean'][~mask]).all())
            self.assertTrue(np.isfinite(series['mean'][mask]).all())
        self.assertTrue(np.all(np.diff(times)==HOUR))

    def test_no_unsupported_h18_nwm_trace(self):
        with self.assertRaises(ValueError):plot.fixed_lead_trace(self.arrays,np.array([REFERENCE_START]),18)

    def test_summary_support_mismatch_rejected(self):
        s=deepcopy(self.summary);s['events'][0]['nwm_paired'][0]['n']+=1
        with self.assertRaisesRegex(ValueError,'support'):plot.validate(s,self.arrays,'03161000')

    def test_same_count_shifted_model_lead_rejected(self):
        s=deepcopy(self.summary);s['events'][0]['models']['LSTM']['seed_mean_and_range'][0]['lead_hours']=2
        with self.assertRaisesRegex(ValueError,'support'):plot.validate(s,self.arrays,'03161000')

    def test_event_selection_and_anchor_change_rejected(self):
        s=deepcopy(self.summary);s['events']=list(reversed(s['events']))
        with self.assertRaisesRegex(ValueError,'replace selected'):plot.validate(s,self.arrays,'03161000')
        s=deepcopy(self.summary);s['events'][0]['event']['observed_peak_q_cms']+=1
        with self.assertRaises(ValueError):plot.validate(s,self.arrays,'03161000')

    def test_decomposition_must_reconcile(self):
        s=deepcopy(self.summary);s['decomposition']['LSTM']['contributions']['REST_OF_ASSESSMENT']['mean']+=.1
        with self.assertRaisesRegex(ValueError,'reconcile'):plot.validate(s,self.arrays,'03161000')

    def test_anchor_error_percent_and_missingness(self):
        item=deepcopy(self.summary['events'][0]);item['nwm_paired'][0]['at_observed_peak']['ratio_to_observed_peak']=.6
        item['nwm_paired'][1]['at_observed_peak']['ratio_to_observed_peak']=None
        values=plot.event_metric_series(item,'NWM','anchor_percent')['mean']
        self.assertAlmostEqual(values[0],-40);self.assertTrue(np.isnan(values[1]))

    def test_render_three_outputs_and_preserve_inputs(self):
        before=self.arrays['observed_q'].copy();pred=self.arrays['prediction_mean__LSTM'].copy()
        with tempfile.TemporaryDirectory() as tmp:
            result=plot.plot_gauge(self.summary,self.arrays,tmp,'03161000')
            for family in ('hydrographs','lead_errors','decomposition'):
                for kind in ('svg','pdf','png'):self.assertGreater(Path(result[family][kind]).stat().st_size,1000)
            self.assertEqual(result['traces']['HYDROGRAPH_PLOT_TRACE']['rows'],1440)
            self.assertEqual(result['traces']['EVENT_METRIC_PLOT_TRACE']['rows'],480)
            self.assertEqual(result['traces']['NSE_GAP_PLOT_TRACE']['rows'],16)
            svg=Path(result['hydrographs']['svg']).read_text()
            self.assertIn('Fixed-lead rolling-origin hydrographs',svg)
            self.assertIn('not confidence intervals',svg)
        np.testing.assert_array_equal(before,self.arrays['observed_q'])
        np.testing.assert_array_equal(pred,self.arrays['prediction_mean__LSTM'])

    def test_explicit_empty_catalog_is_valid_without_replacement(self):
        summary,arrays=fixture(q95=10000.)
        self.assertEqual(plot.validate(summary,arrays,'03161000'),[])

    def test_focused_pair_uses_existing_first_events_only(self):
        summaries={g:self.summary for g in plot.GAUGES};arrays={g:self.arrays for g in plot.GAUGES}
        with tempfile.TemporaryDirectory() as tmp:
            result=plot.plot_largest_pair(summaries,arrays,tmp)
            self.assertEqual(result['trace']['rows'],480)
            for kind in ('svg','pdf','png'):self.assertGreater(Path(result[kind]).stat().st_size,1000)
        bad=deepcopy(summaries);bad['03161000']['events'][0]['event']['observed_peak_rank']=2
        with self.assertRaises(ValueError):plot.plot_largest_pair(bad,arrays,'/not-created')


if __name__=='__main__':unittest.main()
