"""Synthetic Phase A scoring contracts; no study targets, models or file I/O."""
from copy import deepcopy
import hashlib
import json
import unittest

import numpy as np

from hydra_directflow.coverage_extension_scoring import (
    START, END, ORIGINS, HOUR, SEEDS, PRODUCTS, COHORTS,
    validate_calendar, verification_masks, assemble_routed, score_product,
    verify_product_report, aggregate_seed_reports, high_flow_coverage, fixed_events,
)
from hydra_directflow.coverage_analysis import select_observed_events, event_coverage, coverage_by_regime, iso
from hydra_directflow.post_events import POPULATION, CATALOG_RULE
from hydra_directflow.reference import ASSESSMENT_START


def fixture(*, missing_hour=500, constant=False, simple_routes=False):
    origins=np.arange(START,END,HOUR,dtype=np.int64)
    future=origins[:,None]+np.arange(1,19,dtype=np.int64)*HOUR
    values=(20.+np.arange(ORIGINS+18,dtype=np.float64)/100+np.sin(np.arange(ORIGINS+18)/24))
    if constant:values[:]=20.
    flags=np.full(len(values),'["A"]',dtype='<U128');present=np.ones(len(values),bool)
    if missing_hour is not None:flags[missing_hour]='["P"]'
    positions=np.arange(ORIGINS)[:,None]+np.arange(1,19)[None,:]
    labels=values[positions];qualifier=flags[positions]
    target=verification_masks(gauge_id='03161000',origins=origins,forecast_time=future,
                             labels=labels,qualifier_json=qualifier,present_mask=present[positions])
    long=np.ones(ORIGINS,bool);short=np.ones(ORIGINS,bool)
    if not simple_routes:
        long[100:110]=False;long[200:205]=False;short[200:205]=False
    route=np.where(long,'L168',np.where(short,'L24','UNAVAILABLE'))
    common=long&target['label_mask'].all(axis=1)
    return dict(origins=origins,forecast_time=future,labels=labels,label_mask=target['label_mask'],
                original_common=common,eligible168=long,eligible24=short,route=route),qualifier


def identity(product='ROUTED_CLEAN_168_24_V1',seed=17):
    required=('168','24') if product=='ROUTED_CLEAN_168_24_V1' else ('24',) if product=='L24_INPUT_ONLY_V1' else ('168',)
    return dict(gauge_id='03161000',architecture='LSTM',seed=seed,product=product,
        checkpoint_sha256_by_history={k:hashlib.sha256(f'{seed}-{k}'.encode()).hexdigest() for k in required},phase_plan_sha256='a'*64,
        route_map_sha256='b'*64,prediction_sha256=str(seed%10)*64,prediction_barrier_sha256='d'*64,
        input_source_sha256='e'*64,label_source_sha256='f'*64)


def predictions(args,product,offset=2.):
    mask=args['original_common'] if product=='PRIMARY_FROZEN' else args['eligible168'] if product=='L168_INPUT_ONLY_V1' else args['eligible24']
    result=np.full_like(args['labels'],np.nan);result[mask]=args['labels'][mask]+offset
    return result


def event_fixture():
    args,_=fixture(missing_hour=None,simple_routes=True)
    times=np.arange(START,ASSESSMENT_START,HOUR,dtype=np.int64)
    values=np.full(len(times),3.,np.float64);values[100]=15.;values[300]=20.;values[500]=12.
    admitted=np.ones(len(times),bool)
    reference=dict(times=times,values=values,admitted=admitted)
    thresholds=dict(q10_cms=1.,q90_cms=6.,q95_cms=10.,
        fit_start_utc='2018-07-12T00:00:00Z',fit_end_exclusive_utc='2023-05-01T00:00:00Z',
        refit_at_utc=iso(START),fit_population=POPULATION,stopping_or_outer_values_used=False,
        model_scaler=False,simulated_label_delay_hours=2,quantile_method='numpy.linear')
    positions=(args['forecast_time']-START)//HOUR;args['labels']=values[positions]
    args['label_mask']=admitted[positions]
    args['eligible168'][250:341]=False
    args['eligible168'][480:501]=False;args['eligible24'][480:501]=False
    args['route']=np.where(args['eligible168'],'L168',np.where(args['eligible24'],'L24','UNAVAILABLE'))
    args['original_common']=args['eligible168'].copy()
    events=select_observed_events(reference,thresholds,start=START,end=ASSESSMENT_START)
    for event in events:
        cov=event_coverage(event,args['origins'],args['eligible168'],args['original_common'],args['label_mask'],
                           outer_start=START,outer_end=ASSESSMENT_START)
        event['all_lead_requested_cells']=sum(x['requested_cells'] for x in cov)
        event['all_lead_common_comparison_cells']=sum(x['common_comparison_cells'] for x in cov)
        event['observed_peak_supported_leads']=[x['lead_hours'] for x in cov if x['observed_peak_has_common_prediction_case']]
        if event['representative_top3']:event['forecast_case_coverage_by_lead']=cov
    catalog=dict(gauge_id='03161000',fold='final_development',thresholds=thresholds,
        outer_window=dict(start_utc=iso(START),end_exclusive_utc=iso(ASSESSMENT_START)),
        representative_event_rule=CATALOG_RULE,event_selection_uses_model_errors=False,
        event_windows_clipped_to_outer=True,event_windows_hours=[-48,72],
        observed_high_flow_event_count=len(events),observed_high_flow_events=events,
        target_cell_regime_coverage=coverage_by_regime(args['origins'],args['eligible168'],args['original_common'],args['label_mask'],reference,thresholds,start=START,end=ASSESSMENT_START))
    return args,catalog


class PhaseAScoringTests(unittest.TestCase):
    def test_exact_final_calendar_and_protected_failure_before_outcomes(self):
        args,_=fixture();bad=args['forecast_time'].copy();bad[-1,-1]=ASSESSMENT_START
        class Poison:
            def __array__(self,*unused,**kw):raise AssertionError('Outcome decoded before coordinate guard')
        with self.assertRaisesRegex(ValueError,'Protected/crossing'):
            score_product(identity=identity(),**{**args,'forecast_time':bad,'labels':Poison()},predictions=Poison())
        with self.assertRaisesRegex(ValueError,'complete final-development'):
            validate_calendar(args['origins'][:-1],args['forecast_time'][:-1],'03161000')

    def test_quality_rule_rejects_combinations_without_rewriting_qualifiers(self):
        args,_=fixture(missing_hour=None);flags=np.full(args['labels'].shape,'["A"]',dtype='<U128')
        items=[['A'],['A','R'],['R','A','A'],['R'],['A','P'],['A','e'],['A','unknown'],[],None]
        for i,x in enumerate(items):flags[0,i]=json.dumps(x)
        before=flags.copy();values=args['labels'].copy();values[1,0]=np.nan;values[1,1]=-1.;present=np.ones_like(args['label_mask']);present[1,2]=False
        result=verification_masks(gauge_id='03161000',origins=args['origins'],forecast_time=args['forecast_time'],
                                  labels=values,qualifier_json=flags,present_mask=present)
        np.testing.assert_array_equal(result['label_mask'][0,:9],[True,True,True,False,False,False,False,False,False])
        self.assertFalse(result['label_mask'][1,:3].any());np.testing.assert_array_equal(flags,before)
        self.assertEqual(result['exclusion_reason'][1,2],'NO_EXACT_HOUR_RECORD')
        self.assertFalse(result['targets_filled'])

    def test_malformed_qualifier_encoding_is_failure_not_ordinary_exclusion(self):
        args,flags=fixture();flags=flags.copy();flags[0,0]='not-json'
        with self.assertRaisesRegex(ValueError,'Malformed'):
            verification_masks(gauge_id='03161000',origins=args['origins'],forecast_time=args['forecast_time'],
                               labels=args['labels'],qualifier_json=flags,present_mask=np.ones_like(args['label_mask']))

    def test_one_missing_H3_target_does_not_cancel_H18(self):
        args,_=fixture(simple_routes=True);args['eligible168'][497]=False
        args['route']=np.where(args['eligible168'],'L168','L24');args['original_common']=args['eligible168']&args['label_mask'].all(axis=1)
        report=score_product(identity=identity(),**args,predictions=predictions(args,'ROUTED_CLEAN_168_24_V1'))
        panel=report['cohorts']['L24_FALLBACK']
        self.assertEqual(panel['leads'][2]['n'],0);self.assertEqual(panel['leads'][17]['n'],1)
        self.assertIsNone(panel['leads'][2]['rmse']);self.assertAlmostEqual(panel['leads'][17]['rmse'],2.)
        self.assertIsNone(panel['primary_score']);self.assertFalse(panel['leads'][17]['primary_eligible'])

    def test_requested_nonfinite_prediction_fails_even_when_target_invalid(self):
        args,_=fixture(simple_routes=True);p=predictions(args,'L168_INPUT_ONLY_V1');p[497,2]=np.nan
        self.assertFalse(args['label_mask'][497,2])
        with self.assertRaisesRegex(ValueError,'Missing/nonfinite requested'):
            score_product(identity=identity('L168_INPUT_ONLY_V1'),**args,predictions=p)

    def test_invalid_admitted_reference_rejected_and_unknown_history_identity_rejected(self):
        args,_=fixture();p=predictions(args,'ROUTED_CLEAN_168_24_V1');args['labels']=args['labels'].copy();args['labels'][0,0]=-1.
        with self.assertRaisesRegex(ValueError,'Admitted references'):
            score_product(identity=identity(),**args,predictions=p)
        args,_=fixture();bad=identity();bad['history_hours']=168
        with self.assertRaisesRegex(ValueError,'Exact Phase A scoring identity'):
            score_product(identity=bad,**args,predictions=predictions(args,'ROUTED_CLEAN_168_24_V1'))

    def test_fixed_router_uses_entire_correct_vector_without_labels(self):
        args,_=fixture();a=predictions(args,'L168_INPUT_ONLY_V1',2);b=predictions(args,'L24_INPUT_ONLY_V1',7)
        fields={k:args[k] for k in ('origins','forecast_time','eligible168','eligible24','route')}
        result=assemble_routed(gauge_id='03161000',**fields,predictions168=a,predictions24=b)
        np.testing.assert_array_equal(result[args['route']=='L168'],a[args['route']=='L168'])
        np.testing.assert_array_equal(result[args['route']=='L24'],b[args['route']=='L24'])
        self.assertTrue(np.isnan(result[args['route']=='UNAVAILABLE']).all())
        changed=args['labels']+999.;self.assertFalse(np.array_equal(changed,args['labels']))
        again=assemble_routed(gauge_id='03161000',**fields,predictions168=a,predictions24=b)
        np.testing.assert_array_equal(result,again)

    def test_router_cannot_choose_L24_when_L168_complete_or_violate_subset(self):
        args,_=fixture();p=predictions(args,'ROUTED_CLEAN_168_24_V1');bad=args['route'].copy();bad[0]='L24'
        with self.assertRaisesRegex(ValueError,'choose complete L168'):
            score_product(identity=identity(),**{**args,'route':bad},predictions=p)
        short=args['eligible24'].copy();short[0]=False
        with self.assertRaisesRegex(ValueError,'must imply'):
            score_product(identity=identity(),**{**args,'eligible24':short},predictions=p)

    def test_unexpected_common_membership_and_inconsistent_target_hour_fail(self):
        args,_=fixture();p=predictions(args,'ROUTED_CLEAN_168_24_V1');common=args['original_common'].copy();common[0]=False
        with self.assertRaisesRegex(ValueError,'Original common cohort must equal'):
            score_product(identity=identity(),**{**args,'original_common':common},predictions=p)
        y=args['labels'].copy();y[0,1]+=1.
        with self.assertRaisesRegex(ValueError,'Repeated approved target'):
            score_product(identity=identity(),**{**args,'labels':y},predictions=p)

    def test_four_products_have_distinct_truthful_emission_and_history(self):
        args,_=fixture();reports={product:score_product(identity=identity(product),**args,predictions=predictions(args,product)) for product in PRODUCTS}
        routed=reports['ROUTED_CLEAN_168_24_V1']
        self.assertEqual(sum(routed['shared_router_origin_counts'].values()),ORIGINS)
        self.assertEqual(sum(routed['cohorts'][name]['requested_origins'] for name in COHORTS[1:]),ORIGINS)
        self.assertIsNone(routed['history_identity']['fixed_history_hours']);self.assertEqual(routed['history_identity']['mode'],'MIXED_L168_L24_BY_ORIGIN')
        self.assertEqual(reports['L24_INPUT_ONLY_V1']['history_identity']['fixed_history_hours'],24)
        primary=reports['PRIMARY_FROZEN']['cohorts']['ALL_REQUESTED']['leads'][0]
        self.assertGreater(primary['policy_not_emitted_with_eligible_inputs'],0)
        self.assertEqual(reports['L168_INPUT_ONLY_V1']['cohorts']['ALL_REQUESTED']['leads'][0]['policy_not_emitted_with_eligible_inputs'],0)
        self.assertFalse(routed['new_population_score_difference_is_paired_improvement'])

    def test_empty_cohorts_and_unavailable_origins_do_not_disappear(self):
        args,_=fixture();args['eligible168'][:]=False;args['eligible24'][:]=False;args['route'][:]='UNAVAILABLE';args['original_common'][:]=False
        report=score_product(identity=identity(),**args,predictions=np.full_like(args['labels'],np.nan))
        self.assertEqual(report['cohorts']['UNAVAILABLE']['requested_origins'],ORIGINS)
        self.assertEqual(report['cohorts']['ALL_REQUESTED']['leads'][0]['not_emitted_origins'],ORIGINS)
        self.assertIsNone(report['cohorts']['ALL_REQUESTED']['primary_score'])
        self.assertEqual(report['cohorts']['ALL_REQUESTED']['leads'][0]['evaluable_coverage_fraction'],0)

    def test_raw_negative_predictions_are_scored_and_counted_on_all_issued(self):
        args,_=fixture();p=predictions(args,'ROUTED_CLEAN_168_24_V1');p[args['eligible24']]=-1.
        report=score_product(identity=identity(),**args,predictions=p);row=report['cohorts']['ALL_REQUESTED']['leads'][0]
        self.assertEqual(row['negative_frequency'],1.);self.assertEqual(row['negative_frequency_among_emitted'],1.)
        self.assertGreater(row['negative_predictions_emitted'],row['negative_predictions_scored'])
        self.assertFalse(report['negative_predictions_clipped'])

    def test_constant_reference_and_small_new_cohort_keep_descriptive_metrics(self):
        args,_=fixture(constant=True);report=score_product(identity=identity(),**args,predictions=predictions(args,'ROUTED_CLEAN_168_24_V1'))
        row=report['cohorts']['ALL_REQUESTED']['leads'][0];self.assertIsNone(row['nse']);self.assertEqual(row['undefined']['nse'],'CONSTANT_REFERENCE');self.assertEqual(row['rmse'],2.)
        self.assertIsNone(report['cohorts']['L24_FALLBACK']['primary_score']);self.assertEqual(report['cohorts']['L24_FALLBACK']['leads'][0]['rmse'],2.)

    def test_independent_numpy_verifier_detects_metric_and_count_corruption(self):
        args,_=fixture();p=predictions(args,'ROUTED_CLEAN_168_24_V1');report=score_product(identity=identity(),**args,predictions=p)
        checked=verify_product_report(report,identity=identity(),**args,predictions=p)
        self.assertEqual(checked['lead_panels_checked'],90)
        bad=deepcopy(report);bad['cohorts']['ALL_REQUESTED']['leads'][0]['rmse']+=.1
        with self.assertRaisesRegex(ValueError,'metric mismatch'):
            verify_product_report(bad,identity=identity(),**args,predictions=p)
        bad=deepcopy(report);bad['cohorts']['ALL_REQUESTED']['leads'][0]['n']-=1
        with self.assertRaisesRegex(ValueError,'count mismatch'):
            verify_product_report(bad,identity=identity(),**args,predictions=p)
        bad=deepcopy(report);bad['cohorts']['ALL_REQUESTED']['leads'][0]['evaluable_coverage_fraction']=1.
        with self.assertRaisesRegex(ValueError,'coverage fraction'):
            verify_product_report(bad,identity=identity(),**args,predictions=p)
        bad=deepcopy(report);bad['history_identity']['fixed_history_hours']=168
        with self.assertRaisesRegex(ValueError,'routed history identity'):
            verify_product_report(bad,identity=identity(),**args,predictions=p)

    def test_same_checkpoint_cannot_be_claimed_for_two_history_lengths(self):
        args,_=fixture();bad=identity();bad['checkpoint_sha256_by_history']['24']=bad['checkpoint_sha256_by_history']['168']
        with self.assertRaisesRegex(ValueError,'Distinct lookbacks'):
            score_product(identity=bad,**args,predictions=predictions(args,'ROUTED_CLEAN_168_24_V1'))

    def test_all_three_seed_metrics_averaged_not_prediction_ensemble(self):
        args,_=fixture();reports={seed:score_product(identity=identity(seed=seed),**args,predictions=predictions(args,'ROUTED_CLEAN_168_24_V1',offset)) for seed,offset in zip(SEEDS,(1.,2.,4.))}
        result=aggregate_seed_reports(reports);lead=result['cohorts']['ALL_REQUESTED']['leads'][0]
        self.assertAlmostEqual(lead['mean_metrics']['physical_mse'],7.)
        self.assertNotAlmostEqual(lead['mean_metrics']['physical_mse'],(7/3)**2)
        self.assertTrue(result['counts_not_summed_across_seeds']);self.assertEqual(lead['shared_counts']['requested_origins'],ORIGINS)
        self.assertTrue(lead['shared_counts']['primary_eligible'])
        tiny=result['cohorts']['L24_FALLBACK']['leads'][0]['shared_counts']
        self.assertFalse(tiny['primary_eligible']);self.assertIn('TOO_FEW_DISTINCT_TARGET_HOURS',tiny['primary_ineligibility'])
        with self.assertRaisesRegex(ValueError,'All three'):
            aggregate_seed_reports({17:reports[17],29:reports[29]})
        bad=deepcopy(reports);bad[43]['label_mask_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'populations or targets'):
            aggregate_seed_reports(bad)

    def test_fixed_events_recover_fallback_peak_and_retain_still_unavailable_peak(self):
        args,catalog=event_fixture();original=deepcopy(catalog)
        primary=fixed_events(identity=identity('PRIMARY_FROZEN'),**args,predictions=predictions(args,'PRIMARY_FROZEN'),catalog_row=catalog)
        routed=fixed_events(identity=identity(),**args,predictions=predictions(args,'ROUTED_CLEAN_168_24_V1'),catalog_row=catalog)
        self.assertEqual(catalog,original);self.assertEqual(primary['catalog_event_count'],routed['catalog_event_count'])
        self.assertFalse(primary['events'][0]['observed_peak_supported_any_lead']);self.assertTrue(routed['events'][0]['observed_peak_supported_all18_leads'])
        self.assertFalse(routed['events'][2]['observed_peak_supported_any_lead']);self.assertEqual(routed['events'][0]['metrics']['unique_evaluable_target_hours'],120)
        self.assertEqual(routed['events'][0]['metrics']['requested_origin_lead_cells'],2160)
        self.assertLess(routed['events'][0]['metrics']['requested_origins'],ORIGINS)
        lead=routed['events'][0]['metrics']['leads'][0];self.assertNotIn('nse',lead);self.assertEqual(lead['observed_peak_forecast']['error_cms'],2.)
        self.assertFalse(routed['annual_NSE_gate_used_for_short_events'])

    def test_high_flow_cell_unique_hour_and_all_lead_denominators_remain_distinct(self):
        args,catalog=event_fixture();p=predictions(args,'ROUTED_CLEAN_168_24_V1')
        result=high_flow_coverage(identity=identity(),**args,predictions=p,catalog_row=catalog)
        self.assertEqual(result['known_origin_lead_cells'],54);self.assertEqual(result['known_unique_high_target_hours'],3)
        self.assertEqual(result['unique_high_hours_with_any_supported_lead'],2);self.assertEqual(result['unique_high_hours_with_all18_supported_leads'],2)
        self.assertEqual(result['evaluable_origin_lead_cells'],36);self.assertFalse(result['availability_denominator_multiplied_by_models_or_seeds'])

    def test_catalog_cannot_backcast_final_models_or_change_peak_value(self):
        args,catalog=event_fixture();p=predictions(args,'ROUTED_CLEAN_168_24_V1');wrong=deepcopy(catalog);wrong['fold']='cv_3'
        with self.assertRaisesRegex(ValueError,'Original final-development'):
            fixed_events(identity=identity(),**args,predictions=p,catalog_row=wrong)
        changed=deepcopy(catalog);changed['observed_high_flow_events'][0]['observed_peak_q_cms']+=1.
        with self.assertRaisesRegex(ValueError,'Frozen observed peak'):
            fixed_events(identity=identity(),**args,predictions=p,catalog_row=changed)


if __name__=='__main__':unittest.main()
