"""Synthetic complete A6 reporting matrix; no models or study values."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest

import numpy as np

from hydra_directflow.coverage_extension_scoring import aggregate_seed_reports,PRODUCTS,COHORTS,SEEDS,METRICS,ORIGINS
from hydra_directflow.contract import ARCHITECTURES,GAUGE_IDS
from hydra_directflow.reference import digest

PATH=Path(__file__).resolve().parents[1]/'scripts/report_coverage_extension.py'
SPEC=importlib.util.spec_from_file_location('coverage_report_tested',PATH)
reporter=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(reporter)


def fixture():
    blobs={};products=[];aggregates=[];all_scores={};phase='a'*64;barrier='b'*64
    counts=dict(ALL_REQUESTED=ORIGINS,ORIGINAL_COMMON=8000,ADDITIONAL_L168=100,L24_FALLBACK=400,UNAVAILABLE=266)
    routes={'L168':8100,'L24':400,'UNAVAILABLE':266}
    common_mask=np.arange(ORIGINS)<8000;route_array=np.full(ORIGINS,'UNAVAILABLE',dtype='<U11');route_array[:8100]='L168';route_array[8100:8500]='L24'
    router_sha=reporter._array_digest({'route':route_array,'eligible168':route_array=='L168','eligible24':route_array!='UNAVAILABLE'})
    common_sha=reporter._array_digest({'cohort':common_mask});plot_p=np.full((ORIGINS,18),25.,np.float64);plot_p[8500:]=np.nan
    routed_p_sha=reporter._array_digest({'predictions':plot_p})
    source_mask=np.ones(ORIGINS+18,bool);source_mask[8018:8101]=False
    target_mask=source_mask[np.arange(ORIGINS)[:,None]+np.arange(1,19)]
    label_sha=reporter._array_digest({'labels':np.full((ORIGINS,18),20.,np.float64)})
    label_mask_sha=reporter._array_digest({'label_mask':target_mask})
    def add(name,value):
        raw=json.dumps(value).encode();blobs[name]=raw;return dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def panel(gauge,model,seed,product,cohort):
        requested=counts[cohort];rows=[]
        for h in range(1,19):
            long=counts['ORIGINAL_COMMON']+counts['ADDITIONAL_L168'] if cohort=='ALL_REQUESTED' else requested if cohort in ('ORIGINAL_COMMON','ADDITIONAL_L168') else 0
            short=8500 if cohort=='ALL_REQUESTED' else requested if cohort!='UNAVAILABLE' else 0
            issued=8000 if product==PRODUCTS[0] and cohort in ('ALL_REQUESTED','ORIGINAL_COMMON') else 0 if product==PRODUCTS[0] else long if product==PRODUCTS[1] else short
            eligible=long if product in PRODUCTS[:2] else short
            invalid=83 if cohort in ('ALL_REQUESTED','ADDITIONAL_L168') else 0
            valid=requested-invalid;n=issued-(83 if cohort in ('ALL_REQUESTED','ADDITIONAL_L168') and product!=PRODUCTS[0] else 0)
            n=max(0,n);days=min(366,n);qualified=n>=168 and days>=28
            long_error=2.+GAUGE_IDS.index(gauge)*.5+ARCHITECTURES.index(model)*.1+seed/1000
            short_error=long_error+1
            if not n:mse=mae=bias=nse=rmse=neg=None
            else:
                fallback=400 if product==PRODUCTS[3] and cohort=='ALL_REQUESTED' else n if product==PRODUCTS[3] and cohort=='L24_FALLBACK' else n if product==PRODUCTS[2] else 0
                mse=((n-fallback)*long_error**2+fallback*short_error**2)/n;mae=bias=((n-fallback)*long_error+fallback*short_error)/n
                rmse=math.sqrt(mse);nse=1-mse/100;neg=0.
            coordinate=digest([gauge,cohort,h,n]);mask=digest([gauge,cohort,h,n,'mask'])
            rows.append(dict(lead_hours=h,n=n,n_requested=requested,requested_origins=requested,input_eligible_origins=eligible,
                emitted_forecasts=issued,valid_target_count=valid,evaluable_forecasts=n,data_unavailable_origins=requested-eligible,
                policy_not_emitted_with_eligible_inputs=eligible-issued,not_emitted_origins=requested-issued,invalid_or_missing_targets=invalid,
                technical_failures=0,distinct_target_hours=n,distinct_utc_dates=days,negative_predictions_scored=0,negative_predictions_emitted=0,
                forecast_availability_fraction=issued/requested if requested else None,evaluable_coverage_fraction=n/requested if requested else None,
                coverage_among_valid_targets=n/valid if valid else None,negative_frequency_among_emitted=0. if issued else None,
                physical_mse=mse,rmse=rmse,mae=mae,bias=bias,nse=nse,negative_frequency=neg,
                primary_eligible=qualified,primary_ineligibility=[] if qualified else ['TOO_FEW_DISTINCT_TARGET_HOURS'],
                undefined={} if n else {k:'ZERO_SUPPORT' for k in METRICS},scored_coordinate_sha256=coordinate,scoring_mask_sha256=mask))
        primary=math.fsum(row['nse']/18 for row in rows) if all(x['primary_eligible'] for x in rows) else None
        return dict(status='PRIMARY_DEFINED' if primary is not None else 'PRIMARY_UNAVAILABLE',requested_origins=requested,
            cohort_origins=requested,primary_score=primary,leads=rows,origin_cohort_sha256=common_sha if cohort=='ORIGINAL_COMMON' else digest(cohort),
            unique_evaluable_target_hours=max(x['n'] for x in rows)+17 if any(x['n'] for x in rows) else 0)
    for gauge in GAUGE_IDS:
        for model in ARCHITECTURES:
            for seed in SEEDS:
                for product in PRODUCTS:
                    key=(gauge,model,seed,product);prefix='/'.join(map(str,key))
                    identity=dict(gauge_id=gauge,architecture=model,seed=seed,product=product,phase_plan_sha256=phase,
                        prediction_barrier_sha256=barrier,checkpoint_sha256_by_history={'168':digest([model,seed,168]),'24':digest([model,seed,24])},
                        prediction_sha256=digest(key),input_source_sha256=digest([gauge,product]),label_source_sha256=digest(gauge),route_map_sha256=digest([gauge,'route']))
                    score=dict(schema='hydra-phase-a-per-lead-product-score/v1',identity=identity,origin_population=ORIGINS,
                        history_identity=dict(mode='MIXED_L168_L24_BY_ORIGIN' if product==PRODUCTS[3] else 'FIXED_CHECKPOINT_HISTORY',
                            fixed_history_hours=None if product==PRODUCTS[3] else 24 if product==PRODUCTS[2] else 168),
                        shared_router_origin_counts=routes,cohorts={c:panel(gauge,model,seed,product,c) for c in COHORTS},
                        availability_denominator_is_single_gauge_origin=True,input_admission_independent_of_verification=True,
                        routed_weather_soil_attribution_supported=False,negative_predictions_clipped=False,targets_filled=False,
                        optimizer_updates=0,scaler_fits=0,protected_assessment_access=False,
                        coordinates_sha256='1'*64,label_mask_sha256=label_mask_sha,labels_sha256=label_sha,computed_router_choices_sha256=router_sha,
                        predictions_sha256=routed_p_sha if product==PRODUCTS[3] else digest(key))
                    all_scores[key]=score
                    high_count=70 if product==PRODUCTS[0] else 75 if product==PRODUCTS[1] else 90
                    high=dict(identity=identity,q95_cms=100.,known_origin_lead_cells=1800,evaluable_origin_lead_cells=18*high_count,
                        known_unique_high_target_hours=100,unique_high_hours_with_any_supported_lead=high_count,
                        unique_high_hours_with_all18_supported_leads=high_count,unique_high_hours_with_all18_candidate_leads=100,
                        per_lead=[dict(lead_hours=h,known_high_target_cells=100,evaluable_high_forecasts=high_count,
                                      coverage_among_known_high_targets=high_count/100) for h in range(1,19)],protected_assessment_access=False)
                    event_rows=[]
                    for i in range(2):
                        leads=[]
                        for h in range(1,19):
                            issued=0 if i==1 and product in PRODUCTS[:2] else 120
                            leads.append(dict(lead_hours=h,n=issued,requested_origins=120,input_eligible_origins=issued,emitted_forecasts=issued,
                                valid_target_count=120,evaluable_forecasts=issued,peak_forecast_available=bool(issued),peak_candidate_requested=True,
                                peak_target_admissible=True,rmse=2. if issued else None,mae=2. if issued else None,bias=2. if issued else None,
                                physical_mse=4. if issued else None,negative_frequency=0. if issued else None,
                                observed_peak_forecast=dict(observed_q_cms=120.,prediction_q_cms=122.,error_cms=2.) if issued else None,
                                undefined={} if issued else {'rmse':'ZERO_SUPPORT'}))
                        supported=i==0 or product in PRODUCTS[2:]
                        event_rows.append(dict(event_id=f'2024-01-{10+i:02d}T00:00:00Z',rank=i+1,representative_top3=True,
                            start_utc=f'2024-01-{8+i:02d}T00:00:00Z',end_exclusive_utc=f'2024-01-{13+i:02d}T00:00:00Z',
                            observed_peak_supported_leads=list(range(1,19)) if supported else [],observed_peak_candidate_leads=list(range(1,19)),
                            reference_approved_hours=120,reference_requested_hours=120,
                            metrics=dict(leads=leads,unique_evaluable_target_hours=120 if supported else 0,evaluable_origin_lead_cells=2160 if supported else 0)))
                    events=dict(identity=identity,catalog_event_count=2,events_with_any_evaluable_forecast=1 if product in PRODUCTS[:2] else 2,
                        observed_peaks_with_any_forecast=1 if product in PRODUCTS[:2] else 2,observed_peaks_with_all18_forecasts=1 if product in PRODUCTS[:2] else 2,
                        events=event_rows,representative_events_reselected=False,events_with_no_forecasts_retained=True,
                        episodes_are_proven_independent=False,annual_NSE_gate_used_for_short_events=False,protected_assessment_access=False)
                    artifacts={name:add(prefix+'/'+name+'.json',value) for name,value in dict(score=score,high_flow=high,events=events,
                        verification=dict(status='INDEPENDENT_METRICS_AND_COUNTS_MATCH',lead_panels_checked=90,relative_absolute_tolerance=1e-10)).items()}
                    products.append(dict(gauge_id=gauge,architecture=model,seed=seed,product=product,artifacts=artifacts))
            for product in PRODUCTS:
                aggregate=aggregate_seed_reports({seed:all_scores[gauge,model,seed,product] for seed in SEEDS})
                aggregates.append(dict(gauge_id=gauge,architecture=model,product=product,report=add(f'aggregate/{gauge}/{model}/{product}.json',aggregate)))
    index=dict(schema='hydra-phase-a-evaluation-index/v1',status='ALL144_PRODUCTS_SCORED_AND_INDEPENDENTLY_VERIFIED',
        phase_plan_sha256=phase,prediction_barrier_sha256=barrier,original_plan_sha256='c'*64,products=products,aggregates=aggregates,
        original_common_checks=[dict(job_id=f'final_development__{g}__{a}__full_L{h}__seed{s}',status='ORIGINAL_COMMON_METRICS_REPRODUCED')
            for g in GAUGE_IDS for a in ARCHITECTURES for h in (168,24) for s in SEEDS],
        coverage_by_gauge={g:dict(primary_reason_counts={'COMPLETE_L168':8100,'OLDER_Q_HISTORY_GAP_USE_L24':400,'RECENT_Q_HISTORY_INCOMPLETE':266},
            overlapping_input_reasons={'recent_q_gap':266}) for g in GAUGE_IDS},protected_assessment_access=False)
    accounting=dict(plan_sha256=phase,entered_cases=600000,confirmed_cases=600000,entered_batches=9600,confirmed_batches=9600,
                    entered_models=120,confirmed_models=120,read_reserved=1_000_000,output_reserved=20_000_000,
                    infrastructure_resumes=0,peak_combined_rss=2*1024**3,peak_mps=1024**3,minimum_free=10*1024**3,
                    started=1000.,last_clock=2000.,deadline=22600.,original_campaign_deadline=30000.,
                    original_optimizer_updates_before=1151690,original_optimizer_updates_after=1151690)
    return index,blobs,accounting


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=fixture()
    def setUp(self):
        self.index,self.blobs,self.accounting=deepcopy(self.base);self.reads=[]
    def read(self,d):
        self.reads.append(d['path']);body=self.blobs[d['path']]
        self.assertEqual(len(body),d['bytes']);self.assertEqual(hashlib.sha256(body).hexdigest(),d['sha256']);return json.loads(body)
    def change(self,d,fn):
        value=json.loads(self.blobs[d['path']]);fn(value);raw=json.dumps(value).encode();self.blobs[d['path']]=raw
        d.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def build(self):return reporter.build_publication(self.index,self.read,self.accounting)

    def test_complete_publication_retains_all_models_seeds_cohorts_and_unique_origin_denominators(self):
        result=self.build();s=result['summary']
        self.assertEqual(len(result['files']),50);self.assertEqual(len(s['primary_scores']),48)
        self.assertEqual(len(s['paired_routed_vs_l24']),12)
        self.assertTrue(s['all144_product_seed_reports_retained']);self.assertFalse(s['counts_multiplied_by_models_or_seeds'])
        for g in s['coverage_by_gauge']:
            self.assertEqual(g['candidate_origins'],8766);self.assertEqual(g['route_counts'],{'L168':8100,'L24':400,'UNAVAILABLE':266})
        for name,body in result['files'].items():
            self.assertLess(len(body),1_000_000)
            if name.startswith('details/'):
                value=json.loads(body);self.assertEqual(set(value['seeds']),{'17','29','43'})
                self.assertEqual(set(value['aggregate']),set(COHORTS))

    def test_private_index_accounting_fields_and_target_descriptors_are_not_published_or_read(self):
        marker='/'+ 'Users/'+'synthetic/private/target.npz'
        self.index['targets_by_gauge']={g:{'path':marker} for g in GAUGE_IDS};self.index['private_note']=marker
        self.accounting['tasks']=[{'private':marker}]
        result=self.build()
        self.assertNotIn(marker.encode(),b''.join(result['files'].values()));self.assertNotIn(marker,self.reads)

    def test_missing_product_or_seed_is_not_dropped(self):
        self.index['products'].pop()
        with self.assertRaisesRegex(ValueError,'All144'):self.build()

    def test_incomplete_prediction_or_evaluation_barrier_refuses_before_artifact_reads(self):
        self.index['status']='PENDING_PREDICTIONS'
        with self.assertRaisesRegex(ValueError,'Completed development'):self.build()
        self.assertEqual(self.reads,[])

    def test_wrong_phase_or_accounting_cap_cannot_be_hidden(self):
        self.accounting['plan_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'Accounting'):self.build()
        self.accounting['plan_sha256']=self.index['phase_plan_sha256'];self.accounting['entered_cases']=1280001
        with self.assertRaisesRegex(ValueError,'inference accounting'):self.build()

    def test_original_model_checks_and_all_finite_resource_limits_remain_required(self):
        for key,value in [('read_reserved',8*1024**3+1),('output_reserved',1024**3+1),('infrastructure_resumes',2),
                          ('peak_combined_rss',4*1024**3+1),('peak_mps',2*1024**3+1),('minimum_free',8*1024**3-1),
                          ('last_clock',22601.),('original_optimizer_updates_after',1151691)]:
            old=self.accounting[key];self.accounting[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.build()
            self.accounting[key]=old
        self.index['original_common_checks'].pop()
        with self.assertRaisesRegex(ValueError,'All72'):self.build()

    def test_model_specific_availability_is_rejected_not_averaged(self):
        record=next(r for r in self.index['products'] if r['seed']==29 and r['product']==PRODUCTS[0])
        self.change(record['artifacts']['high_flow'],lambda d:d.update(evaluable_origin_lead_cells=1))
        with self.assertRaisesRegex(ValueError,'Coverage cannot depend'):self.build()

    def test_missing_independent_verification_blocks_success_publication(self):
        self.change(self.index['products'][0]['artifacts']['verification'],lambda x:x.update(status='NOT_VERIFIED'))
        with self.assertRaisesRegex(ValueError,'Independent metric'):self.build()

    def test_aggregate_means_are_independently_recomputed(self):
        self.change(self.index['aggregates'][0]['report'],lambda d:d['cohorts']['ALL_REQUESTED'].update(primary_score=0.999999))
        with self.assertRaisesRegex(ValueError,'aggregation differs'):self.build()

    def test_mixed_route_cannot_be_relabelled_L168(self):
        record=next(r for r in self.index['products'] if r['product']==PRODUCTS[3])
        self.change(record['artifacts']['score'],lambda d:d['history_identity'].update(mode='FIXED_CHECKPOINT_HISTORY',fixed_history_hours=168))
        with self.assertRaisesRegex(ValueError,'cannot claim one fixed history'):self.build()

    def test_unsupported_cohorts_remain_unavailable_not_zero(self):
        result=self.build();rows=result['summary']['primary_scores']
        self.assertTrue(all(row['cohorts']['UNAVAILABLE']['primary_score'] is None for row in rows))
        self.assertTrue(all(row['cohorts']['ADDITIONAL_L168']['primary_score'] is None for row in rows))
        self.assertIn('not a paired improvement',result['files']['review.md'].decode())
        self.assertFalse(result['summary']['routed_weather_soil_attribution_supported'])

    def event_plot_data(self,summary):
        origin=np.arange(reporter.START,reporter.END,3600,dtype=np.int64);future=origin[:,None]+np.arange(1,19)*3600
        route=np.full(ORIGINS,'UNAVAILABLE',dtype='<U11');route[:8100]='L168';route[8100:8500]='L24'
        common=np.arange(ORIGINS)<8000;p=np.full((4,3,ORIGINS,18),25.,np.float64);p[:,:,8500:]=np.nan
        times=np.arange(reporter.START,reporter.ASSESSMENT_START,3600,dtype=np.int64)
        mask=np.ones(len(times),bool);mask[8018:8101]=False;values=np.full(len(times),20.,np.float64);values[~mask]=np.nan
        return dict(phase_plan_sha256=summary['phase_plan_sha256'],prediction_barrier_sha256=summary['prediction_barrier_sha256'],
            origin_time=origin,forecast_time=future,route=route,original_common=common,routed_predictions=p,
            reference_time=times,reference_value=values,reference_mask=mask,
            verification_target_values=np.full((ORIGINS,18),20.,np.float64),
            verification_target_mask=mask[np.arange(ORIGINS)[:,None]+np.arange(1,19)])

    def test_fixed_event_plot_data_retains_all_seeds_and_refuses_missing_predictions(self):
        summary=self.build()['summary'];data=self.event_plot_data(summary)
        self.assertIs(reporter.validate_event_plot_data(summary,GAUGE_IDS[0],data),data)
        data['routed_predictions'][2,1,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'all|All four'):
            reporter.validate_event_plot_data(summary,GAUGE_IDS[0],data)

    def test_event_plot_barrier_and_protected_coordinate_checks_precede_prediction_values(self):
        summary=self.build()['summary'];data=self.event_plot_data(summary)
        class Poison:
            def __array__(self,*args,**kwargs):raise AssertionError('Predictions inspected before scope')
        data['routed_predictions']=Poison();data['phase_plan_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'prediction barrier'):
            reporter.validate_event_plot_data(summary,GAUGE_IDS[0],data)
        data['phase_plan_sha256']=summary['phase_plan_sha256'];data['forecast_time'][-1,-1]=reporter.ASSESSMENT_START
        with self.assertRaisesRegex(ValueError,'H18-safe'):
            reporter.validate_event_plot_data(summary,GAUGE_IDS[0],data)

    def test_event_plot_values_must_match_scored_prediction_and_target_identities(self):
        summary=self.build()['summary']
        data=self.event_plot_data(summary);data['routed_predictions'][0,0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'Plotted forecasts differ'):
            reporter.validate_event_plot_data(summary,GAUGE_IDS[0],data)
        data=self.event_plot_data(summary);data['reference_value'][10]+=1
        with self.assertRaisesRegex(ValueError,'Plotted observations differ'):
            reporter.validate_event_plot_data(summary,GAUGE_IDS[0],data)

    def test_eleven_event_catalog_still_fits_unmodified_publication_size_limit(self):
        for record in self.index['products']:
            def expand(d):
                original=d['events'];new=[]
                for i in range(11):
                    row=deepcopy(original[0 if i==0 else 1]);row.update(event_id=f'2024-01-{10+i:02d}T00:00:00Z',rank=i+1,representative_top3=i<3);new.append(row)
                d['events']=new;d['catalog_event_count']=11
                supported=1 if d['identity']['product'] in PRODUCTS[:2] else 11
                for key in ('events_with_any_evaluable_forecast','observed_peaks_with_any_forecast','observed_peaks_with_all18_forecasts'):d[key]=supported
            self.change(record['artifacts']['events'],expand)
        result=self.build()
        self.assertTrue(all(len(body)<1_000_000 for body in result['files'].values()))


if __name__=='__main__':unittest.main()
