"""Synthetic proposed resampling mechanics; no study arrays/models or intervals."""
from datetime import datetime, timezone
import math
import unittest

import numpy as np

from hydra_directflow.reference import digest
from hydra_directflow.scoring import ScoringPolicy, ScoreIdentity, score_physical_h18
from hydra_directflow.uncertainty import (validate_origin_calendar, contiguous_season_segments,
    draw_origin_indices, origin_multiplicities, weighted_physical_h18)

SMALL = ScoringPolicy(minimum_distinct_target_hours=2, minimum_distinct_utc_dates=1)
IDENTITY = ScoreIdentity('LSTM','synthetic','03161000','FULL_HRRR_H2_W4_S2_V1',24,'cv_1',17)


def epoch(text):
    return int(datetime.fromisoformat(text.replace('Z','+00:00')).timestamp())


def calendar(start='2020-07-31T00:00:00Z',end='2021-07-31T00:00:00Z'):
    return np.arange(epoch(start),epoch(end),3600,dtype=np.int64),dict(start_utc=start,end_exclusive_utc=end)


def fixture(n=8, policy=SMALL):
    origins=epoch('2020-01-01T00:00:00Z')+np.arange(n,dtype=np.int64)*3600
    future=origins[:,None]+np.arange(1,19)*3600
    labels=np.broadcast_to(np.arange(1,n+1,dtype=float)[:,None],(n,18)).copy()
    predictions=labels+np.arange(18)[None,:]/10-1.
    comparison=np.ones((n,18),bool)
    return dict(case_ids=np.array(['03161000:'+datetime.fromtimestamp(int(t),timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') for t in origins]),
        origins=origins,forecast_time=future,start_utc='2020-01-01T00:00:00Z',
        end_exclusive_utc=datetime.fromtimestamp(int(origins[-1])+3600,timezone.utc).isoformat(),
        labels=labels,label_mask=np.ones((n,18),bool),predictions=predictions,prediction_mask=np.ones((n,18),bool),
        comparison_mask=comparison,expected_comparison_mask_sha256=digest(comparison.tolist()),multiplicities=np.ones(n,dtype=np.int64),
        identity=IDENTITY,input_cohort_digest='a'*64,label_policy_digest='b'*64,label_source_digest='c'*64,policy=policy)


def ordinary(args):
    return score_physical_h18(**{k:v for k,v in args.items() if k in ('case_ids','origins','forecast_time','labels','label_mask',
        'predictions','prediction_mask','identity','input_cohort_digest','label_policy_digest','label_source_digest','policy')})


class CalendarTests(unittest.TestCase):
    def test_separated_jja_fragments_remain_distinct_and_djf_crosses_year(self):
        origins,kwargs=calendar()
        segments=contiguous_season_segments(origins,**kwargs)
        self.assertEqual([s['season_utc'] for s in segments],['JJA','SON','DJF','MAM','JJA'])
        self.assertEqual([s['hours'] for s in segments],[32*24,91*24,90*24,92*24,60*24])
        self.assertEqual(sum(s['hours'] for s in segments),len(origins))
        self.assertEqual(segments[2]['start_epoch'],epoch('2020-12-01T00:00:00Z'))
        self.assertEqual(segments[2]['end_exclusive_epoch'],epoch('2021-03-01T00:00:00Z'))

    def test_complete_declared_calendar_rejects_gaps_compaction_duplicates_and_reverse(self):
        origins,kwargs=calendar()
        for bad in (np.delete(origins,10),origins[::2],np.append(origins[:10],origins[9:]),origins[::-1].astype(np.uint64)):
            with self.subTest(length=len(bad)),self.assertRaisesRegex(ValueError,'Complete hourly'):
                validate_origin_calendar(bad,**kwargs)
        shifted=origins.copy();shifted[100]+=1
        with self.assertRaises(ValueError):validate_origin_calendar(shifted,**kwargs)
        with self.assertRaises(ValueError):validate_origin_calendar(origins,**{**kwargs,'start_utc':'2020-07-30T00:00:00Z'})

    def test_short_season_fragment_rejects_all_draws_without_shrinking_or_joining(self):
        origins,kwargs=calendar('2020-08-31T00:00:00Z','2021-07-31T00:00:00Z')
        for block in (168,336,672):
            with self.subTest(block=block),self.assertRaisesRegex(ValueError,'no silent shrink'):
                draw_origin_indices(origins,**kwargs,block_hours=block,replicates=1)

    def test_draws_are_reproducible_noncircular_and_preserve_every_season_length(self):
        origins,kwargs=calendar()
        for block in (168,336,672):
            values,receipt=draw_origin_indices(origins,**kwargs,block_hours=block,replicates=3,seed=7)
            same,_=draw_origin_indices(origins,**kwargs,block_hours=block,replicates=3,seed=7)
            np.testing.assert_array_equal(values,same)
            self.assertFalse(values.flags.writeable);self.assertEqual(values.shape,(3,len(origins)))
            for segment in receipt['segments']:
                lo,hi=segment['start_index'],segment['end_index_exclusive']
                for row in values:
                    self.assertTrue(((row[lo:hi]>=lo)&(row[lo:hi]<hi)).all())
                    for pos in range(lo,hi,block):
                        take=min(block,hi-pos);start=int(row[pos])
                        self.assertLessEqual(start,hi-block)
                        np.testing.assert_array_equal(row[pos:pos+take],np.arange(start,start+take))
                for row in values:
                    weights=origin_multiplicities(row,origin_count=len(origins))
                    self.assertEqual(int(weights[lo:hi].sum()),hi-lo)
            self.assertFalse(receipt['study_results_computed'])
            self.assertTrue(receipt['unequal_edge_inclusion_probabilities'])

    def test_one_draw_preserves_joint_model_gauge_seed_lead_vectors(self):
        origins,kwargs=calendar('2020-01-01T00:00:00Z','2020-01-29T00:00:00Z')
        draws,_=draw_origin_indices(origins,**kwargs,block_hours=168,replicates=1)
        # Axis values expose any independent shuffling of gauge/model/seed/lead.
        vectors=np.arange(len(origins)*3*2*3*18).reshape(len(origins),3,2,3,18)
        paired=vectors[draws[0]]
        for slot,original in enumerate(draws[0]):np.testing.assert_array_equal(paired[slot],vectors[original])
        self.assertEqual(int(origin_multiplicities(draws[0],origin_count=len(origins)).sum()),len(origins))

    def test_calendar_forecast_boundary_and_resource_bounds(self):
        origins,kwargs=calendar('2024-07-01T00:00:00Z','2024-07-31T00:00:00Z')
        with self.assertRaisesRegex(ValueError,'Protected'):draw_origin_indices(origins,**kwargs,replicates=1)
        origins,kwargs=calendar()
        for changes in ({'block_hours':24},{'replicates':0},{'replicates':2001},{'seed':-1}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):draw_origin_indices(origins,**kwargs,**changes)
        for indices in (np.array([0,0,-1]),np.array([0.,1.,2.]),np.array([0,1,3])):
            with self.assertRaises(ValueError):origin_multiplicities(indices,origin_count=3)


class WeightedTests(unittest.TestCase):
    def test_unit_weights_reproduce_ordinary_physical_metrics_and_support(self):
        args=fixture();args['predictions'][0,0]=-2.;args['label_mask'][1,1]=False
        expected=ordinary(args);actual=weighted_physical_h18(**args)
        for a,b in zip(actual['leads'],expected['leads']):
            for name in ('physical_mse','rmse','mae','bias','nse','negative_frequency'):
                self.assertAlmostEqual(a[name],b[name],places=13)
            for name in ('distinct_target_hours','distinct_utc_dates','primary_eligible'):
                self.assertEqual(a[name],b[name])
            self.assertEqual(a['weighted_label_count'],b['n'])
        self.assertAlmostEqual(actual['primary_score'],expected['primary_score'],places=13)
        self.assertFalse(actual['percentile_interval_computed']);self.assertEqual(actual['optimizer_updates'],0)

    def test_integer_weights_match_manual_repeated_errors_without_duplicate_coordinates(self):
        args=fixture(4);args['multiplicities']=np.array([0,1,1,2])
        args['predictions'][:,0]=[-99.,-1.,5.,8.]
        result=weighted_physical_h18(**args);row=result['leads'][0]
        truth=np.array([2.,3.,4.,4.]);prediction=np.array([-1.,5.,8.,8.]);errors=prediction-truth
        self.assertAlmostEqual(row['physical_mse'],float(np.mean(errors**2)))
        self.assertAlmostEqual(row['rmse'],math.sqrt(float(np.mean(errors**2))))
        self.assertAlmostEqual(row['mae'],float(np.mean(abs(errors))));self.assertAlmostEqual(row['bias'],float(errors.mean()))
        self.assertAlmostEqual(row['nse'],1-float((errors**2).sum())/float(((truth-truth.mean())**2).sum()))
        self.assertEqual(row['negative_frequency'],.25);self.assertEqual(row['weighted_label_count'],4)
        self.assertEqual(row['distinct_target_hours'],3);self.assertTrue(result['original_coordinates_preserved'])

    def test_repetition_cannot_inflate_distinct_hours_or_utc_dates(self):
        args=fixture(700,ScoringPolicy());args['multiplicities']=np.zeros(700,dtype=int);args['multiplicities'][0]=700
        result=weighted_physical_h18(**args);row=result['leads'][0]
        self.assertTrue(result['baseline_primary_qualified']);self.assertIsNone(result['primary_score'])
        self.assertEqual(row['weighted_label_count'],700);self.assertEqual(row['distinct_target_hours'],1)
        self.assertEqual(row['distinct_utc_dates'],1);self.assertIsNone(row['nse'])
        self.assertIn('CONSTANT_REFERENCE',row['primary_ineligibility'])
        args['multiplicities'][:168]=4;args['multiplicities'][0]=32 # 32+167*4 =700
        result=weighted_physical_h18(**args);row=result['leads'][0]
        self.assertEqual(row['distinct_target_hours'],168);self.assertEqual(row['distinct_utc_dates'],8)
        self.assertIsNone(result['primary_score']);self.assertEqual(row['primary_ineligibility'],['TOO_FEW_DISTINCT_UTC_DATES'])

    def test_mask_binding_and_failed_baseline_cannot_hide_at_zero_weight_or_missing_label(self):
        for mutation in ('digest','zero_weight_nan','zero_weight_mask','missing_label_prediction'):
            args=fixture(4);args['multiplicities']=np.array([0,1,1,2])
            if mutation=='digest':args['comparison_mask'][0,0]=False
            elif mutation=='zero_weight_nan':args['predictions'][0,0]=np.nan
            elif mutation=='zero_weight_mask':args['prediction_mask'][0,0]=False
            else:args['label_mask'][0,0]=False;args['predictions'][0,0]=np.nan
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):weighted_physical_h18(**args)

    def test_unavailable_calendar_rows_remain_present_but_do_not_score(self):
        args=fixture(4);args['comparison_mask'][1]=False
        args['expected_comparison_mask_sha256']=digest(args['comparison_mask'].tolist())
        args['predictions'][1]=np.nan;args['prediction_mask'][1]=False
        result=weighted_physical_h18(**args)
        self.assertEqual(result['original_calendar_hours'],4);self.assertEqual(result['original_comparison_mask_false_cells'],18)
        self.assertEqual(result['leads'][0]['weighted_label_count'],3)
        self.assertEqual(result['leads'][0]['distinct_target_hours'],3)
        args['multiplicities']=np.array([0,4,0,0]);result=weighted_physical_h18(**args)
        self.assertIsNone(result['primary_score']);self.assertEqual(result['leads'][0]['undefined']['nse'],'ZERO_SUPPORT')

    def test_original_support_failure_stays_unavailable_and_bad_weights_reject(self):
        args=fixture(4,ScoringPolicy());result=weighted_physical_h18(**args)
        self.assertFalse(result['baseline_primary_qualified']);self.assertIsNone(result['primary_score'])
        for weights in (np.ones(4,dtype=float),np.array([-1,1,1,3]),np.array([1,1,1,0]),np.zeros(4,dtype=int)):
            args['multiplicities']=weights
            with self.subTest(weights=weights),self.assertRaises(ValueError):weighted_physical_h18(**args)

    def test_constant_and_numeric_underflow_reference_remain_explicit(self):
        for values,reason in (([1.,1.,1.,1.],'CONSTANT_REFERENCE'),([0.,1e-200,2e-200,3e-200],'REFERENCE_VARIANCE_NUMERIC_UNDERFLOW')):
            args=fixture(4);args['labels'][:]=np.array(values)[:,None];args['predictions']=args['labels'].copy()
            result=weighted_physical_h18(**args)
            self.assertIsNone(result['primary_score']);self.assertEqual(result['leads'][0]['undefined']['nse'],reason)
            self.assertEqual(result['leads'][0]['physical_mse'],0.)


if __name__=='__main__':unittest.main()
