"""Synthetic equivalence and family-policy checks; no study data/model calls."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import unittest

import numpy as np

from hydra_directflow.post_uncertainty import (make_draws, split_seed, evaluate_matched,
    evaluate_weights, primary_family, FOLDS, SEEDS, ARCHITECTURES, GAUGES)
from hydra_directflow.reference import digest
from hydra_directflow.scoring import ScoringPolicy, ScoreIdentity
from hydra_directflow.uncertainty import draw_origin_indices, weighted_physical_h18

SMALL = ScoringPolicy(minimum_distinct_target_hours=2, minimum_distinct_utc_dates=1)


def stamp(text):
    return int(datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp())


def fixture(n=1008, panels=2):
    start = '2020-01-01T00:00:00Z'
    origins = stamp(start) + np.arange(n, dtype=np.int64) * 3600
    end = datetime.fromtimestamp(int(origins[-1]) + 3600, timezone.utc).isoformat()
    t = np.arange(n, dtype=float)
    y = 10 + np.sin(t[:, None]/29) + (t[:, None] % 100)/7 + np.arange(18)[None, :]/3
    prediction = np.stack([y + np.cos(t[:, None]/17) + (i+1)*np.arange(18)[None, :]/30 for i in range(panels)])
    prediction[:, 0, 0] = -2.
    label_mask = np.ones((n,18),bool); label_mask[15:20,3] = False; y[~label_mask] = np.nan
    comparison = np.ones(n,bool); comparison[83:100] = False
    present = np.broadcast_to(comparison[None,:,None], prediction.shape).copy()
    prediction[~present] = np.nan
    return origins, dict(start_utc=start,end_exclusive_utc=end), dict(labels=y,label_mask=label_mask,
        predictions=prediction,prediction_mask=present,comparison_mask=comparison,
        expected_comparison_mask_sha256=digest(comparison.tolist()),policy=SMALL)


def reference(origins, bounds, args, weights, panel):
    ids = np.array(['03161000:'+datetime.fromtimestamp(int(t),timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') for t in origins])
    comparison = np.broadcast_to(args['comparison_mask'][:,None], (len(origins),18)).copy()
    return weighted_physical_h18(case_ids=ids, origins=origins,
        forecast_time=origins[:,None]+np.arange(1,19)[None,:]*3600, **bounds,
        labels=args['labels'],label_mask=args['label_mask'],predictions=args['predictions'][panel],
        prediction_mask=args['prediction_mask'][panel],comparison_mask=comparison,
        expected_comparison_mask_sha256=digest(comparison.tolist()),multiplicities=weights,
        identity=ScoreIdentity('LSTM','synthetic','03161000','FULL_HRRR_H2_W4_S2_V1',168,'cv_1',17),
        input_cohort_digest='a'*64,label_policy_digest='b'*64,label_source_digest='c'*64,policy=args['policy'])


class DrawAndScoreTests(unittest.TestCase):
    def test_seed_split_is_fixed_unique_and_existing_sampler_exact(self):
        seeds=[]
        for block in (168,336,672):
            for fold in FOLDS:
                seed,meta=split_seed(block,fold);seeds.append(seed)
                self.assertEqual(meta['child_index'],len(seeds)-1)
                self.assertEqual(meta['spawn_key'],[len(seeds)-1])
        self.assertEqual(len(set(seeds)),12)
        origins,bounds,args=fixture()
        for block in (168,336,672):
            draws=make_draws(origins,**bounds,fold='cv_1',block_hours=block,replicates=3)
            expected,meta=draw_origin_indices(origins,**bounds,block_hours=block,replicates=3,
                seed=split_seed(block,'cv_1')[0])
            np.testing.assert_array_equal(draws.indices,expected)
            self.assertFalse(draws.indices.flags.writeable)
            self.assertFalse(draws.multiplicities.flags.writeable)
            for row,weights in zip(expected,draws.multiplicities):
                np.testing.assert_array_equal(weights,np.bincount(row,minlength=len(origins)))

    def test_full_calendar_guard_before_score_and_short_segments_not_shrunk(self):
        origins,bounds,args=fixture()
        for bad in (origins[::2],origins[::-1],np.delete(origins,8)):
            with self.assertRaisesRegex(ValueError,'Complete hourly'):
                make_draws(bad,**bounds,fold='cv_1',replicates=1)
        protected=np.arange(stamp('2024-07-01T00:00:00Z'),stamp('2024-07-31T00:00:00Z'),3600)
        with self.assertRaisesRegex(ValueError,'Protected'):
            make_draws(protected,start_utc='2024-07-01T00:00:00Z',end_exclusive_utc='2024-07-31T00:00:00Z',fold='final_development',replicates=1)
        short,bounds,_=fixture(100)
        with self.assertRaisesRegex(ValueError,'shorter than block'):
            make_draws(short,**bounds,fold='cv_1',replicates=1)

    def test_fast_block_sums_match_dense_weights_and_existing_scalar_reference(self):
        origins,bounds,args=fixture()
        draws=make_draws(origins,**bounds,fold='cv_1',block_hours=168,replicates=3)
        fast=evaluate_matched(draws,**args)
        dense=evaluate_weights(origins,**bounds,multiplicities=draws.multiplicities,**args)
        for key in ('primary','per_lead_nse','baseline_primary','baseline_per_lead_nse'):
            np.testing.assert_allclose(fast[key],dense[key],rtol=1e-11,atol=1e-11,equal_nan=True)
        for key in ('valid','distinct_target_hours','distinct_utc_dates','weighted_label_count','lead_valid'):
            np.testing.assert_array_equal(fast[key],dense[key])
        for draw in range(3):
            for panel in range(2):
                expected=reference(origins,bounds,args,draws.multiplicities[draw],panel)
                self.assertAlmostEqual(fast['primary'][draw,panel],expected['primary_score'],places=11)
                for lead,row in enumerate(expected['leads']):
                    self.assertAlmostEqual(fast['per_lead_nse'][draw,panel,lead],row['nse'],places=11)
                    self.assertEqual(fast['distinct_target_hours'][draw,lead],row['distinct_target_hours'])
                    self.assertEqual(fast['distinct_utc_dates'][draw,lead],row['distinct_utc_dates'])
                    self.assertEqual(fast['weighted_label_count'][draw,lead],row['weighted_label_count'])

    def test_missing_baseline_prediction_cannot_hide_under_zero_weight_or_missing_label(self):
        origins,bounds,args=fixture(700)
        weights=np.ones((1,700),int);weights[0,0]=0;weights[0,1]=2
        for mutation in ('nan','missing','masked_label'):
            changed=deepcopy(args)
            if mutation=='nan':changed['predictions'][0,0,0]=np.nan
            elif mutation=='missing':changed['prediction_mask'][0,0,0]=False
            else:
                changed['label_mask'][0,0]=False;changed['predictions'][0,0,0]=np.nan
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'Requested missing'):
                evaluate_weights(origins,**bounds,multiplicities=weights,**changed)

    def test_repetition_does_not_create_distinct_hours_or_dates(self):
        origins,bounds,args=fixture(700);args['policy']=ScoringPolicy()
        weights=np.zeros((1,700),int);weights[0,0]=700
        out=evaluate_weights(origins,**bounds,multiplicities=weights,**args)
        self.assertTrue(np.isnan(out['primary']).all())
        self.assertEqual(out['distinct_target_hours'][0,0],1)
        self.assertEqual(out['distinct_utc_dates'][0,0],1)
        self.assertEqual(out['weighted_label_count'][0,0],700)
        self.assertTrue(out['undefined_or_zero_variance'][0,0])
        weights[0]=0;weights[0,:168]=4;weights[0,0]=32
        out=evaluate_weights(origins,**bounds,multiplicities=weights,**args)
        self.assertLess(out['distinct_utc_dates'][0,0],28)
        self.assertTrue(np.isnan(out['primary']).all())

    def test_baseline_ineligibility_is_retained_in_every_draw(self):
        origins,bounds,args=fixture(100);args['policy']=ScoringPolicy()
        out=evaluate_weights(origins,**bounds,multiplicities=np.ones((2,100),int),**args)
        self.assertTrue(out['baseline_unavailable']);self.assertTrue(np.isnan(out['primary']).all())

    def test_unavailable_calendar_rows_stay_unscored_and_inputs_unchanged(self):
        origins,bounds,args=fixture(700)
        original=[x.copy() for x in (args['labels'],args['predictions'],args['comparison_mask'])]
        weights=np.zeros((1,700),int);weights[0,90]=700
        out=evaluate_weights(origins,**bounds,multiplicities=weights,**args)
        self.assertTrue(np.isnan(out['primary']).all());self.assertTrue((out['weighted_label_count']==0).all())
        for actual,expected in zip((args['labels'],args['predictions'],args['comparison_mask']),original):
            np.testing.assert_array_equal(actual,expected)

    def test_constant_and_underflow_reference_do_not_become_defined(self):
        origins,bounds,args=fixture(700)
        for values in (np.ones(700),np.arange(700)*1e-200):
            changed=deepcopy(args);changed['labels'][:]=values[:,None]
            changed['predictions'][:]=values[None,:,None]
            out=evaluate_weights(origins,**bounds,multiplicities=np.ones((1,700),int),**changed)
            self.assertTrue(np.isnan(out['primary']).all())
            self.assertTrue(out['baseline_unavailable'])

    def test_nearly_constant_draw_at_large_reference_offset_matches_two_pass_reference(self):
        origins,bounds,args=fixture(700)
        args['labels'][:]=1e12+np.arange(700)[:,None]/1000
        args['predictions'][:]=args['labels'][None,:,:]+.0005
        weights=np.zeros((1,700),int);weights[0,:2]=350
        out=evaluate_weights(origins,**bounds,multiplicities=weights,**args)
        expected=reference(origins,bounds,args,weights[0],0)
        self.assertAlmostEqual(out['primary'][0,0],expected['primary_score'],places=12)
        np.testing.assert_allclose(out['per_lead_nse'][0,0],[x['nse'] for x in expected['leads']],rtol=1e-12,atol=1e-12)

    def test_mask_digest_and_invalid_weights_are_rejected(self):
        origins,bounds,args=fixture(700)
        with self.assertRaisesRegex(ValueError,'mask identity'):
            evaluate_weights(origins,**bounds,multiplicities=np.ones((1,700),int),
                **{**args,'expected_comparison_mask_sha256':'0'*64})
        for weights in (np.ones((1,700),float),np.zeros((1,700),int),np.r_[[-1],np.ones(698,int),[3]][None,:]):
            with self.subTest(weights=weights.dtype),self.assertRaises(ValueError):
                evaluate_weights(origins,**bounds,multiplicities=weights,**args)

    def test_changed_draw_blocks_origins_or_counts_are_refused(self):
        origins,bounds,args=fixture();draws=make_draws(origins,**bounds,fold='cv_1',replicates=2)
        for field in ('indices','multiplicities','block_starts','block_lengths','origins'):
            value=getattr(draws,field).copy();value.flat[0]+=1
            changed=replace(draws,**{field:value})
            with self.subTest(field=field),self.assertRaises(ValueError):evaluate_matched(changed,**args)


def family_fixture(mode='final'):
    folds=FOLDS[:3] if mode=='cv' else FOLDS[3:]
    seeds=(17,) if mode=='cv' else SEEDS
    arms=(('full',168),('q_only',168)) if mode=='cv' else (('full',168),('q_only',168),('full',24))
    draw,point={},{}
    x=np.arange(2000,dtype=float)/2000
    for fi,fold in enumerate(folds):
        for gi,gauge in enumerate(GAUGES):
            for ai,architecture in enumerate(ARCHITECTURES):
                for arm,hours in arms:
                    for si,seed in enumerate(seeds):
                        key=(fold,gauge,architecture,arm,hours,seed)
                        value=.1*ai+.01*gi+.03*fi+.04*si+(.25 if arm=='full' else 0)+(.02 if hours==168 else 0)
                        point[key]=value;draw[key]=value+x*(1+ai/10)
    return draw,point


class FamilyTests(unittest.TestCase):
    def test_exact_family_sizes_and_all_seed_or_fold_metric_means(self):
        for mode,size,panels in [('final',42,108),('cv',30,72)]:
            draw,point=family_fixture(mode);out=primary_family(draw,point,mode=mode)
            self.assertEqual(out['contrast_count'],size);self.assertEqual(out['required_panel_count'],panels)
            self.assertEqual(out['jointly_valid_draws'],2000)
            self.assertEqual(out['status'],'AVAILABLE_CONDITIONAL_POINTWISE')
            self.assertTrue(out['mean_metrics_not_ensemble_predictions']);self.assertFalse(out['simultaneous_familywise_claim'])
            for row in out['contrasts']:
                if row['kind']=='full_minus_q':self.assertAlmostEqual(row['point_difference'],.25)
                if row['kind']=='L168_minus_L24':self.assertAlmostEqual(row['point_difference'],.02)
            first=out['contrasts'][0];self.assertAlmostEqual(first['lower'],.25);self.assertAlmostEqual(first['upper'],.25)

    def test_joint_invalid_pool_is_shared_and_threshold100_is_inclusive(self):
        draw,point=family_fixture();keys=list(draw)
        draw[keys[0]][:50]=np.nan;draw[keys[-1]][50:100]=np.nan
        out=primary_family(draw,point,mode='final')
        self.assertEqual(out['jointly_invalid_draws'],100);self.assertEqual(out['jointly_valid_draws'],1900)
        self.assertEqual(out['status'],'AVAILABLE_CONDITIONAL_POINTWISE')
        manual=np.percentile(out['contrast_draws'][out['joint_valid_mask']],(2.5,97.5),axis=0,method='linear')
        for index,row in enumerate(out['contrasts']):
            self.assertEqual(row['lower'],manual[0,index]);self.assertEqual(row['upper'],manual[1,index])
        draw[keys[0]][100]=np.nan;out=primary_family(draw,point,mode='final')
        self.assertEqual(out['jointly_invalid_draws'],101)
        self.assertEqual(out['status'],'FAMILY_INTERVALS_UNAVAILABLE')
        self.assertTrue(all(x['lower'] is None and x['upper'] is None for x in out['contrasts']))
        self.assertFalse(out['invalid_draws_redrawn'])

    def test_missing_extra_seed_wrong_fold_or_draw_length_cannot_be_omitted(self):
        draw,point=family_fixture()
        for mode in ('missing','extra','length'):
            dd,pp=dict(draw),dict(point);key=next(iter(dd))
            if mode=='missing':del dd[key];del pp[key]
            elif mode=='extra':dd[('cv_1',)+key[1:]]=dd[key];pp[('cv_1',)+key[1:]]=pp[key]
            else:dd[key]=dd[key][:-1]
            with self.subTest(mode=mode),self.assertRaises(ValueError):primary_family(dd,pp,mode='final')

    def test_any_missing_baseline_disables_family_intervals(self):
        draw,point=family_fixture();point[next(iter(point))]=None
        out=primary_family(draw,point,mode='final')
        self.assertFalse(out['baseline_primary_available']);self.assertEqual(out['status'],'FAMILY_INTERVALS_UNAVAILABLE')
        self.assertTrue(all(x['lower'] is None for x in out['contrasts']))

    def test_family_cannot_relax_draw_count_invalid_limit_or_percentiles(self):
        draw,point=family_fixture()
        short={key:value[:100] for key,value in draw.items()}
        with self.assertRaisesRegex(ValueError,'Exactly2000'):primary_family(short,point,mode='final')
        for extra in ({'max_joint_invalid':101},{'max_joint_invalid':0},{'percentiles':(5,95)}):
            with self.subTest(extra=extra),self.assertRaisesRegex(ValueError,'Declared family policy'):
                primary_family(draw,point,mode='final',**extra)


if __name__=='__main__':unittest.main()
