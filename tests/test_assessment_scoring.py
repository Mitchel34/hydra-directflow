"""Synthetic qualification only: no assessment source, model, or network reads."""
import json
import math
import unittest

import numpy as np

import hydra_directflow.assessment_scoring as scoring
from hydra_directflow.contract import ARCHITECTURES, GAUGE_IDS
from hydra_directflow.assessment_contract import SEEDS
from hydra_directflow.reference import ASSESSMENT_START, ASSESSMENT_END, HOUR


def fixture():
    origins = np.arange(ASSESSMENT_START, ASSESSMENT_END, HOUR, dtype=np.int64)
    hours = np.arange(len(origins) + 17)
    hourly_truth = 12 + 4 * np.sin(hours / 27) + hours / 10000
    index = np.arange(len(origins))[:, None] + np.arange(18)[None, :]
    truth = hourly_truth[index].copy()
    mask = np.ones_like(truth, dtype=bool)
    admitted = np.ones(len(origins), dtype=bool)
    predictions = {(a, s): truth + (s - 17) / 12 for a in ARCHITECTURES for s in SEEDS}
    nwm = truth + .5
    nwm[:, 16:] = np.nan
    nwm_mask = mask.copy(); nwm_mask[:, 16:] = False
    return [GAUGE_IDS[0], origins, truth, mask, admitted, predictions, nwm, nwm_mask]


def invalidate_hour(args, index):
    lookup = np.arange(scoring.N_ORIGINS)[:, None] + np.arange(18)[None, :]
    args[3][lookup == index] = False
    args[2][lookup == index] = np.nan


class AssessmentScoreTests(unittest.TestCase):
    def test_perfect_primary_and_physical_error_metrics(self):
        args = fixture(); result = scoring.score_gauge(*args)
        panel = result['models']['LSTM']['seeds']['17']['primary']
        self.assertEqual(panel['equal_lead_nse'], 1.)
        self.assertEqual(panel['support']['forecast_target_pairs'], 8760 * 18)
        self.assertEqual(panel['support']['distinct_target_hours'], 8777)
        self.assertEqual(panel['leads'][0]['distinct_target_hours'], 8760)
        row = result['models']['LSTM']['seeds']['29']['primary']['leads'][17]
        self.assertAlmostEqual(row['bias'], 1.); self.assertAlmostEqual(row['rmse'], 1.)
        self.assertAlmostEqual(row['physical_mse'], 1.); self.assertAlmostEqual(row['mae'], 1.)
        self.assertAlmostEqual(row['pearson_r'], 1.)
        y = args[2][:, 17]
        self.assertAlmostEqual(row['pbias'], 100 * len(y) / sum(y))
        self.assertEqual(result['last_target_epoch'], ASSESSMENT_END + 17 * HOUR)
        json.dumps(result, allow_nan=False)

    def test_common_primary_and_independent_per_lead_masks(self):
        args = fixture(); args[4][:5] = False; invalidate_hour(args, 120)
        for p in args[5].values(): p[:5] = np.nan
        result = scoring.score_gauge(*args); coverage = result['coverage']
        self.assertEqual(coverage['input_admitted_origins'], 8755)
        self.assertEqual(coverage['primary_complete_target_origins'], 8737)
        self.assertEqual(coverage['input_admitted_with_incomplete_targets'], 18)
        self.assertEqual(coverage['per_lead'][0]['evaluable_origins'], 8754)
        primary_hashes = set()
        for a in ARCHITECTURES:
            for s in SEEDS:
                p = result['models'][a]['seeds'][str(s)]
                primary_hashes.add(p['primary']['support_mask_sha256'])
                self.assertEqual(p['primary']['leads'][0]['n'], 8737)
                self.assertEqual(p['per_lead_valid_target']['leads'][0]['n'], 8754)
        self.assertEqual(len(primary_hashes), 1)

    def test_nwm_pairing_changes_only_matched_support(self):
        args = fixture(); args[7][:200, 0] = False; args[6][:200, 0] = np.nan
        args[6][200:, 0] = args[2][200:, 0]
        result = scoring.score_gauge(*args)
        self.assertEqual(result['coverage']['primary_complete_target_origins'], 8760)
        nwm = result['nwm']['matched']
        self.assertEqual(nwm['leads'][0]['n'], 8560)
        self.assertEqual(nwm['leads'][0]['nse'], 1.)
        for a in ARCHITECTURES:
            for seed in SEEDS:
                paired = result['models'][a]['seeds'][str(seed)]['nwm_matched']
                self.assertEqual(paired['support_mask_sha256'], nwm['support_mask_sha256'])
                self.assertEqual(paired['leads'][0]['scored_target_times_sha256'], nwm['leads'][0]['scored_target_times_sha256'])
                self.assertEqual(paired['equal_lead_nse_leads'], list(range(1, 17)))
        self.assertEqual(nwm['leads'][16]['n'], 0)
        self.assertIn('STRUCTURALLY_UNAVAILABLE', nwm['leads'][16]['undefined']['nse'])
        self.assertEqual(result['nwm']['structurally_unavailable_lead_hours'], [17, 18])

    def test_nwm_absence_never_changes_hydra_eligibility(self):
        args = fixture(); args[6][:] = np.nan; args[7][:] = False
        result = scoring.score_gauge(*args)
        self.assertEqual(result['coverage']['primary_complete_target_origins'], 8760)
        self.assertEqual(result['models']['LSTM']['seeds']['17']['primary']['equal_lead_nse'], 1.)
        self.assertIsNone(result['nwm']['matched']['equal_lead_nse'])

    def test_nwm_unsupported_or_masked_nonfinite_forecasts_fail(self):
        for change in ('unsupported_mask', 'unsupported_value', 'nan_available'):
            with self.subTest(change=change):
                args = fixture()
                if change == 'unsupported_mask': args[7][0, 17] = True
                elif change == 'unsupported_value': args[6][0, 16] = 2.
                else: args[6][0, 0] = np.nan
                with self.assertRaises(ValueError): scoring.score_gauge(*args)

    def test_missing_prediction_on_admitted_origin_fails_even_invalid_target(self):
        args = fixture(); invalidate_hour(args, 0)
        args[5][('Transformer', 43)][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Missing/nonfinite predictions'): scoring.score_gauge(*args)

    def test_all_four_families_and_three_seeds_required(self):
        args = fixture(); args[5].pop(('Mamba2_Transformer', 43))
        with self.assertRaisesRegex(ValueError, 'All four'): scoring.score_gauge(*args)

    def test_all_seed_metrics_aggregated_without_ensemble_or_best_seed(self):
        args = fixture(); result = scoring.score_gauge(*args)
        model = result['models']['LSTM']; aggregate = model['seed_aggregation']['primary']
        expected = [model['seeds'][str(seed)]['primary']['equal_lead_nse'] for seed in SEEDS]
        self.assertAlmostEqual(aggregate['equal_lead_nse']['mean'], sum(expected) / 3)
        self.assertEqual(aggregate['equal_lead_nse']['min'], min(expected))
        self.assertEqual(aggregate['equal_lead_nse']['max'], max(expected))
        mse = aggregate['leads'][0]['physical_mse']['mean']
        errors = [(s - 17) / 12 for s in SEEDS]
        self.assertAlmostEqual(mse, sum(e * e for e in errors) / 3)
        self.assertNotAlmostEqual(mse, (sum(errors) / 3) ** 2)
        self.assertFalse(aggregate['ensemble_predictions_scored'])
        self.assertFalse(aggregate['formal_confidence_intervals'])
        self.assertEqual(set(model['seeds']), {'17', '29', '43'})

    def test_constant_truth_undefined_nse_keeps_physical_metrics_and_all_seeds(self):
        args = fixture(); args[2][:] = 0.
        for p in args[5].values(): p[:] = -1.
        result = scoring.score_gauge(*args)
        row = result['models']['LSTM']['seeds']['17']['primary']['leads'][0]
        self.assertIsNone(row['nse']); self.assertIsNone(row['pbias']); self.assertIsNone(row['pearson_r'])
        self.assertEqual(row['undefined']['nse'], 'CONSTANT_OBSERVED_TARGETS')
        self.assertEqual(row['physical_mse'], 1.); self.assertEqual(row['negative_frequency'], 1.)
        self.assertEqual(row['bias'], -1.)
        aggregate = result['models']['LSTM']['seed_aggregation']['primary']
        self.assertIsNone(aggregate['equal_lead_nse']['mean'])
        self.assertEqual(aggregate['equal_lead_nse']['total_seeds'], 3)
        self.assertFalse(aggregate['equal_lead_nse']['undefined_seeds_omitted'])
        json.dumps(result, allow_nan=False)

    def test_support_floors_are_reporting_gates_not_cohort_filters(self):
        args = fixture(); args[4][:] = False; args[4][:168] = True
        result = scoring.score_gauge(*args)
        p = result['models']['LSTM']['seeds']['17']['primary']; row = p['leads'][0]
        self.assertEqual(row['n'], 168); self.assertEqual(row['nse'], 1.)
        self.assertIn('TOO_FEW_DISTINCT_UTC_DATES', row['primary_ineligibility'])
        self.assertIsNone(p['equal_lead_nse'])
        args[4][:] = False
        # Six hours per day over28UTC dates: exactly168hours and28dates.
        for day in range(28): args[4][day * 24:day * 24 + 6] = True
        result = scoring.score_gauge(*args)
        self.assertEqual(result['models']['LSTM']['seeds']['17']['primary']['equal_lead_nse'], 1.)

    def test_empty_cohort_is_explicit_not_nan_and_no_target_fill(self):
        args = fixture(); args[4][:] = False; args[3][:] = False; args[2][:] = np.nan
        before = args[2].copy(); result = scoring.score_gauge(*args)
        row = result['models']['LSTM']['seeds']['17']['primary']['leads'][0]
        self.assertEqual(row['n'], 0); self.assertIsNone(row['physical_mse'])
        self.assertEqual(row['undefined']['nse'], 'NO_SCORED_PAIRS')
        np.testing.assert_array_equal(args[2], before)
        json.dumps(result, allow_nan=False)

    def test_clocks_fail_before_outcome_array_conversion(self):
        class Poison:
            def __array__(self, *args, **kwargs): raise AssertionError('Read labels before calendar validation')
        args = fixture(); args[1] = args[1][::-1]; args[2] = Poison()
        with self.assertRaisesRegex(ValueError, 'calendar'): scoring.score_gauge(*args)

    def test_duplicate_target_hour_quality_and_value_alignment(self):
        for field in ('value', 'mask'):
            args = fixture()
            if field == 'value': args[2][100, 1] += 2
            else: args[3][100, 1] = False
            with self.assertRaisesRegex(ValueError, 'Repeated'): scoring.score_gauge(*args)

    def test_nonfinite_negative_targets_and_bad_masks_reject(self):
        for change in ('nonfinite', 'negative', 'integer_mask'):
            args = fixture()
            if change == 'nonfinite': args[2][0, 0] = np.nan
            elif change == 'negative': args[2][0, 0] = -1.
            else: args[3] = args[3].astype(int)
            with self.assertRaises(ValueError): scoring.score_gauge(*args)

    def test_unequal_lead_error_uses_arithmetic_mean_nse(self):
        args = fixture()
        for key in args[5]: args[5][key] = args[2] + np.arange(18)[None, :] / 10
        result = scoring.score_gauge(*args)
        p = result['models']['LSTM']['seeds']['17']['primary']
        self.assertAlmostEqual(p['equal_lead_nse'], sum(r['nse'] for r in p['leads']) / 18)
        paired = result['models']['LSTM']['seeds']['17']['nwm_matched']
        self.assertAlmostEqual(paired['equal_lead_nse'], sum(r['nse'] for r in paired['leads'][:16]) / 16)

    def test_overflow_is_failure_not_a_silently_defined_metric(self):
        args = fixture(); args[2][:] = 1e308
        for p in args[5].values(): p[:] = 1e308
        with self.assertRaisesRegex(ValueError, 'overflow'): scoring.score_gauge(*args)


if __name__ == '__main__': unittest.main()
