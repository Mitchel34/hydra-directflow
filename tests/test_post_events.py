"""Frozen event selection and conditional coverage; synthetic values only."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from hydra_directflow.coverage_analysis import (
    select_observed_events, event_coverage, coverage_by_regime, iso,
)
from hydra_directflow.post_events import (
    CATALOG_RULE, HOUR, POPULATION, known_target_calendar, summarize_events,
    summarize_regimes, validate_catalog_row,
)
from hydra_directflow.reference import ASSESSMENT_START


def fixture(*, omit_all=False, hours=700, peak_values=None):
    start = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())
    end = start+hours*HOUR
    times = np.arange(start, end, HOUR, dtype=np.int64)
    values = np.full(len(times), 3., dtype=np.float64)
    for index, value in (peak_values or {100: 15., 300: 20., 500: 12.}).items():
        values[index] = value
    admitted = np.ones(len(times), dtype=bool)
    admitted[250:256] = False
    values[~admitted] = np.nan
    reference = dict(times=times, values=values, admitted=admitted)
    thresholds = dict(q10_cms=1., q90_cms=6., q95_cms=10.,
        fit_start_utc='2018-07-12T00:00:00Z', fit_end_exclusive_utc='2022-05-01T00:00:00Z',
        refit_at_utc=iso(start), fit_population=POPULATION, stopping_or_outer_values_used=False,
        model_scaler=False, simulated_label_delay_hours=2, quantile_method='numpy.linear')
    origins = np.arange(start, end-18*HOUR, HOUR, dtype=np.int64)
    future = origins[:, None]+np.arange(1, 19)[None, :]*HOUR
    positions = (future-start)//HOUR
    target_mask = admitted[positions]
    inputs = np.ones(len(origins), dtype=bool)
    # The largest observed peak is unavailable to every model and is retained.
    inputs[250:341] = False
    if omit_all:
        inputs[:] = False
    common = inputs & target_mask.all(axis=1)
    events = select_observed_events(reference, thresholds, start=start, end=end)
    for event in events:
        event['forecast_case_coverage_by_lead'] = event_coverage(event, origins, inputs, common, target_mask,
                                                               outer_start=start, outer_end=end)
        coverage = event['forecast_case_coverage_by_lead']
        event['all_lead_requested_cells'] = sum(x['requested_cells'] for x in coverage)
        event['all_lead_common_comparison_cells'] = sum(x['common_comparison_cells'] for x in coverage)
        event['observed_peak_supported_leads'] = [x['lead_hours'] for x in coverage if x['observed_peak_has_common_prediction_case']]
        if not event['representative_top3']:
            del event['forecast_case_coverage_by_lead']
    row = dict(gauge_id='03161000', fold='final_development', thresholds=thresholds,
        outer_window=dict(start_utc=iso(start), end_exclusive_utc=iso(end)),
        representative_event_rule=CATALOG_RULE, event_selection_uses_model_errors=False,
        event_windows_clipped_to_outer=True, event_windows_hours=[-48, 72],
        observed_high_flow_event_count=len(events), observed_high_flow_events=events,
        target_cell_regime_coverage=coverage_by_regime(origins, inputs, common, target_mask, reference,
                                                      thresholds, start=start, end=end))
    truth = values[positions][common]
    args = dict(origins=origins[common], forecast_time=future[common], labels=truth,
                label_mask=target_mask[common], predictions={'model-seed17': truth+2.})
    return row, args, reference


class FrozenEventTests(unittest.TestCase):
    def test_all_pinned_metadata_catalogs_validate_without_source_values(self):
        path = Path(__file__).resolve().parents[1]/'reports/development-coverage-regimes-v2.json'
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '8e82f613165a5d452c74a30783b02e1c0ae5fb5d93dbe2e338d0010c215e8c0a')
        rows = json.loads(raw)['fold_regime_and_event_diagnostics']
        self.assertEqual(len(rows), 12)
        for row in rows:
            validate_catalog_row(row)

    def test_unavailable_largest_peak_remains_representative_and_errors_descriptive(self):
        row, args, _ = fixture()
        original = deepcopy(row)
        result = summarize_events(row, **args)
        self.assertEqual(row, original)
        self.assertEqual(result['catalog_events'], 3)
        first = result['events'][0]
        self.assertTrue(first['representative_top3'])
        self.assertFalse(first['observed_peak_supported_any_lead'])
        self.assertIn(first['event_id'], result['representative_event_ids'])
        self.assertFalse(result['unsupported_events_replaced'])
        self.assertFalse(result['event_episodes_are_proven_independent'])
        for event in result['events']:
            for lead in event['per_lead']:
                metrics = lead['models']['model-seed17']
                if metrics['n']:
                    self.assertAlmostEqual(metrics['rmse'], 2.)
                    self.assertAlmostEqual(metrics['mae'], 2.)
                    self.assertAlmostEqual(metrics['bias'], 2.)
                    self.assertEqual(metrics['negative_prediction_frequency'], 0.)
        # The second selected event has only120unique valid hours, not2160floods.
        second = result['events'][1]
        self.assertEqual(second['unique_matched_target_hours'], 120)
        self.assertEqual(second['matched_origin_lead_cells'], 120*18)
        self.assertTrue(second['observed_peak_supported_all18_leads'])

    def test_all_unsupported_events_and_empty_predictions_keep_full_catalog(self):
        row, args, _ = fixture(omit_all=True)
        result = summarize_events(row, **args)
        self.assertEqual(result['catalog_events'], 3)
        self.assertEqual(result['events_with_any_matched_prediction'], 0)
        self.assertEqual(len(result['representative_event_ids']), 3)
        for event in result['events']:
            self.assertEqual(event['unique_matched_target_hours'], 0)
            for lead in event['per_lead']:
                self.assertIsNone(lead['models']['model-seed17']['rmse'])
                self.assertEqual(lead['models']['model-seed17']['status'], 'UNAVAILABLE_NO_MATCHED_TARGETS')

    def test_short_event_support_does_not_receive_annual_nse_gate(self):
        row, args, _ = fixture()
        result = summarize_events(row, **args)
        self.assertFalse(result['annual_NSE_support_rule_applied'])
        metrics = result['events'][1]['per_lead'][0]['models']['model-seed17']
        self.assertEqual(metrics['n'], 120)
        self.assertEqual(metrics['status'], 'AVAILABLE_DESCRIPTIVE')
        self.assertNotIn('nse', metrics)

    def test_no_seed_pooling_or_clipping(self):
        row, args, _ = fixture()
        args['predictions']['model-seed29'] = args['labels']-1.
        args['predictions']['model-negative'] = np.full_like(args['labels'], -1.)
        result = summarize_events(row, **args)
        values = result['events'][1]['per_lead'][0]['models']
        self.assertEqual(values['model-seed17']['bias'], 2.)
        self.assertEqual(values['model-seed29']['bias'], -1.)
        self.assertEqual(values['model-negative']['negative_prediction_frequency'], 1.)
        self.assertEqual(set(values), set(args['predictions']))

    def test_missing_prediction_is_failure_not_a_smaller_cohort(self):
        row, args, _ = fixture()
        args['predictions']['model-seed17'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Missing/nonfinite'):
            summarize_events(row, **args)

    def test_missing_target_or_changed_common_support_rejected(self):
        row, args, _ = fixture()
        args['label_mask'][0, 0] = False
        with self.assertRaisesRegex(ValueError, 'complete common H18'):
            summarize_events(row, **args)
        row, args, _ = fixture()
        for name in ('origins', 'forecast_time', 'labels', 'label_mask'):
            args[name] = args[name][1:]
        args['predictions'] = {k: v[1:] for k, v in args['predictions'].items()}
        # Remove a row inside a catalog window to force frozen-coverage mismatch.
        window_start = row['observed_high_flow_events'][1]['start_epoch']
        position = int(np.searchsorted(args['origins'], window_start))
        for name in ('origins', 'forecast_time', 'labels', 'label_mask'):
            args[name] = np.delete(args[name], position, axis=0)
        args['predictions'] = {k: np.delete(v, position, axis=0) for k, v in args['predictions'].items()}
        with self.assertRaisesRegex(ValueError, 'membership differs'):
            summarize_events(row, **args)

    def test_protected_or_misaligned_times_fail_before_metrics(self):
        row, args, _ = fixture()
        args['forecast_time'][0, 0] = ASSESSMENT_START
        with self.assertRaisesRegex(ValueError, 'development origin/target'):
            summarize_events(row, **args)

    def test_same_valid_hour_must_keep_same_reference_across_forecasts(self):
        row, args, _ = fixture()
        args['labels'][0, 1] += 1.
        with self.assertRaisesRegex(ValueError, 'inconsistent reference'):
            summarize_events(row, **args)

    def test_prediction_errors_cannot_change_catalog_ids_or_ranking(self):
        row, args, _ = fixture()
        first = summarize_events(row, **args)
        args['predictions']['model-seed17'] = args['labels']*1000.
        second = summarize_events(row, **args)
        self.assertEqual(first['representative_event_ids'], second['representative_event_ids'])
        self.assertEqual([e['event_id'] for e in first['events']], [e['event_id'] for e in second['events']])

    def test_catalog_changes_to_rank_window_or_threshold_source_fail(self):
        for what in ('rank', 'window', 'selection', 'threshold'):
            row, _, _ = fixture()
            if what == 'rank': row['observed_high_flow_events'].reverse()
            elif what == 'window': row['observed_high_flow_events'][0]['start_epoch'] -= HOUR
            elif what == 'selection': row['event_selection_uses_model_errors'] = True
            else: row['thresholds']['fit_population'] = 'ONLY_INPUT_ADMITTED_CASES'
            with self.subTest(change=what), self.assertRaises(ValueError):
                validate_catalog_row(row)

    def test_regimes_reconcile_known_full_calendar_denominators_and_not_refit_thresholds(self):
        row, args, _ = fixture()
        result = summarize_regimes(row, **args)
        self.assertFalse(result['thresholds_recomputed'])
        self.assertEqual(result['thresholds'], row['thresholds'])
        self.assertEqual(sum(r['matched_origin_lead_cells'] for r in result['regimes']), args['labels'].size)
        unknown = result['regimes'][-1]
        self.assertEqual(unknown['matched_origin_lead_cells'], 0)
        self.assertGreater(unknown['census_coverage']['requested_target_cells'], 0)
        high = result['regimes'][3]
        self.assertLess(high['matched_origin_lead_cells'], high['census_coverage']['reference_supported_cells'])

    def test_regime_census_mismatch_is_not_silently_reconciled(self):
        row, args, _ = fixture()
        row['target_cell_regime_coverage'][1]['common_comparison_cells'] -= 1
        with self.assertRaisesRegex(ValueError, 'regime membership differs'):
            summarize_regimes(row, **args)

    def test_known_reference_plot_calendar_keeps_gaps_and_separate_catalog_peak(self):
        row, args, reference = fixture()
        args.pop('predictions')
        series = known_target_calendar(row, **args)
        self.assertEqual(len(series['times']), len(reference['times']))
        self.assertFalse(series['gaps_filled'])
        first_peak = row['observed_high_flow_events'][0]['peak_epoch']
        index = int((first_peak-series['times'][0])//HOUR)
        self.assertFalse(series['known_from_admitted_labels'][index])
        self.assertTrue(np.isnan(series['values'][index]))
        self.assertEqual(row['observed_high_flow_events'][0]['observed_peak_q_cms'], 20.)

    def test_float32_target_rounding_cannot_silently_change_regime_ties(self):
        row, args, _ = fixture()
        args['labels'] = args['labels'].astype(np.float32)
        with self.assertRaisesRegex(ValueError, 'Float64 original labels'):
            summarize_regimes(row, **args)

    def test_nonrepresentative_events_keep_counts_without_inventing_reference_denominators(self):
        row, args, _ = fixture(hours=900, peak_values={100: 15., 300: 20., 500: 12., 700: 11.})
        result = summarize_events(row, **args)
        self.assertEqual(result['catalog_events'], 4)
        fourth = result['events'][3]
        self.assertFalse(fourth['representative_top3'])
        self.assertEqual(len(result['representative_event_ids']), 3)
        self.assertEqual(fourth['matched_origin_lead_cells'], 2160)
        self.assertTrue(fourth['observed_peak_supported_all18_leads'])
        for lead in fourth['per_lead']:
            self.assertEqual(lead['calendar_requested_cells'], 120)
            self.assertEqual(lead['matched_evaluable_cells'], 120)
            self.assertIsNone(lead['approved_reference_cells'])
            self.assertIsNone(lead['input_admissible_opportunities'])
            self.assertIsNone(lead['matched_coverage_of_known_reference'])
            self.assertEqual(lead['matched_coverage_of_calendar'], 1.)

    def test_clipped_windows_preserve_lead_specific_calendar_and_peak_support(self):
        row, args, _ = fixture(peak_values={6: 20., 693: 15.})
        result = summarize_events(row, **args)
        left, right = result['events']
        self.assertTrue(left['left_truncated'])
        self.assertTrue(right['right_truncated'])
        self.assertEqual([x['lead_hours'] for x in left['per_lead'] if x['observed_peak_supported']], list(range(1, 7)))
        self.assertEqual([x['lead_hours'] for x in right['per_lead'] if x['observed_peak_supported']], list(range(12, 19)))
        for source, event in zip(row['observed_high_flow_events'], result['events']):
            self.assertEqual(sum(x['calendar_requested_cells'] for x in event['per_lead']), source['all_lead_requested_cells'])

    def test_equal_peak_magnitude_requires_earliest_utc_rank_tie_break(self):
        row, _, _ = fixture()
        # The oldfirstpeak is later inUTC; equal magnitudes must invert that order.
        row['observed_high_flow_events'][0]['observed_peak_q_cms'] = 15.
        with self.assertRaisesRegex(ValueError, 'ranking/tie'):
            validate_catalog_row(row)


if __name__ == '__main__':
    unittest.main()
