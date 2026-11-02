"""Typed bundle tests use synthetic committed shards; zero fits/updates/network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow import corpus
from hydra_directflow.corpus_dataset import inspect_corpus, iter_batches
from hydra_directflow.local_data import GAUGES
from hydra_directflow.reference import digest
from hydra_directflow.trainer_bundle import assemble_bundle
from test_corpus_dataset import synthetic_plan, arrays_for


class TrainerBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root / 'chunks').mkdir(); (self.root / 'staging').mkdir()
        self.plan = synthetic_plan(); (self.root / 'PLAN.json').write_text(json.dumps(self.plan))
        self.roles = {'fit': {'start_utc': self.plan['splits'][0]['start_utc'], 'end_utc': self.plan['splits'][0]['end_utc']},
                      'outer': {'start_utc': self.plan['splits'][1]['start_utc'], 'end_utc': self.plan['splits'][1]['end_utc']}}

    def tearDown(self): self.temp.cleanup()

    def source(self, *, complete=True, missing_label=False):
        for ui, unit in enumerate(self.plan['units']):
            for ci, chunk in enumerate(unit['chunks']):
                if not complete and (ui, ci) != (0, 0): continue
                arrays = arrays_for(chunk)
                if missing_label and (ui, ci) == (0, 0):
                    arrays['target'][0, 0] = np.nan; arrays['target_mask'][0, 0] = False
                    arrays['eligibility_reason'][0] = 'TARGET_INCOMPLETE'
                receipt = {'case_source_digest': digest([ui, ci]), 'cache': {
                    'cycle_keys': {c: digest(c) for c in chunk['cycle_ids']}, 'month_keys': {'2021-08': digest('2021-08')}},
                    'stage_seconds': {'total': .01}}
                corpus._shard_write(self.root, self.plan['plan_sha256'], unit, chunk, arrays, receipt)
        snapshot = inspect_corpus(self.root, expected_plan_sha256=self.plan['plan_sha256'])
        return snapshot, [snapshot.select(gauge_id=GAUGES[0], split=split['name'], history_hours=168,
                         cohort_mode='all_candidates', allow_partial=not complete) for split in self.plan['splits']]

    def build(self, selections, **kwargs):
        return assemble_bundle(selections, feature_view=kwargs.pop('feature_view', 'full'),
                               role_time_bounds=kwargs.pop('role_time_bounds', self.roles),
                               max_total_bytes=kwargs.pop('max_total_bytes', 1024**2),
                               max_batch_reads=kwargs.pop('max_batch_reads', 100), batch_size=2, **kwargs)

    def test_dense_mapping_preserves_tensors_and_orders_explicit_selections_chronologically(self):
        _, selections = self.source()
        bundle = self.build(list(reversed(selections)))
        batches = [batch for selection in selections for batch in iter_batches(selection, feature_view='full', batch_size=2)]
        for name in ('history', 'history_mask', 'weather', 'weather_mask', 'state', 'state_mask', 'target', 'target_mask',
                     'case_id', 'origin_time', 'history_time', 'target_time', 'site_id'):
            np.testing.assert_equal(bundle.arrays[name], np.concatenate([batch[name] for batch in batches]))
            self.assertFalse(bundle.arrays[name].flags.writeable)
        self.assertEqual(bundle.manifest['candidate_selection_parts'][0]['caller_position'], 1)
        self.assertEqual(list(bundle.indices_for('fit')), list(range(6)))
        self.assertEqual(list(bundle.indices_for('outer')), list(range(6, 12)))
        with self.assertRaisesRegex(ValueError, 'no outer-to-stopping'):
            list(bundle.indices_for('inner_stop'))
        self.assertFalse(bundle.manifest['scientific_admission'])
        self.assertEqual(bundle.construction_report['scaler_fits'], 0)

    def test_full_and_q_only_share_membership_roles_history_labels_and_masks(self):
        _, selections = self.source()
        full, q = self.build(selections), self.build(selections, feature_view='q_only')
        self.assertEqual(full.arrays['cohort_admission_digest'].item(), q.arrays['cohort_admission_digest'].item())
        self.assertEqual(full.manifest['role_assignment_digest'], q.manifest['role_assignment_digest'])
        self.assertNotEqual(full.source_identity, q.source_identity)
        for name in ('history', 'history_mask', 'target', 'target_mask', 'case_id', 'target_time'):
            np.testing.assert_equal(full.arrays[name], q.arrays[name])
        self.assertEqual(q.arrays['weather'].shape, (12, 18, 1)); self.assertEqual(q.arrays['state'].shape, (12, 0))
        np.testing.assert_equal(q.arrays['weather'][0, :, 0], np.arange(1, 19))
        self.assertNotIn('source_cycle', full.arrays)
        self.assertTrue(full.manifest['shard_provenance'])

    def test_explicit_cutoff_not_first_later_observation_purges_crossing_labels(self):
        _, selections = self.source()
        roles = {'fit': {'start_utc': '2021-08-23T02:00:00Z', 'end_utc': '2021-08-23T12:00:00Z'},
                 'inner_stop': {'start_utc': '2021-08-23T12:00:00Z', 'end_utc': '2021-08-25T02:00:00Z'}}
        bundle = self.build(selections, role_time_bounds=roles)
        # First actual inner-stop origin is Aug24 02Z, but fit's exact cutoff is Aug23 12Z.
        self.assertEqual(bundle.manifest['roles']['fit']['case_count'], 0)
        self.assertEqual(bundle.manifest['roles']['fit']['target_boundary_purged_cases'], 6)
        self.assertTrue(bundle.manifest['roles']['fit']['zero_row_support'])
        self.assertEqual(bundle.manifest['roles']['inner_stop']['case_count'], 6)
        self.assertEqual(bundle.manifest['unassigned_cases'], 6)
        self.assertEqual(len(bundle.arrays['case_id']), 12)
        self.assertFalse(bundle.manifest['simulated_refit_label_cutoff_implemented'])

    def test_concrete_role_indices_work_directly_with_existing_split_validator(self):
        from hydra_directflow.training import validate_split
        from hydra_directflow.local_data import utc_epoch
        _, selections = self.source()
        roles = {'fit': self.roles['fit'], 'inner_stop': self.roles['outer']}
        bundle = self.build(selections, role_time_bounds=roles)
        fit, stopping = bundle.indices_for('fit'), bundle.indices_for('inner_stop')
        self.assertIs(type(fit), tuple); self.assertIs(type(stopping), tuple)
        self.assertTrue(all(type(index) is int for index in fit + stopping))
        boundary = utc_epoch(roles['inner_stop']['start_utc'])
        self.assertEqual(validate_split(bundle.arrays, fit, stopping, boundary_epoch=boundary), (fit, stopping))
        self.assertEqual(validate_split(bundle.arrays, fit, stopping, boundary_epoch=boundary), (fit, stopping))

    def test_missing_labels_are_preserved_without_changing_role_membership(self):
        _, selections = self.source(missing_label=True)
        bundle = self.build(selections)
        self.assertFalse(bundle.arrays['target_mask'][0, 0]); self.assertTrue(np.isnan(bundle.arrays['target'][0, 0]))
        self.assertEqual(bundle.manifest['roles']['fit']['case_count'], 6)
        self.assertEqual(bundle.manifest['roles']['fit']['complete_target_cases'], 5)
        self.assertEqual(bundle.manifest['roles']['fit']['observed_target_cells'], 6 * 18 - 1)

    def test_partial_snapshots_require_explicit_engineering_acknowledgment(self):
        _, selections = self.source(complete=False)
        with patch('hydra_directflow.trainer_bundle.iter_batches') as reads:
            with self.assertRaisesRegex(ValueError, 'Partial snapshots'):
                self.build(selections)
            reads.assert_not_called()
        bundle = self.build(selections, allow_partial=True)
        self.assertFalse(bundle.manifest['complete_source_snapshot'])
        self.assertEqual(bundle.manifest['roles']['outer']['case_count'], 0)
        self.assertTrue(bundle.manifest['roles']['outer']['zero_row_support'])

    def test_duplicate_overlapping_gauge_or_history_selections_reject_before_reads(self):
        snapshot, selections = self.source()
        other_gauge = snapshot.select(gauge_id=GAUGES[1], split='candidate_validation', history_hours=168, cohort_mode='all_candidates')
        shorter = snapshot.select(gauge_id=GAUGES[0], split='candidate_validation', history_hours=24, cohort_mode='all_candidates')
        for supplied in ([selections[0], selections[0]], [selections[0], other_gauge], [selections[0], shorter]):
            with self.subTest(supplied=supplied), patch('hydra_directflow.trainer_bundle.iter_batches') as reads:
                with self.assertRaises(ValueError): self.build(supplied)
                reads.assert_not_called()

    def test_byte_and_batch_allowances_reject_before_dense_source_reads(self):
        _, selections = self.source()
        for changes in ({'max_total_bytes': 1}, {'max_batch_reads': 0}, {'max_rss_bytes': 1}):
            with self.subTest(changes=changes), patch('hydra_directflow.trainer_bundle.iter_batches') as reads:
                with self.assertRaises(ValueError): self.build(selections, **changes)
                reads.assert_not_called()

    def test_role_overlap_protected_or_naive_times_are_rejected(self):
        _, selections = self.source()
        invalid = [dict(fit={'start_utc': '2021-08-23T00:00:00', 'end_utc': '2021-08-24T00:00:00Z'}),
                   dict(fit={'start_utc': '2024-07-01T00:00:00Z', 'end_utc': '2024-08-01T00:00:00Z'}),
                   dict(fit={'start_utc': '2021-08-23T00:00:00Z', 'end_utc': '2021-08-25T00:00:00Z'},
                        inner_stop={'start_utc': '2021-08-24T00:00:00Z', 'end_utc': '2021-08-26T00:00:00Z'})]
        for roles in invalid:
            with self.subTest(roles=roles), self.assertRaises(ValueError): self.build(selections, role_time_bounds=roles)

    def test_repeated_assembly_identity_excludes_runtime_measurements(self):
        _, selections = self.source()
        first, second = self.build(selections), self.build(selections)
        self.assertEqual(first.source_identity, second.source_identity)
        self.assertEqual(first.manifest['arrays_sha256'], second.manifest['arrays_sha256'])
        self.assertEqual(first.construction_report['dense_array_bytes'], sum(a.nbytes for a in first.arrays.values()))
        self.assertEqual(first.construction_report['batch_reads'], 6)
        with self.assertRaises(TypeError): first.arrays['new'] = np.array(1)


if __name__ == '__main__': unittest.main()
