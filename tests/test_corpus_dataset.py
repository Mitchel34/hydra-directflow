"""Synthetic immutable-shard dataset tests; no fitting, source reads or network."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from hydra_directflow import corpus, corpus_dataset as dataset
from hydra_directflow.local_data import GAUGES, iso, utc_epoch
from hydra_directflow.quality import PRIMARY_APPROVED
from hydra_directflow.reference import digest, ASSESSMENT_START


def synthetic_plan():
    splits = [{"name": "candidate_training", "start_utc": "2021-08-23T02:00:00Z", "end_utc": "2021-08-24T02:00:00Z"},
              {"name": "candidate_validation", "start_utc": "2021-08-24T02:00:00Z", "end_utc": "2021-08-25T02:00:00Z"}]
    units = [unit for split in splits for unit in corpus.partition_split(split, history_hours=168, chunk_hours=2)]
    core = dict(schema=dataset.SCHEMA, splits=splits, history_hours=168, chunk_hours=2,
                input_quality_policy=PRIMARY_APPROVED, target_quality_policy=PRIMARY_APPROVED,
                qualified_index_sha256=dataset.INDEX_SHA256, usgs_manifest_sha256=dataset.USGS_MANIFEST_SHA256,
                candidate_weights_sha256=dataset.WEIGHTS_SHA256, units=units,
                required_cycle_ids=sorted({c for u in units for x in u['chunks'] for c in x['cycle_ids']}),
                calendar_unavailable_cycle_ids=[], implementation_sha256={"synthetic_fixture": "a" * 64})
    return dict(**core, plan_sha256=digest(core), requested_origins=48, candidate_origins=12, purged_origins=36)


def arrays_for(chunk):
    origins = corpus._origins(chunk); times = np.repeat(np.array(origins, np.int64), 3); n = len(times)
    data = dict(origin_time=times, site_id=np.tile(np.arange(3, dtype=np.int64), len(origins)),
                history_time=times[:, None] - np.arange(169, 1, -1) * 3600,
                target_time=times[:, None] + np.arange(1, 19) * 3600,
                case_id=np.array([site + ':' + iso(origin) for origin in origins for site in GAUGES]),
                eligibility_reason=np.array(['COMPLETE'] * n, dtype='<U150'))
    for name, shape in (('history', (n, 168, 2)), ('weather', (n, 18, 4)), ('state', (n, 2)), ('target', (n, 18))):
        data[name], data[name + '_mask'] = np.ones(shape, np.float64), np.ones(shape, bool)
    data['history'][..., 0] = 100.; data['history'][..., 1] = np.arange(169, 1, -1)
    data['target'][:] = 101.; data['state'][:, 0] = .25
    cycles = [max(c for c in chunk['cycle_ids'] if corpus._epoch_cycle(c) <= origin - 7200) for origin in times]
    data['source_cycle'], data['soil_source_cycle'] = np.array(cycles), np.array(cycles)
    data['state'][:, 1] = [(origin - corpus._epoch_cycle(c)) / 3600 for origin, c in zip(times, cycles)]
    data['weather_source_leads'] = np.array([(data['target_time'][i] - corpus._epoch_cycle(c)) // 3600 for i, c in enumerate(cycles)])
    return data


class CorpusDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root / 'chunks').mkdir(); (self.root / 'staging').mkdir()
        self.plan = synthetic_plan()
        (self.root / 'PLAN.json').write_text(json.dumps(self.plan, sort_keys=True))

    def tearDown(self): self.temp.cleanup()

    def write(self, unit_index=0, chunk_index=0, change=None):
        unit = self.plan['units'][unit_index]; chunk = unit['chunks'][chunk_index]
        arrays = arrays_for(chunk)
        if change: change(arrays)
        receipt = {'case_source_digest': digest([unit_index, chunk_index]),
                   'cache': {'cycle_keys': {c: digest(c) for c in chunk['cycle_ids']}, 'month_keys': {'2021-08': digest('2021-08')}},
                   'stage_seconds': {'total': .01}}
        corpus._shard_write(self.root, self.plan['plan_sha256'], unit, chunk, arrays, receipt)
        return self.root / 'chunks' / unit['id'] / f'{chunk_index:03d}'

    def all(self, change_first=None):
        for ui, unit in enumerate(self.plan['units']):
            for ci, _ in enumerate(unit['chunks']):
                self.write(ui, ci, change_first if (ui, ci) == (0, 0) else None)
        return self.inspect()

    def inspect(self): return dataset.inspect_corpus(self.root, expected_plan_sha256=self.plan['plan_sha256'])

    def select(self, snapshot, *, history=168, mode='natural_complete', partial=False, split='candidate_training'):
        return snapshot.select(gauge_id=GAUGES[0], split=split, history_hours=history, cohort_mode=mode, allow_partial=partial)

    def tamper_arrays(self, path, change):
        with np.load(path / 'model.npz', allow_pickle=False) as source:
            arrays = {k: source[k].copy() for k in source.files}
        change(arrays)
        buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays); raw = buffer.getvalue()
        (path / 'model.npz').write_bytes(raw)
        commit = json.loads((path / 'COMMIT.json').read_text())
        commit['payloads']['model.npz'] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        commit['arrays'] = {k: {'shape': list(v.shape), 'dtype': v.dtype.str} for k, v in arrays.items()}
        (path / 'COMMIT.json').write_text(json.dumps(commit))

    def test_partial_snapshot_is_provisional_and_requires_explicit_selection_permission(self):
        self.write()
        snapshot = self.inspect(); report = snapshot.report()
        self.assertFalse(snapshot.complete); self.assertEqual(len(snapshot.missing_chunks), 5)
        self.assertEqual(report['available_candidate_cases'], 6); self.assertEqual(report['planned_cases'], 36)
        self.assertFalse(report['final_scientific_cohort']); self.assertFalse(report['upstream_sources_reaudited'])
        with self.assertRaises(dataset.IncompleteCorpusError): self.select(snapshot)
        chosen = self.select(snapshot, partial=True)
        self.assertEqual(chosen.case_count, 2)
        self.assertEqual(chosen.report()['status'], 'PARTIAL_SNAPSHOT_CANDIDATE_SELECTION')

    def test_complete_snapshot_groups_training_and_validation_without_boundary_leakage(self):
        snapshot = self.all()
        self.assertTrue(snapshot.complete)
        train, val = self.select(snapshot), self.select(snapshot, split='candidate_validation')
        self.assertEqual((train.case_count, val.case_count), (6, 6))
        training = list(dataset.iter_batches(train, feature_view='full', batch_size=1))
        validation = list(dataset.iter_batches(val, feature_view='full'))
        boundary = utc_epoch(self.plan['splits'][0]['end_utc'])
        self.assertTrue(all((b['target_time'] < boundary).all() for b in training))
        self.assertTrue(all((b['origin_time'] >= boundary).all() for b in validation))
        self.assertTrue(all(b['site_id'].tolist() == [0] for b in training))

    def test_l24_natural_counts_do_not_inherit_older_l168_history_exclusions(self):
        def missing_old(a):
            a['history'][0, 0] = np.nan; a['history_mask'][0, 0] = False
            a['eligibility_reason'][0] = 'HISTORY_INCOMPLETE'
        snapshot = self.all(missing_old)
        long = self.select(snapshot); short = self.select(snapshot, history=24)
        common_short = self.select(snapshot, history=24, mode='common_l168_complete')
        common_long = self.select(snapshot, history=168, mode='common_l168_complete')
        self.assertEqual((long.case_count, short.case_count, common_short.case_count), (5, 6, 5))
        self.assertEqual(common_short.cohort_admission_digest, common_long.cohort_admission_digest)
        self.assertNotEqual(common_short.selection_digest, common_long.selection_digest)
        first = next(dataset.iter_batches(short, feature_view='full'))
        self.assertEqual(first['history'].shape, (2, 24, 2)); self.assertTrue(first['history_mask'].all())
        self.assertFalse(first['source_admission_mask'][0]); self.assertTrue(first['view_complete_mask'][0])
        self.assertEqual(first['source_eligibility_reason'][0], 'HISTORY_INCOMPLETE')
        self.assertEqual(first['view_eligibility_reason'][0], 'COMPLETE')
        np.testing.assert_equal(first['history'][0, :, 1], np.arange(25, 1, -1))

    def test_recent_missing_history_remains_excluded_in_both_views(self):
        def missing_recent(a):
            a['history'][0, -1] = np.nan; a['history_mask'][0, -1] = False
            a['eligibility_reason'][0] = 'HISTORY_INCOMPLETE'
        snapshot = self.all(missing_recent)
        self.assertEqual(self.select(snapshot, history=24).case_count, 5)
        self.assertEqual(self.select(snapshot).case_count, 5)
        all_cases = self.select(snapshot, mode='all_candidates')
        self.assertEqual(all_cases.case_count, 6)
        first = next(dataset.iter_batches(all_cases, feature_view='full'))
        self.assertFalse(first['view_complete_mask'][0])

    def test_full_and_q_only_share_cohort_ids_labels_and_bounded_read_only_copies(self):
        selected = self.select(self.all(), history=24)
        full = list(dataset.iter_batches(selected, feature_view='full', batch_size=1))
        q = list(dataset.iter_batches(selected, feature_view='q_only', batch_size=1))
        for left, right in zip(full, q):
            for name in ('history', 'history_mask', 'target', 'target_mask', 'case_id', 'origin_time', 'history_time', 'target_time'):
                np.testing.assert_equal(left[name], right[name])
            self.assertEqual(left['cohort_admission_digest'].item(), right['cohort_admission_digest'].item())
            self.assertEqual(right['weather'].shape, (1, 18, 1)); self.assertEqual(right['state'].shape, (1, 0))
            np.testing.assert_equal(right['weather'][0, :, 0], np.arange(1, 19))
            self.assertTrue(all(not value.flags.writeable for value in right.values()))

    def test_q_only_does_not_resurrect_cases_lacking_full_weather_support(self):
        def missing_weather(a):
            a['weather'][0] = np.nan; a['weather_mask'][0] = False; a['source_cycle'][0] = ''
            a['weather_source_leads'][0] = -1; a['eligibility_reason'][0] = 'NO_COMPLETE_SINGLE_WEATHER_RUN'
        snapshot = self.all(missing_weather)
        selected = self.select(snapshot, history=24)
        self.assertEqual(selected.case_count, 5)
        self.assertEqual(sum(len(b['case_id']) for b in dataset.iter_batches(selected, feature_view='q_only')), 5)

    def test_later_appends_require_new_snapshot_and_never_expand_old_selection(self):
        self.write(); old = self.inspect(); selected = self.select(old, partial=True)
        self.write(0, 1); new = self.inspect()
        self.assertNotEqual(old.snapshot_sha256, new.snapshot_sha256)
        self.assertEqual(selected.case_count, 2)
        self.assertEqual(sum(len(b['case_id']) for b in dataset.iter_batches(selected, feature_view='full')), 2)
        self.assertEqual(self.select(new, partial=True).case_count, 4)
        self.assertEqual(len(old.shards), 1)

    def test_payload_corruption_is_not_silently_downgraded_to_missing_chunk(self):
        path = self.write(); payload = path / 'model.npz'; raw = bytearray(payload.read_bytes()); raw[-1] ^= 1; payload.write_bytes(raw)
        with patch.object(dataset, '_decode', side_effect=AssertionError('Decoded corrupt bytes')):
            with self.assertRaisesRegex(ValueError, 'payload hash'):
                self.inspect()

    def test_missing_committed_payload_and_unplanned_directory_fail(self):
        path = self.write(); (path / 'model.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'members'): self.inspect()
        (self.root / 'chunks' / 'unplanned').mkdir()
        with self.assertRaisesRegex(ValueError, 'Unplanned'): self.inspect()

    def test_duplicate_order_and_boundary_corruption_fail_before_feature_value_decode(self):
        cases = (lambda a: a['case_id'].__setitem__(1, a['case_id'][0]),
                 lambda a: a['origin_time'].__setitem__(0, a['origin_time'][0] + 3600),
                 lambda a: a['target_time'].__setitem__((0, -1), ASSESSMENT_START))
        for change in cases:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                original_root = self.root
                self.root = Path(directory); (self.root / 'chunks').mkdir(); (self.root / 'staging').mkdir()
                (self.root / 'PLAN.json').write_text(json.dumps(self.plan))
                path = self.write(); self.tamper_arrays(path, change)
                decode = dataset._decode
                def guarded(archive, commit, name, shape):
                    if name in ('history', 'weather', 'state', 'target'): self.fail('Read values before clock/ID guard')
                    return decode(archive, commit, name, shape)
                with patch.object(dataset, '_decode', side_effect=guarded), self.assertRaises(ValueError): self.inspect()
                self.root = original_root

    def test_numeric_dtype_and_admission_corruption_are_rejected(self):
        path = self.write(); self.tamper_arrays(path, lambda a: a.__setitem__('history', a['history'].astype('<f4')))
        with self.assertRaisesRegex(ValueError, 'float64'): self.inspect()
        # Reuse a fresh independent fixture to exercise admission rather than dtype.
        with tempfile.TemporaryDirectory() as directory:
            original_root = self.root; self.root = Path(directory)
            (self.root / 'chunks').mkdir(); (self.root / 'staging').mkdir(); (self.root / 'PLAN.json').write_text(json.dumps(self.plan))
            path = self.write(); self.tamper_arrays(path, lambda a: a['admission_mask'].__setitem__(0, False))
            with self.assertRaisesRegex(ValueError, 'admission'): self.inspect()
            self.root = original_root

    def test_snapshot_rejects_changed_commit_even_if_new_payload_is_self_consistent(self):
        path = self.write(); snapshot = self.inspect(); selected = self.select(snapshot, partial=True)
        self.tamper_arrays(path, lambda a: a['target'].__setitem__((0, 0), 102.))
        with self.assertRaisesRegex(ValueError, 'commitment changed'):
            list(dataset.iter_batches(selected, feature_view='full'))

    def test_expected_plan_hash_and_unit_order_are_checked(self):
        with self.assertRaisesRegex(ValueError, 'plan SHA'):
            dataset.inspect_corpus(self.root, expected_plan_sha256='0' * 64)
        changed = deepcopy(self.plan); changed['units'].reverse()
        changed['plan_sha256'] = digest({k: changed[k] for k in dataset.PLAN_KEYS})
        (self.root / 'PLAN.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'unordered'):
            dataset.inspect_corpus(self.root, expected_plan_sha256=changed['plan_sha256'])

    def test_no_implicit_history_or_cohort_policy_and_batch_bound(self):
        snapshot = self.all()
        for change in ({'cohort_mode': 'auto'}, {'history_hours': 720}, {'gauge_id': 'unknown'}, {'split': 'assessment'}):
            args = dict(gauge_id=GAUGES[0], split='candidate_training', history_hours=168, cohort_mode='natural_complete')
            with self.subTest(change=change), self.assertRaises(ValueError): snapshot.select(**{**args, **change})
        with self.assertRaises(ValueError): list(dataset.iter_batches(self.select(snapshot), feature_view='full', batch_size=10000))


if __name__ == '__main__': unittest.main()
