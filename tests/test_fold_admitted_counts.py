"""Fold-count boundary and authority tests; synthetic coordinates, no models or source arrays."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra_directflow import corpus_dataset as dataset
from hydra_directflow.local_data import GAUGES
from hydra_directflow.reference import digest

SPEC = importlib.util.spec_from_file_location('plan_fold_admitted_counts', Path(__file__).resolve().parents[1] / 'scripts/plan_fold_admitted_counts.py')
planner = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(planner)


def records(*timestamps):
    return [(planner.epoch(stamp), GAUGES[0] + ':' + stamp) for stamp in timestamps]


def synthetic_snapshot():
    shards = []
    for index, (split, stamp) in enumerate((
        ('candidate_training', '2019-04-30T05:00:00Z'),
        ('candidate_training', '2023-04-30T05:00:00Z'),
        ('candidate_training', '2023-05-01T00:00:00Z'),
        ('candidate_training', '2023-07-30T03:00:00Z'),
        ('candidate_validation', '2023-07-31T05:00:00Z'))):
        unit = dict(id=f'{split}__{stamp[:7]}__L168', split=split, history_hours=168)
        chunk = dict(index=0, first_origin_utc=stamp, origin_count=2)
        mask = tuple(value != 1 for value in range(6))
        shards.append(dataset.ShardSnapshot(json.dumps(unit), json.dumps(chunk), digest(index), mask, mask))
    return dataset.CorpusSnapshot('synthetic-unused-root', 'a'*64, 'b'*64, 'c'*64, 168,
                                  ('candidate_training','candidate_validation'), tuple(shards), (), 30)


def synthetic_audit(snapshot):
    rows = []
    for split in snapshot.splits:
        for gauge in GAUGES:
            chosen = snapshot.select(gauge_id=gauge, split=split, history_hours=168,
                                     cohort_mode='common_l168_complete', allow_partial=False)
            rows.append(dict(gauge_id=gauge, split=split, candidate_selections={'common_L168':chosen.report()},
                             candidate_counts={'common_L168':chosen.case_count}))
    report = dict(schema='hydra-corpus-admission-audit/v1', status='COMPLETE_SNAPSHOT_CANDIDATE_AUDIT',
                  snapshot=snapshot.report(), implementation_sha256=planner.reader_identity(), splits=planner.CANONICAL_SPLITS,
                  quality=dict(input_policy=planner.PRIMARY_APPROVED, target_policy=planner.PRIMARY_APPROVED,
                               original_flags_reconstructed=False),
                  execution=dict(scientific_cohort_adopted=False, protected_assessment_access=False, optimizer_updates=0, scaler_fits=0),
                  source_identities=dict(qualified_index_sha256=dataset.INDEX_SHA256, usgs_manifest_sha256=dataset.USGS_MANIFEST_SHA256,
                                         candidate_weights_sha256=dataset.WEIGHTS_SHA256),
                  producer_implementation_sha256={'synthetic':'e'*64}, physical_domain_review_needed=False, results=rows)
    report['audit_content_sha256'] = digest(report); report['runtime'] = {'elapsed_seconds':0}
    return report


class FoldAdmittedCountsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.path = Path(self.temp.name) / 'audit.json'
        self.snapshot = synthetic_snapshot(); self.report = synthetic_audit(self.snapshot)
        self.save()

    def tearDown(self): self.temp.cleanup()

    def save(self, rehash=True):
        if rehash:
            self.report['audit_content_sha256'] = digest({k:v for k,v in self.report.items() if k not in ('audit_content_sha256','runtime')})
        raw = (json.dumps(self.report, sort_keys=True) + '\n').encode(); self.path.write_bytes(raw)
        self.sha = hashlib.sha256(raw).hexdigest()

    def load(self, sha=None):
        return planner.load_admission_report(self.path, expected_sha256=sha or self.sha, expected_plan_sha256='a'*64)

    def build(self):
        with patch.object(dataset, 'inspect_corpus', return_value=self.snapshot):
            return planner.build_plan('synthetic-unused-root', self.path,
                                      expected_plan_sha256='a'*64, expected_admission_sha256=self.sha)

    def test_strict18_and_optional19_boundaries_preserve_inclusive_refit_cutoff(self):
        data = records('2023-07-30T03:00:00Z','2023-07-30T04:00:00Z','2023-07-30T05:00:00Z','2023-07-30T06:00:00Z')
        no_delay = planner.fold_windows(simulated_label_delay_hours=0)[-1]['inner_stopping']
        delay = planner.fold_windows(simulated_label_delay_hours=2)[-1]['inner_stopping']
        zero, two = planner.count_role(data, no_delay), planner.count_role(data, delay)
        self.assertEqual((no_delay['purged_origins'],delay['purged_origins']), (18,19))
        self.assertEqual((zero['admitted_cases'],two['admitted_cases']), (3,2))
        self.assertEqual(two['last_admitted_origin_utc'], '2023-07-30T04:00:00Z')
        self.assertEqual(two['last_admitted_target_utc'], '2023-07-30T22:00:00Z')
        self.assertEqual(two['strict_target_window_excluded_common_cases'], 1)
        self.assertEqual(two['refit_label_delay_excluded_common_cases'], 1)
        self.assertEqual(two['case_ids_sha256'], digest([item[1] for item in data[:2]]))

    def test_earlier_fit_boundary_stays18_and_outer_unchanged_under_both_delays(self):
        data = records('2023-04-30T05:00:00Z','2023-04-30T06:00:00Z')
        for delay in (0,2):
            window = planner.fold_windows(simulated_label_delay_hours=delay)[-1]['gradient_fit']
            value = planner.count_role(data, window)
            self.assertEqual(window['purged_origins'], 18); self.assertEqual(value['admitted_cases'], 1)
            self.assertEqual(value['refit_label_delay_excluded_common_cases'], 0)
        values = []
        for delay in (0,2):
            window = planner.fold_windows(simulated_label_delay_hours=delay)[-1]['outer_scoring']
            values.append(planner.count_role(records('2024-07-30T05:00:00Z','2024-07-30T06:00:00Z'), window))
        self.assertEqual([v['admitted_cases'] for v in values], [1,1])
        self.assertEqual(values[0]['case_ids_sha256'], values[1]['case_ids_sha256'])

    def test_missing_boundary_cases_do_not_move_cutoffs_or_force_a_count_difference(self):
        data = records('2023-07-30T03:00:00Z','2023-08-10T00:00:00Z')
        counts = [planner.count_role(data, planner.fold_windows(simulated_label_delay_hours=d)[-1]['inner_stopping']) for d in (0,2)]
        self.assertEqual([c['admitted_cases'] for c in counts], [1,1])
        # A later surviving outer origin does not shift the fixed May gradient-fit end.
        fit = planner.count_role(records('2023-04-30T10:00:00Z','2023-05-10T00:00:00Z'),
                                 planner.fold_windows()[ -1]['gradient_fit'])
        self.assertEqual(fit['admitted_cases'],0); self.assertTrue(fit['zero_row_support'])
        self.assertEqual(fit['case_ids_sha256'],digest([]))

    def test_coordinate_membership_is_exact_sorted_per_gauge_not_dense_calendar(self):
        selected = self.snapshot.select(gauge_id=GAUGES[1], split='candidate_training', history_hours=168,
                                        cohort_mode='common_l168_complete', allow_partial=False)
        coordinates = planner.selected_coordinates(selected)
        self.assertEqual(len(coordinates),4)
        self.assertEqual(coordinates[0][1], GAUGES[1]+':2019-04-30T06:00:00Z')
        self.assertEqual(coordinates[-1][1], GAUGES[1]+':2023-07-30T04:00:00Z')
        wrong = dataset.DatasetSelection(selected.inventory, GAUGES[0], selected.split,168,selected.cohort_mode,
                                         False,selected.selected_rows,selected.cohort_admission_digest,selected.selection_digest)
        with self.assertRaisesRegex(ValueError,'gauge/row'): planner.selected_coordinates(wrong)

    def test_complete_proposal_shapes_parent_bindings_and_deterministic_digest(self):
        result = self.build(); repeated = self.build()
        self.assertEqual(result['fold_counts_sha256'], repeated['fold_counts_sha256'])
        self.assertEqual([s['simulated_label_delay_hours'] for s in result['scenarios']], [0,2])
        self.assertEqual(result['scope_fit_count'],180); self.assertFalse(result['final_scientific_cohort'])
        for scenario in result['scenarios']:
            self.assertEqual([f['name'] for f in scenario['folds']], ['cv_1','cv_2','cv_3','final_development'])
            for fold in scenario['folds']:
                self.assertEqual(len(fold['gauge_results']),3)
                for gauge in fold['gauge_results']:
                    self.assertEqual(tuple(gauge['roles']), planner.ROLES)
                    for role, row in gauge['roles'].items():
                        expected_split = 'candidate_validation' if fold['name']=='final_development' and role=='outer_scoring' else 'candidate_training'
                        self.assertEqual(row['source_split'],expected_split)
                        self.assertLessEqual(row['admitted_cases'],row['parent_selection_cases'])
                        self.assertLessEqual(row['admitted_cases'],row['window']['calendar_origins'])
                        self.assertTrue(dataset._hex(row['case_ids_sha256']))
        self.assertEqual(result['audit_content_sha256'],self.report['audit_content_sha256'])
        self.assertEqual(result['admission_report_sha256'],self.sha)
        self.assertEqual(result['fold_counts_sha256'],digest({k:v for k,v in result.items() if k not in ('fold_counts_sha256','runtime')}))
        self.assertEqual(result['execution']['optimizer_updates'],0)
        self.assertFalse(result['execution']['discharge_magnitude_summaries_computed'])

    def test_file_hash_content_hash_reader_and_quality_tampering_reject(self):
        with self.assertRaisesRegex(ValueError,'report SHA256'): self.load(sha='f'*64)
        original = deepcopy(self.report)
        self.report['physical_domain_review_needed']=True; self.save(rehash=False)
        with self.assertRaisesRegex(ValueError,'content digest'): self.load()
        for mutation, message in ((lambda r:r['implementation_sha256'].__setitem__('scripts/audit_corpus_admission.py','0'*64),'reader/audit code'),
                                  (lambda r:r['quality'].__setitem__('input_policy','RELAXED'),'quality policy'),
                                  (lambda r:r['splits'][0].__setitem__('end_utc','2023-08-01T00:00:00Z'),'Canonical full')):
            self.report=deepcopy(original); mutation(self.report); self.save()
            with self.subTest(message=message), self.assertRaisesRegex(ValueError,message): self.load()

    def test_partial_snapshot_mismatched_commits_and_changed_selection_reject(self):
        original=deepcopy(self.report)
        self.report['snapshot']['missing_chunks']=['missing']; self.save()
        with self.assertRaisesRegex(ValueError,'Complete admission'): self.load()
        self.report=deepcopy(original); self.report['snapshot']['observed_commits'][0]['commit_sha256']='f'*64; self.save()
        with self.assertRaisesRegex(ValueError,'snapshot differs'): self.build()
        self.report=deepcopy(original); self.report['results'][0]['candidate_counts']['common_L168'] += 1; self.save()
        with self.assertRaisesRegex(ValueError,'selection differs'): self.build()

    def test_bad_window_and_unordered_or_duplicate_coordinates_reject(self):
        window = planner.fold_windows()[-1]['inner_stopping']
        changed = dict(window, last_origin_utc='2023-07-30T05:00:00Z')
        with self.assertRaisesRegex(ValueError,'calendar window'): planner.count_role([],changed)
        data=records('2023-07-30T03:00:00Z','2023-07-30T04:00:00Z')
        for values in (list(reversed(data)),[data[0],data[0]]):
            with self.assertRaisesRegex(ValueError,'Unique chronological'): planner.count_role(values,window)


if __name__ == '__main__': unittest.main()
