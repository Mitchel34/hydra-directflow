"""Synthetic admission accounting; no models, raw sources, network or fitted statistics."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow import corpus, corpus_dataset as dataset
from hydra_directflow.local_data import GAUGES, iso
from hydra_directflow.reference import digest
from test_corpus_dataset import synthetic_plan, arrays_for

SPEC = importlib.util.spec_from_file_location('audit_corpus_admission', Path(__file__).resolve().parents[1] / 'scripts/audit_corpus_admission.py')
audit = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(audit)


class CorpusAdmissionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name) / 'corpus'
        self.root.mkdir(); (self.root / 'chunks').mkdir(); (self.root / 'staging').mkdir()
        self.plan = synthetic_plan(); self.save_plan()

    def tearDown(self): self.temp.cleanup()

    def save_plan(self):
        (self.root / 'PLAN.json').write_text(json.dumps(self.plan, sort_keys=True))

    def write(self, ui, ci, change=None):
        unit = self.plan['units'][ui]; chunk = unit['chunks'][ci]
        arrays = arrays_for(chunk)
        if change: change(arrays)
        arrays['eligibility_reason'] = dataset._view_reason(arrays)
        first = dataset._hour(chunk['first_origin_utc']); last = dataset._hour(chunk['last_origin_utc'])
        receipt = {'case_source_digest': digest([ui, ci]), 'cache': {
            'cycle_keys': {c: digest(c) for c in chunk['cycle_ids']},
            'month_keys': {month: digest(month) for month in dataset._months(first - 169*3600, last + 18*3600)}},
            'stage_seconds': {'total': .01}}
        corpus._shard_write(self.root, self.plan['plan_sha256'], unit, chunk, arrays, receipt)

    def populate(self, change_first=None):
        for ui, unit in enumerate(self.plan['units']):
            for ci, _ in enumerate(unit['chunks']): self.write(ui, ci, change_first if (ui, ci) == (0, 0) else None)

    def run_audit(self, **kwargs):
        return audit.audit_corpus(self.root, expected_plan_sha256=self.plan['plan_sha256'], **kwargs)

    def row(self, report, gauge=0, split='candidate_training'):
        return next(row for row in report['results'] if row['gauge_id'] == GAUGES[gauge] and row['split'] == split)

    def test_complete_snapshot_counts_all_gauges_splits_and_binds_provenance(self):
        self.populate(); report = self.run_audit(batch_size=1)
        self.assertEqual(report['status'], 'COMPLETE_SNAPSHOT_CANDIDATE_AUDIT')
        self.assertEqual(report['all_candidate_cases'], 36); self.assertEqual(len(report['results']), 6)
        self.assertTrue(all(row['candidate_counts'] == dict.fromkeys(('all_candidates', 'natural_L24', 'natural_L168', 'common_L24', 'common_L168'), 6) for row in report['results']))
        self.assertEqual(len(report['snapshot']['observed_commits']), 6)
        self.assertEqual(report['producer_implementation_sha256'], {'synthetic_fixture': 'a'*64})
        self.assertEqual(report['implementation_sha256'], audit.code_identity())
        self.assertEqual(report['source_identities']['qualified_index_sha256'], dataset.INDEX_SHA256)
        self.assertFalse(report['execution']['scientific_cohort_adopted']); self.assertFalse(report['execution']['discharge_magnitude_summaries_computed'])
        self.assertEqual(report['audit_content_sha256'], digest({k:v for k,v in report.items() if k not in ('audit_content_sha256', 'runtime')}))

    def test_old_history_missingness_changes_natural_L24_not_matched_L168(self):
        def change(a): a['history'][0, 0] = np.nan; a['history_mask'][0, 0] = False
        self.populate(change); row = self.row(self.run_audit())
        self.assertEqual(row['candidate_counts'], dict(all_candidates=6, natural_L24=6, natural_L168=5, common_L24=5, common_L168=5))
        self.assertEqual(row['natural_L24_cases_rescued_from_L168'], 1)
        self.assertEqual(row['mask_counts']['history_L168_discharge']['missing_cells'], 1)
        self.assertEqual(row['mask_counts']['history_L24_discharge']['missing_cells'], 0)
        self.assertEqual(row['view_exclusion_dimensions']['L168']['HISTORY_INCOMPLETE'], 1)
        self.assertEqual(row['view_exclusion_dimensions']['L24']['HISTORY_INCOMPLETE'], 0)
        self.assertEqual(row['candidate_selections']['common_L24']['cohort_admission_digest'], row['candidate_selections']['common_L168']['cohort_admission_digest'])

    def test_overlapping_exclusions_target_counts_and_independent_soil_availability(self):
        def change(a):
            for field in ('weather', 'state'):
                a[field][0] = np.nan; a[field + '_mask'][0] = False
            a['source_cycle'][0] = ''; a['soil_source_cycle'][0] = ''; a['weather_source_leads'][0] = -1
            a['target'][0, 17] = np.nan; a['target_mask'][0, 17] = False
        self.populate(change); row = self.row(self.run_audit())
        self.assertEqual(row['candidate_counts']['all_candidates'], 6); self.assertEqual(row['candidate_counts']['natural_L24'], 5)
        self.assertEqual(row['view_exclusion_dimensions']['L24']['TARGET_INCOMPLETE'], 1)
        self.assertEqual(row['weather']['unavailable_cases'], 1); self.assertEqual(row['soil']['unavailable_cases'], 1)
        self.assertEqual(row['mask_counts']['target']['missing_cells'], 1)
        self.assertEqual(row['mask_counts']['target_H18']['missing_cells'], 1)
        self.assertEqual(row['mask_counts']['target_H01']['missing_cells'], 0)
        self.assertEqual(sum(row['source_eligibility_counts'].values()), 6)

    def test_older_weather_count_uses_latest_nominal_cycle_not_same_soil_cycle(self):
        def change(a):
            origin = int(a['origin_time'][0]); older = ((origin - 7200) // 21600) * 21600 - 21600
            a['source_cycle'][0] = datetime.fromtimestamp(older, timezone.utc).strftime('%Y%m%d_%HZ')
            a['weather_source_leads'][0] = (a['target_time'][0] - older) // 3600
        self.populate(change); row = self.row(self.run_audit())
        self.assertEqual(row['weather']['selected_cases'], 6)
        self.assertEqual(row['weather']['older_than_latest_nominal_cases'], 1)
        self.assertEqual(row['weather']['common_l168_older_than_nominal_cases'], 1)
        self.assertEqual(row['weather']['older_by_hours_counts'], {'6': 1})
        self.assertEqual(row['weather']['age_hours_counts']['8'], 1)
        self.assertEqual(row['soil']['differs_from_weather_cycle_cases'], 1)
        self.assertEqual(row['soil']['unavailable_cases'], 0)

    def test_weather_domain_diagnostics_do_not_change_admission_or_report_magnitudes(self):
        def change(a):
            a['weather'][0, 0] = [-.1, 0., 1.1, -.1]
        self.populate(change); before = (self.root / 'chunks' / self.plan['units'][0]['id'] / '000' / 'model.npz').read_bytes()
        report = self.run_audit(); row = self.row(report)
        self.assertTrue(report['physical_domain_review_needed'])
        self.assertEqual(row['candidate_counts']['natural_L168'], 6)
        for name in audit.CORE_WEATHER_FEATURES:
            self.assertEqual(row['physical_domain_diagnostics'][name]['violation_cells'], 1)
            self.assertEqual(row['physical_domain_diagnostics'][name]['common_l168_cases_with_violation'], 1)
        self.assertEqual(row['physical_domain_diagnostics']['soil_fraction']['violation_cells'], 0)
        self.assertEqual(before, (self.root / 'chunks' / self.plan['units'][0]['id'] / '000' / 'model.npz').read_bytes())
        self.assertNotIn('minimum', json.dumps(report)); self.assertNotIn('mean', json.dumps(report)); self.assertNotIn('maximum_value', json.dumps(report))

    def test_existing_soil_domain_gate_is_not_relaxed_into_silent_admission(self):
        def change(a): a['state'][0, 0] = 1.01
        self.populate(change)
        with self.assertRaisesRegex(ValueError, 'Soil state age/value'): self.run_audit()

    def test_partial_requires_explicit_diagnostic_and_empty_split_is_visible(self):
        self.write(0, 0)
        with self.assertRaises(dataset.IncompleteCorpusError): self.run_audit()
        report = self.run_audit(diagnostic_partial=True)
        self.assertEqual(report['status'], 'PARTIAL_DIAGNOSTIC_NOT_FINAL_COHORT')
        self.assertEqual(report['all_candidate_cases'], 6); self.assertEqual(len(report['snapshot']['missing_chunks']), 5)
        empty = self.row(report, split='candidate_validation')
        self.assertEqual(empty['candidate_cases'], 0); self.assertEqual(empty['mask_counts']['target']['total_cells'], 0)
        self.assertEqual(empty['physical_domain_diagnostics']['temp_k']['observed_cells'], 0)
        self.assertFalse(empty['candidate_selections']['natural_L24']['final_scientific_cohort'])

    def test_corpus_digest_corruption_and_unsupported_memory_bounds_fail(self):
        self.populate()
        with self.assertRaisesRegex(ValueError, 'plan digest'):
            audit.audit_corpus(self.root, expected_plan_sha256='0'*64)
        for kwargs in ({'max_seconds':0}, {'max_seconds':901}, {'max_rss_bytes':1}, {'batch_size':129}):
            with self.subTest(kwargs=kwargs), patch.object(dataset, 'inspect_corpus', side_effect=AssertionError('Read before bound')):
                with self.assertRaises(ValueError): self.run_audit(**kwargs)
        path = self.root / 'chunks' / self.plan['units'][0]['id'] / '000' / 'model.npz'
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'payload hash/size'): self.run_audit()

    def test_report_publication_is_new_atomic_and_outside_corpus(self):
        self.populate(); report = self.run_audit()
        destination = Path(self.temp.name) / 'audit.json'
        audit.write_report(report, destination, source_root=self.root)
        self.assertEqual(json.loads(destination.read_text())['audit_content_sha256'], report['audit_content_sha256'])
        before = destination.read_bytes()
        with self.assertRaisesRegex(ValueError, 'no overwrites'): audit.write_report(report, destination, source_root=self.root)
        self.assertEqual(destination.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'outside the source corpus'):
            audit.write_report(report, self.root / 'new.json', source_root=self.root)
        self.assertFalse(list(Path(self.temp.name).glob('.admission-*')))

    def test_outside_calendar_warmup_remains_distinct_from_source_gap(self):
        split = dict(name='candidate_training', start_utc='2018-07-12T00:00:00Z', end_utc='2018-07-13T00:00:00Z')
        units = corpus.partition_split(split, history_hours=168, chunk_hours=2)
        core = {key: deepcopy(self.plan[key]) for key in dataset.PLAN_KEYS}
        core.update(splits=[split], units=units,
                    required_cycle_ids=sorted({c for u in units for x in u['chunks'] for c in x['cycle_ids']}),
                    calendar_unavailable_cycle_ids=sorted({c for u in units for x in u['chunks'] for c in x['calendar_unavailable_cycle_ids']}))
        self.plan = dict(**core, plan_sha256=digest(core), requested_origins=24, candidate_origins=6, purged_origins=18)
        self.save_plan()
        original = arrays_for(synthetic_plan()['units'][0]['chunks'][0])
        def warmup_arrays(chunk):
            arrays = {key: value.copy() for key, value in original.items()}
            times = np.repeat(np.array(corpus._origins(chunk), np.int64), 3)
            arrays['origin_time'] = times; arrays['history_time'] = times[:, None] - np.arange(169, 1, -1)*3600
            arrays['target_time'] = times[:, None] + np.arange(1, 19)*3600
            arrays['case_id'] = np.array([site + ':' + iso(int(t)) for t in times[::3] for site in GAUGES])
            for name in ('weather', 'state'): arrays[name][:] = np.nan; arrays[name + '_mask'][:] = False
            arrays['source_cycle'][:] = ''; arrays['soil_source_cycle'][:] = ''; arrays['weather_source_leads'][:] = -1
            return arrays
        with patch(__name__ + '.arrays_for', side_effect=warmup_arrays): self.populate()
        report = self.run_audit(); row = self.row(report)
        self.assertEqual(row['outside_calendar_warmup']['unresolved_cases'], 6)
        self.assertEqual(row['outside_calendar_warmup']['weather_unresolved_cases'], 6)
        self.assertEqual(row['weather']['nominal_cycle_before_acquisition_cases'], 2)
        self.assertEqual(report['calendar_unavailable_cycle_ids'], ['20180711_06Z','20180711_12Z','20180711_18Z'])
        self.assertEqual(row['candidate_counts']['all_candidates'], 6)
        self.assertEqual(row['candidate_counts']['natural_L168'], 0)


if __name__ == '__main__': unittest.main()
