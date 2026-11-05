"""Synthetic prediction/label artifacts only; no models or user source reads."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow import post_data as post
from hydra_directflow.contract import ARCHITECTURES
from hydra_directflow.local_data import iso
from hydra_directflow.quality import PRIMARY_APPROVED
from hydra_directflow.reference import ASSESSMENT_START, digest
from hydra_directflow.scoring import ScoreIdentity, score_physical_h18
from hydra_directflow.trainer_bundle import _array_digest


class PostDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / 'run'
        self.run.mkdir()
        self.source = self.root / 'synthetic-corpus-never-read'
        self.gauge, self.fold = '03161000', 'cv_1'
        self.start = int(datetime(2021, 7, 31, tzinfo=timezone.utc).timestamp())
        self.calendar_hours = 720
        self.positions = np.arange(0, self.calendar_hours, 3, dtype=np.int64)
        self.origin = self.start + self.positions * 3600
        self.future = self.origin[:, None] + np.arange(1, 19, dtype=np.int64) * 3600
        self.ids = np.array([self.gauge + ':' + iso(int(t)) for t in self.origin], dtype='<U29')
        self.truth = (10 + (self.future // 3600) % 23).astype(np.float64)
        self.sites = np.zeros(len(self.origin), np.int64)
        self.label_arrays = dict(origin_time=self.origin, target_time=self.future, site_id=self.sites,
                                 case_id=self.ids, target=self.truth, target_mask=np.ones_like(self.truth, bool))
        self.cohort = 'd' * 64
        self.plan = dict(plan_sha256='a' * 64, protocol_sha256='b' * 64, source_plan_sha256='c' * 64,
                         snapshot_sha256='e' * 64, implementation_sha256={'synthetic.py': 'f' * 64},
                         run_directory=str(self.run), corpus_root=str(self.source), protected_assessment_access=False,
                         jobs=[])
        self.records, self.payloads, self.sidecars, self.metric_payloads = [], [], [], []
        for architecture in ARCHITECTURES:
            for arm in ('full', 'q_only'):
                self.add(architecture, arm)

    def tearDown(self):
        self.temp.cleanup()

    def put(self, directory, name, raw):
        p = directory / name
        p.write_bytes(raw)
        return dict(path=str(p.relative_to(self.run)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def add(self, architecture, arm):
        order = len(self.records)
        job_id = f'{self.fold}__{self.gauge}__{architecture}__{arm}_L168__seed17'
        original = dict(job_id=job_id, order=order, architecture=architecture, input_arm=arm,
                        history_hours=168, seed=17, device='cpu', gauge_id=self.gauge, fold=self.fold,
                        actual_common_cases={'outer_scoring': len(self.origin)},
                        role_case_ids_sha256={'outer_scoring': digest(self.ids.tolist())})
        job = {**original, 'data_job': deepcopy(original)}
        self.plan['jobs'].append(job)
        directory = self.run / 'jobs' / job_id
        directory.mkdir(parents=True)
        schema = 'FULL_HRRR_H2_W4_S2_V1' if arm == 'full' else 'Q_ONLY_H2_HORIZON1_S0_V1'
        raw_sha = _array_digest(self.label_arrays)
        core = dict(schema='hydra-primary-campaign-data/v1', operation='labels', quality_policy=PRIMARY_APPROVED,
                    source_plan_sha256=self.plan['source_plan_sha256'], snapshot_sha256=self.plan['snapshot_sha256'],
                    fold=self.fold, gauge_id=self.gauge, input_arm=arm, history_hours=168,
                    roles={'outer': {'cases': len(self.origin), 'case_ids_sha256': digest(self.ids.tolist())}},
                    case_count=len(self.origin), cohort_admission_digest=self.cohort, raw_arrays_sha256=raw_sha,
                    role_time_bounds={'outer': {'start_utc': iso(self.start),
                                               'end_utc': iso(self.start + (self.calendar_hours + 18) * 3600)}})
        sidecar = dict(**core, source_identity=digest(core), returned_arrays_sha256=raw_sha,
                       construction={'source_bytes_read': 1024})
        sidecar_bytes = json.dumps(sidecar).encode()
        self.put(directory, 'outer-label-source.json', sidecar_bytes)
        self.sidecars.append(sidecar)
        bindings = dict(plan_sha256=self.plan['plan_sha256'], protocol_sha256=self.plan['protocol_sha256'],
                        source_plan_sha256=self.plan['source_plan_sha256'], snapshot_sha256=self.plan['snapshot_sha256'],
                        implementation_sha256=self.plan['implementation_sha256'], job_id=job_id, job_sha256=digest(job))
        receipt = dict(**bindings, status='VERIFIED_TRAIN_EVAL', protected_assessment_access=False,
                       receipt_path=str((directory / 'completion.json').relative_to(self.run)),
                       inference_cohort_admission_digest=self.cohort, outer_case_ids_sha256=digest(self.ids.tolist()),
                       label_source_identity=sidecar['source_identity'], inference_source_identity='1' * 64,
                       training_source_identity='2' * 64, training_cohort_admission_digest='3' * 64,
                       selected_best_epoch=7, hashes={'checkpoint': '4' * 64}, artifacts={})
        meta = {**bindings, **{k: receipt[k] for k in ('training_source_identity', 'training_cohort_admission_digest',
                'inference_source_identity', 'inference_cohort_admission_digest', 'selected_best_epoch')}}
        meta.update(checkpoint_sha256=receipt['hashes']['checkpoint'], case_ids_sha256=digest(self.ids.tolist()),
                    case_count=len(self.origin), prediction_units='m3/s', label_values_opened=False,
                    posthoc_clipping=False, best_only=True, optimizer_constructed=False, scaler_fitted=False)
        predictions = self.truth + (order + 1) * .125
        if order == 7:
            predictions = self.truth - 25
        payload = dict(prediction_q=predictions, case_id=self.ids, origin_time=self.origin,
                       forecast_time=self.future, site_id=self.sites, metadata_json=np.array(json.dumps(meta)))
        self.payloads.append(payload)
        stream = io.BytesIO()
        np.savez_compressed(stream, **payload)
        pred = self.put(directory, 'predictions.npz', stream.getvalue())
        receipt['hashes']['predictions'] = pred['sha256']
        receipt['artifacts']['predictions'] = pred
        panel = score_physical_h18(case_ids=self.ids, origins=self.origin, forecast_time=self.future,
            labels=self.truth, label_mask=np.ones_like(self.truth, bool), predictions=predictions,
            identity=ScoreIdentity(architecture, f'{architecture}-{arm}-L168', self.gauge, schema, 168, self.fold, 17),
            input_cohort_digest=self.cohort, label_policy_digest=digest({'policy': PRIMARY_APPROVED}),
            label_source_digest=sidecar['source_identity'])
        metrics = dict(**bindings, checkpoint_sha256=receipt['hashes']['checkpoint'],
                       predictions_sha256=pred['sha256'], outer_case_ids_sha256=digest(self.ids.tolist()), panel=panel)
        self.metric_payloads.append(metrics)
        metric = self.put(directory, 'metrics.json', json.dumps(metrics).encode())
        receipt['hashes']['metrics'] = metric['sha256']
        receipt['artifacts']['metrics'] = metric
        self.records.append(dict(job=job, receipt=receipt, receipt_sha256=digest(receipt),
                                 label_source_manifest_sha256=hashlib.sha256(sidecar_bytes).hexdigest()))

    def labels(self, corpus_root, job, **kwargs):
        self.assertEqual(str(corpus_root), str(self.source))
        self.assertEqual(job['input_arm'], 'full')
        self.assertEqual(job['history_hours'], 168)
        self.assertEqual(kwargs['expected_outer_cohort_digest'], self.cohort)
        self.assertEqual(kwargs['expected_case_ids_sha256'], digest(self.ids.tolist()))
        self.assertEqual(kwargs['max_read_bytes'], 64 * 1024**2)
        return dict(arrays=self.label_arrays, source_identity=self.sidecars[0]['source_identity'],
                    manifest=self.sidecars[0])

    def build(self, **kwargs):
        with patch.object(post, 'load_outer_labels', side_effect=self.labels) as loader:
            result = post.build_group(self.plan, self.run, self.source, self.records, self.fold, self.gauge, **kwargs)
        self.assertEqual(loader.call_count, 1)
        return result

    def repack(self, index):
        record = self.records[index]
        directory = self.run / 'jobs' / record['job']['job_id']
        stream = io.BytesIO()
        np.savez_compressed(stream, **self.payloads[index])
        item = self.put(directory, 'predictions.npz', stream.getvalue())
        record['receipt']['artifacts']['predictions'] = item
        record['receipt']['hashes']['predictions'] = item['sha256']
        record['receipt_sha256'] = digest(record['receipt'])

    def test_complete_calendar_fixed_cohort_and_independent_metrics(self):
        reservations = []
        arrays, manifest, audit = self.build(read_budget=reservations.append)
        self.assertEqual(arrays['predictions'].shape, (8, 720, 18))
        self.assertEqual(int(arrays['comparison_mask'].sum()), 240)
        self.assertEqual(manifest['calendar_hours'], 720)
        self.assertEqual(manifest['admitted_origins'], 240)
        self.assertEqual(manifest['label_read_bytes_reserved'], 64 * 1024**2)
        self.assertEqual(sum(reservations), manifest['artifact_bytes_read'] + manifest['label_read_bytes_reserved'])
        self.assertTrue(np.isnan(arrays['labels'][~arrays['comparison_mask']]).all())
        self.assertTrue(np.isnan(arrays['predictions'][:, ~arrays['comparison_mask']]).all())
        self.assertFalse(arrays['label_mask'][~arrays['comparison_mask']].any())
        self.assertTrue(np.array_equal(arrays['labels'][self.positions], self.truth))
        self.assertTrue(np.array_equal(arrays['predictions'][0, self.positions], self.payloads[0]['prediction_q']))
        self.assertTrue(all(not x.flags.writeable for x in arrays.values()))
        self.assertEqual(audit['status'], 'ALL_RAW_PREDICTIONS_AND_PHYSICAL_METRICS_MATCH')
        self.assertEqual(audit['panels'], 8)
        for lead in audit['results'][0]['independently_recomputed']['leads']:
            self.assertEqual(lead['physical_mse'], .125**2)
            self.assertEqual(lead['bias'], .125)
            self.assertEqual(lead['negative_predictions_scored'], 0)
        self.assertTrue(any(x['negative_predictions_scored'] > 0 for x in audit['results'][-1]['independently_recomputed']['leads']))

    def test_group_inventory_requires_all_arms_models_without_duplicates(self):
        for records in (self.records[:-1], self.records[:-1] + [self.records[0]]):
            with patch.object(post, 'load_outer_labels', side_effect=AssertionError('Labels opened')):
                with self.assertRaisesRegex(ValueError, 'paired group inventory'):
                    post.build_group(self.plan, self.run, self.source, records, self.fold, self.gauge)

    def test_prediction_hash_failure_is_before_label_values(self):
        descriptor = self.records[0]['receipt']['artifacts']['predictions']
        path = self.run / descriptor['path']
        raw = bytearray(path.read_bytes());raw[-1] ^= 1;path.write_bytes(raw)
        with patch.object(post, 'load_outer_labels', side_effect=AssertionError('Labels opened')):
            with self.assertRaisesRegex(ValueError, 'Artifact SHA differs'):
                post.build_group(self.plan, self.run, self.source, self.records, self.fold, self.gauge)

    def test_published_sidecar_hash_failure_is_before_prediction_values(self):
        record = self.records[0]
        path = self.run / Path(record['receipt']['receipt_path']).parent / 'outer-label-source.json'
        data = json.loads(path.read_text());data['construction']['source_bytes_read'] += 1;path.write_text(json.dumps(data))
        with patch.object(post, '_predictions', side_effect=AssertionError('Predictions decoded')):
            with self.assertRaisesRegex(ValueError, 'Artifact SHA differs'):
                post.build_group(self.plan, self.run, self.source, self.records, self.fold, self.gauge)

    def test_prediction_metadata_and_protected_clocks_fail_before_outcome_decode(self):
        for condition in ('metadata', 'protected'):
            original = deepcopy(self.payloads[0])
            if condition == 'metadata':
                meta = json.loads(self.payloads[0]['metadata_json'].item());meta['plan_sha256'] = '0' * 64
                self.payloads[0]['metadata_json'] = np.array(json.dumps(meta))
            else:
                self.payloads[0]['origin_time'] = np.full_like(self.origin, ASSESSMENT_START)
                self.payloads[0]['forecast_time'] = self.payloads[0]['origin_time'][:, None] + np.arange(1, 19) * 3600
            # Invalid outcome geometry must not be reached before metadata/clocks.
            self.payloads[0]['prediction_q'] = np.array([1.])
            self.repack(0)
            with self.subTest(condition=condition), patch.object(post, 'load_outer_labels', side_effect=AssertionError('Labels opened')):
                with self.assertRaisesRegex(ValueError, 'metadata identity|development origins'):
                    post.build_group(self.plan, self.run, self.source, self.records, self.fold, self.gauge)
            self.payloads[0] = original
            self.repack(0)

    def test_requested_nonfinite_prediction_fails_without_intersection(self):
        self.payloads[0]['prediction_q'] = self.payloads[0]['prediction_q'].copy()
        self.payloads[0]['prediction_q'][5, 3] = np.nan
        self.repack(0)
        with patch.object(post, 'load_outer_labels', side_effect=AssertionError('Labels opened')):
            with self.assertRaisesRegex(ValueError, 'nonfinite requested prediction'):
                post.build_group(self.plan, self.run, self.source, self.records, self.fold, self.gauge)

    def test_label_typed_hash_and_exact_coordinate_join_are_required(self):
        self.label_arrays = {**self.label_arrays, 'target': self.truth.copy()}
        self.label_arrays['target'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'Canonical label bytes/source differ'):
            self.build()

    def test_independent_metric_disagreement_stops_group(self):
        record = self.records[0]
        metrics = deepcopy(self.metric_payloads[0]);metrics['panel']['leads'][0]['rmse'] += .2
        item = self.put(self.run / 'jobs' / record['job']['job_id'], 'metrics.json', json.dumps(metrics).encode())
        record['receipt']['artifacts']['metrics'] = item;record['receipt']['hashes']['metrics'] = item['sha256']
        record['receipt_sha256'] = digest(record['receipt'])
        with self.assertRaisesRegex(ValueError, 'Independent physical metric differs: rmse'):
            self.build()

    def test_read_budget_failure_precedes_body_or_label_read(self):
        def refuse(size):
            raise RuntimeError('synthetic read budget exhausted')
        with patch.object(post, '_read', side_effect=AssertionError('Body opened')):
            with self.assertRaisesRegex(RuntimeError, 'read budget exhausted'):
                self.build(read_budget=refuse)

    def test_check_callback_can_stop_before_source_reads(self):
        def refuse():
            raise RuntimeError('synthetic resource guard')
        with patch.object(post, '_read', side_effect=AssertionError('Body opened')):
            with self.assertRaisesRegex(RuntimeError, 'resource guard'):
                self.build(check=refuse)

    def test_independent_constant_reference_is_undefined_without_discarded_lead(self):
        truth = np.ones_like(self.truth)
        result = post.independent_panel(truth, np.ones_like(truth, bool), truth + 1, self.future)
        self.assertIsNone(result['primary_score'])
        self.assertTrue(all(x['nse'] is None and not x['primary_eligible'] for x in result['leads']))
        self.assertTrue(all(x['physical_mse'] == 1 for x in result['leads']))


if __name__ == '__main__':
    unittest.main()
