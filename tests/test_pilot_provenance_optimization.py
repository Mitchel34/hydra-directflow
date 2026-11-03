"""NumPy-only canonical byte equivalence against the committed legacy helper.

No trainer/model construction, optimizer, scaler fitting, source-data reads or
network. The reference function below is copied verbatim from the local Git
object at b35e0f7e459eefeb1a8cf4ab6dedc2d97ff90e5f, not an old Cloud checkout.
"""
import ast
from collections import OrderedDict, UserDict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow import pilot_training as pilot
from hydra_directflow.pilot_inputs import prepare_h4_inputs, make_fill_policy
from hydra_directflow.reference import digest
from hydra_directflow.training import TrainOnlyTransform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from gap_pilot_fixture import inference_fixture

REFERENCE_COMMIT = 'b35e0f7e459eefeb1a8cf4ab6dedc2d97ff90e5f'
REFERENCE_BLOB_SHA256 = '9095fa00407eebe6f196c26ab7be9f713fbf4ccbb419efc35a4505e80cb1b835'
REFERENCE_FUNCTION_SHA256 = 'e656a314858cab08863999161e4101cd532830cccc0e9e61565a99e4eb6994a0'
REFERENCE_SOURCE = '''def _canonical(value):
    def plain(item):
        if isinstance(item,Mapping): return {key:plain(child) for key,child in item.items()}
        if isinstance(item,np.ndarray): return plain(item.tolist())
        if isinstance(item,np.generic): return plain(item.item())
        if isinstance(item,(tuple,list)): return [plain(child) for child in item]
        return item
    return json.loads(json.dumps(plain(value), sort_keys=True, allow_nan=False))'''
namespace = {'Mapping': Mapping, 'np': np, 'json': json}
exec(compile(REFERENCE_SOURCE, '<pinned-committed-canonical-oracle>', 'exec'), namespace)
legacy_canonical = namespace['_canonical']


def legacy_bytes(value):
    return json.dumps(legacy_canonical(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def prepared_provenance(scenario='clean', *, rows=2, failed=False):
    arrays, provenance, source, _ = inference_fixture()
    sliced = {key: value[:rows].copy() if value.ndim else value.copy() for key, value in arrays.items()}
    native = {key: value[:rows] for key, value in provenance.items()}
    if failed:
        sliced['history'][0, 10:13] = np.nan
        sliced['history_mask'][0, 10:13] = False
    parameters = {name: [dict(mean=0., scale=1., constant=False, cells=100) for _ in range(count)]
                  for name, count in (('history', 2), ('weather', 4), ('state', 2), ('target', 1))}
    saved = TrainOnlyTransform(parameters, 'a' * 64, 'b' * 64)
    return prepare_h4_inputs(sliced, history_provenance=native, saved_h2_transform=saved,
                             inference_source_identity=source, fill_policy=make_fill_policy(), scenario_id=scenario)


class CanonicalParityTests(unittest.TestCase):
    def assert_parity(self, value):
        expected = legacy_bytes(value)
        self.assertEqual(pilot._canonical(value), legacy_canonical(value))
        self.assertEqual(pilot._canonical_bytes(value), expected)
        self.assertEqual(pilot._canonical_digest(value), hashlib.sha256(expected).hexdigest())

    def test_reference_is_pinned_to_the_committed_helper(self):
        self.assertEqual(hashlib.sha256(REFERENCE_SOURCE.encode()).hexdigest(), REFERENCE_FUNCTION_SHA256)
        tree = ast.parse(REFERENCE_SOURCE)
        self.assertEqual([node.name for node in tree.body], ['_canonical'])
        self.assertEqual(len(REFERENCE_COMMIT), 40)

    def test_native_and_numpy_scalar_leaf_types_preserve_exact_bytes(self):
        self.assert_parity({'integer': 2**64-1, 'floating': [-0., 0., 1e-7, 1e20], 'null': None,
            'bool': [True, False, np.bool_(True)], 'numpy': [np.int64(-12), np.uint64(2**64-1),
            np.float32(.1), np.float64(1e-200), np.str_('qualifier')],
            'unicode': ['é', '汉字', '\ud83d\ude00', 'line\nquote"slash\\', '\x00']})

    def test_nested_mappings_tuples_arrays_and_aliases_preserve_flags_and_order(self):
        flags = ('R', 'A', 'A', 'R')
        same = {'ids': ['second', 'first'], 'flags': flags}
        array = np.arange(24, dtype='>i4').reshape(4, 6)[:, ::2]
        array.setflags(write=False)
        value = MappingProxyType({'views': (same, same), 'ordered': OrderedDict([('z', 3), ('a', 1)]),
            'user_mapping': UserDict({'m': np.array([True, False])}), 'noncontiguous': array,
            'empty': [np.empty((0, 3), dtype=np.float64), np.array([], dtype=np.int64)],
            'zero_dim': np.array(np.int64(7))})
        self.assert_parity(value)
        self.assertFalse(array.flags.writeable)
        self.assertEqual(pilot._canonical(value)['views'][0]['flags'], ['R', 'A', 'A', 'R'])

    def test_legacy_numeric_key_stringification_and_lexical_resort_are_retained(self):
        cases = ({2: 'two', 10: 'ten', 1: 'one'}, {'nested': {2: 'two', 10: 'ten'}},
                 {None: np.int64(1)}, {False: 'false', 2: 'two'}, {2.5: 'two', -0.: 'zero'},
                 MappingProxyType({2: {'a': [1, 2]}, 10: 'ten'}))
        for value in cases:
            with self.subTest(value=value):
                self.assertTrue(pilot._needs_key_normalization(value))
                self.assert_parity(value)
        self.assertEqual(pilot._canonical_bytes(cases[0]), b'{"1":"one","10":"ten","2":"two"}')

    def test_object_arrays_retain_nested_mapping_key_behavior(self):
        values = np.empty(2, dtype=object)
        values[0] = {2: 'two', 10: 'ten'}
        values[1] = MappingProxyType({'flags': ('A', 'R'), 'count': np.int32(3)})
        self.assertTrue(pilot._needs_key_normalization(values))
        self.assert_parity({'values': values})

    def test_array_subclass_tolist_cannot_bypass_legacy_key_coercion(self):
        class MetadataArray(np.ndarray):
            def tolist(self): return {2: 'two', 10: 'ten'}
        value = np.array([1], dtype=np.int64).view(MetadataArray)
        self.assertTrue(pilot._needs_key_normalization(value))
        self.assert_parity(value)

    def test_string_keyed_fast_digest_uses_one_encoder_without_json_decode(self):
        value = prepared_provenance('mixed_middle_carry')['provenance']
        expected = hashlib.sha256(legacy_bytes(value)).hexdigest()
        self.assertFalse(pilot._needs_key_normalization(value))
        with patch.object(pilot.json, 'loads', side_effect=AssertionError('Fast path decoded JSON')), \
             patch.object(pilot.json, 'dumps', wraps=json.dumps) as encode:
            self.assertEqual(pilot._canonical_digest(value), expected)
            self.assertEqual(encode.call_count, 1)

    def test_actual_prepared_clean_one_two_and_mixed_gap_metadata_are_byte_identical(self):
        for scenario in ('clean', 'interior_1_early', 'interior_2_recent', 'trailing_1_carry', 'trailing_2_carry', 'mixed_middle_carry'):
            prepared = prepared_provenance(scenario)
            with self.subTest(scenario=scenario):
                self.assertTrue(prepared['case_admissible'].all())
                self.assert_parity(prepared['provenance'])
                self.assert_parity(prepared['receipt'])

    def test_failed_fill_and_ridge_shaped_provenance_remain_exact(self):
        failed = prepared_provenance(failed=True)
        self.assertFalse(failed['case_admissible'][0])
        self.assert_parity(failed['provenance'])
        # Representative serialized ridge support metadata only: no ridge fit.
        value = legacy_canonical(prepared_provenance('trailing_2_carry')['provenance'])
        for item in value['fill_receipts']:
            for slot in item['slot_provenance'][-2:]:
                slot.update(fill_method='RIDGE_DIRECT_ORIGINAL_SIX_LAGS', ridge_model_identity='e'*64,
                            support_source_ids=['synthetic:'+str(i) for i in range(6)],
                            support_source_flags=[['R','A','R']] * 6,
                            support_observation_epochs=[1500000000+i*3600 for i in range(6)])
        self.assert_parity(value)

    def test_batch_order_and_mutations_are_never_hidden_by_a_digest_cache(self):
        value = legacy_canonical(prepared_provenance('interior_1_middle')['provenance'])
        before = pilot._canonical_digest(value)
        changed = deepcopy(value); changed['quality_flags'][0][0] = ['A', 'R', 'A']
        self.assertNotEqual(pilot._canonical_digest(changed), before)
        self.assert_parity(changed)
        changed = deepcopy(value); changed['source_ids'][0][0] += '/different'
        self.assertNotEqual(pilot._canonical_digest(changed), before)
        changed = deepcopy(value)
        for key in changed: changed[key] = changed[key][::-1]
        self.assertNotEqual(pilot._canonical_digest(changed), before)
        self.assert_parity(changed)
        self.assertEqual(pilot._canonical_digest(value), before)

    def test_plain_normalization_still_returns_independent_mutable_json_data(self):
        original = MappingProxyType({'flags': ('R', 'A', 'R'), 'array': np.array([1, 2])})
        normalized = pilot._canonical(original)
        normalized['flags'][0] = 'changed'; normalized['array'][0] = 99
        self.assertEqual(original['flags'], ('R', 'A', 'R'))
        np.testing.assert_array_equal(original['array'], [1, 2])

    def test_invalid_nonfinite_and_unsupported_values_still_reject(self):
        invalid = (float('nan'), {'v': np.float32('inf')}, np.array([1., np.nan]),
                   {'v': {1, 2}}, {'v': b'bytes'}, {'v': complex(1, 2)},
                   {1: 'integer', '2': 'string'}, {'v': np.datetime64('2021-01-01', 'D')})
        for value in invalid:
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, ValueError, RecursionError)):
                    legacy_bytes(value)
                with self.assertRaises((TypeError, ValueError, RecursionError)):
                    pilot._canonical_bytes(value)
        cyclic = []; cyclic.append(cyclic)
        with self.assertRaises((TypeError, ValueError, RecursionError)): legacy_bytes(cyclic)
        with self.assertRaises((TypeError, ValueError, RecursionError)): pilot._canonical_bytes(cyclic)

    def test_original_clock_qualifier_and_source_id_gates_still_reject(self):
        arrays, provenance, _, _ = inference_fixture()
        view = {**arrays, 'target_time': arrays['forecast_time']}
        for mutation in ('clock', 'qualifier', 'source_id'):
            changed = deepcopy(provenance)
            if mutation == 'clock': changed['observation_times'][0][0] += 3600
            elif mutation == 'qualifier': changed['quality_flags'][0][0] = ['P']
            else: changed['source_ids'][0][0] = None
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                pilot._provenance(view, changed)

    def test_hash_call_sites_use_the_equivalent_digest_without_changing_receipt_fields(self):
        source = Path(pilot.__file__).read_text()
        self.assertNotIn('digest(_canonical(', source)
        expected = hashlib.sha256(legacy_bytes({'weights': np.array([1., 2.])})).hexdigest()
        model = type('SyntheticMetadataOnly', (), {'to_dict': lambda self: {'weights': np.array([1., 2.])}})()
        self.assertEqual(pilot._ridge_state_sha256(model), expected)
        self.assertIsNone(pilot._ridge_state_sha256(None))



# Snapshot checks stay NumPy-only: no PilotTrainer/Predictor constructor runs.
from hydra_directflow.pilot_provenance import (admit_history_provenance, validate_native_provenance,
    ValidatedHistoryProvenance, ValidatedProvenanceBatch)
from hydra_directflow.pilot_inputs import make_augmentation_recipe
from gap_pilot_fixture import inputs as synthetic_inputs, epoch as synthetic_epoch


def source_fixture(rows=4):
    arrays, native, source, cohort = synthetic_inputs(synthetic_epoch('2019-01-21T00:00:00Z') + np.arange(rows)*3600)
    arrays['target_time'] = arrays.pop('forecast_time')
    parameters = {name:[dict(mean=0.,scale=1.,constant=False,cells=100) for _ in range(count)]
                  for name,count in (('history',2),('weather',4),('state',2),('target',1))}
    saved = TrainOnlyTransform(parameters,'a'*64,'b'*64)
    return arrays,native,source,cohort,saved


def subset(arrays, rows):
    return {key: value[list(rows)].copy() if value.ndim else value.copy() for key,value in arrays.items()}


class PoisonValues(dict):
    def __getitem__(self,key):
        if key in ('history','weather','state','target','target_mask'):
            raise AssertionError('Snapshot/cache read Q, target or model feature values')
        return super().__getitem__(key)


class OwnedProvenanceTests(unittest.TestCase):
    def assert_preparation_equal(self,left,right):
        self.assertEqual(left['receipt'],right['receipt'])
        self.assertEqual(pilot._canonical_bytes(left['provenance']),legacy_bytes(right['provenance']))
        np.testing.assert_array_equal(left['case_admissible'],right['case_admissible'])
        for field in ('raw_inputs','transformed_inputs'):
            self.assertEqual(set(left[field]),set(right[field]))
            for name in left[field]:np.testing.assert_array_equal(left[field][name],right[field][name])

    def test_snapshot_owned_immutability_and_exact_legacy_source_digest(self):
        arrays,native,source,cohort,_=source_fixture()
        expected=hashlib.sha256(legacy_bytes(native)).hexdigest()
        owned=admit_history_provenance(PoisonValues(arrays),native,source_identity=source,cohort_digest=cohort)
        self.assertEqual(owned.provenance_sha256,expected)
        self.assertEqual(pilot._canonical_digest(owned),expected)
        old_id=owned['source_ids'][0][0];old_clock=int(owned.origins[0])
        native['source_ids'][0][0]='changed';native['quality_flags'][0][0]=['P']
        arrays['origin_time'][0]+=3600;arrays['history_mask'][0,0]=False
        self.assertEqual(owned['source_ids'][0][0],old_id);self.assertEqual(owned['quality_flags'][0][0],('A',))
        self.assertEqual(int(owned.origins[0]),old_clock);self.assertTrue(owned.approved_mask[0,0])
        for value in (owned.origins,owned.approved_mask):
            with self.assertRaises(ValueError):value.setflags(write=True)
        with self.assertRaises(TypeError):owned['source_ids'][0][0]='changed'
        with self.assertRaises(AttributeError):owned.source_identity='f'*64

    def test_interning_preserves_different_source_revisions_and_ordered_duplicate_flags(self):
        arrays,native,source,cohort,_=source_fixture()
        native['quality_flags'][0][1]=['R','A','A'];native['quality_flags'][1][0]=['A','R']
        native['source_ids'][1][0]+='/other-revision'
        owned=admit_history_provenance(arrays,native,source_identity=source,cohort_digest=cohort)
        self.assertEqual(owned['observation_times'][0][1],owned['observation_times'][1][0])
        self.assertNotEqual(owned['source_ids'][0][1],owned['source_ids'][1][0])
        self.assertEqual(owned['quality_flags'][0][1],('R','A','A'));self.assertEqual(owned['quality_flags'][1][0],('A','R'))
        self.assertIs(owned['source_ids'][0][2],owned['source_ids'][1][1])
        self.assertEqual(owned.provenance_sha256,hashlib.sha256(legacy_bytes(native)).hexdigest())

    def test_snapshot_and_raw_adapters_match_all_arrays_payloads_and_receipt_hashes(self):
        arrays,native,source,cohort,saved=source_fixture()
        owned=admit_history_provenance(arrays,native,source_identity=source,cohort_digest=cohort)
        for rows in ((0,1,2,3),(3,0,2)):
            data=subset(arrays,rows);raw={key:[native[key][i] for i in rows] for key in native};view=owned.select(rows)
            for scenario in ('clean','interior_1_early','interior_2_recent','trailing_1_carry','trailing_2_carry','mixed_middle_carry'):
                kwargs=dict(saved_h2_transform=saved,inference_source_identity=source,fill_policy=make_fill_policy(),
                            scenario_id=scenario,epoch=3,mask_seed=17)
                with self.subTest(rows=rows,scenario=scenario):
                    baseline=prepare_h4_inputs(data,history_provenance=raw,**kwargs)
                    result=prepare_h4_inputs(data,history_provenance=view,**kwargs)
                    self.assert_preparation_equal(result,baseline)
                    np.testing.assert_array_equal(result['raw_inputs']['case_id'],data['case_id'])

    def test_failed_fills_and_dynamic_augmentation_remain_equal_without_cached_fill_outputs(self):
        arrays,native,source,cohort,saved=source_fixture()
        arrays['history'][0,10:13]=np.nan;arrays['history_mask'][0,10:13]=False
        owned=admit_history_provenance(arrays,native,source_identity=source,cohort_digest=cohort)
        kwargs=dict(saved_h2_transform=saved,inference_source_identity=source,fill_policy=make_fill_policy())
        baseline=prepare_h4_inputs(arrays,history_provenance=native,**kwargs)
        result=prepare_h4_inputs(arrays,history_provenance=owned.select((0,1,2,3)),**kwargs)
        self.assertFalse(result['case_admissible'][0]);self.assert_preparation_equal(result,baseline)
        arrays,native,source,cohort,saved=source_fixture()
        owned=admit_history_provenance(arrays,native,source_identity=source,cohort_digest=cohort)
        recipe=make_augmentation_recipe(mixture={'interior_2_recent':.5,'mixed_middle_carry':.5},
            gradient_case_ids_sha256=saved.training_case_ids_sha256,source_identity=source,evidence_sha256='e'*64)
        masks=[]
        for epoch in (0,1,7):
            kwargs=dict(saved_h2_transform=saved,inference_source_identity=source,fill_policy=make_fill_policy(),
                        scenario_id='augmentation',epoch=epoch,mask_seed=17,augmentation_recipe=recipe)
            baseline=prepare_h4_inputs(arrays,history_provenance=native,**kwargs)
            result=prepare_h4_inputs(arrays,history_provenance=owned.select((0,1,2,3)),**kwargs)
            self.assert_preparation_equal(result,baseline);masks.append(result['receipt']['artificial_mask_sha256'])
        self.assertGreater(len(set(masks)),1)

    def test_view_rejects_actual_mask_clock_gauge_order_and_binding_changes_before_Q_read(self):
        arrays,native,source,cohort,saved=source_fixture()
        owned=admit_history_provenance(arrays,native,source_identity=source,cohort_digest=cohort);view=owned.select((0,1))
        for mutation in ('mask','clock','future','source','cohort','gauge','order'):
            data=subset(arrays,(0,1));supplied_source=source
            if mutation=='mask':data['history_mask'][0,0]=False
            elif mutation=='clock':data['history_time'][0,0]+=3600
            elif mutation=='future':data['target_time'][0,-1]+=3600
            elif mutation=='source':supplied_source='f'*64
            elif mutation=='cohort':data['cohort_admission_digest']=np.array('f'*64)
            elif mutation=='gauge':
                data['site_id'][:]=1;data['case_id']=np.array([str(case).replace('03161000:','03164000:') for case in data['case_id']])
            else:data=subset(arrays,(1,0))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                prepare_h4_inputs(PoisonValues(data),history_provenance=view,saved_h2_transform=saved,
                    inference_source_identity=supplied_source,fill_policy=make_fill_policy())

    def test_raw_and_snapshot_qualifier_id_timestamp_rejections_match(self):
        arrays,native,source,cohort,_=source_fixture()
        for mutation in ('flags','id','clock'):
            changed=deepcopy(native)
            if mutation=='flags':changed['quality_flags'][0][0]=['e']
            elif mutation=='id':changed['source_ids'][0][0]=None
            else:changed['observation_times'][0][0]+=3600
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):validate_native_provenance(changed,arrays['history_mask'][...,0],arrays['history_time'])
                with self.assertRaises(ValueError):admit_history_provenance(PoisonValues(arrays),changed,source_identity=source,cohort_digest=cohort)

    def test_inference_cache_has_one_entry_and_rechecks_actual_content_not_claimed_hash_or_object_id(self):
        arrays,native,source,cohort,_=source_fixture()
        owner=pilot.PilotBestPredictor.__new__(pilot.PilotBestPredictor)
        with patch.object(pilot,'admit_history_provenance',wraps=admit_history_provenance) as admit:
            first=pilot._inference_provenance_view(owner,PoisonValues(arrays),native,source,cohort)
            second=pilot._inference_provenance_view(owner,PoisonValues(arrays),deepcopy(native),source,cohort)
            self.assertIs(first.snapshot,second.snapshot);self.assertEqual(admit.call_count,1)
            self.assertEqual(owner.provenance_cache_info(),dict(hits=1,misses=1,entries=1,admissions=1,explicit_views=0))
            changed=deepcopy(native);changed['source_ids'][0][0]+='/different'
            third=pilot._inference_provenance_view(owner,PoisonValues(arrays),changed,source,cohort)
            self.assertIsNot(first.snapshot,third.snapshot);self.assertEqual(admit.call_count,2)
            self.assertNotEqual(first.snapshot.provenance_sha256,third.snapshot.provenance_sha256)
            invalid=deepcopy(changed);invalid['quality_flags'][0][0]=['P']
            with self.assertRaises(ValueError):pilot._inference_provenance_view(owner,PoisonValues(arrays),invalid,source,cohort)
            self.assertEqual(owner.provenance_cache_info()['misses'],3);self.assertEqual(owner.provenance_cache_info()['admissions'],2)
            info=owner.provenance_cache_info();info['hits']=999
            self.assertEqual(owner.provenance_cache_info()['hits'],1)

    def test_actual_different_clock_batches_miss_even_when_caller_reuses_asserted_source_identity(self):
        owner=pilot.PilotBestPredictor.__new__(pilot.PilotBestPredictor)
        cohorts=[]
        for hours in range(3):
            arrays,native,_,_=synthetic_inputs(synthetic_epoch('2019-01-21T00:00:00Z')+(hours+np.arange(2))*3600)
            arrays['target_time']=arrays.pop('forecast_time')
            arrays['cohort_admission_digest']=np.array('d'*64)
            view=pilot._inference_provenance_view(owner,PoisonValues(arrays),native,'c'*64,'d'*64)
            cohorts.append(view.snapshot.provenance_sha256)
        self.assertEqual(len(set(cohorts)),3)
        self.assertEqual(owner.provenance_cache_info(),dict(hits=0,misses=3,entries=1,admissions=3,explicit_views=0))

    def test_storage_allowance_rejects_before_expansion_and_snapshot_never_retains_fill_or_Q_data(self):
        arrays,native,source,cohort,_=source_fixture()
        with patch.object(pilot,'_canonical',side_effect=AssertionError('Expanded before memory allowance')):
            with self.assertRaisesRegex(ValueError,'storage allowance'):
                admit_history_provenance(PoisonValues(arrays),native,source_identity=source,cohort_digest=cohort,max_storage_bytes=1)
        owned=admit_history_provenance(PoisonValues(arrays),native,source_identity=source,cohort_digest=cohort)
        self.assertEqual(set(owned),{'observation_times','quality_flags','source_ids'})
        self.assertLess(owned.storage_bytes,owned.max_storage_bytes)
        self.assertFalse(any(key in owned.__dict__ for key in ('history','target','prepared','fill_receipts')))
        self.assertIs(admit_history_provenance(PoisonValues(arrays),owned,source_identity=source,cohort_digest=cohort),owned)
        for indices in ((),(0,0),(999,),tuple(range(129))):
            with self.assertRaises(ValueError):owned.select(indices)


if __name__ == '__main__': unittest.main()
