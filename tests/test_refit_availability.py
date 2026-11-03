"""Synthetic timestamp/scaler isolation checks; no model or optimizer construction."""
from dataclasses import asdict
import copy
import unittest
import numpy as np

from hydra_directflow.local_data import utc_epoch
from hydra_directflow.trainer_bundle import refit_label_mask
from hydra_directflow.training import TrainOnlyTransform
import test_trainer_bundle as bundle_fixtures
from test_training import fixture


class RefitAvailabilityTests(unittest.TestCase):
    def test_actual_last_target_inclusive_delay_and_distinct_strict_boundary(self):
        boundary = utc_epoch('2023-07-31T00:00:00Z')
        origins = boundary - np.array([21,20,19,18],np.int64)*3600
        targets = origins[:,None] + np.arange(1,19)*3600
        zero = refit_label_mask(targets,refit_at_utc='2023-07-31T00:00:00Z',simulated_label_delay_hours=0)
        two = refit_label_mask(targets,refit_at_utc='2023-07-31T00:00:00Z',simulated_label_delay_hours=2)
        self.assertEqual(zero.tolist(),[True,True,True,True])
        self.assertEqual(two.tolist(),[True,True,False,False])
        strict = (targets < boundary).all(axis=1)
        self.assertEqual((strict & zero).tolist(),[True,True,True,False])
        self.assertEqual((strict & two).tolist(),[True,True,False,False])

    def test_bundle_refit_policy_preserves_rows_and_separates_each_exclusion(self):
        helper=bundle_fixtures.TrainerBundleTests();helper.setUp()
        try:
            _, selections=helper.source()
            roles={'fit':{'start_utc':'2021-08-23T02:00:00Z','end_utc':'2021-08-24T01:00:00Z'},
                   'inner_stop':{'start_utc':'2021-08-24T01:00:00Z','end_utc':'2021-08-25T02:00:00Z'}}
            strict=helper.build(selections,role_time_bounds=roles)
            guarded=helper.build(selections,role_time_bounds=roles,
                refit_availability={'refit_at_utc':'2021-08-25T02:00:00Z','simulated_label_delay_hours':2})
            self.assertEqual(strict.manifest['case_count'],guarded.manifest['case_count'])
            self.assertEqual(strict.manifest['cohort_admission_digest'],guarded.manifest['cohort_admission_digest'])
            self.assertEqual(len(strict.indices_for('fit')),5);self.assertEqual(len(guarded.indices_for('fit')),5)
            self.assertEqual(len(strict.indices_for('inner_stop')),6);self.assertEqual(len(guarded.indices_for('inner_stop')),5)
            self.assertEqual(guarded.manifest['roles']['fit']['target_boundary_purged_cases'],1)
            self.assertEqual(guarded.manifest['roles']['fit']['refit_label_availability_purged_cases'],0)
            self.assertEqual(guarded.manifest['roles']['inner_stop']['refit_label_availability_purged_cases'],1)
            self.assertEqual(guarded.manifest['unassigned_cases'],2)
            self.assertNotEqual(strict.manifest['role_assignment_digest'],guarded.manifest['role_assignment_digest'])
            np.testing.assert_equal(strict.arrays['target'],guarded.arrays['target'])
        finally: helper.tearDown()

    def test_refit_policy_must_use_declared_cutoff_and_never_infer_a_surviving_origin(self):
        helper=bundle_fixtures.TrainerBundleTests();helper.setUp()
        try:
            _, selections=helper.source()
            with self.assertRaisesRegex(ValueError,'Refit must equal'):
                helper.build(selections,refit_availability={'refit_at_utc':'2021-08-25T02:00:00Z','simulated_label_delay_hours':2})
            roles={'fit':{'start_utc':'2021-08-23T02:00:00Z','end_utc':'2021-08-24T01:00:00Z'},
                   'inner_stop':{'start_utc':'2021-08-24T01:00:00Z','end_utc':'2021-08-25T02:00:00Z'}}
            with self.assertRaisesRegex(ValueError,'Refit must equal'):
                helper.build(selections,role_time_bounds=roles,
                    refit_availability={'refit_at_utc':'2021-08-25T03:00:00Z','simulated_label_delay_hours':2})
        finally: helper.tearDown()

    def test_scaler_uses_only_gradient_fit_cells_not_inner_outer_or_unassigned(self):
        arrays=fixture()  # Rows0/1/2/3 are explicit synthetic fit/inner/outer/unassigned roles.
        original=TrainOnlyTransform.fit(arrays,(0,))
        changed=copy.deepcopy(arrays)
        for name in ('history','weather','state','target'):
            changed[name][1:]=np.nan;changed[name+'_mask'][1:]=False
        self.assertEqual(asdict(original),asdict(TrainOnlyTransform.fit(changed,(0,))))
        changed['target'][0]+=10
        self.assertNotEqual(asdict(original),asdict(TrainOnlyTransform.fit(changed,(0,))))

    def test_malformed_or_protected_target_coordinates_and_unapproved_lags_fail(self):
        origin=utc_epoch('2023-07-30T00:00:00Z')
        targets=origin+np.arange(1,19)[None,:]*3600
        for change in ('fractional','nonhour','protected','lag'):
            values=targets.copy();lag=2
            if change=='fractional': values=values.astype(float)
            elif change=='nonhour': values[0,-1]+=1
            elif change=='protected': values+=367*86400
            else:lag=1
            with self.subTest(change=change),self.assertRaises(ValueError):
                refit_label_mask(values,refit_at_utc='2023-07-31T00:00:00Z',simulated_label_delay_hours=lag)


if __name__=='__main__':unittest.main()
