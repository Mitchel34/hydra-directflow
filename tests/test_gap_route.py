import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_gap_route import route_masks
from hydra_directflow.gap_policies import classify_history_window, POLICY_IDS
from hydra_directflow.local_data import utc_epoch


class RouteTests(unittest.TestCase):
    def test_closed_form_equals_endpoint_classifier_across_gap_patterns(self):
        masks=np.ones((12,168),bool)
        patterns=[[],[0],[167],[40,41],[40,100],[40,167],[20,21,22],
                  list(range(168)),[0,1,2],[144],[143,144],[3,60,100]]
        cutoff=utc_epoch('2023-08-01T00:00:00Z');times=cutoff-np.arange(169,1,-1)*3600
        for i,indices in enumerate(patterns):masks[i,indices]=False
        before=masks.copy();actual=route_masks(masks,np.ones(12,bool))
        for i in range(12):
            c=classify_history_window(history_times=times,approved_mask=masks[i],
                source_observation_times=[int(t) if m else None for t,m in zip(times,masks[i])],cutoff_epoch=cutoff)
            self.assertEqual(actual['combined_2'][i],c['candidates'][POLICY_IDS['combined_2']]['history_admissible'])
            self.assertEqual(actual['natural_l24'][i],c['natural_l24']['history_admissible'])
        np.testing.assert_array_equal(masks,before)

    def test_union_recovers_complementary_histories_without_dropping_nonq_gate(self):
        masks=np.ones((3,168),bool);masks[0,167]=False;masks[1,30:50]=False
        a=route_masks(masks,np.array([True,True,False]))
        self.assertEqual(a['combined_2'].tolist(),[True,False,False])
        self.assertEqual(a['natural_l24'].tolist(),[False,True,False])
        self.assertEqual(a['short2_or_l24'].tolist(),[True,True,False])

    def test_no_target_argument_or_shape_coercion(self):
        masks=np.ones((2,168),bool)
        for q,n in ((masks.astype(int),np.ones(2,bool)),(masks,np.ones(2,int)),(masks[:,:24],np.ones(2,bool))):
            with self.assertRaises(ValueError):route_masks(q,n)
        with self.assertRaises(TypeError):route_masks(masks,np.ones(2,bool),targets=np.ones((2,18)))


if __name__=='__main__':unittest.main()
