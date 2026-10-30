import unittest
import numpy as np
from p1_followup.analyze import compare
class AnalysisChecks(unittest.TestCase):
    def test_paired_constant_ratio(self):
        a=np.arange(1.,21.)
        r=compare(a,a/2,np.ones(20),np.random.default_rng(1))
        self.assertAlmostEqual(r['gain'],10*np.log10(2))
        self.assertAlmostEqual(r['lo'],r['gain']);self.assertAlmostEqual(r['hi'],r['gain'])
    def test_self_comparison_exact_zero(self):
        a=np.array([.1,100,1,4.])
        r=compare(a,a,a,np.random.default_rng(2))
        for key in ('gain','lo','hi','nmse'):self.assertEqual(r[key],0.)
    def test_clustered_masks_keep_world_count(self):
        errors=np.arange(1.,13.).reshape(4,3)
        r=compare(errors.mean(1),errors.mean(1),np.ones(4),np.random.default_rng(3))
        self.assertEqual(r['n'],4);self.assertEqual(r['gain'],0.)
if __name__=='__main__':unittest.main()
