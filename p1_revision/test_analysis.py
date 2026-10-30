import unittest
import numpy as np
from p1_revision.analyze import paired_gain_change


class AnalysisTests(unittest.TestCase):
    def test_unchanged_condition_has_exact_zero_change(self):
        e=np.arange(1.,11);h=e*np.linspace(.4,1.2,10)
        got=paired_gain_change(e,h,e,h,np.random.default_rng(1))
        self.assertEqual(got,dict(gain_change_db=0.,lo=0.,hi=0.))

    def test_known_paired_loss_of_gain(self):
        e=np.arange(1.,11)
        got=paired_gain_change(e,e/2,e,e,np.random.default_rng(2))
        for v in got.values():self.assertAlmostEqual(v,-10*np.log10(2))

    def test_mismatched_pairing_shape_rejected(self):
        with self.assertRaises(ValueError):
            paired_gain_change(np.ones(2),np.ones(3),np.ones(2),np.ones(2),np.random.default_rng(0))


if __name__=='__main__':unittest.main()
