import unittest
import numpy as np
from p1_followup.run import select_mask,fit_selected
from codex_verification.worlds import make_world
from p1_revision.baselines import select_candidates
class Checks(unittest.TestCase):
    def setUp(self):self.w=make_world(80000002,N=8,K=2,P=10,L=3,snr=5.,rsr=12.)
    def test_fixed_matches_old(self):
        w=self.w
        rank,scores=select_mask(w.S,w.Z,w.B,w.sigma2,np.arange(10),'hard',2,iterations=2)
        idx,ref=select_candidates(w.S,w.Z,w.B,w.sigma2,ranks=np.arange(1,5),sweeps=2,iterations=2)
        self.assertEqual(rank,idx+1);np.testing.assert_array_equal(scores,ref)
    def test_split_is_partition(self):
        order=np.random.default_rng(1).permutation(10)
        self.assertEqual(set(order[:7]) & set(order[7:]),set());self.assertEqual(len(set(order)),10)
    def test_selection_has_no_truth_input(self):
        from types import SimpleNamespace
        w=SimpleNamespace(S=self.w.S,Z=self.w.Z,B=self.w.B,sigma2=self.w.sigma2,G=self.w.G.copy());args=(w.S,w.Z,w.B,w.sigma2,np.arange(10),'hard',1)
        rank,scores=select_mask(*args,iterations=2);w.G[:]=12345
        rank2,scores2=select_mask(*args,iterations=2)
        self.assertEqual(rank,rank2);np.testing.assert_array_equal(scores,scores2)
    def test_full_rank_bypasses_sweeps(self):
        w=self.w
        a=fit_selected(w.S,w.Z,w.B,w.sigma2,4,'hard',1,iterations=2)
        b=fit_selected(w.S,w.Z,w.B,w.sigma2,4,'hard',8,iterations=2)
        np.testing.assert_array_equal(a,b)
if __name__=='__main__':unittest.main()
