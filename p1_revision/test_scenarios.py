import unittest
import numpy as np
from codex_verification import solver
from codex_verification.worlds import make_world
from p1_revision.scenarios import configurations, position_world, select_fraction, estimate_pair


class ScenariosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w=make_world(79_000_001,N=8,K=3,P=10,L=3)

    def test_fraction_matches_verified_selector(self):
        w=self.w
        r,s=select_fraction(w.S,w.Z,w.B,w.sigma2,iterations=3)
        rr,ss=solver.select(w.S,w.Z,w.B,w.sigma2,iterations=3)
        self.assertEqual(r,rr)
        np.testing.assert_array_equal(s,ss)

    def test_zero_mismatch_preserves_exact_world(self):
        self.assertIs(position_world(self.w,0,100), self.w)

    def test_position_formula_and_pairing(self):
        w=self.w
        a=position_world(w,.01,100)
        b=position_world(w,.02,100)
        np.testing.assert_array_equal(2*a.position_offsets,b.position_offsets)
        np.testing.assert_array_equal(a.W,w.W)
        np.testing.assert_array_equal(a.S,w.S)
        np.testing.assert_array_equal(a.B,w.B)  # known broadside reference
        np.testing.assert_allclose(a.Z,np.abs(a.G@a.S+a.B+a.W))
        self.assertEqual(a.sigma2,w.sigma2)
        self.assertGreater(np.linalg.norm(a.G-w.G),0)
        n=np.arange(len(w.G))/2
        g=np.column_stack([w.cfg.c*np.exp(-2j*np.pi*n[:,None]*np.sin(th))@al
                           for th,al in zip(w.theta,w.alpha)])
        np.testing.assert_allclose(g,w.G,atol=1e-13,rtol=1e-13)

    def test_scientific_design(self):
        cs=configurations()
        self.assertEqual(len(cs),30)
        self.assertEqual(sum(c['trials'] for c in cs),12000)
        self.assertEqual(len({c['tag'] for c in cs}),30)
        self.assertEqual(len({c['seed_base'] for c in cs}),12)
        self.assertTrue(all(c['P']-max(1,round(c.get('val_fraction',.3)*c['P']))>=2*c['K'] for c in cs))

    def test_full_refit_uses_all_pilots(self):
        c=dict(iterations=3,select_iterations=2,sweeps=4,val_fraction=.4)
        est,diag=estimate_pair(self.w,c)
        expected=solver.fit(self.w.S,self.w.Z,self.w.B,self.w.sigma2,
            iterations=3,ranks=[diag['rank']],sweeps=4)[0]
        np.testing.assert_array_equal(est['hs'],expected)

    def test_heldout_does_not_enter_candidate_fit(self):
        w=self.w
        # Rank scores may change, but candidates only see the first 7 columns.
        a=solver.fit(w.S[:,:7],w.Z[:,:7],w.B[:,:7],w.sigma2,iterations=2)
        z=w.Z.copy();z[:,7:]*=20
        b=solver.fit(w.S[:,:7],z[:,:7],w.B[:,:7],w.sigma2,iterations=2)
        np.testing.assert_array_equal(a,b)


if __name__=='__main__':
    unittest.main()
