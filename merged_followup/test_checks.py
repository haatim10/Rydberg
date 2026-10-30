import unittest
import numpy as np
import torch
from sahure_expansion.data import world
from codex_verification.solver import fit
from paper2_optshrink_loop.estimator import estimate as original
from sahure.data import Observations, split_observations
from sahure.math import nll
from .estimator import trajectories, refit, context_candidates, validation_scores, select_index, SCHEDULES
from .models import build


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.w, _ = world(0, 202610085000)
        cls.param = dict(rank=4, strength=.5)

    def test_controls_and_replay(self):
        w = self.w
        gs = trajectories(w.Z, w.S, w.B, w.sigma2, self.param)
        for i, (r, t) in enumerate(SCHEDULES):
            np.testing.assert_allclose(gs[i], refit(w.Z,w.S,w.B,w.sigma2,self.param,(r,t)), atol=1e-11)
            if r == 0:
                np.testing.assert_allclose(gs[i], fit(w.S,w.Z,w.B,w.sigma2,iterations=t)[0],atol=1e-11)
            if r == t:
                np.testing.assert_allclose(gs[i], original(w.Z,w.S,w.B,w.sigma2,
                    dict(**self.param,iterations=t,placement='loop')),atol=1e-11)

    def test_validation_isolation(self):
        w=self.w
        a=context_candidates(w.Z,w.S,w.B,w.sigma2,self.param)
        changed=w.Z.copy(); changed[:,21:] *= 1.1
        b=context_candidates(changed,w.S,w.B,w.sigma2,self.param)
        np.testing.assert_array_equal(a,b)
        scores=validation_scores(a,w.Z,w.S,w.B,w.sigma2)
        self.assertEqual(select_index(scores,'release'),int(scores.argmin()))
        self.assertEqual(SCHEDULES[select_index(scores,'em')][0],0)

    def test_learned_finite_and_classical_start(self):
        w=self.w
        obs=Observations(torch.tensor(w.Z[None]),torch.tensor(w.S[None]),
                         torch.tensor(w.B[None]),torch.tensor([w.sigma2],dtype=torch.float64))
        a,b=split_observations(obs,77)
        for name in ('compact','em_transformer'):
            model=build(name,dtype=torch.float64)
            g=model(*a.args())
            if name=='em_transformer':
                expected=fit(a.s[0].numpy(),a.z[0].numpy(),a.b[0].numpy(),w.sigma2,iterations=10)[0]
                np.testing.assert_allclose(g.detach().numpy()[0],expected,rtol=1e-10,atol=1e-10)
            loss=nll(b.z,g@b.s+b.b,b.sigma2).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
