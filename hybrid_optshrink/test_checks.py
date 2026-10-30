import unittest
import copy
import numpy as np
import torch
from sahure_expansion.data import world
from sahure.data import Observations, split_observations
from sahure.math import nll
from codex_verification.solver import fit
from paper2_optshrink_loop.estimator import weights, estimate
from .model import opt_weights, opt_filter, OptController, build


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        w, _ = world(0, 202610110001)
        cls.w = w
        cls.obs = Observations(torch.tensor(w.Z[None]), torch.tensor(w.S[None]),
            torch.tensor(w.B[None]), torch.tensor([w.sigma2], dtype=torch.float64))

    def test_weights_match_numpy(self):
        rng = np.random.default_rng(71)
        s = np.sort(rng.uniform(.01, 8, (4, 16)), axis=-1)[:, ::-1].copy()
        r = np.array([1, 4, 9, 15])
        actual = opt_weights(torch.tensor(s), 17, 16, torch.tensor(r)).numpy()
        np.testing.assert_allclose(actual, weights(s, 17, 16, r), rtol=1e-11, atol=1e-12)

    def test_fixed_and_initial_controller_match_classical(self):
        w = self.w
        for strength in (0., .5, 1.):
            tables = [dict(rank=4, strength=strength)]*7
            expected = estimate(w.Z, w.S, w.B, w.sigma2, dict(rank=4, strength=strength, iterations=10))
            for mode in ('fixed', 'adaptive', 'global'):
                model = OptController(mode, tables=tables).double()
                actual = model(*self.obs.args()).detach().numpy()[0]
                np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)
            if strength == 0:
                np.testing.assert_allclose(actual, fit(w.S,w.Z,w.B,w.sigma2,iterations=10)[0], rtol=2e-9, atol=2e-10)

    def test_filter_gradcheck_and_scale(self):
        gen = torch.Generator().manual_seed(54)
        h = torch.randn(5, 4, dtype=torch.complex128, generator=gen, requires_grad=True)
        fn = lambda x: opt_filter(x, torch.tensor(2))[0]
        self.assertTrue(torch.autograd.gradcheck(fn, (h,), eps=1e-6, atol=2e-5, rtol=2e-4))
        torch.testing.assert_close(fn(3*h), 3*fn(h), rtol=1e-10, atol=1e-11)
        for x in (torch.zeros(5,4,dtype=torch.complex128), torch.eye(5,4,dtype=torch.complex128)):
            self.assertTrue(torch.isfinite(fn(x)).all())

    def test_unrolled_parameter_directional_derivative(self):
        a, b = split_observations(self.obs, 193)
        model = OptController('adaptive', stages=3).double()
        with torch.no_grad(): model.net[-1].bias.fill_(-.1)
        def loss():
            g = model(*a.args())
            return nll(b.z, g@b.s+b.b, b.sigma2).mean()
        value = loss(); value.backward()
        p = model.net[-1].bias
        analytic = float(p.grad[0]); original = float(p.detach()[0]); eps=1e-5
        with torch.no_grad(): p.fill_(original+eps)
        plus = float(loss().detach())
        with torch.no_grad(): p.fill_(original-eps)
        minus = float(loss().detach())
        with torch.no_grad(): p.fill_(original)
        np.testing.assert_allclose(analytic, (plus-minus)/(2*eps), atol=2e-6, rtol=2e-4)

    def test_training_gradients_and_split(self):
        a,b = split_observations(self.obs, 419)
        self.assertEqual(a.s.shape[-1],21);self.assertEqual(b.s.shape[-1],9)
        aa,bb = split_observations(self.obs,419)
        torch.testing.assert_close(a.s,aa.s,rtol=0,atol=0)
        torch.testing.assert_close(b.s,bb.s,rtol=0,atol=0)
        for name in ('adaptive','global','smooth'):
            model=build(name);g=model(*a.args())
            loss=nll(b.z,g@b.s+b.b,b.sigma2).mean();loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        # Withheld magnitudes cannot enter the context-only estimate.
        model=build('adaptive')
        g=model(*a.args()); b.z.mul_(2)
        torch.testing.assert_close(model(*a.args()),g,rtol=0,atol=0)

    def test_optimizer_resume(self):
        a,b=split_observations(self.obs,111)
        model=OptController('adaptive',stages=2).double()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        def step(m,o):
            o.zero_grad(set_to_none=True)
            nll(b.z,m(*a.args())@b.s+b.b,b.sigma2).mean().backward();o.step()
        step(model,optimizer)
        state=copy.deepcopy(model.state_dict());optim=copy.deepcopy(optimizer.state_dict())
        step(model,optimizer)
        resumed=OptController('adaptive',stages=2).double();resumed.load_state_dict(state)
        ro=torch.optim.Adam(resumed.parameters(),lr=.001);ro.load_state_dict(optim)
        step(resumed,ro)
        for name,p in model.state_dict().items():
            torch.testing.assert_close(p,resumed.state_dict()[name],rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
