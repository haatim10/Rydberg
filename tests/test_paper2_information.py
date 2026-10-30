import sys
import unittest
import tempfile
import fcntl
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import iv
from scipy.stats import rice

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper2_information import core as c
from paper2_information.data import world, noise_cell


class Numerics(unittest.TestCase):
    def test_likelihood_and_derivative(self):
        z = np.array([.01, .2, 1., 5.])
        a = np.array([.02, .7, 1.2, 4.])
        v = .8
        expected = -rice.logpdf(z, a/np.sqrt(v/2), scale=np.sqrt(v/2))+np.log(2*z)
        np.testing.assert_allclose(c.nll(z, a, v), expected, atol=1e-12)
        step = 1e-5
        fd = (c.nll(z, a+step, v)-c.nll(z, a-step, v))/(2*step)
        np.testing.assert_allclose(fd, -c.amplitude_score(z, a, v), rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(c.ratio(np.array([0., 1., 10.])), iv(1, [0., 1., 10.])/iv(0, [0., 1., 10.]))

    def test_fisher_lookup_and_limits(self):
        t = np.r_[0., np.geomspace(1e-4, 1e3, 137)]
        exact = c.fisher_weight_quadrature(t, order=240)
        np.testing.assert_allclose(c.amplitude_information(t, 1.), exact, rtol=8e-4, atol=1e-8)
        self.assertEqual(c.amplitude_information(0., 1.), 0.)
        self.assertAlmostEqual(float(c.amplitude_information(1e4, 1.)), 2.)

    def test_fisher_score_covariance(self):
        rng = np.random.default_rng(8723)
        S = (rng.normal(size=(2, 5))+1j*rng.normal(size=(2, 5)))/np.sqrt(2)
        h = np.array([[.2+.4j, -.3+.2j]])
        B = np.full((1, 5), .6+.3j)
        v = .7
        field = h@S+B
        noise = np.sqrt(v/2)*(rng.normal(size=(150000, 5))+1j*rng.normal(size=(150000, 5)))
        z = np.abs(field+noise)
        factor = np.exp(-1j*np.angle(field))*S
        derivative = np.concatenate((factor.real, -factor.imag))
        score = c.amplitude_score(z, np.abs(field), v)@derivative.T
        empirical = score.T@score/len(score)
        expected = c.real_fisher(h, S, B, v)[0]
        self.assertLess(np.linalg.norm(empirical-expected)/np.linalg.norm(expected), .015)
        self.assertLess(np.linalg.norm(score.mean(0)), .03)

    def test_real_gradient_and_rank(self):
        w = world('diagnostic', 'sparse', 15, 3)
        h, S, B, v = w.channel[:1]/np.sqrt(32), w.S[:, :1], w.B[:1, :1]/np.sqrt(32), w.variance/32
        J = c.real_fisher(h, S, B, v)[0]
        self.assertEqual(np.linalg.matrix_rank(J, tol=1e-8), 1)
        eig = np.linalg.eigvalsh(J)
        self.assertGreaterEqual(eig.min(), -1e-10)
        field = h@S+B
        factor = np.exp(-1j*np.angle(field))*S
        g = np.r_[factor.real[:, 0], -factor.imag[:, 0]]
        finite = []
        for i in range(8):
            d = np.zeros_like(h); d[0, i%4] = (1 if i < 4 else 1j)*1e-6
            finite.append(float(((np.abs((h+d)@S+B)-np.abs((h-d)@S+B))/2e-6)[0, 0]))
        np.testing.assert_allclose(g, finite, atol=1e-8)

    def test_shrinkage(self):
        rng = np.random.default_rng(234)
        h = rng.normal(size=(32, 4))+1j*rng.normal(size=(32, 4))
        np.testing.assert_array_equal(c.smooth_structure(h, 0), h)
        x, indices = c._hankel(h)
        u, s, vh = np.linalg.svd(x, full_matrices=False)
        expected = c._unhankel((u*(s**3/(s*s+1.7**2))[:, None, :])@vh, indices, 32)
        np.testing.assert_allclose(c.smooth_structure(h, 1.7), expected, atol=2e-14)
        self.assertLess(np.linalg.norm(c.smooth_structure(h, 1e8)), 1e-10)
        np.testing.assert_array_equal(c.smooth_structure(np.zeros_like(h), 1), np.zeros_like(h))

    def test_filter_bound_nullspace(self):
        rng = np.random.default_rng(349)
        d = rng.normal(size=(3, 4))+1j*rng.normal(size=(3, 4))
        J = np.broadcast_to(np.diag([0., 1., 2., 3., 4., 5., 6., 7.]), (3, 8, 8))
        np.testing.assert_array_equal(c.filter_correction(d, J, 0), d)
        delta = c.filter_correction(d, J, 1e6)
        np.testing.assert_allclose(delta.real[:, 0], d.real[:, 0])
        self.assertLess(np.linalg.norm(delta.imag), 1e-5)
        self.assertLessEqual(np.linalg.norm(delta), np.linalg.norm(d))

    def test_emgs_and_zero_equivalence(self):
        from rydberg_sim.gs import em_gs_channel_rows
        w = world('diagnostic', 'independent', 5, 1)
        cfg = c.Config(iterations=5)
        initial = c.initialize(w.Z/cfg.scale, w.S, w.B/cfg.scale)*cfg.scale
        native = em_gs_channel_rows(w.S, w.Z, w.B, w.variance, max_iter=5, ridge=0., G0=initial).G_hat
        got, _ = c.estimate(w.Z, w.S, w.B, w.variance, cfg)
        np.testing.assert_allclose(got, native, rtol=1e-11, atol=1e-11)
        for method in ('hard', 'soft', 'information'):
            zero, _ = c.estimate(w.Z, w.S, w.B, w.variance, replace(cfg, method=method, alpha=0.))
            np.testing.assert_array_equal(zero, got)
        info, _ = c.estimate(w.Z, w.S, w.B, w.variance, replace(cfg, method='information', tau=0.))
        soft, _ = c.estimate(w.Z, w.S, w.B, w.variance, replace(cfg, method='soft'))
        np.testing.assert_array_equal(info, soft)

    def test_scaling_finite_and_noiseless(self):
        w = world('diagnostic', 'clustered', 80, 2)
        cfg = c.Config(method='information', iterations=3)
        a, d = c.estimate(w.Z, w.S, w.B, w.variance, cfg)
        b, _ = c.estimate(7*w.Z, w.S, 7*w.B, 49*w.variance, replace(cfg, scale=7*cfg.scale))
        np.testing.assert_allclose(b/7, a, atol=1e-10, rtol=1e-10)
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(d['pilot_slots'], 20)
        self.assertFalse(d['oracle_only'])

    def test_determinism_splits_and_noise_rule(self):
        worlds = [world(split, 'sparse', 5, 0) for split in ('tune', 'screen', 'diagnostic')]
        self.assertEqual(len({w.case_id for w in worlds}), 3)
        self.assertEqual(len({w.digest for w in worlds}), 3)
        self.assertEqual(len({w.pilot_id for w in worlds}), 3)
        w = world('screen', 'sparse', 5, 0)
        self.assertEqual(w.digest, worlds[1].digest)
        self.assertEqual(noise_cell(w.S, w.variance), 5)
        cfg = c.Config(method='information', iterations=2)
        a, _ = c.estimate(w.Z, w.S, w.B, w.variance, cfg)
        b, _ = c.estimate(w.Z, w.S, w.B, w.variance, cfg)
        np.testing.assert_array_equal(a, b)

    def test_gradients_repeated_and_small(self):
        import torch
        torch.set_num_threads(1)
        rng = np.random.default_rng(47)
        for h in (np.zeros((6, 2), complex), np.ones((6, 2), complex)*1e-8,
                  rng.normal(size=(6, 2))+1j*rng.normal(size=(6, 2))):
            x = torch.tensor(h, dtype=torch.complex128, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda t: c.smooth_structure_torch(t, .7), (x,), eps=1e-6, atol=1e-5))
            np.testing.assert_allclose(c.smooth_structure_torch(x, .7).detach(), c.smooth_structure(h, .7), atol=1e-12)

    def test_gpu_agreement_if_available(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA not available; GPU agreement is NOT established')
        x = torch.randn((32, 4), dtype=torch.complex128)
        torch.testing.assert_close(c.smooth_structure_torch(x.cuda(), .7).cpu(), c.smooth_structure_torch(x, .7), atol=1e-10, rtol=1e-10)


class RunnerChecks(unittest.TestCase):
    def test_partial_case_resume_and_tamper_guard(self):
        from paper2_information import experiment as e
        with tempfile.TemporaryDirectory() as tmp, patch.object(e, 'OUT', Path(tmp)):
            configs = {'none': c.Config(iterations=1), 'soft': c.Config(method='soft', iterations=1)}
            budget = e.Budget()
            original = e.outcome
            calls = []
            def interrupted(w, cfg):
                calls.append(cfg.method)
                if len(calls) == 2:
                    raise InterruptedError('Simulated interruption')
                return original(w, cfg)
            with patch.object(e, 'outcome', interrupted), self.assertRaises(InterruptedError):
                e.evaluate_record('tune', 'sparse', -5, 0, configs, budget)
            with patch.object(e, 'outcome', wraps=original) as spy:
                first = e.evaluate_record('tune', 'sparse', -5, 0, configs, budget)
                self.assertEqual(spy.call_count, 1)
            second = e.evaluate_record('tune', 'sparse', -5, 0, configs, budget)
            self.assertEqual(first, second)
            file = Path(tmp)/'cases/tune/sparse/-5/000_none.npy'
            with file.open('wb') as stream:
                np.save(stream, np.zeros((32, 4)))
            with self.assertRaises(AssertionError):
                e.evaluate_record('tune', 'sparse', -5, 0, configs, budget)

    def test_budget_guard(self):
        from paper2_information import experiment as e
        with tempfile.TemporaryDirectory() as tmp, patch.object(e, 'OUT', Path(tmp)):
            with patch.object(e.time, 'monotonic', return_value=100):
                budget = e.Budget()
            with patch.object(e.time, 'monotonic', return_value=7301), self.assertRaises(e.BudgetReached):
                budget.check()
            self.assertGreater(e.read(Path(tmp)/'budget.json')['seconds'], 7200)

    def test_dependency_guard(self):
        from paper2_information import experiment as e
        with tempfile.TemporaryDirectory() as tmp, patch.object(e, 'OLD', Path(tmp)):
            lock = Path(tmp)/'run.lock'
            lock.touch()
            e.write(Path(tmp)/'status.json', {'state': 'classical_evaluation'})
            with self.assertRaises(RuntimeError):
                e.dependency_ready()
            with lock.open('r') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertFalse(e.dependency_ready())
            e.write(Path(tmp)/'status.json', {'state': 'complete'})
            self.assertTrue(e.dependency_ready())
            e.write(Path(tmp)/'status.json', {'state': 'budget_exhausted'})
            with self.assertRaises(RuntimeError):
                e.dependency_ready()

    def test_small_end_to_end_and_frozen_selection(self):
        from paper2_information import experiment as e
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(e, OUT=Path(tmp), FAMILIES=('sparse',), SNRS=(-5,), TUNE_N=1, SCREEN_N=1), patch.object(e, 'diagnostics', return_value={}):
            e.run_batch(e.Budget())
            selected = e.read(Path(tmp)/'selection.json')
            self.assertTrue(selected['screen_not_accessed'])
            self.assertEqual(e.read(Path(tmp)/'status.json')['state'], 'complete')
            self.assertIn('paper1_native', e.read(Path(tmp)/'summary.json')['macro_db'])
            before = e.digest(Path(tmp)/'selection.json')
            # Existing tune stage is not re-entered on resume.
            with patch.object(e, 'choose', side_effect=AssertionError('Retuning on resume')):
                e.run_batch(e.Budget())
            self.assertEqual(before, e.digest(Path(tmp)/'selection.json'))


if __name__ == '__main__':
    unittest.main()
