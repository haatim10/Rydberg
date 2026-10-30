"""Numerical and data-access checks; python -m unittest p1_revision.test_baselines -v."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.stats import rice
from codex_verification.solver import fit, select, project, initial
from .baselines import (fit_candidates, posterior_field, rician_gradient, rician_nll,
                        smooth_project, split_columns, select_candidates,
                        estimate_comparators, linearized_ls)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026100701)
        def cn(shape):
            return (rng.standard_normal(shape) + 1j*rng.standard_normal(shape)) / np.sqrt(2)
        self.s = cn((2, 12))
        self.g = cn((8, 2))
        self.b = 3*cn((8, 12))
        self.v = .7
        self.z = abs(self.g @ self.s + self.b + np.sqrt(self.v)*cn((8, 12)))
        self.w = SimpleNamespace(S=self.s, Z=self.z, B=self.b, sigma2=self.v)

    def test_em_exact_parity_with_verified_solver(self):
        for ranks in ([4], [1, 2, 3, 4]):
            expected = fit(self.s, self.z, self.b, self.v, iterations=7, ranks=ranks, sweeps=4)
            actual = fit_candidates(self.s, self.z, self.b, self.v,
                                    iterations=7, ranks=ranks, sweeps=4)
            np.testing.assert_array_equal(actual, expected)
        r, scores = select(self.s, self.z, self.b, self.v, iterations=5, sweeps=4)
        idx, actual_scores = select_candidates(self.s, self.z, self.b, self.v,
                                              ranks=np.arange(1, 5), iterations=5, sweeps=4)
        self.assertEqual(idx+1, r)
        np.testing.assert_array_equal(actual_scores, scores)

    def test_total_complex_variance_likelihood_against_scipy(self):
        for a, v in [(0., .7), (1e-10, 2.), (.0001, 1.), (3., .2), (50., .01)]:
            z = np.linspace(max(1e-5, a - 5*np.sqrt(v)), a + 5*np.sqrt(v), 31)
            mu = np.full(z.shape, complex(a))
            ours = rician_nll(z, mu, v) - np.log(2*z/v)
            expected = -rice.logpdf(z, a/np.sqrt(v/2), scale=np.sqrt(v/2))
            np.testing.assert_allclose(ours, expected, rtol=1e-10, atol=1e-10)

    def test_rician_directional_gradient_and_near_zero_fields(self):
        direction = self.g / np.linalg.norm(self.g)
        for g, b, eps in [(self.g, self.b, 1e-6),
                          (1e-4*self.g, np.zeros_like(self.b), 1e-5),
                          (np.zeros_like(self.g), np.zeros_like(self.b), 1e-6)]:
            fd = (rician_nll(self.z, (g+eps*direction)@self.s+b, self.v).sum()
                  - rician_nll(self.z, (g-eps*direction)@self.s+b, self.v).sum()) / (2*eps)
            packed = 2*rician_gradient(g, self.z, self.s, b, self.v)
            analytic = np.real(np.vdot(packed, direction))
            self.assertAlmostEqual(fd, analytic, delta=2e-7*max(1., abs(analytic)))
        mu = np.array([0j, 1e-20+1e-20j, 1e-10j])
        z = np.array([1., .7, 3.])
        q = posterior_field(z, mu, self.v)
        np.testing.assert_allclose(q, mu*z*z/self.v, rtol=1e-12, atol=1e-35)

    def test_large_bessel_argument_is_finite_and_gradient_agrees(self):
        g = self.g*100
        z = abs(g@self.s+self.b) + .1
        direction = self.g/np.linalg.norm(self.g)
        eps = 1e-5
        fd = (rician_nll(z, (g+eps*direction)@self.s+self.b, .01).sum()
              - rician_nll(z, (g-eps*direction)@self.s+self.b, .01).sum()) / (2*eps)
        packed = 2*rician_gradient(g, z, self.s, self.b, .01)
        self.assertTrue(np.all(np.isfinite(packed)))
        self.assertAlmostEqual(fd, np.real(np.vdot(packed, direction)), delta=1e-4)

    def test_em_preconditioned_gradient_identity(self):
        mu = self.g@self.s+self.b
        q = posterior_field(self.z, mu, self.v)
        sh = self.s.conj().T
        gram = self.s@sh
        direct = np.linalg.solve(gram.T, ((q-self.b)@sh).T).T
        d = rician_gradient(self.g, self.z, self.s, self.b, self.v)
        expressed = self.g - np.linalg.solve(gram.T, (self.v*d).T).T
        np.testing.assert_allclose(direct, expressed, rtol=1e-13, atol=1e-13)

    def test_gradient_step_scaling_matches_formula(self):
        g0 = initial(self.s, self.z, self.b)
        expected = g0 - self.v*rician_gradient(g0, self.z, self.s, self.b, self.v)/np.linalg.norm(self.s, 2)**2
        actual = fit_candidates(self.s, self.z, self.b, self.v, core='gd', iterations=1)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        q, _ = np.linalg.qr(self.s.conj().T)
        s = q.conj().T
        em = fit_candidates(s, self.z, self.b, self.v, core='em', iterations=1)
        gd = fit_candidates(s, self.z, self.b, self.v, core='gd', iterations=1)
        np.testing.assert_allclose(em, gd, rtol=1e-13, atol=1e-13)

    def test_full_rank_and_zero_threshold_are_exact_noops(self):
        g = np.stack([self.g, self.g])
        np.testing.assert_array_equal(project(g, [4, 4], 4), g)
        np.testing.assert_array_equal(smooth_project(g, [0., 0.], 4), g)
        for core in ('em', 'gd'):
            bare = fit_candidates(self.s, self.z, self.b, self.v, core=core, iterations=5)
            smooth = fit_candidates(self.s, self.z, self.b, self.v, core=core,
                                    iterations=5, factors=[0.])
            np.testing.assert_array_equal(smooth, bare)
        mixed = smooth_project(g, [0., .1], 4)
        np.testing.assert_array_equal(mixed[0], self.g)
        self.assertFalse(np.array_equal(mixed[1], self.g))

    def test_smooth_hankel_matches_independent_svd_and_averaging(self):
        expected = np.empty_like(self.g)
        for k in range(2):
            h = np.array([[self.g[i+j, k] for j in range(4)] for i in range(5)])
            u, singular, vh = np.linalg.svd(h, full_matrices=False)
            tau = .2*np.sqrt(np.mean(singular**2))
            hh = (u*(singular**3/(singular**2+tau**2)))@vh
            expected[:, k] = [np.mean([hh[i, j] for i in range(5) for j in range(4) if i+j==n])
                              for n in range(8)]
        actual = smooth_project(self.g[None], [.2], 1)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(smooth_project(np.zeros((2, 8, 2), complex), [.1, 1.], 4), 0)

    def test_context_initializer_never_sees_heldout_columns(self):
        aa, bb = split_columns(12, 2, .3)
        with patch('p1_revision.baselines.initial', wraps=initial) as init:
            select_candidates(self.s, self.z, self.b, self.v, ranks=[2, 4], iterations=2)
            self.assertEqual(init.call_count, 1)
            passed_s, passed_z, passed_b = init.call_args.args
            np.testing.assert_array_equal(passed_s, self.s[:, aa])
            np.testing.assert_array_equal(passed_z, self.z[:, aa])
            np.testing.assert_array_equal(passed_b, self.b[:, aa])
        z2 = self.z.copy(); z2[:, bb] += 100
        base = fit_candidates(self.s[:, aa], self.z[:, aa], self.b[:, aa], self.v, iterations=2)
        changed = fit_candidates(self.s[:, aa], z2[:, aa], self.b[:, aa], self.v, iterations=2)
        np.testing.assert_array_equal(base, changed)

    def test_only_observations_used_and_diagnostics_json_serializable(self):
        class Guarded:
            S, Z, B, sigma2 = self.s, self.z, self.b, self.v
            def __getattr__(self, key):
                raise AssertionError('attempted non-observation access: '+key)
        actual, info = estimate_comparators(Guarded(), iterations=3, select_iterations=2)
        with_truth = SimpleNamespace(**vars(self.w), G=np.full_like(self.g, np.nan), rho=-1, L_k=999)
        comparison, info2 = estimate_comparators(with_truth, iterations=3, select_iterations=2)
        self.assertEqual(set(actual), {'em', 'hs', 'rician_gd', 'rician_hankel', 'em_smooth', 'xu_linearized'})
        for name in actual:
            np.testing.assert_array_equal(actual[name], comparison[name])
            np.testing.assert_array_equal(info['methods'][name]['scores'], info2['methods'][name]['scores'])
        json.dumps(info, allow_nan=False)
        self.assertEqual(info['heldout_columns'], [8, 9, 10, 11])

    def test_xu_ls_recovers_noiseless_linearized_model(self):
        zlinear = abs(self.b) + np.real(np.exp(-1j*np.angle(self.b))*(self.g@self.s))
        actual = linearized_ls(zlinear, self.s, self.b)
        np.testing.assert_allclose(actual, self.g, rtol=1e-12, atol=1e-12)

    def test_candidate_vectorization_matches_serial(self):
        for core, options in [('gd', dict(ranks=[1, 3, 4])), ('em', dict(factors=[0., .2, .8]))]:
            batch = fit_candidates(self.s, self.z, self.b, self.v, core=core, iterations=3, **options)
            key, values = next(iter(options.items()))
            for i, value in enumerate(values):
                single = fit_candidates(self.s, self.z, self.b, self.v, core=core,
                                        iterations=3, **{key: [value]})[0]
                np.testing.assert_allclose(batch[i], single, rtol=1e-13, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
