"""Independent checks: python -m unittest rician_pilot_screen.test_noise -v."""

import unittest

import numpy as np
from scipy.stats import rice
import torch

from sahure.data import Observations
from sahure.math import nll
from .noise import make_views


class RicianViewTests(unittest.TestCase):
    def data(self, z, variance):
        z = torch.as_tensor(z, dtype=torch.float64)
        if z.ndim == 1:
            z = z[:, None, None]
        batch, antennas, pilots = z.shape
        return Observations(
            z,
            torch.ones(batch, 1, pilots, dtype=torch.complex128),
            torch.full((batch, antennas, pilots), 1 + 2j, dtype=torch.complex128),
            torch.full((batch,), variance, dtype=torch.float64),
        )

    def test_determinism_all_pilots_and_no_input_mutation(self):
        original = self.data(torch.arange(48).reshape(2, 3, 8) / 8, 0.7)
        copies = tuple(t.clone() for t in original.args())
        a, b = make_views(original, 0.5, seed=731)
        again = make_views(original, 0.5, generator=torch.Generator().manual_seed(731))
        for observed, repeated in zip((a, b), again):
            for left, right in zip(observed.args(), repeated.args()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(observed.z.shape, original.z.shape)
            self.assertIs(observed.s, original.s)
            self.assertIs(observed.b, original.b)
        for current, saved in zip(original.args(), copies):
            torch.testing.assert_close(current, saved, rtol=0, atol=0)
        torch.testing.assert_close(a.sigma2, original.sigma2 * 1.25)
        torch.testing.assert_close(b.sigma2, original.sigma2 * 5)
        advanced = torch.Generator().manual_seed(731)
        first = make_views(original, 0.5, generator=advanced)[0].z
        second = make_views(original, 0.5, generator=advanced)[0].z
        self.assertFalse(torch.equal(first, second))

    def test_scaling_equivalence(self):
        original = self.data(torch.arange(24).reshape(2, 3, 4) / 5, 0.4)
        scale = 7.0
        scaled = Observations(original.z * scale, original.s, original.b * scale, original.sigma2 * scale**2)
        for base, large in zip(make_views(original, 0.5, seed=7), make_views(scaled, 0.5, seed=7)):
            torch.testing.assert_close(large.z, base.z * scale, rtol=1e-14, atol=1e-14)
            torch.testing.assert_close(large.sigma2, base.sigma2 * scale**2)

    def test_fixed_alpha_and_rng_validation(self):
        original = self.data([1.0, 2.0], 0.7)
        for alpha in (0.0, -1.0, float("nan"), float("inf"), 1e-300, 1e300):
            with self.assertRaises(ValueError):
                make_views(original, alpha, seed=1)
        for alpha in (True, torch.tensor(0.5), [0.5]):
            with self.assertRaises(TypeError):
                make_views(original, alpha, seed=1)
        with self.assertRaises(ValueError):
            make_views(original, 1.0)
        with self.assertRaises(ValueError):
            make_views(original, 1.0, seed=1, generator=torch.Generator())

    def test_float32_output(self):
        original = self.data([1.0, 2.0], 0.7)
        single = Observations(original.z.float(), original.s.cfloat(), original.b.cfloat(), original.sigma2.float())
        for view in make_views(single, 1.0, seed=19):
            self.assertEqual(view.z.dtype, torch.float32)
            self.assertEqual(view.sigma2.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(view.z).all()))

    def test_marginals_and_conditional_independence(self):
        # Fix mu within each replicate: averaging across different signals
        # would create shared-signal dependence even for genuinely independent noise.
        rng = np.random.default_rng(71423)
        count, variance = 80000, 0.7
        for mu in (0j, 0.4 + 0.9j, 3 - 2j):
            w = (rng.normal(size=count) + 1j * rng.normal(size=count)) * np.sqrt(variance / 2)
            original = self.data(np.abs(mu + w), variance)
            for alpha in (0.5, 1.0):
                with self.subTest(mu=mu, alpha=alpha):
                    left, right = make_views(original, alpha, seed=923)
                    arrays = [view.z.numpy().ravel() for view in (left, right)]
                    powers = [x**2 for x in arrays]
                    expected_variances = [variance * (1 + alpha**2), variance * (1 + alpha**-2)]
                    for x, q, v in zip(arrays, powers, expected_variances):
                        # For CN(0,v), E|mu+w|^2=|mu|^2+v and
                        # Var(|mu+w|^2)=v^2+2*v*|mu|^2.
                        mean = abs(mu)**2 + v
                        power_variance = v*v + 2*v*abs(mu)**2
                        self.assertLess(abs(q.mean() - mean), 6 * np.sqrt(power_variance / count))
                        distribution = rice(abs(mu) / np.sqrt(v / 2), scale=np.sqrt(v / 2))
                        for p in (0.2, 0.5, 0.8):
                            empirical = np.mean(x <= distribution.ppf(p))
                            self.assertLess(abs(empirical - p), 6 * np.sqrt(p * (1-p) / count))
                    self.assertLess(abs(np.corrcoef(*powers)[0, 1]), 0.025)
                    # Nonlinear indicator events supplement the zero-covariance check.
                    for p in (0.2, 0.5, 0.8):
                        events = []
                        for x, v in zip(arrays, expected_variances):
                            threshold = rice.ppf(p, abs(mu) / np.sqrt(v/2), scale=np.sqrt(v/2))
                            events.append(x <= threshold)
                        covariance = np.mean(events[0] & events[1]) - np.mean(events[0]) * np.mean(events[1])
                        self.assertLess(abs(covariance), 0.006)
                    # A same-variance target would give the wrong debiased power.
                    if mu == 0j and alpha == 1:
                        self.assertGreater((powers[1] - variance).mean(), 0.8 * variance)

    def test_target_likelihood_matches_independent_scipy_density(self):
        for alpha in (0.5, 1.0):
            for mu in (0j, 0.3 + 0.4j, 2 - 1j):
                original = self.data(np.linspace(0.05, 5, 41), 0.7)
                _, target = make_views(original, alpha, seed=4)
                field = torch.full_like(target.b, mu)
                actual = nll(target.z, field, target.sigma2, full=True).numpy().ravel()
                variance = 0.7 * (1 + alpha**-2)
                expected = -rice.logpdf(target.z.numpy().ravel(), abs(mu) / np.sqrt(variance/2), scale=np.sqrt(variance/2))
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
