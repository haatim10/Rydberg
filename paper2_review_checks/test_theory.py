"""Small numerical checks for the review's qualified theory statements.

Run from the repository root:
  ../../.venv-paper2/bin/python -m unittest paper2_review_checks.test_theory -v

These tests check local formulas and one filtering step. They do not prove
global identifiability, controller differentiability, or estimator convergence.
"""
import unittest

import torch

from sahure.math import hankel, nll, posterior_field, right_solve, smooth_shrink, unhankel
from sahure.model import AdaptiveHankelEstimator, ModelConfig


torch.set_num_threads(1)


def real_magnitude_jacobian(g, s, b):
    """Rows for x=[Re(g); Im(g)], with mu = g^T s + b != 0."""
    mu = g @ s + b
    if torch.any(mu.abs() == 0):
        raise ValueError("The magnitude Jacobian requires nonzero mean fields.")
    radial = mu.conj()[None, :] * s / mu.abs()[None, :]
    return torch.cat((radial.real.T, -radial.imag.T), dim=1)


def weighted_norm(g):
    """Antidiagonal-multiplicity norm; equals ||H(g)||_F exactly."""
    h = hankel(g)
    rows, cols = h.shape[-2:]
    indices = (torch.arange(rows)[:, None] + torch.arange(cols)[None, :]).flatten()
    weights = torch.bincount(indices, minlength=g.shape[-1]).to(g.real.dtype)
    return torch.sqrt((weights * g.abs().square()).sum())


class TheoryChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(803190)
        self.g = torch.randn(3, dtype=torch.complex128)
        self.s = torch.randn(3, 11, dtype=torch.complex128)
        self.b = torch.randn(11, dtype=torch.complex128)

    def test_real_jacobian_central_differences(self):
        x = torch.cat((self.g.real, self.g.imag))
        actual = real_magnitude_jacobian(self.g, self.s, self.b)
        step = 1e-6
        columns = []
        for coordinate in range(x.numel()):
            direction = torch.zeros_like(x)
            direction[coordinate] = step
            plus = x + direction
            minus = x - direction
            ap = (torch.complex(plus[:3], plus[3:]) @ self.s + self.b).abs()
            am = (torch.complex(minus[:3], minus[3:]) @ self.s + self.b).abs()
            columns.append((ap - am) / (2 * step))
        expected = torch.stack(columns, dim=1)
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)

    def test_zero_reference_phase_null_direction(self):
        zero = torch.zeros_like(self.b)
        jacobian = real_magnitude_jacobian(self.g, self.s, zero)
        phase_direction = torch.cat((-self.g.imag, self.g.real))
        torch.testing.assert_close(jacobian @ phase_direction,
                                   torch.zeros(self.s.shape[-1], dtype=torch.float64),
                                   rtol=0, atol=2e-15)
        rotation = torch.exp(torch.tensor(.47j, dtype=torch.complex128))
        torch.testing.assert_close((self.g @ self.s).abs(),
                                   ((rotation * self.g) @ self.s).abs(),
                                   rtol=1e-14, atol=1e-14)

    def test_nonzero_reference_and_full_pilot_rank_not_sufficient(self):
        # A reference lying in the pilot row space retains a shifted phase
        # ambiguity even when P >= 2K and S has full row rank.
        offset = torch.randn_like(self.g)
        reference = offset @ self.s
        self.assertEqual(int(torch.linalg.matrix_rank(self.s)), self.g.numel())
        self.assertTrue(torch.all(reference.abs() > 0))
        rotation = torch.exp(torch.tensor(.61j, dtype=torch.complex128))
        alternative = rotation * (self.g + offset) - offset
        torch.testing.assert_close((self.g @ self.s + reference).abs(),
                                   (alternative @ self.s + reference).abs(),
                                   rtol=1e-14, atol=1e-14)
        direction = self.g + offset
        tangent = torch.cat((-direction.imag, direction.real))
        jacobian = real_magnitude_jacobian(self.g, self.s, reference)
        torch.testing.assert_close(jacobian @ tangent,
                                   torch.zeros(self.s.shape[-1], dtype=torch.float64),
                                   rtol=0, atol=3e-15)

    def test_constant_reference_phase_error_exact_symmetry(self):
        # True B = exp(j phi) B_nominal. An estimator using B_nominal can fit
        # the same magnitudes with G_fit = exp(-j phi) G_true.
        phase = torch.tensor(.2, dtype=torch.float64)
        rotation = torch.exp(1j * phase)
        true_mu = self.g @ self.s + rotation * self.b
        fitting_g = rotation.conj() * self.g
        fitted_mu = fitting_g @ self.s + self.b
        torch.testing.assert_close(true_mu.abs(), fitted_mu.abs(),
                                   rtol=1e-14, atol=1e-14)
        z = true_mu.abs() + .03 * torch.rand(11, dtype=torch.float64)
        torch.testing.assert_close(nll(z, true_mu, .3), nll(z, fitted_mu, .3),
                                   rtol=1e-13, atol=1e-13)
        nmse = (fitting_g - self.g).abs().square().sum() / self.g.abs().square().sum()
        torch.testing.assert_close(nmse, 4 * torch.sin(phase / 2).square(),
                                   rtol=1e-14, atol=1e-14)

    def test_zero_mean_rician_information_first_derivative(self):
        # At zero mean, the density is smooth in real coordinates even though
        # the magnitude Jacobian formula itself is undefined.
        mu = torch.zeros(5, dtype=torch.complex128, requires_grad=True)
        z = torch.tensor([.1, .3, 1., 2., 5.], dtype=torch.float64)
        gradient = torch.autograd.grad(nll(z, mu, .4).sum(), mu)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)

    def test_relaxed_em_surrogate_decrease_without_filter(self):
        # The exact quadratic M-step allows 0 <= gamma <= 2. This property
        # need not survive a subsequent learned structural correction.
        g = torch.randn(4, 3, dtype=torch.complex128)
        b = torch.randn(4, 11, dtype=torch.complex128)
        z = torch.rand(4, 11, dtype=torch.float64) * 3
        variance = .7
        mu = g @ self.s + b
        q = posterior_field(z, mu, variance)
        em = right_solve((q - b) @ self.s.mH, self.s @ self.s.mH)
        surrogate = lambda estimate: (q - estimate @ self.s - b).abs().square().sum() / variance
        before = nll(z, mu, variance).sum()
        for gamma in (0., .1, 1., 1.8, 2.):
            updated = g + gamma * (em - g)
            decrease = surrogate(g) - surrogate(updated)
            expected = gamma * (2 - gamma) * ((em - g) @ self.s).abs().square().sum() / variance
            torch.testing.assert_close(decrease, expected, rtol=1e-12, atol=1e-12)
            self.assertLessEqual(float(nll(z, updated @ self.s + b, variance).sum()),
                                 float(before) + 1e-12)

    def test_shrink_weighted_origin_norm_and_correction_bounds(self):
        for n in (8, 9, 32):
            g = torch.randn(n, dtype=torch.complex128)
            h = hankel(g)
            torch.testing.assert_close(weighted_norm(g), torch.linalg.norm(h),
                                       rtol=1e-14, atol=1e-14)
            q = min(h.shape)
            for relative_tau in (.001, .1, 1., 10.):
                filtered, tau = smooth_shrink(h, torch.tensor(relative_tau, dtype=torch.float64))
                self.assertLessEqual(float(torch.linalg.norm(filtered)),
                                     float(torch.linalg.norm(h)) + 1e-12)
                for beta in (0., .2, 1.):
                    updated = (1 - beta) * g + beta * unhankel(filtered)
                    self.assertLessEqual(float(weighted_norm(updated)),
                                         float(weighted_norm(g)) + 1e-12)
                    bound = beta * min(float(torch.linalg.norm(h)), float(tau) * q ** .5 / 2)
                    self.assertLessEqual(float(weighted_norm(updated - g)), bound + 1e-12)

    def test_repeated_singular_values_filter_only(self):
        # H([0,0,0,1,0,0,0]) is the 4x4 reversal matrix: all singular values
        # equal one. We test only the rational filter, not ordered-SVD features.
        g = torch.zeros(7, dtype=torch.complex128)
        g[3] = 1
        h = hankel(g).clone().requires_grad_()
        torch.testing.assert_close(torch.linalg.svdvals(h), torch.ones(4, dtype=torch.float64))
        filtered, tau = smooth_shrink(h, torch.tensor(.3, dtype=torch.float64))
        expected = h / (1 + tau.square())
        torch.testing.assert_close(filtered, expected, rtol=1e-13, atol=1e-13)
        self.assertLessEqual(float(weighted_norm(unhankel(filtered)).detach()),
                             float(weighted_norm(g)) + 1e-12)
        gradient = torch.autograd.grad(filtered.abs().square().sum(), h)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: smooth_shrink(value, torch.tensor(.3, dtype=torch.float64))[0],
            (h.detach().requires_grad_(),), eps=1e-6, atol=1e-6, rtol=1e-5))

    def test_stage_parameter_count_not_accuracy_claim(self):
        for stages in (5, 10, 20):
            model = AdaptiveHankelEstimator(ModelConfig(stages=stages))
            self.assertEqual(sum(p.numel() for p in model.parameters()), 178 + stages)


if __name__ == "__main__":
    unittest.main()
