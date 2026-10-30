"""Small deterministic LRHD checks; no training or experiment queue is run.

The NumPy reference below spells out the empirical D transform, matrix lift,
anti-diagonal projection, and Algorithm 2 independently of the torch filter.
It checks our declared complex/Frobenius adaptation, not unavailable author code.
"""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.special import i0e, i1e

from codex_verification.solver import fit, initial
from fair_optshrink.estimator import correction, tensors
from sahure.data import Observations
from sahure.math import spectral_initial
from sahure_expansion.data import world
from . import estimator as lrhd


def _lift(x):
    cols = (len(x) - 1) // 2 + 1
    rows = len(x) - cols + 1
    return x[np.arange(rows)[:, None] + np.arange(cols)[None, :]]


def _average(h):
    out = np.zeros(sum(h.shape) - 1, dtype=h.dtype)
    count = np.zeros(len(out), dtype=int)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            out[i + j] += h[i, j]
            count[i + j] += 1
    return out / count


def _weights(s, rows, cols, rank):
    """Reduced-noise-tail empirical D and D' as scalar NumPy arithmetic."""
    result = np.zeros_like(s)
    if s[0] == 0:
        return result
    scaled = s / s[0]
    tail = scaled[rank:]
    left, right = rows - rank, cols - rank
    for i in range(rank):
        z = scaled[i]
        gaps = z * z - tail * tail
        if z <= 0 or np.any(gaps <= 1e-13 * max(z * z, 1e-300)):
            continue
        resolvent = np.sum(z / gaps)
        derivative = np.sum(-(z * z + tail * tail) / gaps**2)
        a = (resolvent + (left - len(tail)) / z) / left
        b = (resolvent + (right - len(tail)) / z) / right
        ap = (derivative - (left - len(tail)) / z**2) / left
        bp = (derivative - (right - len(tail)) / z**2) / right
        result[i] = -2 * a * b / (ap * b + a * bp) * s[0]
    return result


def _numpy_denoise(x, rank, tolerance=1e-5, max_sweeps=1000):
    current = _lift(x)
    for sweep in range(1, max_sweeps + 1):
        u, s, vh = np.linalg.svd(current, full_matrices=False)
        low = (u * _weights(s, *current.shape, rank)) @ vh
        current = _lift(_average(low))
        distance = np.linalg.norm(current - low, 'fro')
        scale = np.linalg.norm(current, 'fro')
        residual = distance / scale if scale > 0 else 0.
        capped = bool(scale > 0 and distance >= tolerance * scale)
        if not capped:
            break
    return _average(current), dict(sweeps=sweep, residual=residual, capped=capped)


def _numpy_estimate(w, param, iterations, placement, max_sweeps):
    g = initial(w.S, w.Z, w.B)
    gram = w.S @ w.S.conj().T
    op = np.linalg.solve(gram.T, w.S.conj()).T

    def filtered(g):
        if param['strength'] == 0:
            return g.copy()
        result = np.column_stack([
            _numpy_denoise(g[:, k], param['rank'], max_sweeps=max_sweeps)[0]
            for k in range(g.shape[1])])
        return g + param['strength'] * (result - g)

    for _ in range(iterations):
        mu = g @ w.S + w.B
        amplitude = np.abs(mu)
        x = 2 * w.Z * amplitude / w.sigma2
        latent = np.divide(mu, amplitude, out=np.zeros_like(mu), where=amplitude > 0)
        latent *= w.Z * i1e(x) / i0e(x)
        g = (latent - w.B) @ op
        if placement == 'loop':
            g = filtered(g)
    return filtered(g) if placement == 'final' else g


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.w = world(0, 20261015399001, snr=5., rsr=12., L=4)[0]
        cls.obs = tensors(dict(z=cls.w.Z[None], s=cls.w.S[None],
                               b=cls.w.B[None], sigma2=np.array([cls.w.sigma2])))
        rng = np.random.default_rng(20261015399002)
        cls.g = torch.tensor((rng.normal(size=(2, 32, 3)) +
                              1j * rng.normal(size=(2, 32, 3))) / np.sqrt(2))

    def test_one_sweep_equals_old_optshrink_correction(self):
        params = [dict(rank=3, strength=.25), dict(rank=12, strength=.75)]
        actual = lrhd.correct(self.g, params, lrhd.empty_diagnostics(), max_sweeps=1)
        torch.testing.assert_close(actual, correction(self.g, params), rtol=2e-13, atol=2e-13)

    def test_independent_numpy_algorithm_and_structural_stop(self):
        for real in (False, True):
            g = self.g[:1, :, :1].clone()
            if real:
                g = g.real.to(torch.complex128)
            got, info = lrhd.denoise(g, 3)
            expected, diagnostic = _numpy_denoise(g.numpy()[0, :, 0], 3)
            np.testing.assert_allclose(got.numpy()[0, :, 0], expected, rtol=2e-8, atol=2e-9)
            self.assertEqual(int(info['sweeps'].item()), diagnostic['sweeps'])
            self.assertEqual(bool(info['capped'].item()), diagnostic['capped'])
            self.assertAlmostEqual(float(info['residual'].item()), diagnostic['residual'], places=10)
            self.assertFalse(bool(info['capped'].item()))
            self.assertLess(float(info['residual'].item()), 1e-5)

    def test_capped_residual_independent_and_deterministic_prefix(self):
        g = self.g[:1, :, :1]
        got, info = lrhd.denoise(g, 3, tolerance=1e-12, max_sweeps=2)
        expected, diagnostic = _numpy_denoise(g.numpy()[0, :, 0], 3, 1e-12, 2)
        np.testing.assert_allclose(got.numpy()[0, :, 0], expected, rtol=2e-12, atol=2e-12)
        self.assertTrue(bool(info['capped'].item()))
        self.assertEqual(int(info['sweeps'].item()), 2)
        self.assertAlmostEqual(float(info['residual'].item()), diagnostic['residual'], places=12)
        again, again_info = lrhd.denoise(g, 3, tolerance=1e-12, max_sweeps=2)
        torch.testing.assert_close(got, again, rtol=0, atol=0)
        for key in info:
            torch.testing.assert_close(info[key], again_info[key], rtol=0, atol=0)

    def test_zero_and_exact_tied_singular_values(self):
        for impulse in (False, True):
            g = torch.zeros((1, 32, 1), dtype=torch.complex128)
            if impulse:
                # A rectangular anti-identity: all sixteen singular values tie.
                g[0, 15, 0] = 1.
            got, info = lrhd.denoise(g, 3)
            self.assertTrue(torch.isfinite(got).all())
            torch.testing.assert_close(got, torch.zeros_like(got), rtol=0, atol=0)
            self.assertEqual(int(info['sweeps'].item()), 1)
            self.assertEqual(float(info['residual'].item()), 0.)
            self.assertFalse(bool(info['capped'].item()))

    def test_noiseless_two_exponential_fixed_point_and_tiny_scale(self):
        n = np.arange(32)
        x = (1.2 + .3j) * np.exp(-.42j * n) + (.7 - .2j) * np.exp(1.17j * n)
        for scale in (1., 1e-100):
            g = torch.tensor(scale * x)[None, :, None]
            got, info = lrhd.denoise(g, 2)
            np.testing.assert_allclose(got.numpy() / scale, g.numpy() / scale,
                                       rtol=2e-12, atol=2e-12)
            self.assertEqual(int(info['sweeps'].item()), 1)
            self.assertFalse(bool(info['capped'].item()))

    def test_complex_phase_scale_and_conjugation_equivariance(self):
        g = self.g[:1, :, :1]
        baseline, _ = lrhd.denoise(g, 4)
        for factor in (.02, 20., np.exp(.73j), 2.3 * np.exp(-1.8j)):
            actual, _ = lrhd.denoise(g * factor, 4)
            torch.testing.assert_close(actual / factor, baseline, rtol=2e-8, atol=2e-9)
        conjugate, _ = lrhd.denoise(g.conj(), 4)
        torch.testing.assert_close(conjugate.conj(), baseline, rtol=2e-8, atol=2e-9)

    def test_per_user_completion_matches_individual_calls(self):
        n = torch.arange(32, dtype=torch.float64)
        exact = torch.exp(.31j * n)
        g = torch.stack((torch.zeros_like(exact), exact, self.g[0, :, 0]), dim=-1)[None]
        ranks = torch.tensor([[1, 1, 3]])
        combined, diag = lrhd.denoise(g, ranks)
        for k in range(3):
            individual, expected = lrhd.denoise(g[..., k:k+1], int(ranks[0, k]))
            torch.testing.assert_close(combined[..., k:k+1], individual, rtol=2e-12, atol=2e-12)
            for key in diag:
                torch.testing.assert_close(diag[key][..., k:k+1], expected[key], rtol=2e-10, atol=2e-12)
        self.assertEqual(int(diag['sweeps'][0, 0]), 1)
        self.assertGreater(int(diag['sweeps'][0, 2]), 1)

    def test_larger_inner_cap_does_not_change_completed_outputs(self):
        g = self.g[:1, :, :1]
        a, da = lrhd.denoise(g, 3, max_sweeps=1000)
        b, db = lrhd.denoise(g, 3, max_sweeps=2000)
        self.assertFalse(bool(da['capped'].any()))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key in da:
            torch.testing.assert_close(da[key], db[key], rtol=0, atol=0)

    def test_no_correction_is_independent_classical_em(self):
        expected = fit(self.w.S, self.w.Z, self.w.B, self.w.sigma2, iterations=10)[0]
        for placement in ('loop', 'final'):
            got, diag = lrhd.estimate(self.obs, [dict(rank=0, strength=0.)],
                                      iterations=10, placement=placement)
            np.testing.assert_allclose(got.numpy()[0], expected, rtol=3e-9, atol=3e-10)
            self.assertEqual(diag['calls'], 0)
            self.assertEqual(diag['sweeps'], 0)

    def test_rician_loop_and_final_against_numpy(self):
        param = dict(rank=4, strength=.5)
        for placement in ('loop', 'final'):
            got, diag = lrhd.estimate(self.obs, [param], iterations=3,
                                      placement=placement, max_sweeps=4)
            expected = _numpy_estimate(self.w, param, 3, placement, 4)
            np.testing.assert_allclose(got.numpy()[0], expected, rtol=3e-8, atol=3e-9)
            self.assertEqual(diag['calls'], 3 if placement == 'loop' else 1)

    def test_selection_uses_only_context_for_fit_and_all_pilots_for_refit(self):
        original_estimate = lrhd.estimate
        fits = []

        def capture(obs, params, **kwargs):
            result = original_estimate(obs, params, **kwargs)
            fits.append((tuple(x.clone() for x in obs.args()), result[0].clone()))
            return result

        z = self.obs.z.clone()
        z[..., 21:] *= 1.7
        changed = Observations(z, self.obs.s, self.obs.b, self.obs.sigma2)
        base = dict(rank=7, strength=.5, iterations=2, placement='loop')
        with patch.object(lrhd, 'estimate', side_effect=capture):
            _, first = lrhd.select_and_estimate(self.obs, base, max_sweeps=2)
            _, second = lrhd.select_and_estimate(changed, base, max_sweeps=2)
        self.assertEqual([row[0][0].shape[-1] for row in fits], [21, 30, 21, 30])
        for a, b in zip(fits[0][0], fits[2][0]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(fits[0][1], fits[2][1], rtol=0, atol=0)
        self.assertFalse(np.allclose(first['scores'], second['scores']))
        self.assertTrue(np.isfinite(first['scores']).all())
        self.assertIn(first['config']['rank'], range(16))
        self.assertEqual(first['config']['iterations'], base['iterations'])

    def test_initialization_and_inputs_unchanged(self):
        before = [x.clone() for x in self.obs.args()]
        with patch.object(lrhd, 'spectral_initial', wraps=spectral_initial) as initializer:
            _, diag = lrhd.estimate(self.obs, [dict(rank=0, strength=0.)], iterations=1)
        initializer.assert_called_once()
        for actual, expected in zip(initializer.call_args.args, self.obs.args()[:3]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for a, b in zip(before, self.obs.args()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(diag['calls'], 0)


if __name__ == '__main__':
    unittest.main()
