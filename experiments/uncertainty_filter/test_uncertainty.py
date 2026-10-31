"""Checks for the phase-posterior uncertainty (see DERIVATION.md).

Run from the repository root:  .venv/bin/python -m pytest experiments/uncertainty_filter
"""
from fractions import Fraction
from math import factorial
import numpy as np
import pytest
import torch

from experiments.uncertainty_filter import uncertainty as U
from tcom_pilot_em_20261018.continuation import posterior

RNG = np.random.default_rng(2026103001)


def test_ratio_matches_existing_posterior_including_small_series():
    z = np.array([[[1e-9, 1e-4, .3, 2., 3., 10.]]])
    mu = np.array([[[1e-9, 2e-5+1e-5j, .2-.1j, 1+1j, 2.5j, 9.]]])
    v = np.array([.4])
    q = posterior(z, mu, v)
    r = U.bessel_ratio(U.concentration(z, mu, v))
    np.testing.assert_allclose(np.abs(q), z*r, rtol=1e-13, atol=0)


def test_one_minus_r2_against_mpmath():
    mpmath = pytest.importorskip('mpmath')
    mpmath.mp.dps = 50
    xs = np.logspace(-6, 9, 61)
    for x in xs:
        r = mpmath.besseli(1, x)/mpmath.besseli(0, x)
        exact = float(1 - r*r)
        assert abs(U.one_minus_r2(x) - exact) <= 1e-12*exact, x
    for x in (999., 1000., 1000.0001, 1001.):     # both sides of the switch
        r = mpmath.besseli(1, x)/mpmath.besseli(0, x)
        assert abs(U.one_minus_r2(x) - float(1 - r*r)) <= 1e-13/x


def test_asymptotic_coefficients_from_hankel_expansion():
    terms = 7
    def series(nu):
        out = []
        for k in range(terms):
            p = Fraction(1)
            for j in range(1, k + 1):
                p *= 4*nu*nu - (2*j - 1)**2
            out.append((-1)**k*p/(factorial(k)*8**k))
        return out
    mul = lambda a, b: [sum(a[i]*b[n - i] for i in range(n + 1)) for n in range(terms)]
    s0, s1 = series(0), series(1)
    inv = [1/s0[0]]
    for n in range(1, terms):
        inv.append(-sum(s0[i]*inv[n - i] for i in range(1, n + 1))/s0[0])
    r = mul(s1, inv)
    one_minus = [(n == 0) - c for n, c in enumerate(mul(r, r))]
    assert one_minus[0] == 0
    assert [float(c) for c in one_minus[1:1 + len(U.ASYMPTOTIC)]] == list(U.ASYMPTOTIC)


def test_zero_mean_gives_uniform_phase_variance():
    z = RNG.uniform(.1, 3, (2, 4, 5))
    var = U.field_variance(z, np.zeros_like(z, dtype=complex), np.array([.7, 1.3]))
    np.testing.assert_allclose(var, z*z, rtol=0, atol=0)


def test_relaxed_uncertainty_has_gamma_squared_scale():
    u = np.array([[.25, 2.], [4., .125]])
    gamma = np.array([.5, 1.5])
    np.testing.assert_allclose(U.relaxed_uncertainty(u, gamma), u*gamma**2)


@pytest.mark.parametrize('kappa', [1e-3, .5, 2., 20., 300.])
def test_mean_and_variance_against_quadrature(kappa):
    phi = np.linspace(-np.pi, np.pi, 400001)
    w = np.exp(kappa*(np.cos(phi) - 1))
    w /= np.trapezoid(w, phi)
    z, theta = 1.7, .4
    y = z*np.exp(1j*(phi + theta))
    mean = np.trapezoid(w*y, phi)
    var = np.trapezoid(w*np.abs(y - mean)**2, phi)
    m = np.exp(1j*theta)*kappa/(2*z)            # v = 1  =>  |M| = kappa / (2 z)
    q = posterior(np.array([[[z]]]), np.array([[[m]]]), np.array([1.]))[0, 0, 0]
    assert abs(q - mean) < 1e-8
    assert abs(U.field_variance(np.array([[[z]]]), np.array([[[m]]]), np.array([1.]))[0, 0, 0] - var) < 1e-8


def test_high_kappa_limit():
    kappa = np.logspace(2, 9, 15)
    product = kappa*U.one_minus_r2(kappa)
    np.testing.assert_allclose(product, 1 + 1/(8*kappa**2), rtol=1e-6)
    assert abs(product[-1] - 1) < 1e-15
    z, a, v = 2., 1.9, 1e-6
    var = U.field_variance(np.array([[[z]]]), np.array([[[a + 0j]]]), np.array([v]))
    assert abs(var[0, 0, 0]/(z*v/(2*a)) - 1) < 1e-9


def test_covariance_propagation_monte_carlo():
    batch, n, k, p = 1, 3, 3, 30
    s = (RNG.standard_normal((batch, k, p)) + 1j*RNG.standard_normal((batch, k, p)))/np.sqrt(2)
    var = RNG.uniform(.1, 2., (batch, n, p))
    cov = U.channel_covariance(var, s)
    gram = s @ s.conj().swapaxes(-1, -2)
    op = np.linalg.solve(gram.swapaxes(-1, -2), s.conj()).swapaxes(-1, -2)   # S^H A^-1
    draws = 200000
    e = (RNG.standard_normal((draws, n, p)) + 1j*RNG.standard_normal((draws, n, p)))*np.sqrt(var/2)
    g = e @ op[0]                                           # rows g_n^T = e_n S^H A^-1
    empirical = np.einsum('dnk,dnj->nkj', g.conj(), g)/draws   # E[conj(g) g^T] = A^-1 S D S^H A^-1
    np.testing.assert_allclose(empirical, cov[0], atol=4e-3*np.abs(cov).max())
    u = U.user_uncertainty(np.ones((batch, n, p)), np.zeros((batch, n, p), complex), s, np.ones(batch))
    np.testing.assert_allclose(u, np.einsum('bnkk->bk', U.channel_covariance(np.ones((batch, n, p)), s)).real/n)


@pytest.mark.parametrize('snr', [50., 60.])
@pytest.mark.parametrize('rsr', [5., 12., 20.])
def test_tau_unc_against_existing_rule_at_high_snr(snr, rsr):
    """Receiver-observable M only (spectral start + 20 exact EM updates)."""
    from tcom_followups_20261017.common import world_batch
    from tcom_runtime_bypass_20261017.sparse_fast import as_numpy, spectral_initial
    from tcom_pilot_em_20261018 import continuation
    obs, _, _ = world_batch(2026103002, range(16), snr, rsr, None)
    z, s, b, v = as_numpy(obs)
    g = continuation.estimate(obs, torch.from_numpy(spectral_initial(z, s, b)), 20).numpy()
    u = U.user_uncertainty(z, g @ s + b, s, v)
    n = z.shape[1]
    ratio = U.tau_uncertainty(u, n, 1.)/U.tau_existing(s, v, n, 1.)
    # Stated relation (DERIVATION.md, Section 5): tau_unc(c) -> tau_noise(c)/sqrt(2).
    np.testing.assert_allclose(ratio, 1/np.sqrt(2), rtol=.03)
    np.testing.assert_allclose(U.tau_uncertainty(u, n, np.sqrt(2)*2.), U.tau_existing(s, v, n, 2.), rtol=.03)
    # The literal expectation in the task (equal c) is off by sqrt(2), about 29 percent.
    assert np.all(np.abs(ratio - 1) > .25)
