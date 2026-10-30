import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from paper2_baseline_oracle import channel_covariance, posterior, psd_root, relative_bayes
from paper2_original_channel import rng_for, draw_channel
from trackD_urformer.dataset import make_fixed_pilots


class OracleTests(unittest.TestCase):
    def test_zero_mean_and_zero_uncertainty(self):
        mu = np.ones((2, 16), dtype=complex)*(1+2j)
        action, risk = relative_bayes(mu, np.zeros((16, 16)))
        np.testing.assert_allclose(action, mu, atol=1e-12)
        np.testing.assert_allclose(risk, 0, atol=1e-10)
        action, risk = relative_bayes(mu*0, np.eye(16))
        np.testing.assert_allclose(action, 0, atol=1e-12)
        np.testing.assert_allclose(risk, 1, atol=1e-12)

    def test_units_and_complex_rotation(self):
        rng = np.random.default_rng(57)
        x = rng.standard_normal((16, 16))+1j*rng.standard_normal((16, 16))
        cov = x@x.conj().T/16
        mu = rng.standard_normal((3, 16))+1j*rng.standard_normal((3, 16))
        delta, risk = relative_bayes(mu, cov)
        q, _ = np.linalg.qr(x)
        d2, r2 = relative_bayes(7*mu@q.T, 49*q@cov@q.conj().T)
        np.testing.assert_allclose(d2, 7*delta@q.T, atol=1e-9)
        np.testing.assert_allclose(r2, risk, atol=1e-10)

    def test_relative_risk_against_direct_monte_carlo(self):
        rng = np.random.default_rng(812)
        x = rng.standard_normal((16, 16))+1j*rng.standard_normal((16, 16))
        cov = x@x.conj().T/32
        mu = (rng.standard_normal(16)+1j*rng.standard_normal(16))*1.3
        delta, risk = relative_bayes(mu[None], cov)
        z = (rng.standard_normal((150000, 16))+1j*rng.standard_normal((150000, 16)))/np.sqrt(2)
        h = mu+z@psd_root(cov).T
        energy = (abs(h)**2).sum(1)
        empirical_action = (h/energy[:, None]).mean(0)/(1/energy).mean()
        np.testing.assert_allclose(delta[0], empirical_action, atol=.008, rtol=0)
        loss = (abs(h-delta[0])**2).sum(1)/energy
        self.assertLess(abs(loss.mean()-risk[0]), 5*loss.std()/np.sqrt(len(loss)))
        # This verifies normalized risk, not the different trace(C)/E||H||² metric.
        ordinary = (abs(h-mu)**2).sum(1)/energy
        self.assertLessEqual(loss.mean(), ordinary.mean()+1e-5)

    def test_posterior_against_direct_gaussian_conditioning(self):
        rng = np.random.default_rng(18)
        root = rng.standard_normal((8, 8))+1j*rng.standard_normal((8, 8))
        prior = root@root.conj().T
        a = rng.standard_normal((12, 8))+1j*rng.standard_normal((12, 8))
        variance = 2.1
        direct = prior-prior@a.conj().T@np.linalg.solve(a@prior@a.conj().T+variance*np.eye(12), a@prior)
        np.testing.assert_allclose(posterior(prior, a.conj().T@a/variance), direct, atol=1e-10)

    def test_vectorization_and_covariance_convention(self):
        s = make_fixed_pilots(4, 20, 777000001)
        g = draw_channel(12000003, 'literal_independent')
        a = np.kron(s.T, np.eye(32))
        np.testing.assert_allclose(a@g.reshape(-1, order='F'), (g@s).reshape(-1, order='F'), atol=1e-12)
        np.testing.assert_allclose(a.conj().T@a, np.kron(s.conj()@s.T, np.eye(32)), atol=1e-12)
        cov = channel_covariance(12000003, 'literal_independent')
        np.testing.assert_allclose(np.diag(cov), 32, atol=1e-12)
        angles = rng_for(12000003, 1).uniform(-np.pi/2, np.pi/2, (4, 4, 10))
        steering = np.exp(1j*np.pi*np.arange(32)[:, None]*np.sin(angles[0].reshape(-1))[None])
        np.testing.assert_allclose(cov[:32, :32], (32/40)*steering@steering.conj().T, atol=1e-12)

    def test_conditional_prior_matches_actual_generator(self):
        trial = 12000003
        angles = rng_for(trial, 1).uniform(-np.pi/2, np.pi/2, (4, 4, 10))
        generator = rng_for(trial, 4)
        gains = (generator.standard_normal((4, 4, 10))+1j*generator.standard_normal((4, 4, 10)))/np.sqrt(2)
        columns = []
        for k in range(4):
            steering = np.exp(1j*np.pi*np.arange(32)[:, None]*np.sin(angles[k].reshape(-1))[None])
            columns.append(np.sqrt(32/40)*steering@gains[k].reshape(-1))
        np.testing.assert_allclose(np.stack(columns, axis=1), draw_channel(trial, 'literal_independent'), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
