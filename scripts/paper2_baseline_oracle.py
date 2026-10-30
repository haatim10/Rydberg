"""Coherent-observation, true-angle genie diagnostic for our channel assumptions.

For H|Y,angles ~ CN(mu,C), the Bayes estimator for per-case normalized
squared error is E[H/||H||² | Y]/E[1/||H||² | Y], not ordinary LMMSE.
The minimum conditional risk is 1-||B||²/A with A=E[1/||H||²],
B=E[H/||H||²]. Laplace integrals evaluate A and B. Averaging this risk
over independent angles and observations gives a numerical lower bound on
the achievable expected NMSE of any magnitude-only estimator under OUR
assumptions. It is NOT a bound on unspecified author channel statistics.
"""
import argparse
import json

import numpy as np
from scipy.linalg import block_diag
from scipy.special import roots_legendre
from scipy.stats import t

from paper2_original_channel import rng_for
from trackD_urformer.dataset import make_fixed_pilots


def channel_covariance(trial, mode):
    if mode == 'literal_independent':
        angles = rng_for(trial, 1).uniform(-np.pi/2, np.pi/2, (4, 4, 10))
    elif mode == 'clustered_5deg':
        centers = rng_for(trial, 2).uniform(-np.pi/2, np.pi/2, (4, 4, 1))
        offsets = np.deg2rad(rng_for(trial, 3).uniform(-5, 5, (4, 4, 10)))
        angles = np.clip(centers+offsets, -np.pi/2, np.pi/2)
    else:
        raise ValueError(mode)
    steering = np.exp(1j*np.pi*np.arange(32)[:, None, None, None]*np.sin(angles)[None])
    return block_diag(*[(32/40)*a@a.conj().T for a in [steering[:, k].reshape(32, 40) for k in range(4)]])


def psd_root(matrix):
    eigenvalues, vectors = np.linalg.eigh((matrix+matrix.conj().T)/2)
    assert eigenvalues.min() > -1e-7*max(1, eigenvalues.max())
    return (vectors*np.sqrt(np.maximum(eigenvalues, 0)))@vectors.conj().T


def posterior(prior, precision):
    root = psd_root(prior)
    covariance = root@np.linalg.solve(np.eye(len(prior))+root@precision@root, root)
    return (covariance+covariance.conj().T)/2


def relative_bayes(mu, covariance, nodes=128, upper=128):
    """Return Bayes action and risk; mu has shape (draws, channel_dimensions)."""
    eig, vectors = np.linalg.eigh(covariance)
    eig = np.maximum(eig, 0)
    # Each row contains a column vector's coordinates in the eigenbasis.
    means = mu@vectors.conj()
    power = abs(means)**2
    scale = power.sum(1)+eig.sum()
    x, w = roots_legendre(nodes)
    u, w = (x+1)*upper/2, w*upper/2
    denom = 1+u[None, :, None]*eig[None, None, :]/scale[:, None, None]
    log_m = -np.log(denom).sum(2)-(u[None, :]/scale[:, None])*(power[:, None, :]/denom).sum(2)
    weighted = np.exp(log_m)*w[None, :]
    a = weighted.sum(1)
    bi = (weighted[:, :, None]/denom).sum(1)
    delta = means*bi/a[:, None]
    risk = 1-(power*bi**2).sum(1)/(scale*a)
    assert np.isfinite(risk).all() and risk.min() > -1e-8 and risk.max() <= 1+1e-8
    return delta@vectors.T, risk


def run(count, draws, modes, start=12000000, seed=2026092319):
    s = make_fixed_pilots(4, 20, 777000001)
    q = 32*float((abs(s)**2).sum())/20
    result = {'scope': 'Numerical genie Bayes minimum of expected per-case NMSE for our declared channel prior, not author model.',
        'n_independent_angle_sets': count, 'posterior_means_per_angle_set': draws,
        'snrs_db': [5, 20], 'seed': seed, 'first_angle_trial': start, 'signal_power': q, 'results': []}
    rng = np.random.default_rng(seed)
    for mode in modes:
        for snr in result['snrs_db']:
            sigma = q/10**(snr/10)
            # Column-stacked vec(H): A = S.T kron I, so A^H A = S.conj() S.T kron I.
            precision = np.kron(s.conj()@s.T, np.eye(32))/sigma
            risks, lmmse_risks, quad_errors = [], [], []
            for j in range(count):
                prior = channel_covariance(start+j, mode)
                cov = posterior(prior, precision)
                root = psd_root(prior-cov)
                random = (rng.standard_normal((draws, 128))+1j*rng.standard_normal((draws, 128)))/np.sqrt(2)
                mu = random@root.T
                _, risk = relative_bayes(mu, cov)
                # Double resolution and integration interval for first angle set.
                if j == 0:
                    _, check = relative_bayes(mu, cov, nodes=256, upper=256)
                    quad_errors.append(float(abs(check-risk).max()))
                    assert quad_errors[-1] < 1e-7
                risks.append(float(risk.mean()))
                lmmse_risks.append(float(np.trace(cov).real/np.trace(prior).real))
            mean, sd = float(np.mean(risks)), float(np.std(risks, ddof=1))
            margin = float(t.ppf(.975, count-1)*sd/np.sqrt(count))
            row = {'mode': mode, 'snr_db': snr, 'relative_bayes_risk': mean,
                'relative_bayes_risk_db': float(10*np.log10(mean)),
                'nominal_MC_ci95_db': (10*np.log10([mean-margin, mean+margin])).tolist(),
                'ratio_of_expected_MSE_to_expected_energy': float(np.mean(lmmse_risks)),
                'max_quadrature_discrepancy': max(quad_errors), 'per_angle_risks': risks}
            result['results'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k != 'per_angle_risks'}), flush=True)
    print('FINAL_RESULT='+json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--draws', type=int, default=32)
    p.add_argument('--modes', nargs='+', default=['literal_independent', 'clustered_5deg'])
    p.add_argument('--start', type=int, default=12000000)
    p.add_argument('--seed', type=int, default=2026092319)
    args = p.parse_args()
    run(args.count, args.draws, args.modes, args.start, args.seed)
