"""Comparable ten-stage starts, independent of experiment selection code.

All inference uses float64/complex128. Model loading is outside inference;
conversion, pilot solves and initialization remain inside each start call.
"""
from pathlib import Path
import numpy as np
import torch
from sahure.train import load_model
from tcom_runtime_repair_20261020.native import NativeEstimator
from tcom_runtime_bypass_20261017.sparse_fast import as_numpy, spectral_initial
from tcom_pilot_em_20261018.continuation import posterior, nll

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/tcom_start_ablation_20261020'
SEEDS = (9101, 9102, 9103)
ADAPTIVE_ROOT = REPO / 'results/tcom_pilot_em_20261018/checkpoints/context30'


def adaptive_path(seed):
    return ADAPTIVE_ROOT / f'seed_{seed}/em_adaptive_best.pt'


def global_path(seed):
    return OUT / f'checkpoints/global/seed_{seed}/em_global_best.pt'


class FixedCore:
    """Exact-Rician EM with predetermined controls and smooth Hankel filtering."""
    shape_constants = NativeEstimator.shape_constants

    def __init__(self, gamma, relative_tau=None, beta=None, noise_c=None):
        self.gamma = np.array(gamma, dtype=np.float64, copy=True)
        if self.gamma.shape != (10,) or not np.all(np.isfinite(self.gamma)):
            raise ValueError('exactly ten finite stage relaxation values required')
        self.relative_tau = None if relative_tau is None else np.array(relative_tau, dtype=np.float64, copy=True)
        self.beta = None if beta is None else np.array(beta, dtype=np.float64, copy=True)
        self.noise_c = noise_c
        self.shapes = {}
        if self.beta is not None:
            if self.beta.shape != (10,) or not np.all((self.beta >= 0) & (self.beta <= 1)):
                raise ValueError('ten gate values in [0,1] required')
            if (self.relative_tau is None) == (noise_c is None):
                raise ValueError('choose one threshold scale')
            if self.relative_tau is not None and (self.relative_tau.shape != (10,) or
                    not np.all(np.isfinite(self.relative_tau) & (self.relative_tau > 0))):
                raise ValueError('positive finite relative thresholds required')
            if noise_c is not None and not (np.isfinite(noise_c) and noise_c > 0):
                raise ValueError('positive finite noise multiplier required')

    def __call__(self, z, s, b, sigma2, *, diagnostics=False):
        z, s, b, v = as_numpy((z, s, b, sigma2))
        batch, n, _ = z.shape
        k = s.shape[1]
        sh = s.conj().swapaxes(-1, -2)
        gram = s @ sh
        op = np.linalg.solve(gram.swapaxes(-1, -2), sh.swapaxes(-1, -2)).swapaxes(-1, -2)
        if self.beta is not None:
            cols, rows, eye, unhankel = self.shape_constants(n)
            cap = min(rows, cols)
            if self.noise_c is not None:
                inverse = np.linalg.solve(gram, np.broadcast_to(np.eye(k, dtype=np.complex128), gram.shape))
                diag = np.diagonal(inverse, axis1=-2, axis2=-1).real
                if not np.all(diag > 0):
                    raise ValueError('positive diagonal of inverse pilot Gram required')
                absolute_tau = self.noise_c * np.sqrt(2 * v[:, None] * diag * max(rows, cols))
        g = spectral_initial(z, s, b)
        trace = []
        for t, gamma in enumerate(self.gamma):
            q = posterior(z, g @ s + b, v)
            dc = g + gamma * ((q - b) @ op - g)
            if self.beta is None:
                g = dc
                beta = tau = np.zeros((batch, k), dtype=np.float64)
            else:
                h = np.lib.stride_tricks.sliding_window_view(dc.swapaxes(-1, -2), cols, axis=-1)
                scale = np.sqrt(np.maximum(np.sum(np.abs(h)**2, axis=(-2, -1)) / cap, 1e-24))
                rel = (absolute_tau / scale if self.noise_c is not None
                       else np.full((batch, k), self.relative_tau[t], dtype=np.float64))
                tau = rel * scale
                hn = h / scale[..., None, None]
                t2 = rel[..., None, None]**2
                hh = hn.conj().swapaxes(-1, -2)
                solved = np.linalg.solve(hh @ hn + t2 * eye, hh)
                hs = (hn - t2 * solved.conj().swapaxes(-1, -2)) * scale[..., None, None]
                structured = (hs.reshape(batch, k, -1) @ unhankel).swapaxes(-1, -2)
                beta = np.full((batch, k), self.beta[t], dtype=np.float64)
                g = dc + beta[:, None, :] * (structured - dc)
            if diagnostics:
                trace.append(dict(g=torch.from_numpy(g.copy()),
                    beta=torch.from_numpy(beta.copy()), tau=torch.from_numpy(tau.copy()),
                    gamma=torch.tensor(gamma, dtype=torch.float64),
                    correction=torch.from_numpy(np.linalg.norm(g-dc, axis=-2) /
                                                np.maximum(np.linalg.norm(dc, axis=-2), 1e-12)),
                    nll=torch.from_numpy(nll(z, g @ s + b, v).mean((-2, -1)))))
        result = torch.from_numpy(g)
        return (result, trace) if diagnostics else result

    def estimate(self, obs, **kwargs):
        return self(*obs.args(), **kwargs)


class NativeGlobalEstimator(FixedCore):
    """Native counterpart of sahure ModelConfig(structure='global')."""
    def __init__(self, model):
        cfg = model.config
        if cfg.core != 'em' or cfg.structure != 'global' or cfg.final_only or cfg.stages != 10:
            raise ValueError('requires the ten-stage global EM model')
        if model.steps.dtype != torch.float64:
            raise ValueError('float64 inference model required')
        self.config = cfg
        gamma = (2 * torch.sigmoid(model.steps.detach())).cpu().numpy()
        relative_tau = (.001 + torch.nn.functional.softplus(model.structural[:, 0].detach())).cpu().numpy()
        beta = torch.sigmoid(model.structural[:, 1].detach()).cpu().numpy()
        if cfg.gate_one:
            beta = np.ones_like(beta)
        super().__init__(gamma, relative_tau=relative_tau, beta=beta)


def classical_configurations():
    """The complete, prespecified 30-member calibration menu."""
    return ([dict(kind='relative', relative_tau=x, beta=b)
             for x in (.01, .03, .1, .3, 1.) for b in (.1, .5, 1.)] +
            [dict(kind='noise', c=x, beta=b)
             for x in (.5, 1., 2., 4., 8.) for b in (.1, .5, 1.)])


def classical_estimator(config):
    kind = config.get('kind', 'relative')
    beta = np.full(10, float(config['beta']))
    if kind == 'relative':
        return FixedCore(np.ones(10), relative_tau=np.full(10, float(config['relative_tau'])), beta=beta)
    if kind == 'noise':
        return FixedCore(np.ones(10), noise_c=float(config['c']), beta=beta)
    raise ValueError('unknown classical threshold scale')


class StartEstimators:
    """Loaded starts; pass an integer fit seed for adaptive/global models."""
    def __init__(self, adaptive_models, global_models=None):
        self.adaptive_models = dict(adaptive_models)
        self.global_models = dict(global_models or {})
        self.adaptive = {int(seed): NativeEstimator(model) for seed, model in self.adaptive_models.items()}
        self.global_ = {int(seed): NativeGlobalEstimator(model) for seed, model in self.global_models.items()}
        self.classical = {}
        self.em10 = FixedCore(np.ones(10))

    def start(self, name, obs, seed=None, config=None, *, diagnostics=False):
        if name in ('adaptive', 'global'):
            if seed is None:
                raise ValueError('a fit seed is required')
            models = self.adaptive if name == 'adaptive' else self.global_
            return models[int(seed)].estimate(obs, diagnostics=diagnostics)
        if name == 'classical':
            if config is None:
                raise ValueError('a prespecified classical configuration is required')
            key = tuple(sorted(config.items()))
            if key not in self.classical:
                self.classical[key] = classical_estimator(config)
            return self.classical[key].estimate(obs, diagnostics=diagnostics)
        if name == 'em10':
            return self.em10.estimate(obs, diagnostics=diagnostics)
        if name == 'spectral':
            if diagnostics:
                raise ValueError('spectral initialization has no stage diagnostics')
            z, s, b, _ = as_numpy(obs)
            return torch.from_numpy(spectral_initial(z, s, b))
        raise ValueError(f'unknown start: {name}')


def load_starts(require_global=True):
    adaptive = {seed: load_model(adaptive_path(seed), dtype=torch.float64)[0] for seed in SEEDS}
    global_models = {}
    for seed in SEEDS:
        path = global_path(seed)
        if require_global or path.exists():
            global_models[seed] = load_model(path, dtype=torch.float64)[0]
    return StartEstimators(adaptive, global_models)
