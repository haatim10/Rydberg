"""Frozen controls and reproducible data; truth is returned only for scoring.

No source in previous experiment packages is modified by this study.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import torch

from sahure.data import Observations
from sahure.train import load_model
from sahure_expansion.data import world
from fair_optshrink.estimator import estimate as classical_estimate, lookup_key

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results' / 'tcom_followups_20261017'
MODEL_SEEDS = (9101, 9102, 9103)
DEVELOPMENT_SEED = 2026101701
TEST_SEED = 2026101712
DATA_SEED = 2026101703
OPT_SELECTION = REPO / 'results/fair_optshrink/frozen_selection.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=default, allow_nan=False) + '\n')


def checkpoint_path(seed):
    return REPO / f'results/sahure_expansion/checkpoints/seed_{seed}/em_adaptive_best.pt'


def effective_trial_id(index, L):
    """Disjoint trial keys across path regimes, shared by all three studies."""
    offset = 0 if L is None else {2: 100000, 4: 200000}.get(L, int(L) * 100000)
    return offset + int(index)


def world_batch(seed, ids, snr, rsr=12., L=None):
    arrays = {key: [] for key in ('z', 's', 'b', 'sigma2')}
    truth, paths = [], []
    for i in ids:
        w, p = world(effective_trial_id(i, L), int(seed), N=32, K=3, P=30,
                     snr=float(snr), rsr=float(rsr), L=L)
        for key, val in zip(arrays, (w.Z, w.S, w.B, w.sigma2)):
            arrays[key].append(val)
        truth.append(w.G)
        paths.append(p)
    obs = Observations(*(torch.as_tensor(np.stack(arrays[key]),
                         dtype=torch.complex128 if key in ('s', 'b') else torch.float64)
                         for key in arrays))
    return obs, np.stack(truth), np.stack(paths)


def load_learned_models():
    return {f'learned_{seed}': load_model(checkpoint_path(seed), dtype=torch.float64)[0]
            for seed in MODEL_SEEDS}


def em_estimate(obs, iterations=100):
    return classical_estimate(obs, dict(rank=0, strength=0., iterations=iterations,
                                        placement='loop'))


def frozen_optshrink(obs, selection=None):
    """Observed noise/RSR lookup; out-of-grid SNR clamps to nearest trained cell."""
    if selection is None:
        selection = json.loads(OPT_SELECTION.read_text())
    keys = lookup_key(obs)
    output = torch.empty((len(obs), obs.z.shape[1], obs.s.shape[1]), dtype=obs.s.dtype)
    for key in sorted(set(keys)):
        ids = torch.tensor([i for i, k in enumerate(keys) if k == key])
        output[ids] = classical_estimate(obs.batch(ids), selection['choices'][key]['selected'])
    return output


def load_estimators():
    models = load_learned_models()
    selection = json.loads(OPT_SELECTION.read_text())
    result = {name: (lambda obs, m=model: m(*obs.args())) for name, model in models.items()}
    result['em100'] = em_estimate
    result['optshrink'] = lambda obs: frozen_optshrink(obs, selection)
    return result


def frozen_inputs():
    paths = [checkpoint_path(seed) for seed in MODEL_SEEDS] + [OPT_SELECTION]
    paths += [REPO / p for p in (
        'sahure/model.py', 'sahure/math.py', 'sahure_expansion/data.py',
        'fair_optshrink/estimator.py', 'hybrid_optshrink/model.py',
        'rydberg_sim/monte_carlo.py', 'rydberg_sim/rng.py')]
    return dict(hashes={str(p.relative_to(REPO)): sha(p) for p in paths},
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                threads=torch.get_num_threads(), development_seed=DEVELOPMENT_SEED,
                test_seed=TEST_SEED, data_seed=DATA_SEED,
                generator_policy='Independent operating point RNG keys include SNR and RSR; '
                'path-regime trial offsets prevent overlap; high-SNR study may explicitly '
                'couple worlds by rescaling a fixed anchor noise realization.')


def scores(estimate, truth):
    estimate = estimate.detach().cpu().numpy() if torch.is_tensor(estimate) else estimate
    return (np.sum(np.abs(estimate - truth) ** 2, axis=(-2, -1)),
            np.sum(np.abs(truth) ** 2, axis=(-2, -1)))
