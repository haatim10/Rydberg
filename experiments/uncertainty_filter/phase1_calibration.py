"""Phase 1: calibration of u_k against the nominal scale (no training).

Usage:  .venv/bin/python -m experiments.uncertainty_filter.phase1_calibration [--workers 4]
Writes results/phase1/records_*.npz and results/phase1/run_manifest.json.
Truth is used only to compute the actual error and the bias fraction.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import scipy
import torch

from tcom_followups_20261017.common import REPO, sha, world_batch, effective_trial_id
from tcom_hybrid_20261020 import inference as hybrid_inference
from tcom_reliability_20261021.inference import BASE_POLICY, BASE_POLICY_PATH
from tcom_pilot_em_20261018 import continuation
from tcom_start_ablation_20261020.starts import load_starts, adaptive_path, global_path
from sahure.data import Observations
from sahure_expansion.data import world
from rydberg_sim.forward import exact_forward
from . import uncertainty as U
from .stages import run

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/'configs/phase1.json').read_text())
OUT = HERE/'results/phase1'
SEEDS = tuple(CONFIG['model_seeds'])
REGIMES = CONFIG['regimes']
BIAS_CHANNELS = 16
BIAS_DRAWS = 16
PARITY_TOL = 1e-10


def cells():
    main = [(snr, rsr, name) for snr in CONFIG['snrs_db'] for rsr in CONFIG['rsrs_db'] for name in REGIMES]
    weak = [(snr, rsr, name) for snr in CONFIG['snrs_db'] for rsr in CONFIG['weak_reference_rsr_db'] for name in REGIMES]
    return main, weak


def random_start(g_spectral, seed, cell_index):
    """Poor start: per-user power from the (observable) spectral start, random direction."""
    rng = np.random.default_rng([seed, cell_index])
    shape = g_spectral.shape
    w = (rng.standard_normal(shape) + 1j*rng.standard_normal(shape))/np.sqrt(2)
    power = np.mean(np.abs(g_spectral)**2, axis=-2, keepdims=True)
    return np.sqrt(power)*w


def routed(obs):
    """Receiver routing (family, extra EM steps, ordinary) for a homogeneous cell."""
    plan = hybrid_inference.plans(obs, BASE_POLICY)
    keys = {(r['family'], r['steps'], r['ordinary']) for r in plan}
    if len(keys) != 1:
        raise ValueError('cell is not homogeneous in routing')
    return keys.pop()


def stage_records(obs, truth, trace, g_final_in):
    """Stack per-stage proxy, nominal scale and actual error.

    For stages 1--10, ``a`` targets the relaxed output ``dc`` that immediately
    enters the Hankel filter. The raw posterior covariance describes the full
    EM map, hence the required gamma-squared factor. The final-filter input
    remains recorded for diagnostics, but it is not a direct conditional-MSE
    target: it has no following relaxed EM map.
    """
    z, s, b, v = (x.detach().cpu().numpy() for x in obs.args())
    n = z.shape[1]
    nominal = np.broadcast_to(U.nominal_scale(s, v)[:, None], (len(z), s.shape[1]))
    a, b_scale, c = [], [], []
    for st in trace:
        a.append(U.relaxed_uncertainty(U.user_uncertainty(z, st['mu'], s, v), st['gamma']))
        b_scale.append(nominal*st['gamma']**2)
        c.append(np.sum(np.abs(st['dc'] - truth)**2, axis=-2)/n)
    a.append(U.user_uncertainty(z, g_final_in @ s + b, s, v))
    b_scale.append(nominal)
    c.append(np.sum(np.abs(g_final_in - truth)**2, axis=-2)/n)
    a, b_scale, c = (np.stack(x, 1) for x in (a, b_scale, c))   # [batch, 11, K]
    return a, b_scale, c


def evaluate_cell(args):
    """All seeds and both starts for one cell; returns arrays and parity diagnostics."""
    cell_index, (snr, rsr, regime) = args
    torch.set_num_threads(1)
    starts = load_starts()
    L = REGIMES[regime]
    obs, truth, _ = world_batch(CONFIG['data_seed'], range(CONFIG['channels_per_cell']), snr, rsr, L)
    family, steps, ordinary = routed(obs)
    out = dict(snr=snr, rsr=rsr, regime=regime, family=family, tail=steps, ordinary=ordinary)
    z, s, b, v = (x.numpy() for x in obs.args())
    from tcom_runtime_bypass_20261017.sparse_fast import spectral_initial
    g_spec = spectral_initial(z, s, b)
    g_rand = random_start(g_spec, CONFIG['random_start_seed'], cell_index)
    parity = 0.
    for seed in SEEDS:
        core = starts.adaptive[seed] if family == 'adaptive' else starts.global_[seed]
        for start_name, g0 in (('receiver', None), ('random', g_rand)):
            g10, trace = run(core, obs, g0)
            if start_name == 'receiver':
                frozen = starts.start(family, obs, seed=seed).numpy()
                parity = max(parity, float(np.max(np.abs(frozen - g10))))
                if parity > PARITY_TOL:
                    raise AssertionError(f'instrumented start differs from frozen start: {parity}')
            g_in = g10 if (ordinary or not steps) else continuation.estimate(obs, torch.from_numpy(g10), steps).numpy()
            if start_name == 'receiver' and not ordinary:
                # Parity with the full frozen receiver output (filter applied to g_in).
                full = hybrid_inference.hybrid(obs, seed, BASE_POLICY, dict(starts=starts)).numpy()
                from tcom_start_budget_20261020 import study as finish_module
                mine = finish_module.finish(obs, torch.from_numpy(g_in)).numpy()
                parity = max(parity, float(np.max(np.abs(full - mine))))
                if parity > PARITY_TOL:
                    raise AssertionError(f'final-filter input does not reproduce receiver: {parity}')
            a, nb, c = stage_records(obs, truth, trace, g_in)
            out[f'{start_name}_{seed}'] = dict(a=a, b=nb, c=c)
    out['parity'] = parity
    return out


def evaluate_bias(args):
    """Repeated noise draws with G, S, B fixed (receiver start, all seeds)."""
    cell_index, (snr, rsr, regime) = args
    torch.set_num_threads(1)
    starts = load_starts()
    L = REGIMES[regime]
    worlds = [world(effective_trial_id(i, L), CONFIG['data_seed'], N=32, K=3, P=30, snr=float(snr), rsr=float(rsr), L=L)[0]
              for i in range(BIAS_CHANNELS)]
    rng = np.random.default_rng([CONFIG['bias_noise_seed'], cell_index])
    zs = np.stack([[exact_forward(w.G, w.S, w.B, w.sigma2, rng_noise=rng).Z for w in worlds] for _ in range(BIAS_DRAWS)])
    s = np.stack([w.S for w in worlds]); b = np.stack([w.B for w in worlds])
    v = np.array([w.sigma2 for w in worlds]); g = np.stack([w.G for w in worlds])
    result = {}
    for seed in SEEDS:
        dcs = []
        family = steps = ordinary = None
        for r in range(BIAS_DRAWS):
            obs = Observations(torch.from_numpy(zs[r]), torch.from_numpy(s), torch.from_numpy(b), torch.from_numpy(v))
            family, steps, ordinary = routed(obs)
            core = starts.adaptive[seed] if family == 'adaptive' else starts.global_[seed]
            g10, trace = run(core, obs)
            g_in = g10 if (ordinary or not steps) else continuation.estimate(obs, torch.from_numpy(g10), steps).numpy()
            dcs.append(np.stack([st['dc'] for st in trace] + [g_in], 1))   # [ch, 11, N, K]
        dcs = np.stack(dcs)                                               # [R, ch, 11, N, K]
        err = dcs - g[None, :, None]
        total = np.mean(np.sum(np.abs(err)**2, axis=-2), axis=0)/32       # [ch, 11, K]
        mean_err = err.mean(0)
        within = np.sum(np.abs(dcs - dcs.mean(0, keepdims=True))**2, axis=-2).sum(0)/(BIAS_DRAWS - 1)/32
        bias = np.sum(np.abs(mean_err)**2, axis=-2)/32 - within/BIAS_DRAWS
        result[seed] = dict(total=total, bias=bias)
    return dict(snr=snr, rsr=rsr, regime=regime, **{f'bias_{k}': v_['bias'] for k, v_ in result.items()},
                **{f'total_{k}': v_['total'] for k, v_ in result.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    main_cells, weak_cells = cells()
    all_cells = list(enumerate(main_cells + weak_cells))
    if args.smoke:
        all_cells = [all_cells[i] for i in (0, 45, 80, 89, 95)]
    tick = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(evaluate_cell, all_cells))
        print(json.dumps(dict(phase='cells', seconds=time.time() - tick, max_parity=max(r['parity'] for r in results))), flush=True)
        bias_cells = [c for c in all_cells if c[1] in main_cells]
        bias = list(pool.map(evaluate_bias, bias_cells))
    suffix = '_smoke' if args.smoke else ''
    arrays = {}
    meta = []
    for i, r in enumerate(results):
        meta.append({k: r[k] for k in ('snr', 'rsr', 'regime', 'family', 'tail', 'ordinary', 'parity')})
        for key, val in r.items():
            if isinstance(val, dict):
                for name, arr in val.items():
                    arrays[f'{i}/{key}/{name}'] = arr
    np.savez_compressed(OUT/f'records{suffix}.npz', **arrays)
    np.savez_compressed(OUT/f'bias{suffix}.npz', **{f'{i}/{k}': v for i, r in enumerate(bias)
                                                   for k, v in r.items() if isinstance(v, np.ndarray)})
    sources = [HERE/f for f in ('uncertainty.py', 'stages.py', 'phase1_calibration.py', 'configs/phase1.json')]
    sources += [REPO/p for p in ('sahure/math.py', 'sahure/model.py', 'tcom_runtime_repair_20261020/native.py',
                                 'tcom_start_ablation_20261020/starts.py', 'tcom_hybrid_20261020/inference.py',
                                 'tcom_pilot_em_20261018/continuation.py', 'tcom_incumbent_guard_20261019/shrink.py',
                                 'tcom_start_budget_20261020/study.py', 'tcom_reliability_20261021/inference.py')]
    checkpoints = [adaptive_path(s) for s in SEEDS] + [global_path(s) for s in SEEDS] + [BASE_POLICY_PATH]
    manifest = dict(cells=meta, bias_cells=[{k: r[k] for k in ('snr', 'rsr', 'regime')} for r in bias],
        bias_channels=BIAS_CHANNELS, bias_draws=BIAS_DRAWS, seconds=time.time() - tick,
        max_parity=max(m['parity'] for m in meta), smoke=args.smoke,
        source_sha256={str(p.relative_to(REPO)): sha(p) for p in sources},
        checkpoint_sha256={str(p.relative_to(REPO)): sha(p) for p in checkpoints},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         torch=torch.__version__, machine=platform.machine(), threads_per_worker=1, workers=args.workers))
    (OUT/f'run_manifest{suffix}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(phase='done', seconds=manifest['seconds'], max_parity=manifest['max_parity'])), flush=True)


if __name__ == '__main__':
    main()
