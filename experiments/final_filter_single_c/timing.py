"""Runtime of the complete call per arm (one CPU thread, float64).

  .venv/bin/python -m experiments.final_filter_single_c.timing 1   # after freezing, before test draws
  .venv/bin/python -m experiments.final_filter_single_c.timing 2   # after all accuracy jobs are idle

Arm A is the frozen gated() call. The other arms run the same upstream calls
(start, EM tail, recovery check) and then their own final filter, including
the u_k computation for D. The first world of every setting is checked
against the offline arm computation.
"""
import argparse
import json
import platform
import time
import numpy as np
import scipy
import torch

from tcom_hybrid_20261020 import inference as hybrid_inference
from tcom_reliability_20261021 import inference as reliability
from tcom_incumbent_guard_20261019 import risk_data
from tcom_pilot_em_20261018 import continuation
from .common import (HERE, RESULTS, SEEDS, REGIMES, GRID_SNRS, GRID_RSRS, SEED, HIGH_ROUTE_DB, RULE, BASE_POLICY,
                     physical_seed, engine, route, receiver_cache, table_c, noise_filter, posterior_filter, observed)
from .test import verify_frozen, arms

CAP_MS = 10.74


@torch.no_grad()
def arm_call(obs, seed, arm, constants, est):
    """Complete call: frozen upstream, frozen recovery decision, arm's final filter."""
    if arm == 'A':
        return reliability.gated(obs, seed, RULE, est, BASE_POLICY).numpy()
    family, steps, ordinary, nominal = route(obs)
    pre = est['starts'].start(family, obs, seed=seed)
    if not ordinary and steps:
        pre = continuation.estimate(obs, pre, steps)
    # At the only reference key where recovery can fire (5) the table filter is the
    # identity, so the frozen recovery check sees exactly this incumbent.
    selected = reliability.from_base(obs, pre, RULE)
    if arm == 'B':
        return selected.numpy()
    high = float(np.mean(nominal)) > HIGH_ROUTE_DB + 1e-10
    if arm == 'C1' or (arm == 'C2' and high):
        return noise_filter(obs, selected, constants[arm])
    if arm == 'C2':
        return selected.numpy()
    if arm == 'D':
        return posterior_filter(obs, selected, constants['D'])
    raise ValueError(arm)


def rows(indices):
    out = []
    for regime, L in REGIMES.items():
        for rsr in GRID_RSRS:
            a = risk_data.make_anchor(physical_seed(SEED['timing'], rsr), indices, L, rsr, snrs=GRID_SNRS,
                                      holdout_seed=SEED['heldout_pilots_and_noise'])
            for j, index in enumerate(indices):
                for snr in GRID_SNRS:
                    out.append((dict(regime=regime, rsr=rsr, snr=snr, index=index), observed(a, snr).batch(slice(j, j + 1))))
    return out


def quantiles(values):
    v = np.asarray(values)
    return dict(n=int(len(v)), p50_ms=float(np.quantile(v, .5)), p90_ms=float(np.quantile(v, .9)),
                max_ms=float(v.max()), mean_ms=float(v.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('window', type=int, choices=(1, 2))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    choice, digest = verify_frozen()
    constants = choice['selected_constants']
    names = tuple(a for a in arms(choice) if a != 'O')
    path = RESULTS/f'timing_window{args.window}.json'
    assert not path.exists()
    indices = tuple(SEED['timing_window_indices'][args.window - 1])
    worlds = rows(indices)
    est = engine()
    # Parity: online arm calls equal offline arm computation (first index of every setting).
    parity = 0.
    for meta, obs in worlds:
        if meta['index'] != indices[0]:
            continue
        cache = receiver_cache(obs, est)
        for i, seed in enumerate(SEEDS):
            g = cache['g_pre'][i]
            high = float(np.mean(cache['nominal_snr'])) > HIGH_ROUTE_DB + 1e-10
            offline = dict(A=cache['frozen'][i], B=g, C1=noise_filter(obs, g, constants['C1']),
                           C2=noise_filter(obs, g, constants['C2']) if high else g)
            if 'D' in names:
                offline['D'] = posterior_filter(obs, g, constants['D'])
            for arm in names:
                diff = float(np.max(np.abs(arm_call(obs, seed, arm, constants, est) - offline[arm])))
                assert diff == 0., (meta, arm, diff)
                parity = max(parity, diff)
    for _, obs in (worlds[0], next(w for w in worlds if w[0]['rsr'] == 5 and w[0]['snr'] == 60)):
        for arm in names:
            for seed in SEEDS:
                arm_call(obs, seed, arm, constants, est)
    rng = np.random.default_rng(np.random.SeedSequence([SEED['timing'], args.window]))
    menu = [(arm, seed) for arm in names for seed in SEEDS]
    samples = {arm: [] for arm in names}
    started = time.time()
    for count, wi in enumerate(rng.permutation(len(worlds)), 1):
        meta, obs = worlds[wi]
        for mi in rng.permutation(len(menu)):
            arm, seed = menu[mi]
            tick = time.perf_counter_ns()
            g = arm_call(obs, seed, arm, constants, est)
            ms = (time.perf_counter_ns() - tick)/1e6
            assert np.isfinite(g).all()
            samples[arm].append(dict(meta, seed=seed, ms=ms))
        if count % 50 == 0:
            print('timing', args.window, count, '/', len(worlds), flush=True)
    summary = {}
    for arm, rec in samples.items():
        full = quantiles([r['ms'] for r in rec])
        target = quantiles([r['ms'] for r in rec if r['rsr'] == 5 and r['snr'] > 30])
        summary[arm] = dict(full_grid=full, target_range=target,
                            absolute_cap_pass=bool(full['p90_ms'] <= CAP_MS and target['p90_ms'] <= CAP_MS))
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(dict(window=args.window, indices=indices, worlds=len(worlds), frozen_sha256=digest,
        online_offline_parity_max_abs=parity, cap_ms=CAP_MS, summary=summary, raw_samples=samples,
        started_unix=started, completed_unix=time.time(),
        environment=dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
                         numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                         torch_threads=torch.get_num_threads())), indent=1) + '\n')
    print(json.dumps({a: dict(full_p90=v['full_grid']['p90_ms'], target_p90=v['target_range']['p90_ms'],
                              full_max=v['full_grid']['max_ms']) for a, v in summary.items()}, indent=1))


if __name__ == '__main__':
    main()
