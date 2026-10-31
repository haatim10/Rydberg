"""Step 4: fresh test draws, generated only after the choices are frozen.

  .venv/bin/python -m experiments.final_filter_single_c.test --workers 4
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch

from tcom_followups_20261017.ber_detector import detect, hypotheses
from .common import (CACHE, HERE, SEEDS, REGIMES, GRID_SNRS, GRID_RSRS, OFF_SNRS, OFF_RSRS, WEAK_RSRS, COUNT,
                     SEED, NOISE_C, HIGH_ROUTE_DB, cellname, make_block, engine, receiver_cache, table_c,
                     noise_filter, posterior_filter, squared_error, observed, sha)

OUT = CACHE/'test'
FROZEN = HERE/'configs/frozen_choices.json'
TEST_SNRS = tuple(sorted(GRID_SNRS + OFF_SNRS))
TEST_RSRS = GRID_RSRS + OFF_RSRS + WEAK_RSRS
_EST = None
_CHOICE = None


def verify_frozen():
    recorded = (HERE/'configs/frozen_choices.sha256').read_text().split()[0]
    if sha(FROZEN) != recorded:
        raise AssertionError('frozen_choices.json does not match its recorded hash')
    return json.loads(FROZEN.read_text()), recorded


def arms(choice):
    names = ['A', 'B', 'C1', 'C2'] + (['D'] if 'D' in choice['selected_constants'] else []) + ['O']
    return tuple(names)


def _init():
    global _EST, _CHOICE
    torch.set_num_threads(1)
    _EST = engine()
    _CHOICE = verify_frozen()


def ber_data(truth, b, v, regime, rsr):
    x, bits = hypotheses(3)
    out = []
    for i in range(len(truth)):
        rng = np.random.default_rng(np.random.SeedSequence(
            [SEED['ber_data'], list(REGIMES).index(regime), int(round(10*rsr)), i]))
        n = (rng.standard_normal((32, 64)) + 1j*rng.standard_normal((32, 64)))/np.sqrt(2)
        out.append(np.abs(truth[i] @ x.T + b[i] + np.sqrt(v[i])*n))
    return np.stack(out), bits


def run_block(job):
    regime, rsr = job
    choice, digest = _CHOICE
    constants = choice['selected_constants']
    names = arms(choice)
    a = make_block(SEED['test'], regime, rsr, TEST_SNRS)
    rows = []
    for snr in TEST_SNRS:
        path = OUT/'cells'/f'{cellname(regime, rsr, snr)}.npz'
        if path.exists():
            continue
        obs = observed(a, snr, P=30)
        c = receiver_cache(obs, _EST)
        tc = table_c(obs, c['ordinary'])
        high = float(np.mean(c['nominal_snr'])) > HIGH_ROUTE_DB + 1e-10
        energy = np.sum(np.abs(a['g'])**2, axis=(-2, -1))
        grid_err = np.empty((len(NOISE_C), len(SEEDS), COUNT))
        estimates = {n: [] for n in names}
        parity = 0.
        for i in range(len(SEEDS)):
            g = c['g_pre'][i]
            filtered = {cc: noise_filter(obs, g, cc) for cc in NOISE_C}
            for j, cc in enumerate(NOISE_C):
                grid_err[j, i] = squared_error(filtered[cc], a['g'])
            arm_a = noise_filter(obs, g, tc)
            if not np.array_equal(arm_a, c['frozen'][i]):
                raise AssertionError(f'arm A differs from frozen output in {path.stem}')
            parity = max(parity, float(np.max(np.abs(arm_a - c['frozen'][i]))))
            estimates['A'].append(arm_a)
            estimates['B'].append(g)
            estimates['C1'].append(filtered[constants['C1']])
            estimates['C2'].append(filtered[constants['C2']] if high else g)
            if 'D' in estimates:
                estimates['D'].append(posterior_filter(obs, g, constants['D']))
        # Oracle: one c per cell, minimizing the 3-fit mean NMSE dB, chosen with truth.
        cell_db = np.mean(10*np.log10(grid_err.sum(-1)/energy.sum()), axis=1)
        oracle_c = float(NOISE_C[int(np.argmin(cell_db))])
        estimates['O'] = [noise_filter(obs, c['g_pre'][i], oracle_c) for i in range(len(SEEDS))]
        estimates = {n: np.stack(v) for n, v in estimates.items()}
        err = np.stack([[squared_error(estimates[n][i], a['g']) for i in range(len(SEEDS))] for n in names])
        # BER through the existing detector; identical channel arrays reuse counts.
        z, s, b, v = (x.numpy() for x in obs.args())
        data, bits = ber_data(a['g'], b[..., :1], v, regime, rsr)
        ber_err = np.zeros((len(names), len(SEEDS), COUNT), dtype=np.int64)
        seen = {}
        for ni, name in enumerate(names):
            for i in range(len(SEEDS)):
                for k in range(COUNT):
                    channel = estimates[name][i, k]
                    key = hashlib.sha256(channel.tobytes()).hexdigest()
                    if key not in seen:
                        decoded = detect(data[k], channel, b[k, :, :1], float(v[k]))
                        seen[key] = int(np.count_nonzero(decoded != bits))
                    ber_err[ni, i, k] = seen[key]
        np.savez_compressed(path, arm_names=np.array(names), err=err, grid_err=grid_err, energy=energy,
            ber_errors=ber_err, bits_per_case=np.array(384), oracle_c=np.array(oracle_c), table_c=np.array(tc),
            high_route=np.array(high), accepted=c['accepted'], triggered=c['triggered'], family=np.array(c['family']),
            steps=np.array(c['steps']), ordinary=np.array(c['ordinary']), nominal_snr=c['nominal_snr'],
            snr=np.array(snr), rsr=np.array(rsr), regime=np.array(regime), frozen_sha256=np.array(digest),
            parity_max_abs=np.array(parity), unique_detector_calls=np.array(len(seen)))
        # Arms are cheap deterministic functions of g_pre and the frozen constants.
        np.savez_compressed(OUT/'estimates'/f'{path.stem}.npz', g_pre=c['g_pre'], truth=a['g'])
        rows.append((path.stem, parity, oracle_c))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    verify_frozen()
    assert (HERE/'results/timing_window1.json').exists(), 'timing window 1 runs before the test draws'
    for d in ('cells', 'estimates'):
        (OUT/d).mkdir(parents=True, exist_ok=True)
    jobs = [(r, x) for x in TEST_RSRS for r in REGIMES]
    tick = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        for rows in pool.map(run_block, jobs):
            for name, parity, oc in rows:
                print(json.dumps(dict(cell=name, parity=parity, oracle_c=oc, seconds=round(time.time() - tick, 1))), flush=True)


if __name__ == '__main__':
    main()
