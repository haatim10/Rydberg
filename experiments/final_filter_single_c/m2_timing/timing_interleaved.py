"""Interleaved runtime of arms A, C2 and C3 (one process, one thread, float64).

Run from the repository root, with nothing else running:

  .venv/bin/python -m experiments.final_filter_single_c.m2_timing.timing_interleaved --tag m2

For every world and fit the three arms run back to back on the same inputs, in an
order rotated per call, so slow drift in machine speed affects all arms alike.
Writes results/timing_<tag>.json and refuses to overwrite it.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import scipy
import torch

from tcom_incumbent_guard_20261019 import risk_data
from ..common import (REGIMES, GRID_SNRS, GRID_RSRS, SEEDS, SEED, physical_seed, engine, receiver_cache, observed, sha)
from .arms import ARMS, arm_call, offline

HERE = Path(__file__).resolve().parent
PRE = json.loads((HERE/'configs/followup_preregistration.json').read_text())['part_1_timing']
CAP_MS = 10.74
INDICES = (900, 901, 902)
ORDERS = (('A', 'C2', 'C3'), ('C2', 'C3', 'A'), ('C3', 'A', 'C2'))


def cpu_name():
    for cmd in (['sysctl', '-n', 'machdep.cpu.brand_string'], ['sh', '-c', "grep -m1 'model name' /proc/cpuinfo"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor()


def worlds():
    out = []
    for regime, L in REGIMES.items():
        for rsr in GRID_RSRS:
            a = risk_data.make_anchor(physical_seed(SEED['timing'], rsr), INDICES, L, rsr, snrs=GRID_SNRS,
                                      holdout_seed=SEED['heldout_pilots_and_noise'])
            for j, index in enumerate(INDICES):
                for snr in GRID_SNRS:
                    out.append((dict(regime=regime, rsr=rsr, snr=snr, index=index), observed(a, snr).batch(slice(j, j + 1))))
    return out


def stats(values):
    v = np.asarray(values)
    return dict(n=int(len(v)), p50_ms=float(np.quantile(v, .5)), p90_ms=float(np.quantile(v, .9)),
                max_ms=float(v.max()), mean_ms=float(v.mean()))


SCOPES = dict(full_grid=lambda m: True, snr_30_60=lambda m: m['snr'] >= 30,
              rsr5_above_30=lambda m: m['rsr'] == 5 and m['snr'] > 30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True, help="'m2' on the Apple M2; anything else is a labelled diagnostic")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    path = HERE/'results'/f'timing_{args.tag}.json'
    assert not path.exists(), f'{path} exists; keep earlier timing records'
    rows = worlds()
    est = engine()
    load_before = os.getloadavg()
    # Online call == offline computation, first index of every setting, every arm and fit.
    parity = 0.
    for meta, obs in rows:
        if meta['index'] != INDICES[0]:
            continue
        cache = receiver_cache(obs, est)
        for i, seed in enumerate(SEEDS):
            for arm in ARMS:
                diff = float(np.max(np.abs(arm_call(obs, seed, arm, est) - offline(cache, i, arm, obs))))
                assert diff == 0., (meta, arm, diff)
    for _, obs in (rows[0], next(r for r in rows if r[0]['rsr'] == 5 and r[0]['snr'] == 60)):
        for arm in ARMS:
            for seed in SEEDS:
                arm_call(obs, seed, arm, est)
    rng = np.random.default_rng(np.random.SeedSequence([SEED['timing'], 0x4D32]))
    samples = {arm: [] for arm in ARMS}
    started, call = time.time(), 0
    for count, wi in enumerate(rng.permutation(len(rows)), 1):
        meta, obs = rows[wi]
        for si in rng.permutation(len(SEEDS)):
            seed = SEEDS[si]
            for arm in ORDERS[call % 3]:
                tick = time.perf_counter_ns()
                g = arm_call(obs, seed, arm, est)
                ms = (time.perf_counter_ns() - tick)/1e6
                assert np.isfinite(g).all()
                samples[arm].append(dict(meta, seed=seed, ms=ms, call=call))
            call += 1
        if count % 50 == 0:
            print('timing', count, '/', len(rows), flush=True)
    load_after = os.getloadavg()
    summary = {}
    for arm, rec in samples.items():
        summary[arm] = {k: stats([r['ms'] for r in rec if f(r)]) for k, f in SCOPES.items()}
        summary[arm]['absolute_cap'] = dict(
            full_grid_and_snr_30_60=bool(summary[arm]['full_grid']['p90_ms'] <= CAP_MS and summary[arm]['snr_30_60']['p90_ms'] <= CAP_MS),
            rsr5_above_30=bool(summary[arm]['rsr5_above_30']['p90_ms'] <= CAP_MS))
    ratios = {arm: {k: summary[arm][k]['p90_ms']/summary['A'][k]['p90_ms'] for k in SCOPES} for arm in ARMS if arm != 'A'}
    # Paired per-call ratio: same world, fit and position block.
    paired = {}
    for arm in ('C2', 'C3'):
        a = {(r['call']): r['ms'] for r in samples['A']}
        paired[arm] = stats([r['ms']/a[r['call']] for r in samples[arm]])
    record = dict(tag=args.tag, is_m2=args.tag == 'm2', method=PRE['method'], limit=PRE['limit'], cap_ms=CAP_MS,
                  indices=INDICES, worlds=len(rows), calls_per_arm=len(samples['A']), online_offline_parity=parity,
                  summary=summary, p90_ratio_vs_A=ratios, paired_call_ratio_vs_A=paired,
                  load_average_before=load_before, load_average_after=load_after,
                  started_unix=started, completed_unix=time.time(), raw_samples=samples,
                  source_sha256={p: sha(HERE/p) for p in ('timing_interleaved.py', 'arms.py')},
                  environment=dict(platform=platform.platform(), machine=platform.machine(), cpu=cpu_name(),
                                   python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                   torch=torch.__version__, torch_threads=torch.get_num_threads()))
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(dict(tag=args.tag, cpu=record['environment']['cpu'],
        p90={a: {k: round(v[k]['p90_ms'], 3) for k in SCOPES} for a, v in summary.items()},
        max={a: round(v['full_grid']['max_ms'], 2) for a, v in summary.items()},
        ratio=ratios, cap={a: v['absolute_cap'] for a, v in summary.items()}), indent=1))


if __name__ == '__main__':
    main()
