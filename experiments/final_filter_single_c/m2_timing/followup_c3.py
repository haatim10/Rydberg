"""Post hoc follow-up: C3 = C2 restricted to settings with a nonzero EM tail.

  .venv/bin/python -m experiments.final_filter_single_c.m2_timing.followup_c3 run --workers 4
  .venv/bin/python -m experiments.final_filter_single_c.m2_timing.followup_c3 analyze

Fresh draws (seed 2026103121). Same metrics and block bootstrap as the main study.
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tcom_followups_20261017.ber_detector import detect, hypotheses
from tcom_incumbent_guard_20261019 import risk_data
from ..common import (CACHE, REGIMES, GRID_SNRS, GRID_RSRS, OFF_SNRS, COUNT, SEEDS, cellname, physical_seed,
                      engine, receiver_cache, observed, squared_error, sha)
from .arms import ARMS, offline

HERE = Path(__file__).resolve().parent
PRE = json.loads((HERE/'configs/followup_preregistration.json').read_text())['part_2_c3']
SEED = PRE['seeds']
OUT = CACHE/'followup_c3'
SNRS = tuple(sorted(GRID_SNRS + OFF_SNRS))
RSRS = GRID_RSRS + (0., 8., 16.)
BOOT = 20000
_EST = None


def _init():
    global _EST
    torch.set_num_threads(1)
    _EST = engine()


def block(regime, rsr):
    return risk_data.make_anchor(physical_seed(SEED['draws'], rsr), range(COUNT), REGIMES[regime], float(rsr),
                                 truth=True, snrs=SNRS, holdout_seed=SEED['heldout_pilots_and_noise'])


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
    a = block(regime, rsr)
    done = []
    for snr in SNRS:
        path = OUT/'cells'/f'{cellname(regime, rsr, snr)}.npz'
        if path.exists():
            continue
        obs = observed(a, snr, P=30)
        c = receiver_cache(obs, _EST)
        est = {arm: np.stack([offline(c, i, arm, obs) for i in range(len(SEEDS))]) for arm in ARMS}
        if not np.array_equal(est['A'], c['frozen']):
            raise AssertionError('arm A differs from frozen output')
        err = np.stack([[squared_error(est[arm][i], a['g']) for i in range(len(SEEDS))] for arm in ARMS])
        z, s, b, v = (x.numpy() for x in obs.args())
        data, bits = ber_data(a['g'], b[..., :1], v, regime, rsr)
        ber = np.zeros((len(ARMS), len(SEEDS), COUNT), dtype=np.int64)
        seen = {}
        for ai, arm in enumerate(ARMS):
            for i in range(len(SEEDS)):
                for k in range(COUNT):
                    key = hashlib.sha256(est[arm][i, k].tobytes()).hexdigest()
                    if key not in seen:
                        seen[key] = int(np.count_nonzero(detect(data[k], est[arm][i, k], b[k, :, :1], float(v[k])) != bits))
                    ber[ai, i, k] = seen[key]
        np.savez_compressed(path, arms=np.array(ARMS), err=err, energy=np.sum(np.abs(a['g'])**2, axis=(-2, -1)),
                            ber_errors=ber, steps=np.array(c['steps']), family=np.array(c['family']),
                            nominal_snr=c['nominal_snr'], accepted=c['accepted'], snr=np.array(snr), rsr=np.array(rsr),
                            regime=np.array(regime))
        done.append(path.stem)
    return done


def run(workers):
    (OUT/'cells').mkdir(parents=True, exist_ok=True)
    jobs = [(r, x) for x in RSRS for r in REGIMES]
    tick = time.time()
    with ProcessPoolExecutor(workers, initializer=_init) as pool:
        for names in pool.map(run_block, jobs):
            print(json.dumps(dict(cells=len(names), last=names[-1] if names else None,
                                  seconds=round(time.time() - tick, 1))), flush=True)


def kind(snr, rsr):
    if rsr in GRID_RSRS and snr in GRID_SNRS:
        return 'grid'
    return 'offgrid_snr' if rsr in GRID_RSRS else f'rsr{rsr:g}'


def scopes_of(row):
    out = [row['kind']]
    if row['kind'] == 'grid' and row['snr'] >= 30:
        out.append('grid_30_60')
    if row['snr'] == 31.25:
        out.append('snr31.25')
    return out


def interval(x):
    return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]


def analyze():
    rng = np.random.default_rng(SEED['bootstrap'])
    w = {(r, x): rng.multinomial(COUNT, np.full(COUNT, 1/COUNT), size=BOOT).T.astype(float) for r in REGIMES for x in RSRS}
    pairs = (('C2', 'A'), ('C3', 'A'), ('C3', 'C2'))
    rows, acc, hashes = [], {}, {}
    for rsr in RSRS:
        for regime in REGIMES:
            for snr in SNRS:
                path = OUT/'cells'/f'{cellname(regime, rsr, snr)}.npz'
                hashes[path.stem] = sha(path)
                with np.load(path) as d:
                    err, energy, ber, steps = d['err'], d['energy'], d['ber_errors'], int(d['steps'])
                ww = w[(regime, rsr)]
                point = np.mean(10*np.log10(err.sum(-1)/energy.sum()), axis=1)
                boot = np.mean(10*np.log10((err @ ww)/(energy @ ww)[None, None]), axis=1)
                rate = ber.mean(1)/384.
                bpoint, bboot = rate.mean(1), rate @ ww/COUNT
                ix = {a: i for i, a in enumerate(ARMS)}
                row = dict(cell=path.stem, kind=kind(snr, rsr), regime=regime, rsr=rsr, snr=snr, em_tail=steps,
                           nmse_db={a: float(point[i]) for a, i in ix.items()}, ber={a: float(bpoint[i]) for a, i in ix.items()},
                           diff={f'{x}-{y}': dict(point=float(point[ix[x]] - point[ix[y]]), ci95=interval(boot[ix[x]] - boot[ix[y]]))
                                 for x, y in pairs},
                           ber_diff={f'{x}-{y}': dict(point=float(bpoint[ix[x]] - bpoint[ix[y]]), ci95=interval(bboot[ix[x]] - bboot[ix[y]]))
                                     for x, y in pairs})
                rows.append(row)
                for sc in scopes_of(row):
                    s = acc.setdefault(sc, dict(n=0, boot=0., err=[], energy=[], pts=[]))
                    s['n'] += 1; s['boot'] = s['boot'] + boot; s['pts'].append(point)
                    s['err'].append(err); s['energy'].append(energy)
    aggregates = {}
    for sc, s in acc.items():
        nb, pts = s['boot']/s['n'], np.mean(s['pts'], axis=0)
        e, en = np.concatenate(s['err'], -1), np.concatenate(s['energy'])
        pooled = np.mean(e.sum(-1)/en.sum(), axis=1)
        sel = [r for r in rows if sc in scopes_of(r)]
        aggregates[sc] = dict(cells=s['n'], mean_setting_nmse_db={a: float(pts[i]) for i, a in enumerate(ARMS)},
            pooled_nmse_db={a: float(10*np.log10(pooled[i])) for i, a in enumerate(ARMS)},
            ber={a: float(np.mean([r['ber'][a] for r in sel])) for a in ARMS},
            diff={f'{x}-{y}': dict(point=float(pts[ARMS.index(x)] - pts[ARMS.index(y)]),
                                   ci95=interval(nb[ARMS.index(x)] - nb[ARMS.index(y)])) for x, y in pairs})
    at31 = [r for r in rows if r['snr'] == 31.25]
    q1 = dict(c2_cells_over_1db=[r['cell'] for r in at31 if r['diff']['C2-A']['point'] > 1.],
              c3_cells_over_1db=[r['cell'] for r in at31 if r['diff']['C3-A']['point'] > 1.])
    q1['passed'] = not q1['c3_cells_over_1db']
    worse = [dict(cell=r['cell'], c3_minus_c2=r['diff']['C3-C2']['point'], em_tail=r['em_tail'])
             for r in rows if r['diff']['C3-C2']['point'] > .3]
    grid = [r for r in rows if r['kind'] == 'grid']
    limits = dict(L1_full=aggregates['grid']['diff']['C3-A']['point'], L1_30_60=aggregates['grid_30_60']['diff']['C3-A']['point'],
                  L2_worst=max(grid, key=lambda r: r['diff']['C3-A']['point'])['cell'],
                  L2_worst_db=max(r['diff']['C3-A']['point'] for r in grid),
                  L3_cells=[r['cell'] for r in grid if r['ber_diff']['C3-A']['ci95'][0] > 0],
                  L2_C2_worst_db=max(r['diff']['C2-A']['point'] for r in grid),
                  L3_C2_cells=[r['cell'] for r in grid if r['ber_diff']['C2-A']['ci95'][0] > 0])
    summary = dict(label='POST HOC follow-up', aggregates=aggregates, Q1_31p25=q1,
                   Q2_cells_C3_worse_than_C2_by_over_0p3dB=worse, limits_descriptive=limits, rows=rows,
                   cell_sha256=hashes, bootstrap=dict(draws=BOOT, seed=SEED['bootstrap']))
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/followup_c3_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    figure(rows)
    print(json.dumps(dict(Q1=q1, n_worse=len(worse), worse_examples=worse[:12], limits=limits,
                          scopes={k: v['diff'] for k, v in aggregates.items()}), indent=1))


def figure(rows):
    colors = dict(C2='#eda100', C3='#1baf7a')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, rsr in zip(axes.ravel(), sorted(RSRS)):
        ax.set_facecolor('#fcfcfb')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(color='#e4e3df', linewidth=.6)
        sub = [r for r in rows if r['rsr'] == rsr]
        for arm, marker in (('C2', 'v'), ('C3', '^')):
            vals = [np.mean([r['diff'][f'{arm}-A']['point'] for r in sub if r['snr'] == s]) for s in SNRS]
            ax.plot(SNRS, vals, color=colors[arm], marker=marker, markersize=3.5, linewidth=1.6, label=f'{arm} − A')
        ax.axhline(0, color='#52514e', linewidth=.8)
        ax.axhline(1, color='#52514e', linewidth=.8, linestyle=':')
        for s in OFF_SNRS:
            ax.axvline(s, color='#e4e3df', linewidth=.8, zorder=0)
        ax.set_title(f'RSR {rsr:g} dB', fontsize=9, loc='left')
    for ax in axes[-1]:
        ax.set_xlabel('SNR (dB); light verticals = off-grid SNRs')
    for ax in axes[:, 0]:
        ax.set_ylabel('NMSE − arm A (dB); positive = worse')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .95))
    (HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/followup_c3_vs_c2.png', dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('run', 'analyze'))
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    run(a.workers) if a.phase == 'run' else analyze()


if __name__ == '__main__':
    main()
