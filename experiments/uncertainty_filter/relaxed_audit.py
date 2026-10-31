"""Post-hoc correction audit for the saved Phase 1 records.

The original Phase 1 records stored the conditional uncertainty of G_EM while
scoring the relaxed G_dc.  This script applies the exact gamma^2 factor to
both uncertainty proxies for stages 1--10, using the frozen checkpoints, and
excludes stage 11: its current final-filter input has no corresponding relaxed
EM map.  It never overwrites the pre-registered Phase 1 files.

Run from the repository root:
  .venv/bin/python -m experiments.uncertainty_filter.relaxed_audit
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch

from sahure.train import load_model

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INPUT = HERE/'results/phase1'
OUT = HERE/'results/phase1_relaxed_audit'
SEEDS = (9101, 9102, 9103)
BOOT, BOOT_SEED = 1000, 2026103101
GATE = dict(margin=.05, tolerance=.02)


def gammas():
    """Exact frozen relaxation factors, indexed by receiver family and seed."""
    roots = {'adaptive': REPO/'results/tcom_pilot_em_20261018/checkpoints/context30',
             'global': REPO/'results/tcom_start_ablation_20261020/checkpoints/global'}
    result = {}
    for family, root in roots.items():
        result[family] = {}
        for seed in SEEDS:
            model, _ = load_model(root/f'seed_{seed}'/f'em_{family}_best.pt', dtype=torch.float64)
            result[family][seed] = (2*torch.sigmoid(model.steps.detach())).cpu().numpy()
    return result


def load():
    """Load saved arrays and put both proxies on the relaxed-output target."""
    manifest = json.loads((INPUT/'run_manifest.json').read_text())
    data, gs, rows = np.load(INPUT/'records.npz'), gammas(), []
    for i, cell in enumerate(manifest['cells']):
        for start in ('receiver', 'random'):
            for seed in SEEDS:
                a, b, c = (data[f'{i}/{start}_{seed}/{key}'] for key in 'abc')
                # Stages 1--10 are the relaxed outputs sent to Hankel filters.
                # Stage 11 is an already-produced final-filter input, not G_dc.
                a, b, c = a[:, :10], b[:, :10], c[:, :10]
                factor = gs[cell['family']][seed][None, :, None]**2
                ch, stages, users = a.shape
                index = np.indices((ch, stages, users)).reshape(3, -1)
                rows.append(dict(a=(a*factor).ravel(), b=(b*factor).ravel(), c=c.ravel(),
                    channel=index[0] + 1000*i, stage=index[1] + 1,
                    start=np.full(a.size, start == 'random'), seed=np.full(a.size, seed),
                    snr=np.full(a.size, cell['snr'], float), rsr=np.full(a.size, cell['rsr'], float),
                    regime=np.full(a.size, cell['regime']), family=np.full(a.size, cell['family'])))
    return manifest, gs, {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}


def spearman(x, y, cluster, rng=None):
    rx, ry = rankdata(np.log(x)), rankdata(np.log(y))
    rho = np.corrcoef(rx, ry)[0, 1]
    if rng is None:
        return float(rho)
    _, inv = np.unique(cluster, return_inverse=True)
    count = inv.max() + 1
    stats = np.stack([np.bincount(inv, weights, count)
                      for weights in (np.ones_like(rx), rx, ry, rx*rx, ry*ry, rx*ry)])
    weights = rng.multinomial(count, np.full(count, 1/count), size=BOOT).astype(float)
    n, sx, sy, sxx, syy, sxy = (weights @ stat for stat in stats)
    cov = sxy/n - sx*sy/n**2
    boot = cov/np.sqrt((sxx/n - (sx/n)**2)*(syy/n - (sy/n)**2))
    return float(rho), boot


def compare(records, mask, rng):
    a, b, c, cluster = (records[key][mask] for key in ('a', 'b', 'c', 'channel'))
    good = (a > 0) & (b > 0) & (c > 0)
    a, b, c, cluster = a[good], b[good], c[good], cluster[good]
    state = rng.bit_generator.state
    rho_a, boot_a = spearman(a, c, cluster, rng)
    rng.bit_generator.state = state
    rho_b, boot_b = spearman(b, c, cluster, rng)
    diff = boot_a - boot_b
    interval = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))]
    return dict(n=int(a.size), rho_a=rho_a, rho_b=rho_b, diff=rho_a-rho_b,
                ci_diff=interval(diff), ci_a=interval(boot_a), ci_b=interval(boot_b))


def main():
    manifest, gamma, records = load()
    rng = np.random.default_rng(BOOT_SEED)
    weak, receiver = records['rsr'] == 0, ~records['start']
    main_receiver = ~weak & receiver
    masks = {'overall (receiver start; stages 1--10; RSR 5/12/20)': main_receiver,
             'poor start (stages 1--10; RSR 5/12/20)': ~weak & records['start'],
             'weak reference (receiver start; stages 1--10)': weak & receiver,
             'weak reference (poor start; stages 1--10)': weak & records['start']}
    masks.update({f'stage {stage}': main_receiver & (records['stage'] == stage) for stage in range(1, 11)})
    masks.update({f'SNR {snr:g}': main_receiver & (records['snr'] == snr)
                  for snr in sorted(set(records['snr']))})
    masks.update({f'RSR {rsr:g}': main_receiver & (records['rsr'] == rsr)
                  for rsr in (5., 12., 20.)})
    groups = {name: compare(records, mask, rng) for name, mask in masks.items()}
    overall = groups['overall (receiver start; stages 1--10; RSR 5/12/20)']
    better = overall['diff'] >= GATE['margin'] and overall['ci_diff'][0] > 0
    worse = [name for name, value in groups.items() if name != 'overall (receiver start; stages 1--10; RSR 5/12/20)'
             and value['rho_a'] < value['rho_b'] - GATE['tolerance']]
    summary = dict(
        status='post_hoc_correction_audit_not_a_new_selection_experiment',
        target='conditional noise-driven MSE of the relaxed output G_dc at stages 1--10',
        excluded='Stage 11 final-filter input: the saved u_k predicts a new EM map, not that existing estimate.',
        correction='Both posterior and nominal MSE proxies multiplied by the exact frozen gamma_t^2.',
        input_records=int(2*len(manifest['cells'])*128*11*3*len(SEEDS)),
        analysed_records=int(records['a'].size),
        per_start_analysed_records=int(records['a'].size//2),
        gamma={family: {str(seed): values.tolist() for seed, values in by_seed.items()}
               for family, by_seed in gamma.items()},
        gate_1=dict(clearly_better_overall=bool(better), groups_worse=worse,
                    passed=bool(better and not worse), criteria=GATE),
        groups=groups,
        provenance=dict(source_records='results/phase1/records.npz', source_manifest='results/phase1/run_manifest.json',
                        source_sha256=manifest['source_sha256'], bootstrap_samples=BOOT, bootstrap_seed=BOOT_SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(gate_1=summary['gate_1'], overall=overall,
                          analysed_records=summary['analysed_records']), indent=2))


if __name__ == '__main__':
    main()
