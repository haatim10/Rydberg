"""Phase 1 statistics, gate verdict and figures from results/phase1/*.npz.

Spearman rho is Pearson correlation of ranks. The bootstrap resamples
clusters (cell, channel) with multinomial weights on the fixed full-sample
ranks, which is the usual fast approximation for large samples.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE/'results/phase1'
CONFIG = json.loads((HERE/'configs/phase1.json').read_text())
GATE = dict(margin=.05, tolerance=.02)
BOOT = 1000
BOOT_SEED = 2026103013
SEEDS = CONFIG['model_seeds']
STAGES = 11
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def load():
    manifest = json.loads((OUT/'run_manifest.json').read_text())
    d = np.load(OUT/'records.npz')
    rows = []
    for i, cell in enumerate(manifest['cells']):
        for start in ('receiver', 'random'):
            for seed in SEEDS:
                a, b, c = (d[f'{i}/{start}_{seed}/{k}'] for k in 'abc')      # [ch, 11, K]
                ch, st, k = a.shape
                idx = np.indices((ch, st, k)).reshape(3, -1)
                rows.append(dict(a=a.ravel(), b=b.ravel(), c=c.ravel(), channel=idx[0] + 1000*i, stage=idx[1] + 1,
                    user=idx[2], cell=np.full(a.size, i), seed=np.full(a.size, seed),
                    start=np.full(a.size, start == 'random'), snr=np.full(a.size, cell['snr'], float),
                    rsr=np.full(a.size, cell['rsr'], float), regime=np.full(a.size, cell['regime'])))
    return manifest, {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def spearman(x, y, cluster, rng=None):
    rx, ry = rankdata(np.log(x)), rankdata(np.log(y))
    rho = np.corrcoef(rx, ry)[0, 1]
    if rng is None:
        return float(rho)
    _, inv = np.unique(cluster, return_inverse=True)
    m = inv.max() + 1
    stats = np.stack([np.bincount(inv, w, m) for w in (np.ones_like(rx), rx, ry, rx*rx, ry*ry, rx*ry)])
    w = rng.multinomial(m, np.full(m, 1/m), size=BOOT).astype(float)
    n, sx, sy, sxx, syy, sxy = (w @ s for s in stats)
    cov = sxy/n - sx*sy/n**2
    boot = cov/np.sqrt((sxx/n - (sx/n)**2)*(syy/n - (sy/n)**2))
    return float(rho), boot


def compare(r, mask, rng):
    """rho_a, rho_b, difference and bootstrap intervals for one record group."""
    a, b, c, cl = r['a'][mask], r['b'][mask], r['c'][mask], r['channel'][mask]
    ok = (a > 0) & (b > 0) & (c > 0)
    a, b, c, cl = a[ok], b[ok], c[ok], cl[ok]
    # Same resampled clusters for both correlations: share the generator state.
    state = rng.bit_generator.state
    rho_a, boot_a = spearman(a, c, cl, rng)
    rng.bit_generator.state = state
    rho_b, boot_b = spearman(b, c, cl, rng)
    diff = boot_a - boot_b
    q = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))]
    return dict(n=int(len(a)), rho_a=rho_a, rho_b=rho_b, diff=rho_a - rho_b, ci_a=q(boot_a), ci_b=q(boot_b), ci_diff=q(diff),
                ratio_a_median=float(np.median(a/c)), ratio_b_median=float(np.median(b/c)),
                ratio_a_geomean=float(np.exp(np.mean(np.log(a/c)))), ratio_b_geomean=float(np.exp(np.mean(np.log(b/c)))))


def bias_fractions(manifest):
    d = np.load(OUT/'bias.npz')
    out = {}
    for i, cell in enumerate(manifest['bias_cells']):
        bias = sum(d[f'{i}/bias_{s}'] for s in SEEDS).sum((0, 2))
        total = sum(d[f'{i}/total_{s}'] for s in SEEDS).sum((0, 2))
        out[f"{cell['regime']}_rsr{cell['rsr']:g}_snr{cell['snr']:g}"] = dict(cell, bias=bias.tolist(), total=total.tolist())
    pooled = {}
    for label, select in (('all', lambda c: True), *[(f'snr{s:g}', lambda c, s=s: c['snr'] == s) for s in CONFIG['snrs_db']],
                          *[(f'rsr{s:g}', lambda c, s=s: c['rsr'] == s) for s in CONFIG['rsrs_db']]):
        cells = [v for v in out.values() if select(v)]
        # Each cell's bias/total ratio per stage, then averaged across cells (cells differ by orders of magnitude).
        ratios = np.array([np.array(v['bias'])/np.array(v['total']) for v in cells])
        pooled[label] = dict(mean_over_cells=ratios.mean(0).tolist(), median_over_cells=np.median(ratios, 0).tolist())
    return out, pooled


def main():
    manifest, r = load()
    rng = np.random.default_rng(BOOT_SEED)
    weak = r['rsr'] == 0
    main_recv = ~weak & ~r['start']
    groups = {'overall (receiver start, RSR 5/12/20)': main_recv,
              'poor start (random init, RSR 5/12/20)': ~weak & r['start'],
              'weak reference (RSR 0, receiver start)': weak & ~r['start'],
              'weak reference (RSR 0, poor start)': weak & r['start']}
    for s in range(1, STAGES + 1):
        groups[f'stage {s}'] = main_recv & (r['stage'] == s)
    for x in CONFIG['rsrs_db']:
        groups[f'RSR {x:g}'] = main_recv & (r['rsr'] == x)
    for x in CONFIG['regimes']:
        groups[f'paths {x}'] = main_recv & (r['regime'] == x)
    for x in CONFIG['snrs_db']:
        groups[f'SNR {x:g}'] = main_recv & (r['snr'] == x)
    results = {name: compare(r, m, rng) for name, m in groups.items()}

    per_cell = {}
    for i, cell in enumerate(manifest['cells']):
        for start in (False, True):
            m = (r['cell'] == i) & (r['start'] == start)
            name = f"{cell['regime']}_rsr{cell['rsr']:g}_snr{cell['snr']:g}_{'random' if start else 'receiver'}"
            per_cell[name] = dict(cell, start='random' if start else 'receiver',
                                  rho_a=spearman(r['a'][m], r['c'][m], None), rho_b=spearman(r['b'][m], r['c'][m], None))
    # Within (cell, stage): does u_k carry information beyond the cell-level noise scale?
    within = []
    for i in range(len(manifest['cells'])):
        for s in range(1, STAGES + 1):
            m = (r['cell'] == i) & (r['stage'] == s) & ~r['start']
            within.append((spearman(r['a'][m], r['c'][m], None), spearman(r['b'][m], r['c'][m], None)))
    within = np.array(within)
    bias_cells, bias_pooled = bias_fractions(manifest)

    overall = results['overall (receiver start, RSR 5/12/20)']
    clearly_better = overall['diff'] >= GATE['margin'] and overall['ci_diff'][0] > 0
    major = [n for n in results if not n.startswith('overall')]
    worse = [n for n in major if results[n]['rho_a'] < results[n]['rho_b'] - GATE['tolerance']]
    verdict = dict(clearly_better_overall=bool(clearly_better), groups_worse=worse,
                   passed=bool(clearly_better and not worse), criteria=CONFIG['gate_1'])
    summary = dict(gate_1=verdict, groups=results, per_cell=per_cell,
                   within_cell_stage=dict(median_rho_a=float(np.nanmedian(within[:, 0])), median_rho_b=float(np.nanmedian(within[:, 1])),
                                          note='Spearman within each (cell, stage) group, receiver start; nominal varies only through S.'),
                   bias_fraction=dict(pooled=bias_pooled, cells=bias_cells,
                                      definition=CONFIG['bias_subset']),
                   records=int(len(r['a'])), max_parity=manifest['max_parity'])
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    figures(r, results, bias_pooled, manifest)
    print(json.dumps(dict(gate_1=verdict, overall=overall), indent=2))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=.6)
    ax.set_axisbelow(True)


def figures(r, results, bias_pooled, manifest):
    fig_dir = HERE/'figures'
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK, 'figure.facecolor': SURFACE})

    # Calibration scatter: predicted vs actual, receiver start, subsample.
    m = (r['rsr'] > 0) & ~r['start']
    idx = np.random.default_rng(0).choice(np.flatnonzero(m), 20000, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('stage', ['#b7d3f2', '#0d3b73'])
    for ax, key, title in ((axes[0], 'a', '(a) predicted uncertainty $u_k$'), (axes[1], 'b', r'(b) nominal scale $v\,\mathrm{tr}[(SS^H)^{-1}]/K$')):
        style(ax)
        sc = ax.scatter(np.log10(r[key][idx]), np.log10(r['c'][idx]), c=r['stage'][idx], cmap=cmap, s=4, alpha=.5, linewidths=0)
        lo, hi = -9, 1
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle='--')
        ax.text(hi - .2, hi - .6, 'y = x', color=MUTED, ha='right', fontsize=8)
        g = results['overall (receiver start, RSR 5/12/20)']
        ax.set_title(f"{title}\nSpearman ρ = {g['rho_' + key]:.3f}", fontsize=10, loc='left')
        ax.set_xlabel('log10 predicted / nominal')
    axes[0].set_ylabel('log10 actual error ||g_dc,k − g_k||² / N')
    cb = fig.colorbar(sc, ax=axes, shrink=.8, pad=.02)
    cb.set_label('stage (11 = final-filter input)', color=INK)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    fig.savefig(fig_dir/'phase1_calibration_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Rank correlation by group: dot plot, u_k vs nominal.
    names = [n for n in results if not n.startswith('SNR')]
    fig, ax = plt.subplots(figsize=(7, 7.5))
    style(ax)
    y = np.arange(len(names))[::-1]
    for key, color, label, off in (('a', BLUE, '(a) u_k', .15), ('b', ORANGE, '(b) nominal', -.15)):
        vals = np.array([results[n]['rho_' + key] for n in names])
        ci = np.array([results[n]['ci_' + key] for n in names])
        ax.errorbar(vals, y + off, xerr=[vals - ci[:, 0], ci[:, 1] - vals], fmt='o', color=color, markersize=5,
                    elinewidth=1.2, capsize=0, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Spearman ρ with actual error (95% cluster-bootstrap interval)')
    ax.legend(loc='lower left', frameon=False, fontsize=9)
    fig.savefig(fig_dir/'phase1_rank_correlation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Bias fraction by stage for each SNR (pooled over RSR/paths).
    fig, ax = plt.subplots(figsize=(7, 4))
    style(ax)
    snrs = CONFIG['snrs_db']
    ramp = cmap(np.linspace(0, 1, len(snrs)))
    stages = np.arange(1, STAGES + 1)
    for colour, s in zip(ramp, snrs):
        ax.plot(stages, bias_pooled[f'snr{s:g}']['mean_over_cells'], color=colour, linewidth=2, marker='o', markersize=4)
    for s in (snrs[0], snrs[-1]):
        ax.annotate(f'{s:g} dB', (STAGES, bias_pooled[f'snr{s:g}']['mean_over_cells'][-1]), xytext=(6, 0),
                    textcoords='offset points', fontsize=8, color=INK, va='center')
    ax.set_xticks(stages)
    ax.set_xlabel('stage (11 = final-filter input)')
    ax.set_ylabel('fraction of actual error that is bias')
    ax.set_ylim(-.05, 1.05)
    ax.set_title('Bias fraction by stage; one line per SNR (light 0 dB → dark 50 dB)', fontsize=10, loc='left')
    fig.savefig(fig_dir/'phase1_bias_fraction.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
