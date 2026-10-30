"""Paired, conditional NMSE analysis of the frozen retrained bypass study."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
os.environ['MPLCONFIGDIR'] = '/private/tmp/tcom_retrained_bypass_matplotlib'
import json
from pathlib import Path
import numpy as np
from .study import OUT, SNRS, RSRS, REGIMES, SEEDS, NAMES, load, cellname, check_selection
from tcom_followups_20261017.common import sha, write_json

BOOT = 20000
BOOT_SEED = 2026101829
COUNT = 64
REPORT_TOLERANCE_DB = 1e-10  # Numerical ties only; no selection or inference changes.
GROUPS = {f'{prefix}_mean': [f'{prefix}_{seed}' for seed in SEEDS]
          for prefix in ('original', 'off', 'fast_bypass', 'em100_bypass')}
METHODS = list(NAMES) + list(GROUPS)
INDEX = {name: i for i, name in enumerate(METHODS)}
CONTRASTS = {'bypass_vs_original': ('original_mean', 'em100_bypass_mean'),
             'bypass_vs_em100': ('em100', 'em100_bypass_mean'),
             'bypass_vs_compact': ('compact_sparse', 'em100_bypass_mean')}
REGIONS = ('ordinary', 'transition', 'stress', 'high_snr', 'all')


def belongs(region, snr):
    return {'ordinary': snr <= 20, 'transition': 20 < snr <= 45,
            'stress': snr >= 50, 'high_snr': snr > 20, 'all': True}[region]


def expand(values):
    return np.concatenate([values, np.stack([values[[INDEX[n] for n in names]].mean(0)
                                             for names in GROUPS.values()])])


def intervals(values, primary=False):
    result = {'pointwise_ci95_db': np.quantile(values, [.025, .975]).tolist()}
    if primary:
        result['bonferroni6_interval_db'] = np.quantile(values, [.05 / 12, 1 - .05 / 12]).tolist()
    return result


def run():
    selection, freeze = check_selection()
    rng = np.random.default_rng(BOOT_SEED)
    weights = {regime: rng.multinomial(COUNT, np.full(COUNT, 1 / COUNT), size=BOOT).T.astype(float)
               for regime, _ in REGIMES}
    sums = {r: np.zeros((len(METHODS), BOOT)) for r in REGIONS}
    rows, points, errors, energies, hashes = [], [], [], [], {}
    maximum_dispatch_error = 0.
    for regime, _ in REGIMES:
        for rsr in RSRS:
            for snr in SNRS:
                path = OUT / 'test/cells' / f'{cellname(regime, rsr, snr)}.npz'
                cell = load(path)
                assert str(cell['selection_sha256']) == freeze
                assert tuple(map(str, cell['method_names'])) == tuple(NAMES)
                err = np.sum(abs(cell['estimates'] - cell['truth_g'][None]) ** 2, axis=(-2, -1))
                energy = np.sum(abs(cell['truth_g']) ** 2, axis=(-2, -1))
                np.testing.assert_array_equal(err, cell['errors'])
                np.testing.assert_array_equal(energy, cell['truth_energy'])
                assert err.shape == (len(NAMES), COUNT)
                if snr <= 20:
                    for seed in SEEDS:
                        original = cell['estimates'][INDEX[f'original_{seed}']]
                        for prefix in ('fast_bypass', 'em100_bypass'):
                            assert np.array_equal(original, cell['estimates'][INDEX[f'{prefix}_{seed}']])
                point = expand(10 * np.log10(err.sum(1) / energy.sum()))
                boot = expand(10 * np.log10((err @ weights[regime]) / (energy @ weights[regime])[None]))
                nm = err / energy[None]
                p95, p99 = (expand(np.quantile(nm, q, axis=1)) for q in (.95, .99))
                severe = expand((nm > 1).mean(1))
                method_rows = {name: dict(nmse_db=float(point[i]),
                    conditional_ci95_db=np.quantile(boot[i], [.025, .975]).tolist(),
                    case_nmse_p95=float(p95[i]), case_nmse_p99=float(p99[i]), severe_rate=float(severe[i]))
                    for name, i in INDEX.items()}
                gains = {key: dict(gain_db=float(point[INDEX[ref]] - point[INDEX[candidate]]),
                    **intervals(boot[INDEX[ref]] - boot[INDEX[candidate]]))
                    for key, (ref, candidate) in CONTRASTS.items()}
                fast = dict(gain_db=float(point[INDEX['original_mean']] - point[INDEX['fast_bypass_mean']]),
                    **intervals(boot[INDEX['original_mean']] - boot[INDEX['fast_bypass_mean']]))
                rows.append(dict(cell=path.stem, regime=regime, rsr=rsr, snr=snr, cases=COUNT,
                                 methods=method_rows, contrasts=gains, fast_bypass_vs_original=fast))
                points.append(point); errors.append(err); energies.append(energy); hashes[path.name] = sha(path)
                for region in REGIONS:
                    if belongs(region, snr): sums[region] += boot
                if 'actual_dispatch_max_abs_error' in cell:
                    maximum_dispatch_error = max(maximum_dispatch_error, float(cell['actual_dispatch_max_abs_error']))
    points = np.stack(points)
    aggregates = {}
    aggregate_boots = {}
    for region in REGIONS:
        keep = np.array([belongs(region, row['snr']) for row in rows])
        scores = points[keep].mean(0)
        boot = sums[region] / keep.sum()
        aggregate_boots[region] = boot
        err = np.concatenate([a for a, use in zip(errors, keep) if use], axis=1)
        energy = np.concatenate([a for a, use in zip(energies, keep) if use])
        nm = err / energy[None]
        pooled = expand(10 * np.log10(err.sum(1) / energy.sum()))
        p95, p99 = (expand(np.quantile(nm, q, axis=1)) for q in (.95, .99))
        severe = expand((nm > 1).mean(1))
        method_rows = {name: dict(nmse_db=float(scores[i]),
            conditional_ci95_db=np.quantile(boot[i], [.025, .975]).tolist(),
            pooled_ratio_of_sums_db=float(pooled[i]), case_nmse_p95=float(p95[i]),
            case_nmse_p99=float(p99[i]), severe_rate=float(severe[i])) for name, i in INDEX.items()}
        contrasts = {}
        for key, (ref, candidate) in CONTRASTS.items():
            seed_gains = {}
            for seed in SEEDS:
                source = f'original_{seed}' if ref == 'original_mean' else ref
                seed_gains[str(seed)] = float(scores[INDEX[source]] - scores[INDEX[f'em100_bypass_{seed}']])
            contrasts[key] = dict(gain_db=float(scores[INDEX[ref]] - scores[INDEX[candidate]]),
                seed_specific_gains_db=seed_gains,
                **intervals(boot[INDEX[ref]] - boot[INDEX[candidate]], primary=region == 'high_snr'))
        fast = dict(gain_db=float(scores[INDEX['original_mean']] - scores[INDEX['fast_bypass_mean']]),
            **intervals(boot[INDEX['original_mean']] - boot[INDEX['fast_bypass_mean']]))
        aggregates[region] = dict(cells=int(keep.sum()), condition_cases=int(keep.sum()) * COUNT,
                                 methods=method_rows, contrasts=contrasts, fast_bypass_vs_original=fast)
    negative = {}
    for key in CONTRASTS:
        worst = min(rows, key=lambda row: row['contrasts'][key]['gain_db'])
        negative[key] = dict(negative_cells=[r['cell'] for r in rows if r['contrasts'][key]['gain_db'] < -REPORT_TOLERANCE_DB],
            pointwise95_harm_cells=[r['cell'] for r in rows if r['contrasts'][key]['pointwise_ci95_db'][1] < -REPORT_TOLERANCE_DB],
            worst_cell=worst['cell'], worst_gain_db=worst['contrasts'][key]['gain_db'])
    fast_worst = min(rows, key=lambda row: row['fast_bypass_vs_original']['gain_db'])
    negative['fast_bypass_vs_original'] = dict(
        negative_cells=[r['cell'] for r in rows if r['fast_bypass_vs_original']['gain_db'] < -REPORT_TOLERANCE_DB],
        pointwise95_harm_cells=[r['cell'] for r in rows if r['fast_bypass_vs_original']['pointwise_ci95_db'][1] < -REPORT_TOLERANCE_DB],
        worst_cell=fast_worst['cell'], worst_gain_db=fast_worst['fast_bypass_vs_original']['gain_db'])
    result = dict(selection_sha256=freeze, protocol_sha256=selection['protocol_sha256'],
        policies=selection['policies'], unique_test_anchors=COUNT * len(RSRS) * len(REGIMES),
        condition_cases=len(rows) * COUNT, bootstrap_index_clusters=COUNT * len(REGIMES),
        ordinary_exact_identity=True, actual_dispatch_max_abs_error=maximum_dispatch_error,
        scoring='Equal-cell mean of dB ratio-of-sums NMSE; groups average three separate fit scores, not estimates. Pooled NMSE and per-fit-averaged tail quantiles are diagnostic.',
        primary='Three contrasts on SNR>20: EM100 bypass versus retrained model, direct EM100, compact matched-budget sparse. NMSE and BER are the six prespecified outcomes.',
        bootstrap=dict(samples=BOOT, seed=BOOT_SEED,
            unit='Multinomial whole-anchor index; shared across all SNR/reference/methods/fits within each path regime.',
            scope='Conditional on these fitted models, selected policies, compact sparse choices and condition grid; excludes training-population and selection uncertainty.',
            multiplicity='Bonferroni6 intervals for high-SNR primary contrasts; all per-cell, ordinary, transition, stress and fast-bypass intervals are descriptive pointwise95.'),
        aggregates=aggregates, negative_cells=negative, numerical_tie_tolerance_db=REPORT_TOLERANCE_DB,
        cells=rows, test_hashes=hashes,
        source_sha256=sha(Path(__file__)))
    timing_path = OUT / 'timing.json'
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        assert timing['selection_sha256'] == freeze
        result.update(timing=timing['methods'], timing_sha256=sha(timing_path))
    write_json(OUT / 'nmse_summary.json', result)
    np.savez_compressed(OUT / 'nmse_bootstrap.npz', method_names=np.array(METHODS),
        regions=np.array(REGIONS), aggregate_bootstraps=np.stack([aggregate_boots[r] for r in REGIONS]),
        cell_names=np.array([r['cell'] for r in rows]), cell_scores=points,
        selection_sha256=np.array(freeze))
    figures(result)
    print(json.dumps(dict(policies=result['policies'], high_snr=aggregates['high_snr']['contrasts'],
                          negative_cells=negative)), flush=True)
    return result


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {
        'original_mean': ('#d73a49', 'Retrained model', '-'),
        'em100_bypass_mean': ('#0969da', 'Retrained + EM100 bypass', '-'),
        'em100': ('#26965c', 'Always exact EM100', '--'),
        'compact_sparse': ('#b07616', 'Fresh compact sparse', '-.'),
        'prior_sparse_full': ('#777777', 'Frozen full sparse', ':'),
        'optshrink': ('#8a63d2', 'Frozen OptShrink', '--'),
    }
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 10.6), sharex=True, sharey=True)
    for i, (regime, _) in enumerate(REGIMES):
        for j, rsr in enumerate(RSRS):
            ax = axes[i, j]
            rr = sorted([r for r in result['cells'] if r['regime'] == regime and r['rsr'] == rsr], key=lambda r: r['snr'])
            for name, (color, label, ls) in styles.items():
                ax.plot([r['snr'] for r in rr], [r['methods'][name]['nmse_db'] for r in rr],
                        color=color, label=label, ls=ls, marker='.', lw=1.5)
            cutoff = result['policies']['em100'][f'{rsr:g}']
            ax.axvline(cutoff, color='#0969da', alpha=.5, ls=':', lw=.9)
            ax.set_title(f'{regime}, reference {rsr:g} dB; cutoff {cutoff:g} dB')
            ax.grid(alpha=.2)
            if i == 2: ax.set_xlabel('Nominal SNR (dB)')
            if j == 0: ax.set_ylabel('Channel NMSE (dB; lower is better)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('Fresh test of the retrained estimator and calibrated EM100 bypass\nSame channel and standardized noise across SNR; unchanged outputs through 20 dB')
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(OUT / 'bypass_nmse.png', dpi=180); fig.savefig(OUT / 'bypass_nmse.svg'); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True)
    for row_index, (key, label) in enumerate((('bypass_vs_original', 'EM100 bypass'), ('fast_bypass_vs_original', 'Ten-step filter-off bypass'))):
        for ax, (regime, _) in zip(axes[row_index], REGIMES):
            for rsr in RSRS:
                rr = sorted([r for r in result['cells'] if r['regime'] == regime and r['rsr'] == rsr], key=lambda r: r['snr'])
                data = [r['contrasts'][key] if key in CONTRASTS else r[key] for r in rr]
                xs = [r['snr'] for r in rr]
                line, = ax.plot(xs, [r['gain_db'] for r in data], marker='.', label=f'Reference {rsr:g} dB')
                ax.fill_between(xs, [r['pointwise_ci95_db'][0] for r in data], [r['pointwise_ci95_db'][1] for r in data], color=line.get_color(), alpha=.12)
            ax.axhline(0, color='black', lw=.8); ax.axvline(20, color='.5', ls=':')
            ax.grid(alpha=.2); ax.set_title(f'{regime}: {label}')
            if row_index == 1: ax.set_xlabel('Nominal SNR (dB)')
        axes[row_index, 0].set_ylabel('Gain over retrained model (dB)')
    axes[0, -1].legend(fontsize=8)
    fig.suptitle('Gains and adverse cells under frozen development-selected policies\nBands: descriptive pointwise conditional 95% intervals')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / 'bypass_gain.png', dpi=180); fig.savefig(OUT / 'bypass_gain.svg'); plt.close(fig)


def attach_timing():
    """Append final latency without repeating the unchanged accuracy analysis."""
    selection, digest = check_selection()
    summary_path = OUT / 'nmse_summary.json'
    result = json.loads(summary_path.read_text())
    timing = json.loads((OUT / 'timing.json').read_text())
    assert result['selection_sha256'] == timing['selection_sha256'] == digest
    result.update(timing=timing['methods'], timing_sha256=sha(OUT / 'timing.json'),
                  timing_analysis_source_sha256=sha(Path(__file__)))
    write_json(summary_path, result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    samples = dict(timing['raw_samples'])
    samples.update({name: [r for part in parts for r in samples[part]] for name, parts in GROUPS.items()})
    methods = [('original_mean', 'Retrained model'), ('em100_bypass_mean', 'EM100 bypass'),
        ('fast_bypass_mean', 'Ten-step bypass'), ('off_mean', 'Always filter off'),
        ('em100', 'Always exact EM100'), ('compact_sparse', 'Fresh compact sparse'),
        ('prior_sparse_matched', 'Frozen matched sparse'), ('prior_sparse_50ms', 'Frozen 50 ms sparse'),
        ('prior_sparse_full', 'Frozen full sparse'), ('optshrink', 'Frozen OptShrink')]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.1), sharey=True)
    for ax, region, title in zip(axes, ('ordinary', 'transition', 'stress'),
            ('Ordinary: 0–20 dB', 'Transition: 22.5–45 dB', 'Stress: 50 and 60 dB')):
        values = [np.quantile([r['ms'] for r in samples[name] if belongs(region, r['snr'])], .9)
                  for name, _ in methods]
        colors = ['#0969da' if name == 'em100_bypass_mean' else '#d98324' if name == 'compact_sparse' else '#8c9dab'
                  for name, _ in methods]
        bars = ax.barh(np.arange(len(methods)), values, color=colors, height=.62)
        for bar, value in zip(bars, values):
            ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, f'{value:.2f}', va='center', fontsize=8)
        ax.set_xscale('log'); ax.set_xlim(.1, max(values) * 2.5)
        ax.set_title(title); ax.set_xlabel('Batch-one p90 time (ms; log scale)'); ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(np.arange(len(methods)), [label for _, label in methods]); axes[0].invert_yaxis()
    fig.suptitle('Fresh isolated end-to-end runtime on one CPU thread\nInitialization, observable lookup and one selected branch included')
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / 'bypass_runtime.png', dpi=180); fig.savefig(OUT / 'bypass_runtime.svg'); plt.close(fig)
    print('Final timing attached; runtime plot saved', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timing-only', action='store_true')
    args = parser.parse_args()
    attach_timing() if args.timing_only else run()
