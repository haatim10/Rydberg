"""Step 5: scoring, acceptance limits, oracle analysis, figures and summary.json.

  .venv/bin/python -m experiments.final_filter_single_c.analysis
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import CACHE, HERE, RESULTS, PREREG, SEEDS, REGIMES, GRID_SNRS, GRID_RSRS, OFF_SNRS, OFF_RSRS, \
    WEAK_RSRS, COUNT, SEED, NOISE_C, cellname, sha
from .test import TEST_SNRS, TEST_RSRS, verify_frozen

BOOT = 20000
L1_DB, L2_DB, CAP_MS = .3, 1., 10.74
COLORS = dict(A='#2a78d6', B='#eb6834', C1='#1baf7a', C2='#eda100', D='#e87ba4', O='#6250d6')
MARKERS = dict(A='o', B='s', C1='^', C2='v', D='D', O='x')
INK, MUTED, GRID_INK, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def load_cells():
    cells = []
    for rsr in TEST_RSRS:
        for regime in REGIMES:
            for snr in TEST_SNRS:
                path = CACHE/'test/cells'/f'{cellname(regime, rsr, snr)}.npz'
                with np.load(path) as d:
                    cells.append(dict({k: d[k] for k in d.files}, path=path, name=path.stem))
    assert len(cells) == 396
    return cells


def kind(cell):
    snr, rsr = float(cell['snr']), float(cell['rsr'])
    if rsr in GRID_RSRS and snr in GRID_SNRS:
        return 'grid'
    if rsr in GRID_RSRS:
        return 'offgrid_snr'
    return f'rsr{rsr:g}'


def weights():
    rng = np.random.default_rng(SEED['bootstrap'])
    return {(r, x): rng.multinomial(COUNT, np.full(COUNT, 1/COUNT), size=BOOT).T.astype(float)
            for r in REGIMES for x in TEST_RSRS}


def cell_metrics(cell, w):
    err, energy = cell['err'], cell['energy']                          # [arm, fit, case], [case]
    point = np.mean(10*np.log10(err.sum(-1)/energy.sum()), axis=1)     # [arm]
    boot = np.mean(10*np.log10((err @ w)/(energy @ w)[None, None]), axis=1)   # [arm, BOOT]
    rate = cell['ber_errors'].mean(1)/float(cell['bits_per_case'])      # [arm, case] mean over fits
    bpoint = rate.mean(1)
    bboot = rate @ w/COUNT
    pooled = np.mean(err.sum(-1)/energy.sum(), axis=1)                 # linear, mean over fits
    return point, boot, bpoint, bboot, pooled


def interval(x):
    return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]


def main():
    choice, digest = verify_frozen()
    cells = load_cells()
    names = tuple(map(str, cells[0]['arm_names']))
    ia = names.index('A')
    w = weights()
    rows = []
    scope_boot = {}
    for cell in cells:
        point, boot, bpoint, bboot, pooled = cell_metrics(cell, w[(str(cell['regime']), float(cell['rsr']))])
        row = dict(cell=cell['name'], kind=kind(cell), regime=str(cell['regime']), rsr=float(cell['rsr']),
                   snr=float(cell['snr']), table_c=float(cell['table_c']), oracle_c=float(cell['oracle_c']),
                   high_route=bool(cell['high_route']), family=str(cell['family']), em_tail=int(cell['steps']),
                   recovery_accepted=int(cell['accepted'].sum()), parity_max_abs=float(cell['parity_max_abs']),
                   nmse_db={n: float(point[i]) for i, n in enumerate(names)},
                   pooled_linear={n: float(pooled[i]) for i, n in enumerate(names)},
                   ber={n: float(bpoint[i]) for i, n in enumerate(names)},
                   nmse_diff_vs_A={n: dict(point=float(point[i] - point[ia]), ci95=interval(boot[i] - boot[ia]))
                                   for i, n in enumerate(names) if n != 'A'},
                   ber_diff_vs_A={n: dict(point=float(bpoint[i] - bpoint[ia]), ci95=interval(bboot[i] - bboot[ia]))
                                  for i, n in enumerate(names) if n != 'A'},
                   grid_nmse_db={f'{c:g}': float(np.mean(10*np.log10(cell['grid_err'][j].sum(-1)/cell['energy'].sum())))
                                 for j, c in enumerate(NOISE_C)})
        rows.append(row)
        for scope in scopes_of(row):
            acc = scope_boot.setdefault(scope, dict(n=0, nmse=0., ber=0., err=[], energy=[]))
            acc['n'] += 1
            acc['nmse'] = acc['nmse'] + boot
            acc['ber'] = acc['ber'] + bboot
            acc['err'].append(cell['err'])
            acc['energy'].append(cell['energy'])

    aggregates = {}
    for scope, acc in scope_boot.items():
        sel = [r for r in rows if scope in scopes_of(r)]
        err = np.concatenate(acc['err'], axis=-1)
        energy = np.concatenate(acc['energy'])
        pooled = np.mean(err.sum(-1)/energy.sum(), axis=1)
        nb, bb = acc['nmse']/acc['n'], acc['ber']/acc['n']
        aggregates[scope] = dict(cells=acc['n'], arms={n: dict(
            mean_setting_nmse_db=float(np.mean([r['nmse_db'][n] for r in sel])),
            pooled_nmse_linear=float(pooled[i]), pooled_nmse_db=float(10*np.log10(pooled[i])),
            ber=float(np.mean([r['ber'][n] for r in sel])),
            nmse_diff_vs_A=None if n == 'A' else dict(
                point=float(np.mean([r['nmse_diff_vs_A'][n]['point'] for r in sel])), ci95=interval(nb[i] - nb[ia])),
            ber_diff_vs_A=None if n == 'A' else dict(
                point=float(np.mean([r['ber_diff_vs_A'][n]['point'] for r in sel])), ci95=interval(bb[i] - bb[ia])))
            for i, n in enumerate(names)})

    timing = {w_: json.loads((RESULTS/f'timing_window{w_}.json').read_text())['summary'] for w_ in (1, 2)}
    a_cap = all(timing[w_]['A']['absolute_cap_pass'] for w_ in (1, 2))
    verdicts = {}
    for n in [x for x in names if x not in ('A', 'O')]:
        grid = [r for r in rows if r['kind'] == 'grid']
        high = [r for r in grid if r['snr'] >= 30]
        l1_full = aggregates['grid']['arms'][n]['nmse_diff_vs_A']['point']
        l1_high = aggregates['grid_30_60']['arms'][n]['nmse_diff_vs_A']['point']
        worst = max(grid, key=lambda r: r['nmse_diff_vs_A'][n]['point'])
        ber_bad = [r['cell'] for r in grid if r['ber_diff_vs_A'][n]['ci95'][0] > 0]
        t = {w_: timing[w_].get(n) for w_ in (1, 2)}
        if t[1] is None:
            l4 = dict(passed=None, note='not timed (no deployable call)')
        elif a_cap:
            ok = all(t[w_]['full_grid']['p90_ms'] <= CAP_MS and t[w_]['target_range']['p90_ms'] <= CAP_MS for w_ in (1, 2))
            l4 = dict(passed=bool(ok), rule='absolute 10.74 ms cap')
        else:
            ok = all(t[w_][s]['p90_ms'] <= 1.02*timing[w_]['A'][s]['p90_ms']
                     for w_ in (1, 2) for s in ('full_grid', 'target_range'))
            l4 = dict(passed=bool(ok), rule='pre-registered substitute: p90 <= 1.02 x arm A p90 (A fails the absolute cap on this machine)',
                      absolute_cap_pass=bool(all(t[w_]['full_grid']['p90_ms'] <= CAP_MS and t[w_]['target_range']['p90_ms'] <= CAP_MS for w_ in (1, 2))))
        secondary = {}
        for k in ('offgrid_snr', 'rsr8', 'rsr16', 'rsr0'):
            sub = [r for r in rows if r['kind'] == k]
            secondary[k] = dict(cells=len(sub),
                worse_than_1db=[r['cell'] for r in sub if r['nmse_diff_vs_A'][n]['point'] > L2_DB],
                ber_ci_above_zero=[r['cell'] for r in sub if r['ber_diff_vs_A'][n]['ci95'][0] > 0],
                mean_diff_db=float(np.mean([r['nmse_diff_vs_A'][n]['point'] for r in sub])))
        v = dict(L1=dict(full_grid_diff_db=l1_full, diff_30_60_db=l1_high, passed=bool(l1_full <= L1_DB and l1_high <= L1_DB)),
                 L2=dict(worst_cell=worst['cell'], worst_diff_db=worst['nmse_diff_vs_A'][n]['point'],
                         cells_over_limit=[r['cell'] for r in grid if r['nmse_diff_vs_A'][n]['point'] > L2_DB],
                         passed=bool(worst['nmse_diff_vs_A'][n]['point'] <= L2_DB)),
                 L3=dict(cells_with_ber_ci_above_zero=ber_bad, passed=not ber_bad),
                 L4=l4, secondary_offgrid=secondary)
        v['replaces_table'] = bool(v['L1']['passed'] and v['L2']['passed'] and v['L3']['passed'] and bool(l4['passed']))
        verdicts[n] = v
    passing = [n for n in ('C1', 'C2', 'D') if n in verdicts and verdicts[n]['replaces_table']]

    single = [n for n in ('C1', 'C2', 'D') if n in names]
    best_single = min(single, key=lambda n: aggregates['grid']['arms'][n]['mean_setting_nmse_db'])
    oracle_rows = []
    for r in rows:
        if r['kind'] != 'grid':
            continue
        oracle_rows.append(dict(cell=r['cell'], snr=r['snr'], rsr=r['rsr'], regime=r['regime'], oracle_c=r['oracle_c'],
            table_c=r['table_c'], O=r['nmse_db']['O'], A=r['nmse_db']['A'], best_single=r['nmse_db'][best_single],
            B=r['nmse_db']['B'], headroom_A_minus_O=r['nmse_db']['A'] - r['nmse_db']['O'],
            cost_single_minus_A=r['nmse_db'][best_single] - r['nmse_db']['A']))
    def avg(scope_rows, key):
        return float(np.mean([x[key] for x in scope_rows]))
    o_high = [x for x in oracle_rows if x['snr'] >= 30]
    oracle = dict(best_single_constant_arm=best_single, rows=oracle_rows,
        averages={s: {k: avg(rs, k) for k in ('O', 'A', 'best_single', 'B', 'headroom_A_minus_O', 'cost_single_minus_A')}
                  for s, rs in (('full_grid', oracle_rows), ('snr_30_60', o_high))},
        max_headroom=dict(value=max(x['headroom_A_minus_O'] for x in oracle_rows),
                          cell=max(oracle_rows, key=lambda x: x['headroom_A_minus_O'])['cell']))
    oracle['conclusion_rule_met'] = bool(oracle['averages']['full_grid']['headroom_A_minus_O'] < L1_DB
                                         and oracle['averages']['snr_30_60']['headroom_A_minus_O'] < L1_DB
                                         and oracle['max_headroom']['value'] < L2_DB)

    summary = dict(frozen_choices=choice['selected_constants'], frozen_sha256=digest, d_gate=choice['d_gate'],
        true_nmse_choice_diagnostic=choice['true_nmse_choice_diagnostic'], arms=names,
        parity_max_abs=max(r['parity_max_abs'] for r in rows), verdicts=verdicts, replaces_table=passing,
        preferred_replacement=passing[0] if passing else None, aggregates=aggregates, oracle=oracle,
        timing={str(k): v for k, v in timing.items()}, timing_absolute_cap_pass_A=a_cap, cells=rows,
        test_cell_sha256={r['cell']: sha(c['path']) for r, c in zip(rows, cells)},
        bootstrap=dict(draws=BOOT, seed=SEED['bootstrap'], blocks='(regime, reference), 128 anchors, shared across SNR/arm/fit'))
    (HERE/'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    figures(rows, names)
    print(json.dumps(dict(choices=choice['selected_constants'], verdicts={n: dict(L1=v['L1'], L2={k: v['L2'][k] for k in ('worst_cell', 'worst_diff_db', 'passed')},
        L3=dict(passed=v['L3']['passed'], cells=len(v['L3']['cells_with_ber_ci_above_zero'])), L4=v['L4'], replaces=v['replaces_table'])
        for n, v in verdicts.items()}, oracle=oracle['averages'], max_headroom=oracle['max_headroom'],
        conclusion=oracle['conclusion_rule_met'], best_single=best_single), indent=1))


def scopes_of(row):
    out = [row['kind']]
    if row['kind'] == 'grid':
        if row['snr'] >= 30:
            out.append('grid_30_60')
        out.append(f"grid_rsr{row['rsr']:g}")
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID_INK, linewidth=.6)
    ax.set_axisbelow(True)


def figures(rows, names):
    fig_dir = HERE/'figures'
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK, 'figure.facecolor': SURFACE})
    rsrs = sorted({r['rsr'] for r in rows})
    for name, key, ylabel, fname in (('NMSE', 'nmse_db', 'NMSE (dB), mean over path regimes', 'nmse_vs_snr.png'),
                                     ('Difference', 'diff', 'NMSE − arm A (dB); positive = worse', 'nmse_diff_vs_snr.png')):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
        for ax, rsr in zip(axes.ravel(), rsrs):
            style(ax)
            sub = [r for r in rows if r['rsr'] == rsr]
            snrs = sorted({r['snr'] for r in sub})
            for arm in names:
                if key == 'diff' and arm == 'A':
                    continue
                vals = [np.mean([(r['nmse_db'][arm] - r['nmse_db']['A']) if key == 'diff' else r['nmse_db'][arm]
                                 for r in sub if r['snr'] == s]) for s in snrs]
                ax.plot(snrs, vals, color=COLORS[arm], marker=MARKERS[arm], markersize=3.5, linewidth=1.6,
                        linestyle='--' if arm == 'O' else '-', label=arm)
            off = [s for s in snrs if s in OFF_SNRS]
            for s in off:
                ax.axvline(s, color=GRID_INK, linewidth=.8, zorder=0)
            if key == 'diff':
                ax.axhline(0, color=MUTED, linewidth=.8)
                ax.axhline(L2_DB, color=MUTED, linewidth=.8, linestyle=':')
            kind_label = 'grid' if rsr in GRID_RSRS else 'off-grid'
            ax.set_title(f'RSR {rsr:g} dB ({kind_label})', fontsize=9, loc='left')
        for ax in axes[-1]:
            ax.set_xlabel('SNR (dB); light verticals = off-grid SNRs')
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(fig_dir/fname, dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
