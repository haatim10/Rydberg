"""Write REPORT.md from summary.json, configs and timing records (no hand-copied numbers)."""
import json
import numpy as np
from .common import HERE, GRID_SNRS, GRID_RSRS

S = json.loads((HERE/'summary.json').read_text())
F = json.loads((HERE/'configs/frozen_choices.json').read_text())


def f(x, d=2, sign=False):
    return 'n/a' if x is None else (f'{x:+.{d}f}' if sign else f'{x:.{d}f}')


def verdict_table():
    out = ['| Arm | L1 full grid (≤ +0.3 dB) | L1 30–60 dB (≤ +0.3 dB) | L2 worst setting (≤ +1.0 dB) | L3 BER harm settings | L4 runtime | Replaces table |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for n, v in S['verdicts'].items():
        mark = lambda ok: 'pass' if ok else '**fail**'
        out.append(f"| {n} | {f(v['L1']['full_grid_diff_db'], sign=True)} ({mark(v['L1']['passed'])}) | "
                   f"{f(v['L1']['diff_30_60_db'], sign=True)} | {f(v['L2']['worst_diff_db'], sign=True)} at `{v['L2']['worst_cell']}` "
                   f"({mark(v['L2']['passed'])}; {len(v['L2']['cells_over_limit'])} over) | "
                   f"{len(v['L3']['cells_with_ber_ci_above_zero'])} ({mark(v['L3']['passed'])}) | {mark(v['L4']['passed'])} | "
                   f"{'**yes**' if v['replaces_table'] else 'no'} |")
    return '\n'.join(out)


def scope_table():
    names = S['arms']
    labels = dict(grid='Full grid (153)', grid_30_60='30–60 dB (81)', grid_rsr5='RSR 5 (51)', grid_rsr12='RSR 12 (51)',
                  grid_rsr20='RSR 20 (51)', offgrid_snr='Off-grid SNR (45)', rsr8='RSR 8 (66)', rsr16='RSR 16 (66)',
                  rsr0='RSR 0, weak (66)')
    out = ['| Scope | ' + ' | '.join(names) + ' | ' + ' | '.join(f'{n} − A [95% CI]' for n in names if n != 'A') + ' |',
           '| --- ' * (2*len(names)) + '|']
    for k, lab in labels.items():
        a = S['aggregates'][k]['arms']
        out.append(f'| {lab} | ' + ' | '.join(f(a[n]['mean_setting_nmse_db']) for n in names) + ' | ' +
                   ' | '.join(f"{f(a[n]['nmse_diff_vs_A']['point'], sign=True)} [{f(a[n]['nmse_diff_vs_A']['ci95'][0], sign=True)}, "
                              f"{f(a[n]['nmse_diff_vs_A']['ci95'][1], sign=True)}]" for n in names if n != 'A') + ' |')
    return '\n'.join(out)


def pooled_ber_table():
    names = S['arms']
    out = ['| Scope | ' + ' | '.join(f'{n} pooled NMSE dB' for n in names) + ' | ' + ' | '.join(f'{n} BER %' for n in names) + ' |',
           '| --- ' * (2*len(names) + 1) + '|']
    for k, lab in (('grid', 'Full grid'), ('grid_30_60', '30–60 dB'), ('grid_rsr5', 'RSR 5'), ('grid_rsr12', 'RSR 12'),
                   ('grid_rsr20', 'RSR 20'), ('offgrid_snr', 'Off-grid SNR'), ('rsr8', 'RSR 8'), ('rsr16', 'RSR 16'), ('rsr0', 'RSR 0')):
        a = S['aggregates'][k]['arms']
        out.append(f'| {lab} | ' + ' | '.join(f(a[n]['pooled_nmse_db']) for n in names) + ' | ' +
                   ' | '.join(f(100*a[n]['ber'], 4) for n in names) + ' |')
    return '\n'.join(out)


def ber_diff_table():
    names = [n for n in S['arms'] if n != 'A']
    out = ['| Scope | ' + ' | '.join(f'{n} − A BER (pp) [95% CI]' for n in names) + ' |', '| --- ' * (len(names) + 1) + '|']
    for k, lab in (('grid', 'Full grid'), ('grid_30_60', '30–60 dB'), ('rsr0', 'RSR 0')):
        a = S['aggregates'][k]['arms']
        out.append(f'| {lab} | ' + ' | '.join(
            f"{100*a[n]['ber_diff_vs_A']['point']:+.4f} [{100*a[n]['ber_diff_vs_A']['ci95'][0]:+.4f}, {100*a[n]['ber_diff_vs_A']['ci95'][1]:+.4f}]"
            for n in names) + ' |')
    return '\n'.join(out)


def runtime_table():
    out = ['| Window | Arm | Full-grid p90 ms (× A) | Target-range p90 ms (× A) | Full-grid max ms |', '| --- | --- | --- | --- | --- |']
    for w, t in S['timing'].items():
        for n, v in t.items():
            out.append(f"| {w} | {n} | {f(v['full_grid']['p90_ms'])} (×{v['full_grid']['p90_ms']/t['A']['full_grid']['p90_ms']:.3f}) | "
                       f"{f(v['target_range']['p90_ms'])} (×{v['target_range']['p90_ms']/t['A']['target_range']['p90_ms']:.3f}) | "
                       f"{f(v['full_grid']['max_ms'], 1)} |")
    return '\n'.join(out)


def oracle_table():
    rows = S['oracle']['rows']
    best = S['oracle']['best_single_constant_arm']
    out = [f'| RSR | SNR | table c | oracle c (paths L2/L4/rnd) | O | A | {best} | B | A − O | {best} − A |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for rsr in GRID_RSRS:
        for snr in GRID_SNRS:
            sel = [r for r in rows if r['rsr'] == rsr and r['snr'] == snr]
            m = lambda k: float(np.mean([r[k] for r in sel]))
            out.append(f"| {rsr:g} | {snr:g} | {sel[0]['table_c']:g} | {'/'.join(f'{r['oracle_c']:g}' for r in sel)} | "
                       f"{f(m('O'))} | {f(m('A'))} | {f(m('best_single'))} | {f(m('B'))} | {f(m('headroom_A_minus_O'), sign=True)} | "
                       f"{f(m('cost_single_minus_A'), sign=True)} |")
    av = S['oracle']['averages']
    for k, lab in (('snr_30_60', '**30–60 dB mean**'), ('full_grid', '**Full-grid mean**')):
        out.append(f"| {lab} | | | | {f(av[k]['O'])} | {f(av[k]['A'])} | {f(av[k]['best_single'])} | {f(av[k]['B'])} | "
                   f"{f(av[k]['headroom_A_minus_O'], sign=True)} | {f(av[k]['cost_single_minus_A'], sign=True)} |")
    return '\n'.join(out)


def curves_table():
    out = ['| c | C1 held-out NLL | C1 NMSE dB | C2 held-out NLL | C2 NMSE dB |', '| --- | --- | --- | --- | --- |']
    for a, b in zip(F['validation_curves']['C1'], F['validation_curves']['C2']):
        out.append(f"| {a['c']:g} | {a['heldout_nll']:.5f} | {a['nmse_db']:.2f} | {b['heldout_nll']:.5f} | {b['nmse_db']:.2f} |")
    A = F['validation_curves']['A']
    out.append(f"| table (A) | {A['heldout_nll']:.5f} | {A['nmse_db']:.2f} | | |")
    return '\n'.join(out)


TEMPLATE = (HERE/'REPORT_TEMPLATE.md').read_text()
values = dict(VERDICTS=verdict_table(), SCOPES=scope_table(), POOLED_BER=pooled_ber_table(), BER_DIFF=ber_diff_table(),
              RUNTIME=runtime_table(), ORACLE=oracle_table(), CURVES=curves_table())
text = TEMPLATE
for k, v in values.items():
    text = text.replace('{{' + k + '}}', v)
assert '{{' not in text
(HERE/'REPORT.md').write_text(text)
print('REPORT.md written')
