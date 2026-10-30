"""Redraw BER figures from frozen results; no estimator or scientific rerun.

Run after make_reliability_material.py if both reporting scripts are used.
The main figure shows 0--30 dB; the companion preserves the full 0--60 dB view.
Zero measured rates are missing on the log axis, with gaps in connecting lines.
Exact values, including every zero, are saved in clean_ber_plot_values.json.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tcom-clean-ber-mpl"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / 'results' / 'tcom_reliability_20261021'
EVIDENCE = HERE / 'evidence' / 'reliability'
if not (SOURCE / 'ber/summary.json').exists():
    SOURCE = EVIDENCE / 'source'
FIGURES = HERE / 'figures'
METHODS = {
    'receiver_mean': dict(label='Proposed receiver', color='#0072B2', marker='o', linestyle='-', zorder=5),
    'compact_sparse': dict(label='Sparse + EM', color='#D55E00', marker='s', linestyle='--', zorder=4),
    'em100': dict(label='EM (100 steps)', color='#009E73', marker='^', linestyle='-.', zorder=3),
    'perfect_csi': dict(label='Known channel', color='#555555', marker='d', linestyle=':', zorder=2),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')


def make_figure(values, snrs, refs, maximum, basename):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.55), sharex=True, sharey=True)
    shown = np.asarray(snrs) <= maximum
    x = np.asarray(snrs)[shown]
    records = []
    for axis, ref in zip(axes, refs):
        for name, style in METHODS.items():
            y = np.asarray(values[f'{name}/rsr{ref:g}'])[shown]
            plotted = np.where(y > 0, y, np.nan)
            line, = axis.plot(x, plotted, markerfacecolor='none', markersize=3.6,
                             markeredgewidth=.9, linewidth=1.15, **style)
            # The displayed arrays must be exact stored rates, with only zeros omitted.
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), plotted)
            assert np.array_equal(np.isnan(line.get_ydata()), y == 0)
            records.append(dict(reference_db=ref, method=name,
                                positive_points=int(np.count_nonzero(y)), zero_points=int(np.count_nonzero(y == 0))))
        axis.set_yscale('log')
        axis.set_ylim(4e-6, .3)
        axis.set_xlim(-.7, maximum + .7)
        axis.set_xticks([0, 10, 20, 30] if maximum == 30 else [0, 20, 40, 60])
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=6))
        axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=30))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(which='major', color='#DADADA', linewidth=.5)
        axis.tick_params(axis='both', which='major', length=3, width=.65)
        axis.tick_params(axis='y', which='minor', length=1.5, width=.5)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_title(f'Reference strength: {ref:g} dB', pad=6)
        axis.set_xlabel('SNR (dB)', labelpad=3)
    axes[0].set_ylabel('Bit error rate', labelpad=3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.52, 1.003), ncol=4, frameon=False,
               handlelength=2.45, columnspacing=1.2, handletextpad=.55)
    fig.subplots_adjust(left=.085, right=.993, bottom=.18, top=.79, wspace=.12)
    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES / f'{basename}.{ext}', dpi=260,
                    metadata={'Creator': 'Frozen-result BER figure script'} if ext == 'pdf' else None)
    plt.close(fig)
    return records


def main():
    bound = ['protocol.json', 'policy.json', 'ber/summary.json', 'ber/audit.json', 'completion.json']
    before = {name: sha(SOURCE / name) for name in bound}
    source, protocol, completion = [load(SOURCE / name) for name in ('ber/summary.json', 'protocol.json', 'completion.json')]
    assert completion['status'] == 'complete' and completion['all_required_audits_passed']
    assert load(SOURCE / 'ber/audit.json')['passed']
    assert source['protocol_sha256'] == before['protocol.json']
    assert source['selection_sha256'] == before['policy.json']
    # Prefer the frozen experiment directory; a ZIP-only reporting rebuild can use
    # the exact source copies retained under evidence/reliability/source.
    snrs, refs = protocol['snrs'], protocol['rsrs']
    values, rows = {}, []
    for name in METHODS:
        for ref in refs:
            rates = []
            for snr in snrs:
                cells = [c for c in source['cells'] if c['method'] == name and c['rsr_db'] == ref and c['snr_db'] == snr]
                assert len(cells) == 3
                assert {c['regime'] for c in cells} == {'L2', 'L4', 'random3to7'}
                # Equal numbers of data bits per path regime; neural cell rates
                # already average the three separate fits, never an ensemble.
                assert {c['bits_per_fit'] for c in cells} == {49152}
                rate = float(np.mean([c['ber'] for c in cells]))
                assert math.isclose(rate, math.fsum(c['ber'] for c in cells) / 3, rel_tol=1e-15, abs_tol=0)
                assert 0 <= rate <= 1 and math.isfinite(rate)
                rates.append(rate)
                rows.append(dict(method=name, reference_db=ref, snr_db=snr, ber=rate,
                                 zero_observed=rate == 0,
                                 regime_rates={c['regime']:c['ber'] for c in cells}))
            values[f'{name}/rsr{ref:g}'] = rates
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
                         'axes.labelsize':8.5, 'axes.titlesize':8.5,
                         'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8,
                         'axes.linewidth':.65, 'pdf.fonttype':42, 'ps.fonttype':42})
    plotted = {
        'main_0_to_30_db':make_figure(values, snrs, refs, 30, 'reliability_ber'),
        'supplement_0_to_60_db':make_figure(values, snrs, refs, 60, 'reliability_ber_full'),
    }
    write_json(EVIDENCE / 'clean_ber_plot_values.json', {
        'snrs_db':snrs, 'references_db':refs, 'labels':{k:v['label'] for k,v in METHODS.items()},
        'aggregation':'Arithmetic mean over the three path regimes, with equal numbers of bits. Neural cell rates already average separately evaluated fits.',
        'main_scope':'SNR 0 through 30 dB; references 5, 12, and 20 dB.',
        'supplement_scope':'Entire frozen SNR grid, 0 through 60 dB.',
        'zero_handling':'Zeros are omitted from the log axis, with line gaps; they are never floored. All exact values, including zeros, are retained here.',
        'values':values, 'rows':rows,
    })
    # Refresh only the BER portion of the inherited combined reporting ledger.
    ledger = load(EVIDENCE / 'figure_values.json')
    ledger['values']['ber'] = values
    ledger['zero_encoding'] = 'Omitted from the logarithmic BER axis, with line gaps. Exact zero rates are retained in clean_ber_plot_values.json.'
    ledger['ber_revision'] = 'make_clean_ber_figure.py: main 0--30 dB, companion 0--60 dB, four methods.'
    write_json(EVIDENCE / 'figure_values.json', ledger)
    generated = [f'figures/{name}.{ext}' for name in ('reliability_ber','reliability_ber_full') for ext in ('pdf','png')]
    assert before == {name:sha(SOURCE/name) for name in bound}
    validation = dict(passed=True, frozen_sources_unchanged=True, exact_rates_checked=len(rows),
                      source_hashes=before, generator_sha256=sha(__file__),
                      generated_hashes={name:sha(HERE/name) for name in generated},
                      plotted_points=plotted, zero_points='Unplotted; connecting lines break at every zero.',
                      numerical_estimation_or_resampling_performed=False)
    write_json(EVIDENCE / 'clean_ber_plot_validation.json', validation)
    inherited = load(EVIDENCE / 'validation.json')
    inherited['generated_hashes'].update({name:sha(HERE/name) for name in generated})
    inherited['ber_figure_revision'] = dict(generator='make_clean_ber_figure.py',
                                           validation='clean_ber_plot_validation.json',
                                           exact_rates_checked=len(rows))
    write_json(EVIDENCE / 'validation.json', inherited)
    manifest={str(path.relative_to(EVIDENCE)):sha(path) for path in sorted(EVIDENCE.rglob('*')) if path.is_file() and path.name!='SHA256SUMS.json'}
    write_json(EVIDENCE/'SHA256SUMS.json', manifest)
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
