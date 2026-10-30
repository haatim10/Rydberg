"""Reporting-only manuscript material from immutable, audited experiment JSON.

No estimation, selection, resampling, or timing is performed. Run from any folder.
"""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / 'results/tcom_reliability_20261021'
EVIDENCE = HERE / 'evidence/reliability'
FIGURES = HERE / 'figures'
LABELS = {
    'receiver_mean': 'Reliability receiver',
    'prior_mean': 'Prior selective EM',
    'latest_mean': 'Full repaired receiver',
    'adaptive40_mean': 'Adaptive start + EM40',
    'global40_mean': 'Global schedule + EM40',
    'classical40': 'Classical start + EM40',
    'em100': 'Spectral EM100',
    'compact_sparse': 'Compact sparse/EM',
    'algebraic20': 'Algebraic start + EM20',
}
FILES = ['protocol.json', 'policy.json', 'nmse_summary.json', 'ber/summary.json',
         'weak_error_concentration.json', 'completion.json', 'RESULTS.txt',
         'timing_window1.json', 'timing_window2.json',
         'audit/selection.json', 'audit/test.json', 'ber/audit.json',
         'audit/summary.json', 'audit/timing.json']
HISTORIES = ['tcom_pilot_em_20261018', 'tcom_retrained_bypass_20261018',
             'tcom_transition_repair_20261019', 'tcom_filter_continuation_20261019',
             'tcom_runtime_repair_20261020', 'tcom_start_budget_20261020',
             'tcom_hybrid_20261020', 'tcom_hybrid_sensitivity_20261020']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def number(value, places=6):
    return f'{value:.{places}f}'


def interval(values, places=3, scale=1):
    return '[' + ', '.join(number(x * scale, places) for x in values) + ']'


def table(caption, label, header, rows, specification, note='', star=True):
    environment = 'table*' if star else 'table'
    text = [f'\\begin{{{environment}}}[!t]', '\\centering',
            '\\caption{' + caption + '}', '\\label{' + label + '}',
            '\\small', '\\setlength{\\tabcolsep}{4pt}',
            '\\begin{tabular}{' + specification + '}', '\\toprule',
            ' & '.join(header) + r' \\', '\\midrule']
    text += [' & '.join(row) + r' \\' for row in rows]
    text += ['\\bottomrule', '\\end{tabular}']
    if note:
        text += ['\\par\\vspace{2pt}',
                 '\\begin{minipage}{0.98\\textwidth}\\footnotesize ' + note + '\\end{minipage}']
    text += [f'\\end{{{environment}}}', '']
    return '\n'.join(text)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    before = {name: sha(SOURCE/name) for name in FILES}
    n, b = load(SOURCE/'nmse_summary.json'), load(SOURCE/'ber/summary.json')
    protocol, policy, completion = (load(SOURCE/name) for name in
                                     ('protocol.json', 'policy.json', 'completion.json'))
    windows = [load(SOURCE/f'timing_window{i}.json') for i in (1, 2)]
    assert completion['status'] == 'complete' and completion['all_required_audits_passed']
    assert completion['report_sha256'] == before['RESULTS.txt']
    assert n['selection_sha256'] == b['selection_sha256'] == before['policy.json']
    assert n['protocol_sha256'] == b['protocol_sha256'] == before['protocol.json']
    for field in ('source_hashes', 'checkpoint_hashes'):
        for name, digest in protocol[field].items():
            assert sha(REPO/name) == digest, name
    for name, digest in completion['artifact_hashes'].items():
        assert sha(SOURCE/name) == digest, name
    for name in ('selection', 'test', 'summary', 'timing'):
        assert load(SOURCE/f'audit/{name}.json')['passed']
    assert load(SOURCE/'ber/audit.json')['passed']

    # All tabular content is generated from the same saved summary fields below.
    validations = []
    main_rows, table_values = [], []
    for name, label in LABELS.items():
        q, r = n['aggregates']['all']['methods'][name], b['regions']['all']['methods'][name]
        values = [q['nmse_db'], q['pooled_ratio_of_sums_linear'], 100*r['ber'],
                  max(w['methods'][name]['p90_ms'] for w in windows),
                  max(w['methods'][name]['weak_target']['p90_ms'] for w in windows)]
        formatted = [number(values[0], 3), number(values[1], 8), number(values[2], 6),
                     number(values[3], 3), number(values[4], 3)]
        main_rows.append([label, *formatted])
        table_values.append(dict(method=name, values=values, displayed=formatted))
    main_text = table(
        'Fresh 153-condition comparison. Lower values are better. Runtime is the slower '
        'of two independently timed windows, in milliseconds.',
        'tab:reliability-main',
        ['Method', r'\shortstack{Mean-cell\\NMSE (dB)}', r'\shortstack{Pooled linear\\NMSE}', 'BER (\\%)',
         r'\shortstack{Full-grid\\p90 (ms)}', r'\shortstack{Weak-target\\p90 (ms)}'], main_rows, 'lrrrrr',
        'Neural rows average metrics from three separately evaluated fits, not ensembles. '
        'Mean-cell NMSE equally averages condition-level dB ratios of summed error to '
        'summed channel energy; pooled linear NMSE first pools all conditions within each '
        'fit and then averages the fit ratios. The weak target is RSR 5~dB and SNR '
        '$>30$~dB (24 conditions). The declared runtime target is 10.74~ms in both scopes '
        'and both windows; it is not a per-call deadline. Compact sparse/EM includes '
        'zero-penalty EM choices and remains best in pooled-linear NMSE and aggregate BER.')
    (HERE/'reliability_main_table.tex').write_text(main_text)
    validations.append(('reliability_main_table.tex', main_rows))

    selected = ['receiver_mean', 'prior_mean', 'latest_mean', 'compact_sparse', 'algebraic20']
    scopes = [('ordinary', 'Ordinary (45)'), ('weak_target', 'Weak target (24)'),
              ('high_all', 'High SNR (45)')]
    region_rows, region_values = [], []
    for scope, label in scopes:
        for metric in ('NMSE', 'BER'):
            vals = ([n['aggregates'][scope]['methods'][m]['nmse_db'] for m in selected]
                    if metric == 'NMSE' else [100*b['regions'][scope]['methods'][m]['ber'] for m in selected])
            row = [label if metric == 'NMSE' else '', 'dB' if metric == 'NMSE' else r'BER (\%)']
            row += [number(v, 3 if metric == 'NMSE' else 6) for v in vals]
            region_rows.append(row)
            region_values.append(dict(scope=scope, metric=metric, methods=selected, values=vals))
    (HERE/'reliability_region_table.tex').write_text(table(
        'Descriptive operating-range results on the same fresh bank.', 'tab:reliability-regions',
        ['Region', 'Metric', 'Reliability', r'\shortstack{Prior selective\\EM}', 'Full repair', 'Sparse/EM', 'Algebraic20'],
        region_rows, 'llrrrrr',
        'Ordinary: SNR $\\leq20$~dB, all references. Weak target: RSR 5~dB and SNR '
        '$>30$~dB. High SNR: SNR $\\geq40$~dB, all references. These scopes overlap '
        'and are not the multiplicity-adjusted primary comparisons. A displayed zero BER '
        'means no observed bit errors; it is neither a zero-error guarantee nor evidence '
        'of population equivalence. The first three methods are bitwise identical in '
        'the ordinary region.'))
    validations.append(('reliability_region_table.tex', region_rows))

    comparison_labels = {
        'receiver_vs_prior': 'Prior selective EM', 'receiver_vs_latest': 'Full repair',
        'receiver_vs_compact': 'Compact sparse/EM', 'receiver_vs_adaptive40': 'Adaptive + EM40',
        'receiver_vs_global40': 'Global + EM40', 'receiver_vs_classical40': 'Classical + EM40',
        'receiver_vs_em100': 'Spectral EM100', 'receiver_vs_algebraic20': 'Algebraic + EM20'}
    effect_rows, harm_rows = [], []
    for key, label in comparison_labels.items():
        x, y = n['aggregates']['all']['contrasts'][key], b['regions']['all']['contrasts'][key]
        primary = key in ('receiver_vs_prior', 'receiver_vs_latest')
        ni = x['bonferroni4_interval_db'] if primary else x['pointwise_ci95_db']
        bi = y['bonferroni4_interval'] if primary else y['bootstrap95']
        effect_rows.append([label, 'Adjusted' if primary else 'Pointwise', number(x['gain_db'], 3),
                            interval(ni), number(100*y['ber_reduction'], 6), interval(bi, 6, 100)])
        hn, hb = n['negative_cells'][key], b['negative_cells'][key]
        harm_rows.append([label, str(len(hn['negative_cells'])), str(len(hn['pointwise95_harm_cells'])),
                          str(len(hb['negative_cells'])), str(len(hb['pointwise95_harm_cells'])),
                          number(hn['worst_gain_db'], 3)])
    extra = table('Paired whole-grid improvements relative to each reference. Positive values '
                  'favor the reliability receiver.', 'tab:reliability-effects',
                  ['Reference', 'Interval', r'\shortstack{NMSE gain\\(dB)}', 'Interval (dB)', r'\shortstack{BER reduction\\(pp)}', 'Interval (pp)'],
                  effect_rows, 'llrrrr', 'Adjusted rows use 98.75\\% percentile intervals '
                  '(Bonferroni across two contrasts and two outcomes for nominal 95\\% family '
                  'coverage). Other rows use descriptive 95\\% intervals. The 20,000 resamples '
                  'are paired within nine independent reference/path blocks; channel indices '
                  'are shared across SNRs and fits within a block. Inference is conditional '
                  'on the fixed fits, policy, and simulation grid. A collapsed empirical '
                  'BER interval does not establish equivalence or noninferiority.')
    extra += table('Condition-level adverse results out of 153 conditions.', 'tab:reliability-harms',
                   ['Reference', r'\shortstack{NMSE\\worse}', r'\shortstack{NMSE\\harm CI}', r'\shortstack{BER\\worse}', r'\shortstack{BER\\harm CI}', r'\shortstack{Worst NMSE\\gain (dB)}'],
                   harm_rows, 'lrrrrr', 'Worse means a negative point improvement; harm CI means '
                   'a descriptive pointwise 95\\% interval lies below zero. Complete adverse '
                   'condition identifiers, fit effects, and intervals are retained in the '
                   'archived summaries. No multiplicity-adjusted subgroup safety claim is made.')
    runtime_rows = []
    for key, record in completion['receiver_latency_extremes'].items():
        maximum = record['maximum_call']
        runtime_rows.append([key.replace('window', 'Window '), number(record['max_ms'], 6),
                             f"{record['over_budget_calls']}/{record['calls']}",
                             maximum['method'].replace('receiver_', ''),
                             maximum['regime'].replace('random3to7', 'Random 3--7'),
                             f"{maximum['rsr']:g}/{maximum['snr']:g}",
                             str(maximum['index']), str(maximum['gate_triggered']).lower()])
    extra += table('Individual receiver timing extremes, retaining both windows.', 'tab:reliability-latency',
                   ['Window', 'Maximum (ms)', '$>10.74$ ms', 'Fit', 'Paths', 'RSR/SNR (dB)', 'Index', 'Gate'],
                   runtime_rows, 'lrrllrrl', 'The second window contains one 22.275833~ms call '
                   'despite satisfying both p90 targets. Its gate was false; these measurements '
                   'do not establish the cause. Pooled p90 eligibility is not a hard per-call deadline.')
    (HERE/'reliability_evidence_tables.tex').write_text(extra)
    validations.extend([('reliability_evidence_tables.tex', effect_rows),
                        ('reliability_evidence_tables.tex', harm_rows),
                        ('reliability_evidence_tables.tex', runtime_rows)])

    # Publication figures: no estimates or confidence intervals are recomputed.
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'lines.linewidth': 1.15, 'axes.linewidth': .65})
    snrs, refs = protocol['snrs'], protocol['rsrs']
    colors = {'receiver_mean':'#0072B2','prior_mean':'#D55E00','latest_mean':'#CC79A7',
              'compact_sparse':'#009E73','global40_mean':'#E69F00','algebraic20':'#444444',
              'adaptive40_mean':'#E69F00','perfect_csi':'#777777'}
    styles = {'receiver_mean':'-', 'prior_mean':'--', 'latest_mean':':',
              'compact_sparse':'-.', 'global40_mean':'--', 'algebraic20':':',
              'adaptive40_mean':'--', 'perfect_csi':':'}
    plot_labels = {**LABELS, 'receiver_mean':'Reliability', 'prior_mean':'Prior selective EM',
                   'latest_mean':'Full repair','global40_mean':'Global + EM40',
                   'algebraic20':'Algebraic + EM20','adaptive40_mean':'Adaptive + EM40',
                   'perfect_csi':'Perfect CSI'}
    plotted = {'nmse':{}, 'ber':{}}
    nmse_names = ['receiver_mean','prior_mean','latest_mean','compact_sparse','global40_mean','algebraic20']
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.65), sharex=True, sharey=True)
    for ax, ref in zip(axes, refs):
        for name in nmse_names:
            vals = [float(np.mean([c['methods'][name]['nmse_db'] for c in n['cells']
                                  if c['rsr']==ref and c['snr']==snr])) for snr in snrs]
            plotted['nmse'][f'{name}/rsr{ref:g}']=vals
            ax.plot(snrs, vals, color=colors[name], ls=styles[name],
                    marker='o' if name=='receiver_mean' else None, ms=2.1, label=plot_labels[name])
        ax.set_title(f'RSR {ref:g} dB'); ax.set_xlabel('SNR (dB)')
        ax.grid(True, color='#dddddd', lw=.45);ax.set_xticks([0,20,40,60]);ax.set_xlim(-1,61)
    axes[0].set_ylabel('Mean-cell NMSE (dB)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3,
               frameon=False, bbox_to_anchor=(.5,1.01), columnspacing=1.5, handlelength=2.6)
    fig.subplots_adjust(left=.088,right=.993,bottom=.18,top=.73,wspace=.14)
    for suffix in ('pdf','png'):fig.savefig(FIGURES/f'reliability_nmse.{suffix}',dpi=240)
    plt.close(fig)

    ber_names = ['receiver_mean','compact_sparse','algebraic20','adaptive40_mean','perfect_csi']
    fig = plt.figure(figsize=(6.8, 2.8))
    outer = fig.add_gridspec(1,3,left=.088,right=.993,bottom=.23,top=.73,wspace=.14)
    positives=[]
    for ref in refs:
        for name in ber_names:
            vals = [float(np.mean([c['ber'] for c in b['cells']
                                  if c['rsr_db']==ref and c['snr_db']==snr and c['method']==name])) for snr in snrs]
            plotted['ber'][f'{name}/rsr{ref:g}']=vals
            positives += [v for v in vals if v>0]
    lower = 10**np.floor(np.log10(min(positives))) / 2
    handles=[]
    for column,ref in enumerate(refs):
        sub=outer[column].subgridspec(2,1,height_ratios=[8,1.2],hspace=.08)
        ax=fig.add_subplot(sub[0]); zero=fig.add_subplot(sub[1],sharex=ax)
        for offset,name in enumerate(ber_names):
            vals=np.array(plotted['ber'][f'{name}/rsr{ref:g}']); pos=vals>0
            ax.plot(snrs,np.where(pos,vals,np.nan),color=colors[name],ls=styles[name],
                    marker='o' if name=='receiver_mean' else None,ms=2.1)
            # The zero row is categorical, outside the logarithmic BER axis.
            zero.plot(np.array(snrs)[~pos],np.full((~pos).sum(),(offset-2)*.12),
                      linestyle='none',marker='v',ms=2.9,color=colors[name],clip_on=False)
        ax.set_yscale('log');ax.set_ylim(lower,.2);ax.set_xlim(-1,61)
        ax.set_title(f'RSR {ref:g} dB');ax.grid(True,color='#dddddd',lw=.45)
        ax.tick_params(labelbottom=False);ax.spines['bottom'].set_visible(False)
        zero.set_ylim(-.42,.42);zero.set_yticks([]);zero.set_xticks([0,20,40,60]);zero.set_xlabel('SNR (dB)')
        zero.spines[['top','right','left']].set_visible(False)
        if column==0:
            ax.set_ylabel('Uncoded BER')
            zero.text(-.04,.5,'0 obs.',transform=zero.transAxes,ha='right',va='center',fontsize=8)
        else:ax.tick_params(labelleft=False)
    handles=[Line2D([0],[0],color=colors[name],ls=styles[name],label=plot_labels[name]) for name in ber_names]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.01),
               columnspacing=1.5,handlelength=2.6)
    fig.text(.5,.017,'Triangles: zero observed errors (separate row; not a finite BER or confidence bound).',
             ha='center',fontsize=8)
    for suffix in ('pdf','png'):fig.savefig(FIGURES/f'reliability_ber.{suffix}',dpi=240)
    plt.close(fig)

    # Read back every generated data row. Values are bound to source summaries below.
    row_count = 0
    for filename,rows in validations:
        saved=(HERE/filename).read_text()
        for row in rows:
            assert ' & '.join(row)+r' \\' in saved,(filename,row)
            row_count+=1
    for row in table_values:
        name=row['method'];q=n['aggregates']['all']['methods'][name]
        assert row['values'][0]==q['nmse_db']
        assert row['values'][1]==q['pooled_ratio_of_sums_linear']
        assert row['values'][2]==100*b['regions']['all']['methods'][name]['ber']
    for name in FILES:
        target=EVIDENCE/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE/name,target)
        assert sha(target)==before[name]
    write_json(EVIDENCE/'table_values.json',dict(main=table_values,regions=region_values,
        paired_effect_rows=effect_rows,adverse_count_rows=harm_rows,latency_rows=runtime_rows))
    write_json(EVIDENCE/'figure_values.json',dict(snrs=snrs,rsrs=refs,values=plotted,
        aggregation='Arithmetic mean of three path-regime condition metrics. Neural metrics already average fits.',
        zero_encoding='Separate categorical row; never a finite floor or bound.'))
    histories={name:dict(path=f'results/{name}/RESULTS.txt',sha256=sha(REPO/f'results/{name}/RESULTS.txt'))
               for name in HISTORIES}
    write_json(EVIDENCE/'historical_report_index.json',histories)
    (EVIDENCE/'INTERPRETATION.txt').write_text(
        'The final table and figures use ONLY the fresh reliability-study bank. Earlier studies\n'
        'are mechanism/selection diagnostics with distinct banks, comparators and primary scopes.\n'
        'Do not pool their estimates or transplant their winning ranks into this table.\n'
        'The compact sparse/EM selector is inherited and frozen; 17 of its 51 bins use factor=0 EM100.\n'
        'The prior selective-EM policy is the earlier exploratory B proposal, whose old C check failed.\n'
        'The present independent C confirms the new gate versus that incumbent only.\n'
        'The current receiver has no adverse condition-mean NMSE against prior, but 32 against full repair.\n'
        'Its mean-cell NMSE advantage does not override compact sparse/EM wins in pooled-linear NMSE and BER.\n'
        'All intervals are conditional on these three fixed fits and the specified simulation grid.\n'
        'All zero-error displays and degenerate empirical intervals remain noninformative about rare-event bounds.\n')
    after={name:sha(SOURCE/name) for name in FILES};assert before==after
    generated=['reliability_main_table.tex','reliability_region_table.tex','reliability_evidence_tables.tex',
               'figures/reliability_nmse.pdf','figures/reliability_nmse.png',
               'figures/reliability_ber.pdf','figures/reliability_ber.png']
    write_json(EVIDENCE/'validation.json',dict(passed=True,source_artifacts_unchanged=True,
        source_binding_verification='All protocol source/checkpoint and completion artifact hashes verified.',
        readback_table_rows=row_count,main_numeric_cells=45,regions_numeric_cells=30,
        figure_aggregation='Three path-regime means from archived condition metrics; no new estimation/resampling.',
        source_hashes=before,generated_hashes={name:sha(HERE/name) for name in generated},
        generator_sha256=sha(__file__)))
    manifest={str(path.relative_to(EVIDENCE)):sha(path) for path in sorted(EVIDENCE.rglob('*')) if path.is_file() and path.name!='SHA256SUMS.json'}
    write_json(EVIDENCE/'SHA256SUMS.json',manifest)
    print(json.dumps(dict(passed=True,readback_table_rows=row_count,generated=generated,evidence_files=len(manifest))))


if __name__ == '__main__':
    main()
