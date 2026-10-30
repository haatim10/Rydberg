"""Assemble the three completed studies without changing any estimator."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT, write_json


def read(relative):
    return json.loads((ROOT / relative).read_text())


def crossings(rows, regime, rsr):
    cells = sorted((r for r in rows if r['regime'] == regime and r['rsr'] == rsr), key=lambda r: r['snr'])
    result = []
    for a, b in zip(cells, cells[1:]):
        ca, cb = a['learned_vs_same_weight_filteroff'], b['learned_vs_same_weight_filteroff']
        if ca['gain_db'] >= 0 and cb['gain_db'] < 0:
            result.append(dict(lower_snr=a['snr'], upper_snr=b['snr'],
                lower_gain=ca['gain_db'], upper_gain=cb['gain_db'],
                lower_ci=ca['ci95_db'], upper_ci=cb['ci95_db']))
    return result


def main():
    sparse = read('sparse/paired_bootstrap.json')
    ber = read('ber/summary_with_sparse.json')
    high = read('high_snr/summary.json')
    timing = read('timing.json')
    audit = read('audit.json')
    assert audit['passed'] and read('high_snr/audit.json')['passed']
    names = [('learned_seed_mean','learned_mean','Your learned estimator'),
             ('sparse','sparse_angle','Angle-domain sparse EM'),
             ('optshrink','optshrink','Frozen offline OptShrink'),
             ('em100','em100','Unfiltered EM100')]
    table = []
    for nmse_name, ber_name, label in names:
        table.append(dict(method=label,
            equal_cell_mean_nmse_db=sparse['all']['equal_cell_nmse_db'][nmse_name],
            mean_BER=ber['overall']['methods'][ber_name]['ber'],
            mean_latency_ms=timing['aggregate'][nmse_name]['mean_ms']))
    boundary = {regime: {str(rsr): crossings(high['cells'],regime,rsr) for rsr in (5,12,20)}
                for regime in ('L2','L4','random3to7')}
    summary = dict(completed=True, ordinary_cells=15, channels_per_cell=64,
        ordinary_worlds=960, data_bits_per_estimator=1474560,
        distinct_pilot_worlds=audit['total_distinct_pilot_worlds'],
        neural_fits=3, high_snr_cells=len(high['cells']), high_snr_anchor_worlds=high['independent_worlds'],
        comparison=table, paired_NMSE_learned_minus_sparse=sparse['all'],
        paired_BER=ber['overall']['paired'], crossover_brackets=boundary,
        scope='Initial synthetic ULA study; 64 independent worlds per cell; N32 K3 P30. '
              'Ordinary comparisons use SNR0,5,10,15,20 and RSR12; high-SNR sweep includes RSR5,12,20. '
              'Pointwise uncertainty conditions on frozen training fits and sparse validation selection. '
              'No universal high-SNR cutoff or deployment switching policy was established.')
    write_json(ROOT / 'summary.json', summary)

    colors = {'learned_seed_mean':'#008569','sparse':'#9d4fa8','optshrink':'#2474ad','em100':'#bd5c25'}
    snrs = [0,5,10,15,20]
    fig, axes = plt.subplots(1,3,figsize=(15,4.4))
    for nmse_name, ber_name, label in names:
        values = [np.mean([r['nmse_db'][nmse_name] for r in sparse['cells'] if r['snr_db']==s]) for s in snrs]
        axes[0].plot(snrs,values,'o-',color=colors[nmse_name],label=label,ms=4)
        values = np.array([ber['by_snr'][str(s)]['methods'][ber_name]['ber'] for s in snrs])
        axes[1].plot(snrs,np.where(values>0, values,np.nan),'o-',color=colors[nmse_name],ms=4)
        zeros=values==0
        axes[1].scatter(np.array(snrs)[zeros],np.full(zeros.sum(),1e-6),marker='v',color=colors[nmse_name],s=25)
    axes[0].set(title='Channel estimation',ylabel='Mean cell NMSE (dB; lower is better)',xlabel='SNR (dB)')
    axes[0].legend(fontsize=7.3,loc='lower left')
    axes[1].set(title='Data detection with the same detector',ylabel='Uncoded BER (log scale)',xlabel='SNR (dB)',yscale='log')
    axes[1].text(.03,.04,'▼ Zero observed errors at a display marker\n   of 10⁻⁶; this is not an upper bound.',transform=axes[1].transAxes,fontsize=7.4)
    for regime,color,label in [('L2','#2474ad','2 paths/user'),('L4','#9d4fa8','4 paths/user'),('random3to7','#008569','3–7 paths/user')]:
        rows=sorted((r for r in high['cells'] if r['regime']==regime and r['rsr']==12),key=lambda r:r['snr'])
        x=[r['snr'] for r in rows]
        v=[r['learned_vs_same_weight_filteroff'] for r in rows]
        axes[2].plot(x,[r['gain_db'] for r in v],'.-',color=color,label=label)
        axes[2].fill_between(x,[r['ci95_db'][0] for r in v],[r['ci95_db'][1] for r in v],color=color,alpha=.1)
    axes[2].axhline(0,color='.2',lw=.7)
    axes[2].axvline(20,color='.5',ls='--',lw=.8,label='Training upper SNR')
    axes[2].set(title='When filtering helps or hurts',ylabel='Benefit vs same weights, filter off (dB)',xlabel='SNR (dB)')
    axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2,which='both')
    axes[0].set_xticks(snrs);axes[1].set_xticks(snrs)
    fig.suptitle('First completed follow-up studies · N=32, K=3, P=30 · RSR=12 dB',fontsize=13)
    fig.text(.5,.012,'64 independent channels per condition. Learned curves average 3 trained fits, not an ensemble. High-SNR curves reuse the same channel/noise anchors.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.045,1,.95])
    fig.savefig(ROOT/'overview.png',dpi=180)
    plt.close(fig)

    lines = ['Three completed follow-up experiments for the merged TCOM draft', '',
        'The new sparse baseline, common-detector BER experiment, and controlled high-SNR sweep are complete. '
        'Existing model checkpoints and manuscript sources were left unchanged.', '',
        'The ordinary comparison uses 960 new channel worlds over 15 conditions: SNR 0,5,10,15,20 dB; '
        'RSR 12 dB; two paths, four paths, or random 3–7 paths per user. Each channel is held fixed for 256 '
        'new QPSK data vectors. The detector enumerates all 64 possible three-user symbol vectors with '
        'the exact referenced-magnitude Rician likelihood. All estimators receive identical pilots and data magnitudes.', '',
        'Method | mean cell NMSE(dB) | BER(%) | warmed batch-one CPU latency(ms)']
    lines += [f"{r['method']} | {r['equal_cell_mean_nmse_db']:.3f} | {100*r['mean_BER']:.4f} | {r['mean_latency_ms']:.2f}" for r in table]
    lines += ['', 'The sparse comparator is slightly more accurate on this balanced test grid and has lower '
        'aggregate BER than the learned estimator. The difference is not uniform: sparse fitting generally '
        'wins at lower SNR, while the learned estimator becomes competitive at higher SNR and wins clearly '
        'for two-path channels at 15–20 dB. The two-path aggregate NMSE difference has a confidence interval '
        'that includes zero. The learned estimator still improves BER over the existing frozen OptShrink '
        'and EM100 controls. This supports an accuracy-versus-computation comparison, with communication '
        'performance measured explicitly, rather than a claim of universal accuracy superiority.',
        f"Measured sparse/learned mean inference-time ratio: {table[1]['mean_latency_ms']/table[0]['mean_latency_ms']:.2f}. "
        'This applies to the implemented solver and selected budgets, not an intrinsic complexity lower '
        'bound; a more optimized or shorter-budget sparse implementation may narrow the gap.']
    d=sparse['all']
    lines += ['',f"Paired learned-minus-sparse mean NMSE difference: {d['learned_minus_sparse_db']:.3f} dB; "
                  f"95% channel-bootstrap interval {[round(x,3) for x in d['paired_95_percentile_CI_db']]}. Negative favors learned."]
    for name, values in ber['overall']['paired'].items():
        lines.append(f"Paired learned-minus-{name} BER difference: {100*values['learned_minus_comparator_ber']:.4f} percentage points; "
                     f"95% interval {[round(100*x,4) for x in values['bootstrap95']]}. Negative favors learned.")
    lines += ['', 'Sparse-estimator design: a 128-column angle dictionary, continuous off-grid generated paths, '
        'exact Rician posterior-field updates, and a complex L1-penalized M-step. Regularization and iteration '
        'budget were selected using 21 fitting and 9 heldout pilots on a separate 180-world development set. '
        'True channel and path count were unavailable to selection. This is a validated first sparse implementation; '
        'it does not establish the strongest possible sparse or off-grid competitor.', '',
        f"The high-SNR study contains {len(high['cells'])} evaluated conditions from {high['independent_worlds']} distinct anchor worlds. "
        'It holds channels, pilots, reference, and standardized noise fixed across SNR. Same-weight filter-off '
        'controls retain the 10 learned step sizes. EM10 and EM100 separate filtering bias from iteration limits.',
        'Mean filtering-benefit sign-change brackets at RSR 12 dB (exploratory 2.5 dB refinement):']
    for regime in ('L2','L4','random3to7'):
        entries=boundary[regime]['12']
        lines.append(f"{regime}: " + ('; '.join(f"{e['lower_snr']:g}–{e['upper_snr']:g} dB" for e in entries) if entries else 'No positive-to-negative crossing detected in tested range.'))
    lines += ['', 'These are descriptive mean crossing brackets, not certified safe cutoffs. Reference strength and path '
        'regime change the boundary; pointwise intervals are in high_snr/summary.json. Classical OptShrink settings '
        'above 20 dB are clamped to their existing 20 dB validation choice and are not optimized for the stress regime.', '',
        'Uncertainty and scope: channel-block bootstrap preserves dependence between all estimators and the three '
        'fixed learned fits. It does not measure variation across new training/development populations. Low or zero '
        'error counts at high SNR need more independent channels; a degenerate zero bootstrap interval is not '
        'evidence of zero population BER. Perfect CSI is an ideal-information reference under vector ML, not a '
        'claimed mathematical lower bound on bit error. No coded throughput, hardware, or universal-channel claim is made.', '',
        'Validation: sparse finite-difference gradient, exact zero-penalty EM equivalence, monotone penalized '
        'likelihood, all 64 noiseless QPSK hypotheses, independent SciPy Rician-density comparison, high-SNR '
        'endpoint replays, and cross-study byte-identical observations/estimates all passed. Original input hashes '
        f"are unchanged. The three studies share data; there are {audit['total_distinct_pilot_worlds']} distinct primary evaluation pilot worlds, excluding development and timing worlds.", '',
        'For the manuscript: include this sparse method and the common-detector BER figure, explain the '
        'low/high-SNR crossover, and state the operating range explicitly. A useful next comparison is '
        'a sparse accuracy-versus-runtime curve at similar inference budgets. A useful method improvement '
        'to investigate is a filter-bypass safeguard selected on fresh validation data. The observed test '
        'crossing must not be presented as an independently validated deployment threshold.', '',
        'Useful outputs: overview.png; sparse/sparse_comparison.png and paired_bootstrap.json; '
        'ber/summary_with_sparse.json and ber_with_sparse_final.png; high_snr/high_snr_filtering_gain.png; '
        'timing.json; audit.json. Individual channel estimates and error counts are retained in the study subfolders.']
    (ROOT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
