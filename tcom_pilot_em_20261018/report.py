"""Assemble audited results without changing the manuscript or study choices."""
import json
import os
from pathlib import Path

os.environ['MPLCONFIGDIR'] = '/private/tmp/tcom_pilot_em_matplotlib'
ROOT = Path(__file__).resolve().parents[1] / 'results/tcom_pilot_em_20261018'


def read(path):
    return json.loads((ROOT / path).read_text())


def table(headers, rows):
    rows = [[str(x) for x in row] for row in [headers] + rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    line = lambda row: '  '.join(x.ljust(w) for x, w in zip(row, widths))
    return '\n'.join([line(rows[0]), line(['-' * w for w in widths])] + [line(r) for r in rows[1:]])


def interval(values, scale=1., digits=3):
    return '[' + ', '.join(f'{scale * x:.{digits}f}' for x in values) + ']'


def training_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, seed in zip(axes, (9101, 9102, 9103)):
        for arm, label, color in [('context21', 'Train with 21 pilots', '#d98324'),
                                  ('context30', 'Train with 30 pilots', '#0969da')]:
            history = read(f'checkpoints/{arm}/seed_{seed}/em_adaptive_history.json')
            xs = [x['epoch'] for x in history]
            ys = [x['val30_nll'] for x in history]
            ax.plot(xs, ys, color=color, marker='.', label=label)
            j = min(range(len(ys)), key=lambda i: ys[i])
            ax.scatter([xs[j]], [ys[j]], s=80, marker='*', color=color)
        ax.set_title(f'Training seed {seed}')
        ax.set_xlabel('Completed epoch')
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Common 30-pilot held-out NLL (lower is better)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9)
    fig.suptitle('Paired training: unchanged architecture and optimization schedule')
    fig.tight_layout(rect=(0, .09, 1, .96))
    fig.savefig(ROOT / 'training_curves.png', dpi=180)
    fig.savefig(ROOT / 'training_curves.svg')
    plt.close(fig)


def main():
    nmse = read('evaluation/nmse_summary.json')
    ber = read('ber/summary.json')
    fixed_ber = read('ber_fixed_tails/summary.json')
    timing = read('evaluation/final_timing.json')
    audits = {name: read(f'audit/{name}.json') for name in ('design', 'training', 'study', 'ber', 'fixed_tails', 'timing')}
    assert all(a['passed'] for a in audits.values()), 'An independent audit did not pass'
    digest = nmse['selection_sha256']
    assert ber['selection_sha256'] == fixed_ber['selection_sha256'] == timing['selection_sha256'] == digest
    assert nmse['timing_sha256'], 'Regenerate NMSE analysis after isolated timing'
    methods = nmse['methods']; bit_methods = ber['overall']['methods']; times = timing['methods']
    labels = [
        ('old_tail0_mean', 'Original learned model'),
        ('old_selected_mean', 'Original + selected EM'),
        ('context21_tail0_mean', 'Fresh 21-pilot control'),
        ('context21_selected_mean', 'Fresh 21-pilot + selected EM'),
        ('context30_tail0_mean', 'Fresh 30-pilot training'),
        ('context30_selected_mean', 'Fresh 30-pilot + selected EM'),
        ('sparse_fresh_context30', 'Fresh sparse, matched budget'),
        ('prior_sparse_matched', 'Prior sparse, matched budget'),
        ('prior_sparse_10ms', 'Prior sparse, 10 ms budget'),
        ('prior_sparse_20ms', 'Prior sparse, 20 ms budget'),
        ('prior_sparse_50ms', 'Prior sparse, 50 ms budget'),
        ('prior_sparse_full', 'Full sparse (100 x 25)'),
        ('em100', 'Optimized exact EM100'),
        ('optshrink', 'Frozen offline OptShrink'),
    ]
    lines = ['PILOT-MATCHED RETRAINING AND SHORT EM CONTINUATION',
             'Fresh test comparison; manuscript not updated.', '',
             'FINDINGS AND DECISION',
             'Pilot-matched training gives a modest improvement: 0.219 dB in equal-cell NMSE and 0.01758 percentage points in BER versus the paired 21-pilot control. One of the three training seeds worsens on both metrics.',
             'The retrained model leads fresh matched-budget sparse by 1.317 dB in equal-cell NMSE, but sparse wins all six 0/5 dB cells, pooled linear NMSE, and overall BER (1.36251% versus 1.43903%). This is not an overall baseline win.',
             'Unregularized EM continuation is rejected for the ordinary range: held-out validation chooses zero steps at every tested SNR, and forced one/two/five-step tails worsen both NMSE and BER.',
             'Keep the retrained checkpoints as candidates, retain zero continuation for ordinary SNR, and address high-SNR bypass separately before promoting a new model or updating the manuscript.', '',
             'MAIN COMPARISON', '']
    rows = []
    for name, label in labels:
        row = methods[name]
        rows.append([label, f"{row['primary_nmse_db']:.3f}",
                     f"{row['pooled_ratio_of_sums_db']:.3f}",
                     f"{100 * bit_methods[name]['ber']:.5f}",
                     f"{times[name]['p90_ms']:.3f}", f"{times[name]['median_ms']:.3f}"])
    lines += [table(['Method', 'Equal-cell NMSE dB', 'Pooled NMSE dB', 'BER %', 'p90 ms', 'Median ms'], rows), '',
              'Lower is better for every table column. Neural rows average scores from three separately evaluated fits; they are not ensembles.',
              'Group latency pools single-model timings across the three fits; it is not the cost of running three models.',
              f"Perfect-channel magnitude-detector BER: {100 * bit_methods['perfect_csi']['ber']:.5f}%.", '']
    lines += ['PRESPECIFIED PAIRED CONTRASTS',
              'Positive NMSE gain or BER reduction means the candidate improves. Adjusted intervals cover eight primary outcomes (four contrasts x two metrics).', '']
    contrast_labels = {'pilot_count': '30-pilot vs paired 21-pilot training',
                       'old_continuation': 'Selected EM vs original model alone',
                       'matched_continuation': 'Selected EM vs 30-pilot model alone',
                       'matched_vs_sparse': '30-pilot selected pipeline vs fresh matched sparse'}
    rows = []
    for key, label in contrast_labels.items():
        a = nmse['primary_contrasts'][key]; b = ber['overall']['contrasts'][key]
        rows.append([label, f"{a['gain_db']:.3f}", interval(a['eight_outcome_simultaneous95_ci']),
                     f"{100 * b['ber_reduction']:.5f}", interval(b['bonferroni8_interval'], 100., 5)])
    lines += [table(['Candidate vs reference', 'NMSE gain dB', 'Adjusted interval', 'BER reduction pp', 'Adjusted interval pp'], rows), '',
              'Intervals use 20,000 paired resamples, with case indices shared across SNR within each path regime.',
              'There are 1,920 test channel worlds and 384 coupled index clusters. Intervals are conditional on the fitted models and frozen choices; they do not include training-population or selection uncertainty.', '']
    lines += ['HELD-OUT EM-TAIL CHOICES', 'Zero, one, two and five exact EM steps were compared on development observations only.', '']
    rows = []
    for family in ('old', 'context21', 'context30'):
        rows.append([family] + [nmse['tail_choices'][family][str(s)]['selected_tail'] for s in (0, 5, 10, 15, 20)])
    lines += [table(['Model family', '0 dB', '5 dB', '10 dB', '15 dB', '20 dB'], rows), '',
              'Continuation starts at the learned estimate and uses the same pilots, exact Rician posterior and unregularized M-step. It does not restart from a spectral initialization.',
              'A reduction in fitting NLL does not imply an improvement in held-out NLL, NMSE or BER.', '']
    rows = []
    for family in ('old', 'context21', 'context30'):
        for tail in (0, 1, 2, 5):
            name = f'{family}_tail{tail}_mean'
            b = fixed_ber['overall']['methods'].get(name, bit_methods.get(name))
            rows.append([family, tail, f"{methods[name]['primary_nmse_db']:.3f}",
                         f"{100 * b['ber']:.5f}" if b else 'not measured', f"{times[name]['p90_ms']:.3f}"])
    lines += [table(['Fixed-tail diagnostic', 'Steps', 'Equal-cell NMSE dB', 'BER %', 'p90 ms'], rows), '',
              'These fixed-tail results are diagnostic. They are not used to replace the development-selected choice.',
              'The fixed-tail BER supplement was frozen before test generation. It covers the original and 30-pilot-trained families, uses identical core data, and leaves all primary contrasts unchanged.', '']
    lines += ['RUNTIME MATCHING',
              'Calibration and final timings ran with other experiment jobs idle, using one CPU thread, batch size one, initialization and all continuation overhead included.',
              'Fresh sparse choices use the same development measurements and a measured family-specific latency cap. The full declared sparse grid was eligible for timing; old cached losses only saved computation.', '']
    rows = []
    for family, row in timing['cap_verification'].items():
        rows.append([family, f"{row['cap_ms']:.3f}", f"{row['selected_pipeline_p90_ms']:.3f}",
                     f"{row['sparse_p90_ms']:.3f}", row['sparse_exceeds_calibrated_cap'], row['sparse_exceeds_fresh_pipeline']])
    lines += [table(['Family', 'Calibrated cap ms', 'Final learned p90 ms', 'Final sparse p90 ms', 'Above cap?', 'Above learned?'], rows), '',
              'Fresh context30 sparse meets its frozen 5.787 ms cap, but its final 5.780 ms p90 is about 10% slower than the remeasured 5.256 ms selected neural pipeline. The final measurements are budget matched by the calibration rule, not exactly equal latency.',
              'Final timing variation is reported directly; there is no post-test sparse reselection.',
              'Runtime is a measurement of these CPU implementations, not hardware-independent algorithmic complexity.', '']
    lines += ['WHERE IMPROVEMENTS HOLD OR FAIL', '']
    for key, label in contrast_labels.items():
        a = nmse['primary_contrasts'][key]
        lines += [label + ':',
                  f"  Fit-specific aggregate NMSE gains (dB): {a['seed_specific_gains_db']}",
                  f"  Worst ordinary cell: {a['worst_cell']} ({a['worst_cell_gain_db']:.3f} dB gain).",
                  '  Negative point-estimate cells: ' + (', '.join(a['negative_point_cells']) or 'none') + '.',
                  '  Descriptive pointwise 95% harm cells: ' + (', '.join(a['pointwise95_harm_cells']) or 'none') + '.']
    risk_rows = []
    risk_names = {'old_tail0_mean', 'context21_tail0_mean', 'context30_selected_mean',
                  'sparse_fresh_context30', 'prior_sparse_full', 'em100', 'optshrink'}
    for name, label in labels:
        if name in risk_names:
            a = methods[name]
            risk_rows.append([label, f"{a['case_nmse_p95']:.5g}", f"{a['case_nmse_p99']:.5g}",
                              f"{100 * a['severe_rate']:.3f}"])
    lines += ['', table(['Tail diagnostic', 'p95 case NMSE (linear)', 'p99 case NMSE (linear)', 'Cases NMSE > 1 (%)'], risk_rows),
              'For neural groups, quantiles are averaged across three individual fits. These diagnostics are not additional primary tests.']
    lines += ['', 'DESCRIPTIVE GUARDS (NOT USED FOR SELECTION)',
              'The ordinary-range choices are reused at these out-of-range points, with the SNR lookup clamped to 20 dB. These are diagnostics, not a tuned high-SNR bypass.',
              'Earlier bypass thresholds were calibrated for the original weights. If a retrained model is adopted, those thresholds need separate recalibration and confirmation.', '']
    guard_names = ['old_tail0_mean', 'old_tail5_mean', 'context21_tail0_mean', 'context30_tail0_mean',
                   'context30_tail5_mean', 'context30_selected_mean', 'sparse_fresh_context30', 'em100']
    rows = []
    for guard in nmse['guards']:
        for name in guard_names:
            rows.append([guard['cell'], name, f"{guard['methods'][name]['nmse_db']:.3f}"])
    lines += [table(['Guard', 'Estimator', 'NMSE dB'], rows), '', 'COMPLETED TRAINING', '']
    train_rows = []
    for name, a in audits['training']['selected'].items():
        train_rows.append([name, a['epoch'], f"{a['val30_nll']:.6f}", a['updates'], f"{a['training_seconds'] / 60.:.2f}"])
    lines += [table(['Model', 'Selected epoch', 'Common validation NLL', 'Updates completed', 'Update-loop minutes'], train_rows),
              'Both fresh arms complete 10,240 optimizer updates. Update-loop time excludes validation and was recorded with concurrent training workers; it is not isolated inference latency.', '', 'PROTOCOL AND INTERPRETATION',
              'Three paired seeds; two fresh training arms; identical 188-parameter ten-stage architecture; 16,384 training and 2,048 validation cases; 20 epochs, batch 32, Adam at 0.001.',
              'Both arms use the same simulated channels, initialization, batch orders and nine held-out measurements. The 21-pilot context is nested inside the 30-pilot context.',
              'A 39-pilot bank is needed only for offline training/validation. Every deployed estimator receives exactly 30 pilot measurements.',
              'Training and checkpoint/tail/sparse selection use observations and design strata, not the true channel. Test truth is used for simulation and scoring only.',
              'Both fresh arms select checkpoints by the same 30-context held-out likelihood. Therefore the fresh 21-pilot control is not an exact reproduction of the original checkpoint-selection procedure.',
              'Checkpoint validation covers 126 broader strata: SNR -10 to 20 dB, RSR 5/12/20 dB, and sparse through dense path regimes. Improved broad validation likelihood need not improve every fit on the narrower primary NMSE test.',
              'The paired fresh comparison isolates training context count under shared selection. Comparisons against the original checkpoint also change the training data and selection protocol.',
              'Changing context count changes both the explicit pilot-count feature and the information/noise properties of the training estimate. This study does not isolate feature extrapolation as the sole mechanism.',
              'Main evaluation: N=32, K=3, P=30, RSR=12 dB; SNR=0/5/10/15/20 dB; two paths, four paths, and random 3-7 paths per user; 128 channels per cell.',
              'BER: 256 fresh joint QPSK vectors per channel, 2,949,120 bits per estimator. All estimators share identical magnitude-domain data and the exact 64-hypothesis Rician ML detector.',
              'A zero observed BER is not proof of zero error probability; the finite channel bank limits rare-error conclusions.',
              'All choices were frozen before generating test/guard data. Original source files, checkpoints, previous results and manuscript were preserved.', '',
              'VERIFICATION',
              'Independent design, completed-training, prediction/selection, core BER, fixed-tail BER and final-runtime audits passed.',
              'Verified nested pilot contexts, held-out isolation, paired initial states/orders/masks, checkpoint minima, source/checkpoint hashes, exact continuation parity, monotonic fitting NLL, test-error recomputation, selected-tail array identity, detector reconstruction and common data.',
              'The runtime audit independently recomputed 1,260 overall/per-SNR statistics from 4,950 positive finite measurements across 55 methods: exact agreement. Fifteen neural groups pool individual fit timings rather than summing an ensemble cost. Raw timing outliers were retained.',
              'Frozen final selection SHA256: ' + digest, '', 'ARTIFACTS',
              'training_summary.json; training_curves.png',
              'evaluation/nmse_summary.json; evaluation/pilot_em_nmse.png; evaluation/pilot_em_runtime.png',
              'evaluation/final_timing.json; evaluation/selection.json',
              'ber/summary.json; ber/ber.png',
              'ber_fixed_tails/summary.json; ber_fixed_tails/ber_fixed_tails.png',
              'audit/design.json; audit/training.json; audit/study.json; audit/ber.json; audit/fixed_tails.json; audit/timing.json', '']
    (ROOT / 'RESULTS.txt').write_text('\n'.join(lines))
    training_figure()
    print(ROOT / 'RESULTS.txt')


if __name__ == '__main__':
    main()
