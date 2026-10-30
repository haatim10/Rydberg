"""Consolidate the frozen bypass study; never edits the manuscript."""
import json
from pathlib import Path
import numpy as np
from .analysis import REGIONS, belongs, GROUPS
from .study import OUT, RSRS, sha


def read(path):
    return json.loads((OUT / path).read_text())


def table(headers, rows):
    cells = [[str(value) for value in row] for row in [headers] + rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    line = lambda row: '  '.join(v.ljust(w) for v, w in zip(row, widths))
    return '\n'.join([line(cells[0]), line(['-' * w for w in widths])] + [line(row) for row in cells[1:]])


def interval(values, scale=1., digits=3):
    return '[' + ', '.join(f'{scale * value:.{digits}f}' for value in values) + ']'


LABELS = [('original_mean', 'Retrained30; always filtered'),
          ('em100_bypass_mean', 'Retrained30 + EM100 bypass'),
          ('fast_bypass_mean', 'Retrained30 + ten-step bypass'),
          ('off_mean', 'Same ten steps; always filter off'),
          ('em100', 'Always optimized exact EM100'),
          ('compact_sparse', 'Fresh compact sparse; matched cap'),
          ('prior_sparse_matched', 'Frozen prior matched sparse'),
          ('prior_sparse_50ms', 'Frozen prior 50 ms sparse'),
          ('prior_sparse_full', 'Frozen full sparse'),
          ('optshrink', 'Frozen offline OptShrink')]


def main():
    nmse, ber, timing, policy = (read(p) for p in ('nmse_summary.json', 'ber/summary.json', 'timing.json', 'policy.json'))
    required_audits = ('checks.json', 'audit/calibration.json', 'audit/selection.json',
                       'audit/test.json', 'audit/summary.json', 'audit/timing.json',
                       'sensitivity/audit.json', 'ber/audit.json', 'audit/adverse_case_parity.json')
    for path in required_audits:
        assert read(path)['passed'], path
    digest = sha(OUT / 'policy.json')
    assert all(x['selection_sha256'] == digest for x in (nmse, ber, timing))
    assert nmse['timing_sha256'] == sha(OUT / 'timing.json')
    samples = dict(timing['raw_samples'])
    samples.update({name: [r for member in members for r in samples[member]] for name, members in GROUPS.items()})
    lines = ['RETRAINED ESTIMATOR WITH A CALIBRATED HIGH-SNR BYPASS',
             'Completed experimental comparison; manuscript not updated.', '',
             'MAIN FINDINGS',
             f"On the declared SNR>20 grid, the primary bypass improves equal-cell NMSE by {nmse['aggregates']['high_snr']['contrasts']['bypass_vs_original']['gain_db']:.3f} dB over the retrained model alone; the six-outcome-adjusted interval is {interval(nmse['aggregates']['high_snr']['contrasts']['bypass_vs_original']['bonferroni6_interval_db'])} dB.",
             f"Observed high-grid BER falls from {100 * ber['primary']['methods']['original_mean']['ber']:.6f}% to {100 * ber['primary']['methods']['em100_bypass_mean']['ber']:.6f}%. Ordinary-range predictions and BER are exactly unchanged on the shared test data.",
             'The bypass improves mean high-grid NMSE versus direct EM100 and compact sparse, but their BER point estimates are lower; those BER differences are unresolved by the adjusted intervals. OptShrink retains a stronger mean high-grid NMSE. This is a performance tradeoff, not baseline dominance.',
             'Eight conditions have worse point-estimate NMSE than the retrained model, with a worst observed loss of 16.415 dB. Their pointwise 95% intervals include zero because the finite bank contains influential rare failures. This uncertainty does not establish safety or negate the observed failures.',
             'Retain this as a candidate operating-range safeguard. A stronger transition/noise-robustness check is needed before describing it as a generally safe replacement. No thresholds or models were adjusted using this test bank.', '',
             'WHAT WAS TESTED',
             'The three completed 30-pilot-trained models remain frozen. A receiver-side rule chooses either the unchanged learned estimator or exact EM100 from spectral initialization. The selected branch runs alone; EM100 is not appended to a learned estimate.',
             'The rule uses supplied noise variance, the user count and known reference power. Its nominal SNR proxy is 10 log10(K/variance), under the existing unit-user-power normalization. It never reads the true channel or realized channel SNR.',
             'One cutoff per reference bin is calibrated on fresh 30-fitting/9-held-out pilot measurements, shared by all three fits and all path regimes. Filtering is retained at or below the cutoff. An explicit guard preserves the model at nominal SNR <=20 dB.', '',
             'FROZEN CUTOFFS', '']
    lines += [table(['Reference dB', 'Primary EM100 cutoff dB', 'Diagnostic ten-step cutoff dB'],
        [[r, policy['policies']['em100'][f'{r:g}'], policy['policies']['same_steps'][f'{r:g}']] for r in RSRS]),
        'Cutoff 60 means no bypass within the tested 0-60 dB domain. Deployment outside that domain is unvalidated.', '',
        'DEVELOPMENT TRADEOFF AT THE FIRST SWITCHED SNR',
        'Held-out NLL change (EM100 minus retrained): positive favors retaining filtering; negative favors EM100.', '']
    switch_rows = []
    for ref in RSRS:
        cutoff = policy['policies']['em100'][f'{ref:g}']
        chosen = next(row for row in policy['validation_tables']['em100'][f'{ref:g}'] if row['cutoff_db'] == cutoff)
        switched = [row for row in chosen['cells'] if row['snr'] > cutoff]
        if switched:
            row = min(switched, key=lambda x: x['snr'])
            switch_rows.append([ref, row['snr']] + [f'{value:.4f}' for value in row['by_regime']] + [f"{row['excess_nll']:.4f}"])
    lines += [table(['Reference dB', 'First switched SNR dB', 'L2 NLL change', 'L4 NLL change', 'Random3-7 NLL change', 'Mean change'], switch_rows),
        'The shared cutoff minimizes the average held-out criterion. It can trade a loss in one path regime for gains in others; this is not a per-regime safety guarantee.', '',
        'COMPARISON BY OPERATING RANGE',
        'Ordinary = 0-20 dB; transition = 22.5-45 dB; stress = 50 and 60 dB; high-SNR = all tested points above 20 dB. Reference levels are 5/12/20 dB; path regimes are 2, 4, and random 3-7 paths per user.',
        'Lower is better. NMSE averages condition-level dB ratio-of-sums scores equally; pooled NMSE is also shown because aggregation can change rankings.',
        'Neural groups average metrics from three separate fits, not an ensemble. Runtime pools single-fit calls and includes initialization, observable lookup and the selected branch.', '']
    for region in ('ordinary', 'transition', 'stress', 'high_snr', 'all'):
        b = ber['primary'] if region == 'high_snr' else ber['overall'] if region == 'all' else ber[region]
        n = nmse['aggregates'][region]
        lines += [region.upper() + f" ({n['cells']} conditions)", '']
        rows = []
        for name, label in LABELS:
            times = [r['ms'] for r in samples[name] if belongs(region, r['snr'])]
            rows.append([label, f"{n['methods'][name]['nmse_db']:.3f}",
                f"{n['methods'][name]['pooled_ratio_of_sums_db']:.3f}",
                f"{100 * b['methods'][name]['ber']:.6f}", f'{np.quantile(times, .9):.3f}', f'{np.median(times):.3f}'])
        lines += [table(['Estimator', 'Equal-cell NMSE dB', 'Pooled NMSE dB', 'BER %', 'p90 ms', 'Median ms'], rows), '']
    lines += ['PRIMARY PAIRED CONTRASTS (SNR >20 dB)',
              'Positive gain/reduction favors the EM100 bypass. Intervals use a Bonferroni correction across six prespecified outcomes: three comparisons x NMSE/BER.', '']
    rows = []
    for key, n in nmse['aggregates']['high_snr']['contrasts'].items():
        b = ber['primary']['contrasts'][key]
        rows.append([key, f"{n['gain_db']:.3f}", interval(n['bonferroni6_interval_db']),
                     f"{100 * b['ber_reduction']:.6f}", interval(b['bonferroni6_interval'], 100., 6)])
    lines += [table(['Comparison', 'NMSE gain dB', 'Adjusted interval', 'BER reduction pp', 'Adjusted interval pp'], rows), '',
              'Intervals condition on these three trained models, the frozen policies, sparse selections and declared condition grid. They do not include training-population or threshold-selection uncertainty.', '',
              'ADVERSE CELLS AND SEED VARIATION', '']
    for key, row in nmse['negative_cells'].items():
        worst = next(cell for cell in nmse['cells'] if cell['cell'] == row['worst_cell'])
        effect = worst['fast_bypass_vs_original'] if key == 'fast_bypass_vs_original' else worst['contrasts'][key]
        lines += [key + ':', f"  Negative point-estimate cells: {len(row['negative_cells'])}; descriptive pointwise 95% harm cells: {len(row['pointwise95_harm_cells'])}.",
                  f"  Worst cell: {row['worst_cell']}; gain {row['worst_gain_db']:.3f} dB; descriptive pointwise 95% interval {interval(effect['pointwise_ci95_db'])}.",
                  '  Negative cells: ' + (', '.join(row['negative_cells']) or 'none')]
    for key, row in nmse['aggregates']['high_snr']['contrasts'].items():
        lines.append(key + ' fit-specific NMSE gains: ' + str(row['seed_specific_gains_db']))
    lines += ['',
        'Post-test diagnosis of L2/reference 5 dB: two channel anchors dominate the direct EM100 error floor from 37.5 through 60 dB. Their individual NMSEs remain about 0.091 and 0.070. A targeted replay agrees with canonical 100-step EM within 3.34e-15; this is behavior of the specified initialized estimator, not a fast-backend discrepancy.',
        'This diagnosis did not change the policy or search for different starts/iteration counts. Channel-error harm need not imply BER harm under the fixed finite QPSK test; the two co-outcomes remain separate.']
    lines += ['', '60 dB STRESS CASES', '']
    rows = []
    for cell in nmse['cells']:
        if cell['snr'] != 60:
            continue
        b = next(row for row in ber['cells'] if row['regime'] == cell['regime']
                 and row['rsr_db'] == cell['rsr'] and row['snr_db'] == 60
                 and row['method'] == 'em100_bypass_mean')
        rows.append([cell['rsr'], cell['regime'],
            f"{cell['methods']['original_mean']['nmse_db']:.3f}",
            f"{cell['methods']['em100_bypass_mean']['nmse_db']:.3f}",
            f"{cell['methods']['em100']['nmse_db']:.3f}",
            f"{cell['methods']['compact_sparse']['nmse_db']:.3f}", f"{100 * b['ber']:.6f}"])
    lines += [table(['Reference dB', 'Paths', 'Retrained NMSE dB', 'Bypass NMSE dB',
                     'EM100 NMSE dB', 'Compact sparse NMSE dB', 'Bypass BER %'], rows), '']
    lines += ['', 'LATENCY AND BASELINE FAIRNESS',
              'The compact sparse grid contains the three structures selected in the preceding matched-training study, each with five positive penalties, plus exact EM100. It is a compact declared search, not a globally optimized sparse benchmark.',
              'Eligibility was frozen before development from isolated end-to-end timing: the cap is the maximum of filtered-branch and EM100-branch p90. Eligible candidates were selected separately for every nominal SNR/reference using the same held-out likelihood as the bypass.',
              'Seven of the 16 compact candidates met the frozen cap. The exact EM100 candidate measured 5.18354 ms against the 5.13045 ms cap and was excluded from compact selection without relaxing the rule; it remained a direct unrestricted baseline and the primary bypass branch.',
              'The older matched/50 ms/full sparse and OptShrink baselines are frozen transfer controls. Their previous lookup tables are clamped outside their calibration domain; they are not newly optimized high-SNR comparators.',
              'Final timing uses a separate isolated, single-thread, batch-one window on an Apple M2 CPU over the same condition grid. Raw outliers are retained. This is implementation-specific latency, not a hard real-time guarantee.',
              json.dumps(timing['cap_verification'], indent=2),
              f"The primary all-grid p90 exceeds the frozen cap by {100 * (timing['cap_verification']['primary_pipeline_p90_ms'] / timing['cap_verification']['calibrated_cap_ms'] - 1):.2f}%. Compact sparse stays below both the frozen cap and the measured pipeline runtime. No reselection followed this final timing.", '', 'PRIMARY BYPASS BRANCH TIMINGS', '']
    branch = timing['methods']['em100_bypass_mean']['by_branch']
    lines += [table(['Executed branch', 'p90 ms', 'Median ms', 'Calls'],
                    [[name, f"{r['p90_ms']:.3f}", f"{r['median_ms']:.3f}", r['calls']] for name, r in branch.items()]), '',
              'CUTOFF STABILITY ON DEVELOPMENT DATA', '']
    selection_audit = read('audit/selection.json')
    lines += [table(['Policy/reference', 'Frozen cutoff dB', 'Bootstrap frequency of frozen cutoff'],
                    [[name, r['selected_cutoff_db'], f"{100 * r['selected_cutoff_bootstrap_frequency']:.2f}%"]
                     for name, r in selection_audit['reviews'].items()]),
              'These frequencies diagnose the finite calibration bank. They are not probabilities of deployment optimality.', '',
              'SUPPLIED-NOISE SENSITIVITY',
              'Fixed-policy diagnostic: random3-7 paths, all three reference strengths, at each primary cutoff and +/-2.5 dB plus 20/60 dB, with supplied variance errors -3/0/+3 dB.', '']
    sensitivity = read('sensitivity/summary.json')
    rows = []
    for cell in sensitivity['cells']:
        method = cell['methods']['em100_bypass_mean']
        rows.append([cell['rsr'], cell['actual_snr_db'], cell['supplied_variance_offset_db'],
                     f"{method['nmse_db']:.3f}", f"{method['degradation_vs_correct_variance_db']:.3f}"])
    lines += [table(['Reference dB', 'Actual SNR dB', 'Variance error dB', 'Bypass NMSE dB', 'Loss vs correct noise dB'], rows),
              'Near the transition, a +3 dB supplied-variance error delays switching and produces observed losses of 1.942 and 2.310 dB at actual SNR 32.5 dB for reference levels 12 and 20 dB. The respective pointwise 95% intervals are [0.936, 2.920] and [1.294, 3.231] dB. The largest observed noise-mismatch loss, 7.458 dB at reference 5/SNR 37.5/+3 dB variance error, has a wide interval [-8.089, 13.261] dB. These are focused diagnostics, not simultaneous general robustness guarantees.',
              '192 unique channel/reference anchors share 64 local-index clusters. Physical observations remain fixed; the supplied variance affects both routing and inference. Positive loss means worse NMSE. No sensitivity BER was measured.', '',
              'INTERPRETATION AND LIMITS',
              'Ordinary-range prediction identity is checked, including actual wrapper calls. Identical predictions with the same detector/data imply identical BER in that range.',
              'High-SNR gains of the EM100 branch are inherited from classical EM100. The bypass is an engineering extension that preserves learned regularization where selected; it is not a newly learned controller or proof of new estimator novelty.',
              'The guarantee at <=20 dB refers to the supplied-noise proxy. Noise miscalibration can change both routing and estimation; separate fixed-policy sensitivity results are diagnostic only.',
              'Zero observed bit errors and degenerate empirical bootstrap intervals are not proof of zero BER or a rare-event bound. No extra bits were added after observing test outcomes.',
              'Each fit/estimator is evaluated on 3,760,128 bits over the full grid, of which 2,654,208 are in the high-SNR region. Reusing the same physical data across estimators preserves paired comparisons; it does not create independent bit trials across methods.',
              'The main study uses 432 development channel/reference anchors (48 per regime/reference) and 576 fresh test channel/reference anchors (64 per pair), each reused along 17 SNR points. There are 9,792 operating cases, not 9,792 independent channel draws.',
              'Paired resampling uses 192 local-index clusters, preserving the SNR/reference coupling within each of three path regimes. The BER study enumerates all 64 joint QPSK symbols once per channel, with independent data noise shared across methods and rescaled across SNR.',
              'All calibration, policies and baseline choices were frozen before test generation. Existing checkpoints, source files, previous results and manuscripts were preserved.', '',
              'VERIFICATION',
              'Preflight, calibration, development selection, test prediction, NMSE summary, final timing, supplied-noise sensitivity, BER and targeted adverse-case EM parity audits all passed.',
              'Audit records: ' + '; '.join(required_audits), '',
              'ARTIFACTS',
              'nmse_summary.json; bypass_nmse.png; bypass_gain.png; bypass_runtime.png',
              'ber/summary.json; ber/ber.png',
              'timing.json; policy.json; calibration.json; protocol.json; sensitivity/summary.json',
              'Frozen policy SHA256: ' + digest, '']
    (OUT / 'RESULTS.txt').write_text('\n'.join(lines))
    print(OUT / 'RESULTS.txt')


if __name__ == '__main__':
    main()
