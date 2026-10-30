"""Consolidate completed runtime and bypass evidence without changing models."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results/tcom_runtime_bypass_20261017'


def read(name):
    return json.loads((ROOT / name).read_text())


def interval(values):
    return '[' + ', '.join(f'{x:.3f}' for x in values) + ']'


def main():
    runtime = read('runtime/nmse_summary.json')
    ber = read('runtime_ber/summary.json')
    bypass = read('bypass/summary.json')
    noise = read('bypass_noise_sensitivity/summary.json')
    # Require completed independent audits before presenting a final report.
    for path in ('runtime_audit.json', 'bypass_audit/audit.json', 'bypass_audit/study_audit.json'):
        audit = read(path)
        assert audit.get('passed', False), path
    assert bypass['ordinary_exact_identity']
    timing = bypass['timing']
    example = next(r for r in bypass['cells'] if r['regime'] == 'random3to7' and r['rsr'] == 12 and r['snr'] == 60)
    labels = [('original_mean', 'Original learned estimator'),
              ('off_mean', 'Always disable filtering; same ten learned steps'),
              ('fast_bypass_mean', 'Calibrated ten-step bypass'),
              ('em100', 'Always exact EM100'),
              ('em100_bypass_mean', 'Calibrated EM100 fallback')]
    lines = ['Completed high-SNR bypass experiment', '',
        'This is a separately calibrated engineering safeguard around the frozen estimator. '
        'No neural weights were trained or changed. At inference it uses only the supplied '
        'noise variance, number of users, and known reference power. It executes one branch '
        'per channel realization; it does not run both candidates and pick using the true channel.', '',
        'The nominal SNR proxy is 10 log10(K / supplied noise variance), using the existing '
        'unit-power-per-user normalization. Reference power selects the nearest bin in '
        '{5, 12, 20} dB. Filtering remains unchanged at or below the selected cutoff; '
        'the fallback is used strictly above it. A separate guard preserves the original '
        'estimator at proxy SNR <=20 dB.', '',
        'Reference strength dB | ten-step cutoff dB | EM100 cutoff dB']
    for ref in ('5', '12', '20'):
        lines.append(f"{ref} | {bypass['policies']['same_steps'][ref]:g} | {bypass['policies']['em100'][ref]:g}")
    lines += ['',
        'Calibration used 216 new development anchors: 24 per path regime/reference pair. '
        'For each anchor, 21 pilot magnitudes were available for initialization and fitting; '
        'nine were withheld for exact Rician likelihood scoring. The same cutoff is shared '
        'across all path regimes and all three trained fits. The 13-point SNR grid spans '
        '10 to 60 dB. Both policy tables and all development losses were frozen before '
        'generating the fresh test bank.', '',
        f"Test scope: {bypass['unique_test_anchors']} channel/reference anchors, "
        f"{bypass['condition_cases']} condition-cases over 117 cells. Within an anchor, "
        'the channel, pilots, reference and standardized noise remain fixed across SNR. '
        'The repeated SNR cases are not independent channel draws. Confidence intervals '
        'resample 192 case-index clusters, sharing indices across SNR and reference within '
        'each path regime; trained fits and selected policies are held fixed.', '',
        'Equal-cell NMSE over the prespecified high-SNR grid (SNR >20 dB; lower is better)',
        'Method | mean NMSE dB']
    for name, label in labels:
        lines.append(f"{label} | {bypass['aggregates']['high_snr']['methods'][name]['nmse_db']:.3f}")
    for arm in ('fast_bypass_mean', 'em100_bypass_mean'):
        result = bypass['aggregates']['high_snr']['gains_vs_original'][arm]
        lines.append(f"{dict(labels)[arm]}: gain over original {result['gain_db']:.3f} dB; "
                     f"simultaneous 95% interval {interval(result['simultaneous95_ci_db'])}.")
    lines += ['', 'Concrete example: random 3-7 paths per user, RSR 12 dB, SNR 60 dB',
              'Method | NMSE dB | fresh batch-one p90 ms at this SNR/reference (pooled path regimes)']
    for name, label in labels:
        lines.append(f"{label} | {example['methods'][name]['nmse_db']:.3f} | "
                     f"{timing[name]['by_condition']['rsr12_snr60']['p90_ms']:.3f}")
    lines += ['',
        'All reference/path conditions at SNR 60 dB',
        'Reference dB | path regime | original NMSE dB | fast bypass NMSE dB | EM100 fallback NMSE dB']
    for row in bypass['cells']:
        if row['snr'] == 60:
            lines.append(f"{row['rsr']:g} | {row['regime']} | "
                         f"{row['methods']['original_mean']['nmse_db']:.3f} | "
                         f"{row['methods']['fast_bypass_mean']['nmse_db']:.3f} | "
                         f"{row['methods']['em100_bypass_mean']['nmse_db']:.3f}")
    lines += ['',
        'The weak-reference RSR 5 dB cases retain substantial error floors even with '
        'EM100. The 12 dB example must not be generalized to every reference strength.', '',
        f"Same-window branch p90 timings: original estimator directly "
        f"{timing['original_mean']['p90_ms']:.3f} ms; fast bypass's filtered branch "
        f"{timing['fast_bypass_mean']['by_branch']['original']['p90_ms']:.3f} ms and "
        f"unfiltered branch {timing['fast_bypass_mean']['by_branch']['off']['p90_ms']:.3f} ms; "
        f"EM100 bypass's filtered branch {timing['em100_bypass_mean']['by_branch']['original']['p90_ms']:.3f} ms "
        f"and EM100 branch {timing['em100_bypass_mean']['by_branch']['em100']['p90_ms']:.3f} ms.", '',
        'Implementation distinction: the EM100 reference point in the first runtime figure '
        'uses the original Torch implementation (13.27 ms). This bypass uses the numerically '
        'equivalent optimized NumPy/SciPy implementation (about 5.25 ms direct in the new '
        'timing window). The original timing is not an unavoidable cost of EM100. Optimized '
        'zero-penalty EM was also eligible in the runtime-matched candidate search. Cross-study '
        'timings use different input grids and windows; compare branches within the same window.', '',
        'All tested 10/15/20 dB predictions are bitwise identical to the original estimator '
        'under correctly supplied noise. The ordinary-range BER therefore remains unchanged '
        'for the same detector and data. A new high-SNR BER claim is not made here.', '',
        'Remaining limitations and costs']
    for arm in ('fast_bypass_mean', 'em100_bypass_mean'):
        negative = bypass['negative_cells'][arm]
        worst_gain = min(bypass['cells'], key=lambda r: r['gains_vs_original'][arm]['gain_db'])
        lines.append(f"{dict(labels)[arm]}: {len(negative['point_estimate_cells'])} of 117 cells "
                     f"have worse point-estimate NMSE than the original; "
                     f"{len(negative['pointwise95_harm_cells'])} have pointwise 95% intervals wholly on the harmful side.")
        if negative['point_estimate_cells']:
            harm = worst_gain['gains_vs_original'][arm]
            lines.append(f"Material exception: {worst_gain['cell']}, original "
                         f"{worst_gain['methods']['original_mean']['nmse_db']:.3f} dB versus bypass "
                         f"{worst_gain['methods'][arm]['nmse_db']:.3f} dB. Gain "
                         f"{harm['gain_db']:.3f} dB; pointwise 95% interval {interval(harm['ci95_db'])}. "
                         'This variant should not be described as a universally safe improvement.')
        rows = [r for r in bypass['cells'] if r['snr'] > 20]
        worst = max(rows, key=lambda r: r['penalties_vs_always_off'][arm]['penalty_db'])
        penalty = worst['penalties_vs_always_off'][arm]
        lines.append(f"Largest high-SNR penalty versus always disabling filtering: "
                     f"{penalty['penalty_db']:.3f} dB at {worst['cell']}; "
                     f"pointwise 95% interval {interval(penalty['ci95_db'])}. "
                     'The validation-selected cutoff was retained without test-based adjustment.')
    residual = next(r for r in bypass['residual_60db'] if r['regime'] == 'random3to7' and r['rsr'] == 12)
    lines += [f"In the 60 dB example, ten unfiltered learned steps still trail exact EM100 by "
        f"{residual['em100_gain_vs_off10_db']:.3f} dB. Disabling filtering removes the harmful "
        'filter action but does not remove the finite-iteration error. The EM100 fallback '
        'addresses that limitation at a separately measured computation cost.', '',
        'Supplied-noise sensitivity (RSR 12 dB, random 3-7 paths, same 64 fresh anchors)',
        'Actual SNR | supplied variance error dB | receiver SNR proxy | fast bypass fraction | '
        'fast-bypass NMSE dB | EM100-bypass NMSE dB']
    for row in noise['cells']:
        lines.append(f"{row['actual_snr_db']:g} | {row['variance_offset_db']:+g} | "
                     f"{row['supplied_snr_proxy_db']:g} | {row['fast_bypass_rate']:.0%} | "
                     f"{row['methods']['fast_bypass_mean']['nmse_db']:.3f} | "
                     f"{row['methods']['em100_bypass_mean']['nmse_db']:.3f}")
    lines += ['',
        'A positive supplied-variance error means the receiver assumes more noise than was '
        'actually present, reducing its SNR proxy. The guarantee through 20 dB applies to '
        'that proxy, not to unknown actual SNR when noise is miscalibrated. These nine '
        'diagnostic cells do not establish robustness to arbitrary mismatch. Threshold '
        'transfer beyond the tested dimensions, reference bins, channel distribution and '
        'power normalization also remains unverified.', '',
        'Development-only resampling also shows cutoff-selection uncertainty: at RSR 5, '
        'EM100 selects the frozen 40 dB cutoff in 63.4% of resamples and 35 dB in 36.35%. '
        'The test confidence intervals condition on the selected policy and do not include '
        'this calibration uncertainty. Centered and compensated likelihood summation '
        'confirm that numerical cancellation does not explain the selected thresholds.', '',
        'Timing includes initialization and observable-only dispatch, with preloaded models '
        'and one requested CPU thread in float64/complex128. The filtered and unfiltered '
        'branches each retain their own computational cost. There is no simultaneous dual '
        'inference or online held-out pilot scoring. These local p90 timings are not hard '
        'real-time guarantees.', '',
        'Checks passed: identical frozen checkpoint states; exact ordinary-range preservation; '
        'filter-off equivalence to a zero gate; high-SNR EM100 numerical parity; heterogeneous '
        'batch dispatch; one branch per world; independent cutoff recomputation; unchanged '
        'source/checkpoint/development hashes; policy frozen before test; raw test-error '
        'recomputation and fresh dispatch replays.', '',
        'Figures: bypass/bypass_nmse.png, bypass/bypass_gain.png, '
        'bypass_noise_sensitivity/noise_sensitivity.png.',
        'Full results: bypass/summary.json, bypass/timing.json, '
        'bypass_noise_sensitivity/summary.json, bypass/validation_diagnostic.json. '
        'Raw predictions and losses are retained.',
        'Independent audits: bypass_audit/audit.json and bypass_audit/study_audit.json.']
    bypass_text = '\n'.join(lines) + '\n'
    (ROOT / 'RESULTS_BYPASS.txt').write_text(bypass_text)
    gain = -runtime['contrasts']['sparse_matched']['gain_db']
    lead = [
        'Runtime-matched sparse comparison and high-SNR bypass: completed evidence', '',
        f"At similar measured runtime, the learned estimator has a {gain:.3f} dB advantage "
        'on the prespecified equal-cell NMSE score. Sparse remains better at low SNR, '
        'on pooled linear NMSE, and on aggregate BER. This supports a conditional '
        'efficiency claim, not universal superiority.',
        f"Measured p90 latency: learned {runtime['timing']['learned_mean']['p90_ms']:.3f} ms; "
        f"matched sparse {runtime['timing']['sparse_matched']['p90_ms']:.3f} ms. "
        f"BER: learned {100*ber['overall']['methods']['learned_mean']['ber']:.5f}%; "
        f"matched sparse {100*ber['overall']['methods']['sparse_matched']['ber']:.5f}%.", '',
        'The high-SNR safeguard is implemented and tested with separate development and '
        'test channels. It preserves the original ordinary-range estimator and switches '
        'to unfiltered ten-step inference or EM100 using only known receiver quantities. '
        'The detailed report includes intermediate-SNR losses, finite-iteration limits, '
        'noise sensitivity, latency, and conditional uncertainty.', '',
        'Material exception: the fast ten-step bypass worsens the two-path, RSR 5 dB, '
        'SNR 45 dB cell by about 6.08 dB. The separate EM100 fallback has no negative '
        'cell-level mean gain on the tested grid and is the better-supported candidate '
        'for the paper, with its additional runtime and calibration limits stated. '
        'Neither result establishes a universal guarantee.', '',
        'For the merged paper: present runtime versus accuracy with the low-SNR/BER '
        'tradeoff, followed by the calibrated bypass as an operating-range safeguard. '
        'Describe it as an engineering extension to the frozen model. These experiments '
        'strengthen the empirical evidence but do not by themselves establish publication '
        'novelty or acceptance readiness.', '',
        'Frozen checkpoints and original manuscript/source files were not edited. '
        'New implementation: tcom_runtime_bypass_20261017/. '
        'All artifacts for this study: results/tcom_runtime_bypass_20261017/.', '',
        '=' * 72, '', (ROOT / 'RESULTS_RUNTIME.txt').read_text(), '=' * 72, '', bypass_text]
    (ROOT / 'RESULTS.txt').write_text('\n'.join(lead))
    print(ROOT / 'RESULTS.txt')


if __name__ == '__main__':
    main()
