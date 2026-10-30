"""Reporting only; consumes frozen results and cannot select an estimator."""
import json
from pathlib import Path
import time
import numpy as np
from . import study
from .analysis import belongs
from tcom_followups_20261017.common import sha,write_json


def read(path):
    return json.loads(path.read_text())


def main():
    out=study.OUT
    policy,digest=study.check_selection()
    nmse=read(out/'nmse_summary.json')
    ber=read(out/'ber/summary.json')
    windows=[read(out/f'timing_window{i}.json') for i in (1,2)]
    audits={name:read(out/f'audit/{name}.json') for name in ('selection','test','summary','boundary','timing')}
    audits['ber']=read(out/'ber/audit.json')
    assert all(a['passed'] for a in audits.values())
    passed=all(w['cap_verification']['passed'] for w in windows)
    labels=[('original_mean','Retrained learned only'),('old_bypass_mean','Old EM100 bypass'),
            ('prior_repair_mean','Earlier transition repair'),('current_repair_mean','Canonical latest'),
            ('repair_mean','Native latest (same algorithm)'),('em100','Direct spectral EM100'),
            ('compact_sparse','Frozen compact sparse/EM'),('prior_sparse_full','Full sparse'),
            ('optshrink','OptShrink')]
    raw=dict(windows[0]['raw_samples'])
    raw.update({name:[r for part in parts for r in raw[part]] for name,parts in study.GROUPS.items()})
    lines=[
        'RUNTIME IMPLEMENTATION COMPARISON — FRESH FROZEN EVALUATION',
        'The paper has not been updated. All earlier sources, weights, policies and results remain unchanged.',
        '',
        'IMPLEMENTATION',
        'The same frozen 188-parameter learned estimator now executes in NumPy/SciPy float64/complex128. It retains all 10 stages, observable features, learned weights, posterior functions, and smooth Hankel solves. Pilot-only linear algebra and shape constants are reused; framework overhead is reduced.',
        'All continuation budgets (including 100), initializations, lookup choices and final noise-scaled filters are unchanged. No retraining, new sparse selection, precision reduction, posterior approximation, early stopping or cached observation answers.',
        'Every learned baseline uses the same optimized learned kernel. Canonical latest is an explicit implementation control. EM100 and sparse kernels are unchanged. prior_compact and compact_sparse are identical controls, not independent evidence.',
        '',
        'FIXED RUNTIME TARGET',
        f'Both prespecified whole-grid p90 windows <=10.74 ms: {passed}. Audit validity is separate from target acceptance.',
    ]
    for w in windows:
        methods=w['methods']
        lines.append(f"Window{w['window']}: native whole-grid p90={methods['repair_mean']['p90_ms']:.6f}ms; canonical={methods['current_repair_mean']['p90_ms']:.6f}ms; native weak-warm100 p90={methods['repair_mean']['weak_warm100']['p90_ms']:.6f}ms; compact={methods['compact_sparse']['p90_ms']:.6f}ms.")
        excess={k:v['p90_ms'] for k,v in methods['repair_mean']['by_condition'].items() if v['p90_ms']>10.74}
        lines.append('  Native conditional p90 above the pooled target: '+json.dumps(excess,sort_keys=True))
        lines.append('  Native per-fit whole-grid p90: '+json.dumps({str(s):methods[f'repair_{s}']['p90_ms'] for s in study.SEEDS}))
    lines += [
        'Timing uses batch size one and one CPU thread, with complete calls including conversion, dispatch, initialization, learned inference, EM and filtering; only fixed loading/shape constants are outside the warm call. Three fresh timing indices per full 153-condition window. Window 1 precedes fresh accuracy generation; window 2 follows completion of all owned accuracy workers. No sentinel normalization, moving budget or repeated attempts until passing.',
        '',
        'NUMERICAL PRESERVATION',
        f"All 153 main conditions passed the frozen coefficient tolerance atol=rtol=1e-10. Maximum absolute native/canonical channel difference={nmse['numerical_parity']['maximum_absolute_error']:.12g}; bitwise-identical condition/fit arrays={nmse['numerical_parity']['bitwise_equal_cell_model_pairs']}/459.",
        'The 54 additional dense boundary conditions are audited separately and excluded from main-grid weighting. Numerical equivalence is not a new accuracy gain or a proof of population BER equivalence.',
        '',
        'FRESH ACCURACY AND RUNTIME COMPARISONS',
        'NMSE is equal-cell mean of dB ratio-of-sums; BER is percent. Runtime is within-region p90 from timing window1. Three fits average individual metrics, not predictions.',
    ]
    for region,title in [('all','ALL 153 CONDITIONS'),('weak_warm100','WEAK REFERENCE WARM100: 18 CONDITIONS'),
                         ('strong_high','STRONG REFERENCE 40–60 dB: 30 CONDITIONS'),('strong_transition','EARLIER STRONG TRANSITION: 18 CONDITIONS')]:
        lines += ['',title,f"{'Method':38s} {'NMSEdB':>12s} {'BER%':>12s} {'p90ms':>12s}"]
        for name,label in labels:
            m=nmse['aggregates'][region]['methods'][name]
            b=ber['regions'][region]['methods'][name]
            latency=np.quantile([r['ms'] for r in raw[name] if belongs(region,r['snr'],r['rsr'],r['regime'])],.9)
            lines.append(f"{label:38s} {m['nmse_db']:12.6f} {100*b['ber']:12.6f} {latency:12.6f}")
    lines += ['', 'CANONICAL VERSUS NATIVE']
    for region in ('all','weak_warm100'):
        n=nmse['aggregates'][region]['contrasts']['repair_vs_current']
        b=ber['regions'][region]['contrasts']['repair_vs_current']
        lines.append(f"{region}: NMSE gain={n['gain_db']:.12g}dB, Bonferroni4 interval={n['bonferroni4_interval_db']}; BER reduction={b['ber_reduction']:.12g}, Bonferroni4 interval={b['bonferroni4_interval']}.")
    lines += ['', 'CELLWISE BASELINE COMPARISONS']
    for name,item in nmse['negative_cells'].items():
        lines.append(f"{name}: mean-negative={len(item['negative_cells'])}/153; descriptive pointwise95 harm={len(item['pointwise95_harm_cells'])}; worst={item['worst_cell']} ({item['worst_gain_db']:+.9g}dB).")
        if item['negative_cells']:
            lines.append('  Mean-negative cells: '+', '.join(item['negative_cells']))
    lines += [
        'These are paired implementation checks, not claims of scientific superiority. Bonferroni4 applies only to the canonical/native contrast in two regions times NMSE/BER. All other intervals and baseline comparisons are descriptive.',
        'BER uses all 64 joint Gray-QPSK vectors per channel, identical fresh data magnitudes/reference/noise and the same exact Rician magnitude-domain detector. Non-bitwise channel arrays are independently decoded. Zero errors and collapsed bootstrap intervals do not establish zero BER or noninferiority; bits within a channel are not independent channel trials.',
        '',
        'INTERPRETATION FOR THE PAPER',
        'This work supports the accuracy-versus-computation evidence for the existing estimator. It is an implementation contribution, not a new estimator principle. High-SNR gains belong to the classical spectral EM100 plus noise-scaled Hankel branch; they are not attributable to the learned controller.',
        'The separately developed observable paired raw-channel-MSE criterion currently calibrates policies offline. Its moment identity is related to established nonlinear-estimation constructions; a distinct contribution needs careful prior-art positioning and stronger analysis of selection reliability. The bare identity is not claimed as a theoretical breakthrough.',
        '',
        'PROVENANCE',
        f"Protocol SHA256: {policy['protocol_sha256']}",
        f"Policy SHA256: {digest}",
        f"BER protocol SHA256: {policy['ber_protocol_sha256']}",
        'Fresh seeds: test 2026102002, data 2026102003, timing 2026102004, holdout 2026102005, bootstrap 2026102009.',
        'All independent audits passed: '+', '.join(audits),
        'Main test: 128 anchors for each of 9 reference/path settings, reused across 17 SNRs; 153 conditions. Bootstrap: 20,000 paired local-index resamples, 384 coupled clusters. Boundary: 54 additional conditions.',
        'Results remain conditional on the simulated channel distribution, known reference, correctly supplied noise variance, fixed CPU/library environment and tested grids.',
    ]
    (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    record=dict(phase='complete' if passed else 'comparison_complete_runtime_target_failed',
                comparison_complete=True,runtime_target_met_in_both_windows=passed,all_audits_passed=True,
                paper_updated=False,protocol_sha256=policy['protocol_sha256'],selection_sha256=digest,
                result_sha256=sha(out/'RESULTS.txt'),report_source_sha256=sha(__file__),completed_unix=time.time())
    write_json(out/'completion.json',record)
    write_json(out/'status.json',record)
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
