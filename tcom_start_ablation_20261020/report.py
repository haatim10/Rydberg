"""Reporting only: assemble audited results without selecting any algorithm.

This file is deliberately outside the frozen scientific source list. It reads
completed artifacts and never trains, estimates, decodes, resamples or times.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import os

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/tcom_start_ablation_20261020'
SEEDS = (9101, 9102, 9103)
CAP_MS = 10.74
PRIMARY = ('adaptive_vs_global', 'adaptive_vs_classical')
SCOPES = ('all', 'weak_warm100', 'strong_high', 'ordinary')
SCOPE_LABELS = {
    'all': 'All 153 conditions',
    'weak_warm100': 'Weak reference, SNR 37.5–60 dB: 18 conditions',
    'strong_high': 'Strong references, SNR 40–60 dB: 30 conditions',
    'ordinary': 'Ordinary SNR 0–20 dB: 45 conditions',
}
LABELS = {
    'adaptive_mean': 'Adaptive learned start',
    'global_mean': 'Global learned schedule',
    'classical': 'Classical structured start',
    'spectral': 'Spectral start',
    'em10': 'EM10 start',
    'budget_adaptive_mean': 'Adaptive, extended budget',
    'budget_global_mean': 'Global, extended budget',
    'budget_classical': 'Classical, extended budget',
    'budget_spectral': 'Spectral, extended budget',
    'budget_em10': 'EM10, extended budget',
    'latest_mean': 'Frozen latest estimator',
    'em100': 'Direct EM100',
    'compact_sparse': 'Frozen compact sparse',
    'optshrink': 'Frozen OptShrink',
    'perfect_csi': 'Perfect CSI reference',
}
MAIN_METHODS = ('adaptive_mean', 'global_mean', 'classical', 'spectral', 'em10',
                'latest_mean', 'em100', 'compact_sparse', 'optshrink')
BUDGET_METHODS = ('budget_adaptive_mean', 'budget_global_mean',
                  'budget_classical', 'budget_spectral', 'budget_em10')
REQUIRED_AUDITS = ('selection', 'test', 'summary', 'timing')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        stream.write(value)
        temporary = stream.name
    os.replace(temporary, path)


def required_paths():
    return [
        'protocol.json', 'policy.json', 'classical_selection.json', 'budget_selection.json',
        'nmse_summary.json', 'raw_start_summary.json', 'ber/summary.json',
        'timing_window1.json', 'timing_window2.json', 'ber/audit.json',
        *(f'audit/{name}.json' for name in REQUIRED_AUDITS),
    ]


def evidence():
    missing = [name for name in required_paths() if not (OUT / name).exists()]
    if missing:
        raise RuntimeError('Final reporting requires completed artifacts: ' + ', '.join(missing))
    items = {name: load(OUT / name) for name in required_paths()}
    policy = items['policy.json']
    selection_hash = sha(OUT / 'policy.json')
    protocol_hash = sha(OUT / 'protocol.json')
    assert policy['protocol_sha256'] == protocol_hash
    protocol = items['protocol.json']
    for field in ('source_hashes', 'checkpoint_hashes'):
        for name, digest in protocol[field].items():
            assert sha(REPO / name) == digest, name
    for name, digest in policy['selection_evidence'].items():
        assert sha(OUT / name) == digest, name
    for name, digest in policy['ber_source_hashes'].items():
        assert sha(REPO / name) == digest, name
    assert policy['ber_protocol_sha256'] == sha(OUT / 'ber/protocol.json')
    nmse, raw, ber = (items[name] for name in
                      ('nmse_summary.json', 'raw_start_summary.json', 'ber/summary.json'))
    for result in (nmse, raw, ber):
        assert result['selection_sha256'] == selection_hash
    assert nmse['protocol_sha256'] == raw['protocol_sha256'] == protocol_hash
    assert ber['protocol_sha256'] == policy['ber_protocol_sha256']
    assert nmse['raw_start_summary_sha256'] == sha(OUT / 'raw_start_summary.json')
    assert nmse['raw_start_bootstrap_sha256'] == sha(OUT / 'raw_start_bootstrap.npz')
    assert not nmse['raw_start_diagnostic'] and raw['raw_start_diagnostic']
    assert len(nmse['cells']) == len(raw['cells']) == 153
    assert nmse['unique_test_anchors'] == 1152 and nmse['bootstrap_index_clusters'] == 384
    assert ber['regions']['all']['cells'] == 153

    audits = {name: items[f'audit/{name}.json'] for name in REQUIRED_AUDITS}
    audits['ber'] = items['ber/audit.json']
    for name, audit in audits.items():
        assert audit['passed'] is True, name
        assert audit['selection_sha256'] == selection_hash, name
    assert audits['test']['cells'] == audits['ber']['cells'] == 153
    assert audits['summary']['summary_sha256'] == sha(OUT / 'nmse_summary.json')
    assert audits['summary']['raw_start_summary_sha256'] == sha(OUT / 'raw_start_summary.json')
    assert audits['summary']['final_timing_bindings_verified'] is True
    assert audits['selection']['classical_selection_sha256'] == sha(OUT / 'classical_selection.json')
    assert audits['selection']['budget_selection_sha256'] == sha(OUT / 'budget_selection.json')
    assert audits['ber']['completed_unix'] >= (OUT / 'ber/summary.json').stat().st_mtime
    assert len(audits['ber']['cell_hashes']) == 153
    for name, digest in audits['ber']['cell_hashes'].items():
        assert sha(OUT / 'ber/cells' / (name + '.npz')) == digest
    for name, digest in audits['test']['test_prediction_hashes'].items():
        assert sha(OUT / 'test/cells' / name) == digest

    windows = {f'window{i}': items[f'timing_window{i}.json'] for i in (1, 2)}
    for key, window in windows.items():
        filename = f'timing_{key}.json'
        assert window['selection_sha256'] == selection_hash
        assert window['protocol_sha256'] == protocol_hash
        assert nmse['timing_hashes'][filename] == sha(OUT / filename)
        assert audits['timing']['timing_hashes'][filename] == sha(OUT / filename)
        assert nmse['timing_windows'][key]['methods'] == window['methods']
        assert nmse['timing_windows'][key]['fixed_cap_ms'] == CAP_MS
    assert nmse['timing'] == windows['window1']['methods']
    hashes = {name: sha(OUT / name) for name in required_paths()}
    return items, audits, windows, hashes


def interval(values, scale=1., digits=5):
    return '[' + ', '.join(f'{v * scale:.{digits}g}' for v in values) + ']'


def status(bounds):
    if bounds[0] > 0:
        return 'supports lower error for the adaptive start'
    if bounds[1] < 0:
        return 'supports higher error for the adaptive start'
    if bounds == [0., 0.]:
        return 'records an empirical tie; it does not prove population equivalence'
    return 'does not resolve an error difference at the adjusted confidence level'


def table(header, rows):
    strings = [[str(value) for value in header], *[[str(value) for value in row] for row in rows]]
    widths = [max(len(row[i]) for row in strings) for i in range(len(header))]
    lines = [' | '.join(value.ljust(widths[i]) for i, value in enumerate(strings[0]))]
    lines.append('-+-'.join('-' * w for w in widths))
    lines.extend(' | '.join(value.ljust(widths[i]) for i, value in enumerate(row))
                 for row in strings[1:])
    return '\n'.join(lines)


def ber_failures(ber, contrast):
    rows = [r for r in ber['cell_contrasts'] if r['contrast'] == contrast]
    assert len(rows) == 153
    def identity(row):
        return f"{row['regime']}_rsr{row['rsr_db']:g}_snr{row['snr_db']:g}"
    negative = [r for r in rows if r['ber_reduction'] < -1e-15]
    supported = [r for r in rows if r['bootstrap95'][1] < -1e-15]
    worst = sorted(rows, key=lambda r: r['ber_reduction'])[:5]
    return dict(negative_cells=[identity(r) for r in negative],
                pointwise95_harm_cells=[identity(r) for r in supported],
                worst_cells=[dict(cell=identity(r), ber_reduction=r['ber_reduction'],
                                  bootstrap95=r['bootstrap95']) for r in worst])


def compose(items, audits, windows, hashes):
    nmse, raw, ber = (items[name] for name in
                      ('nmse_summary.json', 'raw_start_summary.json', 'ber/summary.json'))
    policy, budget = items['policy.json'], items['budget_selection.json']
    lines = ['CONTROLLED START COMPARISON', '']
    primary = {}
    for key, competitor in (('adaptive_vs_global', 'the trained global schedule'),
                            ('adaptive_vs_classical', 'the calibrated classical structured start')):
        n = nmse['aggregates']['all']['contrasts'][key]
        b = ber['regions']['all']['contrasts'][key]
        primary[key] = dict(nmse=n, ber=b)
        lines.append(f"Against {competitor}, the whole-grid common-100 comparison "
                     f"{status(n['bonferroni4_interval_db'])} in NMSE and "
                     f"{status(b['bonferroni4_interval'])} in BER.")
    lines += [
        '',
        'Every experimental start is used at every condition, followed by the same 100 exact EM '
        'updates and final noise/reference-dependent filter. The global control learns 30 per-stage '
        'constants offline and has no observation-conditioned controller; its relative thresholds '
        'retain scale normalization. The classical control uses one configuration selected from '
        'the prespecified validation grid. Classical calibration minimizes raw-start held-out '
        'likelihood loss, rather than loss after the 100-update continuation. The common final '
        'filter was inherited from earlier adaptive-led selection, so this comparison is conditional '
        'on that fixed filter. The control does not equalize model parameter count. This comparison '
        'does not select or replace the deployed estimator.',
        '',
        'PRIMARY WHOLE-GRID EFFECTS',
        'Positive differences mean lower error for the adaptive start. BER differences are percentage points.',
    ]
    rows = []
    for key, description in (('adaptive_vs_global', 'Adaptive vs global'),
                             ('adaptive_vs_classical', 'Adaptive vs classical')):
        n, b = primary[key]['nmse'], primary[key]['ber']
        rows.append((description, f"{n['gain_db']:+.6g}", interval(n['bonferroni4_interval_db']),
                     f"{100*b['ber_reduction']:+.6g}", interval(b['bonferroni4_interval'], 100)))
    lines += [table(('Comparison', 'NMSE gain dB', 'Bonferroni4 interval dB',
                     'BER reduction pp', 'Bonferroni4 interval pp'), rows),
              'The two contrasts across NMSE and BER form four prespecified outcomes. '
              'All subgroup, raw-start, budget-arm and reference comparisons below are descriptive.', '']

    for scope in SCOPES:
        n, b = nmse['aggregates'][scope], ber['regions'][scope]
        lines += [SCOPE_LABELS[scope].upper(),
                  'The first five rows use the common 100-update continuation; remaining rows are separate references.']
        rows = [(LABELS[name], f"{n['methods'][name]['nmse_db']:.5f}",
                 f"{100*b['methods'][name]['ber']:.7g}") for name in MAIN_METHODS]
        rows.append((LABELS['perfect_csi'], '—', f"{100*b['methods']['perfect_csi']['ber']:.7g}"))
        lines += [table(('Method', 'NMSE dB (lower)', 'BER % (lower)'), rows), '']

    lines += ['RAW STARTS BEFORE CONTINUATION AND FINAL FILTER',
              'These are descriptive NMSE diagnostics; no raw-start BER experiment was run.']
    rows = [(LABELS[name], *[f"{raw['aggregates'][scope]['methods'][name]['nmse_db']:.5f}"
                            for scope in SCOPES])
            for name in ('adaptive_mean', 'global_mean', 'classical', 'spectral', 'em10')]
    lines += [table(('Start', 'All dB', 'Weak18 dB', 'Strong-high30 dB', 'Ordinary45 dB'), rows), '']

    lines += ['TRAINING-FIT VARIATION',
              'Groups average the three individual metrics, never the predicted channels. '
              'Three fits do not triple the number of independent channels.']
    rows = []
    for scope in ('all', 'weak_warm100'):
        for family in ('adaptive', 'global'):
            for seed in SEEDS:
                name = f'{family}_{seed}'
                n = nmse['aggregates'][scope]['methods'][name]['nmse_db']
                b = ber['regions'][scope]['methods'][name]
                rows.append((scope, family, seed, f'{n:.5f}', f"{100*b['ber']:.7g}",
                             b['bit_errors'], b['bits_per_fit']))
    lines += [table(('Scope', 'Start', 'Fit', 'NMSE dB', 'BER %', 'Bit errors', 'Bits'), rows), '']

    failures = {}
    lines += ['CONDITION-SPECIFIC LOSSES',
              'Counts cover all 153 conditions. Pointwise intervals below are descriptive and '
              'do not constitute additional familywise claims. Full lists are saved in completion.json.']
    rows = []
    for contrast in nmse['contrasts']:
        n = nmse['negative_cells'][contrast]
        b = ber_failures(ber, contrast)
        failures[contrast] = dict(nmse=n, ber=b)
        rows.append((contrast, len(n['negative_cells']), len(n['pointwise95_harm_cells']),
                     len(b['negative_cells']), len(b['pointwise95_harm_cells'])))
    lines.append(table(('Contrast (candidate vs reference)', 'NMSE worse', 'NMSE harm CI',
                        'BER worse', 'BER harm CI'), rows))
    for contrast in PRIMARY:
        n, b = failures[contrast]['nmse'], failures[contrast]['ber']
        lines.append(f"{contrast}: NMSE is worse in {len(n['negative_cells'])} cells, "
                     f"with {len(n['pointwise95_harm_cells'])} pointwise intervals supporting harm; "
                     f"worst {n['worst_cell']} ({n['worst_gain_db']:+.6g} dB gain). "
                     f"BER is worse in {len(b['negative_cells'])} cells, with "
                     f"{len(b['pointwise95_harm_cells'])} pointwise intervals supporting harm.")
        lines.append('Five lowest BER differences: ' + '; '.join(
            f"{r['cell']}: {100*r['ber_reduction']:+.6g} pp "
            f"{interval(r['bootstrap95'], 100)}" for r in b['worst_cells']))
    lines += ['', 'SEPARATE COMPUTATION-BUDGET COMPARISON',
              'Continuation counts were selected using timing alone, with a 10.0 ms calibration '
              'threshold for both whole-grid and weak-region p90. Accuracy did not select the counts. '
              'Unequal continuation counts assess computational allocation, not initialization alone.']
    budget_selection = {}
    rows = []
    for family in ('adaptive', 'global', 'classical', 'spectral', 'em10'):
        count = policy['budget_steps'][family]
        candidates = budget['scores'][family]
        eligible = [int(k) for k, value in candidates.items() if value['eligible']]
        selected_ok = bool(candidates[str(count)]['eligible'])
        budget_selection[family] = dict(continuation_updates=count, eligible_counts=eligible,
                                        calibration_eligible=selected_ok,
                                        fallback_no_eligible_count=not bool(eligible))
        assert policy['budget_fallback_to_100'][family] == (not bool(eligible))
        assert audits['selection']['budget_fallback_no_eligible'][family] == (not bool(eligible))
        rows.append((family, count, ', '.join(map(str, eligible)) or 'none',
                     'yes' if selected_ok else 'NO; fallback retained'))
    lines += [table(('Start family', 'Selected continuation', 'Calibration-eligible counts',
                     'Selected count eligible'), rows),
              'Budget-arm accuracy (descriptive; NMSE dB / BER %):']
    rows = []
    for name in BUDGET_METHODS:
        row = [LABELS[name]]
        for scope in SCOPES:
            n = nmse['aggregates'][scope]['methods'][name]['nmse_db']
            b = ber['regions'][scope]['methods'][name]['ber']
            row.append(f'{n:.5f} / {100*b:.7g}')
        rows.append(row)
    lines += [table(('Budget arm', 'All', 'Weak18', 'Strong-high30', 'Ordinary45'), rows), '']

    lines += ['COMPLETE PIPELINE RUNTIME',
              'The literal cap is 10.74 ms in BOTH final timing windows, applied to the '
              'whole-grid pooled batch-one p90. Weak-region p90 is reported separately. '
              'A family pools individual fit calls; it is not an ensemble runtime. '
              'All overruns remain visible and no algorithm is selected by this report.']
    runtime = {}
    rows, conditional_rows = [], []
    for name in (*MAIN_METHODS[:5], *BUDGET_METHODS, *MAIN_METHODS[5:]):
        a, b = windows['window1']['methods'][name], windows['window2']['methods'][name]
        passed = a['p90_ms'] <= CAP_MS and b['p90_ms'] <= CAP_MS
        assert passed == audits['timing']['target_met_in_both_windows_by_method'][name]
        conditional = {}
        for key, item in (('window1', a), ('window2', b)):
            bins = item['by_condition']
            overruns = [condition for condition, stats in bins.items() if stats['p90_ms'] > CAP_MS]
            worst = max(bins, key=lambda condition: bins[condition]['p90_ms'])
            conditional[key] = dict(overrun_count=len(overruns), overruns=overruns,
                                    worst_condition=worst, worst_p90_ms=bins[worst]['p90_ms'])
        runtime[name] = dict(window1=a, window2=b, cap_ms=CAP_MS, both_windows_pass=passed,
                             conditional_overruns=conditional)
        rows.append((LABELS[name], f"{a['p90_ms']:.4f}", f"{b['p90_ms']:.4f}",
                     f"{a['weak_warm100']['p90_ms']:.4f}", f"{b['weak_warm100']['p90_ms']:.4f}",
                     'PASS' if passed else 'FAIL'))
        conditional_rows.append((LABELS[name], *[
            f"{conditional[key]['overrun_count']}/51; "
            f"{conditional[key]['worst_p90_ms']:.4f} ms ({conditional[key]['worst_condition']})"
            for key in ('window1', 'window2')]))
    lines += [table(('Pipeline', 'W1 all p90 ms', 'W2 all p90 ms',
                     'W1 weak p90 ms', 'W2 weak p90 ms', '10.74 ms cap'), rows),
              'Condition-specific p90 overruns are descriptive; the declared acceptance rule uses the pooled whole grid.',
              table(('Pipeline', 'Window 1: overruns; worst bin', 'Window 2: overruns; worst bin'),
                    conditional_rows),
              'Individual-fit whole-grid p90 ranges (window 1; window 2):']
    for family in ('adaptive', 'global', 'budget_adaptive', 'budget_global', 'latest'):
        ranges = []
        for key in ('window1', 'window2'):
            values = [windows[key]['methods'][f'{family}_{seed}']['p90_ms'] for seed in SEEDS]
            ranges.append(f'{min(values):.4f}–{max(values):.4f} ms')
        lines.append(f'{family}: ' + '; '.join(ranges))

    zero_scopes = {}
    for scope in SCOPES:
        zero = [name for name, value in ber['regions'][scope]['methods'].items()
                if name in LABELS and value['ber'] == 0]
        zero_scopes[scope] = zero
    lines += [
        '', 'INTERPRETATION AND VERIFICATION',
        'The test contains 128 anchors for each of nine reference/path combinations: 1,152 physical '
        'anchors reused across 17 SNR values, and 384 index clusters for the coupled bootstrap. '
        'Each channel transmits all 64 joint Gray-QPSK vectors once, with independent data noise '
        'shared across estimators. Every detector observes magnitudes only and evaluates the same '
        '64-hypothesis exact Rician likelihood.',
        'NMSE is the equal-cell mean of dB ratios of summed channel errors to summed channel energy. '
        'BER averages the same channel samples. The 20,000 paired bootstrap draws share local '
        'indices across references and SNR within each path regime. Intervals are conditional on '
        'the fixed trained weights, calibrated controls and study grid.',
        'Zero observed errors, zero-versus-zero differences and collapsed empirical intervals '
        'do not establish zero population BER, statistical equivalence or population noninferiority. '
        'Bit counts are not independent channel trials.',
        f"Classical configuration: {json.dumps(policy['classical_config'], sort_keys=True)}.",
        'Required audits passed: selection, test predictions, raw/final summaries, both timing '
        'windows, and BER. The BER audit independently regenerates data and redecodes every method '
        'on the prespecified spot cases. This report does not edit the manuscript or make a '
        'publication/novelty claim.',
        '',
        'ARTIFACTS',
        'nmse_summary.json; raw_start_summary.json; ber/summary.json; '
        'timing_window1.json; timing_window2.json; audit/*.json; ber/audit.json.',
        'Figures: start_nmse.png, raw_start_nmse.png, start_gains.png, start_runtime.png, ber/ber.png.',
        f"Policy SHA256: {sha(OUT / 'policy.json')}",
    ]
    completion = dict(
        status='complete', all_required_audits_passed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        selection_sha256=sha(OUT / 'policy.json'), protocol_sha256=sha(OUT / 'protocol.json'),
        source_sha256=sha(__file__), artifact_hashes=hashes,
        primary_outcomes=primary, primary_multiplicity=4,
        scientific_decision='Comparison only; no algorithm selected or deployed by this report.',
        conditional_losses=failures, budget_selection=budget_selection,
        runtime=runtime, zero_observed_error_scopes=zero_scopes,
        counts=dict(cells=153, anchors_per_cell=128, physical_anchors=1152,
                    coupled_index_clusters=384, vectors_per_channel=64,
                    bits_per_cell_per_fit=49152, bits_full_grid_per_fit=7520256),
        audit_paths=[f'audit/{name}.json' for name in REQUIRED_AUDITS] + ['ber/audit.json'],
        manuscript_written_by_report=False,
        limitations='Conditional fixed-model/control comparison; zero BER does not prove '
                    'equivalence; budget arms have unequal continuation counts.',
    )
    return '\n'.join(lines) + '\n', completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-ready', action='store_true',
                        help='List missing required artifacts without generating a report.')
    args = parser.parse_args()
    if args.check_ready:
        missing = [name for name in required_paths() if not (OUT / name).exists()]
        print(json.dumps(dict(required_files_present=not missing, missing=missing,
                              reports_written=False)))
        return
    items, audits, windows, hashes = evidence()
    text, completion = compose(items, audits, windows, hashes)
    atomic_text(OUT / 'RESULTS.txt', text)
    completion['report_sha256'] = sha(OUT / 'RESULTS.txt')
    atomic_text(OUT / 'completion.json', json.dumps(completion, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='complete', report=str(OUT / 'RESULTS.txt'),
                          completion=str(OUT / 'completion.json'), required_audits_passed=True)))


if __name__ == '__main__':
    main()
