"""Reporting only: consume completed, audited hybrid-study artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/tcom_hybrid_20261020'
SEEDS = (9101, 9102, 9103)
CAP_MS = 10.74
PRIMARY = ('hybrid_vs_latest', 'hybrid_vs_compact')
SCOPES = ('all', 'ordinary', 'transition_all', 'high_all', 'weak_warm100', 'strong_high')
LABELS = dict(
    hybrid_mean='Hybrid', hybrid0_mean='Same hybrid, zero additional EM',
    latest_mean='Frozen latest estimator', adaptive40_mean='Adaptive start + EM40',
    global40_mean='Global schedule + EM40', classical40='Classical start + EM40',
    em100='Direct EM100', compact_sparse='Compact sparse', optshrink='OptShrink',
    perfect_csi='Perfect CSI reference')
METHODS = tuple(name for name in LABELS if name != 'perfect_csi')
AUDIT_NAMES = ('selection', 'test', 'summary', 'timing')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        stream.write(value)
        temporary = stream.name
    os.replace(temporary, path)


def required():
    return ['protocol.json', 'policy.json', 'nmse_summary.json', 'raw_start_summary.json',
            'ber/summary.json', 'timing_window1.json', 'timing_window2.json', 'ber/audit.json',
            'runtime_calibration.json', 'crossover_selection.json',
            'tail_proposals.json', 'tail_confirmation.json',
            *(f'audit/{name}.json' for name in AUDIT_NAMES)]


def evidence():
    missing = [name for name in required() if not (OUT / name).exists()]
    if missing:
        raise RuntimeError('Final reporting awaits: ' + ', '.join(missing))
    data = {name: load(OUT / name) for name in required()}
    policy, protocol = data['policy.json'], data['protocol.json']
    selected, protocol_hash = sha(OUT / 'policy.json'), sha(OUT / 'protocol.json')
    assert policy['protocol_sha256'] == protocol_hash
    for field in ('source_hashes', 'checkpoint_hashes'):
        for name, digest in protocol[field].items():
            assert sha(REPO / name) == digest, name
    assert set(policy['selection_evidence']) == {
        'runtime_calibration.json', 'crossover_selection.json',
        'tail_proposals.json', 'tail_confirmation.json'}
    for name, digest in policy['selection_evidence'].items():
        assert sha(OUT / name) == digest, name
    for name, digest in policy['ber_source_hashes'].items():
        assert sha(REPO / name) == digest, name
    assert policy['ber_protocol_sha256'] == sha(OUT / 'ber/protocol.json')
    nmse, raw, ber = (data[n] for n in
                      ('nmse_summary.json', 'raw_start_summary.json', 'ber/summary.json'))
    for result in (nmse, raw, ber):
        assert result['selection_sha256'] == selected
    assert nmse['protocol_sha256'] == raw['protocol_sha256'] == protocol_hash
    assert ber['protocol_sha256'] == policy['ber_protocol_sha256']
    assert nmse['raw_start_summary_sha256'] == sha(OUT / 'raw_start_summary.json')
    assert nmse['raw_start_bootstrap_sha256'] == sha(OUT / 'raw_start_bootstrap.npz')
    assert not nmse['raw_start_diagnostic'] and raw['raw_start_diagnostic']
    assert len(nmse['cells']) == len(raw['cells']) == 153
    assert nmse['unique_test_anchors'] == 1152 and nmse['bootstrap_index_clusters'] == 384
    assert ber['regions']['all']['cells'] == 153
    audits = {name: data[f'audit/{name}.json'] for name in AUDIT_NAMES}
    audits['ber'] = data['ber/audit.json']
    for name, item in audits.items():
        assert item['passed'] is True and item['selection_sha256'] == selected, name
    assert audits['test']['cells'] == audits['ber']['cells'] == 153
    assert audits['summary']['summary_sha256'] == sha(OUT / 'nmse_summary.json')
    assert audits['summary']['raw_start_summary_sha256'] == sha(OUT / 'raw_start_summary.json')
    assert audits['summary']['final_timing_bindings_verified']
    assert audits['ber']['completed_unix'] >= (OUT / 'ber/summary.json').stat().st_mtime
    assert len(audits['ber']['cell_hashes']) == 153
    for name, digest in audits['ber']['cell_hashes'].items():
        assert sha(OUT / 'ber/cells' / (name + '.npz')) == digest
    for name, digest in audits['test']['test_prediction_hashes'].items():
        assert sha(OUT / 'test/cells' / name) == digest
    windows = {f'window{i}': data[f'timing_window{i}.json'] for i in (1, 2)}
    for key, item in windows.items():
        filename = f'timing_{key}.json'
        assert item['selection_sha256'] == selected and item['protocol_sha256'] == protocol_hash
        assert nmse['timing_hashes'][filename] == audits['timing']['timing_hashes'][filename] == sha(OUT / filename)
        assert nmse['timing_windows'][key]['methods'] == item['methods']
        assert nmse['timing_windows'][key]['fixed_cap_ms'] == CAP_MS
    assert nmse['timing'] == windows['window1']['methods']
    return data, audits, windows, {name: sha(OUT / name) for name in required()}


def table(header, rows):
    strings = [[str(v) for v in header], *[[str(v) for v in row] for row in rows]]
    widths = [max(len(row[i]) for row in strings) for i in range(len(header))]
    rendered = [' | '.join(v.ljust(widths[i]) for i, v in enumerate(strings[0])),
                '-+-'.join('-' * w for w in widths)]
    rendered.extend(' | '.join(v.ljust(widths[i]) for i, v in enumerate(row))
                    for row in strings[1:])
    return '\n'.join(rendered)


def interval(values, scale=1):
    return '[' + ', '.join(f'{scale * v:.6g}' for v in values) + ']'


def interpretation(bounds):
    if bounds[0] > 0:
        return 'supports lower error for the hybrid'
    if bounds[1] < 0:
        return 'supports higher error for the hybrid'
    if bounds == [0., 0.]:
        return 'records an empirical tie, without proving population equivalence'
    return 'does not resolve a difference at the adjusted confidence level'


def failure_rows(ber, contrast):
    cells = [r for r in ber['cell_contrasts'] if r['contrast'] == contrast]
    assert len(cells) == 153
    def name(row):
        return f"{row['regime']}_rsr{row['rsr_db']:g}_snr{row['snr_db']:g}"
    return dict(
        negative_cells=[name(r) for r in cells if r['ber_reduction'] < -1e-15],
        pointwise95_harm_cells=[name(r) for r in cells if r['bootstrap95'][1] < -1e-15],
        worst_cells=[dict(cell=name(r), ber_reduction=r['ber_reduction'],
                          bootstrap95=r['bootstrap95'])
                     for r in sorted(cells, key=lambda r: r['ber_reduction'])[:5]])


def policy_description(data):
    policy = data['policy.json']
    proposals = data['tail_proposals.json']['proposals']
    confirmation = data['tail_confirmation.json']['confirmed']
    crossover = data['crossover_selection.json']['crossover_db']
    assert crossover == policy['crossover_db']
    tails = policy['tails']
    references = sorted(tails, key=float)
    snrs = sorted(tails[references[0]], key=float)
    assert references == ['5', '12', '20'] and len(snrs) == 17
    rows, selected, rejected = [], {}, []
    for snr in snrs:
        cells = []
        for ref in references:
            choice = tails[ref][snr]
            family, steps = choice['family'], choice['steps']
            assert family in ('adaptive', 'global') and steps in (0, 10, 20, 40)
            key = f'{family}:{steps}'
            selected[key] = selected.get(key, 0) + 1
            if float(snr) <= 20:
                assert choice == {'family': 'adaptive', 'steps': 0}
                cells.append('adaptive / 0 (raw)')
                continue
            proposed = proposals[ref][snr]
            confirmed = confirmation[ref][snr]
            assert family == proposed['family'] == confirmed['family']
            assert steps == confirmed['steps']
            assert confirmed['proposed_steps'] == proposed['proposed_steps']
            expected = proposed['proposed_steps'] if confirmed['accepted'] else 0
            assert steps == expected
            suffix = ''
            if proposed['proposed_steps'] > 0 and not confirmed['accepted']:
                suffix = ' (C rejected)'
                rejected.append(dict(rsr_db=float(ref), snr_db=float(snr),
                                     proposed_steps=proposed['proposed_steps'], final_steps=0))
            cells.append(f'{family} / {steps}{suffix}')
        rows.append((snr, *cells))
    lines = [
        'FROZEN START AND EM POLICY',
        f'The shared transition crossover is {crossover:g} dB. Ordinary SNR <=20 dB '
        'returns the adaptive raw output exactly; SNR >=40 dB uses the global schedule. '
        'Between them, the adaptive start is used through the crossover and the global '
        'start above it. Outside the ordinary branch, the inherited final filter is applied.',
        'The table gives selected start / additional EM updates. C rejected means the '
        'independent confirmation bank rejected the proposed nonzero tail, so the evaluated '
        'policy retains zero extra EM; no alternative tail was tested on that bank.',
        table(('SNR dB', 'RSR 5', 'RSR 12', 'RSR 20'), rows),
        '',
        'Selection uses three independent development banks: A chooses the crossover '
        'from complete zero-tail outputs; B proposes a runtime-eligible tail; C confirms '
        'that single proposal against zero in every path regime. The one-sided Student-t '
        'bounds with nominal multiplicity corrections are approximate screening rules, '
        'not finite-sample guarantees of likelihood, NMSE or BER safety.',
        'Calibration eligibility requires both whole-grid and weak-region pooled p90 <=10.0 ms. '
        'Zero remains an explicit fallback even if its calibration misses that screening cap. '
        'Final acceptance separately checks the literal 10.74 ms cap in both runtime windows.',
    ]
    runtime_rows = []
    for family, choices in data['runtime_calibration.json']['scores'].items():
        for steps in sorted(choices, key=int):
            item = choices[steps]
            runtime_rows.append((family, steps, f"{item['p90_ms']:.4f}",
                                 f"{item['weak_p90_ms']:.4f}", item['eligible'],
                                 selected.get(f'{family}:{steps}', 0)))
    lines += [table(('Start', 'Extra EM', 'Calibration p90 ms', 'Weak p90 ms',
                     'Eligible', 'Selected bins / 51'), runtime_rows), '']
    return lines, dict(crossover_db=crossover, tails=tails,
                       selected_reference_snr_bin_counts=selected,
                       confirmation_rejections_to_zero=rejected)


def compose(data, audits, windows, hashes):
    nmse, raw, ber = (data[n] for n in
                      ('nmse_summary.json', 'raw_start_summary.json', 'ber/summary.json'))
    lines = ['HYBRID ESTIMATOR: FRESH PAIRED COMPARISON', '']
    primary = {}
    for key, label in (('hybrid_vs_latest', 'the frozen latest estimator'),
                       ('hybrid_vs_compact', 'compact sparse')):
        n = nmse['aggregates']['all']['contrasts'][key]
        b = ber['regions']['all']['contrasts'][key]
        primary[key] = dict(nmse=n, ber=b)
        lines.append(f"Against {label}, the whole-grid comparison "
                     f"{interpretation(n['bonferroni4_interval_db'])} in NMSE and "
                     f"{interpretation(b['bonferroni4_interval'])} in BER.")
    if all(choice['steps'] == 0 for by_snr in data['policy.json']['tails'].values()
           for choice in by_snr.values()):
        lines.append('The confirmed policy selected zero additional EM everywhere. '
                     'The hybrid and its zero-tail control are therefore the same estimator '
                     'on this grid; this is not evidence that additional EM cannot help.')
    if any(primary[key]['nmse']['bonferroni4_interval_db'][1] < 0 for key in PRIMARY):
        lines.append('The confirmed hybrid does not meet the intended accuracy objective '
                     'and these results do not support replacing the previous estimator.')
    lines += [
        '',
        'The hybrid start and additional EM count are frozen from observable development '
        'criteria and runtime eligibility before this fresh test. The no-EM arm uses exactly '
        'the same chosen start and final filter with zero additional EM. It isolates the '
        'continuation; it is not the raw pre-filter starting estimate.',
        'Both starts already execute ten EM/structure stages. Zero tail means no ADDITIONAL '
        'EM updates. The global schedule contains 30 globally trained parameters; the adaptive '
        'controller contains 188 parameters and conditions its decisions on observations. '
        'The fixed schedule still uses normalization and known noise variance.',
        'Primary claims concern hybrid versus latest and compact over all 153 conditions. '
        'The two contrasts across NMSE and BER form four outcomes with Bonferroni adjustment. '
        'Other methods, subgroups, raw starts and no-EM comparisons are descriptive.',
        'This report compares the fixed receivers; it does not automatically select, deploy '
        'or recommend replacing an estimator.',
        '',
        'PRIMARY EFFECTS',
        'Positive differences mean lower error for hybrid. BER differences are percentage points.',
    ]
    rows = []
    for key in PRIMARY:
        n, b = primary[key]['nmse'], primary[key]['ber']
        rows.append((key, f"{n['gain_db']:+.6g}", interval(n['bonferroni4_interval_db']),
                     f"{100*b['ber_reduction']:+.6g}", interval(b['bonferroni4_interval'], 100)))
    lines += [table(('Contrast', 'NMSE gain dB', 'Adjusted interval dB',
                     'BER reduction pp', 'Adjusted interval pp'), rows), '']
    policy_lines, policy_detail = policy_description(data)
    lines += policy_lines
    for scope in SCOPES:
        n, b = nmse['aggregates'][scope], ber['regions'][scope]
        rows = []
        for name in METHODS:
            item = n['methods'][name]
            pooled = item['pooled_ratio_of_sums_db']
            rows.append((LABELS[name], f"{item['nmse_db']:.5f}", f'{pooled:.5f}',
                         f"{item['pooled_ratio_of_sums_linear']:.7g}",
                         f"{100*b['methods'][name]['ber']:.7g}"))
        rows.append((LABELS['perfect_csi'], '—', '—', '—',
                     f"{100*b['methods']['perfect_csi']['ber']:.7g}"))
        lines += [f"{scope.upper()} ({n['cells']} conditions)",
                  table(('Method', 'Equal-cell NMSE dB', 'Pooled ratio dB',
                         'Pooled linear NMSE', 'BER %'), rows), '']
    lines += ['RAW STARTS BEFORE FINAL FILTERING',
              'Raw adaptive/global starts have NMSE diagnostics only. They are different '
              'from hybrid0, which includes the same chosen final filter as hybrid.']
    rows = []
    for family in ('adaptive_mean', 'global_mean'):
        rows.append((family, *[f"{raw['aggregates'][s]['methods'][family]['nmse_db']:.5f}"
                                for s in SCOPES]))
    lines += [table(('Raw start', *SCOPES), rows), '',
              'FIT SPREAD AND ACTUAL BIT COUNTS',
              'Three fits average individual metrics, never channels, and do not triple the channel sample size.']
    rows = []
    for scope in ('all', 'ordinary', 'high_all', 'weak_warm100'):
        for family in ('hybrid', 'hybrid0', 'latest'):
            for seed in SEEDS:
                name = f'{family}_{seed}'
                n, b = nmse['aggregates'][scope]['methods'][name], ber['regions'][scope]['methods'][name]
                rows.append((scope, name, f"{n['nmse_db']:.5f}", f"{100*b['ber']:.7g}",
                             b['bit_errors'], b['bits_per_fit']))
    lines += [table(('Scope', 'Fit', 'NMSE dB', 'BER %', 'Errors', 'Bits'), rows), '']
    losses, rows = {}, []
    for key in nmse['contrasts']:
        n, b = nmse['negative_cells'][key], failure_rows(ber, key)
        losses[key] = dict(nmse=n, ber=b)
        rows.append((key, len(n['negative_cells']), len(n['pointwise95_harm_cells']),
                     len(b['negative_cells']), len(b['pointwise95_harm_cells']),
                     n['worst_cell'], f"{n['worst_gain_db']:+.6g}"))
    lines += ['CONDITION-SPECIFIC LOSSES',
              'Pointwise harm intervals are descriptive. Full cell lists are saved in completion.json.',
              table(('Contrast', 'NMSE worse', 'NMSE harm CI', 'BER worse', 'BER harm CI',
                     'Worst NMSE cell', 'Gain dB'), rows), '']
    lines += ['COMPLETE PIPELINE RUNTIME',
              'Acceptance uses the literal 10.74 ms whole-grid pooled batch-one p90 in BOTH '
              'final windows. Weak-region and conditional p90s are descriptive. These are '
              'host-specific observed quantiles, not hard per-call deadlines. A family pools '
              'individual fit calls rather than adding runtimes into an ensemble.']
    runtimes, rows, conditional_rows = {}, [], []
    for name in METHODS:
        a, b = windows['window1']['methods'][name], windows['window2']['methods'][name]
        passed = a['p90_ms'] <= CAP_MS and b['p90_ms'] <= CAP_MS
        assert passed == audits['timing']['target_met_in_both_windows_by_method'][name]
        conditional = {}
        for key, item in (('window1', a), ('window2', b)):
            bins = item['by_condition']
            over = [k for k, v in bins.items() if v['p90_ms'] > CAP_MS]
            worst = max(bins, key=lambda k: bins[k]['p90_ms'])
            conditional[key] = dict(overrun_count=len(over), overruns=over,
                                    worst_condition=worst, worst_p90_ms=bins[worst]['p90_ms'])
        runtimes[name] = dict(window1=a, window2=b, both_windows_pass=passed,
                              cap_ms=CAP_MS, conditional_overruns=conditional)
        rows.append((LABELS[name], f"{a['p90_ms']:.4f}", f"{b['p90_ms']:.4f}",
                     f"{a['weak_warm100']['p90_ms']:.4f}", f"{b['weak_warm100']['p90_ms']:.4f}",
                     'PASS' if passed else 'FAIL'))
        conditional_rows.append((LABELS[name], *[
            f"{conditional[key]['overrun_count']}/51; "
            f"{conditional[key]['worst_p90_ms']:.4f} ms ({conditional[key]['worst_condition']})"
            for key in ('window1', 'window2')]))
    lines += [table(('Pipeline', 'W1 p90 ms', 'W2 p90 ms', 'W1 weak p90', 'W2 weak p90', 'Cap'), rows),
              table(('Pipeline', 'W1 condition overruns; worst', 'W2 condition overruns; worst'),
                    conditional_rows), '']
    lines += [
        'INTERPRETATION AND VERIFICATION',
        'The primary NMSE is an equal-cell mean of dB error-to-energy ratios. The pooled linear '
        'ratio is computed separately for each fit and then averaged arithmetically across '
        'fits; its dB value is the log of that mean. It weights errors and channel energy '
        'differently from the primary mean-cell dB measure. The grid '
        'has 108 of 153 conditions above 20 dB although the inherited training ends at 20 dB; '
        'neither weighting is a deployment-weighted guarantee.',
        'There are 128 channels per reference/path setting, 1,152 physical anchors reused across '
        '17 SNR values, and 384 local-index clusters for the 20,000 paired bootstrap draws. '
        'Indices are shared across reference, SNR, estimator and training fit within each path '
        'regime. Uncertainty is conditional on the fixed weights, selected policy and grid.',
        'All methods use the same independent magnitude-domain data: each channel transmits all '
        '64 joint unit-power Gray-QPSK vectors once. The detector enumerates those 64 hypotheses '
        'using the exact Rician likelihood. N=32, K=3 and P=30; reference and noise variance are known.',
        'Zero observed errors and collapsed empirical intervals do not prove zero population BER, '
        'equivalence or population noninferiority. Bit counts are not independent channel trials.',
        'The additional EM count is a finite continuation, not a guarantee of a converged optimum. '
        'The learned global schedule lacks an observation-conditioned controller but retains '
        'scale normalization. These comparisons do not cover every classical estimator.',
        'Routing still has boundaries: the start changes above 30 dB; the nearest-bin strong-reference '
        'filter changes at 31.25 dB. In the separate exploratory B-proposal arm, a ten-update '
        'tail also starts across that strong-reference boundary, and the weak-reference tail '
        'changes from zero to 40 updates across 38.75 dB. These B tails are absent from the '
        'confirmed zero-tail hybrid. The earlier strong-reference filter-off jump around 40 dB '
        'remains repaired: c=4, beta=1 filtering continues from 37.5 dB upward. This gridded '
        'test does not prove continuity between sampled conditions or robustness to variance mismatch.',
        'Selection, prediction, raw/final summary, timing and BER audits all passed. The report '
        'does not edit the paper or make a publication or novelty claim.',
        '',
        'Artifacts: nmse_summary.json, raw_start_summary.json, ber/summary.json, '
        'timing_window1.json, timing_window2.json, audit/*.json and ber/audit.json.',
        'Figures: hybrid_nmse.png, hybrid_gains.png, hybrid_runtime.png, raw_start_nmse.png, ber/ber.png.',
        f"Policy SHA256: {sha(OUT / 'policy.json')}",
    ]
    completion = dict(
        status='complete', all_required_audits_passed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        selection_sha256=sha(OUT / 'policy.json'), protocol_sha256=sha(OUT / 'protocol.json'),
        source_sha256=sha(__file__), artifact_hashes=hashes, primary_outcomes=primary,
        primary_multiplicity=4, frozen_hybrid_policy=policy_detail,
        conditional_losses=losses, runtime=runtimes,
        counts=dict(cells=153, anchors_per_cell=128, physical_anchors=1152,
                    coupled_index_clusters=384, vectors_per_channel=64,
                    bits_per_cell_per_fit=49152, bits_full_grid_per_fit=7520256),
        scientific_decision='Comparison only; no algorithm selected or deployed by the report.',
        manuscript_written_by_report=False,
        limitations='Conditional fixed-policy inference; no equivalence claim from zero errors; '
                    'host-specific pooled p90 is not a hard deadline.',
        audit_paths=[f'audit/{n}.json' for n in AUDIT_NAMES] + ['ber/audit.json'])
    return '\n'.join(lines) + '\n', completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-ready', action='store_true')
    args = parser.parse_args()
    if args.check_ready:
        missing = [name for name in required() if not (OUT / name).exists()]
        print(json.dumps(dict(required_files_present=not missing, missing=missing, reports_written=False)))
        return
    data, audits, windows, hashes = evidence()
    text, completion = compose(data, audits, windows, hashes)
    write(OUT / 'RESULTS.txt', text)
    completion['report_sha256'] = sha(OUT / 'RESULTS.txt')
    write(OUT / 'completion.json', json.dumps(completion, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='complete', report=str(OUT / 'RESULTS.txt'),
                          required_audits_passed=True)))


if __name__ == '__main__':
    main()
