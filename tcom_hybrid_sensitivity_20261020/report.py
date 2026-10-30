"""Reporting only; never changes the parent policy or primary conclusions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/tcom_hybrid_sensitivity_20261020'
PARENT = REPO / 'results/tcom_hybrid_20261020'
SCOPES = ('all', 'ordinary', 'transition_all', 'high_all', 'weak_warm100', 'strong_high')
LABELS = {
    'proposal_mean': 'Unconfirmed B proposals',
    'hybrid_mean': 'Confirmed hybrid (zero EM)',
    'latest_mean': 'Frozen latest',
    'compact_sparse': 'Compact sparse',
    'adaptive40_mean': 'Adaptive + EM40',
    'global40_mean': 'Global schedule + EM40',
    'classical40': 'Classical + EM40',
    'em100': 'Direct EM100',
    'optshrink': 'OptShrink',
}
AUDITS = ('selection', 'test', 'summary', 'timing')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        stream.write(text)
        temporary = stream.name
    os.replace(temporary, path)


def table(header, rows):
    text = [[str(v) for v in header], *[[str(v) for v in row] for row in rows]]
    widths = [max(len(row[i]) for row in text) for i in range(len(header))]
    lines = [' | '.join(v.ljust(widths[i]) for i, v in enumerate(text[0])),
             '-+-'.join('-' * w for w in widths)]
    lines.extend(' | '.join(v.ljust(widths[i]) for i, v in enumerate(row)) for row in text[1:])
    return '\n'.join(lines)


def interval(values, scale=1):
    return '[' + ', '.join(f'{v * scale:.6g}' for v in values) + ']'


def required():
    return ['protocol.json', 'nmse_summary.json', 'nmse_bootstrap.npz', 'ber/summary.json',
            'ber/bootstrap.npz', 'timing_window1.json', 'timing_window2.json', 'ber/audit.json',
            *(f'audit/{name}.json' for name in AUDITS)]


def evidence():
    missing = [str(OUT / p) for p in required() if not (OUT / p).exists()]
    if not (PARENT / 'completion.json').exists():
        missing.append(str(PARENT / 'completion.json'))
    if missing:
        raise RuntimeError('Final sensitivity report awaits: ' + ', '.join(missing))
    data = {name: load(OUT / name) for name in required() if name.endswith('.json')}
    protocol, digest = data['protocol.json'], sha(OUT / 'protocol.json')
    parent_digest = sha(PARENT / 'policy.json')
    assert protocol['main_policy_sha256'] == parent_digest
    for field in ('source_hashes', 'checkpoint_hashes'):
        for name, value in protocol[field].items():
            assert sha(REPO / name) == value, name
    for name, value in protocol['parent_input_hashes'].items():
        assert sha(PARENT / name) == value, name
    complete = load(PARENT / 'completion.json')
    assert complete['status'] == 'complete' and complete['all_required_audits_passed']
    assert complete['selection_sha256'] == parent_digest
    assert complete['report_sha256'] == sha(PARENT / 'RESULTS.txt')
    for name, value in complete['artifact_hashes'].items():
        assert sha(PARENT / name) == value
    audits = {name: data[f'audit/{name}.json'] for name in AUDITS}
    audits['ber'] = data['ber/audit.json']
    for name, item in audits.items():
        assert item['passed'] and item['selection_sha256'] == digest, name
        assert item['parent_selection_sha256'] == parent_digest
    nmse, ber = data['nmse_summary.json'], data['ber/summary.json']
    for item in (nmse, ber):
        assert item['selection_sha256'] == item['protocol_sha256'] == digest
        assert item['parent_selection_sha256'] == parent_digest
        assert item['exploratory'] and item['primary'] is False and item['adjusted_intervals'] is False
    for cell, value in audits['test']['test_prediction_hashes'].items():
        assert sha(OUT / 'test/cells' / cell) == value
    for cell, value in audits['ber']['cell_hashes'].items():
        assert sha(OUT / 'ber/cells' / cell) == value
    windows = {f'window{i}': data[f'timing_window{i}.json'] for i in (1, 2)}
    for key, item in windows.items():
        assert item['selection_sha256'] == digest and item['protocol_sha256'] == digest
        filename = f'timing_{key}.json'
        assert audits['timing']['timing_hashes'][filename] == sha(OUT / filename)
    assert audits['summary']['summary_sha256'] == sha(OUT / 'nmse_summary.json')
    assert audits['summary']['ber_summary_sha256'] == sha(OUT / 'ber/summary.json')
    assert audits['summary']['nmse_bootstrap_sha256'] == sha(OUT / 'nmse_bootstrap.npz')
    assert audits['summary']['ber_bootstrap_sha256'] == sha(OUT / 'ber/bootstrap.npz')
    hashes = {name: sha(OUT / name) for name in required()}
    hashes['parent_completion'] = sha(PARENT / 'completion.json')
    return data, audits, windows, hashes


def compose(data, audits, windows, hashes):
    protocol, nmse, ber = (data[name] for name in ('protocol.json', 'nmse_summary.json', 'ber/summary.json'))
    policy = protocol['proposal_policy']
    failed = protocol['failed_confirmation']
    assert len(failed) == 21 and all(not r['C_accepted'] and r['C_selected_steps'] == 0 for r in failed)
    confirmation_guards = [guard for row in failed for guard in row['C_guards'].values()]
    positive_means = sum(guard['mean_improvement'] > 0 for guard in confirmation_guards)
    passing_bounds = sum(guard['passed'] for guard in confirmation_guards)
    against_hybrid = nmse['aggregates']['all']['contrasts']['proposal_vs_hybrid']
    against_latest = nmse['aggregates']['all']['contrasts']['proposal_vs_latest']
    ber_against_hybrid = ber['regions']['all']['contrasts']['proposal_vs_hybrid']
    lines = [
        'EXPLORATORY B-PROPOSAL SENSITIVITY — NOT THE CONFIRMED HYBRID',
        '',
        'The parent primary remains the C-confirmed hybrid with zero additional EM in every '
        'condition. This attached arm executes the 21 nonzero tails proposed by bank B, all '
        'of which failed the prespecified independent bank-C confirmation. There was no '
        'relaxed threshold, second choice, new search or change to the parent estimator.',
        'These fresh-test comparisons measure the unconfirmed proposals. They do not '
        'retroactively confirm them, establish selection safety, or replace the parent '
        'primary results. All intervals below are descriptive pointwise 95% intervals.',
        f"The proposals change whole-grid mean-cell NMSE by {against_hybrid['gain_db']:+.4f} dB "
        f"relative to the confirmed hybrid and {against_latest['gain_db']:+.4f} dB relative to "
        f"the previous estimator (positive favors proposals). Their BER reduction versus the "
        f"confirmed hybrid is {100 * ber_against_hybrid['ber_reduction']:+.6f} percentage points. "
        'This channel-error/bit-error tradeoff does not overturn the failed C confirmation.',
        'Both starts already execute ten EM/structure stages; zero tail means no ADDITIONAL '
        'EM updates. The global schedule has 30 globally trained parameters, while the '
        '188-parameter adaptive controller conditions decisions on observations. The fixed '
        'schedule retains normalization and dependence on known noise variance.',
        f'The shared A crossover remains {policy["crossover_db"]:g} dB. The exact parent '
        '128 channel anchors, 30 fitting pilots and saved 64-vector Gray-QPSK data are reused '
        'for all 153 conditions. This is a paired extension, not a second independent sample.',
        '',
        'WHOLE-GRID DESCRIPTIVE EFFECTS',
        'Positive values favor the unconfirmed proposal. BER differences are percentage points.',
    ]
    rows = []
    for key in nmse['contrasts']:
        n = nmse['aggregates']['all']['contrasts'][key]
        b = ber['regions']['all']['contrasts'][key]
        rows.append((key, f"{n['gain_db']:+.6g}", interval(n['pointwise_ci95_db']),
                     f"{100 * b['ber_reduction']:+.6g}", interval(b['bootstrap95'], 100)))
    lines += [table(('Contrast', 'NMSE gain dB', 'Pointwise interval dB',
                     'BER reduction pp', 'Pointwise interval pp'), rows), '']
    for scope in SCOPES:
        n, b = nmse['aggregates'][scope], ber['regions'][scope]
        rows = []
        for name, label in LABELS.items():
            value = n['methods'][name]
            rows.append((label, f"{value['nmse_db']:.5f}",
                         f"{value['pooled_ratio_of_sums_linear']:.7g}",
                         f"{100 * b['methods'][name]['ber']:.7g}"))
        lines += [f"{scope.upper()} ({n['cells']} conditions)",
                  table(('Method', 'Mean-cell NMSE dB', 'Pooled linear NMSE', 'BER %'), rows), '']
    rows = []
    for row in failed:
        rows.append((row['reference_key'], row['snr_key'], row['family'],
                     row['proposed_steps'], row['C_selected_steps']))
    lines += ['THE 21 UNCONFIRMED TAILS', table(
        ('RSR dB', 'SNR dB', 'Start', 'Exploratory EM', 'Confirmed EM'), rows), '']
    losses, rows = {}, []
    for key in nmse['contrasts']:
        n, b = nmse['negative_cells'][key], ber['negative_cells'][key]
        losses[key] = dict(nmse=n, ber=b)
        rows.append((key, len(n['negative_cells']), len(n['pointwise95_harm_cells']),
                     len(b['negative_cells']), len(b['pointwise95_harm_cells']),
                     n['worst_cell'], f"{n['worst_gain_db']:+.6g}"))
    lines += ['ALL CONDITION-SPECIFIC LOSSES', 'Complete lists are retained in completion.json.',
              table(('Contrast', 'NMSE worse', 'NMSE harm CI', 'BER worse', 'BER harm CI',
                     'Worst NMSE cell', 'Gain dB'), rows), '']
    rows = []
    for scope in SCOPES:
        for seed in (9101, 9102, 9103):
            name = f'proposal_{seed}'
            n = nmse['aggregates'][scope]['methods'][name]
            b = ber['regions'][scope]['methods'][name]
            rows.append((scope, seed, f"{n['nmse_db']:.5f}", f"{100 * b['ber']:.7g}",
                         b['bit_errors'], b['bits_per_fit']))
    lines += ['PROPOSAL FIT SPREAD AND ACTUAL COUNTS',
              table(('Scope', 'Fit', 'NMSE dB', 'BER %', 'Errors', 'Bits'), rows), '']
    runtime, rows = {}, []
    for name in ('proposal_mean', 'hybrid_mean', 'latest_mean', 'compact_sparse'):
        a, b = (windows[key]['methods'][name] for key in ('window1', 'window2'))
        passed = a['p90_ms'] <= 10.74 and b['p90_ms'] <= 10.74
        assert passed == audits['timing']['target_met_in_both_windows_by_method'][name]
        overruns = {key: [c for c, value in windows[key]['methods'][name]['by_condition'].items()
                          if value['p90_ms'] > 10.74] for key in windows}
        runtime[name] = dict(window1=a, window2=b, both_windows_pass=passed,
                             conditional_overruns=overruns)
        rows.append((LABELS[name], f"{a['p90_ms']:.4f}", f"{b['p90_ms']:.4f}",
                     f"{a['weak_warm100']['p90_ms']:.4f}", f"{b['weak_warm100']['p90_ms']:.4f}",
                     'PASS' if passed else 'FAIL', f"{len(overruns['window1'])}/{len(overruns['window2'])}"))
    lines += ['ATTACHED SAME-WINDOW RUNTIME COMPARISON',
              'Both isolated windows time complete proposal, confirmed hybrid, latest and '
              'compact pipelines. Family samples pool individual fits, never an ensemble sum.',
              table(('Pipeline', 'W1 p90 ms', 'W2 p90 ms', 'W1 weak p90', 'W2 weak p90',
                     '10.74 cap', 'W1/W2 bin overruns'), rows), '',
              'INTERPRETATION',
              f'{positive_means}/{len(confirmation_guards)} path-regime mean benefits for the '
              f'21 B proposals were positive on bank C, but only {passing_bounds} of their '
              'approximate lower bounds passed; every proposal failed at '
              'least one path regime. Rejection does not prove no EM benefit. Conversely, '
              'favorable exploratory results do not repair the failed confirmation.',
              'The paired bootstrap uses 20,000 draws of 128 local indices within each of '
              'three regimes, shared across references, SNRs, fits and methods: 384 coupled '
              'index clusters, with 1,152 physical channel anchors. Fit averages are metric '
              'averages. Pooled linear NMSE averages fit-specific pooled ratios arithmetically.',
              'The grid has 108/153 conditions above 20 dB; equal-cell averages are not '
              'deployment-weighted claims. N=32, K=3, P=30, known reference and variance, '
              'and the exact magnitude-only 64-hypothesis detector define this comparison.',
              'Routing boundaries remain: the start changes above 30 dB; the strong-reference '
              'filter and exploratory ten-update tail change across the 31.25 dB nearest-bin '
              'boundary; the weak-reference exploratory tail changes from zero to 40 updates '
              'across 38.75 dB. The prior strong-reference filter-off jump near 40 dB remains '
              'repaired, with c=4, beta=1 filtering retained from 37.5 dB upward. The gridded '
              'test is not a continuity proof or a variance-mismatch robustness claim.',
              'Zero observed BER and collapsed intervals do not prove equivalence, zero '
              'population BER or noninferiority. The host-specific pooled p90 cap is not '
              'a hard per-call deadline. All adverse conditions and runtime overruns are retained.',
              'Parent completion and all five attached audits passed. No paper edits, deployment '
              'decision, novelty claim or publication conclusion are made here.',
              '', 'Figures: exploratory_nmse.png and exploratory_ber.png.']
    completion = dict(status='complete', exploratory=True, primary=False,
        parent_primary_unchanged=True, confirmed_parent_extra_em_everywhere=0,
        failed_confirmation_count=21, all_required_audits_passed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        selection_sha256=sha(OUT / 'protocol.json'),
        parent_selection_sha256=protocol['main_policy_sha256'],
        parent_completion_sha256=sha(PARENT / 'completion.json'),
        artifact_hashes=hashes, source_sha256=sha(__file__), conditional_losses=losses,
        runtime=runtime, primary_claims_added=False, manuscript_written=False)
    return '\n'.join(lines) + '\n', completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-ready', action='store_true')
    args = parser.parse_args()
    if args.check_ready:
        missing = [name for name in required() if not (OUT / name).exists()]
        if not (PARENT / 'completion.json').exists():
            missing.append('parent completion.json')
        print(json.dumps(dict(ready=not missing, missing=missing, report_written=False)))
        return
    data, audits, windows, hashes = evidence()
    text, completion = compose(data, audits, windows, hashes)
    write(OUT / 'RESULTS.txt', text)
    completion['report_sha256'] = sha(OUT / 'RESULTS.txt')
    write(OUT / 'completion.json', json.dumps(completion, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='complete', exploratory=True, parent_primary_unchanged=True)))


if __name__ == '__main__':
    main()
