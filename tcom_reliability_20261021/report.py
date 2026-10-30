"""Reporting only: verified reliability-check comparison and operational status."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/tcom_reliability_20261021'
SCOPES = ('all','weak_target','ordinary','transition_all','high_all','weak_warm100','strong_high')
PRIMARY = ('receiver_vs_prior','receiver_vs_latest')
LABELS = {
    'receiver_mean':'A-selected reliability receiver', 'prior_mean':'Prior selective EM',
    'latest_mean':'Frozen latest', 'compact_sparse':'Compact sparse',
    'adaptive40_mean':'Adaptive start + EM40', 'global40_mean':'Global schedule + EM40',
    'classical40':'Classical start + EM40', 'em100':'Direct EM100', 'algebraic20':'Standalone algebraic + EM20'}
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
    return ['protocol.json','policy.json','nmse_summary.json','nmse_bootstrap.npz',
            'weak_error_concentration.json','ber/summary.json','ber/bootstrap.npz',
            'timing_window1.json','timing_window2.json','ber/audit.json',
            *(f'audit/{name}.json' for name in AUDITS)]


def evidence():
    missing=[name for name in required() if not (OUT/name).exists()]
    if missing:
        raise RuntimeError('Final report awaits: '+', '.join(missing))
    data={name:load(OUT/name) for name in required() if name.endswith('.json')}
    protocol,policy=data['protocol.json'],data['policy.json']
    digest,protocol_digest=sha(OUT/'policy.json'),sha(OUT/'protocol.json')
    assert policy['protocol_sha256']==protocol_digest
    for field in ('source_hashes','checkpoint_hashes'):
        for name,value in protocol[field].items():
            assert sha(REPO/name)==value,name
    for name,value in policy['selection_evidence'].items():
        assert sha(OUT/name)==value,name
    audits={name:data[f'audit/{name}.json'] for name in AUDITS}
    audits['ber']=data['ber/audit.json']
    for name,item in audits.items():
        assert item['passed'] and item['selection_sha256']==digest,name
        assert item['protocol_sha256']==protocol_digest,name
    nmse,ber=data['nmse_summary.json'],data['ber/summary.json']
    for item in (nmse,ber):
        assert item['selection_sha256']==digest and item['protocol_sha256']==protocol_digest
        assert item['confirmation']==policy['confirmation']
        assert item['candidate_rule']==policy['candidate_rule']
        assert item['operational_rule']==policy['operational_rule']
        assert item['bootstrap_index_clusters']==1152 and item['unique_test_anchors']==1152
    for cell,value in audits['test']['test_prediction_hashes'].items():
        assert sha(OUT/'test/cells'/cell)==value
    for cell,value in audits['ber']['cell_hashes'].items():
        path=OUT/'ber/cells'/cell
        if not path.exists():path=path.with_suffix('.npz')
        assert sha(path)==value
    assert audits['summary']['summary_sha256']==sha(OUT/'nmse_summary.json')
    assert audits['summary']['ber_summary_sha256']==sha(OUT/'ber/summary.json')
    assert audits['summary']['nmse_bootstrap_sha256']==sha(OUT/'nmse_bootstrap.npz')
    assert audits['summary']['ber_bootstrap_sha256']==sha(OUT/'ber/bootstrap.npz')
    assert nmse['weak_error_concentration_sha256']==sha(OUT/'weak_error_concentration.json')
    assert audits['summary']['weak_error_concentration_sha256']==sha(OUT/'weak_error_concentration.json')
    windows={f'window{i}':data[f'timing_window{i}.json'] for i in (1,2)}
    for key,item in windows.items():
        assert item['selection_sha256']==digest and item['protocol_sha256']==protocol_digest
        assert audits['timing']['timing_hashes'][f'timing_{key}.json']==sha(OUT/f'timing_{key}.json')
    hashes={name:sha(OUT/name) for name in required()}
    hashes.update({name:sha(OUT/name) for name in policy['selection_evidence']})
    return data,audits,windows,hashes


def compose(data,audits,windows,hashes):
    protocol,policy,nmse,ber=(data[name] for name in ('protocol.json','policy.json','nmse_summary.json','ber/summary.json'))
    confirmation=policy['confirmation'];candidate=policy['candidate_rule'];operational=policy['operational_rule']
    assert operational==(candidate if confirmation['passed'] else None)
    status=confirmation['status']
    lines=['REFERENCE-AWARE RELIABILITY CHECK: FRESH PAIRED COMPARISON','',
        f'Independent confirmation status: {status.upper()}.',
        ('The evaluated receiver is the frozen A-selected proposal and C confirmed it.' if confirmation['passed'] else
         'The evaluated receiver is the frozen A-selected proposal. It is unconfirmed; the operational recommendation remains the prior selective-EM estimator.'
         if candidate is not None else
         'Development A selected the incumbent. The receiver arm retains the prior selective-EM estimator.'),
        'Fresh-test outcomes do not change the selected candidate or the confirmation decision. '
        'No model has been deployed and no manuscript has been edited.',
        'Receiver versus prior and versus latest over all 153 conditions are the two primary '
        'contrasts. Their NMSE and BER outcomes form four tests with Bonferroni adjustment. '
        'Weak-reference, other regional and baseline results are descriptive.','',
        'PRIMARY EFFECTS','Positive values favor the receiver; BER reductions are percentage points.']
    primary={};rows=[]
    for key in PRIMARY:
        n=nmse['aggregates']['all']['contrasts'][key];b=ber['regions']['all']['contrasts'][key]
        primary[key]=dict(nmse=n,ber=b)
        rows.append((key,f"{n['gain_db']:+.6g}",interval(n['bonferroni4_interval_db']),
                     f"{100*b['ber_reduction']:+.6g}",interval(b['bonferroni4_interval'],100)))
    overall=nmse['aggregates']['all']['methods']; overall_ber=ber['regions']['all']['methods']
    lines += [table(('Contrast','NMSE gain dB','Adjusted interval dB','BER reduction pp','Adjusted interval pp'),rows),'',
        'The receiver improves mean-cell NMSE versus both primary references. Its observed '
        'BER equals the prior selective-EM receiver, and the adjusted BER interval versus '
        'latest includes zero. Compact sparse remains better under pooled-linear NMSE '
        f"({overall['compact_sparse']['pooled_ratio_of_sums_linear']:.8g} versus "
        f"{overall['receiver_mean']['pooled_ratio_of_sums_linear']:.8g} for the receiver) "
        f"and aggregate BER ({100*overall_ber['compact_sparse']['ber']:.6g}% versus "
        f"{100*overall_ber['receiver_mean']['ber']:.6g}%). The mean-cell NMSE advantage "
        'therefore does not imply superiority under every NMSE aggregation or in detection.','',
        'SELECTED CHECK AND OPERATIONAL STATUS',
        'A-selected candidate: '+json.dumps(candidate,sort_keys=True),
        'C-authorized operational rule: '+json.dumps(operational,sort_keys=True),
        'An absent rule means the frozen prior estimator. The receiver column always denotes '
        'the A-selected candidate, even when C fails; it must not be relabeled as the operational incumbent.',
        'The check uses fitted magnitude residuals with known reference and variance. It is a '
        'heuristic diagnostic, not a calibrated failure probability. Only the same 30 fitting '
        'pilots are available online. The algebraic fallback and any additional EM/filter work '
        'are accepted only by the frozen observable likelihood rule.',
        'The standalone algebraic20 baseline is reported in every accuracy, BER and runtime '
        'comparison table. It applies the algebraic start and 20 EM updates independently '
        'of the receiver gate; an invalid algebraic solve uses the frozen spectral start '
        'with 20 EM updates. Its final filter matches the defined baseline pipeline.',
        'Pilot-design scope: the current 15-column algebraic least-squares implementation '
        'requires sufficiently rich, amplitude-varying fitting pilots; this study uses 30 '
        'Gaussian fitting pilots. With constant-modulus/QPSK fitting pilots, the three '
        'diagonal nuisance columns are collinear, so rank is at most 13 and the current '
        'full-rank-15 validity gate rejects the solve. The gated receiver then retains the '
        'prior, while standalone algebraic20 takes its declared spectral-plus-EM20 branch. '
        'This is a limitation of this formulation and validity rule, not a general claim '
        'that channel recovery with constant-modulus pilots is impossible. The 64 QPSK '
        'vectors used here are separate data/probes, not the 30 fitting pilots.',
        'Development A and independent C use heldout observations and known QPSK probes. '
        'C requires a positive pooled weak-target NLL lower bound and empirical probe BER '
        'nonincrease in every path regime. This is not a formal BER noninferiority guarantee.',
        'The prior selective-EM policy is inherited exactly from the earlier exploratory B '
        'proposals; those proposals failed their original C confirmation. This experiment '
        'does not retroactively confirm that prior study.','']
    for scope in SCOPES:
        n,b=nmse['aggregates'][scope],ber['regions'][scope];rows=[]
        for name,label in LABELS.items():
            item=n['methods'][name]
            rows.append((label,f"{item['nmse_db']:.5f}",f"{item['pooled_ratio_of_sums_linear']:.7g}",
                f"{100*b['methods'][name]['ber']:.7g}",f"{item['case_nmse_p95']:.4g}",
                f"{item['case_nmse_p99']:.4g}",f"{item['case_nmse_max']:.4g}"))
        lines += [f"{scope.upper()} ({n['cells']} conditions)",
            table(('Method','Mean-cell NMSE dB','Pooled linear','BER %','Case p95','Case p99','Case max'),rows),'']
    lines += ['CHECK TRIGGERS AND ACCEPTED FALLBACKS','All counts retain the separate training fits.']
    rows=[]
    for scope in ('all','weak_target','ordinary','strong_high'):
        for seed,item in nmse['aggregates'][scope]['gate'].items():
            rows.append((scope,seed,item['cases'],item['triggers'],item['accepted'],item['rank_invalid'],
                         f"{100*item['trigger_fraction']:.4f}",f"{100*item['accepted_fraction']:.4f}"))
    lines += [table(('Scope','Fit','Cases','Triggered','Accepted','Rank invalid','Trigger %','Accepted %'),rows),'']
    losses,rows={},[]
    for key in nmse['contrasts']:
        n,b=nmse['negative_cells'][key],ber['negative_cells'][key];losses[key]=dict(nmse=n,ber=b)
        rows.append((key,len(n['negative_cells']),len(n['pointwise95_harm_cells']),len(b['negative_cells']),
            len(b['pointwise95_harm_cells']),n['worst_cell'],f"{n['worst_gain_db']:+.6g}"))
    lines += ['CONDITION-SPECIFIC LOSSES','Complete adverse-cell lists are retained in completion.json; pointwise harm intervals are descriptive.',
        table(('Contrast','NMSE worse','NMSE harm CI','BER worse','BER harm CI','Worst NMSE cell','Gain dB'),rows),'']
    rows=[]
    for scope in ('all','weak_target','ordinary','strong_high'):
        for family in ('receiver','prior','latest'):
            for seed in (9101,9102,9103):
                name=f'{family}_{seed}';n=nmse['aggregates'][scope]['methods'][name];b=ber['regions'][scope]['methods'][name]
                rows.append((scope,name,f"{n['nmse_db']:.5f}",f"{100*b['ber']:.7g}",b['bit_errors'],b['bits_per_fit']))
    lines += ['FIT SPREAD AND ACTUAL BIT COUNTS',table(('Scope','Fit','NMSE dB','BER %','Errors','Bits'),rows),'']
    concentration=data['weak_error_concentration.json'];rows=[]
    for r in concentration['weak_target_by_regime']:
        if not r['method'].startswith(('receiver_','prior_','latest_')):continue
        rows.append((r['regime'],r['method'],r['top1_local_index'],f"{100*r['top1_error_share']:.4f}",
                     f"{r['case_nmse_p95']:.5g}",f"{r['case_nmse_p99']:.5g}",f"{r['case_nmse_max']:.5g}"))
    lines += ['WEAK-TARGET ERROR CONCENTRATION',
        'For each fit/path regime, sum each physical anchor’s numerator error over the eight '
        'coupled SNRs, then identify the largest contributor. Its share is not a failure '
        'probability; repeated SNR appearances are not independent failures. Complete per-cell '
        'and baseline diagnostics are retained in weak_error_concentration.json.',
        table(('Regime','Fit','Top anchor','Error share %','Case p95','Case p99','Case max'),rows),'']
    runtime,rows={},[]
    for name,label in LABELS.items():
        a,b=(windows[k]['methods'][name] for k in ('window1','window2'))
        fullpass=all(w['p90_ms']<=10.74 for w in (a,b))
        weakpass=all(w['weak_target']['p90_ms']<=10.74 for w in (a,b))
        runtime[name]=dict(window1=a,window2=b,full_grid_both_windows_pass=fullpass,
            weak_target_both_windows_pass=weakpass,
            conditional_overruns={key:[c for c,v in windows[key]['methods'][name]['by_condition'].items()
                                      if v['p90_ms']>10.74] for key in windows})
        rows.append((label,f"{a['p90_ms']:.4f}",f"{b['p90_ms']:.4f}",
            f"{a['weak_target']['p90_ms']:.4f}",f"{b['weak_target']['p90_ms']:.4f}",
            'PASS' if fullpass else 'FAIL','PASS' if weakpass else 'FAIL'))
    lines += ['COMPLETE-PIPELINE RUNTIME',protocol['timing']['acceptance'],
        table(('Pipeline','W1 p90 ms','W2 p90 ms','W1 target p90','W2 target p90','Full cap','Target cap'),rows),'']
    rows=[]
    for key in ('window1','window2'):
        item=windows[key]['methods']['receiver_mean'];branch=item['triggered_branch']
        rows.append((key,item['trigger_count'],item['accepted_count'],
                     'no triggers' if branch is None else f"{branch['p90_ms']:.4f}",
                     len(runtime['receiver_mean']['conditional_overruns'][key])))
    lines += [table(('Window','Triggered calls','Accepted calls','Triggered-branch p90 ms','Overrun bins /51'),rows),'']
    latency_extremes,rows={},[]
    for key in ('window1','window2'):
        samples=[dict(method=name,**sample)
                 for name,records in windows[key]['raw_samples'].items()
                 if name.startswith('receiver_') for sample in records]
        assert len(samples)==windows[key]['methods']['receiver_mean']['calls']
        maximum=max(samples,key=lambda sample:sample['ms'])
        overruns=[sample for sample in samples if sample['ms']>10.74]
        latency_extremes[key]=dict(max_ms=maximum['ms'],calls=len(samples),
            over_budget_calls=len(overruns),maximum_call=maximum,over_budget_samples=overruns)
        rows.append((key,f"{maximum['ms']:.6f}",f'{len(overruns)}/{len(samples)}',
                     maximum['method'],maximum['regime'],maximum['rsr'],maximum['snr'],
                     maximum['index'],str(maximum['gate_triggered']).lower()))
    lines += ['INDIVIDUAL RECEIVER LATENCIES',
        table(('Window','Maximum ms','Calls >10.74 ms','Maximum-call fit','Regime','Reference','SNR','Index','Gate triggered'),rows),
        'The receiver passes the declared p90 targets in both windows, but an individual '
        'window-2 call exceeds 10.74 ms. Its gate was false; these measurements do not establish '
        'the cause of the outlier. Latencies vary substantially between the two windows; '
        'the slower second window remains part of the reported acceptance check. The p90 '
        'target is not a hard per-call deadline.','',
        'INTERPRETATION AND VERIFICATION',
        'Both learned starts contain ten EM/structure stages before any continuation. The '
        'global schedule has 30 globally trained parameters; the 188-parameter adaptive controller '
        'conditions decisions on observations. A zero extra tail does not mean zero estimation iterations.',
        'Reference domains in the fresh generator separate both channel/noise and random path '
        'counts. The 20,000 paired bootstrap draws resample 128 indices independently inside each '
        'of nine (regime, reference) blocks, and share those indices across SNR/method/fit inside '
        'each block: 1,152 physical anchor clusters. Fit means average metrics, not predictions.',
        'Mean-cell NMSE first computes 10 log10(total squared channel error / total channel '
        'energy) within each condition and training fit, then averages those dB values equally '
        'over conditions and fits. Pooled linear NMSE instead sums errors and channel energy '
        'over all conditions before dividing, separately for each fit, then averages the fit '
        'ratios arithmetically. The summaries use different weights and can rank methods '
        'differently; neither is the mean of individual-channel NMSE ratios. Group p95/p99/max '
        'values average corresponding fit diagnostics. The grid has 108/153 conditions above '
        '20 dB; these are not deployment-weighted effects.',
        'Every estimator sees the same fresh data and magnitude-only 64-hypothesis Rician ML '
        'detector. N=32, K=3, P=30, known reference/variance, 64 joint QPSK vectors once/channel. '
        'Zero observed BER and collapsed intervals do not prove equivalence, zero population BER, '
        'rare-event bounds or population noninferiority. The host-specific p90 cap is not a hard deadline.',
        'OptShrink was omitted prospectively from this cap-focused run because prior measured '
        'runtime greatly exceeded 10.74 ms. No historical result is presented as a fresh comparison.',
        'All required source/data/arithmetic/BER/timing audits passed. No test-driven tuning, '
        'deployment decision, manuscript edit or publication claim is made by this report.','',
        'Figures: reliability_nmse.png and reliability_ber.png.']
    completion=dict(status='complete',all_required_audits_passed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=sha(OUT/'policy.json'),
        protocol_sha256=sha(OUT/'protocol.json'),confirmation=confirmation,
        candidate_rule=candidate,operational_rule=operational,primary_outcomes=primary,primary_multiplicity=4,
        conditional_losses=losses,runtime=runtime,receiver_latency_extremes=latency_extremes,
        artifact_hashes=hashes,source_sha256=sha(__file__),
        counts=dict(cells=153,anchors=1152,bootstrap_clusters=1152,bits_per_fit=7520256),
        manuscript_written=False,no_test_based_reselection=True)
    return '\n'.join(lines)+'\n',completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-ready', action='store_true')
    args = parser.parse_args()
    if args.check_ready:
        missing = [name for name in required() if not (OUT / name).exists()]
        print(json.dumps(dict(ready=not missing, missing=missing, report_written=False)))
        return
    data, audits, windows, hashes = evidence()
    text, completion = compose(data, audits, windows, hashes)
    write(OUT / 'RESULTS.txt', text)
    completion['report_sha256'] = sha(OUT / 'RESULTS.txt')
    write(OUT / 'completion.json', json.dumps(completion, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='complete',confirmation=completion['confirmation']['status'],required_audits_passed=True)))


if __name__ == '__main__':
    main()
