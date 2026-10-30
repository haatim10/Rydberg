"""Report the frozen correction and fresh comparisons without changing the paper."""
import json
import numpy as np
from .analysis import belongs,PRIMARY,HISTORICAL
from .study import OUT,SEEDS,GROUPS,sha

LABELS=[('original_mean','Retrained model'),('old_bypass_mean','Old spectral-EM100 bypass'),
    ('prior_repair_mean','Previous transition repair'),('current_repair_mean','Current correction'),('repair_mean','New refinement'),
    ('em100','Direct spectral EM100'),('prior_compact','Frozen current compact sparse'),
    ('compact_sparse','New matched sparse/EM family'),('prior_sparse_full','Frozen full sparse'),('optshrink','Frozen OptShrink')]

def read(name):return json.loads((OUT/name).read_text())
def interval(values,scale=1,digits=4):return '['+', '.join(f'{x*scale:.{digits}f}' for x in values)+']'
def table(headers,rows):
    cells=[[str(x) for x in row] for row in [headers]+rows]
    widths=[max(len(r[i]) for r in cells) for i in range(len(headers))]
    fmt=lambda row:'  '.join(x.ljust(w) for x,w in zip(row,widths))
    return '\n'.join([fmt(cells[0]),fmt(['-'*w for w in widths])]+[fmt(r) for r in cells[1:]])

def main():
    n,b,t,p=map(read,['nmse_summary.json','ber/summary.json','timing.json','policy.json'])
    audit_names=['checks.json','risk_checks.json','risk_data_checks.json','shrink_checks.json','audit/calibration.json','audit/repair_selection.json','audit/selection.json',
        'audit/test.json','audit/summary.json','audit/timing.json','ber/audit.json','sensitivity/audit.json',
        'audit/sensitivity.json','historical_replay/audit.json']
    for name in audit_names:assert read(name)['passed'],name
    digest=sha(OUT/'policy.json')
    assert all(x['selection_sha256']==digest for x in (n,b,t))
    assert n['timing_sha256']==sha(OUT/'timing.json')
    for name in ('audit/selection.json','audit/test.json','audit/summary.json','audit/timing.json','ber/audit.json','audit/sensitivity.json','historical_replay/audit.json'):
        assert read(name)['selection_sha256']==digest,name
    for record,key,artifact in [('audit/summary.json','summary_sha256','nmse_summary.json'),
        ('audit/summary.json','bootstrap_sha256','nmse_bootstrap.npz'),('audit/timing.json','timing_sha256','timing.json'),
        ('audit/sensitivity.json','sensitivity_summary_sha256','sensitivity/summary.json'),
        ('historical_replay/audit.json','summary_sha256','historical_replay/summary.json')]:
        assert read(record)[key]==sha(OUT/artifact),(record,artifact)
    samples=dict(t['raw_samples'])
    samples.update({name:[row for member in members for row in samples[member]] for name,members in GROUPS.items()})
    def latency(name,region):
        values=[r['ms'] for r in samples[name] if belongs(region,r['snr'],r['rsr'],r['regime'])]
        return float(np.median(values)),float(np.quantile(values,.9))
    target=n['aggregates']['strong_transition'];bt=b['regions']['strong_transition']
    old=target['contrasts']['repair_vs_old'];prev=target['contrasts']['repair_vs_prior'];current=target['contrasts']['repair_vs_current']
    replay=read('historical_replay/summary.json')
    calibrated_cap=t['cap_verification']['calibrated_cap_ms']
    prior_outside=[c for c in n['cells'] if c['contrasts']['repair_vs_prior']['gain_db'] < -1e-10 and not belongs('strong_transition',c['snr'],c['rsr'],c['regime'])]
    prior_outside_max=max((-c['contrasts']['repair_vs_prior']['gain_db'] for c in prior_outside),default=0.)
    lines=['INCUMBENT-GUARDED TRANSITION REFINEMENT: FRESH NMSE, BER AND RUNTIME',
        'Completed experiment. The manuscript, prior studies and retrained checkpoints remain unchanged.','',
        'KEY RESULTS',
        f"Primary target NMSE gains: {old['gain_db']:+.4f} dB versus old bypass and {prev['gain_db']:+.4f} dB versus previous repair, {current['gain_db']:+.4f} dB versus current correction. Positive means improvement.",
        f"Primary observed BER: old bypass {100*bt['methods']['old_bypass_mean']['ber']:.6f}%, previous repair {100*bt['methods']['prior_repair_mean']['ber']:.6f}%, current correction {100*bt['methods']['current_repair_mean']['ber']:.6f}%, refinement {100*bt['methods']['repair_mean']['ber']:.6f}%.",
        f"Primary p90 latency: correction {latency('repair_mean','strong_transition')[1]:.3f} ms, previous repair {latency('prior_repair_mean','strong_transition')[1]:.3f} ms, old bypass {latency('old_bypass_mean','strong_transition')[1]:.3f} ms.",
        f"Overall correction p90 is {t['methods']['repair_mean']['p90_ms']:.4f} ms versus the calibrated cap of {calibrated_cap:.4f} ms ({100*(t['methods']['repair_mean']['p90_ms']/calibrated_cap-1):+.2f}%). Matched sparse/EM p90 is {t['methods']['compact_sparse']['p90_ms']:.4f} ms, "+('below both the calibration cap and fresh correction p90.' if not t['cap_verification']['compact_exceeds_calibrated_cap'] and not t['cap_verification']['compact_exceeds_fresh_repair'] else 'with budget exceedances retained in the runtime section.'),
        f"Exposed historical replay: {len(replay['flagged_cells_remaining_negative_vs_prior'])}/8 flagged conditions remain worse than previous repair. This is a regression diagnostic, not independent evidence.",
        f"Fresh historical-condition subset: {sum(c['cell'] in HISTORICAL and c['contrasts']['repair_vs_prior']['gain_db'] < -1e-10 for c in n['cells'])}/8 mean NMSE losses versus previous repair. This reuses the condition names, not the exposed channels.",
        f"Fresh full-grid trade-offs: {len(n['negative_cells']['repair_vs_old']['negative_cells'])}/153 negative mean gains versus old bypass; {len(n['negative_cells']['repair_vs_prior']['negative_cells'])}/153 versus previous repair; {len(n['negative_cells']['repair_vs_current']['negative_cells'])}/153 versus current correction; {len(n['negative_cells']['repair_vs_original']['negative_cells'])}/153 versus original model. All raw gains and descriptive intervals are retained below.",
        f"Fresh previous-repair losses comprise {len(n['negative_cells']['repair_vs_prior']['target_negative_cells'])} target conditions and {len(prior_outside)} conditions outside the target; the latter have losses no larger than {prior_outside_max:.9g} dB. The worst previous-repair loss is {-n['negative_cells']['repair_vs_prior']['worst_gain_db']:.9g} dB at {n['negative_cells']['repair_vs_prior']['worst_cell']}.",
        f"Target-region NMSE advantages: {target['methods']['prior_sparse_full']['nmse_db']-target['methods']['repair_mean']['nmse_db']:+.4f} dB versus full sparse and {target['methods']['optshrink']['nmse_db']-target['methods']['repair_mean']['nmse_db']:+.4f} dB versus OptShrink. Interpret these gains together with the all-grid comparison below.",
        'All 135 conditions outside the declared target preserve the current correction exactly. All 51 weak-reference conditions preserve the previous repair; all 45 ordinary-range outputs preserve the original retrained model.',
        'Every adverse condition, fitted-model variation, zero-error limitation and timing-cap comparison is retained below.','',
        'WHAT CHANGED AND HOW IT WAS SELECTED',
        'The existing three 30-pilot trained models are frozen. Reference bin 5 retains the previous repair exactly. Stronger reference bins retain the original estimator through the nearest 30-dB SNR bin and use canonical spectral EM100 from the nearest 40-dB bin upward.',
        'Only reference bins 12/20 at SNR 32.5, 35, 37.5 may change. The candidate menu is recorded in protocol.json; development-only screens compared EM tails, convex channel blends, and final noise-scaled Hankel shrinkage.',
        'The three frozen fits receive 30 context pilots. Each fresh development anchor adds 9 independently generated Gaussian held-out pilots after the context is accepted. Held-out pilots are used for policy selection only; deployed estimates use 30 pilots.',
        'Selection compares each context-only estimate against four incumbents: original, old bypass, previous repair and current correction. A candidate must have a one-sided 95% paired-bootstrap upper bound on observable channel-risk difference at or below zero in EACH path regime. Three fit deltas are averaged within anchor before 4,000 coupled resamples.',
        'The observable risk is an unbiased difference of raw coefficient MSE divided by N*K under independent proper Gaussian held-out pilots and correct noise second moment. It uses intensity residuals around the pair midpoint, known pilots/reference and supplied noise variance. It does not use true channels, true path counts, or true channel-energy normalization. Finite pairwise scores need not be transitive.',
        'Selection minimizes mean risk versus the fixed current correction among guarded candidates. There is no likelihood preference band. Held-out likelihood is retained as a diagnostic, not the selection objective.',
        'Individual development bounds have no simultaneous coverage over candidates/bins and cannot guarantee NMSE or BER noninferiority. An empty guard intersection retains the current correction with an explicit fallback flag. Such a fallback does not claim to satisfy the missing guards.',
        'The final Hankel filter uses a pilot-Gram/noise-derived heuristic threshold and a smooth singular-value shrinker. Its scale is not a known covariance of the nonlinear estimate; it assumes no true channel rank.',
        'Routing uses supplied noise variance, known reference power and nearest declared bins. It never uses true channels or path counts. Path regimes are calibration strata and reporting labels only.',
        'All choices are frozen before fresh test generation. The matched sparse/EM family is retuned only in the same six bins using the same 384-anchor development banks; the 45 remaining choices preserve the current frozen compact policy exactly. Runtime eligibility is recalibrated for all 16 candidates after refinement selection and before sparse selection.',
        'The full sparse, OptShrink and previous compact policies remain frozen transfer controls, with their original calibration scope. The compact menu is a limited declared family, not a globally optimal sparse method.','',
        'FRESH CONFIRMATION DESIGN',
        'The fresh test contains 1,152 channel/reference anchors, reused across 17 SNRs for 19,584 operating cases in 153 conditions. All estimators share pilot observations, data observations and the magnitude-domain Rician detector.',
        'The primary region contains all 18 conditions at references 12/20 and SNR 32.5/35/37.5, including all three path regimes. Refinement versus old bypass, previous repair and current correction, times NMSE and BER, form six prespecified outcomes.',
        'Primary intervals use 20,000 paired whole-anchor resamples and two-sided 99.1667% percentile intervals for nominal 95% Bonferroni coverage across six outcomes. Shared indices preserve coupling across references, SNRs, methods and fits within each regime.',
        'Resampling uses 384 local-index clusters (128 in each path regime). The 1,152 channel/reference combinations and 19,584 operating cases are not treated as independent channel draws.',
        'Intervals condition on the fixed checkpoints, selected policies and test grid; they exclude training-population and selection uncertainty. Other regions, subgroups and individual-cell intervals are descriptive pointwise 95%.',
        'NMSE is the equal-condition mean of dB ratio-of-sums scores. Pooled linear NMSE is also reported. Three-fit groups average individual metrics, never channel estimates; they do not triple independent channels.',
        'BER exhausts all 64 joint Gray-QPSK vectors per channel with common independent data noise, using the same exact 64-hypothesis detector. There are 884,736 target bits per fit and 7,520,256 full-grid bits per fit. Zero observed errors and degenerate [0,0] bootstrap intervals do not establish zero BER or population noninferiority.','',
        'PRIMARY EFFECTS']
    for key in PRIMARY:
        nr=target['contrasts'][key];br=bt['contrasts'][key]
        lines +=[key+':',f"  NMSE gain {nr['gain_db']:.4f} dB; adjusted interval {interval(nr['bonferroni6_interval_db'])} dB.",
            f"  BER reduction {100*br['ber_reduction']:.6f} percentage points; adjusted interval {interval(br['bonferroni6_interval'],100,6)}.",
            '  Fit-specific NMSE gains: '+str(nr['seed_specific_gains_db'])]
    lines+=['Target bit-error counts by fitted model, in seed order '+str(SEEDS)+':']
    for group in ('original','old_bypass','prior_repair','current_repair','repair'):
        lines +=[group+': '+str([round(bt['methods'][f'{group}_{seed}']['ber']*884736) for seed in SEEDS])]
    lines+=['','COMPARISON TABLES','All values lower is better. CPU latency is batch-one, one Apple M2 thread, including lookup, initialization and all inference steps; models and fixed tables are preloaded.','']
    regions=('strong_transition','strong_transition_L2','strong_transition_L4','strong_transition_random3to7',
        'strong_transition_snr32.5','strong_transition_snr35','strong_transition_snr37.5','historical_flagged','weak_transition','ordinary','high_snr','all')
    for region in regions:
        nr=n['aggregates'][region];br=b['regions'][region];rows=[]
        for name,label in LABELS:
            median,p90=latency(name,region)
            rows.append([label,f"{nr['methods'][name]['nmse_db']:.4f}",f"{nr['methods'][name]['pooled_ratio_of_sums_db']:.4f}",f"{100*br['methods'][name]['ber']:.6f}",f'{p90:.3f}',f'{median:.3f}'])
        lines +=[region.upper()+f" ({nr['cells']} conditions)",table(['Estimator','Mean NMSE dB','Pooled NMSE dB','BER %','p90 ms','Median ms'],rows),'']
    lines+=['EVERY ADVERSE CONDITION','Negative gain means correction is worse. A reporting tolerance of 1e-10 dB distinguishes numerical identity; exact scores remain in JSON.','']
    for key,row in n['negative_cells'].items():
        lines +=[key+':',f"  Negative means {len(row['negative_cells'])}/153, including {len(row['target_negative_cells'])}/18 primary conditions; pointwise 95% harm intervals in {len(row['pointwise95_harm_cells'])} conditions.",
            f"  Worst gain {row['worst_gain_db']:.6f} dB at {row['worst_cell']}.",'  Negative cells: '+(', '.join(row['negative_cells']) or 'none')]
    lines+=['','EVERY PRIMARY AND HISTORICALLY FLAGGED CONDITION','']
    rows=[]
    for c in n['cells']:
        if not belongs('strong_transition',c['snr'],c['rsr'],c['regime']) and c['cell'] not in HISTORICAL:continue
        rows.append([c['cell'],*[f"{c['methods'][m]['nmse_db']:.5f}" for m in ('original_mean','old_bypass_mean','prior_repair_mean','current_repair_mean','repair_mean')],
            interval(c['contrasts']['repair_vs_old']['pointwise_ci95_db']),interval(c['contrasts']['repair_vs_prior']['pointwise_ci95_db'])])
    lines+=[table(['Condition','Original','Old bypass','Prior repair','Current correction','Refinement','Gain vs old CI95','Gain vs prior CI95'],rows),'',
        'PRIMARY TAIL DIAGNOSTICS','Per-channel linear-NMSE quantiles, averaged across fits; severe means NMSE>1.',
        table(['Estimator','p95 NMSE','p99 NMSE','Severe %'],[[label,f"{target['methods'][name]['case_nmse_p95']:.6g}",f"{target['methods'][name]['case_nmse_p99']:.6g}",f"{100*target['methods'][name]['severe_rate']:.4f}"] for name,label in LABELS]),'',
        'FROZEN POLICY',table(['SNR dB','Reference 5','Reference 12','Reference 20'],[[snr]+[p['repair_choices'][ref][snr]['choice'] for ref in ('5','12','20')] for snr in p['repair_choices']['5']]),
        'Selected calibrated-bin records (including fallback/admissibility metadata):',
        json.dumps({ref:{s:p['repair_choices'][ref][s] for s in ('32.5','35','37.5')} for ref in ('12','20')},indent=2),'',
        'RUNTIME ELIGIBILITY AND CONDITIONAL OVERRUNS',
        'The eligibility cap is the selected correction pooled p90 across all 153 conditions and three fixed fits. It is not a per-condition or worst-case deadline. Final isolated timing uses three repetitions and randomized method order; all raw samples are retained without reselection.',json.dumps(t['cap_verification'],indent=2)]
    cap=t['cap_verification']['calibrated_cap_ms']
    for name in ('repair_mean','compact_sparse'):
        excess=[]
        for ref,snr in sorted({(r['rsr'],r['snr']) for r in samples[name]}):
            value=float(np.quantile([r['ms'] for r in samples[name] if r['rsr']==ref and r['snr']==snr],.9))
            if value>cap:excess.append([ref,snr,f'{value:.4f}',f'{100*(value/cap-1):.2f}'])
        lines +=[f'{name}: {len(excess)}/51 conditional p90 values exceed the pooled cap.',table(['Reference','SNR','p90 ms','Above cap %'],excess)]
    lines+=['','EXPOSED-BANK REGRESSION REPLAY',
        'These are the previously exposed channels, replayed once after final selection. They diagnose the requested historical failures and sibling conditions; they are not independent confirmation and never trigger reselection.',
        'Flagged cells still worse than previous repair: '+str(replay['flagged_cells_remaining_negative_vs_prior']),
        table(['Condition','Correction NMSE dB','Gain vs old','Gain vs prior','Gain vs current','Gain vs original'],[[c['cell'],f"{c['methods']['repair_mean']['nmse_db']:.6f}",*[f"{c['contrasts'][key]['gain_db']:.6f}" for key in ('repair_vs_old_bypass','repair_vs_prior_repair','repair_vs_current_repair','repair_vs_original')]] for c in replay['cells']]),'',
        'SUPPLIED-NOISE SENSITIVITY',
        'Physical observations are fixed while supplied noise variance changes by -3/0/+3 dB in 45 predeclared conditions. Positive mismatch loss compares the SAME correction under incorrect versus correct variance; it is not a gain versus another estimator. Comparator gains supply both methods the same incorrect variance. No sensitivity BER or tuning is performed.',
        table(['Paths','Ref','SNR','Variance error','Branch','Correction NMSE','Mismatch loss','Loss CI95'],[[c['regime'],c['rsr'],c['actual_snr_db'],c['supplied_variance_offset_db'],c['repair_choice'],f"{c['methods']['repair_mean']['nmse_db']:.4f}",f"{c['methods']['repair_mean']['degradation_vs_correct_variance_db']:.4f}",interval(c['methods']['repair_mean']['degradation_bootstrap95'])] for c in read('sensitivity/summary.json')['cells']]),'',
        'FILTER-OFF BOUNDARY DIAGNOSTIC',
        'The prespecified rule preserves the current correction from the 40-dB bin upward. The following positive differences mean the new estimate becomes worse when SNR rises from 37.5 to 40 dB because the added filter switches off. This descriptive check is not a new primary outcome and does not trigger reselection.',
        table(['Paths','Ref','New37.5dB NMSE','New40dB NMSE','Worsening dB'],[[regime,ref,
            f"{next(c for c in n['cells'] if c['regime']==regime and c['rsr']==ref and c['snr']==37.5)['methods']['repair_mean']['nmse_db']:.4f}",
            f"{next(c for c in n['cells'] if c['regime']==regime and c['rsr']==ref and c['snr']==40)['methods']['repair_mean']['nmse_db']:.4f}",
            f"{next(c for c in n['cells'] if c['regime']==regime and c['rsr']==ref and c['snr']==40)['methods']['repair_mean']['nmse_db']-next(c for c in n['cells'] if c['regime']==regime and c['rsr']==ref and c['snr']==37.5)['methods']['repair_mean']['nmse_db']:+.4f}"]
            for regime in ('L2','L4','random3to7') for ref in (12,20)]),'',
        'LIMITS',
        'This study evaluates a combined wrapper around existing learned weights. It does not demonstrate a newly learned controller, global convergence, universal safety or dominance over all baselines.',
        'The four-incumbent guard is an empirical development screen. Empty-set fallback and metric mismatch remain explicit limitations; NMSE, BER and runtime must be interpreted together.',
        'Reference/SNR interpolation beyond the declared nearest-bin rule is unvalidated. Noise misspecification can change both routing and inference. Preserving the weak branch preserves its previously identified mismatch vulnerability too.',
        'The exposed historical replay and the fresh test are kept separate. The manuscript is unchanged.','',
        'VERIFICATION','All required checks passed: '+', '.join(audit_names),'Frozen policy SHA256: '+digest,'',
        'ARTIFACTS','nmse_summary.json; repair_nmse.png; repair_target.png; repair_runtime.png; ber/summary.json; ber/ber.png; timing.json; historical_replay/summary.json; sensitivity/summary.json; policy.json; protocol.json','']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines))
    print(OUT/'RESULTS.txt')

if __name__=='__main__':main()
