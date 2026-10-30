"""Human-readable evidence report; no estimation or model selection."""
import json
from pathlib import Path
import numpy as np
from . import study, analysis

LABELS={
 'original_mean':'Retrained learned model', 'old_bypass_mean':'Old EM100 bypass',
 'prior_repair_mean':'Earlier transition repair', 'current_repair_mean':'Latest refinement (before this fix)',
 'repair_mean':'Continuous filter (new)', 'em100':'Direct spectral EM100',
 'prior_compact':'Previously frozen compact baseline', 'compact_sparse':'New runtime-matched sparse/EM baseline',
 'prior_sparse_full':'Full sparse baseline', 'optshrink':'OptShrink baseline'}

def main():
    out=study.OUT;policy,digest=study.check_selection()
    nm=json.loads((out/'nmse_summary.json').read_text());ber=json.loads((out/'ber/summary.json').read_text())
    boundary=json.loads((out/'boundary/summary.json').read_text());timing=json.loads((out/'timing.json').read_text())
    checks=json.loads((out/'checks.json').read_text());guard=json.loads((out/'development/guard_summary.json').read_text())
    assert nm['selection_sha256']==ber['selection_sha256']==boundary['selection_sha256']==timing['selection_sha256']==digest
    audits={p.stem:json.loads(p.read_text()) for p in sorted((out/'audit').glob('*.json'))}
    assert {'repair_selection','calibration','selection','test','timing','summary','boundary_summary'}.issubset(audits)
    assert all(a.get('passed') for a in audits.values())
    ber_audit=json.loads((out/'ber/audit.json').read_text());assert ber_audit['passed']
    audits['ber']=ber_audit
    execution_audit=json.loads((out/'ber/execution_audit.json').read_text());assert execution_audit['passed']
    audits['ber_execution']=execution_audit
    samples=dict(timing['raw_samples']);samples.update({n:[r for m in members for r in samples[m]] for n,members in study.GROUPS.items()})
    lines=[]
    def add(text=''):lines.append(str(text))
    def latency(name,region):
        return np.quantile([r['ms'] for r in samples[name] if analysis.belongs(region,r['snr'],r['rsr'],r['regime'])],.9)
    add('40 dB FILTER SWITCH-OFF REPAIR — FRESH COMPARISON')
    add('Frozen study: tcom_filter_continuation_20261019')
    add('The paper has not been updated. Earlier studies, results and checkpoints are unchanged.')
    add()
    add('CHANGE AND VALIDATION')
    add('Keep the inherited c=4, beta=1 final Hankel filter on spectral EM100 in strong-reference bins 40, 42.5, 45, 50, 60 and above. The same recipe already applies at 37.5. The nearest-bin cutoff was 38.75dB; the filter now remains active across it. Tau is computed from actual supplied noise variance and tends to zero with that variance. No finite high-SNR switch-off remains in this branch; no neural retraining or new parameter search.')
    add('Only 30 of 153 grid conditions change. The other 123 are exactly the latest incumbent, including all 18 previous strong-transition conditions and all 51 weak-reference conditions. Ordinary 45 remain exactly original. These are output identities, not statistical noninferiority claims.')
    add('One fixed recipe was declared before development, globally accepted after all 30 independent-heldout observable-risk upper95 guards passed. Each guard used 384 anchors/regime, 30 fitting pilots plus 9 independent validation-only pilots, 4000 paired resamples. The bounds are development diagnostics, not simultaneous population guarantees.')
    add('The final policy and compact comparator were frozen before fresh test data. Deployment uses 30 pilots. Development never exported true channels. Fresh testing uses 128 anchors per reference/path setting; three fitted neural seeds average individual metrics and do not multiply the sample size. High-branch outputs are fit-independent.')
    add('NMSE: equal-cell mean of dB ratio-of-sums NMSE. Main 153-condition grid aggregates exclude 54 added boundary cells. 20,000 coupled channel-index bootstrap resamples per regime. Primary 30 cells compare new versus latest and matched compact on NMSE and BER: four outcomes, Bonferroni-adjusted adjusted intervals. All other intervals are descriptive.')
    add()
    for region,berkey,title in [('strong_high','primary','PRIMARY: STRONG REFERENCES, 40–60dB (30 conditions)'),('strong_transition','strong_transition','PRESERVED EARLIER TRANSITION (18 conditions)'),('all','overall','FULL 153-CONDITION GRID')]:
        add(title);add(f"{'Method':48} {'NMSE dB':>11} {'BER %':>12} {'p90 ms':>11}")
        for name,label in LABELS.items():
            add(f"{label:48} {nm['aggregates'][region]['methods'][name]['nmse_db']:11.6f} {100*ber[berkey]['methods'][name]['ber']:12.6f} {latency(name,region):11.4f}")
        add()
    add('PRIMARY EFFECTS (positive gain/reduction favors the new filter)')
    for key in ('repair_vs_current','repair_vs_compact'):
        a=nm['aggregates']['strong_high']['contrasts'][key];b=ber['primary']['contrasts'][key]
        add(f"{key}: NMSE gain={a['gain_db']:.6f}dB, Bonferroni4 interval={a['bonferroni4_interval_db']}; BER reduction={b['ber_reduction']:.9g}, adjusted interval={b['bonferroni4_interval']}")
    add('Latest incumbent equals direct EM100 throughout these 30 conditions. The fresh compact selector also chose EM100 in all 10 changed bins, so both primary NMSE effects are numerically identical. These are duplicate numerical evidence, not independent confirmation.')
    add('BER uses the same 64-vector joint Gray-QPSK alphabet, data magnitudes, reference, variance and exact magnitude-domain Rician ML detector for every estimator. 384 bits/channel,49,152 bits/condition/fit,1,474,560 bits per primary-region fit. Perfect CSI is an explicitly labeled diagnostic. Zero observed errors and collapsed bootstrap intervals do not establish zero BER, equivalence or noninferiority; no independent-bit binomial bounds are used.')
    add()
    add('BOUNDARY:37.5→40dB (positive change means worsening)')
    add(f"{'Regime':14} {'Ref dB':>7} {'Latest change':>15} {'New change':>15}")
    for r in boundary['changes_37_5_to_40']:
        add(f"{r['regime']:14} {r['rsr']:7g} {r['methods']['current_repair_mean']['change_db']:15.6f} {r['methods']['repair_mean']['change_db']:15.6f}")
    worsening=[r for r in boundary['adjacent_changes'] if r['methods']['repair_mean']['change_db']>1e-10]
    add(f"Dense boundary probe:72cells at 12 SNRs, 54 additional to main grid; {len(worsening)} of 66 adjacent new-filter changes worsen on the sample mean.")
    for r in worsening:add(f"  {r['regime']} ref{r['rsr']:g} {r['lower_snr']:g}→{r['upper_snr']:g}: {r['methods']['repair_mean']['change_db']:+.9g}dB")
    add('Fixed-observation epsilon tests separately verify continuity across the routing midpoint: new outputs match the same explicit formula on either side. Smooth sampled NMSE curves alone would not prove estimator continuity or globally monotone performance.')
    add()
    add('CELLWISE GAINS AND POSSIBLE HARMS')
    for key,item in nm['negative_cells'].items():
        add(f"{key}: mean-negative={len(item['negative_cells'])}/153; primary-negative={len(item['target_negative_cells'])}/30; pointwise95 harm={len(item['pointwise95_harm_cells'])}; worst={item['worst_cell']} ({item['worst_gain_db']:+.9g}dB)")
        if item['negative_cells']:add('  Mean-negative cells: '+', '.join(item['negative_cells']))
    add('All 30 changed cells: latest NMSE / new NMSE / gain, in dB')
    for row in nm['cells']:
        if analysis.belongs('strong_high',row['snr'],row['rsr'],row['regime']):
            add(f"  {row['cell']:32} {row['methods']['current_repair_mean']['nmse_db']:11.5f} {row['methods']['repair_mean']['nmse_db']:11.5f} {row['contrasts']['repair_vs_current']['gain_db']:10.5f}")
    add()
    add('RUNTIME AND COMPARATOR FAIRNESS')
    add('Apple M2 CPU, one thread; preloaded models/tables, batch-one complete estimator call including dispatch, initialization, neural inference when used, EM and filtering. Calibration and final timing run with other experiment jobs idle. Final 20 methods×153 conditions×3 repeats=9,180 measured calls; timing is separate from cached BER decoding.')
    add(json.dumps(timing['cap_verification'],sort_keys=True))
    add('The measured pooled runtime cap was exceeded and is not claimed as satisfied. Read-only audit found similar calibration-to-final p90 increases for the new method (1.749x), direct EM100 (1.748x), and the identical-config compact mix (1.772x), including unchanged learned/EM branches. These windows used different prespecified anchors 100 and 200. Shared slowdown cannot identify the host cause; no normalization, retiming or reselection was performed. Within the final window, prior refinement pooled p90 was 18.7482582 ms versus new 18.788992 ms.')
    add('The calibration cap is a pooled whole-grid p90, not a per-condition limit. All 16 compact sparse/EM candidates were considered for fresh eligibility; the 10 changed reference/SNR bins were refreshed by heldout likelihood, and 41 other choices retained. A slower full sparse estimator is also shown as an unrestricted accuracy comparator. Check below for conditional cap exceedances.')
    cap=timing['cap_verification']['calibrated_cap_ms']
    for name in ('repair_mean','compact_sparse'):
        over={k:v['p90_ms'] for k,v in timing['methods'][name]['by_condition'].items() if v['p90_ms']>cap}
        add(f'{name} conditional p90 above pooled calibration cap: '+json.dumps(over,sort_keys=True))
    add('New compact choices in changed bins:')
    for ref in study.STRONG_RSRS:
        for snr in study.TARGET_SNRS:
            x=policy['compact_choices'][f'{ref:g}'][f'{snr:g}'];add(f"  ref{ref:g} SNR{snr:g}: {x['candidate']}, heldout NLL={x['mean_nll']:.8g}")
    add()
    add('LIMITATIONS')
    add('Conditional simulation evidence under the declared channel, reference and noise model; no proof of uniform superiority, BER gain or extrapolation beyond tested 60dB. Threshold scaling is heuristic. Unknown/mismatched noise, finite-iteration EM local solutions, other path distributions and hardware may change performance. Existing lower-SNR routing transitions remain outside this change. The design was motivated by prior observed failures, but the new confirmatory bank and policy freeze are separate from that evidence.')
    add('The filter tends toward identity as supplied variance decreases; this alone does not establish exact zero-noise numerical validity. The added boundary points are diagnostic, not extra independently weighted confirmation.')
    add()
    add('AUDIT AND REPRODUCIBILITY')
    add('Preflight passed: '+json.dumps({k:checks[k] for k in ('conditions','changed_conditions','exact_current_conditions','exact_earlier_transition_conditions','exact_weak_reference_conditions','maximum_actual_scalar_batch_error')}))
    add('Development guards: '+str(sum(x['passed'] for x in guard['guards']))+'/30 passed.')
    add('Independent audit files: '+', '.join(audits))
    add('Policy SHA256: '+digest);add('Protocol SHA256: '+study.sha(out/'protocol.json'))
    add('BER protocol SHA256: '+study.sha(out/'ber/protocol.json'))
    add('Fresh seeds: '+json.dumps(dict(development=study.DEV_SEED,test=study.TEST_SEED,data=study.DATA_SEED,timing=study.TIMING_SEED,holdout=study.MASK_SEED,bootstrap=study.BOOT_SEED)))
    add('Main data: test/cells; boundary data: boundary/cells; development and timing raw evidence retained in this result directory. Figures: repair_nmse.png, repair_target.png, repair_runtime.png, ber/ber.png, boundary/nmse_boundary.png, boundary/nmse_adjacent_changes_readable.png (reporting-only layout correction; rendering hashes in boundary/plot_rendering.json).')
    path=out/'RESULTS.txt';path.write_text('\n'.join(lines)+'\n');print(path)

if __name__=='__main__':main()
