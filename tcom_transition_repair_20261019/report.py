"""Build a reviewable experiment report; never edits manuscript files."""
import json
import numpy as np
from .analysis import belongs,REGIONS,CONTRASTS,PRIMARY
from .study import OUT,SEEDS,GROUPS,sha

LABELS=[('original_mean','Retrained model'),('old_bypass_mean','Previous spectral-EM100 bypass'),
    ('repair_mean','New transition repair'),('em100','Direct spectral EM100'),('prior_compact','Frozen previous compact sparse'),
    ('compact_sparse','Matched sparse/EM family'),('prior_sparse_full','Frozen full sparse'),('optshrink','Frozen OptShrink')]

def read(name):return json.loads((OUT/name).read_text())
def interval(x,scale=1,digits=3):return '['+', '.join(f'{v*scale:.{digits}f}' for v in x)+']'
def table(headers,rows):
    cells=[[str(x) for x in row] for row in [headers]+rows];widths=[max(len(r[i]) for r in cells) for i in range(len(headers))]
    fmt=lambda r:'  '.join(x.ljust(w) for x,w in zip(r,widths))
    return '\n'.join([fmt(cells[0]),fmt(['-'*w for w in widths])]+[fmt(r) for r in cells[1:]])

def main():
    n,b,t,p=map(read,['nmse_summary.json','ber/summary.json','timing.json','policy.json'])
    audits=['checks.json','audit/calibration.json','audit/repair_selection.json','audit/selection.json','audit/test.json','audit/summary.json','audit/timing.json','ber/audit.json','sensitivity/audit.json','audit/sensitivity.json']
    for name in audits:assert read(name)['passed'],name
    digest=sha(OUT/'policy.json');assert all(x['selection_sha256']==digest for x in (n,b,t));assert n['timing_sha256']==sha(OUT/'timing.json')
    for name in ('audit/selection.json','audit/test.json','audit/summary.json','audit/timing.json','ber/audit.json','audit/sensitivity.json'):
        assert read(name)['selection_sha256']==digest,name
    for audit_name,key,artifact in (
        ('audit/repair_selection.json','repair_selection_sha256','repair_selection.json'),
        ('audit/calibration.json','calibration_sha256','calibration.json'),
        ('audit/summary.json','summary_sha256','nmse_summary.json'),
        ('audit/summary.json','bootstrap_sha256','nmse_bootstrap.npz'),
        ('audit/timing.json','timing_sha256','timing.json'),
        ('audit/sensitivity.json','sensitivity_summary_sha256','sensitivity/summary.json'),
        ('audit/sensitivity.json','sensitivity_primary_audit_sha256','sensitivity/audit.json')):
        assert read(audit_name)[key]==sha(OUT/artifact),(audit_name,artifact)
    times=dict(t['raw_samples']);times.update({name:[r for member in members for r in times[member]] for name,members in GROUPS.items()})
    positive_primary=sum(n['aggregates']['weak_transition']['contrasts'][key]['bonferroni4_interval_db'][0]>0 for key in PRIMARY)+sum(b['regions']['weak_transition']['contrasts'][key]['bonferroni4_interval'][0]>0 for key in PRIMARY)
    identity_cells=3*sum(c['choice']=='original' for cells in p['repair_choices'].values() for c in cells.values())
    target_p90={name:float(np.quantile([r['ms'] for r in times[name] if belongs('weak_transition',r['snr'],r['rsr'],r['regime'])],.9)) for name,_ in LABELS}
    cap=t['cap_verification']['calibrated_cap_ms'];overruns={}
    for name in ('repair_mean','compact_sparse'):
        overruns[name]=[]
        for ref,snr in sorted({(r['rsr'],r['snr']) for r in times[name]}):
            value=float(np.quantile([r['ms'] for r in times[name] if r['rsr']==ref and r['snr']==snr],.9))
            if value>cap:overruns[name].append((ref,snr,value))
    lines=['WEAK-REFERENCE TRANSITION REPAIR: FRESH NMSE, BER AND RUNTIME COMPARISON',
        'Completed experiment. Manuscript and all earlier study outputs remain unchanged.','',
        'KEY FINDINGS',
        f"In the predeclared weak-reference transition target, repair changes mean NMSE by {n['aggregates']['weak_transition']['contrasts']['repair_vs_old']['gain_db']:+.3f} dB relative to the previous bypass and {n['aggregates']['weak_transition']['contrasts']['repair_vs_original']['gain_db']:+.3f} dB relative to the retrained model (positive is improvement).",
        f"Observed target BER: previous bypass {100*b['regions']['weak_transition']['methods']['old_bypass_mean']['ber']:.6f}%, repair {100*b['regions']['weak_transition']['methods']['repair_mean']['ber']:.6f}%. The adjusted intervals below determine whether the differences are resolved by this finite sample.",
        f"{positive_primary}/4 prespecified primary NMSE/BER comparisons have strictly positive family-adjusted improvement intervals. This inference is conditional on the fixed models, policies and test grid.",
        f"Target cells with worse mean NMSE: {len(n['negative_cells']['repair_vs_old']['target_negative_cells'])} versus previous bypass, {len(n['negative_cells']['repair_vs_original']['target_negative_cells'])} versus retrained model. All three path-regime target results and every adverse cell follow.",
        f"Across the full grid, {len(n['negative_cells']['repair_vs_old']['negative_cells'])} conditions have worse mean NMSE than the old bypass, including {len(n['negative_cells']['repair_vs_old']['pointwise95_harm_cells'])} with pointwise 95% intervals supporting harm. The worst observed loss is {-n['negative_cells']['repair_vs_old']['worst_gain_db']:.3f} dB. The repair is not uniformly better than that policy.",
        f"The rule retains the original estimate in {identity_cells}/153 conditions and changes it in {153-identity_cells}. Comparing against the original therefore includes many exact identities.",
        f"Overall observed BER remains lower for the matched sparse/EM family ({100*b['regions']['all']['methods']['compact_sparse']['ber']:.6f}%) than for the repair ({100*b['regions']['all']['methods']['repair_mean']['ber']:.6f}%); the repair preserves the original estimator's low-SNR behavior.",
        f"Primary-window p90 latency is {target_p90['repair_mean']:.3f} ms for repair versus {target_p90['old_bypass_mean']:.3f} ms for the old bypass ({target_p90['repair_mean']/target_p90['old_bypass_mean']:.2f} times as long). This target-window cost must not be hidden by the overall median.",
        f"Final all-grid p90 latency: repair {t['methods']['repair_mean']['p90_ms']:.3f} ms; matched sparse/EM family {t['methods']['compact_sparse']['p90_ms']:.3f} ms. Original outputs remain exactly unchanged through nominal 20 dB.",
        '', 'DESIGN AND INTERPRETATION',
        'The three existing30-pilot retrained checkpoints remain frozen. At nominal SNR<=20 dB the original estimator is returned exactly. Above20 dB a development-selected lookup uses supplied noise variance and known reference power to choose the original estimator,5/20/100 exact EM steps initialized by its output, or spectral-initialized EM100.',
        'Every deployed decision uses receiver-observable quantities. True channels and path counts are absent from inference. A path regime is used only as a calibration stratum and a test reporting label.',
        'The routing proxy is 10 log10(K/sigma^2), with K=3 and the declared unit-user-power normalization. Nearest declared SNR/reference bins determine the lookup; this is not an estimate of realized channel SNR.',
        'Initialization is the hypothesis under test: retaining the learned channel as the EM starting point may avoid the poor solutions reached by the previous fresh spectral fallback. Longer continuation is not assumed universally better.',
        'The tested repair combines learned initialization and guarded routing. This comparison does not isolate their separate causal contributions, and warm100 and spectral100 have different initialization costs.',
        'An exploratory screen used only an already available DEVELOPMENT observation subset to define this small menu. New calibration and new test banks are independent of that screen and the exposed previous test bank.',
        'Fresh development uses96 channel/reference anchors per pair, with30 fitting pilots and9 held-out pilots. Each nonidentity candidate must have a one-sided95% paired-bootstrap upper bound on held-out NLL change<=0 in EACH of three path regimes. Original is always admissible; constraints are never relaxed. Eligible choices minimize mean held-out NLL, using the frozen preference order for ties.',
        'This is a conservative development screen, not a guarantee of NMSE/BER safety. Multiple selection decisions, finite data and metric mismatch remain. No test outcomes change the rule.',
        'The primary target is reference5 dB and SNR32.5–45 dB:18 conditions spanning2,4,and random3–7 paths. Two comparisons, repair versus previous bypass and versus retrained model, times NMSE andBER form four primary outcomes.',
        'Separate path-regime target scores, every adverse condition, tails and all-grid results are reported so a pooled improvement cannot hide the original weakness.',
        'Testing uses128 fresh channel/reference anchors per pair:1,152 unique channel/reference draws, reused at17 SNRs to give19,584 operating cases. Within each path regime, shared local indices preserve coupling across reference/SNR/method/fit in20,000 bootstrap resamples.',
        'Three-fit groups average individual estimator metrics; they are not ensembles and do not triple independent channels. NMSE is equal-cell mean dB ratio-of-sums. Pooled NMSE is shown separately because aggregation can change rankings.',
        'BER uses every64-vector joint GrayQPSK symbol once per channel, fresh data noise shared across estimators, and the same exact64-hypothesis magnitude-domain Rician detector. There are7,520,256 bits per estimator/fit across the full grid and884,736 in the primary target. Zero observed errors do not prove zero BER.',
        '', 'PRIMARY TARGET RESULTS','']
    target=n['aggregates']['weak_transition'];bt=b['regions']['weak_transition']
    lines +=['Target bit errors by fitted model (seed order '+', '.join(map(str,SEEDS))+'); each fit uses the same 884,736 target bits:',
        *[f"  {group}: "+str([round(bt['methods'][f'{group}_{seed}']['ber']*884736) for seed in SEEDS]) for group in ('original','old_bypass','repair')],
        'BER improvement versus the original can be concentrated in particular fitted models. Pooled target intervals do not establish improvement in every path subgroup or every fit. Zero-error empirical intervals can collapse to [0,0] and provide no rare-event guarantee.','']
    for key in PRIMARY:
        nr=target['contrasts'][key];br=bt['contrasts'][key]
        lines += [key+':',f"  NMSE gain {nr['gain_db']:.4f} dB; family-adjusted interval {interval(nr['bonferroni4_interval_db'],digits=4)} dB.",
            f"  BER reduction {100*br['ber_reduction']:.6f} percentage points; family-adjusted interval {interval(br['bonferroni4_interval'],100,6)}."]
    lines+=['Primary intervals are two-sided 98.75% bootstrap percentile intervals, adjusted across four prespecified outcomes for nominal 95% family coverage.',
        'Intervals condition on the fixed checkpoints, selected policy and declared grid. They exclude training-population and selection uncertainty. Other intervals below are descriptive pointwise95%.', '', 'COMPARISON BY OPERATING RANGE',
        'Ordinary=0–20 dB; transition=22.5–45 dB; stress=50/60 dB; high-SNR=all tested points above20 dB. Reference levels5/12/20 dB;2/4/random3–7 paths per user.',
        'All columns lower is better. Latency is batch-one on one Apple M2 CPU thread, including observable lookup, initialization, learned inference and all selected EM updates. Models and fixed lookup tables are preloaded.','']
    for region in ('weak_transition','weak_transition_L2','weak_transition_L4','weak_transition_random3to7','ordinary','transition','stress','high_snr','all'):
        nr=n['aggregates'][region];br=b['regions'][region];rows=[]
        for name,label in LABELS:
            vals=[r['ms'] for r in times[name] if belongs(region,r['snr'],r['rsr'],r['regime'])]
            rows.append([label,f"{nr['methods'][name]['nmse_db']:.3f}",f"{nr['methods'][name]['pooled_ratio_of_sums_db']:.3f}",
                f"{100*br['methods'][name]['ber']:.6f}",f'{np.quantile(vals,.9):.3f}',f'{np.median(vals):.3f}'])
        lines +=[region.upper()+f" ({nr['cells']} conditions)",table(['Estimator','Equal-cell NMSE dB','Pooled NMSE dB','Observed BER %','p90 ms','Median ms'],rows),'']
    lines+=['ADVERSE CONDITIONS AND FIT VARIATION','Negative gain means repair is worse than its comparator. A numerical tie tolerance of1e-10 dB is used only for reporting identical estimates.','']
    for key,row in n['negative_cells'].items():
        worst=next(c for c in n['cells'] if c['cell']==row['worst_cell']);effect=worst['contrasts'][key]
        lines +=[key+':',f"  Negative means: {len(row['negative_cells'])} overall, {len(row['target_negative_cells'])} in target; pointwise95% harm intervals: {len(row['pointwise95_harm_cells'])} overall.",
            f"  Worst observed gain: {row['worst_gain_db']:.3f} dB in {row['worst_cell']}; interval {interval(effect['pointwise_ci95_db'])}.",
            '  All negative cells: '+(', '.join(row['negative_cells']) or 'none'),
            '  Target fit-specific NMSE gains: '+str(target['contrasts'][key]['seed_specific_gains_db'])]
    lines+=['','EVERY WEAK-REFERENCE TARGET CONDITION','']
    rows=[]
    for cell in n['cells']:
        if not belongs('weak_transition',cell['snr'],cell['rsr'],cell['regime']):continue
        rows.append([cell['regime'],cell['snr'],*[f"{cell['methods'][name]['nmse_db']:.3f}" for name in ('original_mean','old_bypass_mean','repair_mean','em100','compact_sparse')],
            interval(cell['contrasts']['repair_vs_original']['pointwise_ci95_db'])])
    lines +=[table(['Paths','SNR dB','Retrained','Previous bypass','Repair','EM100','Matched sparse/EM','Repair gain over original CI95'],rows),'',
        'TAIL DIAGNOSTICS IN TARGET','p95/p99 are quantiles of per-channel linear NMSE, averaged across fitted models. Severe means NMSE>1.','']
    lines +=[table(['Estimator','p95 case NMSE','p99 case NMSE','Severe %'],[[label,f"{target['methods'][name]['case_nmse_p95']:.6g}",
        f"{target['methods'][name]['case_nmse_p99']:.6g}",f"{100*target['methods'][name]['severe_rate']:.4f}"] for name,label in LABELS]),'']
    lines +=['RUNTIME AND BASELINE FAIRNESS',
        'The new repair rule is selected before isolated timing calibration. The compact sparse cap is the measured p90 of that actual rule; eligible sparse configurations are then selected on the same fresh development observations before any tests.',
        'That cap pools all 153 conditions and three fixed fits. It is neither a per-condition latency match nor a worst-branch runtime guarantee.',
        'All16 declared compact candidates passed calibration eligibility. The compact menu is a limited family, not a global sparse-optimization claim. The previous compact sparse table, full sparse and OptShrink remain frozen transfer controls. Their earlier lookup calibration is not repeated for the new study.',
        'Some strong-reference warm100/spectral100 development losses differ by only about1e-8 to1e-6. These are near-equivalent finite-development choices, well above the measured arithmetic discrepancy but not evidence of a meaningful deployment advantage. The rule is not optimized for minimum runtime within a practical-equivalence tolerance.',
        f"The fresh matched family selects exact zero-penalty EM100 in {sum(c['candidate']=='em100' for cells in p['compact_choices'].values() for c in cells.values())}/51 lookup bins, including {sum(p['compact_choices']['5'][snr]['candidate']=='em100' for snr in ('32.5','35','37.5','40','42.5','45'))}/6 weak-reference target bins. In those cells its prediction equals direct EM100; it is a selected sparse/EM family, not an independent sparse solution.",
        'Fresh final timing runs separately after accuracy work, with random method order and3 repetitions per condition. All raw calls and outliers are retained. Calibration caps are measured quantiles, not hard deadlines; final cap violations are reported without reselection.',
        f"The primary-window repair p90 is {100*(target_p90['repair_mean']/cap-1):+.2f}% relative to the pooled calibration cap. {len(overruns['repair_mean'])}/51 repair condition-specific p90 values exceed that pooled cap; the matched sparse/EM family has {len(overruns['compact_sparse'])}/51. Each condition pools paths and fixed fits. These values show why pooled eligibility is not a per-condition runtime guarantee.",
        table(['Reference dB','SNR dB','Repair conditional p90 ms','Above pooled cap %'],[[ref,snr,f'{value:.4f}',f'{100*(value/cap-1):.2f}'] for ref,snr,value in overruns['repair_mean']]),
        json.dumps(t.get('cap_verification',{}),indent=2),'', 'FROZEN POLICY',table(['Nominal SNR dB','Reference5 dB','Reference12 dB','Reference20 dB'],[[snr]+[p['repair_choices'][ref][snr]['choice'] for ref in ('5','12','20')] for snr in p['repair_choices']['5']]),'',
        'SUPPLIED-NOISE SENSITIVITY',
        'Fixed physical observations at reference5/SNR35,40,45 and reference12/20/SNR32.5, all three path regimes, with supplied-variance offsets -3/0/+3 dB. The supplied variance changes both inference and routing. No sensitivity BER or retuning follows this diagnostic.',
        'Positive noise-mismatch loss means the repair is worse than with correctly supplied variance. Intervals below are descriptive pointwise95%.',
        'Each mismatch loss compares the same repair under incorrect versus correct variance. It is not a repair-versus-original effect; that separate comparison supplies both estimators the same incorrect variance.',
        table(['Paths','Reference dB','Actual SNR dB','Variance error dB','Selected branch','Repair NMSE dB','Mismatch loss dB','Loss CI95'],
            [[c['regime'],c['rsr'],c['actual_snr_db'],c['supplied_variance_offset_db'],c['repair_choice'],
              f"{c['methods']['repair_mean']['nmse_db']:.3f}",f"{c['methods']['repair_mean']['degradation_vs_correct_variance_db']:.3f}",
              interval(c['methods']['repair_mean']['degradation_bootstrap95'])] for c in read('sensitivity/summary.json')['cells']]),'',
        'LIMITS AND SCIENTIFIC ATTRIBUTION',
        'Any gains from a learned-initialized EM tail belong to the combined estimator. They do not establish a new learned controller, a global convergence result or uniform baseline superiority.',
        'The guarantee of exact original behavior at or below20 dB refers to the supplied-noise SNR proxy. Misspecified noise can move an observation to a different lookup entry. Nearest-grid/reference lookup outside the declared operating grid is not validated.',
        'This follow-up is independent confirmation of a repair motivated by exposed earlier failures. Previous test scores are not reused as independent evidence for the new claim.',
        'The manuscript has not been changed. Interpret the target, subgroup, BER and latency results together before promoting the repair.', '',
        'VERIFICATION','All required audits passed: '+', '.join(audits),
        'Frozen policy SHA256: '+digest,'','ARTIFACTS',
        'nmse_summary.json; nmse_bootstrap.npz; repair_nmse.png; repair_target.png; repair_runtime.png',
        'ber/summary.json; ber/ber.png; timing.json; sensitivity/summary.json; policy.json; protocol.json','']
    text='\n'.join(lines)
    for old,new in [('existing30','existing 30'),('Above20','Above 20'),('estimator,5/20/100','estimator, 5/20/100'),('uses96','uses 96'),('with30','with 30'),('and9','and 9'),('one-sided95%','one-sided 95%'),('reference5','reference 5'),('SNR32.5','SNR 32.5'),(':18',': 18'),('spanning2,4,and','spanning 2, 4, and'),('andBER','and BER'),('uses128','uses 128'),(':1,152',': 1,152'),('at17','at 17'),('give19,584','give 19,584'),('in20,000','in 20,000'),('every64','every 64'),('exact64','exact 64'),('are7,520,256','are 7,520,256'),('and884,736','and 884,736'),('All16','All 16'),('about1e-8 to1e-6','about 1e-8 to 1e-6'),('pointwise95%','pointwise 95%'),('above20','above 20'),('levels5/12/20','levels 5/12/20'),(';2/4/','; 2/4/'),('of1e-10','of 1e-10'),('declared16','declared 16'),('and3 repetitions','and 3 repetitions'),('SNR35,40,45','SNR 35, 40, 45'),('below20','below 20')]:text=text.replace(old,new)
    (OUT/'RESULTS.txt').write_text(text);print(OUT/'RESULTS.txt')

if __name__=='__main__':main()
