"""Paired NMSE analysis of the independently tested dual-comparator transition correction."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/private/tmp/tcom_dual_guard_matplotlib'
import json
from pathlib import Path
import numpy as np
from .study import OUT,SNRS,RSRS,REGIMES,SEEDS,NAMES,GROUPS,TEST_COUNT,load,cellname,check_selection
from tcom_followups_20261017.common import sha,write_json

BOOT=20000;BOOT_SEED=2026101939;COUNT=TEST_COUNT;TOL=1e-10
CONTRASTS=dict(repair_vs_old=('old_bypass_mean','repair_mean'),repair_vs_prior=('prior_repair_mean','repair_mean'),repair_vs_original=('original_mean','repair_mean'),
               repair_vs_em100=('em100','repair_mean'),repair_vs_compact=('compact_sparse','repair_mean'))
PRIMARY=('repair_vs_old','repair_vs_prior')
REGIONS=('ordinary','transition','stress','high_snr','all','weak_transition','weak_transition_L2','weak_transition_L4','weak_transition_random3to7',
    'strong_transition','strong_transition_L2','strong_transition_L4','strong_transition_random3to7',
    'strong_transition_snr32.5','strong_transition_snr35','strong_transition_snr37.5',
    'historical_flagged','historical_flagged_L2','historical_flagged_L4','historical_flagged_random3to7')
HISTORICAL=('L2_rsr20_snr32.5','L2_rsr20_snr42.5','L2_rsr20_snr45','L2_rsr20_snr50','L2_rsr20_snr60',
    'L4_rsr12_snr32.5','L4_rsr20_snr32.5','random3to7_rsr12_snr32.5','random3to7_rsr20_snr32.5','random3to7_rsr20_snr37.5')
METHODS=list(NAMES)+list(GROUPS);INDEX={name:i for i,name in enumerate(METHODS)}


def belongs(region,snr,rsr=None,regime=None):
    weak=rsr==5 and 32.5<=snr<=45
    strong=rsr in (12,20) and 32.5<=snr<=37.5
    historical=regime is not None and rsr is not None and cellname(regime,rsr,snr) in HISTORICAL
    if region.startswith('strong_transition_snr'):return strong and snr==float(region[len('strong_transition_snr'):])
    for prefix,member in (('weak_transition_',weak),('strong_transition_',strong),('historical_flagged_',historical)):
        if region.startswith(prefix):return member and regime==region[len(prefix):]
    return dict(ordinary=snr<=20,transition=20<snr<=45,stress=snr>=50,high_snr=snr>20,all=True,
        weak_transition=weak,strong_transition=strong,historical_flagged=historical)[region]


def expand(values):
    return np.concatenate([values,np.stack([values[[INDEX[n] for n in names]].mean(0) for names in GROUPS.values()])])


def intervals(samples,primary=False):
    result=dict(pointwise_ci95_db=np.quantile(samples,[.025,.975]).tolist())
    if primary:result['bonferroni4_interval_db']=np.quantile(samples,[.05/8,1-.05/8]).tolist()
    return result


def run():
    selection,digest=check_selection();rng=np.random.default_rng(BOOT_SEED)
    weights={r:rng.multinomial(COUNT,np.full(COUNT,1/COUNT),size=BOOT).T.astype(float) for r,_ in REGIMES}
    sums={r:np.zeros((len(METHODS),BOOT)) for r in REGIONS}
    rows=[];errors=[];energies=[];points=[];hashes={};dispatch=0.
    for regime,_ in REGIMES:
        for ref in RSRS:
            for snr in SNRS:
                path=OUT/'test/cells'/f'{cellname(regime,ref,snr)}.npz';a=load(path)
                assert str(a['selection_sha256'])==digest
                assert tuple(map(str,a['method_names']))==tuple(NAMES)
                err=np.sum(abs(a['estimates']-a['truth_g'][None])**2,axis=(-2,-1));energy=np.sum(abs(a['truth_g'])**2,axis=(-2,-1))
                np.testing.assert_array_equal(err,a['errors']);np.testing.assert_array_equal(energy,a['truth_energy'])
                assert err.shape==(len(NAMES),COUNT)
                if snr<=20:
                    for seed in SEEDS:
                        for prefix in ('old_bypass','prior_repair','repair'):
                            np.testing.assert_array_equal(a['estimates'][INDEX[f'original_{seed}']],a['estimates'][INDEX[f'{prefix}_{seed}']])
                if ref==5:
                    for seed in SEEDS:np.testing.assert_array_equal(a['estimates'][INDEX[f'prior_repair_{seed}']],a['estimates'][INDEX[f'repair_{seed}']])
                score=expand(10*np.log10(err.sum(1)/energy.sum()))
                boot=expand(10*np.log10((err@weights[regime])/(energy@weights[regime])[None]))
                nm=err/energy[None];p95,p99=(expand(np.quantile(nm,q,axis=1)) for q in (.95,.99));severe=expand((nm>1).mean(1))
                methods={name:dict(nmse_db=float(score[i]),conditional_ci95_db=np.quantile(boot[i],[.025,.975]).tolist(),
                    case_nmse_p95=float(p95[i]),case_nmse_p99=float(p99[i]),severe_rate=float(severe[i])) for name,i in INDEX.items()}
                contrasts={key:dict(gain_db=float(score[INDEX[refname]]-score[INDEX[candidate]]),
                    **intervals(boot[INDEX[refname]]-boot[INDEX[candidate]])) for key,(refname,candidate) in CONTRASTS.items()}
                rows.append(dict(cell=path.stem,regime=regime,rsr=ref,snr=snr,cases=COUNT,methods=methods,contrasts=contrasts))
                points.append(score);errors.append(err);energies.append(energy);hashes[path.name]=sha(path)
                for region in REGIONS:
                    if belongs(region,snr,ref,regime):sums[region]+=boot
                dispatch=max(dispatch,float(a.get('actual_dispatch_max_abs_error',0)))
    points=np.stack(points);aggregates={};boots={}
    for region in REGIONS:
        keep=np.array([belongs(region,r['snr'],r['rsr'],r['regime']) for r in rows]);score=points[keep].mean(0);boot=sums[region]/keep.sum();boots[region]=boot
        err=np.concatenate([e for e,k in zip(errors,keep) if k],axis=1);energy=np.concatenate([e for e,k in zip(energies,keep) if k]);nm=err/energy[None]
        pooled=expand(10*np.log10(err.sum(1)/energy.sum()));p95,p99=(expand(np.quantile(nm,q,axis=1)) for q in (.95,.99));severe=expand((nm>1).mean(1))
        methods={name:dict(nmse_db=float(score[i]),conditional_ci95_db=np.quantile(boot[i],[.025,.975]).tolist(),pooled_ratio_of_sums_db=float(pooled[i]),
            case_nmse_p95=float(p95[i]),case_nmse_p99=float(p99[i]),severe_rate=float(severe[i])) for name,i in INDEX.items()}
        contrasts={}
        for key,(reference,candidate) in CONTRASTS.items():
            gains={}
            for seed in SEEDS:
                source=f'{reference[:-5]}_{seed}' if reference.endswith('_mean') else reference
                gains[str(seed)]=float(score[INDEX[source]]-score[INDEX[f'repair_{seed}']])
            contrasts[key]=dict(gain_db=float(score[INDEX[reference]]-score[INDEX[candidate]]),seed_specific_gains_db=gains,
                **intervals(boot[INDEX[reference]]-boot[INDEX[candidate]],region=='strong_transition' and key in PRIMARY))
        aggregates[region]=dict(cells=int(keep.sum()),condition_cases=int(keep.sum())*COUNT,methods=methods,contrasts=contrasts)
    negative={}
    for key in CONTRASTS:
        worst=min(rows,key=lambda r:r['contrasts'][key]['gain_db'])
        target=[r for r in rows if belongs('strong_transition',r['snr'],r['rsr'],r['regime'])]
        negative[key]=dict(negative_cells=[r['cell'] for r in rows if r['contrasts'][key]['gain_db'] < -TOL],
            pointwise95_harm_cells=[r['cell'] for r in rows if r['contrasts'][key]['pointwise_ci95_db'][1] < -TOL],
            target_negative_cells=[r['cell'] for r in target if r['contrasts'][key]['gain_db'] < -TOL],
            worst_cell=worst['cell'],worst_gain_db=worst['contrasts'][key]['gain_db'])
    result=dict(selection_sha256=digest,protocol_sha256=selection['protocol_sha256'],unique_test_anchors=COUNT*9,condition_cases=len(rows)*COUNT,
        bootstrap_index_clusters=COUNT*3,ordinary_exact_identity=True,weak_reference_exact_identity=True,actual_dispatch_max_abs_error=dispatch,
        scoring='Equal-cell mean dB ratio-of-sums NMSE. Three-fit groups average individual metrics, never estimates. Pooled linear NMSE and tail rates are diagnostics.',
        primary='References12/20 dB and SNR32.5/35/37.5: correction versus old bypass and versus prior repair; NMSE and BER form four prespecified outcomes.',
        bootstrap=dict(samples=BOOT,seed=BOOT_SEED,unit='Whole-channel multinomial local index shared across SNR/reference/method/fit within regime',
            scope='Conditional on frozen models, policies and grid; excludes training-population/selection uncertainty',
            multiplicity='Bonferroni4 on two target-region contrasts times NMSE/BER. Other intervals descriptive pointwise95.'),
        aggregates=aggregates,negative_cells=negative,numerical_tie_tolerance_db=TOL,cells=rows,test_hashes=hashes,source_sha256=sha(__file__))
    write_json(OUT/'nmse_summary.json',result)
    np.savez_compressed(OUT/'nmse_bootstrap.npz',method_names=np.array(METHODS),regions=np.array(REGIONS),aggregate_bootstraps=np.stack([boots[r] for r in REGIONS]),
        cell_names=np.array([r['cell'] for r in rows]),cell_scores=points,selection_sha256=np.array(digest))
    figures(result)
    print(json.dumps(dict(primary=aggregates['strong_transition']['contrasts'],negative_cells=negative)),flush=True)


def figures(result):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'original_mean':('#d73a49','Retrained model','-'),'old_bypass_mean':('#d99322','Previous EM100 bypass','--'),
        'prior_repair_mean':('#7b52ab','Prior repair',':'),'repair_mean':('#0969da','New correction','-'),'em100':('#26965c','Direct EM100','--'),
        'compact_sparse':('#9b6718','Matched sparse/EM family','-.'),'prior_sparse_full':('#777777','Frozen full sparse',':'),
        'optshrink':('#8a63d2','Frozen OptShrink','--')}
    fig,axes=plt.subplots(3,3,figsize=(13.8,10.6),sharex=True,sharey=True)
    for i,(regime,_) in enumerate(REGIMES):
        for j,ref in enumerate(RSRS):
            ax=axes[i,j];rr=sorted([r for r in result['cells'] if r['regime']==regime and r['rsr']==ref],key=lambda r:r['snr'])
            for name,(color,label,ls) in styles.items():ax.plot([r['snr'] for r in rr],[r['methods'][name]['nmse_db'] for r in rr],color=color,label=label,ls=ls,marker='.',lw=1.5)
            if ref in (12,20):ax.axvspan(32.5,37.5,color='#0969da',alpha=.06)
            ax.set_title(f'{regime}, reference {ref:g} dB');ax.grid(alpha=.2)
            if i==2:ax.set_xlabel('Nominal SNR (dB)')
            if j==0:ax.set_ylabel('Channel NMSE (dB; lower is better)')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,fontsize=8)
    fig.suptitle('Fresh test: short EM refinement with two development comparators\nShading marks the predeclared strong-reference transition target')
    fig.tight_layout(rect=(0,.065,1,.95));fig.savefig(OUT/'repair_nmse.png',dpi=180);fig.savefig(OUT/'repair_nmse.svg');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(11.5,10),sharex=True)
    for i,(regime,_) in enumerate(REGIMES):
        for j,ref in enumerate((12,20)):
            ax=axes[i,j]
            rr=sorted([r for r in result['cells'] if r['regime']==regime and r['rsr']==ref and 32.5<=r['snr']<=37.5],key=lambda r:r['snr'])
            for key,label,color in [('repair_vs_old','Gain over old bypass','#d99322'),('repair_vs_prior','Gain over prior repair','#0969da')]:
                data=[r['contrasts'][key] for r in rr];x=[r['snr'] for r in rr]
                ax.plot(x,[r['gain_db'] for r in data],marker='.',label=label,color=color)
                ax.fill_between(x,[r['pointwise_ci95_db'][0] for r in data],[r['pointwise_ci95_db'][1] for r in data],alpha=.12,color=color)
            ax.axhline(0,color='black',lw=.8);ax.set_title(f'{regime}, reference {ref} dB');ax.grid(alpha=.2)
            if i==2:ax.set_xlabel('Nominal SNR (dB)')
            if j==0:ax.set_ylabel('NMSE gain (dB; positive is better)')
    axes[0,1].legend(fontsize=8)
    fig.suptitle('Strong-reference transition: every predeclared condition shown\nBands are descriptive pointwise conditional 95% intervals')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/'repair_target.png',dpi=180);fig.savefig(OUT/'repair_target.svg');plt.close(fig)


def attach_timing():
    _,digest=check_selection();path=OUT/'nmse_summary.json';summary=json.loads(path.read_text());timing=json.loads((OUT/'timing.json').read_text())
    assert summary['selection_sha256']==timing['selection_sha256']==digest
    summary.update(timing=timing['methods'],timing_sha256=sha(OUT/'timing.json'),timing_analysis_source_sha256=sha(__file__));write_json(path,summary)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    samples=dict(timing['raw_samples']);samples.update({n:[r for m in members for r in samples[m]] for n,members in GROUPS.items()})
    methods=[('original_mean','Retrained'),('old_bypass_mean','Previous bypass'),('prior_repair_mean','Prior repair'),('repair_mean','New correction'),('em100','Direct EM100'),
        ('prior_compact','Previous compact sparse'),('compact_sparse','Matched sparse/EM family'),('prior_sparse_full','Frozen full sparse'),('optshrink','Frozen OptShrink')]
    fig,axes=plt.subplots(1,3,figsize=(13.5,5.2),sharey=True)
    for ax,region,title in zip(axes,('strong_transition','weak_transition','high_snr'),('Strong-reference transition target','Preserved weak-reference target','All SNR >20 dB')):
        values=[np.quantile([r['ms'] for r in samples[n] if belongs(region,r['snr'],r['rsr'],r['regime'])],.9) for n,_ in methods]
        bars=ax.barh(np.arange(len(methods)),values,color=['#0969da' if n=='repair_mean' else '#d98324' if n=='compact_sparse' else '#8c9dab' for n,_ in methods])
        for bar,value in zip(bars,values):ax.text(value*1.06,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center',fontsize=8)
        ax.set_xscale('log');ax.set_xlim(.1,max(values)*3);ax.set_title(title);ax.set_xlabel('Batch-one p90 latency (ms; log scale)');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(np.arange(len(methods)),[x[1] for x in methods]);axes[0].invert_yaxis()
    fig.suptitle('Fresh isolated runtime on one CPU thread\nLookup, initialization, learned inference and any EM continuation included')
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/'repair_runtime.png',dpi=180);fig.savefig(OUT/'repair_runtime.svg');plt.close(fig)
    print('Timing attached; runtime figure saved',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--timing-only',action='store_true');args=p.parse_args()
    attach_timing() if args.timing_only else run()
