"""Frozen paired scoring for the observable-validation-selected hybrid."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/private/tmp/tcom_hybrid_matplotlib'
import json
import numpy as np
from .study import OUT,SNRS,RSRS,REGIMES,SEEDS,NAMES,GROUPS,RAW_NAMES,TEST_COUNT,BOOT_SEED,load,cellname,check_selection
from tcom_followups_20261017.common import sha,write_json

BOOT=20000;COUNT=TEST_COUNT;TOL=1e-10
REGIONS=('all','high_all','transition_all','weak_warm100','strong_high','strong_transition','ordinary','weak','rsr5','rsr12','rsr20',
    'all_L2','all_L4','all_random3to7','weak_warm100_L2','weak_warm100_L4','weak_warm100_random3to7')
PRIMARY=('hybrid_vs_latest','hybrid_vs_compact')
CONTRASTS=dict(hybrid_vs_latest=('latest_mean','hybrid_mean'),hybrid_vs_compact=('compact_sparse','hybrid_mean'),
    hybrid_vs_adaptive40=('adaptive40_mean','hybrid_mean'),hybrid_vs_global40=('global40_mean','hybrid_mean'),
    hybrid_vs_classical40=('classical40','hybrid_mean'),hybrid_vs_em100=('em100','hybrid_mean'),
    hybrid_vs_optshrink=('optshrink','hybrid_mean'),hybrid_vs_noem=('hybrid0_mean','hybrid_mean'),
    adaptive_vs_global=('global_mean','adaptive_mean'))


def belongs(region,snr,rsr=None,regime=None):
    scopes=dict(all=True,high_all=snr>=40,transition_all=20<snr<40,weak_warm100=rsr==5 and snr>=37.5,strong_high=rsr in (12,20) and snr>=40,
        strong_transition=rsr in (12,20) and 32.5<=snr<=37.5,ordinary=snr<=20,weak=rsr==5)
    if region in scopes:return scopes[region]
    if region.startswith('rsr'):return rsr==float(region[3:])
    for prefix in ('all_','weak_warm100_'):
        if region.startswith(prefix):return scopes[prefix[:-1]] and regime==region[len(prefix):]
    raise KeyError(region)


def intervals(samples,primary=False):
    result=dict(pointwise_ci95_db=np.quantile(samples,[.025,.975]).tolist())
    if primary:result['bonferroni4_interval_db']=np.quantile(samples,[.05/8,1-.05/8]).tolist()
    return result


def score_dataset(selection,digest,raw=False):
    names=tuple(RAW_NAMES if raw else NAMES)
    groups={name:members for name,members in GROUPS.items() if all(n in names for n in members)}
    methods=list(names)+list(groups);index={name:i for i,name in enumerate(methods)}
    contrasts={name:pair for name,pair in CONTRASTS.items() if all(n in methods for n in pair)}
    def expand(values):
        return np.concatenate([values,np.stack([values[[index[n] for n in members]].mean(0) for members in groups.values()])])
    rng=np.random.default_rng(BOOT_SEED)
    weights={r:rng.multinomial(COUNT,np.full(COUNT,1/COUNT),size=BOOT).T.astype(float) for r,_ in REGIMES}
    sums={r:np.zeros((len(methods),BOOT)) for r in REGIONS}
    rows=[];errors=[];energies=[];points=[];hashes={}
    estimate_key,error_key,name_key=('raw_estimates','raw_errors','raw_names') if raw else ('estimates','errors','method_names')
    for regime,_ in REGIMES:
        for ref in RSRS:
            for snr in SNRS:
                path=OUT/'test/cells'/f'{cellname(regime,ref,snr)}.npz';a=load(path)
                assert str(a['selection_sha256'])==digest and str(a['protocol_sha256'])==selection['protocol_sha256']
                assert tuple(map(str,a[name_key]))==names
                err=np.sum(abs(a[estimate_key]-a['truth_g'][None])**2,axis=(-2,-1));energy=np.sum(abs(a['truth_g'])**2,axis=(-2,-1))
                np.testing.assert_array_equal(err,a[error_key]);np.testing.assert_array_equal(energy,a['truth_energy'])
                assert err.shape==(len(names),COUNT) and np.isfinite(err).all() and (energy>0).all()
                point=expand(10*np.log10(err.sum(1)/energy.sum()))
                boot=expand(10*np.log10((err@weights[regime])/(energy@weights[regime])[None]))
                nm=err/energy[None];p95,p99=(expand(np.quantile(nm,q,axis=1)) for q in (.95,.99));severe=expand((nm>1).mean(1))
                values={name:dict(nmse_db=float(point[i]),conditional_ci95_db=np.quantile(boot[i],[.025,.975]).tolist(),
                    case_nmse_p95=float(p95[i]),case_nmse_p99=float(p99[i]),severe_rate=float(severe[i])) for name,i in index.items()}
                comparisons={key:dict(gain_db=float(point[index[refname]]-point[index[candidate]]),
                    **intervals(boot[index[refname]]-boot[index[candidate]])) for key,(refname,candidate) in contrasts.items()}
                rows.append(dict(cell=path.stem,regime=regime,rsr=ref,snr=snr,cases=COUNT,methods=values,contrasts=comparisons))
                points.append(point);errors.append(err);energies.append(energy);hashes[path.name]=sha(path)
                for region in REGIONS:
                    if belongs(region,snr,ref,regime):sums[region]+=boot
    assert len(rows)==153
    points=np.stack(points);aggregates={};boots={}
    for region in REGIONS:
        keep=np.array([belongs(region,r['snr'],r['rsr'],r['regime']) for r in rows]);point=points[keep].mean(0);boot=sums[region]/keep.sum();boots[region]=boot
        err=np.concatenate([e for e,k in zip(errors,keep) if k],axis=1);energy=np.concatenate([e for e,k in zip(energies,keep) if k]);nm=err/energy[None]
        pooled_linear=expand(err.sum(1)/energy.sum());pooled=10*np.log10(pooled_linear);p95,p99=(expand(np.quantile(nm,q,axis=1)) for q in (.95,.99));severe=expand((nm>1).mean(1))
        values={name:dict(nmse_db=float(point[i]),conditional_ci95_db=np.quantile(boot[i],[.025,.975]).tolist(),
            pooled_ratio_of_sums_linear=float(pooled_linear[i]),pooled_ratio_of_sums_db=float(pooled[i]),case_nmse_p95=float(p95[i]),case_nmse_p99=float(p99[i]),severe_rate=float(severe[i])) for name,i in index.items()}
        comparisons={}
        for key,(reference,candidate) in contrasts.items():
            gains={}
            for seed in SEEDS:
                left=f'{reference[:-5]}_{seed}' if reference.endswith('_mean') else reference
                right=f'{candidate[:-5]}_{seed}' if candidate.endswith('_mean') else candidate
                gains[str(seed)]=float(point[index[left]]-point[index[right]])
            comparisons[key]=dict(gain_db=float(point[index[reference]]-point[index[candidate]]),seed_specific_gains_db=gains,
                **intervals(boot[index[reference]]-boot[index[candidate]],not raw and region=='all' and key in PRIMARY))
        aggregates[region]=dict(cells=int(keep.sum()),condition_cases=int(keep.sum())*COUNT,methods=values,contrasts=comparisons)
    negative={}
    for key in contrasts:
        worst=min(rows,key=lambda r:r['contrasts'][key]['gain_db'])
        negative[key]=dict(negative_cells=[r['cell'] for r in rows if r['contrasts'][key]['gain_db'] < -TOL],
            pointwise95_harm_cells=[r['cell'] for r in rows if r['contrasts'][key]['pointwise_ci95_db'][1] < -TOL],
            weak_warm100_negative_cells=[r['cell'] for r in rows if belongs('weak_warm100',r['snr'],r['rsr'],r['regime']) and r['contrasts'][key]['gain_db'] < -TOL],
            worst_cell=worst['cell'],worst_gain_db=worst['contrasts'][key]['gain_db'])
    result=dict(selection_sha256=digest,protocol_sha256=selection['protocol_sha256'],raw_start_diagnostic=raw,
        method_names=methods,groups=groups,contrasts=contrasts,unique_test_anchors=COUNT*9,condition_cases=153*COUNT,bootstrap_index_clusters=COUNT*3,
        scoring='Equal-cell mean dB ratio-of-sums NMSE. Groups average metrics across fits, never channel estimates. Pooled linear NMSE averages fit-specific error/energy ratios arithmetically; pooled dB is its10log10. Pooled and case-tail metrics are diagnostic.',
        primary=('None: raw starts are descriptive diagnostics.' if raw else 'Whole-grid hybrid versus frozen latest and compact. NMSE and BER form four prespecified outcomes. Fixed-start, zero-EM and subgroup comparisons are descriptive.'),
        bootstrap=dict(samples=BOOT,seed=BOOT_SEED,unit='Whole-channel multinomial local index shared across SNR/reference/method/fit within each regime',
            scope='Conditional on frozen trained weights, calibrated controls and grid; excludes training-population and selection uncertainty.',
            multiplicity='Bonferroni4 only for the two hybrid primary contrasts in all153, with their two BER outcomes. All other intervals descriptive pointwise95.'),
        aggregates=aggregates,negative_cells=negative,numerical_tie_tolerance_db=TOL,cells=rows,test_hashes=hashes,source_sha256=sha(__file__))
    stem='raw_start' if raw else 'nmse'
    write_json(OUT/f'{stem}_summary.json',result)
    np.savez_compressed(OUT/f'{stem}_bootstrap.npz',method_names=np.array(methods),regions=np.array(REGIONS),
        aggregate_bootstraps=np.stack([boots[r] for r in REGIONS]),cell_names=np.array([r['cell'] for r in rows]),cell_scores=points,selection_sha256=np.array(digest))
    return result


def run():
    selection,digest=check_selection()
    raw=score_dataset(selection,digest,raw=True);result=score_dataset(selection,digest)
    result['raw_start_summary_sha256']=sha(OUT/'raw_start_summary.json')
    result['raw_start_bootstrap_sha256']=sha(OUT/'raw_start_bootstrap.npz')
    write_json(OUT/'nmse_summary.json',result);figures(result,raw)
    print(json.dumps(dict(primary={key:result['aggregates']['all']['contrasts'][key] for key in PRIMARY},negative_cells=result['negative_cells'])),flush=True)


def figures(result,raw):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main_styles={'hybrid_mean':('#0969da','Selected hybrid','-'),'hybrid0_mean':('#bf3989','Hybrid, zero EM','--'),
        'latest_mean':('#8a63d2','Frozen latest',':'),'adaptive40_mean':('#26965c','Adaptive + 40 EM','-.'),
        'global40_mean':('#d99322','Global + 40 EM','--'),'classical40':('#777777','Classical + 40 EM',':'),
        'compact_sparse':('#795548','Frozen compact','-.'),'optshrink':('#d73a49','Frozen OptShrink','--')}
    raw_styles={'adaptive_mean':('#0969da','Raw adaptive','-'),'global_mean':('#bf3989','Raw global','--')}
    for data,styles,stem,title in ((result,main_styles,'hybrid_nmse','Selected hybrid and frozen comparison pipelines'),
            (raw,raw_styles,'raw_start_nmse','Raw starts before any continuation or final filter')):
        fig,axes=plt.subplots(3,3,figsize=(13.8,10.6),sharex=True,sharey=True)
        for i,(regime,_) in enumerate(REGIMES):
            for j,ref in enumerate(RSRS):
                ax=axes[i,j];rr=sorted([r for r in data['cells'] if r['regime']==regime and r['rsr']==ref],key=lambda r:r['snr'])
                for name,(color,label,ls) in styles.items():
                    ax.plot([r['snr'] for r in rr],[r['methods'][name]['nmse_db'] for r in rr],color=color,label=label,ls=ls,marker='.',lw=1.5)
                ax.set_title(f'{regime}, reference {ref:g} dB',fontsize=11);ax.grid(alpha=.2);ax.set_xticks((0,10,20,30,40,50,60))
                if i==2:ax.set_xlabel('Nominal SNR (dB)')
                if j==0:ax.set_ylabel('Channel NMSE (dB; lower is better)')
        h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.012),ncol=3,fontsize=8)
        fig.suptitle(title+'\nFresh paired test; three-fit metrics averaged, not estimates',y=.97)
        fig.subplots_adjust(left=.08,right=.985,bottom=.14,top=.895,hspace=.25,wspace=.13)
        fig.savefig(OUT/f'{stem}.png',dpi=180);fig.savefig(OUT/f'{stem}.svg');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13.8,10),sharex=True)
    for i,(regime,_) in enumerate(REGIMES):
        for j,ref in enumerate(RSRS):
            ax=axes[i,j];rr=sorted([r for r in result['cells'] if r['regime']==regime and r['rsr']==ref],key=lambda r:r['snr'])
            for key,label,color in [('hybrid_vs_latest','Versus frozen latest','#bf3989'),('hybrid_vs_compact','Versus frozen compact','#26965c')]:
                data=[r['contrasts'][key] for r in rr];x=[r['snr'] for r in rr]
                ax.plot(x,[r['gain_db'] for r in data],marker='.',label=label,color=color)
                ax.fill_between(x,[r['pointwise_ci95_db'][0] for r in data],[r['pointwise_ci95_db'][1] for r in data],alpha=.12,color=color)
            ax.axhline(0,color='black',lw=.8);ax.set_title(f'{regime}, reference {ref:g} dB');ax.grid(alpha=.2)
            if i==2:ax.set_xlabel('Nominal SNR (dB)')
            if j==0:ax.set_ylabel('Hybrid NMSE gain (dB)')
    axes[0,1].legend(fontsize=8);fig.suptitle('Selected hybrid paired comparisons\nPositive means hybrid is better; bands are descriptive pointwise95 intervals')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/'hybrid_gains.png',dpi=180);fig.savefig(OUT/'hybrid_gains.svg');plt.close(fig)


def attach_timing():
    """Bind completed timing records, without recomputing any accuracy result."""
    _,digest=check_selection();path=OUT/'nmse_summary.json';summary=json.loads(path.read_text())
    paths=[OUT/f'timing_window{i}.json' for i in (1,2)]
    windows={f'window{i}':json.loads(p.read_text()) for i,p in enumerate(paths,1)}
    for data in windows.values():assert data['selection_sha256']==summary['selection_sha256']==digest
    summary.update(timing=windows['window1']['methods'],timing_windows={n:dict(methods=v['methods'],fixed_cap_ms=10.74) for n,v in windows.items()},
        timing_hashes={p.name:sha(p) for p in paths},timing_analysis_source_sha256=sha(__file__))
    write_json(path,summary)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=(('hybrid_mean','Selected hybrid'),('hybrid0_mean','Hybrid, zero EM'),('latest_mean','Frozen latest'),
        ('adaptive40_mean','Adaptive + 40 EM'),('global40_mean','Global + 40 EM'),('classical40','Classical + 40 EM'),
        ('em100','EM100'),('compact_sparse','Frozen compact'))
    fig,axes=plt.subplots(1,2,figsize=(12.8,6.4),sharey=True)
    for ax,(window,data) in zip(axes,windows.items()):
        values=[data['methods'][name]['p90_ms'] for name,_ in names]
        bars=ax.barh(np.arange(len(names)),values,color=['#0969da' if i==0 else '#8c9dab' for i in range(len(names))])
        for bar,value in zip(bars,values):ax.text(value+.08,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center',fontsize=8)
        ax.axvline(10.74,color='#d73a49',ls='--',label='Fixed10.74 ms cap');ax.set_title(window)
        ax.set_xlim(0,max(max(values)*1.14,12));ax.set_xlabel('Whole-grid batch-one p90 latency (ms)');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(np.arange(len(names)),[label for _,label in names]);axes[0].invert_yaxis();axes[1].legend(fontsize=8)
    fig.suptitle('Complete selected and frozen pipelines\nBoth independent timing windows use the same fixed 10.74 ms cap')
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'hybrid_runtime.png',dpi=180);fig.savefig(OUT/'hybrid_runtime.svg');plt.close(fig)
    print('Both final timing records attached',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--timing-only',action='store_true');args=parser.parse_args()
    attach_timing() if args.timing_only else run()
