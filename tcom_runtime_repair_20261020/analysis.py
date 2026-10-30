"""Fresh numerical-equivalence comparison for the unchanged runtime implementation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/private/tmp/tcom_runtime_repair_matplotlib'
import json
import numpy as np
from .study import OUT,SNRS,RSRS,REGIMES,SEEDS,NAMES,GROUPS,TEST_COUNT,load,cellname,check_selection
from tcom_followups_20261017.common import sha,write_json

BOOT=20000;BOOT_SEED=2026102009;COUNT=TEST_COUNT;TOL=1e-10
CONTRASTS=dict(repair_vs_current=('current_repair_mean','repair_mean'),repair_vs_old=('old_bypass_mean','repair_mean'),repair_vs_prior=('prior_repair_mean','repair_mean'),repair_vs_original=('original_mean','repair_mean'),
               repair_vs_em100=('em100','repair_mean'),repair_vs_compact=('compact_sparse','repair_mean'))
PRIMARY=('repair_vs_current',)
PRIMARY_REGIONS=('all','weak_warm100')
REGIONS=('all','weak_warm100','strong_high','strong_transition','ordinary','weak','rsr5','rsr12','rsr20',
    'all_L2','all_L4','all_random3to7','weak_warm100_L2','weak_warm100_L4','weak_warm100_random3to7')
PARITY_ATOL=1e-10;PARITY_RTOL=1e-10
METHODS=list(NAMES)+list(GROUPS);INDEX={name:i for i,name in enumerate(METHODS)}
BOUNDARY_EXTRA=(38.,38.5,38.75,39.,39.5,40.5,41.,41.5,42.)
BOUNDARY_SNRS=tuple(sorted((37.5,40.,42.5)+BOUNDARY_EXTRA))
BOUNDARY_METHODS=('current_repair_mean','repair_mean','em100')


def belongs(region,snr,rsr=None,regime=None):
    scopes=dict(all=True,weak_warm100=rsr==5 and snr>=37.5,
        strong_high=rsr in (12,20) and snr>=40,
        strong_transition=rsr in (12,20) and 32.5<=snr<=37.5,
        ordinary=snr<=20,weak=rsr==5)
    if region in scopes:return scopes[region]
    if region.startswith('rsr'):return rsr==float(region[3:])
    for prefix in ('all_','weak_warm100_'):
        if region.startswith(prefix):return scopes[prefix[:-1]] and regime==region[len(prefix):]
    raise KeyError(region)


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
    rows=[];errors=[];energies=[];points=[];hashes={};dispatch=0.;parity_rows=[]
    for regime,_ in REGIMES:
        for ref in RSRS:
            for snr in SNRS:
                path=OUT/'test/cells'/f'{cellname(regime,ref,snr)}.npz';a=load(path)
                assert str(a['selection_sha256'])==digest
                assert tuple(map(str,a['method_names']))==tuple(NAMES)
                err=np.sum(abs(a['estimates']-a['truth_g'][None])**2,axis=(-2,-1));energy=np.sum(abs(a['truth_g'])**2,axis=(-2,-1))
                np.testing.assert_array_equal(err,a['errors']);np.testing.assert_array_equal(energy,a['truth_energy'])
                assert err.shape==(len(NAMES),COUNT)
                fit_parity=[]
                for seed in SEEDS:
                    canonical=a['estimates'][INDEX[f'current_repair_{seed}']]
                    native=a['estimates'][INDEX[f'repair_{seed}']]
                    difference=np.abs(native-canonical);limit=PARITY_ATOL+PARITY_RTOL*np.abs(canonical)
                    assert np.all(difference<=limit)
                    fit_parity.append(dict(seed=seed,maximum_absolute_error=float(difference.max()),
                        maximum_tolerance_fraction=float((difference/limit).max()),bitwise_equal=bool(np.array_equal(native,canonical))))
                parity_rows.append(dict(cell=path.stem,models=fit_parity))
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
    assert len(rows)==153
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
                **intervals(boot[INDEX[reference]]-boot[INDEX[candidate]],region in PRIMARY_REGIONS and key in PRIMARY))
        aggregates[region]=dict(cells=int(keep.sum()),condition_cases=int(keep.sum())*COUNT,methods=methods,contrasts=contrasts)
    negative={}
    for key in CONTRASTS:
        worst=min(rows,key=lambda r:r['contrasts'][key]['gain_db'])
        target=[r for r in rows if belongs('weak_warm100',r['snr'],r['rsr'],r['regime'])]
        negative[key]=dict(negative_cells=[r['cell'] for r in rows if r['contrasts'][key]['gain_db'] < -TOL],
            pointwise95_harm_cells=[r['cell'] for r in rows if r['contrasts'][key]['pointwise_ci95_db'][1] < -TOL],
            target_negative_cells=[r['cell'] for r in target if r['contrasts'][key]['gain_db'] < -TOL],
            worst_cell=worst['cell'],worst_gain_db=worst['contrasts'][key]['gain_db'])
    result=dict(selection_sha256=digest,protocol_sha256=selection['protocol_sha256'],unique_test_anchors=COUNT*9,condition_cases=len(rows)*COUNT,
        bootstrap_index_clusters=COUNT*3,actual_dispatch_max_abs_error=dispatch,
        numerical_parity=dict(atol=PARITY_ATOL,rtol=PARITY_RTOL,passed=True,cells=parity_rows,
            maximum_absolute_error=max(f['maximum_absolute_error'] for r in parity_rows for f in r['models']),
            maximum_tolerance_fraction=max(f['maximum_tolerance_fraction'] for r in parity_rows for f in r['models']),
            bitwise_equal_cell_model_pairs=sum(f['bitwise_equal'] for r in parity_rows for f in r['models']),
            total_cell_model_pairs=len(parity_rows)*len(SEEDS)),
        scoring='Equal-cell mean dB ratio-of-sums NMSE. Three-fit groups average individual metrics, never estimates. Pooled linear NMSE and tail rates are diagnostics.',
        primary='Optimized versus canonical latest implementation in all153 conditions and weak-reference warm10018 conditions; NMSE and BER form four prespecified outcomes. Roundoff-scale effects are not estimator superiority or population-equivalence claims.',
        bootstrap=dict(samples=BOOT,seed=BOOT_SEED,unit='Whole-channel multinomial local index shared across SNR/reference/method/fit within regime',
            scope='Conditional on frozen models, policies and grid; excludes training-population/selection uncertainty',
            multiplicity='Bonferroni4 on one canonical/native contrast in two regions times NMSE/BER. Other intervals descriptive pointwise95.'),
        aggregates=aggregates,negative_cells=negative,numerical_tie_tolerance_db=TOL,cells=rows,test_hashes=hashes,source_sha256=sha(__file__))
    write_json(OUT/'nmse_summary.json',result)
    np.savez_compressed(OUT/'nmse_bootstrap.npz',method_names=np.array(METHODS),regions=np.array(REGIONS),aggregate_bootstraps=np.stack([boots[r] for r in REGIONS]),
        cell_names=np.array([r['cell'] for r in rows]),cell_scores=points,selection_sha256=np.array(digest))
    figures(result)
    print(json.dumps(dict(primary={r:aggregates[r]['contrasts']['repair_vs_current'] for r in PRIMARY_REGIONS},negative_cells=negative)),flush=True)


def figures(result):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'original_mean':('#d73a49','Retrained model','-'),'old_bypass_mean':('#d99322','Previous EM100 bypass','--'),
        'prior_repair_mean':('#7b52ab','Previous repair',':'),'current_repair_mean':('#d15a8a','Canonical latest','--'),'repair_mean':('#0969da','Optimized latest','-'),'em100':('#26965c','Direct EM100','--'),
        'compact_sparse':('#9b6718','Matched sparse/EM family','-.'),'prior_sparse_full':('#777777','Frozen full sparse',':'),
        'optshrink':('#8a63d2','Frozen OptShrink','--')}
    fig,axes=plt.subplots(3,3,figsize=(13.8,10.6),sharex=True,sharey=True)
    for i,(regime,_) in enumerate(REGIMES):
        for j,ref in enumerate(RSRS):
            ax=axes[i,j];rr=sorted([r for r in result['cells'] if r['regime']==regime and r['rsr']==ref],key=lambda r:r['snr'])
            for name,(color,label,ls) in styles.items():ax.plot([r['snr'] for r in rr],[r['methods'][name]['nmse_db'] for r in rr],color=color,label=label,ls=ls,marker='.',lw=1.5)
            if ref in (12,20):ax.axvspan(40,60,color='#0969da',alpha=.06)
            ax.set_title(f'{regime}, reference {ref:g} dB');ax.grid(alpha=.2)
            if i==2:ax.set_xlabel('Nominal SNR (dB)')
            if j==0:ax.set_ylabel('Channel NMSE (dB; lower is better)')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Fresh test: optimized implementation of the same estimator\nCurves show performance; canonical/native differences are checked separately')
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(OUT/'repair_nmse.png',dpi=180);fig.savefig(OUT/'repair_nmse.svg');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13.8,10),sharex=True)
    for i,(regime,_) in enumerate(REGIMES):
        for j,ref in enumerate(RSRS):
            ax=axes[i,j]
            rr=sorted([r for r in result['cells'] if r['regime']==regime and r['rsr']==ref],key=lambda r:r['snr'])
            for key,label,color in [('repair_vs_current','Canonical minus optimized NMSE','#0969da')]:
                data=[r['contrasts'][key] for r in rr];x=[r['snr'] for r in rr]
                ax.plot(x,[r['gain_db'] for r in data],marker='.',label=label,color=color)
                ax.fill_between(x,[r['pointwise_ci95_db'][0] for r in data],[r['pointwise_ci95_db'][1] for r in data],alpha=.12,color=color)
            ax.axhline(0,color='black',lw=.8);ax.set_title(f'{regime}, reference {ref} dB');ax.grid(alpha=.2)
            if i==2:ax.set_xlabel('Nominal SNR (dB)')
            if j==0:ax.set_ylabel('NMSE difference (dB)')
    axes[0,1].legend(fontsize=8)
    fig.suptitle('Implementation differences across all references and SNRs\nBands are descriptive pointwise conditional 95% intervals')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/'repair_target.png',dpi=180);fig.savefig(OUT/'repair_target.svg');plt.close(fig)


def attach_timing():
    _,digest=check_selection();path=OUT/'nmse_summary.json';summary=json.loads(path.read_text())
    windows={f'window{i}':json.loads((OUT/f'timing_window{i}.json').read_text()) for i in (1,2)}
    for data in windows.values():assert data['selection_sha256']==summary['selection_sha256']==digest
    summary.update(timing=windows['window1']['methods'],
        timing_windows={name:dict(methods=data['methods'],fixed_cap_ms=10.74) for name,data in windows.items()},
        timing_hashes={f'timing_window{i}.json':sha(OUT/f'timing_window{i}.json') for i in (1,2)},
        timing_analysis_source_sha256=sha(__file__))
    write_json(path,summary)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.4),sharey=True)
    names=(('current_repair_mean','Canonical latest'),('repair_mean','Optimized latest'),('compact_sparse','Frozen compact'),('em100','Spectral EM100'))
    for ax,(window,data) in zip(axes,windows.items()):
        raw=dict(data['raw_samples'])
        raw.update({name:[r for member in members for r in raw[member]] for name,members in GROUPS.items() if all(member in raw for member in members)})
        labels=[];values=[]
        for region in ('all','weak_warm100'):
            for name,label in names:
                labels.append(f'{label} | '+('whole grid' if region=='all' else 'weak warm100'))
                values.append(float(np.quantile([r['ms'] for r in raw[name] if belongs(region,r['snr'],r['rsr'],r['regime'])],.9)))
        bars=ax.barh(np.arange(len(values)),values,color=['#0969da' if i%4==1 else '#8c9dab' for i in range(len(values))])
        for bar,value in zip(bars,values):ax.text(value+.1,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center',fontsize=8)
        ax.axvline(10.74,color='#d73a49',ls='--',label='Fixed10.74 ms target');ax.set_title(window)
        ax.set_xlabel('Batch-one p90 latency (ms)');ax.set_xlim(0,max(max(values)*1.16,12));ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(np.arange(len(labels)),labels);axes[0].invert_yaxis();axes[1].legend(fontsize=8)
    fig.suptitle('Two prespecified isolated runtime windows\nInitialization, conversion, routing and all estimator updates included')
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/'repair_runtime.png',dpi=180);fig.savefig(OUT/'repair_runtime.svg');plt.close(fig)
    print('Both timing windows attached; runtime figure saved',flush=True)


def boundary_only():
    """Describe coupled boundary curves; never include dense points in main scores.

    Differences are paired changes in ratio-of-sums dB NMSE, higher minus lower
    SNR. Positive changes mean worsening. These curves cannot distinguish every
    physical-noise effect from a numerical/local-solution effect; they are not a
    continuity theorem or an additional confirmatory hypothesis family.
    """
    selection,digest=check_selection()
    rng=np.random.default_rng(BOOT_SEED)
    weights={r:rng.multinomial(COUNT,np.full(COUNT,1/COUNT),size=BOOT).T.astype(float) for r,_ in REGIMES}
    raw_names=tuple(dict.fromkeys(n for name in BOUNDARY_METHODS for n in GROUPS.get(name,(name,))))
    raw_index={name:i for i,name in enumerate(raw_names)}
    take=[INDEX[name] for name in raw_names]

    def boundary_expand(values):
        return np.stack([values[[raw_index[n] for n in GROUPS.get(name,(name,))]].mean(0) for name in BOUNDARY_METHODS])

    rows=[];hashes={};scores={};boots={};dispatch=0.
    for regime,_ in REGIMES:
        for ref in (12.,20.):
            anchor=None
            for snr in BOUNDARY_SNRS:
                directory='boundary/cells' if snr in BOUNDARY_EXTRA else 'test/cells'
                path=OUT/directory/f'{cellname(regime,ref,snr)}.npz';a=load(path)
                assert str(a['selection_sha256'])==digest
                assert tuple(map(str,a['method_names']))==tuple(NAMES)
                assert a['estimates'].shape[0:2]==(len(NAMES),COUNT)
                if anchor is None:
                    anchor={name:a[name].copy() for name in ('truth_g','indices','local_indices','s','b')}
                else:
                    for name,value in anchor.items():np.testing.assert_array_equal(a[name],value)
                errors=np.sum(abs(a['estimates']-a['truth_g'][None])**2,axis=(-2,-1))
                energy=np.sum(abs(a['truth_g'])**2,axis=(-2,-1))
                np.testing.assert_array_equal(errors,a['errors']);np.testing.assert_array_equal(energy,a['truth_energy'])
                for seed in SEEDS:
                    np.testing.assert_allclose(a['estimates'][INDEX[f'repair_{seed}']],a['estimates'][INDEX[f'current_repair_{seed}']],atol=PARITY_ATOL,rtol=PARITY_RTOL)
                chosen_errors=errors[take]
                score=boundary_expand(10*np.log10(chosen_errors.sum(1)/energy.sum()))
                boot=boundary_expand(10*np.log10((chosen_errors@weights[regime])/(energy@weights[regime])[None]))
                methods={name:dict(nmse_db=float(score[i]),conditional_ci95_db=np.quantile(boot[i],[.025,.975]).tolist())
                         for i,name in enumerate(BOUNDARY_METHODS)}
                gain=score[0]-score[1];gain_boot=boot[0]-boot[1]
                rows.append(dict(cell=path.stem,regime=regime,rsr=ref,snr=snr,cases=COUNT,source_directory=directory,
                    methods=methods,repair_vs_current=dict(gain_db=float(gain),**intervals(gain_boot))))
                scores[(regime,ref,snr)]=score;boots[(regime,ref,snr)]=boot
                hashes[str(path.relative_to(OUT))]=sha(path)
                dispatch=max(dispatch,float(a.get('actual_dispatch_max_abs_error',0)))
    assert len(rows)==72

    def change_record(regime,ref,lower,upper):
        delta=scores[(regime,ref,upper)]-scores[(regime,ref,lower)]
        delta_boot=boots[(regime,ref,upper)]-boots[(regime,ref,lower)]
        methods={name:dict(change_db=float(delta[i]),change_per_snr_db=float(delta[i]/(upper-lower)),
                    pointwise_ci95_db=np.quantile(delta_boot[i],[.025,.975]).tolist()) for i,name in enumerate(BOUNDARY_METHODS)}
        return dict(regime=regime,rsr=ref,lower_snr=lower,upper_snr=upper,methods=methods,
            worsening_reduction_vs_current=dict(change_db=float(delta[0]-delta[1]),
                pointwise_ci95_db=np.quantile(delta_boot[0]-delta_boot[1],[.025,.975]).tolist()))

    adjacent=[change_record(regime,ref,lower,upper) for regime,_ in REGIMES for ref in (12.,20.)
              for lower,upper in zip(BOUNDARY_SNRS[:-1],BOUNDARY_SNRS[1:])]
    endpoint=[change_record(regime,ref,37.5,40.) for regime,_ in REGIMES for ref in (12.,20.)]
    pooled=[]
    for lower,upper in list(zip(BOUNDARY_SNRS[:-1],BOUNDARY_SNRS[1:]))+[(37.5,40.)]:
        delta=np.mean([scores[(r,ref,upper)]-scores[(r,ref,lower)] for r,_ in REGIMES for ref in (12.,20.)],axis=0)
        boot=np.mean([boots[(r,ref,upper)]-boots[(r,ref,lower)] for r,_ in REGIMES for ref in (12.,20.)],axis=0)
        pooled.append(dict(lower_snr=lower,upper_snr=upper,cells_at_each_endpoint=6,
            methods={name:dict(change_db=float(delta[i]),pointwise_ci95_db=np.quantile(boot[i],[.025,.975]).tolist())
                     for i,name in enumerate(BOUNDARY_METHODS)},
            worsening_reduction_vs_current=dict(change_db=float(delta[0]-delta[1]),
                pointwise_ci95_db=np.quantile(boot[0]-boot[1],[.025,.975]).tolist())))
    result=dict(selection_sha256=digest,protocol_sha256=selection['protocol_sha256'],source_sha256=sha(__file__),
        scope='Descriptive coupled boundary diagnostic only; no extra primary hypothesis and no dense-point contribution to main153 aggregates.',
        snrs=list(BOUNDARY_SNRS),additional_snrs=list(BOUNDARY_EXTRA),cells=len(rows),additional_cells=54,shared_main_cells=18,
        unique_channel_reference_anchors=COUNT*6,bootstrap_index_clusters=COUNT*3,condition_cases=len(rows)*COUNT,
        historical_lookup_boundary_db=38.75,
        routing_note='Both compared implementations use the identical already-repaired policy across the historical38.75dB boundary; this sweep checks numerical preservation, not a newly selected cutoff.',
        change_sign='Higher-SNR minus lower-SNR NMSE in dB; positive means worsening. Worsening reduction is canonical change minus native change.',
        paired_anchor_identity=True,actual_dispatch_max_abs_error=dispatch,
        bootstrap=dict(samples=BOOT,seed=BOOT_SEED,unit='Whole-channel local index shared across SNR/reference/method/fit within regime',
            scope='Descriptive conditional pointwise95 intervals; no multiplicity-adjusted claims and no training/selection uncertainty.'),
        methods=list(BOUNDARY_METHODS),curves=rows,adjacent_changes=adjacent,changes_37_5_to_40=endpoint,
        equal_cell_changes=pooled,cell_hashes=hashes)
    destination=OUT/'boundary';destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'summary.json',result)
    boundary_figures(result)
    print(json.dumps(dict(boundary_cells=len(rows),changes_37_5_to_40=endpoint)),flush=True)


def boundary_figures(result):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'current_repair_mean':('#d15a8a','Canonical latest','--'),
            'repair_mean':('#0969da','Optimized latest','-'),'em100':('#26965c','Direct EM100',':')}
    fig,axes=plt.subplots(3,2,figsize=(11.5,10.2),sharex=True,sharey=True)
    for i,(regime,_) in enumerate(REGIMES):
        for j,ref in enumerate((12.,20.)):
            ax=axes[i,j]
            rows=sorted([row for row in result['curves'] if row['regime']==regime and row['rsr']==ref],key=lambda r:r['snr'])
            for name,(color,label,linestyle) in styles.items():
                ax.plot([r['snr'] for r in rows],[r['methods'][name]['nmse_db'] for r in rows],
                    color=color,label=label,ls=linestyle,marker='.',lw=1.5)
            ax.axvline(38.75,color='#555555',lw=.8,ls=':')
            ax.set_title(f'{regime}, reference {ref:g} dB');ax.grid(alpha=.2)
            if i==2:ax.set_xlabel('Nominal SNR (dB)')
            if j==0:ax.set_ylabel('Channel NMSE (dB; lower is better)')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Fresh numerical-parity boundary sweep, separate from the main benchmark\nDotted line: historical38.75 dB boundary; both methods use the same policy')
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(OUT/'boundary/nmse_boundary.png',dpi=180);fig.savefig(OUT/'boundary/nmse_boundary.svg');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(11.5,10.2),sharex=True)
    for i,(regime,_) in enumerate(REGIMES):
        for j,ref in enumerate((12.,20.)):
            ax=axes[i,j]
            rows=[row for row in result['adjacent_changes'] if row['regime']==regime and row['rsr']==ref]
            for name,(color,label,linestyle) in styles.items():
                ax.plot([(r['lower_snr']+r['upper_snr'])/2 for r in rows],[r['methods'][name]['change_db'] for r in rows],
                    color=color,label=label,ls=linestyle,marker='.',lw=1.5)
            ax.axhline(0,color='black',lw=.8);ax.axvline(38.75,color='#555555',lw=.8,ls=':')
            ax.set_title(f'{regime}, reference {ref:g} dB');ax.grid(alpha=.2)
            if i==2:ax.set_xlabel('Midpoint of adjacent SNR samples (dB)')
            if j==0:ax.set_ylabel('NMSE change (dB)')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Every adjacent boundary change, with coupled channels and noise\nRaw changes, not slopes; sample spacing is0.25 or0.5 dB')
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(OUT/'boundary/nmse_adjacent_changes.png',dpi=180);fig.savefig(OUT/'boundary/nmse_adjacent_changes.svg');plt.close(fig)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group()
    mode.add_argument('--timing-only',action='store_true');mode.add_argument('--boundary-only',action='store_true');args=p.parse_args()
    if args.timing_only:attach_timing()
    elif args.boundary_only:boundary_only()
    else:run()
