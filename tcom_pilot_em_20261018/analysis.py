"""Fixed primary contrasts and paired channel bootstrap for pilot/EM study."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/tcom_pilot_em_matplotlib'
import json
import numpy as np
from .study import OUT,SNRS,REGIMES,FAMILIES,SEEDS,TAILS,CONTRASTS,PRIOR_NAMES,load,check_selection
from tcom_followups_20261017.common import sha,write_json

BOOT=20000;BOOT_SEED=2026101894


def groups(names):
    return {f'{f}_{kind}_mean':[names.index(f'{f}_{s}_{kind}') for s in SEEDS] for f in FAMILIES for kind in ['tail0','tail1','tail2','tail5','selected']}


def expand(values,group):return np.concatenate([values,np.stack([values[i].mean(0) for i in group.values()])])


def intervals(values):
    return dict(pointwise_ci95=np.quantile(values,[.025,.975]).tolist(),eight_outcome_simultaneous95_ci=np.quantile(values,[.003125,.996875]).tolist())


def main():
    selection,tails_record,freeze=check_selection()
    files=[OUT/'test'/f'{regime}_snr{snr:g}.npz' for snr in SNRS for regime,L in REGIMES]
    first=load(files[0]);names=list(map(str,first['method_names']));group=groups(names);methods=names+list(group);ni={n:i for i,n in enumerate(methods)}
    rng=np.random.default_rng(BOOT_SEED)
    weights={regime:rng.multinomial(128,np.full(128,1/128),size=BOOT) for regime,L in REGIMES}
    total_boot=np.zeros((len(methods),BOOT));scores=[];rows=[];errs=[];energies=[];case_nmse=[];hashes={}
    def cell(path,weight):
        a=load(path);assert str(a['selection_sha256'])==freeze and list(map(str,a['method_names']))==names
        error=np.sum(abs(a['estimates']-a['truth_g'][None])**2,axis=(-2,-1));np.testing.assert_array_equal(error,a['errors'])
        en=np.sum(abs(a['truth_g'])**2,axis=(-2,-1));np.testing.assert_array_equal(en,a['truth_energy'])
        assert float(a['maximum_continuation_nll_increase'])<=1e-10
        point=expand(10*np.log10(error.sum(1)/en.sum()),group)
        boot=expand(10*np.log10((error@weight.T)/(en@weight.T)[None]),group)
        nmse=error/en[None];severe=expand((nmse>1).mean(1),group);p95=expand(np.quantile(nmse,.95,axis=1),group);p99=expand(np.quantile(nmse,.99,axis=1),group)
        details={n:dict(nmse_db=float(point[i]),ci95_db=np.quantile(boot[i],[.025,.975]).tolist(),severe_rate=float(severe[i]),case_nmse_p95=float(p95[i]),case_nmse_p99=float(p99[i])) for n,i in ni.items()}
        contrasts={key:dict(reference=reference,candidate=candidate,gain_db=float(point[ni[reference]]-point[ni[candidate]]),ci95_db=np.quantile(boot[ni[reference]]-boot[ni[candidate]],[.025,.975]).tolist()) for key,(reference,candidate) in CONTRASTS.items()}
        row=dict(cell=path.stem,snr=float(a['snr']),rsr=float(a['rsr']),regime=str(a['regime']),guard=bool(a['guard']),cases=len(en),methods=details,contrasts=contrasts)
        return row,point,boot,error,en,nmse
    for path in files:
        a=load(path);row,point,boot,error,en,nmse=cell(path,weights[str(a['regime'])])
        rows.append(row);scores.append(point);total_boot+=boot/len(files);errs.append(error);energies.append(en);case_nmse.append(nmse);hashes[path.name]=sha(path)
    primary=np.mean(scores,axis=0);allerror=np.concatenate(errs,axis=1);allenergy=np.concatenate(energies);allnmse=np.concatenate(case_nmse,axis=1)
    pooled=expand(10*np.log10(allerror.sum(1)/allenergy.sum()),group);severe=expand((allnmse>1).mean(1),group)
    p95=expand(np.quantile(allnmse,.95,axis=1),group);p99=expand(np.quantile(allnmse,.99,axis=1),group)
    summary_methods={n:dict(primary_nmse_db=float(primary[i]),conditional_ci95_db=np.quantile(total_boot[i],[.025,.975]).tolist(),pooled_ratio_of_sums_db=float(pooled[i]),
      severe_rate=float(severe[i]),case_nmse_p95=float(p95[i]),case_nmse_p99=float(p99[i])) for n,i in ni.items()}
    contrasts={}
    for key,(reference,candidate) in CONTRASTS.items():
        rb=total_boot[ni[reference]];cb=total_boot[ni[candidate]];difference=rb-cb
        def indices(n):return group[n] if n in group else [ni[n]]*3
        ri,ci=indices(reference),indices(candidate)
        gains={str(seed):float(primary[ri[j]]-primary[ci[j]]) for j,seed in enumerate(SEEDS)}
        worst=min(rows,key=lambda r:r['contrasts'][key]['gain_db'])
        contrasts[key]=dict(reference=reference,candidate=candidate,gain_db=float(primary[ni[reference]]-primary[ni[candidate]]),
          **intervals(difference),seed_specific_gains_db=gains,worst_cell=worst['cell'],worst_cell_gain_db=worst['contrasts'][key]['gain_db'],
          negative_point_cells=[r['cell'] for r in rows if r['contrasts'][key]['gain_db']<0],
          pointwise95_harm_cells=[r['cell'] for r in rows if r['contrasts'][key]['ci95_db'][1]<0])
    guard_rng=np.random.default_rng(2026101895);guards=[]
    for path in sorted((OUT/'guards').glob('*.npz')):
        a=load(path);count=len(a['truth_energy']);weight=guard_rng.multinomial(count,np.full(count,1/count),size=BOOT)
        row,*_=cell(path,weight);guards.append(row);hashes['guards/'+path.name]=sha(path)
    result=dict(selection_sha256=freeze,protocol_sha256=selection['protocol_sha256'],tail_choices=tails_record['choices'],primary_contrasts=contrasts,
      methods=summary_methods,cells=rows,guards=guards,primary_metric='Equal15-cell average of per-cell dB ratio of summed errors to summed truth energies; neural groups average three individual fit scores, never estimates',
      bootstrap=dict(samples=BOOT,seed=BOOT_SEED,unit='Multinomial case-count weights perpathregime, shared acrossSNR and allmethods/fits',
        channel_worlds=1920,index_clusters=384,scope='Conditional on fixed fitted checkpoints and validationselectedtails/sparsecontrols; excludes training-population and selection uncertainty',
        multiplicity='Four declared contrasts times two quality metrics(NMSE,BER)=eight; adjusted individual percentile interval99.375%. Per-cell and guard intervals descriptive pointwise95.'),
      tail_metrics='Group percase p95/p99 are averages of three fit-specific quantiles; severe threshold is percaseNMSE>1.',
      paired_training_caveat='Both fresh context21/context30 arms select checkpoints using30-context/9-heldout likelihood. Thus theircontrast isolates paired trainingcontext count under sharedselection, while oldvsfresh also changes data andselection.',
      guard_policy='Guard localids1000..1063; noise lookup clamps to20dB, no guard-specific tailorbaseline calibration.',test_hashes=hashes,analysis_source_sha256=sha(__file__))
    if (OUT/'final_timing.json').exists():
        timing=json.loads((OUT/'final_timing.json').read_text());assert timing['selection_sha256']==freeze
        result.update(timing=timing['methods'],cap_verification=timing['cap_verification'],timing_sha256=sha(OUT/'final_timing.json'))
    write_json(OUT/'nmse_summary.json',result)
    np.savez_compressed(OUT/'nmse_bootstrap.npz',method_names=np.array(methods),cell_scores=np.stack(scores),primary_bootstrap=total_boot)
    figures(result)
    print(json.dumps(dict(primary_contrasts=contrasts,tail_choices=tails_record['choices'])),flush=True)


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'old_tail0_mean':'Original model','context21_tail0_mean':'Fresh 21-pilot training',
      'context30_selected_mean':'Fresh 30-pilot training (selected tail: 0)','sparse_fresh_context30':'Fresh sparse, matched latency','prior_sparse_50ms':'Prior sparse, 50 ms budget','em100':'Optimized EM100'}
    colors=['#606060','#e08c27','#0969da','#d73a49','#29944c','#9270ac']
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),sharey=True)
    for ax,(regime,L) in zip(axes,REGIMES):
        rr=sorted([r for r in result['cells'] if r['regime']==regime],key=lambda r:r['snr'])
        for (name,label),color in zip(labels.items(),colors):ax.plot([r['snr'] for r in rr],[r['methods'][name]['nmse_db'] for r in rr],label=label,color=color,marker='.',lw=1.6)
        ax.set_title({'L2':'Two paths','L4':'Four paths','random3to7':'Random 3–7 paths'}[regime]);ax.set_xlabel('SNR (dB)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Channel NMSE (dB; lower is better)');h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Fresh test after pilot-count matching and held-out EM-tail selection')
    fig.tight_layout(rect=(0,.17,1,.95));fig.savefig(OUT/'pilot_em_nmse.png',dpi=180);fig.savefig(OUT/'pilot_em_nmse.svg');plt.close(fig)
    if 'timing' not in result:return
    fig,axs=plt.subplots(1,2,figsize=(14,6.4),sharey=True,gridspec_kw={'width_ratios':[1.25,1]});family_colors={'old':'#606060','context21':'#e08c27','context30':'#0969da'}
    family_labels={'old':'Original model','context21':'Fresh 21-pilot training','context30':'Fresh 30-pilot training'}
    annotation_offsets={'old':(-14,-12),'context21':(-12,9),'context30':(6,0)}
    for family in FAMILIES:
        names=[f'{family}_tail{t}_mean' for t in TAILS]
        x=[result['timing'][n]['p90_ms'] for n in names];y=[result['methods'][n]['primary_nmse_db'] for n in names]
        for ax in axs:
            ax.plot(x,y,'o-',color=family_colors[family],label=family_labels[family]+'; fixed tails')
            name=f'{family}_selected_mean';ax.scatter(result['timing'][name]['p90_ms'],result['methods'][name]['primary_nmse_db'],marker='*',s=135,color=family_colors[family])
            name=f'sparse_fresh_{family}';ax.scatter(result['timing'][name]['p90_ms'],result['methods'][name]['primary_nmse_db'],marker='D',s=40,color=family_colors[family])
        for xx,yy,t in zip(x,y,TAILS):axs[1].annotate(str(t),(xx,yy),xytext=annotation_offsets[family],textcoords='offset points',fontsize=8,color=family_colors[family])
    for name,label,marker,color in [('prior_sparse_matched','Prior sparse matched','s','#d73a49'),('prior_sparse_10ms','Prior sparse 10 ms','s','#b16637'),('prior_sparse_20ms','Prior sparse 20 ms','s','#9762ae'),('prior_sparse_50ms','Prior sparse 50 ms','s','#29894b'),('prior_sparse_full','Full sparse 100 × 25','s','#111111'),('em100','Optimized EM100','^','#7b80a7'),('optshrink','Offline OptShrink','^','#927a61')]:
        for ax in axs:ax.scatter(result['timing'][name]['p90_ms'],result['methods'][name]['primary_nmse_db'],marker=marker,color=color,label=label)
    axs[0].set_xscale('log');axs[0].set_xlabel('Batch-one p90 time (ms; log scale)');axs[0].set_ylabel('Equal-cell channel NMSE (dB; lower is better)');axs[0].set_title('Full latency range')
    axs[1].set_xlim(4.9,6.05);axs[1].set_xlabel('Batch-one p90 time (ms; linear scale)');axs[1].set_title('Fast estimators: enlarged view')
    fig.suptitle('Accuracy and measured end-to-end latency\nNumbers: EM tail length; stars: selected tail 0; diamonds: freshly matched sparse',fontsize=13)
    for ax in axs:ax.grid(alpha=.2)
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',fontsize=8,ncol=4)
    fig.tight_layout(rect=(0,.14,1,.92));fig.savefig(OUT/'pilot_em_runtime.png',dpi=180);fig.savefig(OUT/'pilot_em_runtime.svg');plt.close(fig)

if __name__=='__main__':main()
