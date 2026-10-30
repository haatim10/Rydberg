"""Declared channel metrics and conditional paired bootstrap for runtime study."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/tcom_runtime_matplotlib'
import json
from pathlib import Path
import numpy as np
from .runtime_study import OUT,SNRS,REGIMES,CAP_NAMES,load
from tcom_followups_20261017.common import sha,write_json

BOOTSTRAP=3000


def main():
    selection=json.loads((OUT/'validation_selection.json').read_text());freeze=sha(OUT/'validation_selection.json')
    files=[OUT/'test'/f'{regime}_snr{snr:g}.npz' for snr in SNRS for regime,L in REGIMES]
    arrays=[load(p) for p in files]
    names=list(map(str,arrays[0]['method_names']));M=len(names);methods=names+['learned_mean']
    ni={n:i for i,n in enumerate(methods)};learned=[ni[f'learned_{s}'] for s in (9101,9102,9103)]
    rng=np.random.default_rng(20261017460)
    # Same latent path-count vector recurs across SNR at a given local index.
    resamples={regime:rng.integers(0,64,(BOOTSTRAP,64)) for regime,L in REGIMES}
    cell_scores=[];cell_boot=[];cells=[];all_errors=[];all_truth=[];all_severe=[]
    for p,a in zip(files,arrays):
        assert str(a['selection_sha256'])==freeze;assert list(map(str,a['method_names']))==names
        recalc=np.sum(abs(a['estimates']-a['truth_g'][None])**2,axis=(-2,-1))
        np.testing.assert_array_equal(recalc,a['errors'])
        e=a['errors'];en=a['truth_energy'];N=e.shape[1]
        score=10*np.log10(e.sum(1)/en.sum());score=np.r_[score,score[learned].mean()]
        ids=resamples[str(a['regime'])];b=10*np.log10(e[:,ids].sum(-1)/en[ids].sum(-1)[None])
        b=np.concatenate([b,b[learned].mean(0,keepdims=True)])
        severe=(e/en[None]>1).mean(1);severe=np.r_[severe,severe[learned].mean()]
        details={n:dict(nmse_db=float(score[i]),ci95_db=np.quantile(b[i],[.025,.975]).tolist(),severe_rate=float(severe[i])) for i,n in enumerate(methods)}
        contrasts={cap:dict(gain_db=float(score[-1]-score[ni[cap]]),ci95_db=np.quantile(b[-1]-b[ni[cap]],[.025,.975]).tolist(),positive_means='sparse better than average learned score') for cap in CAP_NAMES}
        cells.append(dict(cell=p.stem,snr=float(a['snr']),regime=str(a['regime']),cases=N,methods=details,sparse_vs_learned=contrasts))
        cell_scores.append(score);cell_boot.append(b);all_errors.append(e);all_truth.append(en);all_severe.append(severe)
    scores=np.stack(cell_scores);boots=np.stack(cell_boot);primary=scores.mean(0);bootprimary=boots.mean(0)
    severe=np.mean(all_severe,axis=0);pooled=10*np.log10(np.concatenate(all_errors,axis=1).sum(1)/np.concatenate(all_truth).sum())
    pooled=np.r_[pooled,pooled[learned].mean()]
    summary_methods={n:dict(primary_nmse_db=float(primary[i]),primary_conditional_ci95_db=np.quantile(bootprimary[i],[.025,.975]).tolist(),pooled_ratio_of_sums_db=float(pooled[i]),severe_rate=float(severe[i])) for i,n in enumerate(methods)}
    # Four declared sparse-vs-learned contrasts: Bonferroni 98.75% individual intervals.
    alpha=.05/len(CAP_NAMES)
    contrasts={cap:dict(gain_db=float(primary[-1]-primary[ni[cap]]),pointwise_ci95_db=np.quantile(bootprimary[-1]-bootprimary[ni[cap]],[.025,.975]).tolist(),familywise95_adjusted_ci_db=np.quantile(bootprimary[-1]-bootprimary[ni[cap]],[alpha/2,1-alpha/2]).tolist(),positive_means='sparse better') for cap in CAP_NAMES}
    result=dict(selection_sha256=freeze,primary_criterion='Equal15-cell average of per-cell dB ratio-of-sums NMSE; learned group averages three model scores, not predictions',
       bootstrap=dict(samples=BOOTSTRAP,seed=20261017460,unit='Shared case-index resampling across SNR within each path regime, paired methods and learned fits; preserves reused random path-count vectors',scope='Conditional on trained checkpoints, validation selection and test strata; excludes training-population and selection uncertainty',declared_family=CAP_NAMES),
       unique_channel_worlds=15*64,bootstrap_index_clusters=3*64,bootstrap_refinement='Analysis-only shared-index bootstrap added before test scoring to preserve cross-SNR path-count dependence; point estimates and selection unchanged',methods=summary_methods,contrasts=contrasts,cells=cells,
       selected_controls=selection['selections'],caps_ms=selection['caps_ms'],test_hashes={p.name:sha(p) for p in files},
       source_sha256=sha(__file__),timing_implementation_note='The EM100 reference uses the original Torch implementation. Optimized exact EM was eligible as zero-penalty sparse candidates; optimized NumPy EM100 latency is separately reported in the bypass study.')
    timingpath=OUT/'final_timing.json'
    if timingpath.exists():
        timing=json.loads(timingpath.read_text());assert timing['selection_sha256']==freeze
        result.update(final_timing_sha256=sha(timingpath),timing={n:{k:v for k,v in m.items() if k!='samples'} for n,m in timing['methods'].items()},cap_verification=timing['cap_verification'])
    write_json(OUT/'nmse_summary.json',result)
    np.savez_compressed(OUT/'nmse_bootstrap.npz',method_names=np.array(methods),cell_scores=scores,primary_bootstrap=bootprimary)
    figures(result)
    print(json.dumps(dict(primary={n:summary_methods[n]['primary_nmse_db'] for n in methods},contrasts=contrasts)),flush=True)


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'learned_mean':'#0969da','sparse_matched':'#d73a49','sparse_10ms':'#ce6d16','sparse_20ms':'#8a63d2','sparse_50ms':'#248a3d','fast_full':'#404040','optshrink':'#8d7662','em100':'#6c7785'}
    labels={'learned_mean':'SAHURE (three-fit mean)','sparse_matched':'Sparse: matched budget','sparse_10ms':'Sparse: 10 ms budget','sparse_20ms':'Sparse: 20 ms budget','sparse_50ms':'Sparse: 50 ms budget','fast_full':'Sparse: full 100 × 25','optshrink':'Offline OptShrink','em100':'EM100 (original Torch)'}
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),sharey=True)
    for ax,(regime,L) in zip(axes,REGIMES):
        rr=sorted([r for r in result['cells'] if r['regime']==regime],key=lambda r:r['snr'])
        for n in ('learned_mean','sparse_matched','sparse_10ms','sparse_20ms','sparse_50ms','fast_full','optshrink'):
            ax.plot([r['snr'] for r in rr],[r['methods'][n]['nmse_db'] for r in rr],color=colors[n],label=labels[n],marker='.',lw=1.5)
        ax.set_title({'L2':'Two paths','L4':'Four paths','random3to7':'Random 3–7 paths'}[regime]);ax.set_xlabel('SNR (dB)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Channel NMSE (dB; lower is better)')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,fontsize=8)
    fig.suptitle('Fresh test cases after validation-only sparse selection; RSR 12 dB')
    fig.tight_layout(rect=(0,.16,1,.95));fig.savefig(OUT/'nmse_by_snr.png',dpi=180);fig.savefig(OUT/'nmse_by_snr.svg');plt.close(fig)
    if 'timing' not in result:return
    fig,ax=plt.subplots(figsize=(9,5.8))
    for n in labels:
        t=result['timing'][n]['p90_ms'];m=result['methods'][n];lo,hi=m['primary_conditional_ci95_db']
        ax.errorbar(t,m['primary_nmse_db'],yerr=[[m['primary_nmse_db']-lo],[hi-m['primary_nmse_db']]],fmt='o',color=colors[n],capsize=3,label=f"{labels[n]} ({t:.2f} ms)")
    ax.set_xscale('log');ax.set_xlabel('Measured batch-one p90 processing time (ms, log scale)');ax.set_ylabel('Equal-cell channel NMSE (dB; lower is better)')
    ax.set_title('Measured runtime and channel accuracy\nBudgets use calibration p90; points use fresh verification timing')
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
    fig.text(.5,.015,'Optimized exact EM was eligible among the zero-penalty sparse candidates.',ha='center',fontsize=8,color='.3')
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'runtime_accuracy.png',dpi=180);fig.savefig(OUT/'runtime_accuracy.svg');plt.close(fig)

if __name__=='__main__':main()
