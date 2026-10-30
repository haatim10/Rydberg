"""Paired anchor-cluster analysis of the prespecified frozen bypass study."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/tcom_bypass_matplotlib'
import json
import numpy as np
from .bypass_study import OUT,SNRS,RSRS,REGIMES,SEEDS,NAMES,load,cellname,check_selection
from tcom_followups_20261017.common import write_json,sha

BOOT=3000
GROUPS={'original_mean':list(range(3)),'off_mean':list(range(3,6)),
        'fast_bypass_mean':list(range(6,9)),'em100_bypass_mean':list(range(10,13))}
METHODS=list(NAMES)+list(GROUPS)
NI={name:i for i,name in enumerate(METHODS)}
ARMS=('fast_bypass_mean','em100_bypass_mean')


def expanded(values):
    return np.concatenate([values,np.stack([values[ids].mean(0) for ids in GROUPS.values()])])


def summarize(scores,boots):
    return {name:dict(nmse_db=float(scores[i]),conditional_ci95_db=np.quantile(boots[i],[.025,.975]).tolist()) for i,name in enumerate(METHODS)}


def main():
    selection,freeze=check_selection();rng=np.random.default_rng(20261017660)
    resamples={regime:rng.integers(0,64,(BOOT,64)) for regime,L in REGIMES}
    rows=[];all_scores=[];all_boots=[];all_errors=[];all_energy=[];all_nmse=[];hashes={};identity=True;dispatch_max=0.
    for regime,L in REGIMES:
        for rsr in RSRS:
            a=load(OUT/'test'/'anchors'/f'{regime}_rsr{rsr:g}.npz')
            assert str(a['selection_sha256'])==freeze
            for snr in SNRS:
                path=OUT/'test'/'cells'/f'{cellname(regime,rsr,snr)}.npz';c=load(path)
                assert str(c['selection_sha256'])==freeze and tuple(map(str,c['method_names']))==NAMES
                error=np.sum(abs(c['estimates']-a['g'][None])**2,axis=(-2,-1))
                np.testing.assert_array_equal(error,c['errors'])
                energy=np.sum(abs(a['g'])**2,axis=(-2,-1));np.testing.assert_array_equal(energy,c['truth_energy'])
                if snr<=20:
                    assert np.array_equal(c['estimates'][6:9],c['estimates'][:3])
                    assert np.array_equal(c['estimates'][10:13],c['estimates'][:3])
                    assert not c['bypass_fast'].any() and not c['bypass_em100'].any()
                ids=resamples[regime];score=expanded(10*np.log10(error.sum(1)/energy.sum()))
                boot=expanded(10*np.log10(error[:,ids].sum(-1)/energy[ids].sum(-1)[None]))
                nmse=error/energy[None];severe=expanded((nmse>1).mean(1));p95=expanded(np.quantile(nmse,.95,axis=1));p99=expanded(np.quantile(nmse,.99,axis=1))
                methods=summarize(score,boot)
                for name,i in NI.items():methods[name].update(severe_rate=float(severe[i]),case_nmse_p95=float(p95[i]),case_nmse_p99=float(p99[i]))
                gains={arm:dict(gain_db=float(score[NI['original_mean']]-score[NI[arm]]),
                  ci95_db=np.quantile(boot[NI['original_mean']]-boot[NI[arm]],[.025,.975]).tolist(),positive_means='bypass improves original') for arm in ARMS}
                versus_off={arm:dict(penalty_db=float(score[NI[arm]]-score[NI['off_mean']]),
                  ci95_db=np.quantile(boot[NI[arm]]-boot[NI['off_mean']],[.025,.975]).tolist(),positive_means='bypass is worse than always filter-off') for arm in ARMS}
                rows.append(dict(cell=path.stem,regime=regime,rsr=rsr,snr=snr,cases=64,methods=methods,gains_vs_original=gains,
                  penalties_vs_always_off=versus_off,
                  fast_bypass_fraction=float(c['bypass_fast'].mean()),em100_bypass_fraction=float(c['bypass_em100'].mean())))
                all_scores.append(score);all_boots.append(boot);all_errors.append(error);all_energy.append(energy);all_nmse.append(nmse)
                dispatch_max=max(dispatch_max,float(c['actual_dispatch_max_abs_error']));hashes[path.name]=sha(path)
    scores=np.stack(all_scores);boots=np.stack(all_boots);aggregates={}
    for name,mask in [('ordinary',np.array([r['snr']<=20 for r in rows])),('high_snr',np.array([r['snr']>20 for r in rows])),('all',np.ones(len(rows),bool))]:
        point=scores[mask].mean(0);b=boots[mask].mean(0);methods=summarize(point,b)
        e=np.concatenate([e for e,m in zip(all_errors,mask) if m],axis=1);en=np.concatenate([e for e,m in zip(all_energy,mask) if m]);nm=np.concatenate([e for e,m in zip(all_nmse,mask) if m],axis=1)
        pooled=expanded(10*np.log10(e.sum(1)/en.sum()));severe=expanded((nm>1).mean(1));p95=expanded(np.quantile(nm,.95,axis=1));p99=expanded(np.quantile(nm,.99,axis=1))
        for method,i in NI.items():methods[method].update(pooled_ratio_of_sums_db=float(pooled[i]),severe_rate=float(severe[i]),case_nmse_p95=float(p95[i]),case_nmse_p99=float(p99[i]))
        contrasts={arm:dict(gain_db=float(point[NI['original_mean']]-point[NI[arm]]),
          pointwise_ci95_db=np.quantile(b[NI['original_mean']]-b[NI[arm]],[.025,.975]).tolist(),
          simultaneous95_ci_db=np.quantile(b[NI['original_mean']]-b[NI[arm]],[.0125,.9875]).tolist()) for arm in ARMS}
        aggregates[name]=dict(cells=int(mask.sum()),methods=methods,gains_vs_original=contrasts)
    negative={arm:dict(point_estimate_cells=[r['cell'] for r in rows if r['gains_vs_original'][arm]['gain_db']<0],
      pointwise95_harm_cells=[r['cell'] for r in rows if r['gains_vs_original'][arm]['ci95_db'][1]<0],
      worst_cell=min(rows,key=lambda r:r['gains_vs_original'][arm]['gain_db'])['cell'],
      minimum_gain_db=min(r['gains_vs_original'][arm]['gain_db'] for r in rows)) for arm in ARMS}
    retained_filter_penalty={arm:dict(point_estimate_cells=[r['cell'] for r in rows if r['penalties_vs_always_off'][arm]['penalty_db']>0],
      pointwise95_cells=[r['cell'] for r in rows if r['penalties_vs_always_off'][arm]['ci95_db'][0]>0],
      worst_cell=max(rows,key=lambda r:r['penalties_vs_always_off'][arm]['penalty_db'])['cell'],
      maximum_penalty_db=max(r['penalties_vs_always_off'][arm]['penalty_db'] for r in rows)) for arm in ARMS}
    residual=[]
    for i,row in enumerate(rows):
        if row['snr']!=60:continue
        diff=scores[i,NI['off_mean']]-scores[i,NI['em100']];b=boots[i,NI['off_mean']]-boots[i,NI['em100']]
        residual.append(dict(regime=row['regime'],rsr=row['rsr'],off10_nmse_db=float(scores[i,NI['off_mean']]),em100_nmse_db=float(scores[i,NI['em100']]),
          em100_gain_vs_off10_db=float(diff),ci95_db=np.quantile(b,[.025,.975]).tolist()))
    result=dict(selection_sha256=freeze,protocol_sha256=selection['protocol_sha256'],policies=selection['policies'],
      unique_test_anchors=64*3*3,condition_cases=len(rows)*64,bootstrap_index_clusters=64*3,
      scoring='Equal-cell mean of dB ratio-of-sums channelNMSE. Learned groups average three fit scores, never predictions. Group p95/p99 are averages of the per-fit case-NMSE quantiles.',
      primary='high_snr (SNR>20), two gains vs original. Ordinary range and all-range summaries are descriptive.',
      bootstrap=dict(samples=BOOT,seed=20261017660,unit='Whole-anchor case index, shared across all SNR/reference within each regime and paired methods/fits',
        scope='Conditional on frozen checkpoints, policy selection and listed condition grid; not training-population or policy-selection uncertainty',
        multiplicity='Two high-SNR aggregate contrasts have Bonferroni simultaneous95 intervals. Per-cell intervals are pointwise95 and negative-cell flags are diagnostic, not familywise claims.'),
      ordinary_exact_identity=True,actual_dispatch_max_abs_error=dispatch_max,aggregates=aggregates,negative_cells=negative,
      penalties_vs_always_off=retained_filter_penalty,
      residual_60db=residual,cells=rows,test_hashes=hashes,source_sha256=sha(__file__))
    if (OUT/'timing.json').exists():
        t=json.loads((OUT/'timing.json').read_text());assert t['selection_sha256']==freeze
        result.update(timing=t['methods'],timing_sha256=sha(OUT/'timing.json'))
    write_json(OUT/'summary.json',result)
    np.savez_compressed(OUT/'bootstrap.npz',method_names=np.array(METHODS),cell_names=np.array([r['cell'] for r in rows]),cell_scores=scores,cell_bootstraps=boots)
    figures(result)
    print(json.dumps(dict(policies=result['policies'],high_snr=aggregates['high_snr']['gains_vs_original'],negative_cells=negative,residual_60db=residual)),flush=True)


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style={'original_mean':('#d73a49','Frozen SAHURE','-'),'off_mean':('#8c8c8c','Same weights, filter off','--'),
      'fast_bypass_mean':('#0969da','Calibrated 10-stage bypass','-'),'em100':('#26965c','Always EM100','--'),
      'em100_bypass_mean':('#8a63d2','Calibrated EM100 bypass','-')}
    fig,axes=plt.subplots(3,3,figsize=(13.5,10.3),sharex=True,sharey=True)
    for i,(regime,L) in enumerate(REGIMES):
        for j,rsr in enumerate(RSRS):
            ax=axes[i,j];rr=sorted([r for r in result['cells'] if r['regime']==regime and r['rsr']==rsr],key=lambda r:r['snr'])
            for name,(color,label,ls) in style.items():ax.plot([r['snr'] for r in rr],[r['methods'][name]['nmse_db'] for r in rr],color=color,ls=ls,label=label,marker='.',lw=1.6)
            ax.axvline(20,color='.5',ls=':',lw=.8)
            ax.set_title(f'{regime}, RSR {rsr:g} dB');ax.grid(alpha=.2)
            if i==2:ax.set_xlabel('Nominal SNR (dB)')
            if j==0:ax.set_ylabel('Channel NMSE (dB; lower is better)')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Fresh test of validation-calibrated high-SNR safeguards\nSame channels and standardized noise across SNR; original output preserved through 20 dB')
    fig.tight_layout(rect=(0,.065,1,.95));fig.savefig(OUT/'bypass_nmse.png',dpi=180);fig.savefig(OUT/'bypass_nmse.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
    for ax,(regime,L) in zip(axes,REGIMES):
        for rsr in RSRS:
            rr=sorted([r for r in result['cells'] if r['regime']==regime and r['rsr']==rsr],key=lambda r:r['snr'])
            x=[r['snr'] for r in rr];c=[r['gains_vs_original']['fast_bypass_mean'] for r in rr]
            ax.plot(x,[r['gain_db'] for r in c],marker='.',label=f'RSR {rsr:g} dB')
            ax.fill_between(x,[r['ci95_db'][0] for r in c],[r['ci95_db'][1] for r in c],alpha=.12)
        ax.axhline(0,color='k',lw=.7);ax.axvline(20,color='.5',ls=':');ax.set_title(regime);ax.set_xlabel('Nominal SNR (dB)');ax.grid(alpha=.2)
    axes[0].set_ylabel('10-stage bypass improvement over original (dB)');axes[-1].legend()
    fig.suptitle('Fixed development-selected policy; bands are pointwise conditional 95% intervals')
    fig.tight_layout();fig.savefig(OUT/'bypass_gain.png',dpi=180);fig.savefig(OUT/'bypass_gain.svg');plt.close(fig)

if __name__=='__main__':main()
