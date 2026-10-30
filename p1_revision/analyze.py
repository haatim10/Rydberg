"""Audit and summarize frozen reviewer experiments, with paired uncertainty."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from p1_revision.reuse import paired_summary, write_json, write_csv

REPO=Path(__file__).resolve().parents[1]
DEFAULT=REPO/'results/p1_revision/campaign'
LABELS=dict(em='EM-GS',hs='HS-GS',rician_gd='Rician GD',
            rician_hankel='Rician GD + Hankel',em_smooth='EM-GS + shrinkage',
            xu_linearized='Xu 1D linearized LS')
METHODS=tuple(LABELS)


def paired_gain_change(em0,hs0,em1,hs1,rng,n_boot=4000):
    """Change in the ratio-of-sums HS gain, pairing worlds across conditions."""
    arrays=[np.asarray(a) for a in (em0,hs0,em1,hs1)]
    if len({a.shape for a in arrays})!=1 or arrays[0].ndim!=1:
        raise ValueError('same one-dimensional paired shape required')
    if not all(np.all(np.isfinite(a)) and np.all(a>0) for a in arrays):
        raise ValueError('positive finite errors required')
    indices=rng.integers(len(arrays[0]),size=(n_boot,len(arrays[0])))
    a,b,c,d=[x[indices].sum(axis=1) for x in arrays]
    samples=10*np.log10(c/d)-10*np.log10(a/b)
    point=10*np.log10(arrays[2].sum()/arrays[3].sum())-10*np.log10(arrays[0].sum()/arrays[1].sum())
    lo,hi=np.quantile(samples,[.025,.975])
    return dict(gain_change_db=float(point),lo=float(lo),hi=float(hi))


def read_cell(root,c,fp):
    files=sorted((root/'raw'/c['tag']).glob('*.npz'))
    parts=[];diags=[]
    for p in files:
        with np.load(p,allow_pickle=False) as f:
            if str(f['fingerprint'])!=fp:
                raise ValueError('mixed source fingerprints')
            parts.append({k:f[k].copy() for k in f.files if k not in ('count','fingerprint','diagnostics_json')})
            diags.extend(json.loads(str(f['diagnostics_json'])))
    if not parts:
        raise ValueError(f'no data: {c["tag"]}')
    data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    expected=np.arange(c['seed_base'],c['seed_base']+c['trials'])
    np.testing.assert_array_equal(data['trial'],expected)
    if len(diags)!=len(expected):raise ValueError('missing diagnostics')
    if not all(np.all(np.isfinite(v)) for v in data.values()):
        raise ValueError('nonfinite raw values')
    return data,diags,files


def analyze(root=DEFAULT, make_plots=True):
    root=Path(root)
    completion=json.loads((root/'COMPLETE.json').read_text())
    if not completion['complete']:raise ValueError('campaign is incomplete')
    manifest=json.loads((root/'manifest.json').read_text())
    rng=np.random.default_rng(20261007);rows=[];raw={};hashes={}
    for c in manifest['cells']:
        data,diags,files=read_cell(root,c,manifest['source_fingerprint']);raw[c['tag']]=data
        for p in files:hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
        for method in METHODS:
            if 'num_'+method not in data:continue
            stats=paired_summary(data['num_em'],data['num_'+method],data['den'],rng)
            against_hs=paired_summary(data['num_hs'],data['num_'+method],data['den'],rng)
            if c['kind']=='baselines':
                rank=[d['methods'][method]['selected_rank'] for d in diags]
                threshold=[d['methods'][method]['selected_threshold'] for d in diags]
                times=[d['methods'][method]['seconds_select']+d['methods'][method]['seconds_final_fit'] for d in diags]
                fallback=(np.mean(np.asarray(threshold)==0) if method=='em_smooth'
                          else np.mean(np.asarray(rank)==(c['N']+1)//2) if method in ('hs','rician_hankel') else None)
            else:
                rank=data['rank'] if method=='hs' else [None]*len(data['trial'])
                threshold=[None]*len(data['trial'])
                fallback=float(np.mean(data['rank']==(c['N']+1)//2)) if method=='hs' else None
                times=data['seconds_hs_total'] if method=='hs' else data['seconds_em']
            row={k:c.get(k) for k in ('kind','tag','N','K','P','snr','rsr','L','val_fraction','position_std')}
            row.update(method=method,n=stats['n'],nmse_db=stats['hs_nmse_db'],
                nmse_lo=stats['hs_nmse_db_lo'],nmse_hi=stats['hs_nmse_db_hi'],
                gain_vs_em_db=stats['ratio_gain_db'],gain_vs_em_lo=stats['ratio_gain_db_lo'],gain_vs_em_hi=stats['ratio_gain_db_hi'],
                median_gain_vs_em_db=stats['median_gain_db'],
                gain_vs_hs_db=against_hs['ratio_gain_db'],gain_vs_hs_lo=against_hs['ratio_gain_db_lo'],gain_vs_hs_hi=against_hs['ratio_gain_db_hi'],
                median_gain_vs_hs_db=against_hs['median_gain_db'],fallback_fraction=fallback,
                selected_rank_median=float(np.median(rank)) if all(x is not None for x in rank) else None,
                selected_rank_mean=float(np.mean(rank)) if all(x is not None for x in rank) else None,
                selected_threshold_median=float(np.median(threshold)) if all(x is not None for x in threshold) else None,
                true_rho_median=float(np.median(data['rho_true'])),
                em_proxy_rho_median=float(np.median(data['rho_em'])),
                median_parallel_workload_seconds=float(np.median(times)))
            rows.append(row)
    # The split study reuses the identical channel/pilots/noise at every fraction.
    split_rows=[]
    for c in manifest['cells']:
        if c['kind']!='holdout':continue
        ref=next(x for x in manifest['cells'] if x['kind']=='holdout' and x['P']==c['P'] and x['snr']==c['snr'] and x['val_fraction']==.3)
        a,b=raw[ref['tag']],raw[c['tag']]
        for k in ('trial','den','num_em'):np.testing.assert_array_equal(a[k],b[k])
        stat=paired_summary(a['num_hs'],b['num_hs'],b['den'],rng)
        split_rows.append(dict(P=c['P'],snr=c['snr'],fraction=c['val_fraction'],n=c['trials'],
            gain_vs_30pct_db=stat['ratio_gain_db'],lo=stat['ratio_gain_db_lo'],hi=stat['ratio_gain_db_hi']))
    mismatch_rows=[]
    for c in manifest['cells']:
        if c['kind']!='mismatch':continue
        ref=next(x for x in manifest['cells'] if x['kind']=='mismatch' and x['L']==c['L'] and x['position_std']==0)
        a,b=raw[ref['tag']],raw[c['tag']]
        np.testing.assert_array_equal(a['trial'],b['trial'])
        mismatch_rows.append(dict(L=c['L'],position_std=c['position_std'],n=c['trials'],
            **paired_gain_change(a['num_em'],a['num_hs'],b['num_em'],b['num_hs'],rng)))
    out=root/'analysis';out.mkdir(exist_ok=True)
    write_csv(out/'summary.csv',rows);write_csv(out/'split_comparison.csv',split_rows)
    write_csv(out/'mismatch_comparison.csv',mismatch_rows)
    result=dict(manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        source_fingerprint=manifest['source_fingerprint'],analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_dependencies_sha256={'p1_revision/reuse.py':hashlib.sha256((REPO/'p1_revision/reuse.py').read_bytes()).hexdigest()},
        condition_trials=sum(c['trials'] for c in manifest['cells']),
        independent_base_worlds=sum(c['trials'] for i,c in enumerate(manifest['cells']) if c['seed_base'] not in {x['seed_base'] for x in manifest['cells'][:i]}),
        interval='4000 paired world bootstrap resamples; pointwise percentile 95%',
        timing_note='campaign times are under parallel load, not serial algorithm speed benchmarks',
        records=rows,split_comparison=split_rows,mismatch_comparison=mismatch_rows,raw_sha256=hashes)
    write_json(out/'summary.json',result)
    if make_plots:
        plots(rows,out)
    write_json(out/'audit.json',dict(passed=True,cells=len(manifest['cells']),
        unique_condition_trial_pairs=sum(c['trials'] for c in manifest['cells']),
        duplicate_condition_trials=0,paired_split_worlds_verified=True,
        complete_before_analysis=True,source_fingerprint=manifest['source_fingerprint']))
    return result


def plots(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    colors=dict(zip(METHODS,['#555555','#d62728','#9467bd','#1f77b4','#2ca02c','#a07828']))
    def save(fig,name):
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
        plt.close(fig)
    def series(ax,rs,x,key,lo,hi,label,color):
        rs=sorted(rs,key=lambda r:r[x]);xx=[r[x] for r in rs];yy=np.array([r[key] for r in rs])
        lower=np.array([r[lo] for r in rs]);upper=np.array([r[hi] for r in rs])
        # A percentile interval need not contain the original point estimate.
        # Draw the actual endpoints rather than clipping asymmetric lengths.
        ax.plot(xx,yy,marker='o',markersize=4,linewidth=1.5,label=label,color=color)
        ax.vlines(xx,lower,upper,color=color,linewidth=1)
        ax.plot(xx,lower,linestyle='none',marker='_',markersize=5,color=color)
        ax.plot(xx,upper,linestyle='none',marker='_',markersize=5,color=color)
    fig,axs=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for ax,snr in zip(axs,(0.,10.)):
        for method in METHODS:
            rs=[r for r in rows if r['kind']=='baselines' and r['snr']==snr and r['method']==method]
            series(ax,rs,'L','nmse_db','nmse_lo','nmse_hi',LABELS[method],colors[method])
        ax.set(title=f'SNR = {snr:g} dB',xlabel='Paths per user',xticks=[3,7,14]);ax.grid(alpha=.2)
    axs[0].set_ylabel('NMSE (dB)');axs[0].legend(fontsize=7.6,loc='best')
    fig.suptitle('Structured comparisons: N=32, K=3, P=30, RSR=12 dB',fontsize=11)
    fig.tight_layout();save(fig,'structured_baselines')
    fig,axs=plt.subplots(2,2,figsize=(8,6),sharex=True)
    for ax,(p,snr) in zip(axs.flat,((p,s) for p in (10,30) for s in (0.,10.))):
        rs=[dict(r,fraction_percent=100*r['val_fraction']) for r in rows if r['kind']=='holdout' and r['P']==p and r['snr']==snr and r['method']=='hs']
        series(ax,rs,'fraction_percent','gain_vs_em_db','gain_vs_em_lo','gain_vs_em_hi','HS-GS gain',colors['hs'])
        ax.axhline(0,color='gray',linestyle=':',linewidth=.8)
        ax.set(title=f'P={p}, SNR={snr:g} dB',xticks=[10,20,30,40]);ax.grid(alpha=.2)
    for ax in axs[-1]:ax.set_xlabel('Pilots held out during rank selection (%)')
    for ax in axs[:,0]:ax.set_ylabel('HS-GS gain over EM-GS (dB)')
    fig.suptitle('All pilots are reused for the final fit',fontsize=11)
    fig.tight_layout();save(fig,'heldout_fraction')
    fig,axs=plt.subplots(1,2,figsize=(8,3.5),sharey=True)
    for ax,l in zip(axs,(3,12)):
        for method in ('em','hs'):
            rs=[r for r in rows if r['kind']=='mismatch' and r['L']==l and r['method']==method]
            series(ax,rs,'position_std','nmse_db','nmse_lo','nmse_hi',LABELS[method],colors[method])
        ax.set(title=f'L={l} paths per user',xlabel=r'Position-error standard deviation / $\lambda$',xticks=[0,.005,.01,.02]);ax.grid(alpha=.2);ax.legend()
    axs[0].set_ylabel('NMSE (dB)')
    fig.suptitle('Position mismatch: N=32, P=30, SNR=5 dB, calibrated reference',fontsize=10)
    fig.tight_layout();save(fig,'position_mismatch')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    args=ap.parse_args();result=analyze(args.root)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','raw_sha256','split_comparison')},indent=2))
