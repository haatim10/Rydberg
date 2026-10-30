"""World-paired analyses. Mask replicates are never counted as independent worlds."""
import json,hashlib
from pathlib import Path
import numpy as np
from p1_followup.run import ROOT,OUT,CELLS,write
from p1_revision.analyze import read_cell

def compare(ref,err,den,rng):
    ref,err,den=map(np.asarray,(ref,err,den));idx=rng.integers(len(err),size=(4000,len(err)))
    samples=10*np.log10(ref[idx].sum(1)/err[idx].sum(1))
    lo,hi=np.quantile(samples,[.025,.975])
    return dict(n=len(err),gain=float(10*np.log10(ref.sum()/err.sum())),lo=float(lo),hi=float(hi),nmse=float(10*np.log10(err.sum()/den.sum())))

def historical():
    root=ROOT/'results/p1_revision/campaign';m=json.loads((root/'manifest.json').read_text());rows=[]
    for c in m['cells']:
        if c['kind']!='baselines':continue
        data,diags,_=read_cell(root,c,m['source_fingerprint'])
        rank=np.array([d['methods']['hs']['selected_rank'] for d in diags]);true=np.minimum(data['L_k'],16)
        assert np.all(true==c['L'])
        gain=10*np.log10(data['num_em']/data['num_hs'])
        groups={}
        for name,mask in [('below',rank<c['L']),('equal',rank==c['L']),('above',rank>c['L'])]:
            groups[name]=dict(n=int(mask.sum()),harm_fraction=float(np.mean(gain[mask]<0)) if mask.any() else None,median_gain=float(np.median(gain[mask])) if mask.any() else None)
        rows.append(dict(L=c['L'],snr=c['snr'],n=len(rank),median_selected=float(np.median(rank)),harm_fraction=float(np.mean(gain<0)),groups=groups))
    write(OUT/'analysis/historical_rank_harm.json',dict(source='Existing 2400 baseline worlds, not new trials',rows=rows));return rows

def main():
    assert json.loads((OUT/'COMPLETE.json').read_text())['complete']
    protocol=json.loads((OUT/'protocol.json').read_text());rng=np.random.default_rng(20261008)
    rows=[];splits=[];raw=[];checks=[]
    for c in CELLS:
        worlds=[json.loads((OUT/'raw'/f'{c["seed"]+i}.json').read_text()) for i in range(c['n'])]
        for i,w in enumerate(worlds):
            assert w['trial']==c['seed']+i and w['source_sha256']==protocol['source_sha256']
            assert w['L']==c['L'] and w['snr']==c['snr'] and len(w['rows'])==11
            for r in w['rows']:
                assert len(set(r['heldout']))==9 and all(0<=v<30 for v in r['heldout'])
                assert np.isfinite(r['error']) and r['error']>0
                candidates=list(range(1,17)) if r['kind']=='hard' else [0,.05,.1,.2,.4,.8,1.6]
                assert r['selected']==candidates[int(np.argmin(r['scores']))]
                if r['selected']==(16 if r['kind']=='hard' else 0):
                    assert abs(r['error']-w['em_error'])<1e-10*max(1,w['em_error'])
        raw+=worlds
        den=np.array([w['den'] for w in worlds]);em=np.array([w['em_error'] for w in worlds])
        def values(kind,q,split,key='error'):
            return np.array([next(r[key] for r in w['rows'] if r['kind']==kind and r['sweeps']==q and r['split']==split) for w in worlds])
        for kind in ('hard','smooth'):
            ref=values(kind,4,0)
            for q in (1,2,4,8):
                err=values(kind,q,0)
                rows.append(dict(L=c['L'],snr=c['snr'],kind=kind,sweeps=q,vs_four=compare(ref,err,den,rng),vs_em=compare(em,err,den,rng)))
        ref=values('hard',4,0)
        error=np.stack([values('hard',4,m) for m in (1,2,3)],axis=1)
        ranks=np.stack([values('hard',4,m,'selected') for m in (1,2,3)],axis=1)
        allranks=np.column_stack((values('hard',4,0,'selected'),ranks))
        # Average errors across random masks within world FIRST, then bootstrap worlds.
        splits.append(dict(L=c['L'],snr=c['snr'],random_mean_vs_fixed=compare(ref,error.mean(1),den,rng),
          any_rank_change_fraction=float(np.mean(np.ptp(allranks,axis=1)>0)),
          random_vs_fixed_rank_disagreement=float(np.mean(ranks!=allranks[:,:1])),
          median_within_world_error_range_db=float(np.median(np.ptp(10*np.log10(np.column_stack((ref,error))),axis=1))),
          random_only_median_error_range_db=float(np.median(np.ptp(10*np.log10(error),axis=1))),
          masks=3,worlds=len(worlds)))
    old=historical()
    write(OUT/'analysis/summary.json',dict(sweeps=rows,splits=splits,historical=old,interval='4000 paired world bootstrap percentile resamples; pointwise 95%; random-mask errors averaged within world before resampling',analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    write(OUT/'analysis/audit.json',dict(passed=True,worlds=len(raw),estimates=sum(len(w['rows'])+1 for w in raw),rank_argmin_verified=True,full_rank_and_zero_filter_bypass_verified=True,source_hashes_verified=True,independent_unit='world, not random partition',raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'raw').glob('*.json'))}))
    plot(rows,splits,old)
    print(json.dumps(dict(splits=splits,historical=old),indent=2))

def plot(rows,splits,old):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(10,4.6),sharex=True)
    for ax,(snr,l) in zip(axes.flat,((s,l) for s in (0,10) for l in (3,7,14))):
        for kind,label,color in [('hard','Hard rank','#cf443c'),('smooth','Smooth filter','#197d55')]:
            rs=[r for r in rows if r['L']==l and r['snr']==snr and r['kind']==kind]
            x=[r['sweeps'] for r in rs];y=[r['vs_em']['gain'] for r in rs]
            ax.plot(x,y,'o-',label=label,color=color,markersize=3)
            ax.vlines(x,[r['vs_em']['lo'] for r in rs],[r['vs_em']['hi'] for r in rs],color=color,lw=1)
        ax.axhline(0,color='gray',lw=.6);ax.set_title(f'L={l}, SNR={snr} dB');ax.set_xticks([1,2,4,8]);ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Sweeps per application')
    for ax in axes[:,0]:ax.set_ylabel('Gain over EM-GS (dB)')
    axes[0,0].legend(fontsize=8);fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(OUT/f'analysis/sweep_sensitivity.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    x=np.arange(6);labels=[f'L={r["L"]}\n{r["snr"]} dB' for r in splits]
    rs=[r['random_mean_vs_fixed'] for r in splits]
    axes[0].plot(x,[r['gain'] for r in rs],'o',color='#215a9a')
    axes[0].vlines(x,[r['lo'] for r in rs],[r['hi'] for r in rs],color='#215a9a');axes[0].axhline(0,color='gray',lw=.7)
    axes[0].set_ylabel('Random-split gain over fixed split (dB)');axes[0].set_xticks(x,labels)
    axes[1].bar(x,[100*r['random_vs_fixed_rank_disagreement'] for r in splits],color='#215a9a')
    axes[1].set_ylabel('Random vs. fixed rank disagreement (%)');axes[1].set_xticks(x,labels)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(OUT/f'analysis/split_stability.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':main()
