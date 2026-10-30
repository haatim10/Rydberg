"""Paired channel-world uncertainty and plots for the frozen sparse study."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import write_json
from .sparse import OUT, SNRS, REGIMES


def main():
    rng=np.random.default_rng(2026101717)
    repetitions=2000
    cell_results=[]; bootstraps=[]
    for snr in SNRS:
        for L in REGIMES:
            label='random3-7' if L is None else f'L{L}'
            data=np.load(OUT/f'test_snr{int(snr)}_{label}.npz')
            energy=data['truth_energy']; count=len(energy)
            ids=rng.integers(count,size=(repetitions,count))
            total_energy=energy[ids].sum(1)
            names=['sparse','em100','optshrink']+[f'learned_{s}' for s in (9101,9102,9103)]
            score={name:10*np.log10(data[name+'_error'].sum()/energy.sum()) for name in names}
            score['learned_seed_mean']=np.mean([score[f'learned_{s}'] for s in (9101,9102,9103)])
            bootstrap={name:10*np.log10(data[name+'_error'][ids].sum(1)/total_energy) for name in names}
            bootstrap['learned_seed_mean']=np.mean([bootstrap[f'learned_{s}'] for s in (9101,9102,9103)],axis=0)
            difference=bootstrap['learned_seed_mean']-bootstrap['sparse']
            point=score['learned_seed_mean']-score['sparse']
            cell_results.append(dict(snr_db=snr,path_regime=label,nmse_db=score,
                learned_minus_sparse_db=point,paired_95_percentile_CI_db=np.quantile(difference,[.025,.975]).tolist()))
            bootstraps.append(difference)
    results=dict(bootstrap_repetitions=repetitions,bootstrap_seed=2026101717,
        inference='Channel-world paired bootstrap independently within each of 15 cells; seed-mean learned score, no ensemble. CI conditions on the selected controls and trained models, and excludes training/development variability.',
        sign='negative learned-minus-sparse dB favors the learned estimator',cells=cell_results)
    for label in ['all']+['random3-7' if L is None else f'L{L}' for L in REGIMES]:
        ids=[i for i,row in enumerate(cell_results) if label=='all' or row['path_regime']==label]
        boot=np.mean([bootstraps[i] for i in ids],axis=0)
        results[label]=dict(learned_minus_sparse_db=float(np.mean([cell_results[i]['learned_minus_sparse_db'] for i in ids])),
            paired_95_percentile_CI_db=np.quantile(boot,[.025,.975]).tolist(),
            equal_cell_nmse_db={name:float(np.mean([cell_results[i]['nmse_db'][name] for i in ids]))
                for name in ['sparse','learned_seed_mean','optshrink','em100']})
    write_json(OUT/'paired_bootstrap.json',results)
    colors={'learned_seed_mean':'#bd4c23','sparse':'#168773','optshrink':'#4169ad','em100':'#888888'}
    labels={'learned_seed_mean':'Learned (mean of 3 seeds)','sparse':'Angle sparse EM','optshrink':'Offline OptShrink','em100':'EM100'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True,constrained_layout=True)
    for ax,L in zip(axes,REGIMES):
        label='random3-7' if L is None else f'L{L}'
        rows=[r for r in cell_results if r['path_regime']==label]
        for name in colors:
            ax.plot([r['snr_db'] for r in rows],[r['nmse_db'][name] for r in rows],
                    marker='o',ms=4,color=colors[name],label=labels[name])
        ax.set_title('Random 3–7 paths per user' if L is None else f'{L} paths per user')
        ax.set_xlabel('SNR (dB)');ax.grid(alpha=.22)
    axes[0].set_ylabel('NMSE (dB; lower is better)')
    axes[-1].legend(fontsize=8,loc='upper right')
    fig.suptitle('Exact referenced-magnitude likelihood · N=32, K=3, P=30 · RSR=12 dB',fontsize=11)
    fig.savefig(OUT/'sparse_comparison.png',dpi=180)
    print(json.dumps({k:results[k] for k in ('all','L2','L4','random3-7')},indent=2))


if __name__=='__main__':main()
