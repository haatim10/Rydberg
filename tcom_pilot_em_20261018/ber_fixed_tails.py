"""Prespecified descriptive BER of actual fixed1/2/5-step EM continuations.

This supplement cannot alter checkpoint/tail/sparse selection or the four core
primary contrasts. It consumes the exact data saved by the core BER module.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import ber as core
from .training_protocol import sha,write_json
from tcom_followups_20261017.ber_detector import detect

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'results/tcom_pilot_em_20261018/ber_fixed_tails'
FAMILIES=('old','context30');TAILS=(0,1,2,5)
NAMES=tuple(f'{family}_{seed}_tail{tail}' for family in FAMILIES for tail in TAILS for seed in core.SEEDS)
ADDITIONAL=tuple(name for name in NAMES if not name.endswith('_tail0'))
GROUPS={f'{family}_tail{tail}_mean':tuple(f'{family}_{seed}_tail{tail}' for seed in core.SEEDS)
    for family in FAMILIES for tail in TAILS}


def freeze():
    paths=[Path(__file__),Path(core.__file__),REPO/'tcom_followups_20261017/ber_detector.py']
    spec=dict(version=1,name='Descriptive fixed-tail BER supplement',
        purpose='Compare actual EM lengths1,2,5 even when heldout validation selects0; no changes to primary contrasts or frozen selection.',
        additional_detector_methods=list(ADDITIONAL),baseline_methods=[n for n in NAMES if n.endswith('_tail0')],
        families=list(FAMILIES),tails=list(TAILS),model_seeds=list(core.SEEDS),
        pilot_seed=2026101804,data_seed=2026101805,cases_per_cell=128,vectors_per_case=256,
        cells=[list(c) for c in core.cells()],core_protocol_sha256=sha(core.OUT/'protocol.json'),
        data_policy='Read only the exact coreBER saved magnitudes/reference/variance and separately saved transmitted bits. Never regenerate or select new data for this supplement.',
        duplicates='Bitwise-identical saved channel arrays may reuse core or earlier supplement error counts. Every reuse is recorded and verified against exact channel equality.',
        uncertainty='Descriptive pointwise95% paired coupled-block bootstrap,20000 resamples seed2026101894, same128 local case weights across SNRs within each ofthree independent regimes. Fixed training fits; no ensemble and no primary inference upgrade.',
        effect='Reference tail0 BER minus fixed-tail BER, positive denotes lower errors after continuation.',
        selection='Every prespecified family/tail/SNR/path cell reported. No supplement output may influence checkpoints, selected tails, sparse budgets or the four primary contrasts.',
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in paths})
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None)
        if old!=spec:raise RuntimeError('Fixed-tail BER protocol/source changed after freeze')
    else:
        assert not list((core.SOURCE/'test').glob('*.npz')),'Freeze the supplement before fresh test generation'
        write_json(path,dict(**spec,frozen_unix=time.time()))
    return sha(path)


def fingerprint(g):return hashlib.sha256(g.tobytes()).hexdigest()


def evaluate(digest,available_only=False):
    selection_digest=core.verify_selection();complete=True
    for regime,snr in core.cells():
        label=f'{regime}_snr{snr}';result=OUT/'cells'/(label+'.npz')
        corepath=core.OUT/'cells'/(label+'.npz');sourcepath=core.SOURCE/'test'/(label+'.npz')
        if not corepath.exists() and available_only:complete=False;continue
        if result.exists():
            a=core.load(result)
            assert str(a['protocol_sha256'])==digest and str(a['core_cell_sha256'])==sha(corepath)
            assert str(a['source_sha256'])==sha(sourcepath)
            print('Fixed-tail BER cached',label,flush=True);continue
        a=core.load(sourcepath);base=core.load(corepath)
        assert str(a['selection_sha256'])==selection_digest==str(base['selection_sha256'])
        assert str(base['source_sha256'])==sha(sourcepath)
        assert str(base['protocol_sha256'])==sha(core.OUT/'protocol.json')
        datapath=core.OUT/'data'/(label+'.npz');truthpath=core.OUT/'truth'/(label+'.npz')
        assert str(base['data_sha256'])==sha(datapath) and str(base['truth_sha256'])==sha(truthpath)
        data=core.load(datapath);truth=core.load(truthpath)
        source_names=list(map(str,a['method_names']));base_names=list(map(str,base['method_names']))
        g=a['estimates'][[source_names.index(name) for name in NAMES]]
        cache={}
        for mi,name in enumerate(base_names):
            channel=a['truth_g'] if name=='perfect_csi' else a['estimates'][source_names.index(name)]
            cache[fingerprint(channel)]=(channel,base['errors_per_user'][mi],'core/'+name)
        errors=np.zeros((len(NAMES),core.CASES,3),dtype=np.int64)
        seconds=np.zeros((len(NAMES),core.CASES));duplicate_of=[];evaluations=0
        for mi,name in enumerate(NAMES):
            key=fingerprint(g[mi]);cached=cache.get(key)
            if cached is not None:
                assert np.array_equal(cached[0],g[mi]);errors[mi]=cached[1];duplicate_of.append(cached[2]);continue
            duplicate_of.append('');evaluations+=1
            for case in range(core.CASES):
                tick=time.perf_counter();decoded=detect(data['z'][case],g[mi,case],data['b'][case],float(data['sigma2'][case]))
                seconds[mi,case]=time.perf_counter()-tick
                errors[mi,case]=np.count_nonzero(decoded!=truth['bits'][case],axis=(0,2))
            cache[key]=(g[mi],errors[mi].copy(),'fixed/'+name)
        core.save(result,method_names=np.array(NAMES),errors_per_user=errors,bits_per_user=base['bits_per_user'],
            duplicate_of=np.array(duplicate_of),detector_wall_seconds=seconds,
            indices=base['indices'],protocol_sha256=digest,selection_sha256=selection_digest,
            source_sha256=sha(sourcepath),core_cell_sha256=sha(corepath),data_sha256=sha(datapath),truth_sha256=sha(truthpath))
        print(json.dumps(dict(cell=label,new_detector_methods=evaluations,reused_methods=len(NAMES)-evaluations)),flush=True)
    return complete


def analyze(digest):
    selection_digest=core.verify_selection();rng=np.random.default_rng(core.BOOTSTRAP_SEED)
    weights={r:rng.multinomial(core.CASES,np.full(core.CASES,1/core.CASES),size=core.BOOTSTRAPS).astype(float) for r in core.REGIMES}
    names=NAMES+tuple(GROUPS);rates=[];replicates=[];all_counts=[];rows=[]
    bits_per_world=3*2*core.VECTORS
    for regime,snr in core.cells():
        a=core.load(OUT/'cells'/f'{regime}_snr{snr}.npz')
        assert tuple(map(str,a['method_names']))==NAMES and str(a['protocol_sha256'])==digest
        assert str(a['selection_sha256'])==selection_digest
        counts=a['errors_per_user'].sum(-1);all_counts.append(counts);perworld=counts/bits_per_world
        grouped=np.stack([perworld[[NAMES.index(n) for n in parts]].mean(0) for parts in GROUPS.values()])
        perworld=np.concatenate((perworld,grouped));samples=perworld@weights[regime].T/core.CASES
        rates.append(perworld.mean(-1));replicates.append(samples)
        for mi,name in enumerate(names):
            row=dict(regime=regime,snr_db=snr,method=name,ber=float(perworld[mi].mean()),bootstrap95=core.interval(samples[mi]))
            if name in GROUPS:row['errors_per_training_seed']={n:int(counts[NAMES.index(n)].sum()) for n in GROUPS[name]}
            else:row['bit_errors']=int(counts[mi].sum())
            rows.append(row)
    rates,replicates,all_counts=map(np.stack,(rates,replicates,all_counts))
    def aggregate(indices):
        methods={};contrasts={}
        for mi,name in enumerate(names):
            row=dict(ber=float(rates[indices,mi].mean()),bootstrap95=core.interval(replicates[indices,mi].mean(0)),
                bits_per_fit=len(indices)*core.CASES*bits_per_world)
            if name in GROUPS:row['errors_per_training_seed']={n:int(all_counts[indices,NAMES.index(n)].sum()) for n in GROUPS[name]}
            else:row['bit_errors']=int(all_counts[indices,mi].sum())
            methods[name]=row
        for family in FAMILIES:
            reference=f'{family}_tail0_mean';ri=names.index(reference)
            for tail in (1,2,5):
                candidate=f'{family}_tail{tail}_mean';ci=names.index(candidate)
                delta=rates[indices,ri]-rates[indices,ci]
                samples=(replicates[indices,ri]-replicates[indices,ci]).mean(0)
                contrasts[f'{family}_tail{tail}']=dict(reference=reference,candidate=candidate,
                    ber_reduction=float(delta.mean()),bootstrap95=core.interval(samples),
                    improved_cells=int((delta>0).sum()),harmed_cells=int((delta<0).sum()),tied_cells=int((delta==0).sum()),total_cells=len(indices))
        return dict(methods=methods,fixed_tail_vs_zero=contrasts)
    summary=dict(protocol_sha256=digest,core_protocol_sha256=sha(core.OUT/'protocol.json'),selection_sha256=selection_digest,
        overall=aggregate(np.arange(len(core.cells()))),
        by_snr={str(s):aggregate(np.array([i for i,c in enumerate(core.cells()) if c[1]==s])) for s in core.SNRS},
        by_regime={r:aggregate(np.array([i for i,c in enumerate(core.cells()) if c[0]==r])) for r in core.REGIMES},
        cells=rows,interpretation='Prespecified descriptive supplement; pointwise intervals, no multiplicity-adjusted claim and no effect on the four primary contrasts or any selection. Group rates average three separate model fits; no prediction ensemble.',
        zero_error_policy='Zero observed errors are unresolved rare-event performance, not zeroBER or a bound.',
        uncertainty='20000 paired coupled-block resamples conditional on fixed trained fits;1920 channel realizations in384 clusters. Samecase weights across SNR and allmethods within eachregime.')
    write_json(OUT/'summary.json',summary);plot(rows)
    print(json.dumps(summary['overall']['fixed_tail_vs_zero']),flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={0:'#555555',1:'#009e73',2:'#0072b2',5:'#d55e00'}
    fig,axes=plt.subplots(2,3,figsize=(12.8,7.4),sharex=True,sharey=True);floor=1e-6
    for ri,family in enumerate(FAMILIES):
        for ci,regime in enumerate(core.REGIMES):
            ax=axes[ri,ci]
            for tail in TAILS:
                method=f'{family}_tail{tail}_mean';items=[r for r in rows if r['regime']==regime and r['method']==method]
                values=np.array([r['ber'] for r in items]);positive=values>0
                ax.plot(core.SNRS,np.where(positive,values,np.nan),marker='o',markersize=3.5,color=colors[tail],label=f'{tail} extra EM steps')
                ax.scatter(np.array(core.SNRS)[~positive],np.full((~positive).sum(),floor),marker='v',color=colors[tail],s=25)
            ax.set_yscale('log');ax.set_ylim(.65e-6,.3);ax.set_xticks(core.SNRS);ax.grid(alpha=.2,which='both')
            if ri==0:ax.set_title({'L2':'2 paths/user','L4':'4 paths/user','random3to7':'3–7 paths/user'}[regime])
            if ci==0:ax.set_ylabel(('Original learned' if family=='old' else 'Fresh30-pilot training')+'\nUncoded BER')
            if ri==1:ax.set_xlabel('SNR (dB)')
    handles,labels=axes[0,0].get_legend_handles_labels();handles.append(Line2D([0],[0],color='#555555',marker='v',linestyle='none'));labels.append('Zero errors (marker only)')
    fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=8.3,bbox_to_anchor=(.5,.025))
    fig.suptitle('Fixed short EM continuation: same data and exact magnitude-domain detector',fontsize=11)
    fig.text(.5,.008,'Descriptive comparisons, mean of3 separate fits. Zero-error markers at10⁻⁶ are not estimates or bounds.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.97));fig.savefig(OUT/'ber_fixed_tails.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'ber_fixed_tails.pdf',bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');p.add_argument('--available-only',action='store_true');p.add_argument('--analyze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);digest=freeze()
    if args.freeze_only:return
    if not args.analyze_only and not evaluate(digest,args.available_only):return
    analyze(digest)


if __name__=='__main__':main()
