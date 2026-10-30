"""Frozen bounded BER study for retrained-model high-SNR safeguards.

Enumerate the uniform64-vector QPSK alphabet once per channel. Each transmitted
vector gets an independent data-noise draw, with common standardized noise
across SNRs and methods. No extra bits or cells are added after seeing errors.
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
from tcom_followups_20261017.ber_detector import detect,hypotheses,verify_detector
from tcom_pilot_em_20261018.training_protocol import sha,write_json

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'results/tcom_retrained_bypass_20261018';OUT=SOURCE/'ber'
SNRS=(0.,5.,10.,15.,20.,22.5,25.,27.5,30.,32.5,35.,37.5,40.,42.5,45.,50.,60.)
RSRS=(5.,12.,20.);REGIMES=('L2','L4','random3to7');SEEDS=(9101,9102,9103)
CASES=64;VECTORS=64;BOOTSTRAPS=20000;BOOTSTRAP_SEED=2026101829;DATA_SEED=2026101823
GROUPS={f'{family}_mean':tuple(f'{family}_{seed}' for seed in SEEDS)
    for family in ('original','off','fast_bypass','em100_bypass')}
USED=tuple(n for members in GROUPS.values() for n in members)+(
    'em100','optshrink','prior_sparse_matched','prior_sparse_50ms','prior_sparse_full','compact_sparse')
NAMES=('perfect_csi',)+USED
CONTRASTS=dict(bypass_vs_original=('original_mean','em100_bypass_mean'),
    bypass_vs_em100=('em100','em100_bypass_mean'),bypass_vs_compact=('compact_sparse','em100_bypass_mean'))


def cells():return [(r,ref,s) for r in REGIMES for ref in RSRS for s in SNRS]

def label(regime,rsr,snr):return f'{regime}_rsr{rsr:g}_snr{snr:g}'

def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}

def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**arrays);temporary.replace(path)


def freeze():
    detector=REPO/'tcom_followups_20261017/ber_detector.py'
    checkpoint_record=json.loads((REPO/'results/tcom_pilot_em_20261018/evaluation/checkpoints.json').read_text())
    selected={name:item for name,item in checkpoint_record['checkpoints'].items() if name.startswith('context30_')}
    assert len(selected)==3
    for item in selected.values():assert sha(REPO/item['path'])==item['sha256']
    spec=dict(version=1,name='Retrained-model bypass: bounded full-grid magnitude-domain BER',
        N=32,K=3,P=30,snrs=list(SNRS),rsrs=list(RSRS),regimes=list(REGIMES),cases_per_cell=CASES,
        test_seed=2026101822,data_seed=DATA_SEED,source_methods=list(USED),additional_reference='perfect_csi',
        checkpoint_record=selected,neural_groups=GROUPS,
        modulation='All64 equally likely joint GrayQPSK vectors once per channel; unit power per user, six bits per vector. This is a stratified uniform-alphabet expectation, not64 independent random symbol selections.',
        data_noise='Independent proper complex Gaussian noise for each transmitted vector and antenna; independent of pilot noise. SeedSequence([2026101823, regime_index, reference_index, local_index]) draws unit-variance noise. Reuse this standard noise across all SNRs and scale by sqrt(sigma2).',
        identical_inputs='Every channel estimate sees exactly the same new data magnitudes, known reference and variance, and exact64-hypothesis Rician ML detector. No coherent data available to any detector.',
        inference='True channel used only to generate data and for explicitly labeled perfectCSI reference. No BER-dependent checkpoint, threshold, sparse choice, cell selection or extra bit allocation.',
        bounds='64 vectors/channel is fixed before tests; report zero errors without interpreting a collapsed bootstrap interval as a rare-event bound. Do not use independent-bit binomial confidence intervals.',
        duplicates='Bitwise-identical saved channel arrays may share detector outputs; record and audit every reuse. Duplicate reuse is not an estimator-runtime measurement.',
        primary_scope='All declared SNRs strictly greater than20dB, equal weighting of reference/path/SNR cells.',
        primary_contrasts=CONTRASTS,improvement_sign='reference BER minus candidate BER',
        descriptive_scopes=dict(ordinary='SNR<=20',transition='20<SNR<=45',stress='SNR in50,60'),
        bootstrap=dict(resamples=BOOTSTRAPS,seed=BOOTSTRAP_SEED,
            rule='Multinomial64-case weights in regime order L2,L4,random3to7, shared across all reference strengths, SNRs, methods and fits within a regime; regimes independent.',
            pointwise_quantiles=[.025,.975],primary_bonferroni6_quantiles=[.05/12,1-.05/12],
            interpretation='Conditional on fixed fitted models and frozen policies. Three fits average metrics, never predictions; they do not triple independent test channels.'),
        source_hashes={str(Path(__file__).relative_to(REPO)):sha(__file__),str(detector.relative_to(REPO)):sha(detector)})
    spec=json.loads(json.dumps(spec));path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None)
        if old!=spec:raise RuntimeError('BER source/protocol changed after freeze')
    else:
        assert not list((SOURCE/'test/cells').glob('*.npz')),'Freeze before any test cells'
        write_json(path,dict(**spec,frozen_unix=time.time()))
    return sha(path)


def selection_hash():
    from .study import check_selection
    _,digest=check_selection();return digest


def standard_noise(regime,rsr,local):
    rng=np.random.default_rng(np.random.SeedSequence([DATA_SEED,REGIMES.index(regime),RSRS.index(rsr),int(local)]))
    return (rng.standard_normal((32,VECTORS))+1j*rng.standard_normal((32,VECTORS)))/np.sqrt(2)


def data_for(channel,reference,sigma2,regime,rsr,local):
    x,bits=hypotheses(3)
    z=np.abs(channel@x.T+np.asarray(reference).reshape(32,1)+np.sqrt(sigma2)*standard_noise(regime,rsr,local))
    return z,bits


def evaluate(digest,available_only=False):
    selected=selection_hash();complete=True
    for regime,rsr,snr in cells():
        name=label(regime,rsr,snr);source=SOURCE/'test/cells'/(name+'.npz');destination=OUT/'cells'/(name+'.npz')
        if not source.exists() and available_only:complete=False;continue
        if destination.exists():
            old=load(destination)
            assert str(old['protocol_sha256'])==digest and str(old['source_sha256'])==sha(source)
            assert str(old['selection_sha256'])==selected
            print('BER cached',name,flush=True);continue
        a=load(source)
        assert str(a['selection_sha256'])==selected
        assert int(a['seed'])==2026101822
        assert str(a['protocol_sha256'])==sha(SOURCE/'protocol.json')
        assert float(a['snr'])==snr and float(a['rsr'])==rsr and str(a['regime'])==regime
        source_names=list(map(str,a['method_names']));assert tuple(source_names)==USED
        assert a['z'].shape==(CASES,32,30) and a['s'].shape==(CASES,3,30)
        offset={'L2':100000,'L4':200000,'random3to7':0}[regime]
        np.testing.assert_array_equal(a['indices'],np.arange(CASES)+offset)
        np.testing.assert_array_equal(a['local_indices'],np.arange(CASES))
        truth=a['truth_g'];assert truth.shape==(CASES,32,3)
        channels=np.concatenate((truth[None],a['estimates']))
        assert channels.shape==(len(NAMES),CASES,32,3) and np.isfinite(channels).all()
        reference=a['b'][...,:1];variance=a['sigma2']
        assert np.array_equal(a['b'],np.broadcast_to(reference,a['b'].shape))
        data=np.stack([data_for(truth[i],reference[i],float(variance[i]),regime,rsr,i)[0] for i in range(CASES)])
        bits=hypotheses(3)[1]
        datapath=OUT/'data'/(name+'.npz')
        save(datapath,z=data,b=reference,sigma2=variance,indices=a['indices'],
            transmitted_bits=bits,hypothesis_symbols=hypotheses(3)[0],protocol_sha256=digest)
        errors=np.zeros((len(NAMES),CASES,3),dtype=np.int64);seconds=np.zeros((len(NAMES),CASES));reuse=[];seen={}
        for mi,method in enumerate(NAMES):
            fingerprint=hashlib.sha256(channels[mi].tobytes()).hexdigest();previous=seen.get(fingerprint)
            if previous is not None:
                assert np.array_equal(channels[mi],channels[previous]);errors[mi]=errors[previous];reuse.append(NAMES[previous]);continue
            seen[fingerprint]=mi;reuse.append('')
            for i in range(CASES):
                start=time.perf_counter();decoded=detect(data[i],channels[mi,i],reference[i],float(variance[i]))
                seconds[mi,i]=time.perf_counter()-start
                errors[mi,i]=np.count_nonzero(decoded!=bits,axis=(0,2))
        save(destination,method_names=np.array(NAMES),errors_per_user=errors,
            bits_per_user=np.full((CASES,3),2*VECTORS,dtype=np.int64),duplicate_of=np.array(reuse),
            detector_wall_seconds=seconds,indices=a['indices'],source_sha256=sha(source),data_sha256=sha(datapath),
            protocol_sha256=digest,selection_sha256=selected)
        print(json.dumps(dict(cell=name,unique_detector_methods=len(seen),methods=len(NAMES),
            total_errors={method:int(errors[mi].sum()) for mi,method in enumerate(NAMES)})),flush=True)
    return complete


def interval(samples,adjusted=False):
    return np.quantile(samples,[.05/12,1-.05/12] if adjusted else [.025,.975]).tolist()


def memberships(regime,rsr,snr):
    return ['all','ordinary' if snr<=20 else 'high_snr']+(['transition' if snr<=45 else 'stress',f'high_rsr:{rsr:g}',f'high_regime:{regime}'] if snr>20 else [])+[f'snr:{snr:g}',f'rsr:{rsr:g}',f'regime:{regime}']


def analyze(digest):
    selected=selection_hash();names=NAMES+tuple(GROUPS)
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    weights={r:rng.multinomial(CASES,np.full(CASES,1/CASES),size=BOOTSTRAPS).astype(float) for r in REGIMES}
    accum={};rows=[];cell_contrasts=[]
    for regime,rsr,snr in cells():
        name=label(regime,rsr,snr);a=load(OUT/'cells'/(name+'.npz'))
        assert tuple(map(str,a['method_names']))==NAMES
        assert str(a['protocol_sha256'])==digest and str(a['selection_sha256'])==selected
        counts=a['errors_per_user'].sum(-1);rate=counts/(3*2*VECTORS)
        rate=np.concatenate((rate,np.stack([rate[[NAMES.index(n) for n in members]].mean(0) for members in GROUPS.values()])))
        samples=rate@weights[regime].T/CASES;point=rate.mean(-1)
        for contrast,(reference,candidate) in CONTRASTS.items():
            ri,ci=names.index(reference),names.index(candidate)
            cell_contrasts.append(dict(regime=regime,rsr_db=rsr,snr_db=snr,contrast=contrast,
                reference=reference,candidate=candidate,ber_reduction=float(point[ri]-point[ci]),
                bootstrap95=interval(samples[ri]-samples[ci]),interpretation='descriptive pointwise'))
        for key in memberships(regime,rsr,snr):
            if key not in accum:accum[key]=dict(cells=0,rates=np.zeros(len(names)),samples=np.zeros_like(samples),errors=np.zeros(len(NAMES),dtype=np.int64),case_errors=np.zeros(len(NAMES),dtype=np.int64),regimes=set())
            store=accum[key];store['cells']+=1;store['rates']+=point;store['samples']+=samples
            store['errors']+=counts.sum(-1);store['case_errors']+=(counts>0).sum(-1);store['regimes'].add(regime)
        for mi,method in enumerate(names):
            row=dict(regime=regime,rsr_db=rsr,snr_db=snr,method=method,ber=float(point[mi]),bootstrap95=interval(samples[mi]),bits_per_fit=CASES*3*2*VECTORS,channel_anchors=CASES)
            if method in GROUPS:
                row['errors_per_training_seed']={n:int(counts[NAMES.index(n)].sum()) for n in GROUPS[method]}
                row['channels_with_errors_per_fit']={n:int((counts[NAMES.index(n)]>0).sum()) for n in GROUPS[method]}
            else:row.update(bit_errors=int(counts[mi].sum()),channels_with_errors=int((counts[mi]>0).sum()))
            rows.append(row)
    def finish(key,primary=False):
        store=accum[key];ncell=store['cells'];point=store['rates']/ncell;samples=store['samples']/ncell
        methods={};contrasts={}
        for mi,name in enumerate(names):
            row=dict(ber=float(point[mi]),bootstrap95=interval(samples[mi]),bits_per_fit=ncell*CASES*3*2*VECTORS)
            if name in GROUPS:
                row['errors_per_training_seed']={n:int(store['errors'][NAMES.index(n)]) for n in GROUPS[name]}
                row['operating_cases_with_errors_per_fit']={n:int(store['case_errors'][NAMES.index(n)]) for n in GROUPS[name]}
                sr=[float(point[NAMES.index(n)]) for n in GROUPS[name]];row['training_seed_ber_range']=[min(sr),max(sr)]
            else:row.update(bit_errors=int(store['errors'][mi]),operating_cases_with_errors=int(store['case_errors'][mi]))
            methods[name]=row
        for contrast,(reference,candidate) in CONTRASTS.items():
            ri,ci=names.index(reference),names.index(candidate);rep=samples[ri]-samples[ci]
            record=dict(reference=reference,candidate=candidate,ber_reduction=float(point[ri]-point[ci]),bootstrap95=interval(rep))
            if primary:
                adjusted=interval(rep,True);record.update(bonferroni6_interval=adjusted,positive_reduction_supported_familywise=adjusted[0]>0)
            contrasts[contrast]=record
        return dict(cells=ncell,operating_cases=ncell*CASES,coupled_clusters=len(store['regimes'])*CASES,methods=methods,contrasts=contrasts)
    summary=dict(protocol_sha256=digest,selection_sha256=selected,primary=finish('high_snr',True),overall=finish('all'),
        ordinary=finish('ordinary'),transition=finish('transition'),stress=finish('stress'),
        by_snr={f'{s:g}':finish(f'snr:{s:g}') for s in SNRS},by_rsr={f'{r:g}':finish(f'rsr:{r:g}') for r in RSRS},
        by_regime={r:finish(f'regime:{r}') for r in REGIMES},
        primary_by_rsr={f'{r:g}':finish(f'high_rsr:{r:g}') for r in RSRS},
        primary_by_regime={r:finish(f'high_regime:{r}') for r in REGIMES},cells=rows,cell_contrasts=cell_contrasts,
        confidence='20000 paired coupled-block resamples,64 local indices reused across every reference and SNR within regime. Conditional on the fixed models/policies;192 resampling clusters. Bonferroni6 applies only to the three high-SNR primary contrasts across NMSE/BER cooutcomes.',
        zero_errors='Zero observed errors and a degenerate empirical bootstrap interval do not establish zeroBER or a rare-event bound. One error per cell/fit is1/24576; symbols/bits within a channel are not independent channel trials.',
        coverage='64 transmitted jointQPSK vectors per channel enumerate the complete uniform alphabet once; independent data noise per vector. 576 physical channel anchors reused across17 SNRs,9792 operating cases. No extra bits added after test results.')
    write_json(OUT/'summary.json',summary);plot(rows)
    print(json.dumps(summary['primary']['contrasts']),flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    styles={'perfect_csi':('Perfect CSI','#777777','--'),'original_mean':('Retrained filtered','#333333','-'),
        'em100_bypass_mean':('Calibrated EM100 bypass','#0072b2','-'),'fast_bypass_mean':('Same-step bypass diagnostic','#009e73',':'),
        'em100':('Direct EM100','#cc79a7','--'),'compact_sparse':('Fresh compact sparse','#d55e00','-')}
    fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,sharey=True);floor=1e-6
    for ri,rsr in enumerate(RSRS):
        for ci,regime in enumerate(REGIMES):
            ax=axes[ri,ci]
            for method,(label_text,color,style) in styles.items():
                selected=[r for r in rows if r['regime']==regime and r['rsr_db']==rsr and r['method']==method]
                values=np.array([r['ber'] for r in selected]);positive=values>0
                ax.plot(SNRS,np.where(positive,values,np.nan),marker='o',markersize=2.6,color=color,linestyle=style,label=label_text)
                ax.scatter(np.array(SNRS)[~positive],np.full((~positive).sum(),floor),marker='v',s=13,color=color)
            ax.set_yscale('log');ax.set_ylim(.65e-6,.6);ax.set_xticks([0,20,30,40,50,60]);ax.grid(alpha=.18,which='both')
            if ri==0:ax.set_title({'L2':'2 paths/user','L4':'4 paths/user','random3to7':'3–7 paths/user'}[regime])
            if ci==0:ax.set_ylabel(f'RSR {rsr:g} dB\nUncoded BER')
            if ri==2:ax.set_xlabel('SNR (dB)')
    handles,labels=axes[0,0].get_legend_handles_labels();handles.append(Line2D([0],[0],color='#555555',marker='v',linestyle='none'));labels.append('Zero observed errors (marker only)')
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(.5,.023))
    fig.suptitle('Retrained-model high-SNR safeguards: identical magnitude-domain data and detector',fontsize=12)
    fig.text(.5,.009,'Each channel uses all64 jointQPSK symbols once. Zero-error markers at10⁻⁶ are display markers, not estimates or bounds.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,.97));fig.savefig(OUT/'ber.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'ber.pdf',bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');p.add_argument('--available-only',action='store_true');p.add_argument('--analyze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);digest=freeze()
    if args.freeze_only:return
    write_json(OUT/'detector_checks.json',verify_detector())
    if not args.analyze_only and not evaluate(digest,args.available_only):return
    analyze(digest)


if __name__=='__main__':main()
