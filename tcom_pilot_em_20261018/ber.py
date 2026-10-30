"""Fresh paired estimate-then-detect BER for pilot matching and EM tails.

The detector is the already independently validated exact64-hypothesis Rician
likelihood. This module never fits, tunes, or selects a channel estimator.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from tcom_followups_20261017.ber_detector import detect,draw_data,verify_detector
from tcom_followups_20261017.common import effective_trial_id
from .training_protocol import sha,write_json

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'results/tcom_pilot_em_20261018'
SOURCE=BASE/'evaluation';OUT=BASE/'ber'
SNRS=(0,5,10,15,20);REGIMES=('L2','L4','random3to7')
FAMILIES=('old','context21','context30');SEEDS=(9101,9102,9103)
CASES=128;VECTORS=256;BOOTSTRAPS=20000;BOOTSTRAP_SEED=2026101894
GROUPS={f'{family}_{kind}_mean':tuple(f'{family}_{seed}_{kind}' for seed in SEEDS)
    for family in FAMILIES for kind in ('tail0','selected')}
NEURAL=tuple(name for members in GROUPS.values() for name in members)
BASELINES=('sparse_fresh_context30','prior_sparse_matched','prior_sparse_10ms',
    'prior_sparse_20ms','prior_sparse_50ms','prior_sparse_full','em100','optshrink')
USED=NEURAL+BASELINES
NAMES=('perfect_csi',)+USED
CONTRASTS=dict(pilot_count=('context21_tail0_mean','context30_tail0_mean'),
    old_continuation=('old_tail0_mean','old_selected_mean'),
    matched_continuation=('context30_tail0_mean','context30_selected_mean'),
    matched_vs_sparse=('sparse_fresh_context30','context30_selected_mean'))


def cells():return [(regime,snr) for regime in REGIMES for snr in SNRS]


def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)


def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}


def freeze():
    detector=REPO/'tcom_followups_20261017/ber_detector.py'
    spec=dict(version=1,name='Fresh paired BER: context-count training and exact EM continuation',
        N=32,K=3,P=30,RSR_dB=12,SNR_dB=list(SNRS),path_regimes=list(REGIMES),
        cases_per_cell=CASES,main_cells=15,total_channel_realizations=1920,coupled_resampling_clusters=384,
        pilot_seed=2026101804,data_seed=2026101805,
        data_seed_rule='SeedSequence([2026101805, cell_index, local_case_index]); cell order regime L2,L4,random3to7, then SNR0,5,10,15,20.',
        independent_QPSK_vectors_per_channel=VECTORS,modulation='Unit-power Gray QPSK, two bits/user/symbol, three users.',
        source_methods=list(USED),additional_method='perfect_csi',neural_metric_groups=GROUPS,
        detector='Identical exact Rician magnitude-only ML over all64 jointQPSK hypotheses for every estimate. Known reference and variance. No coherent data observations are available.',
        common_data='One fresh bit/noise realization per test channel, shared by every detector; independent of pilot noise. Truth is used only for generation, scoring and explicitly labeled perfectCSI reference.',
        duplicates='Bitwise-identical saved channel arrays within a cell may reuse exact detector outputs; these are recorded as duplicate_of, never treated as a timing advantage.',
        selection='Consume only predictions that cite the fully frozen evaluation/selection.json. No BER-based estimator, tail, checkpoint, hyperparameter or test-cell selection.',
        aggregation='Each ofthree fixed neuralfits evaluated separately; groups average BER, never estimates or decisions. The three fits do not triple the independent test data.',
        primary_contrasts=CONTRASTS,improvement_sign='reference BER minus candidate BER; positive means candidate improves',
        uncertainty=dict(method='Paired coupled-block percentile bootstrap conditional on fixed models and selections',
            resamples=BOOTSTRAPS,seed=BOOTSTRAP_SEED,
            construction='Per-regime multinomial128-case weights in L2,L4,random3to7 order, shared across every SNR and method; regimes independent.',
            pointwise_quantiles=[.025,.975],primary_bonferroni_quantiles=[.003125,.996875],
            family='Eight co-primary outcomes: four frozen contrasts times NMSE/BER; individual99.375% intervals provide Bonferroni95% family coverage.'),
        zeros='Zero observed errors are not zero BER or a bound; collapsed empirical bootstrap intervals cannot resolve unseen rare events.',
        guard_policy='No guard BER; only the15 prespecified ordinary-range cells. Guard NMSE handled separately.',
        timing='No detector work during isolated calibration/final estimator timing. Detector walltime is diagnostic only.',
        source_hashes={str(Path(__file__).relative_to(REPO)):sha(__file__),str(detector.relative_to(REPO)):sha(detector)},
        training_protocol_sha256=sha(BASE/'training_protocol.json'))
    # JSON-normalize tuple values before comparing the immutable record.
    spec=json.loads(json.dumps(spec));path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None)
        if old!=spec:raise RuntimeError('BER protocol/source changed after freeze')
    else:write_json(path,dict(**spec,frozen_unix=time.time()))
    return sha(path)


def verify_selection():
    from .study import check_selection
    selection,_,digest=check_selection()
    if tuple(selection['sparse_choices'])!=tuple(str(s) for s in SNRS):
        raise ValueError('Unexpected selected operating range')
    return digest


def evaluate(protocol_digest,available_only=False):
    selection_digest=verify_selection();complete=True
    for cell_index,(regime,snr) in enumerate(cells()):
        label=f'{regime}_snr{snr}';source=SOURCE/'test'/(label+'.npz');out=OUT/'cells'/(label+'.npz')
        if not source.exists() and available_only:complete=False;continue
        if out.exists():
            previous=load(out)
            assert str(previous['protocol_sha256'])==protocol_digest and str(previous['source_sha256'])==sha(source)
            assert str(previous['selection_sha256'])==selection_digest
            print('BER cached',label,flush=True);continue
        a=load(source)
        assert int(a['seed'])==2026101804 and float(a['snr'])==snr and float(a['rsr'])==12
        assert str(a['regime'])==regime and not bool(a['guard'])
        assert str(a['selection_sha256'])==selection_digest
        assert source.stat().st_mtime>=json.loads((SOURCE/'selection.json').read_text())['frozen_unix']
        source_names=list(map(str,a['method_names']));assert len(set(source_names))==len(source_names)
        assert all(name in source_names for name in USED)
        truth=a['truth_g'];assert truth.shape==(CASES,32,3)
        assert a['z'].shape==(CASES,32,30) and a['s'].shape==(CASES,3,30)
        estimates=np.concatenate((truth[None],a['estimates'][[source_names.index(n) for n in USED]]))
        assert estimates.shape==(len(NAMES),CASES,32,3) and np.isfinite(estimates).all()
        L={'L2':2,'L4':4,'random3to7':None}[regime]
        np.testing.assert_array_equal(a['indices'],[effective_trial_id(i,L) for i in range(CASES)])
        reference=a['b'][...,:1];variance=a['sigma2']
        assert np.array_equal(a['b'],np.broadcast_to(reference,a['b'].shape))
        data_z=[];data_bits=[];symbol_indices=[]
        for case in range(CASES):
            rng=np.random.default_rng(np.random.SeedSequence([2026101805,cell_index,case]))
            z,bits,indices=draw_data(truth[case],reference[case],float(variance[case]),rng,VECTORS)
            data_z.append(z);data_bits.append(bits);symbol_indices.append(indices)
        data_z,data_bits,symbol_indices=map(np.stack,(data_z,data_bits,symbol_indices))
        save(OUT/'data'/(label+'.npz'),z=data_z,b=reference,sigma2=variance,indices=a['indices'],protocol_sha256=protocol_digest)
        save(OUT/'truth'/(label+'.npz'),bits=data_bits,symbol_indices=symbol_indices,indices=a['indices'],protocol_sha256=protocol_digest)
        errors=np.zeros((len(NAMES),CASES,3),dtype=np.int64);seconds=np.zeros((len(NAMES),CASES))
        fingerprints={};duplicate_of=[]
        for mi,name in enumerate(NAMES):
            fingerprint=hashlib.sha256(estimates[mi].tobytes()).hexdigest()
            previous=fingerprints.get(fingerprint)
            if previous is not None:
                assert np.array_equal(estimates[previous],estimates[mi]);errors[mi]=errors[previous]
                duplicate_of.append(NAMES[previous]);continue
            fingerprints[fingerprint]=mi;duplicate_of.append('')
            for case in range(CASES):
                tick=time.perf_counter()
                decoded=detect(data_z[case],estimates[mi,case],reference[case],float(variance[case]))
                seconds[mi,case]=time.perf_counter()-tick
                errors[mi,case]=np.count_nonzero(decoded!=data_bits[case],axis=(0,2))
        save(out,method_names=np.array(NAMES),errors_per_user=errors,
            bits_per_user=np.full((CASES,3),2*VECTORS,dtype=np.int64),
            duplicate_of=np.array(duplicate_of),detector_wall_seconds=seconds,
            squared_channel_error=np.sum(np.abs(estimates-truth[None])**2,axis=(-2,-1)),
            truth_energy=np.sum(np.abs(truth)**2,axis=(-2,-1)),indices=a['indices'],
            source_sha256=sha(source),selection_sha256=selection_digest,protocol_sha256=protocol_digest,
            data_sha256=sha(OUT/'data'/(label+'.npz')),truth_sha256=sha(OUT/'truth'/(label+'.npz')))
        print(json.dumps(dict(cell=label,detector_evaluated_methods=len(fingerprints),
            method_count=len(NAMES),ber={name:float(errors[mi].sum()/(CASES*3*2*VECTORS)) for mi,name in enumerate(NAMES)})),flush=True)
    return complete


def interval(samples,adjusted=False):
    return np.quantile(samples,[.003125,.996875] if adjusted else [.025,.975]).tolist()


def analyze(protocol_digest):
    selection_digest=verify_selection()
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    weights={regime:rng.multinomial(CASES,np.full(CASES,1/CASES),size=BOOTSTRAPS).astype(float) for regime in REGIMES}
    names=NAMES+tuple(GROUPS);bits_per_world=3*2*VECTORS
    rows=[];rates=[];replicates=[];all_counts=[]
    for regime,snr in cells():
        a=load(OUT/'cells'/f'{regime}_snr{snr}.npz')
        assert tuple(map(str,a['method_names']))==NAMES
        assert str(a['protocol_sha256'])==protocol_digest and str(a['selection_sha256'])==selection_digest
        counts=a['errors_per_user'].sum(-1);all_counts.append(counts)
        perworld=counts/bits_per_world
        group_rates=np.stack([perworld[[NAMES.index(n) for n in members]].mean(0) for members in GROUPS.values()])
        perworld=np.concatenate((perworld,group_rates))
        samples=perworld@weights[regime].T/CASES
        rates.append(perworld.mean(-1));replicates.append(samples)
        for mi,name in enumerate(names):
            row=dict(regime=regime,snr_db=snr,method=name,ber=float(perworld[mi].mean()),
                bootstrap95=interval(samples[mi]),bits_per_fit=CASES*bits_per_world,independent_channels=CASES)
            if name in GROUPS:
                row['errors_per_training_seed']={n:int(counts[NAMES.index(n)].sum()) for n in GROUPS[name]}
                seed_rates=[float(perworld[NAMES.index(n)].mean()) for n in GROUPS[name]]
                row['training_seed_ber_range']=[min(seed_rates),max(seed_rates)]
            else:row['bit_errors']=int(counts[mi].sum())
            rows.append(row)
    rates,replicates,all_counts=map(np.stack,(rates,replicates,all_counts))
    def aggregate(indices,primary=False):
        methods={}
        for mi,name in enumerate(names):
            row=dict(ber=float(rates[indices,mi].mean()),bootstrap95=interval(replicates[indices,mi].mean(0)),
                bits_per_fit=len(indices)*CASES*bits_per_world,
                channel_realizations=len(indices)*CASES,
                coupled_clusters=len({cells()[i][0] for i in indices})*CASES)
            if name in GROUPS:
                row['errors_per_training_seed']={n:int(all_counts[indices,NAMES.index(n)].sum()) for n in GROUPS[name]}
                sr=[float(rates[indices,NAMES.index(n)].mean()) for n in GROUPS[name]]
                row['training_seed_ber_range']=[min(sr),max(sr)]
            else:row['bit_errors']=int(all_counts[indices,mi].sum())
            methods[name]=row
        contrasts={}
        for key,(reference,candidate) in CONTRASTS.items():
            ri,ci=names.index(reference),names.index(candidate)
            point=float((rates[indices,ri]-rates[indices,ci]).mean())
            samples=(replicates[indices,ri]-replicates[indices,ci]).mean(0)
            result=dict(reference=reference,candidate=candidate,ber_reduction=point,bootstrap95=interval(samples),
                relative_error_reduction=point/methods[reference]['ber'] if methods[reference]['ber'] else None,
                improved_cells=int((rates[indices,ri]>rates[indices,ci]).sum()),tied_cells=int((rates[indices,ri]==rates[indices,ci]).sum()),total_cells=len(indices))
            if primary:
                adjusted=interval(samples,adjusted=True)
                result.update(bonferroni8_interval=adjusted,positive_reduction_supported_familywise=adjusted[0]>0)
            contrasts[key]=result
        return dict(methods=methods,contrasts=contrasts)
    summary=dict(protocol_sha256=protocol_digest,selection_sha256=selection_digest,
        overall=aggregate(np.arange(len(cells())),primary=True),
        by_snr={str(s):aggregate(np.array([i for i,c in enumerate(cells()) if c[1]==s])) for s in SNRS},
        by_regime={r:aggregate(np.array([i for i,c in enumerate(cells()) if c[0]==r])) for r in REGIMES},cells=rows,
        uncertainty='20000 paired coupled-block resamples; same128 local indices across SNRs within regime and all methods. Three regimes independently resampled:1920 channel realizations,384 coupled clusters. Conditional on nine fitted models and frozen selections; three neuralfits do not triple independent test data. Primary overall intervals additionally adjusted across four contrasts times NMSE/BER (8 outcomes). Individual cells and other summaries descriptive.',
        zero_error_policy='Zero observed errors are not a zeroBER claim or confidence bound. One error per cell corresponds to1/196608 bits; the finite128-channel bank limits rare-failure conclusions.',
        scope='Uncoded magnitude-domain vector-ML GrayQPSK; N32,K3,P30,RSR12; synthetic channels. No coding/throughput/hardware claim.')
    write_json(OUT/'summary.json',summary)
    with (OUT/'ber.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['regime','snr_db','method','ber','ci_low','ci_high','bit_errors'])
        writer.writeheader()
        for row in rows:writer.writerow({k:row.get(k,'') for k in ('regime','snr_db','method','ber','bit_errors')}|dict(ci_low=row['bootstrap95'][0],ci_high=row['bootstrap95'][1]))
    plot(rows)
    print(json.dumps(summary['overall']['contrasts']),flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    styles={
        'perfect_csi':('Perfect CSI','#555555','--'),
        'old_tail0_mean':('Original learned','#999999',':'),
        'old_selected_mean':('Original + selected EM','#56b4e9','-'),
        'context21_tail0_mean':('Fresh21-pilot training','#cc79a7','--'),
        'context30_tail0_mean':('Fresh30-pilot training','#009e73','--'),
        'context30_selected_mean':('Fresh30 + selected EM','#0072b2','-'),
        'sparse_fresh_context30':('Fresh matched-budget sparse','#d55e00','-')}
    fig,axes=plt.subplots(1,3,figsize=(13.4,5.2),sharey=True);floor=1e-6
    for ax,regime in zip(axes,REGIMES):
        for method,(label,color,style) in styles.items():
            selected=[r for r in rows if r['regime']==regime and r['method']==method]
            values=np.array([r['ber'] for r in selected]);positive=values>0
            ax.plot(SNRS,np.where(positive,values,np.nan),marker='o',markersize=3.6,color=color,linestyle=style,label=label)
            ax.scatter(np.array(SNRS)[~positive],np.full((~positive).sum(),floor),marker='v',s=25,color=color)
        ax.set_yscale('log');ax.set_ylim(.65e-6,.3);ax.set_xticks(SNRS);ax.set_xlabel('SNR (dB)')
        ax.set_title({'L2':'2 paths/user','L4':'4 paths/user','random3to7':'3–7 paths/user'}[regime]);ax.grid(alpha=.2,which='both')
    axes[0].set_ylabel('Uncoded bit error rate')
    handles,labels=axes[-1].get_legend_handles_labels()
    handles.append(Line2D([0],[0],color='#555555',marker='v',linestyle='none'));labels.append('Zero observed errors (marker only)')
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(.5,.04))
    fig.suptitle('Pilot-count matching and exact EM continuation: identical data and detector',fontsize=11)
    fig.text(.5,.012,'Neural curves average three separately evaluated fits. Zero-error markers at10⁻⁶ are display markers, not estimates or bounds.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.18,1,.98));fig.savefig(OUT/'ber.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'ber.pdf',bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');p.add_argument('--available-only',action='store_true');p.add_argument('--analyze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    checks=verify_detector();digest=freeze();write_json(OUT/'detector_checks.json',checks)
    if args.freeze_only:return
    if not args.analyze_only and not evaluate(digest,args.available_only):return
    analyze(digest)


if __name__=='__main__':main()
