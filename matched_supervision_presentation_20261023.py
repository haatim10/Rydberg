"""Additional checks and readable views; frozen study outputs are not edited."""
import itertools, json, shutil
import numpy as np
import torch
from scipy.stats import rice
from matched_supervision_20261023.common import OUT, PROJECT, SEEDS, SNRS, DATA_MASTER, power, write, frozen, contexts, load_observations
from matched_supervision_20261023.models import build
from tcom_followups_20261017.ber_detector import draw_data, detect

ARMS=('proposed_supervised','proposed_selfsupervised','urformer_supervised','urformer_selfsupervised')
LABELS=('Proposed, with channel labels','Proposed, without channel labels',
        'URformer, with channel labels','URformer, without channel labels')
STYLES=(('#168052','-','*'),('#168052','--','*'),('#2168a5','-','o'),('#2168a5','--','o'))


@torch.no_grad()
def independent_checks():
    val,_=load_observations('val');x=contexts(val,slice(0,4));precision=[]
    for arm in ARMS:
        arch=arm.split('_')[0]
        for seed in SEEDS:
            state=torch.load(OUT/'training'/arm/str(seed)/'final.pt',map_location='cpu',weights_only=False)
            m=build(arch,seed);m.load_state_dict(state['model']);m.eval();p32=m(*x)
            xx=tuple(t.to(torch.complex128 if t.is_complex() else torch.float64) for t in x)
            p64=m.double()(*xx)
            relative=float(((p32-p64).abs().square().sum()/p64.abs().square().sum()).sqrt())
            assert relative<1e-3
            precision.append(dict(arm=arm,seed=seed,relative_float32_float64_error=relative))
    with np.load(OUT/'data/test_raw.npz') as f:h=f['h'][:512].astype(np.complex128)
    bits=np.array(list(itertools.product((0,1),repeat=8)),dtype=np.int64).reshape(256,4,2)
    symbols=((1-2*bits[...,0])+1j*(1-2*bits[...,1]))/np.sqrt(2)
    replay=[]
    for snr in (0.,5.,20.):
        v=power(20)/10**(snr/10);b=np.full((32,1),np.sqrt(10*power(20)))
        with np.load(OUT/'ber'/f'P20_snr{snr:g}.npz') as f:
            names=list(f['method_names']);counts=f['bit_errors'].copy()
        with np.load(OUT/'predictions'/f'P20_snr{snr:g}.npz') as predictions:
            for i in (0,17,511):
                rng=np.random.default_rng(np.random.SeedSequence([DATA_MASTER,i]))
                z,truth,_=draw_data(h[i],b,v,rng,vectors=32)
                for j,name in enumerate(names):
                    g=h[i] if name=='known_channel' else predictions[name][i]
                    decoded=detect(z,g,b,v)
                    assert np.count_nonzero(decoded!=truth)==counts[j,i]
                    amp=np.abs(g@symbols.T+b).T
                    scores=-rice.logpdf(z[:,:4].T[:,None,:],b=amp[None]/np.sqrt(v/2),scale=np.sqrt(v/2)).sum(-1)
                    np.testing.assert_array_equal(bits[scores.argmin(-1)],decoded[:4])
                replay.append(dict(snr=snr,anchor=i,methods=len(names)))
    write(OUT/'additional_inference_audit.json',dict(passed=True,final_precision=precision,
          independent_full_Rice_density_decoding=replay,stored_BER_counts_replayed=True))


def ber_intervals():
    arrays={}
    for label in ('final','best'):
        for arm in ARMS:
            values=[]
            for snr in SNRS:
                with np.load(OUT/'ber'/f'P20_snr{snr:g}.npz') as f:
                    names=list(f['method_names']);c=f['bit_errors']
                    values.append(np.stack([c[names.index(f'{arm}_{s}_{label}')] for s in SEEDS]).mean(0)/256)
            arrays[(arm,label)]=np.stack(values)
    contrasts={'proposed_vs_UR_with_labels':(ARMS[2],ARMS[0]),
               'proposed_vs_UR_without_labels':(ARMS[3],ARMS[1]),
               'labels_help_proposed':(ARMS[1],ARMS[0]),'labels_help_UR':(ARMS[3],ARMS[2])}
    rng=np.random.default_rng(2026102344);boot={}
    for _ in range(2000):
        ids=rng.integers(0,512,512)
        means={key:v[:,ids].mean(1) for key,v in arrays.items()}
        for label in ('final','best'):
            for name,(left,right) in contrasts.items():
                boot.setdefault((label,name),[]).append(100*(means[(left,label)]-means[(right,label)]))
    result={}
    for (label,name),values in boot.items():
        left,right=contrasts[name]
        result[label+'_'+name]=dict(SNR=SNRS,
            gain_percentage_points=(100*(arrays[(left,label)].mean(1)-arrays[(right,label)].mean(1))).tolist(),
            ci95_percentage_points=np.quantile(values,[.025,.975],axis=0).T.tolist())
    write(OUT/'additional_BER_intervals.json',dict(contrasts=result,
        interpretation='Positive means the named benefit has lower BER. Descriptive paired channel bootstrap, conditional on three fits. Zero-count intervals do not prove equal or zero population BER.'))


def present():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((OUT/'summary.json').read_text())
    rows=sorted([r for r in result['nmse'] if r['P']==20],key=lambda r:r['snr'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.9),sharey=True)
    handles=[]
    for ax,label,title in zip(axes,('final','best'),('Final weights after 50 rounds','Weights chosen using validation data')):
        for i,(arm,name,(color,line,marker)) in enumerate(zip(ARMS,LABELS,STYLES)):
            h,=ax.plot(SNRS,[r['scores'][arm+'_'+label]['nmse_db'] for r in rows],
                color=color,ls=line,marker=marker,lw=1.8,ms=7 if marker=='*' else 5,
                markerfacecolor='white' if i%2 else color,label=name)
            if label=='final':handles.append(h)
        h,=ax.plot(SNRS,[r['scores']['EM100']['nmse_db'] for r in rows],':',color='#bd762b',lw=1.6,label='EM100, same starting estimate')
        if label=='final':handles.append(h)
        ax.set(title=title,xlabel='SNR (dB)',xticks=SNRS);ax.grid(alpha=.2)
    axes[0].set_ylabel('Channel error: NMSE (dB)')
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=8.5,bbox_to_anchor=(.5,.055),frameon=False)
    fig.text(.5,.015,'Same 20,000 training examples, 50 rounds, 3 runs, and 20 input pilots. Lower error is better. No test channel is supplied to a model.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.19,1,1))
    fig.savefig(PROJECT/'Matched_Training_Final_And_Validation.png',dpi=190)
    fig.savefig(OUT/'final_and_validation.svg');plt.close(fig)
    # All modes have the same validation quantity; training losses have different units.
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,arch,title in zip(axes,('proposed','urformer'),('Proposed model','URformer implementation')):
        for mode,color,line in (('supervised','#9b4c18','-'),('selfsupervised','#2168a5','--')):
            histories=[]
            for seed in SEEDS:
                h=json.loads((OUT/'training'/f'{arch}_{mode}'/str(seed)/'history.json').read_text())
                y=np.array([r['validation_heldout_nll'] for r in h['history']]);histories.append(y)
                ax.plot(np.arange(1,51),y,color=color,alpha=.22,lw=.8)
            ax.plot(np.arange(1,51),np.mean(histories,axis=0),color=color,ls=line,lw=2,
                    label='With channel labels' if mode=='supervised' else 'Without channel labels')
        ax.set(title=title,xlabel='Training round',ylabel='Validation likelihood loss (lower is better)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(PROJECT/'Matched_Training_Validation_Curves.png',dpi=170);plt.close(fig)
    lines=['MATCHED TRAINING: RESULTS AND CHECKPOINT CHOICE','',
           'All four cases: same 20,000 training channels, 2,000 validation channels, 50 rounds, three independent fits, Adam settings, batch order and starting channel estimate.',
           '32 antennas, four users, 20 input pilots. Nine separate pilot targets are available during training for every case.',
           'With labels: train on the true channel. Without labels: train on nine withheld noisy magnitude measurements.',
           'No model gets the true channel at test time. Known reference and noise variance are supplied equally to all methods.',
           'Every fit finished all 50 rounds. Primary results use final weights. Secondary results use the checkpoint with the lowest shared validation likelihood loss; no true validation channel is used for this selection.',
           'These are our implementations. Published URformer accuracy remains unreproduced. Equal updates do not guarantee that either model has its best possible training recipe.',
           'This comparison covers the ordinary-SNR proposed core, not the high-SNR hybrid recovery branches.','']
    for label in ('final','best'):
        lines += [('FINAL 50-ROUND WEIGHTS' if label=='final' else 'COMMON VALIDATION-CHOSEN WEIGHTS'),
                  'SNR | Proposed + labels | Proposed no labels | URformer + labels | URformer no labels | EM100',
                  'NMSE (dB); lower is better:']
        for row in rows:
            values=[row['scores'][a+'_'+label]['nmse_db'] for a in ARMS]+[row['scores']['EM100']['nmse_db']]
            lines.append(f"{row['snr']:4g} | "+' | '.join(f'{v:9.4f}' for v in values))
        lines+=['BER (%); same data and exact Rician detector:']
        for row in result['ber']:
            values=[100*row['scores'][a+'_'+label]['ber'] for a in ARMS]+[100*row['scores']['EM100']['ber']]
            lines.append(f"{row['snr']:4g} | "+' | '.join(f'{v:9.6f}' for v in values))
        lines+=['']
    lines+=['RUNTIME: measured final weights only. Full call including common starting estimate and pilot solve; one CPU thread, batch one, float64, PyTorch for every method.',
            'P20 p90 is the slower value across three fits and two fixed timing windows. Weight loading and data detection are excluded.']
    for name,r in result['runtime'].items():
        lines.append(f"{name}: P20 p90 {r['worst_fit_window_P20_p90_ms']:.3f} ms; all-pilot-count p90 {r['worst_fit_window_p90_ms']:.3f} ms; observed maximum over full grid {r['max_ms']:.3f} ms.")
    lines+=['','VALIDATION-CHOSEN EPOCHS:']
    for arm in ARMS:lines.append(arm+': '+str([result['training'][f'{arm}_{s}']['best_epoch'] for s in SEEDS]))
    lines+=['','CHECKS: original preflight and full training replay passed. Additional audits replayed every cached training initializer, every saved NMSE score, selected BER counts using full Rice densities, and final-weight numerical precision.',
            'BER uses 512 physical channels and 32 data vectors per channel, with the same bits/noise across fits. Zero observed errors do not establish zero population BER.',
            'Paired intervals are descriptive and conditional on the three fitted models. Full histories, per-fit scores, original plots, weights and channel predictions are retained locally.',
            'The manuscript and earlier experiment records are unchanged.']
    (PROJECT/'Matched_Training_Comparison.txt').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);frozen()
    independent_checks();ber_intervals();present();frozen()
    print('Independent checks, paired BER intervals and both checkpoint comparisons complete.',flush=True)
