"""Report every predeclared arm; no result-driven model selection."""
import json,zipfile,time
import numpy as np
from .common import *


ARMS=tuple(f'{a}_{m}' for a in ARCHES for m in MODES)
LABELS={'proposed_supervised':'Proposed: with channel labels',
        'proposed_selfsupervised':'Proposed: without channel labels',
        'urformer_supervised':'URformer: with channel labels',
        'urformer_selfsupervised':'URformer: without channel labels'}


def summary():
    frozen();assert json.loads((OUT/'accuracy_complete.json').read_text())['passed']
    assert json.loads((OUT/'ber_complete.json').read_text())['passed']
    assert json.loads((OUT/'training_audit.json').read_text())['passed']
    rows=[];arrays={};berrows=[]
    for p,snr in GRID:
        with np.load(OUT/'test'/f'P{p}_snr{snr:g}.npz') as a:
            scores={}
            for arm in ARMS:
                for label in ('final','best'):
                    values=np.stack([a[f'{arm}_{s}_{label}'] for s in SEEDS]);arrays[(p,snr,arm,label)]=values
                    scores[f'{arm}_{label}']=dict(nmse_db=float(10*np.log10(values.mean())),
                        nmse_linear=float(values.mean()),per_fit_nmse_db=(10*np.log10(values.mean(1))).tolist())
            for name in ('GS100','EM10','EM100'):scores[name]=dict(nmse_db=float(10*np.log10(a[name].mean())))
        rows.append(dict(P=p,snr=snr,scores=scores))
    for snr in SNRS:
        with np.load(OUT/'ber'/f'P20_snr{snr:g}.npz') as a:
            names=list(a['method_names']);counts=a['bit_errors'];scores={}
            for arm in ARMS:
                for label in ('final','best'):
                    cc=np.stack([counts[names.index(f'{arm}_{s}_{label}')] for s in SEEDS])
                    scores[f'{arm}_{label}']=dict(ber=float(cc.sum()/(cc.size*256)),
                        total_bit_errors=int(cc.sum()),total_bits=int(cc.size*256),per_fit_ber=(cc.sum(1)/(512*256)).tolist())
            for name in ('GS100','EM10','EM100','known_channel'):
                cc=counts[names.index(name)];scores[name]=dict(ber=float(cc.sum()/(len(cc)*256)),total_bit_errors=int(cc.sum()),total_bits=len(cc)*256)
        berrows.append(dict(snr=snr,scores=scores))
    windows=[json.loads((OUT/'timing'/f'window{i}.json').read_text()) for i in (1,2)]
    timings={}
    for arm in (*ARMS,'GS100','EM10','EM100'):
        names=[f'{arm}_{s}_final' for s in SEEDS] if arm in ARMS else [arm]
        scores=[w['scores'][n] for w in windows for n in names]
        p20=[]
        for w in windows:
            for name in names:p20.append(float(np.quantile([r['ms'] for r in w['records'] if r['method']==name and r['P']==20],.9)))
        timings[arm]=dict(worst_fit_window_p90_ms=max(v['p90_ms'] for v in scores),
            worst_fit_window_P20_p90_ms=max(p20),max_ms=max(v['max_ms'] for v in scores),
            over_budget=sum(v['over_budget'] for v in scores),all_p90_under10_74=all(v['p90_ms']<=10.74 for v in scores))
    # Descriptive paired intervals conditional on the three fitted models.
    contrasts={
        'proposed_vs_urformer_with_labels':('urformer_supervised','proposed_supervised'),
        'proposed_vs_urformer_without_labels':('urformer_selfsupervised','proposed_selfsupervised'),
        'labels_help_proposed':('proposed_selfsupervised','proposed_supervised'),
        'labels_help_urformer':('urformer_selfsupervised','urformer_supervised')}
    intervals={};rng=np.random.default_rng(2026102343)
    mean_channels={arm:np.stack([arrays[(20,s,arm,'final')].mean(0) for s in SNRS]) for arm in ARMS}
    boot={name:[] for name in contrasts}
    for _ in range(2000):
        ids=rng.integers(0,2000,2000)
        db={arm:10*np.log10(v[:,ids].mean(1)) for arm,v in mean_channels.items()}
        for name,(left,right) in contrasts.items():boot[name].append(db[left]-db[right])
    for name,(left,right) in contrasts.items():
        values=np.array(boot[name]);point=10*np.log10(mean_channels[left].mean(1))-10*np.log10(mean_channels[right].mean(1))
        intervals[name]=dict(positive_meaning='First method has more error; the named benefit is positive.',per_snr_gain_db=point.tolist(),
            per_snr_ci95=np.quantile(values,[.025,.975],axis=0).T.tolist(),mean_setting_gain_db=float(point.mean()),
            mean_setting_ci95=np.quantile(values.mean(1),[.025,.975]).tolist())
    histories={}
    for arm in ARMS:
        for seed in SEEDS:
            h=json.loads((OUT/'training'/arm/str(seed)/'history.json').read_text())
            histories[f'{arm}_{seed}']=dict(best_epoch=h['best_epoch'],epochs=len(h['history']),
                train_seconds=sum(v['training_seconds'] for v in h['history']),
                val_seconds=sum(v['validation_seconds'] for v in h['history']))
    result=dict(status='complete',arms=ARMS,nmse=rows,ber=berrows,runtime=timings,
        paired_descriptive_intervals=intervals,training=histories,
        primary='Final50epochs,common spectral initialization and identical data/optimizer settings.',
        secondary='Best checkpoint from the same heldout validation likelihood for every arm.',
        limitation='Our URformer implementation,not a verified replication of published accuracy. Matched training updates are not equal compute or proof of optimal convergence.')
    write(OUT/'summary.json',result);return result


def plots(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'proposed_supervised':('#178252','-','*'), 'proposed_selfsupervised':('#178252','--','*'),
            'urformer_supervised':('#1967a3','-','o'),'urformer_selfsupervised':('#1967a3','--','o')}
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8))
    for ax,scope in zip(axes,('snr','pilots')):
        rows=sorted([r for r in result['nmse'] if r['P']==20] if scope=='snr' else [r for r in result['nmse'] if r['snr']==5.],key=lambda r:r['snr'] if scope=='snr' else r['P'])
        x=[r['snr'] if scope=='snr' else r['P'] for r in rows]
        for arm in ARMS:
            color,line,mark=styles[arm]
            ax.plot(x,[r['scores'][arm+'_final']['nmse_db'] for r in rows],color=color,ls=line,marker=mark,label=LABELS[arm],lw=1.8)
        for name,color,line in (('EM100','#d77b28',':'),('EM10','#777777',':')):
            ax.plot(x,[r['scores'][name]['nmse_db'] for r in rows],color=color,ls=line,label=name+'; same start',lw=1.5)
        ax.set(xlabel='SNR (dB)' if scope=='snr' else 'Number of pilots',ylabel='NMSE (dB)',xticks=x,
            title='20 input pilots; four users' if scope=='snr' else 'SNR 5 dB; all models trained with 20 input pilots')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=7.5)
    fig.text(.5,.015,'All four models trained from scratch on the same data for 50 epochs. True-channel labels are used only during supervised training.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(OUT/'matched_nmse.png',dpi=190);fig.savefig(OUT/'matched_nmse.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.6,4.4))
    for arm in ARMS:
        c,l,m=styles[arm];y=[r['scores'][arm+'_final']['ber'] for r in result['ber']]
        # True zero counts are omitted, never replaced with an arbitrary floor.
        ax.semilogy(SNRS,[v if v>0 else np.nan for v in y],color=c,ls=l,marker=m,label=LABELS[arm])
    for name,c in (('EM100','#d77b28'),('known_channel','#444444')):
        y=[r['scores'][name]['ber'] for r in result['ber']]
        ax.semilogy(SNRS,[v if v>0 else np.nan for v in y],':',color=c,label=name)
    ax.set(xlabel='SNR (dB)',ylabel='Bit error rate',title='Same data observations and detector; 20 input pilots')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'matched_ber.png',dpi=190);plt.close(fig)


def plain(result):
    lines=['FAIR ARCHITECTURE AND TRAINING-LABEL COMPARISON','',
        'Every model was trained from scratch on the same 20,000 examples for 50 epochs.',
        'Four users, 32 antennas, 20 input pilots, and nine separate pilot targets available during training.',
        'Same starting channel estimate, optimizer, learning-rate schedule, batch order, precision and validation rule.',
        'Only supervised training receives true-channel targets. No model sees the true test channel.',
        'The no-label versions learn from the nine withheld pilot measurements.',
        'Primary results use the final epoch. Separate validation-selected results are retained in summary.json.',
        'The pilot-count sweep tests transfer of the same models trained with 20 pilots.',
        '', 'NMSE in dB. Lower is better.',
        'SNR | Proposed with labels | Proposed without labels | URformer with labels | URformer without labels | EM100']
    for r in sorted([r for r in result['nmse'] if r['P']==20],key=lambda r:r['snr']):
        values=[r['scores'][a+'_final']['nmse_db'] for a in ARMS]+[r['scores']['EM100']['nmse_db']]
        lines.append(f"{r['snr']:4g} | "+' | '.join(f'{v:10.4f}' for v in values))
    lines+=['','BER in percent. Same 512 channels and 32 data symbols per channel for all methods.']
    for r in result['ber']:
        values=[100*r['scores'][a+'_final']['ber'] for a in ARMS]+[100*r['scores']['EM100']['ber']]
        lines.append(f"{r['snr']:4g} dB | "+' | '.join(f'{v:10.6f}' for v in values))
    lines+=['','Runtime: complete batch-one calls, one CPU thread, PyTorch for both architectures.',
            'Slower p90 across the three fits and two fixed timing windows; model loading and data detection excluded.']
    for arm,v in result['runtime'].items():lines.append(f"{arm}: all-condition p90 {v['worst_fit_window_p90_ms']:.3f} ms; P20 p90 {v['worst_fit_window_P20_p90_ms']:.3f} ms; maximum {v['max_ms']:.3f} ms.")
    lines+=['','Limits: this evaluates our documented URformer implementation. Its published accuracy has not been reproduced.',
            'The same number of training updates does not mean the same training computation or prove either model is optimally tuned.',
            'Intervals are descriptive and conditional on the three trained fits. Zero observed bit errors do not prove zero population BER.',
            'The manuscript and all earlier experiment records were preserved.']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    import shutil
    for name in ('matched_nmse.png','matched_ber.png','RESULTS.txt','summary.json'):
        shutil.copy2(OUT/name,PROJECT/('Matched_Training_'+name))
    print('\n'.join(lines),flush=True)


def finish():
    result=summary();plots(result);plain(result);frozen()
    paths=[*CODE.glob('*.py'),CODE/'README.txt']
    paths += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.txt','.png','.svg') and p.name not in ('completion.json','execution.lock')]
    paths += list((OUT/'test').glob('*.npz'))+list((OUT/'ber').glob('*.npz'))
    dest=PROJECT/'Matched_Training_Evidence.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(REPO)))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    write(OUT/'completion.json',dict(status='complete',protocol_sha256=sha(OUT/'protocol.json'),summary_sha256=sha(OUT/'summary.json'),
        evidence_zip_sha256=sha(dest),finished_unix=time.time(),full_weights_and_predictions='Retained locally under training/ and predictions/; omitted from the small evidence ZIP.'))


if __name__=='__main__':finish()
