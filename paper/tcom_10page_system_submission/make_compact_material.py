"""Generate compact-follow-up tables/figure from completed audited results."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=REPO/'results/compact_baseline'
LABELS={'none':'Learned steps only','global':'Global filtering',
    'adaptive':'Spectral/state controller','recurrent':'Recurrent control',
    'opt_selected':'OptShrink, offline','rank_loop':'OptShrink, block rank'}


def read(path): return json.loads(path.read_text())


def row(*items): return ' & '.join(map(str,items))+r'\\'+'\n'


def table(caption,label,columns,header,rows):
    return (r'\begin{table}[!t]'+'\n'+f'\\caption{{{caption}}}\n\\label{{{label}}}\\centering\\footnotesize\n'+
        r'\setlength{\tabcolsep}{3pt}'+'\n'+f'\\begin{{tabular}}{{@{{}}{columns}@{{}}}}\\toprule\n'+
        row(*header)+r'\midrule'+'\n'+''.join(rows)+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{table}'+'\n')


def main():
    assert read(ROOT/'status.json')['state']=='complete'
    s=read(ROOT/'summary.json');assert s['audit']['status']=='passed'
    assert s['independent_headline_recalculation']=='passed'
    for source,digest in read(ROOT/'manifest.json')['sources'].items():
        assert hashlib.sha256((REPO/source).read_bytes()).hexdigest()==digest,source
    arrays=np.load(ROOT/'all_errors.npz',allow_pickle=False)
    e,p,names=arrays['squared_error'],arrays['channel_power'],arrays['names'].tolist()
    assert e.shape==(2,128,14,16) and p.shape==(2,128,16)
    for key,value in s['primary'].items():
        ii=[names.index(key)] if key.startswith(('opt_','rank_')) else [names.index(key+'_'+str(seed)) for seed in (9101,9102,9103)]
        literal=np.mean([10*np.log10(sum(float(e[b,c,m,i]) for b in range(2) for i in range(16))/
            sum(float(p[b,c,i]) for b in range(2) for i in range(16))) for c in range(126) for m in ii])
        np.testing.assert_allclose(literal,value,rtol=1e-12,atol=1e-12)
    rows=[]
    for key in LABELS:
        count=s['parameters'].get(key,'---')
        rows.append(row(LABELS[key],count,f'{s["primary"][key]:.3f}',f'{s["timing"][key]["mean_ms"]:.2f}'))
    (HERE/'compact_table.tex').write_text(table(
        'Fresh matched compact comparison: 4,032 primary worlds and 64 separate stress worlds. Lower NMSE is better. Neural scores average three models, not an ensemble. All timings include initialization, use one PyTorch CPU thread and count rank search/refitting.',
        'tab:compact','lrrr',['Method','Weights','NMSE (dB)','Time (ms)'],rows))
    lines=[r'\subsection{Full Compact-Control Results}',r'\input{compact_table.tex}']
    rows=[]
    for key,v in s['contrasts'].items():
        lo,hi=v['simultaneous95_macro_ci']
        rows.append(row(LABELS[key],f'{v["gain_db"]:+.3f}',f'[{lo:+.3f}, {hi:+.3f}]',f'{v["worst_cell_deterioration_db"]:.3f}'))
    lines.append(table('188-parameter gains: positive favors the spectral/state controller. Three predeclared comparisons with Bonferroni-adjusted paired intervals. Worst-cell harm is descriptive, not an upper confidence bound.',
        'tab:compact-contrasts','lrrr',['Comparator','Gain (dB)',r'95\% interval','Worst harm'],rows))
    rows=[]
    for key in LABELS:
        rate=s['primary_threshold_rates'][key]
        rows.append(row(LABELS[key],f'{s["ratio_of_sums_pooled"][key]:.3f}',*[f'{100*rate[str(t)]:.2f}' for t in (.5,1.,2.)]))
    lines.append(table('Different error summaries on the same primary worlds. Pooled NMSE pools linear errors before taking dB and is not the primary score. Threshold rates are percentages.',
        'tab:compact-tails','lrrrr',['Method','Pooled (dB)',r'$>.5$',r'$>1$',r'$>2$'],rows))
    rows=[]
    for key in LABELS:
        rows.append(row(LABELS[key],*[f'{s["cells"][cell][key]["nmse_db"]:.3f}' for cell in ('weak_reference','nearly_noiseless')]))
    lines.append(table('Separate stress tests, 32 paired worlds each. Weak reference: SNR 5/RSR 0 dB. Nearly noiseless: SNR 60/RSR 12 dB. Neither enters selection or the primary average.',
        'tab:compact-stress','lrr',['Method','Weak reference','Nearly noiseless'],rows))
    rows=[]
    for key in ('none','global','adaptive','recurrent'):
        rows.append(row(LABELS[key],*[f'{s["seed_primary"][key][str(seed)]:.3f}' for seed in (9101,9102,9103)]))
    lines.append(table('Primary scores by training initialization. These seeds share the same training data and do not represent independent training populations.',
        'tab:compact-seeds','lrrr',['Method','9101','9102','9103'],rows))
    rows=[]
    for seed in (9101,9102,9103):
        folder=ROOT/'checkpoints'/f'seed_{seed}'
        best=torch.load(folder/'recurrent_best.pt',map_location='cpu',weights_only=False)
        last=torch.load(folder/'recurrent_latest.pt',map_location='cpu',weights_only=False)
        assert last['epoch']==20 and len(last['history'])==20
        rows.append(row(seed,best['epoch'],f'{best["best_score"]:.5f}',f'{last["train_seconds"]/60:.2f}'))
    lines.append(table('Recurrent training records. Time includes validation and parallel-worker contention; it is not an isolated-device training benchmark.',
        'tab:compact-training','rrrr',['Seed','Selected epoch','Held-out NLL','Minutes'],rows))
    audit=s['audit']
    lines.append(f'The audit regenerated all {audit["regenerated_test"]:,} test worlds, replayed {audit["neural_replays"]:,} representative neural predictions and independently replayed {audit["independent_classical_replays"]} classical predictions in NumPy. Maximum absolute replay difference was ${audit["replay_max_abs_error"]:.2e}$. Headline scores were also recomputed with scalar loops. These numerical checks do not establish statistical optimality or global recovery.')
    specs=read(ROOT/'manifest.json')['config']['specs']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'legend.fontsize':7,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.7),layout='constrained')
    snrs=list(range(-10,21,5))
    short=dict(none='Steps (10)',global_='Global (30)',adaptive='Ours (188)',
        recurrent='LSTM (148)',opt_selected='Opt offline',rank_loop='Opt rank')
    short['global']=short.pop('global_')
    for key,style in zip(LABELS,('o--','s--','o-','d-','v:','^:')):
        values=[np.mean([s['cells'][sp['id']][key]['nmse_db'] for sp in specs[:126] if sp['snr']==snr]) for snr in snrs]
        axes[0].plot(snrs,values,style,label=LABELS[key],markersize=3,lw=1)
        axes[1].scatter(s['timing'][key]['mean_ms'],s['primary'][key],s=24)
        offsets={'none':(5,2),'global':(-4,-13),'adaptive':(5,-1),
            'recurrent':(5,5),'opt_selected':(4,5),'rank_loop':(-5,-12)}
        axes[1].annotate(short[key],(s['timing'][key]['mean_ms'],s['primary'][key]),fontsize=7,
            xytext=offsets[key],ha='right' if key in ('global','rank_loop') else 'left',textcoords='offset points')
    axes[0].set(xlabel='SNR (dB)',ylabel='Mean cell NMSE (dB)',xticks=snrs);axes[0].legend(frameon=False,fontsize=6)
    axes[1].set(xlabel='Mean inference time (ms, log scale)',ylabel='Primary macro NMSE (dB)',xscale='log',xlim=(1.1,220),ylim=(-14.8,-9.1))
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(HERE/'figures/compact_comparison.pdf');plt.close(fig)
    lines.append(r'\begin{figure*}[!t]\centering\includegraphics[width=\textwidth]{compact_comparison.pdf}\caption{Fresh compact-control comparison. Left: each SNR point averages 18 reference/path cells. Right: primary accuracy and measured serial inference time. These descriptive summaries do not imply cellwise confidence guarantees.}\label{fig:compact}\end{figure*}')
    (HERE/'compact_supplement_results.tex').write_text('\n\n'.join(lines)+'\n')
    evidence=HERE/'evidence/compact_baseline';evidence.mkdir(parents=True,exist_ok=True)
    for name in ('manifest.json','frozen_selection.json','benchmark.json','summary.json','audit.json','RESULTS.txt','status.json'):
        shutil.copy2(ROOT/name,evidence/name)
    shutil.copy2(REPO/'compact_baseline/PROTOCOL.txt',evidence/'PROTOCOL.txt')
    for seed in (9101,9102,9103):
        shutil.copy2(ROOT/'checkpoints'/f'seed_{seed}'/'history.json',evidence/f'history_{seed}.json')
    print('Generated compact tables, figure and evidence from audited results.')


if __name__=='__main__': main()
