"""Build manuscript tables/figures from audited, frozen results (no fitting)."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NAMES = {'learned': 'Compact estimator (188 weights)',
         'opt_selected': 'Offline OptShrink, selected placement',
         'opt_loop': 'Offline OptShrink, in-loop',
         'opt_final': 'Offline OptShrink, final-only',
         'rank_joint': 'Online rank + placement selection',
         'rank_loop': 'Online rank, in-loop',
         'rank_final': 'Online rank, final-only',
         'hs_one': 'HS-Rician, one sweep', 'em100': 'EM-GS, 100 updates'}
NAMES.update({f'opt_budget{t}':f'OptShrink, {t} updates' for t in (5,10,20,50,100)})
NAMES['opt_old']='Original narrow OptShrink lookup'

def read(path):
    return json.loads(path.read_text())

def audited(name):
    root = REPO / 'results' / name
    assert read(root/'status.json')['state'] == 'complete'
    s = read(root/'summary.json')
    assert s['audit']['status'] == 'passed'
    a = np.load(root/'all_errors.npz', allow_pickle=False)
    e, p = a['squared_error'], a['channel_power']
    assert e.shape[0:2] == (2,128) and e.shape[-1] == 16
    assert np.isfinite(e).all() and np.isfinite(p).all() and (p > 0).all()
    names = a['names'].tolist()
    for n, value in s['primary'].items():
        ii = [i for i,k in enumerate(names) if k.startswith('learned_')] if n == 'learned' else [names.index(n)]
        # Deliberately scalar across cells and worlds, independent of analysis einsums.
        score = np.mean([10*np.log10(sum(float(e[b,c,m,j]) for b in range(2) for j in range(16))/
                      sum(float(p[b,c,j]) for b in range(2) for j in range(16))) for c in range(126) for m in ii])
        np.testing.assert_allclose(score,value,atol=1e-12,rtol=1e-12)
    return root,s

def fmt(v):
    return f'{v:.3f}'

def row(*cols):
    return ' & '.join(map(str,cols)) + r'\\' + '\n'

def table(caption,label,columns,header,rows,wide=False):
    env='table*' if wide else 'table'
    return (f'\\begin{{{env}}}[!t]\n\\caption{{{caption}}}\n\\label{{{label}}}\\centering\\footnotesize\n'
            r'\setlength{\tabcolsep}{4pt}'+'\n'+f'\\begin{{tabular}}{{@{{}}{columns}@{{}}}}\\toprule\n'+
            row(*header)+r'\midrule'+'\n'+''.join(rows)+r'\bottomrule'+'\n'+
            f'\\end{{tabular}}\n\\end{{{env}}}\n')

def main():
    roots,stats=zip(*(audited(n) for n in ('fair_optshrink','optshrink_rank_check')))
    fair,rank=stats
    specs=read(roots[0]/'manifest.json')['config']['specs']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'axes.titlesize':9,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42})
    snrs=list(range(-10,21,5)); paths=[None,2,4,8,12,16]
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.45),layout='constrained',sharey=True)
    gains=[]
    for s,comparator in ((fair,'opt_selected'),(rank,'rank_loop')):
        gains.append(np.array([[np.mean([s['cells'][sp['id']][comparator]['nmse_db']-s['cells'][sp['id']]['learned']['nmse_db']
            for sp in specs[:126] if sp['snr']==snr and sp['L']==l]) for snr in snrs] for l in paths]))
    limit=np.ceil(max(abs(g).max() for g in gains)*2)/2
    for ax,g,title in zip(axs,gains,('A. Broad offline tuning','B. Fresh online-rank test')):
        im=ax.imshow(g,cmap='RdBu',norm=TwoSlopeNorm(0,vmin=-limit,vmax=limit),aspect='auto')
        ax.set(xticks=range(7),xticklabels=snrs,yticks=range(6),yticklabels=['3-7','2','4','8','12','16'],xlabel='SNR (dB)',title=title)
        for i in range(6):
            for j in range(7):
                ax.text(j,i,f'{g[i,j]:+.1f}',ha='center',va='center',fontsize=7,color='white' if abs(g[i,j])>limit*.65 else 'black')
    axs[0].set_ylabel('Paths per user')
    fig.colorbar(im,ax=axs,shrink=.87,label='Learned gain (dB); positive is better')
    fig.savefig(HERE/'figures/latest_gains.pdf');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(6.5,2.7),layout='constrained')
    for ax,root,title,comparator in zip(axs,roots,('A. Broad offline tuning','B. Online in-loop rank selection'),('opt_selected','rank_loop')):
        data=np.load(root/'error_distributions.npz',allow_pickle=False)
        for name,label,color in [('learned','Compact','#0072B2'),(comparator,'OptShrink selector','#D55E00'),('hs_one','HS-Rician','#009E73')]:
            values=data[name]; x=np.r_[0,values]; y=np.r_[1,1-np.arange(1,len(values)+1)/len(values)]
            ax.step(x,y,where='post',label=label,color=color,lw=1.2)
        for t in (.5,1,2): ax.axvline(t,color='.75',ls=':',lw=.7)
        ax.set(title=title,xlabel='Per-world linear NMSE',ylabel='Fraction above threshold',xlim=(0,2.1),ylim=(0,1))
        ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(HERE/'figures/latest_distributions.pdf');plt.close(fig)

    fig,ax=plt.subplots(figsize=(3.2,2.4),layout='constrained')
    keys=[f'opt_budget{t}' for t in (5,10,20,50,100)]
    ax.plot([fair['timing'][k]['mean_ms'] for k in keys],[fair['primary'][k] for k in keys],'o-',color='#D55E00',label='Fixed-budget OptShrink')
    for t,k in zip((5,10,20,50,100),keys): ax.annotate(str(t),(fair['timing'][k]['mean_ms'],fair['primary'][k]),xytext=(3,5),textcoords='offset points',fontsize=7)
    for k,mark,color,label in [('learned','*','#0072B2','Compact'),('opt_selected','s','#009E73','Offline selected')]:
        ax.scatter(fair['timing'][k]['mean_ms'],fair['primary'][k],marker=mark,color=color,s=65,label=label,zorder=5)
    ax.set(xlabel='Mean CPU time (ms, log scale)',ylabel='Macro NMSE (dB); lower is better',xscale='log',ylim=(-14,-9.2))
    ax.xaxis.set_major_locator(FixedLocator([2,5,10,20,50]))
    ax.xaxis.set_major_formatter(FixedFormatter(['2','5','10','20','50']))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=.2);ax.legend(frameon=False,loc='upper right')
    fig.savefig(HERE/'figures/latest_cost.pdf');plt.close(fig)

    rows=[]
    for k,label in NAMES.items():
        if k.startswith('opt_budget') or k=='opt_old': continue
        vals=[]
        for s in stats:
            vals.extend([fmt(s['primary'][k]),f"{s['timing'][k]['mean_ms']:.2f}"] if k in s['primary'] else ['---','---'])
        rows.append(row(label,*vals))
    main_table=table('Broad comparisons on separate fresh banks. Study A tests offline tuning; Study B adds online rank selection on a new bank. Each study has 4,032 primary worlds and 64 excluded stress worlds. Lower NMSE is better. Times use the same PyTorch CPU backend and include selection/refitting. Learned metrics average three frozen models, not an ensemble.',
        'tab:broad','lrrrr',['Method','A: NMSE (dB)','Time (ms)','B: NMSE (dB)','Time (ms)'],rows,True)
    (HERE/'latest_table.tex').write_text(main_table)

    # Supplementary results preserve every comparator, aggregation and declared contrast.
    lines=[r'\section{Broad OptShrink Comparisons: Full Numerical Record}',
      'Study A tunes OptShrink on the original 2,048 validation observations and masks. Study B inherits that lookup and selects rank from each new block of observations. Neither study retrains the compact estimator. Each has two independent test banks, 126 primary cells and two separate stress cells. All 30 pilots are reused for the final fit, without additional observations. The inference lookup uses supplied noise and reference strength, not true channel power or path count.',
      'Study A uses test-bank seeds 202610210001 and 202610310001; Study B uses 202610610001 and 202610710001. Each cell adds 100 times its zero-based index to the bank seed. Both use 16 worlds per cell and bank. The selected-model seeds are 9101, 9102 and 9103. Sources, settings and checkpoints are hashed before test generation. Settings remain frozen after the test data are inspected.',
      'The primary score averages 126 per-cell dB ratio-of-sums NMSE values, after pooling the two equal-sized test banks within each cell. It is not the dB of one pooled error ratio. Learned scores average the three separate model scores. Bootstrap resampling preserves pairing within each bank/cell; confidence intervals condition on these weights and the offline validation choices. They do not cover repeated training or offline selection uncertainty.',
      r'\subsection{Declared Contrasts and All Comparators}',
      'The numerical tables separate the declared average-dB contrasts, alternative error summaries, individual models, test banks and stress cases. These summaries are not interchangeable: a gain in one average does not establish a gain in every cell or at every error threshold. All individual-cell scores are retained in the evidence records.']
    rows=[]
    for code,s,comparators in [('A',fair,['opt_selected','opt_loop','opt_final']),('B',rank,['rank_joint','rank_loop','rank_final'])]:
        for k in comparators:
            d=s['contrasts'][k];ci=d['simultaneous95_macro_ci']
            rows.append(row(code,NAMES[k],fmt(d['gain_db']),f'[{ci[0]:.3f}, {ci[1]:.3f}]',fmt(d['worst_cell_deterioration_db'])))
    lines.append(table(r'Learned gain (positive favors learning), three-contrast-adjusted 95\% intervals within each study, and largest observed primary-cell deterioration. Cellwise safety is not established.',
        'tab:broad-ci','llrrr',['Study','Comparator','Gain (dB)','Adjusted interval','Worst harm (dB)'],rows,True))
    for code,s in [('A',fair),('B',rank)]:
        rows=[row(NAMES.get(k,k.replace('_',r'\_')),fmt(v),f"{s['timing'][k]['mean_ms']:.2f}") for k,v in s['primary'].items()]
        lines.append(table(f'Study {code}: all primary averages and mean end-to-end CPU time. Fixed-budget labels denote EM updates, not matched runtime.',
            'tab:all-'+code,'lrr',['Method','NMSE (dB)','Time (ms)'],rows))
    rows=[]
    for code,s,comp in [('A',fair,'opt_selected'),('B',rank,'rank_loop')]:
        for k in ('learned',comp,'hs_one','em100'):
            rates=s['primary_threshold_rates'][k]
            rows.append(row(code,NAMES[k],fmt(s['ratio_of_sums_pooled'][k]),fmt(s['mean_case_linear'][k]),
                *[f'{100*rates[str(t)]:.2f}' for t in (.5,1.,2.)]))
    lines.append(table('Alternative summaries on primary worlds. Pooled NMSE is in dB; per-case mean NMSE is linear. Last three columns are percentages above the fixed thresholds. Zero observed exceedances do not imply zero population risk.',
        'tab:broad-tails','llrrrrr',['Study','Method','Pooled','Case mean','$>0.5$','$>1$','$>2$'],rows,True))
    lines.append(r'''\begin{figure*}[!t]
\centering\includegraphics[width=\textwidth]{latest_distributions.pdf}
\caption{Primary empirical exceedance curves over the displayed threshold range, separately by study. Learned curves average the three model-specific empirical distributions on the same worlds; predictions are not independent channels. All values, including larger errors, are retained in the evidence arrays. A lower rate at one threshold does not imply a lower rate at every threshold.}
\end{figure*}
\begin{figure}[!t]
\centering\includegraphics[width=\columnwidth]{latest_cost.pdf}
\caption{Study A accuracy--cost curve. Labels are fixed update budgets, each tuned separately for placement, rank and mixing. The selected lookup may use different budgets by noise/reference pair. No equal-runtime optimum is claimed.}
\end{figure}''')
    rows=[]
    for code,s in [('A',fair),('B',rank)]:
        for seed,val in s['learned_seed_primary'].items(): rows.append(row(code,'Model '+seed,fmt(val)))
        for k in ('learned','opt_selected') + (('rank_joint','rank_loop') if code=='B' else ()):
            for b,val in enumerate(s['bank_primary'][k]): rows.append(row(code,NAMES[k]+f', bank {b+1}',fmt(val)))
    lines.append(table('Training-seed and test-bank summaries, kept distinct. Bank dB scores need not average to the score obtained after pooling their linear errors.',
        'tab:broad-seeds','llr',['Study','Subset','NMSE (dB)'],rows,True))
    rows=[]
    for code,s,comp in [('A',fair,'opt_selected'),('B',rank,'rank_loop')]:
        for label,key,vals in [('SNR','snr',snrs),('RSR','rsr',[5,12,20]),('Paths','L',paths)]:
            for v in vals:
                ids=[sp['id'] for sp in specs[:126] if sp[key]==v]
                lc=np.mean([s['cells'][i]['learned']['nmse_db'] for i in ids]); cc=np.mean([s['cells'][i][comp]['nmse_db'] for i in ids])
                rows.append(row(code,label+' '+('3--7' if v is None else str(v)),fmt(lc),fmt(cc),fmt(cc-lc)))
    # Split into one column per study instead of a page-breaking table.
    for code in ('A','B'):
        rr=[r for r in rows if r.startswith(code+' &')]
        lines.append(table(f'Study {code}: descriptive marginal averages over the other primary conditions. OptShrink is '+('offline-selected' if code=='A' else 'online in-loop rank selection')+'. These are not additional independent tests.',
            'tab:marginal-'+code,'llrrr',['Study','Condition','Compact','OptShrink','Gain'],rr))
    rows=[]
    for code,s in [('A',fair),('B',rank)]:
        for cell in ('weak_reference','nearly_noiseless'):
            for k in ('learned','opt_selected','em100')+(('rank_joint','rank_loop') if code=='B' else ()):
                rows.append(row(code,'Weak reference' if cell=='weak_reference' else 'Nearly noiseless',NAMES[k],fmt(s['cells'][cell][k]['nmse_db'])))
    lines.append(table('Stress conditions, excluded from primary averages. Weak reference: SNR 5/RSR 0 dB. Nearly noiseless: SNR 60/RSR 12 dB. Both use random three-to-seven-path channels.',
        'tab:broad-stress','lllr',['Study','Condition','Method','NMSE (dB)'],rows,True))
    lines += [r'\subsection{Frozen Settings and Numerical Verification}',
       'The online selector searches a shared rank, not separate ranks for each user. Its mixing strength and iteration count are inherited from the matching in-loop or final-only row in the offline lookup. Thus the rank comparison does not exhaust all possible classical policies. Candidate initialization uses only the first 21 pilots. After scoring the other nine, the winner starts again from all 30. Joint selection evaluates both placements and counts both searches in timing. Search work shared while generating several experimental arms is not used to understate standalone timing.']
    choices=read(roots[0]/'frozen_selection.json')['choices'];rows=[]
    for key,cfg in choices.items():
        vals=[]
        for p in ('loop','final'): vals += [str(cfg[p]['rank']),str(cfg[p]['strength']),str(cfg[p]['iterations'])]
        rows.append(row(key.replace(':',' / '),*vals,cfg['selected']['placement']))
    lines.append(table('Broad offline settings. Within each placement, columns give rank, mixing factor and update count. The selected column chooses between those placements. Rank zero denotes no filtering.',
        'tab:settings-broad','lrrrrrrl',['SNR / RSR','Loop $r$',r'$\beta$','$T$','Final $r$',r'$\beta$','$T$','Selected'],rows,True))
    for code,root,s in zip(('A','B'),roots,stats):
        audit=s['audit'];replays=audit.get('representative_replays',audit.get('representative_prediction_replays'))
        error=audit.get('independent_rank_max_abs_error',audit.get('independent_classical_max_abs_error'))
        mantissa,exponent=f'{error:.2e}'.split('e')
        error_tex='$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
        lines.append(f'Study {code}: all {audit["regenerated_test"]:,} test worlds regenerated exactly; {replays:,} representative predictions replayed. Maximum recorded independent '+
          ('rank-selector' if code=='B' else 'classical')+f' absolute output difference was {error_tex}. An independent scalar-loop calculation reproduced every primary score. These checks test the saved implementation and data, not an external reproduction of the study.')
    lines.append('Study A also regenerated all 2,048 validation observations and replayed selection from saved scores. Study B independently checked both rank-score vectors, chosen settings and all-pilot refits with NumPy/SciPy. Its new audit did not independently replay HS-Rician again; that backend check is in Study A. Both studies passed finite-output, source-hash, checkpoint-hash and freeze-before-generation checks. No failed trial was removed. Full per-cell values and source records are supplied as JSON evidence in the manuscript archive; full observation and prediction arrays remain in the experimental archive.')
    (HERE/'latest_supplement.tex').write_text('\n\n'.join(lines)+'\n')
    provenance={}
    for root,s in zip(roots,stats):
        for name in ('summary.json','manifest.json','frozen_selection.json','audit.json','RESULTS.txt','status.json','all_errors.npz','error_distributions.npz'):
            path=root/name
            shutil.copy2(path,HERE/'evidence'/f'{root.name}_{name}')
            provenance[str(path.relative_to(REPO))]=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol=REPO/root.name/'PROTOCOL.txt'
        shutil.copy2(protocol,HERE/'evidence'/f'{root.name}_PROTOCOL.txt')
        provenance[str(protocol.relative_to(REPO))]=hashlib.sha256(protocol.read_bytes()).hexdigest()
    for name in ('hybrid_optshrink','hybrid_optshrink_repair','merged_followup','hankel_error_check'):
        path=REPO/'results'/name/'RESULTS.txt';shutil.copy2(path,HERE/'evidence'/f'{name}_RESULTS.txt')
        provenance[str(path.relative_to(REPO))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/'evidence'/'figure_source_hashes.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Built evidence-linked figures/tables; both primary calculations independently matched.')

if __name__ == '__main__':
    main()
