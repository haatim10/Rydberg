"""Produce the report only after validation selection and independent confirmation."""
from pathlib import Path
import os,json,time
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'matplotlib_cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit import CFG,sha,write,verify_protected,generators,grid

def table(headers,rows):
    clean=lambda x:str(x).replace('|',r'\|')
    return '\n'.join(['| '+' | '.join(map(clean,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(clean,r))+' |' for r in rows])
def label(spec):
    return 'Independent rays' if spec['distribution']=='independent' else spec['distribution'].capitalize()+' '+str(spec['cluster_spread_deg'])+'°'
def main():
    assert json.loads((ROOT/'compute_complete.json').read_text())['passed']
    x=json.loads((ROOT/'numerical_results.json').read_text());checks=json.loads((ROOT/'preflight_checks.json').read_text())
    assert checks['passed']
    assert not x['training_gate_passed'],'Training gate passed: perform the authorized training before issuing the final report.'
    specs=generators();names=[s['name'] for s in specs];chosen=x['validation_selection']['selected_generator']['name']
    vals={a['generator']['name']:a for a in x['validation']};conf={a['generator']['name']:a for a in x['confirmation']}
    structures={(r['split'],r['generator']['name']):r for r in x['structure']};covs={r['generator']['name']:r for r in x['covariances']}
    cells={(r['split'],r['generator']['name'],r['P'],r['snr_db']):r for r in x['rows']}
    reference=x['published'];get=lambda split,name,p,s,m:cells[split,name,p,s]['metrics'][m]['nmse_db']
    counts=dict(validation_generators=len(vals),validation_cells=sum(r['split']=='validation' for r in x['rows']),confirmation_cells=sum(r['split']=='confirmation' for r in x['rows']),classical_passes=sum(v['classical_pass'] for v in vals.values()),clustered_both_passes=sum(vals[n]['passes_both'] for n in names if n!='independent'))
    assert counts['clustered_both_passes']==0,'A candidate needs its requested neural follow-up.'
    verdicts=dict(clustering_reduces_conditional_dimension='confirmed',independent_generator_has_white_covariance_or_no_learnable_structure='rejected',tested_cluster_spreads_make_global_covariance_genie_LMMSE_reach_all_published_targets='rejected',published_URformer_gain_is_caused_by_clustered_subrays='inconclusive; a nonlinear estimator can exploit per-channel cluster structure that global LMMSE cannot use')
    followup=dict(status='not_triggered',reason='No clustered generator passes the requested combined classical-match and global-covariance LMMSE gate.',overfit_64_samples='not_run',v3_three_seed_retraining='not_run',v1_encoder_sensitivity='not_run')
    x.update(verdicts=verdicts,counts=counts,preflight_checks=checks,task4=followup,protected_files_unchanged=verify_protected(),estimator_scope='The locked configuration uses the phrase linear posterior. The implemented estimator is linear MMSE from second moments, not the exact Bayesian posterior mean for the non-Gaussian clustered-channel mixture. It minimizes ordinary MSE within the linear class; evaluation uses mean per-channel NMSE.')
    write('summary.json',x)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    # All validation curves; confirmation is highlighted separately in the report.
    colors=plt.get_cmap('viridis')(np.linspace(.07,.92,5));styles={}
    for s in specs:
        if s['name']=='independent':styles[s['name']]=dict(color='#202020',ls='-',lw=2.3)
        else:styles[s['name']]=dict(color=colors[CFG['spread_degrees'].index(s['cluster_spread_deg'])],ls='-' if s['distribution']=='laplacian' else '--',lw=1.2)
    fig,axs=plt.subplots(2,2,figsize=(11.6,8.2))
    for row,figure in enumerate(('fig3','fig4')):
        pts=[r for r in reference[figure] if figure=='fig4' or r['x'] in CFG['snr_db']]
        xx=[r['x'] for r in pts]
        for col,method in enumerate(('GS100','EM100')):
            ax=axs[row,col]
            for spec in specs:
                yy=[get('validation',spec['name'],20,int(r['x']),method) if figure=='fig3' else get('validation',spec['name'],int(r['x']),5,method) for r in pts]
                ax.plot(xx,yy,**styles[spec['name']])
            ax.plot(xx,[r[method] for r in pts],'o',color='#cc4134',ms=5,zorder=10,label='Published (approx.)')
            ax.set(title=('GS, 100 iterations' if col==0 else 'EM-GS, 100 iterations')+(' | 20 pilots' if row==0 else ' | 5 dB SNR'),xlabel='SNR (dB)' if row==0 else 'Pilots',ylabel='NMSE (dB)',xticks=xx);ax.grid(alpha=.2)
    from matplotlib.lines import Line2D
    legend=[Line2D([0],[0],color='#202020',lw=2.3,label='Independent rays'),Line2D([0],[0],marker='o',color='#cc4134',ls='',label='Published (approx.)')]+[Line2D([0],[0],color=c,lw=2,label=f'{s}°') for c,s in zip(colors,CFG['spread_degrees'])]
    fig.legend(handles=legend,loc='lower center',ncol=7,frameon=False,bbox_to_anchor=(.5,.025),fontsize=9)
    fig.text(.5,.012,'Solid: Laplacian. Dashed: Gaussian. All curves use 2,000 validation channels per generator and condition.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.105,1,1));fig.savefig(ROOT/'classical_overlay.png',dpi=180);fig.savefig(ROOT/'classical_overlay.svg');plt.close(fig)
    # The requested oracle screen, distinct from classical matching.
    fig,axs=plt.subplots(1,2,figsize=(11.6,4.6))
    for ax,figure in zip(axs,('fig3','fig4')):
        pts=[r for r in reference[figure] if figure=='fig4' or r['x'] in CFG['snr_db']];xx=[r['x'] for r in pts]
        for spec in specs:
            yy=[get('validation',spec['name'],20,int(r['x']),'genie_LMMSE') if figure=='fig3' else get('validation',spec['name'],int(r['x']),5,'genie_LMMSE') for r in pts]
            ax.plot(xx,yy,**styles[spec['name']])
        yy=[get('validation','independent',20,int(r['x']),'genie_LS') if figure=='fig3' else get('validation','independent',int(r['x']),5,'genie_LS') for r in pts]
        ax.plot(xx,yy,':',color='#737373',lw=2,label='Genie LS, independent rays')
        ax.plot(xx,[r['URformer'] for r in pts],'o-',color='#cc4134',lw=1.8,label='Published URformer (approx.)')
        ax.set(title='20 pilots' if figure=='fig3' else 'SNR = 5 dB',xlabel='SNR (dB)' if figure=='fig3' else 'Pilots',ylabel='NMSE (dB)',xticks=xx);ax.grid(alpha=.2)
    fig.legend(handles=[Line2D([0],[0],color='#202020',label='Global-covariance genie LMMSE, all 11 generators'),Line2D([0],[0],color='#737373',ls=':',label='Genie LS, independent rays'),Line2D([0],[0],color='#cc4134',marker='o',label='Published URformer (approx.)')],loc='lower center',ncol=1,frameon=False,fontsize=9,bbox_to_anchor=(.5,.005))
    fig.tight_layout(rect=(0,.22,1,1));fig.savefig(ROOT/'oracle_screen.png',dpi=180);fig.savefig(ROOT/'oracle_screen.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11.6,4.2));xs=CFG['spread_degrees']
    for d,color in [('laplacian','#2368a2'),('gaussian','#ce7e26')]:
        ss=[s for s in specs if s['distribution']==d]
        axs[0].plot(xs,[structures['validation',s['name']]['conditional_effective_rank_entropy']['mean'] for s in ss],'o-',color=color,label=d.capitalize())
        axs[1].plot(xs,[covs[s['name']]['effective_rank_entropy'] for s in ss],'o-',color=color,label=d.capitalize())
    axs[0].axhline(structures['validation','independent']['conditional_effective_rank_entropy']['mean'],color='#333',ls='--',label='Independent rays')
    axs[1].axhline(covs['independent']['effective_rank_entropy'],color='#333',ls='--',label='Independent rays')
    for ax,title in zip(axs,['Within each channel (given its ray angles)','Covariance across many random channels']):
        ax.set(title=title,xlabel='Angular spread (degrees, standard deviation)',ylabel='Entropy effective rank',xscale='log',xticks=xs,xticklabels=xs,ylim=(0,32));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'spatial_structure.png',dpi=180);plt.close(fig)
    selected_spec=x['validation_selection']['selected_generator'];selected_label=label(selected_spec)
    lines=['# URformer cluster audit','',
      '**Result: none of the ten clustered generators passes the requested LMMSE screen. Neural training was therefore not triggered.** Clustering clearly makes each individual channel simpler, but the global covariance does not learn where the clusters are in a new channel. This result does not rule out gains from a nonlinear learned estimator.','',
      f'The audit evaluated eleven generators: independent rays, five Laplacian spreads and five Gaussian spreads. Each used a separate covariance estimate from 20,000 MIMO draws and 2,000 validation draws. The selected negative case ({selected_label}) and the independent-ray control were then checked on 2,000 new confirmation draws. The original `reproduction_ensemble` noise definition was retained.','',
      f'{counts["classical_passes"]} of the 11 generators meet the earlier classical-curve tolerance. None of the ten clustered generators meets the combined classical and LMMSE target requirements.','',
      '## Verdicts','',
      table(['Claim','Verdict','Reason'],[
       ['Small spreads create simpler channels','Confirmed','The effective dimension within each channel falls sharply.'],
       ['The independent-ray channel is spatially white and leaves no prior to learn','Rejected','Uniform physical angles produce correlated antenna responses, even with independent rays.'],
       ['Clustered rays make global-covariance genie LMMSE reach the requested published targets','Rejected for the tested settings','No spread passes all target values; complete per-point gaps are below.'],
       ['Clustering explains the published URformer gain','Inconclusive','The requested global LMMSE screen does not test a nonlinear, per-channel cluster estimator.']]),'',
      '## What was held fixed and what changed','',
      'There are 32 antennas, 4 users, 4 clusters per user and 10 rays per cluster. Cluster centres are drawn independently for every channel and user from uniform angles between −90° and 90°. Each offset has the requested standard deviation; Laplacian scale is spread/sqrt(2). The Gaussian standard deviation is the spread itself. Angles are reflected into the allowed interval using arcsin(sin(angle)). This reflection preserves the steering response. It also preserves the uniform marginal angle distribution for symmetric offsets.','',
      'Path gains remain independent CN(0,1), with channel scaling sqrt(32/40). Expected power per channel entry remains 32; individual channels are not forced to a fixed norm. Pilots, reference power, noise samples and CN(0,1) starting estimates are paired across generators. Noise power uses the same expected signal power as the earlier validated audit. Both classical algorithms run 100 iterations.','',
      'All covariance estimates use a separate seed stream: 20,000 independent channel matrices, each with four identically distributed users. The common spatial covariance is estimated by pooling their 80,000 user vectors. Separate per-user estimates are also saved. The population mean is known to be zero; no validation or confirmation labels enter the covariance estimate.','',
      'The genie observes the full complex field, so its phase is known. LS uses the pilot pseudoinverse. LMMSE uses the estimated population covariance and the full pilot Gram matrix, including correlations between user estimates. It does not receive the current channel’s cluster centres or ray angles. The implementation agrees with a separate full-matrix LMMSE calculation; details are in preflight_checks.json.','',
      'LMMSE here means a linear estimate based on second moments. It is not the exact Bayesian posterior mean for the clustered-channel mixture. It is designed for ordinary mean squared error and is scored using the same mean per-channel NMSE as all the other methods.','',
      '## Why clustered channels can have the same global covariance','',
      'Within one channel, nearby rays point in similar directions, so a few spatial patterns carry most of its energy. But the cluster centres move from one channel to the next. Averaging over all their positions restores a broad global covariance. Independent zero-mean path gains remove cross-ray terms from that covariance.','',
      'For this uniform physical-angle distribution and half-wavelength spacing, the population covariance is R[m,n] = 32 J0(pi(m−n)). It is the same for every tested spread under the stated reflection rule. This follows from the angular integral and [Bessel’s integral representation](https://dlmf.nist.gov/10.9.E1). The adjacent-antenna correlation is about −0.304, so this covariance is not white. Separate empirical estimates show small Monte Carlo differences, which are retained rather than rescaled away.','',
      '**A failed global-LMMSE screen is not an impossibility bound.** A nonlinear estimator could infer cluster locations from each observation and exploit structure that the global linear estimator misses. A genie supplied with the true current angles would be a different, more informed benchmark. We did not silently substitute that benchmark for the one requested.','',
      '## Spatial rank and effective rank','',
      'A single user channel is a vector and has matrix rank one by itself. To measure its spatial structure, the table reports the rank of its 32-by-40 steering matrix and the power effective rank of its conditional covariance given those ray angles. Separately, it reports the rank of the global covariance estimated from the 20,000-draw set. These quantities do not depend on SNR or pilot count.','',
      'Numerical steering rank uses max(32,40) times machine epsilon times its largest singular value. Effective rank is exp(−sum p log p), where p contains normalized covariance eigenvalues. Participation rank, 1/sum(p²), is also reported. Small numerical singular values can count toward rank while contributing almost no energy; effective rank is more useful for that distinction.','',
      table(['Generator','Conditional numerical rank, mean','Conditional entropy rank, mean','Conditional participation rank, mean','Global numerical rank','Global entropy rank','Mean entry power, covariance set'],[[label(s),f"{structures['validation',s['name']]['conditional_steering_numerical_rank']['mean']:.3f}",f"{structures['validation',s['name']]['conditional_effective_rank_entropy']['mean']:.3f}",f"{structures['validation',s['name']]['conditional_effective_rank_participation']['mean']:.3f}",covs[s['name']]['numerical_rank'],f"{covs[s['name']]['effective_rank_entropy']:.3f}",f"{covs[s['name']]['mean_per_entry_power']:.3f}"] for s in specs]),'',
      '![Conditional and global spatial rank](spatial_structure.png)','',
      'Full rank counts, 5th/95th percentiles, per-channel values and the actual 32-by-4 channel matrix ranks are saved in structure/ and summary.json. Global covariance matrices and their spectra are saved in covariance/.','',
      '## Validation screening and selection','',
      'The classical match tolerance is unchanged: joint curve RMSE at most 1.5 dB and maximum absolute error at most 3 dB, across both methods and both published figures. The requested LMMSE gate requires reaching or beating every target: Figure 3 at 5 and 20 dB, and all six Figure 4 pilot counts. The stricter Figure 4 reading at P = 20 is retained separately. The primary gate adds no digitization allowance; a ±0.2 dB reading sensitivity is also recorded.','',
      table(['Generator','Classical RMSE (dB)','Classical max error (dB)','Classical pass','LMMSE at 5 dB, P20','LMMSE at 20 dB, P20','Worst LMMSE target gap (dB)','Both pass'],[[label(s),f"{vals[s['name']]['classical_rmse_db']:.3f}",f"{vals[s['name']]['classical_max_error_db']:.3f}",vals[s['name']]['classical_pass'],f"{get('validation',s['name'],20,5,'genie_LMMSE'):.3f}",f"{get('validation',s['name'],20,20,'genie_LMMSE'):.3f}",f"{vals[s['name']]['worst_lmmse_target_gap_db']:.3f}",vals[s['name']]['passes_both']] for s in specs]),'',
      'Positive target gaps mean the genie LMMSE error is worse than the published URformer value. Lower NMSE is better. The gate is applied to the mean of per-channel linear errors, converted to dB afterward. The published values are approximate figure readings inherited unchanged from the SNR audit.','',
      '**Published Figure 3 reference, 20 pilots (approximate NMSE in dB)**','',
      table(['SNR (dB)','GS100','EM-GS100','URformer'],[[r['x'],r['GS100'],r['EM100'],r['URformer']] for r in reference['fig3'] if r['x'] in CFG['snr_db']]),'',
      '**Published Figure 4 reference, 5 dB SNR (approximate NMSE in dB)**','',
      table(['Pilots','GS100','EM-GS100','URformer'],[[r['x'],r['GS100'],r['EM100'],r['URformer']] for r in reference['fig4']]),'',
      '![All classical validation curves](classical_overlay.png)','',
      '![Global LMMSE against published URformer](oracle_screen.png)','',
      f'No clustered candidate qualified. The prespecified fallback selected **{selected_label}** solely as the closest nonpassing case among classical matches. Its worst LMMSE target gap was {vals[chosen]["worst_lmmse_target_gap_db"]:.3f} dB on validation. This was a negative confirmation, not a declaration of success. The choice was written to validation_selection.json before generating the confirmation observations. No candidate was reselected from confirmation results.','',
      '## Independent confirmation','',
      table(['Generator','Classical RMSE (dB)','Classical max error (dB)','Classical pass','Worst LMMSE target gap (dB)','Both pass'],[[label(a['generator']),f"{a['classical_rmse_db']:.3f}",f"{a['classical_max_error_db']:.3f}",a['classical_pass'],f"{a['worst_lmmse_target_gap_db']:.3f}",a['passes_both']] for a in x['confirmation']]),'',
      table(['Generator','Conditional numerical rank, mean','Conditional entropy rank, mean','Entry power, mean'],[[label(a['generator']),f"{structures['confirmation',a['generator']['name']]['conditional_steering_numerical_rank']['mean']:.3f}",f"{structures['confirmation',a['generator']['name']]['conditional_effective_rank_entropy']['mean']:.3f}",f"{structures['confirmation',a['generator']['name']]['channel_per_entry_power']['mean']:.3f}"] for a in x['confirmation']]),'']
    for name in (chosen,'independent'):
        spec=next(s for s in specs if s['name']==name)
        lines += [f'**{label(spec)} — confirmation NMSE (dB)**','',table(['Pilots','SNR (dB)','GS100','EM-GS100','Genie LS','Genie LMMSE'],[[p,snr,*[f"{get('confirmation',name,p,snr,m):.4f}" for m in ('GS100','EM100','genie_LS','genie_LMMSE')]] for p,snr in grid()]),'']
    lines += ['## Full validation values','',
      'Each row below uses 2,000 channels. P = 20 / SNR = 5 dB appears once in these results and twice in the published-figure comparisons. Ranks above apply to all these rows.','']
    for spec in specs:
        name=spec['name'];lines += [f'**{label(spec)} — validation NMSE (dB)**','',table(['Pilots','SNR (dB)','GS100','EM-GS100','Genie LS','Genie LMMSE'],[[p,snr,*[f"{get('validation',name,p,snr,m):.4f}" for m in ('GS100','EM100','genie_LS','genie_LMMSE')]] for p,snr in grid()]),'']
    lines+=['## LMMSE target gaps at every requested published point','',
      'Each number is computed LMMSE NMSE minus the published URformer NMSE, in dB. Positive values fail the target.','']
    headers=['Generator']+[('Fig3 '+str(r['snr_db'])+' dB' if r['figure']=='fig3' else 'Fig4 P'+str(r['P'])) for r in vals['independent']['targets']]
    lines += [table(headers,[[label(s),*[f"{r['gap_db']:.3f}" for r in vals[s['name']]['targets']]] for s in specs]),'',
      'All per-point GS/EM absolute errors against published values are recorded in summary.json under validation and confirmation. The raw mean-normalized errors for every method and channel are stored in cells/.','',
      '## Calibration at P = 20, SNR = 5 dB','',
      table(['Generator','Mean entry power','Realized aggregate SNR (dB)','Realized aggregate RSR (dB)','Distance of global covariance from analytic law'],[[label(s),f"{cells['validation',s['name'],20,5]['calibration']['mean_per_entry_channel_power']:.3f}",f"{cells['validation',s['name'],20,5]['calibration']['aggregate_realized_snr_db']:.3f}",f"{cells['validation',s['name'],20,5]['calibration']['aggregate_RSR_db']:.3f}",f"{covs[s['name']]['relative_distance_to_uniform_angle_theory']:.4f}"] for s in specs]),'',
      'The covariance distance is a relative Frobenius norm. Finite-sample powers fluctuate because channels are not normalized individually. All estimators receive the same known noise variance and reference.','',
      '## Task 4: conditional training','',
      table(['Requested experiment','Status','Reason'],[['64-sample overfit test','Not run','No spread passed the requested screen.'],['v3 retraining: 3 seeds, 20,000 samples, 50 epochs','Not run','No spread passed the requested screen.'],['v1 encoder sensitivity: FFN(LN(A))+A','Not run','The conditional training branch was not triggered.']]),'',
      'No new claim about URformer accuracy, v1 versus v3, or trainability follows from this audit. The smallest change needed to test the broader clustering explanation would be a benchmark that can adapt to each channel’s cluster locations, rather than screening only through one global covariance. That is a different follow-up, and has not been added here.','',
      '## Reproducibility and checks','',
      f'Master seed: {CFG["master_seed"]}. Covariance, validation and confirmation use distinct split keys {CFG["splits"]["covariance"]}, {CFG["splits"]["validation"]} and {CFG["splits"]["confirmation"]}. All cases use float64/complex128. Three workers, each with one numerical thread, evaluate independent generators. The fixed pilot seed is unchanged from the earlier study.','',
      f'Independent checks passed: full-matrix versus separable LMMSE, GS/EM versus the earlier PyTorch implementation, zero-spread rank, exact independent-ray formula, noise calibration, and the reflected-angle covariance identity. All {x["protected_files_unchanged"]} protected files, including the complete SNR audit, were unchanged.','',
      'Run from the repository root with the existing environment: `../../.venv-paper2/bin/python -B experiments/urformer_cluster_audit/checks.py`, then `audit.py`, `report.py` and `finalize.py` from the same directory. Completed numerical studies refuse silent reruns. Source/configuration hashes are locked before validation generation. Every covariance sample stream has a hash, and final file hashes are recorded in manifest.json.','',
      'The approximately 95% Monte Carlo intervals in summary.json describe sampling error for each mean NMSE. They are not simultaneous intervals or uncertainty in the published figure readings. Selection used validation means only. Confirmation did not fit a covariance, tune a spread or change a threshold.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write('report_complete.json',dict(passed=True,neural_training_performed=False,report_sha256=sha(ROOT/'REPORT.md'),summary_sha256=sha(ROOT/'summary.json'),finished_unix=time.time()))
    print('Report generated. Selected negative confirmation:',chosen,'validation and confirmation gaps:',vals[chosen]['worst_lmmse_target_gap_db'],conf[chosen]['worst_lmmse_target_gap_db'],flush=True)

if __name__=='__main__':main()
