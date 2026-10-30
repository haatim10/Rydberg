"""Build report from the locked non-learning audit; no new training or selection."""
from pathlib import Path
import os,json,hashlib,time
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'matplotlib_cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import CFG,REPO,write,sha,snapshot

def table(headers,rows):
    clean=lambda x:str(x).replace('|',r'\|')
    return '\n'.join(['| '+' | '.join(map(clean,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(clean,r))+' |' for r in rows])

def main():
    assert json.loads((ROOT/'compute_complete.json').read_text())['passed']
    assert json.loads((ROOT/'independent_checks.json').read_text())['passed']
    o=json.loads((ROOT/'task1_oracle.json').read_text());c=json.loads((ROOT/'task23_classical.json').read_text())
    refs=json.loads((ROOT/'reference/curve_readings.json').read_text());versions=json.loads((ROOT/'version_audit.json').read_text())
    selected=c['validation_ranking'][0]['convention'];confirmed={r['convention']:r for r in c['confirmation_ranking']}
    normal=confirmed['reproduction_ensemble'];shift=confirmed['paper_shifted']
    accepted=[r['convention'] for r in c['validation_ranking'] if r['within_predeclared_tolerance']]
    # A common 15dB shift is rejected only if both shifted readings are decisively worse.
    rejection=all(confirmed[n]['rmse_db']>normal['rmse_db']+8 for n in ('paper_shifted','unit_channel_assumed')) and normal['rmse_db']<3
    assert rejection,'Audit did not settle the shared-shift question; inspect before stopping.'
    prior=json.loads((REPO/'results/urformer_reproduction_20261022/summary.json').read_text()) if (REPO/'results/urformer_reproduction_20261022/summary.json').exists() else {}
    prior_test=[]
    for r in prior['rows']:
        if r['P']==20 and r['snr'] in (5,20):
            published=next(v['URformer'] for v in refs['curves']['fig3'] if v['x']==r['snr'])
            prior_test.append(dict(snr_db=r['snr'],final_nmse_db=r['URformer_final']['nmse_db'],validation_selected_nmse_db=r['URformer_best']['nmse_db'],published_approx_db=published))
    prior_test.sort(key=lambda r:r['snr_db'])
    training=[];histories=[]
    for seed in (2026102201,2026102202,2026102203):
        h=json.loads((REPO/f'results/urformer_reproduction_20261022/training/{seed}/history.json').read_text())['history'];histories.append(h)
        best=min(h,key=lambda r:r['validation_nmse']);last=h[-1]
        training.append(dict(seed=seed,epochs=len(h),best_recorded_epoch=best['epoch'],best_validation_db=10*np.log10(best['validation_nmse']),final_validation_db=10*np.log10(last['validation_nmse']),first_training_nmse_db=10*np.log10(h[0]['train_nmse']),final_training_nmse_db=10*np.log10(last['train_nmse'])))
    secondary=dict(verdict='inconclusive',observed_validation_deterioration='confirmed in existing supervised training histories; consistent with overfitting',records=training,
        caution='A worse final validation score establishes deterioration after the best saved epoch; it does not prove the cause or explain the full published gap. Single-SNR training and loss comparisons were not performed in this audit.')
    summary=dict(config=CFG,array_factor_db=float(10*np.log10(CFG['M'])),published=refs,task1=o,task2_and_3=c,task6=versions,independent_checks=json.loads((ROOT/'independent_checks.json').read_text()),SNR_preserving_scale_check=json.loads((ROOT/'scaling_check.json').read_text()),
        hypotheses=dict(common_M_factor_SNR_shift=dict(verdict='rejected',scope='A shared shift applied consistently to GS,EM-GS and the learned estimator.',confirmation_existing_rmse_db=normal['rmse_db'],confirmation_shifted_rmse_db=shift['rmse_db']),
        neural_only_normalization=dict(verdict='inconclusive',reason='Classical plots cannot establish the data/noise normalization used by a separate neural training pipeline.'),
        SNR_preserving_amplitude_scaling=dict(verdict='inconclusive',reason='Scaling H, B, W and the starting estimate together preserves SNR and classical normalized error, but can affect the training of the neural blocks. This audit tests noise conventions, not this separate scaling question.'),
        training_secondary=secondary,unreachable_published_accuracy=dict(verdict='inconclusive',reason=o['reachability'])),
        selection=dict(validation_winner=selected,validation_tolerance_passes=accepted,confirmation_used_for_selection=False,reading_uncertainty_db=.2),
        tasks4_5=dict(status='not_run_under_user_stop_rule',reason='Task 3 rejects the proposed common shift; the closest scale is the already tested unshifted family. No supported normalization correction calls for repeating full neural training. Task5 diagnostics remain explicitly pending.',
          earlier_training_reused_only_for_reporting=True),
        earlier_training=training,earlier_P20_test_results=prior_test)
    write('summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(11.5,8.4))
    styles=[('reproduction_ensemble','#2168a5','-','Existing / vector SNR'),('ours','#258a51','--','Realized signal power'),('paper_shifted','#d98122','-','Noise /32, realized power'),('unit_channel_assumed','#ab6b9e','--','Noise /32, expected power'),('array_signal_scalar_noise','#b83c3c',':','Array signal / scalar noise')]
    for row,figure in enumerate(('fig3','fig4')):
        pts=[r for r in refs['curves'][figure] if figure=='fig4' or r['x'] in CFG['snr_db']]
        for col,method in enumerate(('GS100','EM100')):
            ax=axs[row,col];x=[r['x'] for r in pts]
            ax.plot(x,[r[method] for r in pts],'ko',ms=5,label='Published (approx.)',zorder=8)
            for convention,color,ls,label in styles:
                ys=[]
                for r in pts:
                    p,s=(20,r['x']) if figure=='fig3' else (r['x'],5)
                    cell=next(v for v in c['rows'] if v['split']=='confirmation' and v['convention']==convention and v['P']==p and v['snr_db']==s)
                    ys.append(cell['metrics'][method]['nmse_db'])
                ax.plot(x,ys,color=color,ls=ls,lw=1.8,label=label)
            ax.set(title=method+(' | 20 pilots' if row==0 else ' | nominal 5 dB'),xlabel='Nominal SNR (dB)' if row==0 else 'Number of pilots',ylabel='NMSE (dB)',xticks=x);ax.grid(alpha=.2)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,.025),frameon=False)
    fig.text(.5,.008,'2,000 fresh confirmation channels per point. Convention ranking fixed using separate validation channels.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.13,1,1));fig.savefig(ROOT/'baseline_overlay.png',dpi=180);fig.savefig(ROOT/'baseline_overlay.svg');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11.5,3.8),sharey=True)
    for ax,h,rec in zip(axs,histories,training):
        ax.plot([r['epoch'] for r in h],[10*np.log10(r['train_nmse']) for r in h],label='Training (online epoch mean)')
        ax.plot([r['epoch'] for r in h],[10*np.log10(r['validation_nmse']) for r in h],label='Validation (end of epoch)')
        ax.axvline(rec['best_recorded_epoch'],ls=':',color='gray');ax.set(title=f"Existing run {rec['seed']}",xlabel='Epoch',ylabel='NMSE (dB)');ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'existing_training_curves.png',dpi=170);plt.close(fig)
    lines=['# URformer SNR audit','',
       '**Verdict: the classical curves reject a common +15.05 dB SNR shift as the explanation for the reproduction gap.** The data identify the unshifted SNR family, not the authors\' exact implementation. Neural-only normalization and training causes remain unresolved.','',
       'In simple terms: if every method had received 32 times less noise, the ordinary GS and EM-GS methods would also have improved by about 15 dB. They do not show that improvement in the published figures. Our original noise setting is much closer to those figures. Changing the noise for every method would therefore make the reproduction less faithful.','',
       f'This audit used 2,000 independent oracle channels, 2,000 validation channels and 2,000 separate confirmation channels. All six conventions used the same paired channels, unit noise, pilot bank and random CN(0,1) starting estimates within each split. GS and EM-GS each ran 100 iterations. Mean per-channel linear NMSE was averaged before converting to dB.','',
       'All new outputs are in this directory. Earlier studies and checkpoints were read only. No neural training was performed.','',
       '## Hypothesis verdicts','',
       table(['Hypothesis','Verdict','Evidence/limit'],[
        ['Shared +15.05 dB SNR shift','Rejected',f"Confirmation baseline RMSE: existing {normal['rmse_db']:.3f} dB; noise/32 {shift['rmse_db']:.3f} dB."],
        ['Correct vector-power SNR creates a factor M','Rejected analytically','M cancels between total signal power and total noise power.'],
        ['Neural pipeline used a different normalization from classical baselines','Inconclusive','Authors code or sample data needed; classical curves alone cannot settle this.'],
        ['Input amplitude was rescaled while preserving SNR','Inconclusive','Scaling channel, reference, noise and starting estimate together can leave classical NMSE unchanged but affect neural training.'],
        ['Training explains a further fixed 3-5 dB gap','Inconclusive','Earlier histories show training improvement with validation deterioration; neither the cause nor a universal gap reduction is established.'],
        ['Published -21.6 dB is mathematically unreachable at our 5 dB','Inconclusive','It beats the tested genie LS estimators, but these are not universal Bayesian lower bounds.']]),'',
       '## Task 1: oracle references under the existing generator','',
       table(['SNR (dB)','Genie complex LS','K/((P-K) SNR)','Exact fixed-pilot expectation','True-DoA projected LS'],[[r['snr_db'],f"{r['genie_LS']['nmse_db']:.4f}",f"{r['requested_random_pilot_theory_db']:.4f}",f"{r['fixed_pilot_conditional_theory_db']:.4f}",f"{r['oracle_DoA_projection']['nmse_db']:.4f}"] for r in o['rows']]),'',
       'Entries are NMSE in dB. The requested theory expression assumes an ensemble of random Gaussian pilots and normalized expected channel energy. Our reproduction uses one fixed pilot matrix and mean per-channel error ratios. Its exact conditional reference is E[M sigma^2 tr((SS^H)^-1)/||H||_F^2]. The numerical and exact conditional values agree within sampling error.','',
       'With 40 independent rays per user and 32 antennas, the true steering span is almost always the entire antenna space. The tiny projection gain is therefore expected. Numerical rank uses max(M,40) times machine epsilon times the largest singular value. Rank counts across 8,000 user channels: '+str(o['steering_rank_counts'])+'.','',
       '**Reachability:** -21.6 dB at nominal 5 dB is below both tested oracle-LS errors. That makes a prior/distribution/normalization audit useful, but does not prove impossibility. Gaussian/cluster priors can improve on LS even when the steering span is full rank. The task\'s proposed "ceiling" is not a valid ceiling on all learned estimators.','',
       '## Task 2: explicit SNR conventions','',
       'Let q(H)=mean|HS|^2, q_bar=32||S||_F^2/P and rho=10^(SNR_dB/10). The same known reference B=sqrt(10 q_bar) is used throughout, so ensemble RSR is 10 dB. Individual realized RSR varies; the reported ratio excludes the reference from signal power.','',
       table(['Argument','Noise variance sigma^2','Meaning'],[
        ['reproduction_ensemble','q_bar/rho','Actual frozen reproduction convention.'],
        ['ours','q(H)/rho','Requested per-realization convention; differs from the frozen reproduction.'],
        ['paper_vector','M q_bar/(M rho)=q_bar/rho','Proper vector signal/noise ratio; exactly equivalent to existing convention.'],
        ['paper_shifted','q(H)/(32 rho)','Requested +15.05 dB hypothesis.'],
        ['unit_channel_assumed','q_bar/(32 rho)','Expected-power analogue: noise computed as if channel entries had unit power.'],
        ['array_signal_scalar_noise','32 q_bar/rho','Array-summed signal divided by per-antenna noise; produces -15.05 dB, not +15.05 dB.']]),'',
       'Per-entry channel power is near 32 under our unit-variance path-gain assumption. That fact alone does not shift SNR: both generated signal power and the noise calibration already include it. Both signs of an inconsistent array/scalar convention were tested. None is attributed to the authors without evidence. The scalar meaning in the paper is not sufficient to establish their actual generator.','',
       'A separate question remains: multiplying the channel, reference, noise and initial estimate by the same constant preserves SNR and the classical normalized error. An independent check using a scale of 1/sqrt(32) confirmed this to numerical precision; see scaling_check.json. Neural training can still react to that change in input size. Therefore this audit rejects a shared noise/SNR shift; it does not rule out every possible channel or network-input normalization difference.','',
       'Calibration below uses the confirmation split, P = 20, nominal 5 dB. Per-case values are means of realized dB ratios; aggregate SNR/RSR are ratios of summed powers. Full distributions and every point are in summary.json.','',
       table(['Convention','Mean |H|^2','Mean realized SNR dB','Aggregate SNR dB','Mean realized RSR dB','Aggregate RSR dB'],[[n,*[f'{v:.4f}' for v in (r['calibration']['per_entry_channel_power']['mean'],r['calibration']['per_antenna_snr_realized_noise_db']['mean'],r['calibration']['aggregate_realized_snr_db'],r['calibration']['realized_RSR_db']['mean'],r['calibration']['aggregate_RSR_db'])]] for n in CFG['conventions'] for r in c['rows'] if r['split']=='confirmation' and r['convention']==n and r['P']==20 and r['snr_db']==5]),'',
       '## Task 3: classical curves and convention selection','',
       'Convention ranking used validation draws only, with a predeclared descriptive tolerance: joint RMSE ≤ 1.5 dB and maximum absolute error ≤ 3 dB across both methods and both figures. This is a coarse reproduction check, not proof of identical settings. Approximate digitization uncertainty is 0.2 dB. The P = 20 / SNR = 5 dB location has separate readings in Figures 3 and 4; all comparison rows retain their figure identity.','',
       table(['Convention','Validation RMSE','Confirmation RMSE','Confirmation mean abs error','Confirmation max abs error','Validation tolerance'],[[r['convention'],f"{r['rmse_db']:.3f}",f"{confirmed[r['convention']]['rmse_db']:.3f}",f"{confirmed[r['convention']]['mean_abs_error_db']:.3f}",f"{confirmed[r['convention']]['max_abs_error_db']:.3f}",str(r['within_predeclared_tolerance'])] for r in c['validation_ranking']]),'',
       f'Validation winner: `{selected}`. This selection was saved before confirmation draws were evaluated. Small differences within the unshifted family do not identify the authors\' exact normalization. The large shifted-curve mismatch rejects a single common +15 dB explanation.','',
       '![Classical baseline overlays](baseline_overlay.png)','',
       '### Published readings','']
    for figure,points in refs['curves'].items():
        lines += [figure+' (approximate values):','',table(['SNR dB' if figure=='fig3' else 'Pilots','GS','EM-GS100','URformer'],[[r['x'],r['GS100'],r['EM100'],r['URformer']] for r in points]),'']
    lines+=['Figure 4 at P = 15 shows about −20.2 dB for URformer. The v1 prose claimed −25 dB; v3 corrects it to −20 dB, consistent with the plot.','',
        '### Absolute dB error at every requested point','']
    for convention in CFG['conventions']:
        lines += [f'**{convention}, confirmation draws**','',table(['Figure','P','SNR','Method','Computed NMSE','Published approx.','Absolute error'],[[r['figure'],r['P'],r['snr_db'],r['method'],f"{r['measured_db']:.4f}",r['published_approx_db'],f"{r['absolute_error_db']:.4f}"] for r in confirmed[convention]['points']]),'']
    lines += ['## Tasks 4-5: stopping decision and training evidence','',
       'The user requested stopping when a step clearly settles the normalization question. Task 3 rejects a shared +15 dB shift. The unshifted scale already has completed three-seed, 20,000-example, 50-epoch supervised training in the earlier reproduction. Repeating it would not test a newly supported normalization correction. We therefore stopped before new neural training. This does not claim that the entire URformer reproduction gap has been solved.','',
       table(['Requested task','Status','What is available'],[
        ['4: retrain under matching convention','Not rerun under stop rule','Earlier unshifted supervised runs completed; no new normalization was supported.'],
        ['5a: 64-sample overfit test','Not run','Still a useful follow-up; failing −30 dB alone would not uniquely prove a code bug.'],
        ['5b: per-SNR train/validation curves','Not run','Existing aggregate epoch curves plotted below; per-SNR epoch histories were not saved.'],
        ['5c: single-SNR versus mixed training','Not run','Cannot attribute the gap to mixed-SNR training yet.'],
        ['5d: loss variants','Not run','No test-based loss selection.'],
        ['5e: FilterNet/Transformer ablations and filter curve','Not run','Architecture-version and module tests remain pending.']]),'',
       'Earlier supervised results at 20 pilots, averaged across three seeds (no new training or evaluation):','',
       table(['SNR (dB)','Final weights NMSE (dB)','Validation-selected weights NMSE (dB)','Published URformer, approximate (dB)'],[[r['snr_db'],f"{r['final_nmse_db']:.4f}",f"{r['validation_selected_nmse_db']:.4f}",r['published_approx_db']] for r in prior_test]),'',
       'These earlier test results are reported for context only. They did not select the normalization convention in this audit.','',
       'Existing supervised validation evidence (no new training):','',
       table(['Seed','Epochs','Best epoch among saved histories','Best validation NMSE dB','Final validation NMSE dB','First training NMSE dB','Final training NMSE dB'],[[r['seed'],r['epochs'],r['best_recorded_epoch'],*[f"{r[k]:.4f}" for k in ('best_validation_db','final_validation_db','first_training_nmse_db','final_training_nmse_db')]] for r in training]),'',
       '![Existing aggregate training curves](existing_training_curves.png)','',
       'Training values are online minibatch means during an epoch; validation uses fixed weights at its end. They are not identical evaluation procedures. These histories show validation deterioration after early epochs, but do not establish a particular 3–5 dB benefit from changing the loss, or prove a bug. In this audit, no neural fitting occurred, and H entered only simulation, scoring and oracle diagnostics.','',
       '## Task 6: version audit','',
       'Sources: [v1 HTML](https://arxiv.org/html/2509.12586v1) and [v3 PDF](https://arxiv.org/pdf/2509.12586v3), with a local copy in reference/v3.pdf. The latest PDF pages 3–4 were rendered and inspected.','',
       table(['Item','v1','v3 / current code'],[
        ['FFN normalization','FFN(LN(A))+A','FFN(A)+A; current code follows v3.'],
        ['Initialization','Not specified for URformer','Random complex Gaussian, normalized variance; paper-setting code uses CN(0,1).'],
        ['FilterNet','Small MLP','Two-layer scalar network, output in (0,1); hidden width/activation still assumptions.'],
        ['Layer indices','Gate t; filter/Former unindexed','Stage t-1 indices explicit; prose still has g(t). Current code uses separate stage weights.'],
        ['Training details','20,000 samples','Adds Adam, cosine schedule over 50 epochs, batch size 32, learning rate 0.001.'],
        ['Transduction','Constants normalized away','Explicit known G; current code divides it out.'],
        ['Channel notation','Mentions unused large-scale beta','Drops beta sentence; tilde(h)-to-h normalization still unstated.'],
        ['Pilot count','P = 20 in table','P = 20 in Figure 3 text.'],
        ['Figure 4 target at 15 pilots','Prose says −25 dB','Prose corrected to −20 dB; plot about −20.2 dB.']]),'',
       'The SNR/RSR expressions, steering scaling, user-token layout, residual addition, NMSE objective and 100 classical iterations are unchanged. Full implementation notes are in version_audit.json. The missing FFN normalization is a real architectural difference, not evidence that either printed version was used for the published curves. Ask the authors which encoder produced the figures before treating it as the cause.','',
       '## Smallest useful next checks','',
       'Keep the current noise definition. First run the 64-sample fitting check to see whether the network can learn a tiny dataset at all. Next compare the two printed encoder versions using the same training data and a separate validation set. If both can fit the small dataset, compare training at 5 dB alone and 20 dB alone with mixed-SNR training. These tests isolate basic learning problems before a larger set of loss and module experiments. They have not been run in this audit.','',
       '## Suggested paper paragraph','',
       f'We audited the normalization in our independent URformer implementation using paired classical baselines and 2,000 separate confirmation channels. The existing per-antenna power definition and a consistent vector-power definition are equivalent. Their classical-curve RMSE against approximate published readings was {normal["rmse_db"]:.2f} dB, whereas dividing noise variance by 32 increased it to {shift["rmse_db"]:.2f} dB. Thus a shared 15.05 dB SNR shift does not explain the reproduction gap. Unspecified channel/training details and a difference in feed-forward layer normalization between paper versions remain unresolved. We report our implementation as an independent reimplementation and do not infer superiority over the published URformer results.','',
       '## Questions for the authors','',
       '1. Could you share the data generator or one seeded batch at nominal 5 dB, including H,S,B,sigma^2 and W, and the measured powers used to calculate SNR/RSR? Is exactly the same generator used for neural and classical results?',
       '2. How are path gains scaled, and is the displayed tilde(h) normalized before becoming h? Are channel estimates normalized before entering the neural blocks and rescaled afterward? Are subrays independent or drawn around a shared cluster center; what angular spread and array spacing are used?',
       '3. What pilot distribution, energy and reuse/refresh policy are used? Is the reference a constant positive field or a complex spatial vector?',
       '4. Do you train one mixed-SNR model or separate models for each SNR and pilot count? What training/validation distributions and checkpoint selection rule are used?',
       '5. Which encoder generated the figures: v1\'s FFN(LN(A))+A or v3\'s FFN(A)+A? Please specify head count, FFN/output-layer dimensions, activations, dropout, stage weight sharing and gate initialization.',
       '6. What do normalized initial-channel variance and its redraw policy mean in code? Are additional optimizer settings, clipping, warm starts or intermediate losses used?','',
       '## Reproducibility and verification','',
       'Run from the repository root using the existing numerical environment: `../../.venv-paper2/bin/python -B experiments/urformer_snr_audit/run.py`, then `checks.py`, `scaling_check.py`, `report.py` and `finalize.py` (same directory). Completed study files intentionally prevent silent reruns. Configuration, source and file hashes are retained. Seed 2026102701 determines new data splits; the unchanged pilot-bank seed is 2026102240 with stream 77. No new experiment uses an earlier final-test draw.','',
       'Independent NumPy versus frozen PyTorch 100-step GS/EM checks passed. Proper vector and existing generators are bitwise identical; variance-factor signs were verified. An independent 10,000-pilot Wishart check confirms the requested random-pilot theory under its assumptions.','',
       'A startup raster-color mismatch was corrected before data generation; the failed log and original receipt are retained. Hashes of the selected earlier source files, checkpoints and results were checked before and after the audit; see protected_verification.json for the count. `summary.json` contains every tabulated numeric result, and per-channel errors are in `cells/`.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    questions=lines[lines.index('## Questions for the authors')+2:lines.index('## Reproducibility and verification')]
    (ROOT/'author_questions.txt').write_text('\n'.join(questions).strip()+'\n')
    assert snapshot()==json.loads((ROOT/'protected_before.json').read_text())
    write('complete.json',dict(passed=True,main_hypothesis='rejected_for_common_shift',neural_training_performed=False,report_sha256=sha(ROOT/'REPORT.md'),summary_sha256=sha(ROOT/'summary.json'),finished_unix=time.time()))
    print('Report complete. Validation winner:',selected,'confirmation existing/shift RMSE:',normal['rmse_db'],shift['rmse_db'],flush=True)

if __name__=='__main__':main()
