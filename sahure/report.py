"""Paired uncertainty and pilot figures; no model or checkpoint selection."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

LABELS={'em_10':'EM-GS (10)','em_100':'EM-GS (100)','hs_gs':'HS-GS + rank search',
    'rician_100':'Rician GD (100)','rician_converged':'Rician GD (convergence)',
    'amplitude_100':'Amplitude GD (100)','amplitude_converged':'Amplitude GD (convergence)',
    'rician_hard':'Rician GD + hard Hankel','linearized_ls':'Linearized LS',
    'em_none':'Learned EM: no structure','em_global':'Learned EM: global shrinkage',
    'em_adaptive':'Learned EM: adaptive shrinkage','gd_none':'Learned GD: no structure',
    'gd_global':'Learned GD: global shrinkage','gd_adaptive':'Learned GD: adaptive shrinkage',
    'em_constant_control':'EM: adaptive replaced by constants',
    'gd_constant_control':'GD: adaptive replaced by constants'}


def paired_summary(error,reference,energy,indices):
    tiny=np.finfo(float).tiny
    error=np.maximum(error,tiny);reference=np.maximum(reference,tiny)
    nmse=10*np.log10(error.sum()/energy.sum())
    gain=10*np.log10(reference.sum()/error.sum())
    med=np.median(10*np.log10(reference/error))
    boot_error=error[indices].sum(1);boot_ref=reference[indices].sum(1)
    ci=lambda x:np.percentile(x,[2.5,97.5]).tolist()
    return dict(nmse_db=float(nmse),nmse_ci=ci(10*np.log10(boot_error/energy[indices].sum(1))),
        gain_vs_hs_db=float(gain),gain_vs_hs_ci=ci(10*np.log10(boot_ref/boot_error)),
        paired_median_vs_hs_db=float(med),paired_median_vs_hs_ci=ci(np.median(10*np.log10(reference[indices]/error[indices]),axis=1)))


def make_report(out,config):
    out=Path(out)
    rows=[json.loads(x) for x in (out/'trials.jsonl').read_text().splitlines()]
    methods=list(dict.fromkeys(r['method'] for r in rows))
    by={m:sorted([r for r in rows if r['method']==m],key=lambda r:r['trial']) for m in methods}
    count=config['test_cases'];expected=list(range(count))
    for m in methods:
        if [r['trial'] for r in by[m]]!=expected: raise ValueError(f'incomplete or duplicate results: {m}')
    if len(methods)!=17: raise ValueError('pilot requires all 17 learned/control/classical estimators')
    error={m:np.array([r['error_energy'] for r in by[m]]) for m in methods}
    energy=np.array([r['true_energy'] for r in by['hs_gs']])
    indices=np.random.default_rng(99101).integers(count,size=(config['bootstrap_resamples'],count))
    summary={m:paired_summary(error[m],error['hs_gs'],energy,indices) for m in methods}
    hs_time=np.median([r['median_seconds'] for r in by['hs_gs']])
    for m in methods:
        times=np.array([r['median_seconds'] for r in by[m]])
        summary[m].update(median_seconds=float(np.median(times)),time_iqr_seconds=np.percentile(times,[25,75]).tolist(),
            speedup_vs_hs=float(hs_time/np.median(times)))
    frozen=json.loads((out/'frozen_selection.json').read_text());core=frozen['primary_core']
    comparisons={}
    structural_diagnostics={}
    for c in ('em','gd'):
        beta=np.array([r['diagnostics']['beta'] for r in by[c+'_adaptive']])
        correction=np.array([r['diagnostics']['correction'] for r in by[c+'_adaptive']])
        structural_diagnostics[c]=dict(gate_min=float(beta.min()),gate_mean=float(beta.mean()),
            gate_max=float(beta.max()),fraction_gate_above_099=float((beta>.99).mean()),
            mean_first_correction=float(correction[:,0].mean()),mean_last_correction=float(correction[:,-1].mean()))
        for ref in (c+'_global',c+'_none',c+'_constant_control'):
            result=paired_summary(error[c+'_adaptive'],error[ref],energy,indices)
            comparisons[f'{c}_adaptive_vs_{ref}']=dict(gain_db=result['gain_vs_hs_db'],ci=result['gain_vs_hs_ci'])
    result=dict(methods=summary,comparisons=comparisons,structural_diagnostics=structural_diagnostics,primary_core=core,test_cases=count,
        training_seeds=1,bootstrap_resamples=config['bootstrap_resamples'],fixed_snr_db=config['snr'],fixed_rsr_db=config['rsr'])
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plots(out,config,by,error,energy,summary,frozen)
    main=summary[core+'_adaptive'];adapt=comparisons[f'{core}_adaptive_vs_{core}_global']
    positive=main['gain_vs_hs_db']>=.3 and main['gain_vs_hs_ci'][0]>0 and main['speedup_vs_hs']>=2
    efficiency=main['gain_vs_hs_db']>=-.2 and main['speedup_vs_hs']>=2
    verdict=('Promising accuracy and speed at the pilot condition.' if positive else
             'Promising efficiency at the pilot condition, without the combined accuracy target.' if efficiency else
             'The primary adaptive model does not meet either predeclared accuracy–speed target at this pilot condition.')
    total_train=sum(x['training_seconds'] for x in frozen['checkpoints'].values())
    hardware_path=out/'hardware.json'
    hardware=json.loads(hardware_path.read_text()) if hardware_path.exists() else None
    hardware_text=(f' Hardware: {hardware["cpu"]}, {hardware["physical_cores"]} physical cores, '
                   f'{hardware["memory_bytes"]/2**30:g} GiB RAM; see `hardware.json`.') if hardware else ''
    def interval(x):return f'[{x[0]:.3f}, {x[1]:.3f}]'
    lines=['# Self-supervised adaptive Hankel: exploratory pilot', '',verdict,'',
        f'Completed six models, {config["epochs"]} epochs each, and {count} fresh paired test worlds. '
        'There is one training seed, one SNR (5 dB), and one single-user RSR (12 dB). '
        'This is an implementation and feasibility result, not a completed publication experiment.','',
        f'The primary core selected by validation held-out likelihood is **{core.upper()}**. '
        f'Its adaptive model has NMSE **{main["nmse_db"]:.3f} dB**, gain over HS-GS '
        f'**{main["gain_vs_hs_db"]:.3f} dB** (pointwise paired 95% interval '
        f'{interval(main["gain_vs_hs_ci"])}), and **{main["speedup_vs_hs"]:.2f}×** median inference speedup.','',
        '## Results','',
        '| Estimator | NMSE (dB) | Gain vs HS-GS (dB), 95% CI | Median inference (ms) | Speedup vs HS-GS |',
        '|---|---:|---:|---:|---:|']
    for m in sorted(methods,key=lambda m:summary[m]['nmse_db']):
        x=summary[m];lines.append(f'| {LABELS[m]} | {x["nmse_db"]:.3f} | {x["gain_vs_hs_db"]:.3f} {interval(x["gain_vs_hs_ci"])} | {1000*x["median_seconds"]:.2f} | {x["speedup_vs_hs"]:.2f}× |')
    lines += ['', 'Positive gain means lower total squared channel error than the reference. '
        'NMSE is 10 log10(sum error / sum true-channel energy). Intervals use 4,000 paired resamples of whole test worlds. '
        'They do not measure variation across training seeds. Timing is the median across worlds of three batch-one repeats.','',
        '## Does adaptation add value?','',
        '| Adaptive model compared with | Gain (dB) | Paired 95% CI |','|---|---:|---:|']
    for name,x in comparisons.items(): lines.append(f'| {name} | {x["gain_db"]:.3f} | {interval(x["ci"])} |')
    lines += ['', 'The constant-output intervention averages the adaptive network outputs on validation observations using all 30 pilots, matching deployment, '
        'then freezes them per stage. It keeps the learned update scalars. This differs from separately training the global-shrinkage model. '
        'A small gap to this intervention weakens the case that sample-dependent decisions are necessary. '
        'All these comparisons are within the same 3–7-path distribution; held-out-complexity adaptation has not yet been established.','',
        f'The simpler {core.upper()} global-shrinkage model already improves on HS-GS by '
        f'{summary[core+"_global"]["gain_vs_hs_db"]:.3f} dB. The adaptive network adds '
        f'{adapt["gain_db"]:.3f} dB, with interval {interval(adapt["ci"])}. '
        'Therefore, the full gain over HS-GS should not be attributed to measurement-dependent adaptation alone.','',
        'The learned gates stay almost fully open in this pilot: '
        f'EM minimum/mean {structural_diagnostics["em"]["gate_min"]:.4f}/{structural_diagnostics["em"]["gate_mean"]:.4f}; '
        f'GD minimum/mean {structural_diagnostics["gd"]["gate_min"]:.4f}/{structural_diagnostics["gd"]["gate_mean"]:.4f}. '
        'Both exceed 0.99 at every measured user/stage. This does **not** demonstrate learning when to turn structure off. '
        'A fixed-gate ablation and higher-complexity channels remain necessary to assess that claim.','',
        '## Training and measurement protocol','',
        f'- Train/validation/test sizes: {config["train_cases"]:,} / {config["val_cases"]:,} / {count}; disjoint master seeds in `config.json`.',
        '- N=32, K=3, P=30, independent per-user path counts uniform on 3–7. Paper 1 channel, Gaussian pilots, reference and aggregate-SNR calibration reused.',
        '- Each epoch redraws context/held-out pilot sets: 21 context and 9 loss pilots. Initialization and every model feature use context only.',
        '- Validation uses a fixed pilot split and held-out Rician likelihood only. Test inference uses all 30 pilots. No channel-label loss or selection.',
        '- Six models: EM/GD × none/global/adaptive structure. Parameter counts: 10 / 30 / 188. Adam 0.001, batches 32, 10 stages, 20 epochs, float32 training.',
        f'- Total measured training plus per-epoch validation: {total_train/60:.2f} minutes. Data generation, preflight and test evaluation are additional.',
        '- All inference timings include spectral initialization, feature extraction and structural operations. HS-GS timing includes its exhaustive rank search. CPU single-thread, double precision for every estimator.'+hardware_text,
        '- Classical comparisons use the vectorized NumPy/SciPy implementation; learned models use PyTorch. Speed is implementation-specific, not an operation-count theorem.',
        '- Before test evaluation, unused spectral-feature computations were removed from the global controls. All six arms gave bit-identical outputs and parameter gradients on the equivalence fixture. The original training source and both hashes are saved; the live training run retained its original implementation.',
        f'- Global hard-Hankel Rician rank is {frozen["hard_rank"]}, selected by held-out validation NLL (25 candidate updates, one sweep). Final run uses 100 updates. HS-GS retains Paper 1 selection (magnitude residual, 25 candidate updates, four sweeps) and 100 final updates.',
        '- Rician and amplitude gradient references share initialization and 100-step budgets; additional convergence-controlled runs cap at 1,000 and report actual convergence.',
        '- Xu comparison solves the 1-D linearized least-squares objective; it still receives the same exact nonlinear noisy observations. It is not a reproduction of Xu’s 2-D rank-constrained estimator.','',
        '## Numerical verification','',
        'The test suite checks density normalization and SciPy agreement, total complex noise variance, '
        'Wirtinger/autograd/finite-difference agreement, zero and large fields, the EM preconditioner identity, '
        'repository initializer/EM/GD parity, Hankel inversion, solve/SVD shrinkage equivalence, repeated singular values, '
        'zero-gate behavior, user permutation, float32/64 agreement, held-out isolation, truth-file independence, '
        'mask behavior, checkpoint reload, and the fixed-output intervention. See the saved test log.','',
        '## Publication scope and next decision','',
        verdict,
        'The full multi-SNR success criterion cannot be evaluated by this fixed-SNR pilot. '
        'No claim of transfer, hardware validation, supervised equivalence, superiority to URformer, or a learned rank is supported here. '
        'The architecture retains exact Rician data processing and learns a smooth spatial correction without channel labels. '
        'Its empirical contribution must be qualified by the comparison table above.','',
        'The larger balanced training campaign, three-seed repetition, SNR/RSR and complexity sweeps, '
        'transfer tests, supervised comparator, separately trained final-only model and remaining ablations are deferred to the next decision point. '
        'The five pilot figure groups are marked as fixed-condition diagnostics; they are not substitutes for the planned SNR/RSR and transfer figures.','',
        '## Files','',
        '- `config.json`, phase manifests: configuration, seeds, source hashes, software and hardware metadata.',
        '- `frozen_selection.json`: checkpoint hashes, validation decisions and fixed classical settings, saved before test generation.',
        '- `checkpoints/`: best and last models, optimizer state, training histories and constant controls.',
        '- `data/*_observations.npz`: label-free training inputs. `data/*_truth.npz`: evaluation-only labels/diagnostics.',
        '- `trials.jsonl`: every method/world error, energy, repeated timing, rank or per-stage diagnostics.',
        '- `summary.json`: aggregate values and paired uncertainty.',
        '- `figures/`: five pilot figure groups in PDF and PNG.',
        '- `../../sahure/MATHEMATICS.md`, `LITERATURE.md`, and `README.md`: derivation, overlap assessment and reproduction commands.','']
    for name in ('rician_converged','amplitude_converged'):
        converged=sum(r['diagnostics']['converged'] for r in by[name])
        lines.insert(lines.index('## Numerical verification'),f'{LABELS[name]} reached the declared stopping tolerance in {converged}/{count} trials. Remaining trials hit the cap; no global optimum is claimed.\n')
    (out/'REPORT.md').write_text('\n'.join(lines))


def plots(out,config,by,error,energy,summary,frozen):
    os.environ.setdefault('MPLCONFIGDIR',str((out/'mpl_cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.dpi':180,'axes.grid':True,'grid.alpha':.18})
    directory=out/'figures';directory.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(directory/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(directory/f'{name}.png',bbox_inches='tight');plt.close(fig)
    names=sorted(summary,key=lambda m:summary[m]['nmse_db'])
    fig,ax=plt.subplots(figsize=(10,7))
    values=np.array([summary[m]['nmse_db'] for m in names]);bounds=np.array([summary[m]['nmse_ci'] for m in names])
    ax.errorbar(values,np.arange(len(names)),xerr=np.maximum(0,np.stack((values-bounds[:,0],bounds[:,1]-values))),fmt='o',color='#225ea8',capsize=3)
    ax.set_yticks(range(len(names)),[LABELS[m] for m in names]);ax.invert_yaxis()
    ax.set_xlabel('NMSE (dB), lower is better');ax.set_title(f'Pilot accuracy: SNR 5 dB, single-user RSR 12 dB\n{config["test_cases"]} paired trials; pointwise 95% bootstrap intervals')
    save(fig,'01_fixed_condition_accuracy')
    rho=np.array([np.mean(r['rho']) for r in by['hs_gs']]);edges=np.quantile(rho,[0,.25,.5,.75,1])
    fig,ax=plt.subplots(figsize=(7,4.5))
    for name in ('em_adaptive','gd_adaptive','em_global','gd_global'):
        xx=[];yy=[];lo=[];hi=[]
        for j in range(4):
            mask=(rho>=edges[j])&((rho<=edges[j+1]) if j==3 else (rho<edges[j+1]))
            xx.append(np.mean(rho[mask]));yy.append(10*np.log10(error['hs_gs'][mask].sum()/error[name][mask].sum()))
            sample=np.random.default_rng(3200+j).integers(mask.sum(),size=(config['bootstrap_resamples'],mask.sum()))
            interval=np.percentile(10*np.log10(error['hs_gs'][mask][sample].sum(1)/error[name][mask][sample].sum(1)),[2.5,97.5])
            lo.append(interval[0]);hi.append(interval[1])
        line,=ax.plot(xx,yy,'o-',label=LABELS[name]);ax.fill_between(xx,lo,hi,color=line.get_color(),alpha=.10)
    ax.axhline(0,color='k',lw=.8);ax.set_xlabel('Mean true normalized effective rank (evaluation only)')
    ax.set_ylabel('Gain over HS-GS (dB)');ax.set_title('Pilot complexity diagnostic: four equal-count bins\nPointwise 95% bands; not a complexity-transfer test');ax.legend(fontsize=8)
    save(fig,'02_within_distribution_complexity')
    fig,ax=plt.subplots(figsize=(9,5))
    chosen=['em_10','em_100','hs_gs','rician_100','rician_hard','em_none','em_global','em_adaptive','gd_none','gd_global','gd_adaptive']
    for j,name in enumerate(chosen):
        x=summary[name];ax.scatter(1000*x['median_seconds'],x['nmse_db'],s=65,
            marker=('o','s','^','D')[j//3],label=LABELS[name])
    ax.set_xscale('log');ax.set_xlabel('Median batch-one inference time (ms, log scale)');ax.set_ylabel('NMSE (dB)')
    ax.set_title('Accuracy and computation at the pilot condition\nInitialization and HS-GS rank selection included');ax.margins(.12)
    ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),fontsize=8,frameon=False)
    save(fig,'03_accuracy_runtime')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    learned=list(frozen['checkpoints'])
    for name in learned:
        history=json.loads((out/'checkpoints'/f'{name}_history.json').read_text())
        axes[0].plot([x['epoch'] for x in history],[x['val_nll'] for x in history],label=LABELS[name])
    axes[0].set_xlabel('Epoch');axes[0].set_ylabel('Validation held-out NLL / measurement');axes[0].legend(fontsize=7)
    xx=np.arange(3);width=.35
    for j,core in enumerate(('em','gd')):
        values=np.array([summary[core+'_'+s]['gain_vs_hs_db'] for s in ('none','global','adaptive')])
        bounds=np.array([summary[core+'_'+s]['gain_vs_hs_ci'] for s in ('none','global','adaptive')])
        axes[1].bar(xx+(j-.5)*width,values,width,label=core.upper(),
            yerr=np.maximum(0,np.stack((values-bounds[:,0],bounds[:,1]-values))),capsize=3)
    axes[1].set_xticks(xx,['No structure','Global','Adaptive']);axes[1].axhline(0,color='k',lw=.8)
    axes[1].set_ylabel('Gain over HS-GS (dB)');axes[1].legend()
    fig.suptitle('Self-supervised training and structural controls\nOne seed; supervised comparison not yet run');fig.tight_layout()
    save(fig,'04_training_structural_controls')
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,core in enumerate(('em','gd')):
        diag=[r['diagnostics'] for r in by[core+'_adaptive']]
        beta=np.array([x['beta'] for x in diag]);tau=np.array([x['tau'] for x in diag]);corr=np.array([x['correction'] for x in diag])
        axes[row,0].plot(np.arange(1,11),beta.mean((0,2)),label='Mean gate')
        axes[row,0].plot(np.arange(1,11),corr.mean((0,2)),label='Mean relative correction')
        axes[row,0].set_xlabel('Stage');axes[row,0].set_title(core.upper()+': gate and actual correction');axes[row,0].legend(fontsize=8)
        axes[row,1].plot(np.arange(1,11),tau.mean((0,2)),'o-');axes[row,1].set_xlabel('Stage');axes[row,1].set_title('Mean absolute threshold')
        axes[row,2].scatter(rho,corr[:,-1].mean(1),s=9,alpha=.5)
        axes[row,2].set_xlabel('Mean true normalized effective rank');axes[row,2].set_ylabel('Final-stage relative correction')
    fig.suptitle('Structural diagnostics at SNR 5 dB\nNo claim that the gate estimates rank; transfer tests deferred');fig.tight_layout()
    save(fig,'05_structural_diagnostics')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/sahure_pilot')
    args=parser.parse_args();out=Path(args.output);make_report(out,json.loads((out/'config.json').read_text()))
