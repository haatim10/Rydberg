"""Report fixed comparisons; bootstrap worlds, keep training seeds paired."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from .protocol import ROOT,CONFIG,write_json,sha


def ci(values): return np.percentile(values,[2.5,97.5]).tolist()


def seed_nmse_and_bootstrap(errors,energy,counts):
    """Do not turn predictions from multiple training seeds into extra worlds."""
    errors=np.atleast_2d(errors)
    return 10*np.log10(errors.sum(1)/energy.sum()),10*np.log10((errors@counts.T)/(counts@energy)[None])


def summarize():
    frozen=json.loads((ROOT/'frozen_selection.json').read_text());core=frozen['primary_core']
    methods=list(dict.fromkeys(x['method'] for x in frozen['models'].values()))+frozen['classical']
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);summary={};boots={}
    for spec in frozen['cells']:
        directory=ROOT/'cells'/spec['id']
        if not (directory/'status.json').exists(): raise RuntimeError(f'incomplete: {spec["id"]}')
        with np.load(directory/'hs_gs.npz') as f: energy=f['energy'].copy()
        errors={}
        for method in methods:
            paths=[directory/f'{name}.npz' for name,x in frozen['models'].items() if x['method']==method]
            if not paths: paths=[directory/f'{method}.npz']
            arrays=[]
            for p in paths:
                with np.load(p) as f:
                    np.testing.assert_array_equal(f['energy'],energy);arrays.append(f['error'].copy())
            errors[method]=np.stack(arrays)
        counts=rng.multinomial(len(energy),np.full(len(energy),1/len(energy)),size=CONFIG['bootstrap_resamples']).astype(float)
        estimates={m:seed_nmse_and_bootstrap(e,energy,counts) for m,e in errors.items()}
        nmse={m:v[0] for m,v in estimates.items()};bsnmse={m:v[1] for m,v in estimates.items()}
        cell={};boot={}
        for m in methods:
            gains=nmse['hs_gs']-nmse[m];bgains=(bsnmse['hs_gs']-bsnmse[m]).mean(0)
            cell[m]=dict(nmse_db=float(nmse[m].mean()),nmse_ci=ci(bsnmse[m].mean(0)),
                seed_nmse_db=nmse[m].tolist(),seed_nmse_sd=float(nmse[m].std(ddof=1)) if len(nmse[m])>1 else None,
                gain_vs_hs_db=float(gains.mean()),gain_vs_hs_ci=ci(bgains),seed_gain_vs_hs_db=gains.tolist())
            boot[m]=bgains
        comparisons={}
        for c in ['em','gd']:
            for ref in [f'{c}_global',f'{c}_none',f'{c}_constant_control']:
                differences=nmse[ref]-nmse[f'{c}_adaptive']
                bdiff=(bsnmse[ref]-bsnmse[f'{c}_adaptive']).mean(0)
                comparisons[f'{c}_adaptive_vs_{ref}']=dict(gain_db=float(differences.mean()),ci=ci(bdiff),seed_gains=differences.tolist())
        records=[json.loads(line) for line in (directory/'classical.jsonl').read_text().splitlines()]
        convergence={}
        for method in ['rician_converged','amplitude_converged']:
            rows=[v for v in records if v['method']==method]
            convergence[method]=dict(fraction_converged=float(np.mean([v['info']['converged'] for v in rows])),
                mean_updates=float(np.mean([v['info']['iterations'] for v in rows])),cases=len(rows))
        reference_ratios=dict(single_user_rsr_db=spec['rsr'],expected_aggregate_rsr_db=float(spec['rsr']-10*np.log10(spec['K'])))
        summary[spec['id']]=dict(spec=spec,methods=cell,comparisons=comparisons,convergence=convergence,reference_ratios=reference_ratios)
        boots[spec['id']]=boot
    snr_ids=[v['cell'] for v in frozen['axes']['snr']]
    aggregate={}
    for m in methods:
        seedgains=np.stack([summary[c]['methods'][m]['seed_gain_vs_hs_db'] for c in snr_ids]).mean(0)
        aggregate[m]=dict(mean_per_snr_gain_db=float(seedgains.mean()),seed_mean_per_snr_gain_db=seedgains.tolist(),
            paired_world_ci=ci(np.stack([boots[c][m] for c in snr_ids]).mean(0)))
    timings=[json.loads(x) for x in (ROOT/'timings.jsonl').read_text().splitlines()]
    timing={}
    for m in methods:
        names=[n for n,x in frozen['models'].items() if x['method']==m] or [m]
        seedlatency=[];alltimes=[]
        for name in names:
            records=sorted([r for r in timings if r['method']==name],key=lambda x:x['trial'])
            if [r['trial'] for r in records]!=list(range(CONFIG['timing_trials'])): raise RuntimeError(f'incomplete timing {name}')
            times=np.array([np.median(r['seconds']) for r in records])
            seedlatency.append(float(np.median(times)));alltimes.extend(times.tolist())
        timing[m]=dict(median_seconds=float(np.median(alltimes)),seed_median_seconds=seedlatency,
                       iqr_seconds=np.percentile(alltimes,[25,75]).tolist())
    for m in methods: timing[m]['speedup_vs_hs']=timing['hs_gs']['median_seconds']/timing[m]['median_seconds']
    main=aggregate[f'{core}_adaptive'];speed=timing[f'{core}_adaptive']['speedup_vs_hs']
    criteria=dict(combined=main['mean_per_snr_gain_db']>=.3 and main['paired_world_ci'][0]>0 and speed>=2,
        efficiency=main['mean_per_snr_gain_db']>=-.2 and speed>=2,
        timing_scope='N32 K3 P30 SNR5 RSR12 random 3-7 paths; 100 paired worlds, three serial repeats',
        self_supervision_vs_supervised='not tested in this stage',
        independently_trained_final_only='not tested in this stage')
    duration_path=ROOT/'training_duration.json'
    if duration_path.exists(): training_duration=json.loads(duration_path.read_text())
    else:
        starts=list((ROOT/'checkpoints').glob('seed_*/manifest.json'))
        ends=list((ROOT/'checkpoints').glob('seed_*/*_latest.pt'))
        training_duration=dict(
            main_training_span_seconds=(max(p.stat().st_mtime for p in ends)-min(p.stat().st_mtime for p in starts)) if starts and ends else None,
            accumulated_epoch_worker_seconds=sum(e.get('train_seconds',0.) for e in frozen['models'].values()),
            definition='Main training span from earliest worker manifest to final latest checkpoint; excludes preflight and data preparation. Accumulated epoch worker time includes training and validation, with overlapping workers; it is not CPU time.',
            preflight_seconds=json.loads((ROOT/'preflight.json').read_text())['one_seed_one_epoch_seconds'] if (ROOT/'preflight.json').exists() else None)
        write_json(duration_path,training_duration)
    result=dict(primary_core=core,config=CONFIG,cells=summary,aggregate_snr=aggregate,timing=timing,criteria=criteria,
        number_cells=len(summary),unique_world_sets=len(summary)-2,
        training_duration=training_duration,
        uncertainty='4000 paired world resamples, conditional on three trained models; seed variation reported separately',
        averaging='mean of per-training-seed dB summaries, not an ensemble of predictions')
    write_json(ROOT/'summary.json',result);return frozen,result


def plots(frozen,result):
    os.environ.setdefault('MPLCONFIGDIR',str((ROOT/'mpl_cache').resolve()))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','figure.dpi':140})
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    core=result['primary_core'];primary=f'{core}_adaptive'
    styles={'hs_gs':('HS-GS','#222222','s'),f'{core}_none':('Learned steps only','#999999','^'),
            f'{core}_global':('Global shrinkage','#2785b3','D'),primary:('Adaptive shrinkage','#ce4438','o')}
    def finish(fig,name,rect=None):
        fig.tight_layout(rect=rect);fig.savefig(out/f'{name}.pdf');fig.savefig(out/f'{name}.png',dpi=180);plt.close(fig)
    def axis_curve(ax,axis,metric='nmse_db'):
        points=frozen['axes'][axis];x=[p['value'] for p in points]
        for m,(label,color,marker) in styles.items():
            values=[result['cells'][p['cell']]['methods'][m] for p in points]
            y=np.array([v[metric] for v in values])
            intervals=np.array([v['nmse_ci' if metric=='nmse_db' else 'gain_vs_hs_ci'] for v in values])
            ax.plot(x,y,marker=marker,label=label,color=color,ms=4)
            ax.fill_between(x,intervals[:,0],intervals[:,1],alpha=.12,color=color)
        ax.grid(alpha=.2);ax.set_ylabel('NMSE (dB)' if metric=='nmse_db' else 'Gain over HS-GS (dB)')
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    axis_curve(axs[0],'snr');axs[0].set_xlabel('Aggregate SNR (dB)')
    axis_curve(axs[1],'rsr');axs[1].set_xlabel('Single-user RSR (dB)');axs[1].legend(fontsize=8)
    finish(fig,'01_snr_reference')
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    axis_curve(axs[0],'paths','gain_vs_hs_db');axs[0].axhline(0,color='k',lw=.6);axs[0].set_xlabel('Paths per user')
    axs[0].legend(fontsize=8,loc='lower right')
    for c,col in [('em','#2785b3'),('gd','#ce4438')]:
        pts=frozen['axes']['paths'];v=[result['cells'][p['cell']]['comparisons'][f'{c}_adaptive_vs_{c}_global'] for p in pts]
        axs[1].errorbar([p['value'] for p in pts],[q['gain_db'] for q in v],
            yerr=np.array([[q['gain_db']-q['ci'][0],q['ci'][1]-q['gain_db']] for q in v]).T,label=c.upper(),color=col,marker='o')
    axs[1].axhline(0,color='k',lw=.6);axs[1].set_xlabel('Paths per user');axs[1].set_ylabel('Adaptive gain over global (dB)');axs[1].legend();axs[1].grid(alpha=.2)
    finish(fig,'02_complexity')
    base=next(p['cell'] for p in frozen['axes']['snr'] if p['value']==5)
    fig,ax=plt.subplots(figsize=(10,4.5))
    selected=['hs_gs','em_10','em_100','rician_10','rician_100','rician_converged','amplitude_100','linearized_ls',f'{core}_none',f'{core}_global',primary]
    labels={'hs_gs':'HS-GS','em_10':'EM-GS (10 updates)','em_100':'EM-GS (100 updates)',
        'rician_10':'Rician GD (10 updates)','rician_100':'Rician GD (100 updates)',
        'rician_converged':'Rician GD (stopping rule)','amplitude_100':'Amplitude GD (100 updates)',
        'linearized_ls':'Linearized least squares',f'{core}_none':'Learned steps only',
        f'{core}_global':'Global shrinkage',primary:'Adaptive shrinkage'}
    markers=['s','o','^','v','D','P','X','*','<','>','h']
    for j,m in enumerate(selected):
        x=result['timing'][m]['median_seconds']*1000;y=result['cells'][base]['methods'][m]['nmse_db']
        color=styles[m][1] if m in styles else plt.get_cmap('tab10')(j%10)
        ax.scatter(x,y,s=65,marker=markers[j],color=color,label=labels[m],edgecolors='white',linewidths=.5)
    ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=8,frameon=False)
    ax.set_xscale('log');ax.set_xlabel('Serial batch-one inference (ms)');ax.set_ylabel('NMSE at SNR 5 dB (dB)');ax.grid(alpha=.2)
    finish(fig,'03_accuracy_runtime')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for j,c in enumerate(['em','gd']):
        names=[f'{c}_none',f'{c}_global',f'{c}_adaptive',f'{c}_constant_control']
        for i,m in enumerate(names):
            v=result['aggregate_snr'][m]
            axs[j].scatter(np.full(3,i)+[-.08,0,.08],v['seed_mean_per_snr_gain_db'],s=22,color='#2785b3',label='Individual training seeds' if i==0 else None)
            axs[j].errorbar(i,v['mean_per_snr_gain_db'],yerr=[[v['mean_per_snr_gain_db']-v['paired_world_ci'][0]],[v['paired_world_ci'][1]-v['mean_per_snr_gain_db']]],color='#ce4438',fmt='D',capsize=3,label='Mean and paired-world 95% interval' if i==0 else None)
        axs[j].set_xticks(range(4),['Steps only','Global','Adaptive','Constants'],rotation=15);axs[j].set_title(f'{c.upper()} core')
        axs[j].set_ylabel('Mean per-SNR gain over HS-GS (dB)');axs[j].axhline(0,color='k',lw=.6);axs[j].grid(axis='y',alpha=.2)
    axs[0].legend(fontsize=7,loc='lower left')
    finish(fig,'04_controls_training_seeds')
    fig,axs=plt.subplots(2,3,figsize=(12,7.2))
    for ax,axis in zip(axs.flat,['pilots','array','users','noise_offset','channel']):
        points=frozen['axes'][axis];x=np.arange(len(points))
        for m,(label,color,marker) in styles.items():
            v=[result['cells'][p['cell']]['methods'][m] for p in points]
            y=np.array([r['gain_vs_hs_db'] for r in v]);intervals=np.array([r['gain_vs_hs_ci'] for r in v])
            ax.plot(x,y,marker=marker,color=color,label=label);ax.fill_between(x,intervals[:,0],intervals[:,1],alpha=.1,color=color)
        ax.set_xticks(x,[str(p['value']) for p in points]);ax.set_title(axis.replace('_',' '));ax.axhline(0,color='k',lw=.6);ax.grid(alpha=.2)
        ax.set_ylabel('Gain over HS-GS (dB)')
    ax=axs.flat[-1]
    for point in frozen['axes']['snr']:
        cell=point['cell'];truth=np.load(ROOT/'data'/f'{cell}_truth.npz');rho=truth['rho'].mean(1)
        corr=[]
        for seed in CONFIG['seeds']:
            with np.load(ROOT/'cells'/cell/f'{primary}_s{seed}.npz') as f: corr.append(f['correction'].mean((1,2)))
        corr=np.mean(corr,axis=0);order=np.argsort(rho)
        bins=np.array_split(order,min(5,len(order)))
        ax.plot([rho[b].mean() for b in bins],[corr[b].mean() for b in bins],marker='.',label=f'{point["value"]} dB')
    ax.set_xlabel('Mean true normalized effective rank');ax.set_ylabel('Mean relative correction');ax.set_title('Diagnostics by SNR');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9,frameon=False)
    finish(fig,'05_transfer_diagnostics',rect=(0,.05,1,1))
    # Gate and threshold are shown separately from actual correction strength.
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for axis,key in zip(axs,['beta','relative_tau']):
        for point in frozen['axes']['snr']:
            with np.load(ROOT/'data'/f'{point["cell"]}_truth.npz') as t: rho=t['rho'].mean(1)
            values=[]
            for seed in CONFIG['seeds']:
                with np.load(ROOT/'cells'/point['cell']/f'{primary}_s{seed}.npz') as f: values.append(f[key].mean((1,2)))
            y=np.mean(values,axis=0);bins=np.array_split(np.argsort(rho),min(5,len(rho)))
            axis.plot([rho[b].mean() for b in bins],[y[b].mean() for b in bins],marker='.',label=f'{point["value"]} dB')
        axis.set_xlabel('Mean true normalized effective rank');axis.set_ylabel('Gate' if key=='beta' else 'Relative threshold');axis.grid(alpha=.2)
        if key=='beta':
            axis.set_ylim(0,1.04)
            axis.text(.5,.5,'Gate remains approximately 1',ha='center',transform=axis.transAxes)
    axs[1].legend(fontsize=7,ncol=2);finish(fig,'05b_gate_threshold_diagnostics')


def report(frozen,result):
    core=result['primary_core'];m=f'{core}_adaptive';a=result['aggregate_snr'][m];t=result['timing'][m]
    interval=lambda x:f'[{x[0]:.3f}, {x[1]:.3f}]'
    lines=['# Mixed-condition self-supervised Hankel evaluation','',
        f'Completed 18 main models: two update cores, three structural variants, and three training seeds. '
        f'Each model received 20 epochs on the same 16,384 training worlds. Checkpoints used 2,048 independent validation worlds.','',
        f'The primary core selected by mean validation likelihood is **{core.upper()}**. Its equally weighted mean gain over HS-GS across seven SNR values is '
        f'**{a["mean_per_snr_gain_db"]:.3f} dB**, paired-world 95% interval **{interval(a["paired_world_ci"])}**. '
        f'Median inference speedup at the declared reference condition is **{t["speedup_vs_hs"]:.2f}×**.','',
        f'Combined accuracy–speed target met: **{result["criteria"]["combined"]}**. Efficiency target met: **{result["criteria"]["efficiency"]}**.','',
        'The improvement is not uniform. HS-GS is better on the two- and three-path tests and at RSR 0 dB. '
        'At SNR 20 dB, the positive mean gain has an interval that includes zero. '
        'The mean gate is almost one, so the evidence supports learned shrinkage strength rather than an on/off structural decision.','',
        '## What was tested','',
        'Training spans seven SNRs, three single-user RSRs, random 3–7 paths and fixed 2/4/8/12/16 paths per user. '
        'Half the training cases use random path counts. Validation gives equal weight to all 126 SNR/RSR/path-regime strata. '
        'RSR denotes reference power relative to one user; the expected reference-to-total-signal ratio is RSR minus 10 log10(K). '
        'For K=3, a single-user RSR of 12 dB corresponds to 7.229 dB relative to aggregate signal power. '
        'The three training seeds share worlds but change initialization, batch order and training masks. '
        'No true channels enter the training loss, estimator inputs or checkpoint selection.','',
        'Each epoch splits the 30 pilot columns into 21 context columns and nine held-out columns. '
        'The model uses only the context columns to estimate the channel; the loss measures its prediction of the held-out amplitudes. '
        'At deployment, weights are frozen and all available pilots are used in one pass. '
        'This changes the pilot-count feature from its training value, so transfer performance must be measured rather than assumed.','',
        f'There are {result["number_cells"]} test conditions, each with 400 paired trials. Two assumed-noise variants reuse the reference worlds, '
        f'giving {400*result["unique_world_sets"]:,} distinct test worlds. Learned weights remain frozen for every condition. '
        'Classical baselines are evaluated once per world; their results are shared across training-seed comparisons.','',
        '## SNR results','',
        '| SNR (dB) | Adaptive NMSE (dB) | Gain vs HS-GS (dB), 95% interval | Adaptive gain vs global (dB), 95% interval |',
        '|---:|---:|---:|---:|']
    for p in frozen['axes']['snr']:
        c=result['cells'][p['cell']];v=c['methods'][m];d=c['comparisons'][f'{core}_adaptive_vs_{core}_global']
        lines.append(f'| {p["value"]} | {v["nmse_db"]:.3f} | {v["gain_vs_hs_db"]:.3f} {interval(v["gain_vs_hs_ci"])} | {d["gain_db"]:.3f} {interval(d["ci"])} |')
    lines+=['','## Complexity and adaptation','',
        'Path counts 3 and 6 were not fixed-count training strata, but lie within the random 3–7 training distribution. '
        'Therefore this sweep tests complexity dependence; it is not evidence of generalization to wholly unseen path-count support.','',
        '| Paths per user | Gain vs HS-GS (dB) | Adaptive gain vs global (dB), 95% interval | Adaptive gain vs constant intervention (dB), 95% interval |',
        '|---:|---:|---:|---:|']
    for p in frozen['axes']['paths']:
        c=result['cells'][p['cell']];v=c['methods'][m];d=c['comparisons'][f'{core}_adaptive_vs_{core}_global'];q=c['comparisons'][f'{core}_adaptive_vs_{core}_constant_control']
        lines.append(f'| {p["value"]} | {v["gain_vs_hs_db"]:.3f} | {d["gain_db"]:.3f} {interval(d["ci"])} | {q["gain_db"]:.3f} {interval(q["ci"])} |')
    lines+=['','The gate and actual correction answer different questions. These diagnostics hold SNR at 5 dB and RSR at 12 dB; '
        'values average over stages, users, worlds and training seeds.','',
        '| Paths per user | Mean gate | Mean relative threshold | Mean relative correction |','|---:|---:|---:|---:|']
    for p in frozen['axes']['paths']:
        values={k:[] for k in ['beta','relative_tau','correction']}
        for seed in CONFIG['seeds']:
            with np.load(ROOT/'cells'/p['cell']/f'{m}_s{seed}.npz') as f:
                for key in values: values[key].append(float(f[key].mean()))
        lines.append(f'| {p["value"]} | {np.mean(values["beta"]):.4f} | {np.mean(values["relative_tau"]):.4f} | {np.mean(values["correction"]):.4f} |')
    lines+=['','## Transfer and mismatch','',
        '| Axis | Value | Adaptive gain vs HS-GS (dB), 95% interval |','|---|---|---:|']
    for axis in ['rsr','pilots','array','users','channel','noise_offset']:
        for p in frozen['axes'][axis]:
            v=result['cells'][p['cell']]['methods'][m]
            lines.append(f'| {axis} | {p["value"]} | {v["gain_vs_hs_db"]:.3f} {interval(v["gain_vs_hs_ci"])} |')
    lines+=['','Different P/N/K and alternative channel distributions are transfer tests. Reference ratios 0/10/15/25 dB were not training values. '
        'The unequal-power channel has ten paths with powers proportional to 0.5^ell; all alternative families are normalized to the base world’s realized total channel energy. '
        'The clustered family uses four centers with ten rays each and ±5° spread. The noise-offset axis changes the supplied variance only.','',
        '## Runtime and controls','',
        '| Method | Mean per-SNR gain vs HS-GS (dB) | Training-seed range (dB) | Median inference (ms) | Speedup |',
        '|---|---:|---:|---:|---:|']
    for name,v in result['aggregate_snr'].items():
        timing=result['timing'][name];seeds=v['seed_mean_per_snr_gain_db']
        lines.append(f'| {name} | {v["mean_per_snr_gain_db"]:.3f} | {min(seeds):.3f} to {max(seeds):.3f} | {1000*timing["median_seconds"]:.2f} | {timing["speedup_vs_hs"]:.2f}× |')
    lines+=['','Fraction of reference runs meeting the stated stopping criterion before the 1,000-update cap:','',
        '| SNR (dB) | Rician gradient | Amplitude gradient |','|---:|---:|---:|']
    for point in frozen['axes']['snr']:
        conv=result['cells'][point['cell']]['convergence']
        lines.append(f'| {point["value"]} | {100*conv["rician_converged"]["fraction_converged"]:.1f}% | {100*conv["amplitude_converged"]["fraction_converged"]:.1f}% |')
    lines+=['','Timing uses CPU, one thread, double precision and batch size one, with initialization and HS-GS rank selection included. '
        'The timing pass runs after parallel accuracy workers exit. It uses 100 reference-condition worlds and three repeats per estimator and world. '
        'Speedup is specific to this implementation and machine; it is not established for every transfer condition.','',
        f'The main parallel training span was {result["training_duration"]["main_training_span_seconds"]/3600:.2f} hours, '
        f'plus {result["training_duration"]["preflight_seconds"]/60:.2f} minutes for the retained first-epoch preflight. '
        'This span excludes data preparation. See `training_duration.json` for the timing definition.','',
        '## Interpretation and limits','',
        'Numbers average per-seed dB summaries; they are not results from averaging three predicted channels. '
        'Intervals use 4,000 paired resamples of entire worlds and are conditional on the three trained models. '
        'Training-seed variation is reported separately. Intervals are pointwise, without adjustment for multiple comparisons.','',
        'The gate is not an estimated rank. Threshold and gate can compensate for one another, so the figures also show the actual relative correction. '
        'Global controls learn fixed relative thresholds and gates per stage; their absolute threshold still scales with the current Hankel RMS singular value. '
        'A reduction in validation likelihood does not guarantee lower channel NMSE. '
        'The convergence-controlled classical runs have a 1,000-update cap; reaching that cap does not establish convergence.','',
        'These are simulations of the stated receiver model, not measurements from a physical Rydberg receiver. '
        'Low-reference failures can involve reduced identifiability as well as optimization difficulty; exact likelihood alone does not resolve either issue.','',
        'This stage does not include a matched supervised network, a separately trained final-only network, or the remaining architecture ablations. '
        'It cannot establish equivalence to supervised learning or prove that interleaving is necessary. '
        'Publication claims should follow the individual operating-point results, including harmed regions, rather than only the overall average.','',
        '## What this supports and what comes next','',
        'A supported provisional contribution is a small, self-supervised, measurement-conditioned Hankel shrinkage module '
        'within ten exact-Rician updates. It improves the declared average accuracy–runtime comparison against HS-GS without channel labels '
        'or deployment-time rank search. The adaptive-versus-global comparisons support using measurement-dependent outputs in this experiment. '
        'They do not establish that every feature or the gate is necessary.','',
        'The GD core has a larger test SNR-average gain, but EM remains the primary result because it was selected using validation likelihood '
        'before test generation. This difference also shows why validation likelihood and channel NMSE should not be treated as identical criteria.','',
        'The next focused checks are a gate fixed at one, feature ablations separating spectral information from noise/residual information, '
        'a matched supervised model, and a separately trained final-only model. The diagnostics show much stronger variation with SNR '
        'than with true effective rank within each SNR group; they do not establish discovery of a universal complexity threshold. '
        'This is substantial experimental evidence for developing a paper, but the remaining controls and a current literature comparison '
        'are still needed before calling it submission-ready.','',
        '## Figures','',
        '- [SNR and reference strength](figures/01_snr_reference.pdf): primary EM core, mean per-seed NMSE and pointwise intervals.',
        '- [Channel complexity](figures/02_complexity.pdf): gain over HS-GS and adaptive-versus-global comparisons.',
        '- [Accuracy and runtime](figures/03_accuracy_runtime.pdf): reference setting, serial batch-one timing.',
        '- [Controls and training seeds](figures/04_controls_training_seeds.pdf): individual seeds and conditional paired-world intervals.',
        '- [Transfer and correction diagnostics](figures/05_transfer_diagnostics.pdf): changes in system dimensions, channels and supplied noise.',
        '- [Gate and threshold diagnostics](figures/05b_gate_threshold_diagnostics.pdf): the gate is nearly saturated; threshold varies with SNR.','',
        '## Reproduction','',
        'Configuration: `config.json`. Frozen selection and code/checkpoint hashes: `frozen_selection.json`. '
        'Training curves and optimizer checkpoints: `checkpoints/`. Per-world errors and learned predictions: `cells/`. '
        'Full numerical summaries: `summary.json`. Serial timings: `timings.jsonl`. Audit: `audit.json`. '
        'Figures are available as PDF and PNG in `figures/`. See the module README for commands.','']
    recoveries=list((ROOT/'numerical_recovery').glob('cell_*_trial_*.json'))
    if recoveries:
        lines+=['## Numerical recovery','',
            f'{len(recoveries)} HS-GS trial required an alternative SVD driver after NumPy reported nonconvergence on finite inputs. '
            'The same trial was retained, with the same initialization, rank search, updates and projection equations. '
            'SciPy gesvd was used only for a failing matrix; reconstruction and orthogonality were checked. '
            'The recovery is recorded in `numerical_recovery/` and independently rerun by the audit. '
            'No trained checkpoint or original estimator source was changed.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    write_json(ROOT/'analysis_sources.json',{str(p):sha(p) for p in [Path('sahure_expansion')/name for name in ['report.py','audit.py','test_expansion.py','e2e_smoke.py','recover_svd.py']]})
    frozen,result=summarize();plots(frozen,result);report(frozen,result)
    print(json.dumps(dict(report=str(ROOT/'REPORT.md'),primary=result['primary_core'],criteria=result['criteria'])),flush=True)
