"""Post-freeze audit and plain-language report; truth is evaluator-only."""
import json
from pathlib import Path
import numpy as np
import torch
from sahure_expansion.data import world
from sahure.data import split_observations
from paper2_optshrink_loop.estimator import estimate as independent_estimate
from .run import (ROOT,ARMS,SEEDS,SPECS,CONFIG,sha,load,write,tensors,
                  initialize,learned_models,controls,validation,bank)


def group_names():
    groups={n:[n] for n in ('opt_strong','opt_tuned','opt_residual','opt_near_initial','em100')}
    for arm in ARMS:
        for which in ('initial','best','final'):
            groups[arm+'_'+which]=[f'{arm}_s{s}_{which}' for s in SEEDS]
    return groups


def metric(errors,power,names,group,ii=None):
    if ii is None:ii=np.arange(len(power))
    return float(np.mean([10*np.log10(errors[names.index(n),ii].sum()/power[ii].sum()) for n in group]))


def audit(check):
    initialize();frozen=json.loads((ROOT/'frozen_selection.json').read_text())
    for p,h in {**frozen['data_hashes'],**frozen['checkpoints']}.items():
        if sha(p)!=h:raise RuntimeError('audit hash mismatch: '+p)
    if frozen['manifest_sha256']!=sha(ROOT/'manifest.json'):raise RuntimeError('manifest hash')
    selection=json.loads((ROOT/'classical_selection.json').read_text())
    if sha(ROOT/'classical_selection.json')!=frozen['classical_sha256']:raise RuntimeError('selection hash')
    identities=set();regenerated=0;data_hashes={}
    for manifest in sorted((ROOT/'data').glob('*/*/manifest.json')):
        meta=json.loads(manifest.read_text());data=load(manifest.with_name('observations.npz'))
        if set(data['trial_ids'])&identities:raise RuntimeError('duplicate trial identity')
        identities.update(data['trial_ids'])
        for p,h in meta['hashes'].items():
            if sha(p)!=h:raise RuntimeError('data hash')
            data_hashes[p]=h
        truth=load(manifest.with_name('truth.npz')) if meta['split']=='test' else None
        if truth is not None:
            np.testing.assert_array_equal(data['trial_ids'],truth['trial_ids'])
            if meta['freeze']!=sha(ROOT/'frozen_selection.json'):raise RuntimeError('test freeze identity')
        for i in range(meta['count']):
            check('audit_data',split=meta['split'],cell=meta['spec']['id'],trial=i)
            w,_=world(i,meta['seed'],**{k:meta['spec'][k] for k in ('snr','rsr','L')})
            for k,v in zip(('z','s','b','sigma2'),(w.Z,w.S,w.B,w.sigma2)):
                np.testing.assert_array_equal(data[k][i],v)
            if truth is not None:np.testing.assert_array_equal(truth['g'][i],w.G)
            regenerated+=1
    models=learned_models();replays=0;classical_max_error=0.
    old=json.loads(Path('results/paper2_optshrink_loop/frozen_selection.json').read_text())['noise_aware']
    with torch.no_grad():
        for j,spec in enumerate(SPECS):
            data=load(ROOT/'data/test'/spec['id']/'observations.npz')
            obs=tensors({k:v[:1] for k,v in data.items()})
            output=load(ROOT/'test_outputs'/f'{spec["id"]}.npz');names=output['names'].tolist()
            np.testing.assert_array_equal(data['trial_ids'],output['trial_ids'])
            if str(output['selection_sha256'])!=sha(ROOT/'frozen_selection.json'):raise RuntimeError('output selection')
            for name,model in models.items():
                check('audit_predictions',cell=spec['id'],method=name)
                # Single-case versus batch replay can differ at floating-point roundoff.
                np.testing.assert_allclose(model(*obs.args()).numpy()[0],output['estimates'][names.index(name),0],rtol=2e-8,atol=2e-9)
                replays+=1
            level=str(int(np.clip(spec['snr'],-10,20)))
            cfg=old[level]['loop']
            expected=independent_estimate(data['z'][0],data['s'][0],data['b'][0],float(data['sigma2'][0]),cfg)
            actual=output['estimates'][names.index('opt_strong'),0]
            np.testing.assert_allclose(actual,expected,rtol=2e-8,atol=2e-9)
            classical_max_error=max(classical_max_error,float(abs(actual-expected).max()))
    va,ids,_=bank('val',CONFIG['val_per_cell'],CONFIG['val_seed'],check)
    a,b=split_observations(va,202610118999);validation_replay={}
    for seed in SEEDS:
        hashes={}
        for arm in ARMS:
            key=f'{arm}_s{seed}'
            state=torch.load(ROOT/'training'/f'{key}_latest.pt',map_location='cpu',weights_only=False)
            if state['epoch']!=CONFIG['epochs']:raise RuntimeError('incomplete history')
            for field in ('mask_hash','order_hash'):
                values=[row[field] for row in state['history']]
                if field in hashes and hashes[field]!=values:raise RuntimeError('training order/mask mismatch')
                hashes[field]=values
            score=validation(models[key+'_best'],a,b,ids)
            np.testing.assert_allclose(score,state['best'],rtol=1e-11,atol=1e-11)
            validation_replay[key]=score
    result=dict(status='passed',regenerated_worlds=regenerated,distinct_worlds=len(identities),
        test_worlds=CONFIG['test_per_cell']*len(SPECS),checkpoint_replays=replays,
        independent_classical_replays=len(SPECS),classical_max_abs_error=classical_max_error,
        validation_replays=validation_replay,matched_orders=True,matched_masks=True,
        test_outputs={str(p):sha(p) for p in sorted((ROOT/'test_outputs').glob('*.npz'))},
        data_hashes=data_hashes,scope='conditional numerical audit, not external replication')
    write(ROOT/'audit.json',result);return result


def analyze(check=lambda *a,**k:None):
    verification=audit(check);groups=group_names();cells={};raw={};seed_scores={}
    for spec in SPECS:
        check('analysis',cell=spec['id'])
        output=load(ROOT/'test_outputs'/f'{spec["id"]}.npz')
        truth=load(ROOT/'data/test'/spec['id']/'truth.npz')['g']
        names=output['names'].tolist();power=(abs(truth)**2).sum((-2,-1))
        errors=(abs(output['estimates']-truth[None])**2).sum((-2,-1));raw[spec['id']]=(errors,power,names)
        cell={}
        for group,members in groups.items():
            ii=[names.index(n) for n in members]
            cell[group]=dict(nmse_db=metric(errors,power,names,members),
                severe_error_rate=float((errors[ii]/power>1).mean()),
                mean_case_nmse=float((errors[ii]/power).mean()))
        cx={}
        for arm in ARMS:
            ii=[output['context_names'].tolist().index(f'{arm}_s{s}_best') for s in SEEDS]
            ce=(abs(output['context_estimates'][ii]-truth[None])**2).sum((-2,-1))
            cx[arm]=float(np.mean(10*np.log10(ce.sum(-1)/power.sum())))
        diag={}
        for name,d in zip(output['diagnostic_names'],output['diagnostics']):
            active=d[...,2]>0
            diag[str(name)]=dict(mean_beta=float(d[...,0][active].mean()),
                mean_relative_correction=float(d[...,1][active].mean()),
                mean_gate_slope=float(d[...,3][active].mean()),
                exact_zero_slope_fraction=float((d[...,3][active]==0).mean()),
                beta_below_001_fraction=float((d[...,0][active]<.01).mean()),
                beta_above_099_fraction=float((d[...,0][active]>.99).mean()))
        cells[spec['id']]=dict(methods=cell,context21_nmse_db=cx,diagnostics=diag)
        seed_scores[spec['id']]={n:metric(errors,power,names,[n]) for n in names if n.endswith(('_initial','_best','_final'))}
    primary={name:float(np.mean([cells[sp['id']]['methods'][name]['nmse_db'] for sp in SPECS[:7]])) for name in groups}
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);contrasts={}
    for comparator in ('opt_residual','opt_strong','opt_near_initial','opt_tuned','global_best','clipped_best'):
        gains=[cells[sp['id']]['methods'][comparator]['nmse_db']-cells[sp['id']]['methods']['adaptive_best']['nmse_db'] for sp in SPECS]
        samples=[]
        for repeat in range(CONFIG['bootstrap_samples']):
            if repeat%100==0:check('paired_uncertainty',comparator=comparator,draw=repeat)
            values=[]
            for sp in SPECS[:7]:
                errors,power,names=raw[sp['id']];ii=rng.integers(len(power),size=len(power))
                values.append(metric(errors,power,names,groups[comparator],ii)-metric(errors,power,names,groups['adaptive_best'],ii))
            samples.append(np.mean(values))
        contrasts[comparator]=dict(gain_db=float(np.mean(gains[:7])),ci95=np.quantile(samples,[.025,.975]).tolist(),
            worst_primary_deterioration_db=float(max(-np.asarray(gains[:7]))),
            worst_stress_deterioration_db=float(max(-np.asarray(gains[7:]))))
    gates={}
    for comparator in ('opt_residual','opt_strong'):
        c=contrasts[comparator]
        gates[comparator]=bool(c['gain_db']>=.3 and c['ci95'][0]>0 and c['worst_primary_deterioration_db']<=.2
                              and c['worst_stress_deterioration_db']<=.5)
    gates['global_control']=contrasts['global_best']['gain_db']>0
    decision='eligible_for_independent_confirmation' if all(gates.values()) else 'stop_expansion_of_repaired_hybrid'
    timing=json.loads((ROOT/'timing.json').read_text())
    times={name:dict(mean_ms=float(1000*np.mean(v)),median_ms=float(1000*np.median(v))) for name,v in timing['seconds'].items()}
    summary=dict(decision=decision,gates=gates,primary=primary,contrasts=contrasts,cells=cells,
        per_seed_per_cell=seed_scores,timing=times,audit=verification,
        caveats=['Two initializations share one training dataset; not two dataset replicates.',
                 'Pointwise intervals condition on training/tuning; small development screen.',
                 'No full convergence, novelty, hardware or general-optimality claim.',
                 'SVD derivatives checked away from ties, not globally differentiable; sigmoid can still become numerically saturated at extreme logits.',
                 'Do not retune a failed pilot on these test outcomes.'])
    write(ROOT/'summary.json',summary)
    lines=['CORRECTED GATE + STRONG OPTSHRINK: FEASIBILITY RESULTS','',f'Decision: {decision}',
        'Two initializations, one shared training bank; this is not publication confirmation.',
        'All settings/checkpoints frozen before 704 new paired test worlds were generated.',
        'Main metric averages seven per-SNR ratio-of-sums dB NMSEs. Lower is better.','',
        'PRIMARY AVERAGE']
    lines += [f'  {n}: {v:.3f} dB' for n,v in primary.items()]
    lines += ['','ADAPTIVE SELECTED-CHECKPOINT GAINS (positive helps)']
    for n,c in contrasts.items():
        lines.append(f"  versus {n}: {c['gain_db']:.3f} dB; conditional pointwise95% CI {c['ci95']}; worst primary deterioration {c['worst_primary_deterioration_db']:.3f}; worst stress deterioration {c['worst_stress_deterioration_db']:.3f} dB")
    chosen=['opt_strong','opt_tuned','opt_residual','opt_near_initial','global_best','clipped_best','adaptive_best','em100']
    lines += ['','PER-CELL RESULTS','cell | '+' | '.join(chosen)]
    for sp in SPECS:
        lines.append(sp['id']+' | '+' | '.join(f"{cells[sp['id']]['methods'][n]['nmse_db']:.3f}" for n in chosen))
    lines += ['','TIMING: same PyTorch backend, one CPU thread, float64, batch one, 12 worlds x3; milliseconds']
    lines += [f"  {n}: mean {v['mean_ms']:.3f}; median {v['median_ms']:.3f}" for n,v in times.items()]
    lines += ['','SELECTED GATE DIAGNOSTICS: beta mean / exact-zero local-slope fraction']
    for sp in SPECS:
        for arm in ('adaptive','clipped'):
            dd=[cells[sp['id']]['diagnostics'][f'{arm}_s{s}_best'] for s in SEEDS]
            lines.append(f"  {sp['id']} {arm}: {np.mean([d['mean_beta'] for d in dd]):.6f} / {np.mean([d['exact_zero_slope_fraction'] for d in dd]):.6f}")
    lines += ['','Interpretation and limits',
        'All OptShrink variants share the strong baseline rank and per-noise iteration schedule.',
        'Repaired adaptive/global gates start at strength .95 where the baseline uses1, otherwise .5. They are NOT exactly the full-strength baseline at initialization.',
        'Opt_near_initial controls the changed initialization; clipped is the old gate retrained with the same strong rank and iteration schedule.',
        'Both initializations use identical observations but different matched training orders/masks.',
        'Validation-selected/final predictions, per-seed results, severe-error rates, correction magnitudes and21-pilot diagnostics are in summary.json.',
        'Twenty-one-pilot versus30-pilot evaluation does not isolate distribution shift from extra measurements.',
        'Exact singular-value ties/rank boundaries are not covered by native-SVD gradient smoothness.',
        'No finite-gradient failure was silently bypassed. A passed gate justifies a separately frozen confirmation, not a publication claim.',
        'A failed gate remains failed; no automatic new tuning/training is authorized.',
        f"Audit passed: {verification['regenerated_worlds']} regenerated worlds, {verification['checkpoint_replays']} checkpoint replays, {verification['independent_classical_replays']} independent classical replays.",
        'Historical results and manuscript were not modified.']
    report=ROOT/'RESULTS.txt'
    report.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:32]),flush=True)


if __name__=='__main__':analyze()

