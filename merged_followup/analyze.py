"""Read-only replays/statistics; writes only this development study's report."""
import json
from collections import Counter
import hashlib
import numpy as np
import torch
from .run import ROOT, SPECS, ARMS, initialize, sha, data_world, old_config, validation, SEED
from .models import build
from .estimator import estimate, context_candidates, validation_scores, SCHEDULES
from paper2_optshrink_loop.run import load, write
from sahure.data import load_observations


def interval(x):return np.quantile(x,[.025,.975]).tolist()


def summarize(bank):
    rng=np.random.default_rng(202610089501)
    summary={};boot={};identities=[];pilots=set();replay=[];rows_by_cell={}
    for spec in SPECS:
        rows=[load(p) for p in sorted((ROOT/bank/spec['id']).glob('*.npz'))]
        if len(rows)!=32:raise RuntimeError(f'incomplete {bank}/{spec["id"]}: {len(rows)}')
        rows_by_cell[spec['id']]=rows
        names=list(rows[0]['names']);err={n:[] for n in names};energy=[];context={}
        for row in rows:
            if list(row['names'])!=names:raise RuntimeError('inconsistent method names')
            if not np.isfinite(row['estimates']).all():raise RuntimeError('nonfinite estimate')
            identities.append(str(row['trial_id']))
            pilots.add(hashlib.sha256(row['s'].tobytes()).hexdigest())
            g=row['truth'];en=float(np.vdot(g,g).real);energy.append(en)
            for name,hat in zip(names,row['estimates']):
                e=float(np.vdot(hat-g,hat-g).real)
                np.testing.assert_allclose(e,np.sum(abs(hat-g)**2),rtol=1e-13,atol=1e-13)
                err[name].append(e)
            for name,hat in zip(row['context_names'],row['context_estimates']):
                context.setdefault(str(name),[]).append(float(np.vdot(hat-g,hat-g).real))
        idx=rng.integers(0,32,(4000,32));energy=np.array(energy)
        summary[spec['id']]={};boot[spec['id']]={}
        for k,name in enumerate(names):
            e=np.array(err[name]);ratios=e/energy
            boot[spec['id']][name]=10*np.log10(e[idx].sum(1)/energy[idx].sum(1))
            item=dict(nmse_db=float(10*np.log10(e.sum()/energy.sum())),
                mean_case_linear_nmse=float(ratios.mean()),
                severe_percent=float(100*np.mean(ratios>1)),
                threshold_percent={str(t):float(100*np.mean(ratios>t)) for t in (.5,1.,2.)},
                mean_ms=float(1000*np.mean([row['seconds'][k] for row in rows])))
            if name in context:
                item['context_nmse_db']=float(10*np.log10(np.sum(context[name])/energy.sum()))
                item['refit_minus_context_db']=item['nmse_db']-item['context_nmse_db']
            summary[spec['id']][name]=item
        if bank=='classical':
            summary[spec['id']]['schedules']={n:dict(Counter(
                str(tuple(int(x) for x in row['schedules'][k])) for row in rows))
                for k,n in enumerate(rows[0]['context_names'])}
    if len(set(identities))!=352:raise RuntimeError('duplicated world IDs')
    primary=[s['id'] for s in SPECS[:7]]
    names=list(rows_by_cell[primary[0]][0]['names'])
    macro={n:float(np.mean([summary[c][n]['nmse_db'] for c in primary])) for n in names}
    comparisons={}
    pairs=[('observed_release','observed_stop'),('observed_release','fixed_release'),
           ('observed_stop','opt_frozen')] if bank=='classical' else [
           ('compact_best','em_transformer_best'),('compact_best','opt_frozen')]
    for method,reference in pairs:
        gain=macro[reference]-macro[method]
        distribution=np.mean([boot[c][reference]-boot[c][method] for c in primary],axis=0)
        comparisons[f'{method}_vs_{reference}']=dict(gain_db=gain,pointwise_ci95=interval(distribution),
            worst_primary_deterioration_db=max(summary[c][method]['nmse_db']-summary[c][reference]['nmse_db'] for c in primary))
    return dict(cells=summary,macro_nmse_db=macro,comparisons=comparisons,worlds=352,
        distinct_pilot_arrays=len(pilots),ids=identities,
        uncertainty='4000 paired resamples within each cell; fixed fitted models; pointwise, not multiplicity-adjusted'),rows_by_cell


@torch.no_grad()
def audit(a,b):
    initialize();out=dict(source_hashes_verified=True,worlds=704,matched_training_orders=True,
                          matched_training_masks=True,replays=[])
    afrozen=json.loads((ROOT/'release_selection.json').read_text())
    for p,h in afrozen['tuning_hashes'].items():assert sha(p)==h
    bfrozen=json.loads((ROOT/'learned/frozen_selection.json').read_text())
    for p,h in bfrozen['checkpoints'].items():assert sha(p)==h
    all_a={str(r['trial_id']) for rows in a.values() for r in rows}
    all_b={str(r['trial_id']) for rows in b.values() for r in rows}
    tune={str(load(p)['trial_id']) for p in afrozen['tuning_hashes']}
    assert not (all_a&all_b or all_a&tune or all_b&tune)
    out['tuning_A_B_world_ids_disjoint']=True
    data=ROOT.parent/'sahure_expansion/data'
    val=load_observations(data/'val_observations.npz');ids=np.load(data/'val_conditions.npy')
    histories=[]
    for name in ARMS:
        histories.append(json.loads((ROOT/'learned'/f'{name}_history.json').read_text()))
        assert len(histories[-1])==12
        model=build(name)
        initial_score=validation(model,val,ids)
        out[name+'_initial_validation_nll']=initial_score
        ck=torch.load(ROOT/'learned'/f'{name}_best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);model.eval()
        v=validation(model,val,ids)
        np.testing.assert_allclose(v,ck['best'],atol=1e-8,rtol=1e-8)
        recorded=[initial_score]+[h['val_nll'] for h in histories[-1]]
        assert ck['best_epoch']==int(np.argmin(recorded))
        out[name+'_validation_replay_abs']=abs(v-ck['best'])
    for key in ('order_hash','mask_hash'):
        assert [r[key] for r in histories[0]]==[r[key] for r in histories[1]]
    for bank,rows_by_cell,seed in [('classical',a,202610086101),('learned_test',b,202610088101)]:
        digest=sha(ROOT/('release_selection.json' if bank=='classical' else 'learned/frozen_selection.json'))
        for j,spec in enumerate(SPECS):
            for i,row in enumerate(rows_by_cell[spec['id']]):
                assert str(row['selection_sha256'])==digest
                # Replay every generated observation/truth, also detecting input mutation.
                w,d=data_world(seed+100*j,i,spec)
                for k in ('z','s','b','variance','trial_id'):np.testing.assert_array_equal(row[k],d[k])
                np.testing.assert_array_equal(row['truth'],w.G)
            if spec['id'] not in ('snr_-10','snr_+5','nearly_noiseless'):continue
            row=rows_by_cell[spec['id']][0]
            z,s,bb,v=row['z'],row['s'],row['b'],float(row['variance'])
            if bank=='classical':
                g,info=estimate(z,s,bb,v,dict(**old_config(v),mode='release'))
                expected=row['estimates'][list(row['names']).index('observed_release')]
                np.testing.assert_array_equal(g,expected)
                assert tuple(row['schedules'][2])==tuple(info['schedule'])
                gs=context_candidates(z,s,bb,v,old_config(v));scores=validation_scores(gs,z,s,bb,v)
                assert tuple(info['schedule'])==SCHEDULES[int(scores.argmin())]
                out['replays'].append(dict(bank=bank,cell=spec['id'],max_abs=float(abs(g-expected).max())))
            else:
                for name in ARMS:
                    model=build(name,dtype=torch.float64)
                    ck=torch.load(ROOT/'learned'/f'{name}_best.pt',map_location='cpu',weights_only=False)
                    model.load_state_dict(ck['model']);model.eval()
                    g=model(torch.tensor(z[None]),torch.tensor(s[None]),torch.tensor(bb[None]),
                            torch.tensor([v],dtype=torch.float64))[0].numpy()
                    expected=row['estimates'][list(row['names']).index(name+'_best')]
                    np.testing.assert_array_equal(g,expected)
                    out['replays'].append(dict(bank=bank,cell=spec['id'],model=name,max_abs=float(abs(g-expected).max())))
    out['all_observation_truth_replays_exact']=True
    out['status']='passed'
    return out


def main():
    torch.set_num_threads(1)
    status=json.loads((ROOT/'status.json').read_text())
    if status['state']!='complete' or status['phase']!='all':raise RuntimeError('full run not complete')
    a,arows=summarize('classical');b,brows=summarize('learned_test')
    checked=audit(arows,brows);checked['analysis_source_sha256']=sha(__file__)
    write(ROOT/'audit.json',checked)
    criteria=[]
    for key in ('observed_release_vs_observed_stop','observed_release_vs_fixed_release'):
        c=a['comparisons'][key]
        criteria += [c['gain_db']>=.3,c['worst_primary_deterioration_db']<=.25]
    criteria += [a['cells']['nearly_noiseless']['observed_release']['nmse_db']<
                 a['cells']['nearly_noiseless']['opt_frozen']['nmse_db']]
    decision='passes_small_screen_needs_review' if all(criteria) else 'stop_expansion_of_simple_release'
    summary=dict(classical=a,learned=b,release_screen_decision=decision,
        training=json.loads((ROOT/'learned/frozen_selection.json').read_text())['summary'],audit='audit.json')
    write(ROOT/'summary.json',summary)
    lines=['MERGED PAPER FOLLOW-UP: DEVELOPMENT RESULTS','',
        'This is a bounded, one-training-seed pilot, not final confirmation.',
        'No claim of verified novelty or published-URformer reproduction.',
        'All previous studies and the manuscript were left unchanged.','',
        'A. Stronger classical controls and filter-release screen',f'Decision: {decision}',
        'Lower NMSE is better; positive comparative gain means the first method helps.']
    for n,v in a['macro_nmse_db'].items():lines.append(f'  {n}: {v:.3f} dB')
    for n,c in a['comparisons'].items():lines.append(f'  {n}: gain {c["gain_db"]:.3f} dB; pointwise CI {c["pointwise_ci95"]}; worst primary deterioration {c["worst_primary_deterioration_db"]:.3f} dB')
    lines+=['','B. Matched learned comparison (same data, masks, order and NLL selection)',
        'Compact: 188 parameters. Exact-EM Transformer: 1,585,920 parameters.',
        'The Transformer keeps the exact Bessel ratio and is not published URformer.',
        'Both received 4096 worlds x 12 epochs; neither is claimed fully trained.']
    for n,v in b['macro_nmse_db'].items():lines.append(f'  {n}: {v:.3f} dB')
    for n,c in b['comparisons'].items():lines.append(f'  {n}: gain {c["gain_db"]:.3f} dB; pointwise CI {c["pointwise_ci95"]}')
    for n,info in summary['training'].items():lines.append(f'  {n}: selected epoch {info["best_epoch"]}/12, training+validation {info["train_seconds"]:.1f} s')
    lines += ['The Transformer is worse than EM100 on this primary average. This pilot',
        'does not establish a strong learned benchmark or a win over published',
        'URformer. A bounded training adequacy check is still required before a',
        'publication claim about compact versus large learned estimators.',
        'The compact model is still improving at its final epoch; convergence is',
        'not demonstrated. Its small mean advantage over OptShrink has a paired',
        'interval crossing zero, so no average-accuracy superiority is established.']
    lines+=['','Per-cell NMSE (no pooling of the two independent banks):']
    for title,results,names in [('A',a,['opt_frozen','observed_stop','fixed_release','observed_release','em100']),
                               ('B',b,['compact_best','em_transformer_best','opt_frozen','em100'])]:
        lines.append(title+': cell | '+' | '.join(names))
        for spec in SPECS:
            lines.append(spec['id']+' | '+' | '.join(f'{results["cells"][spec["id"]][n]["nmse_db"]:.3f}' for n in names))
    lines+=['','Limits and next decision',
        'A smaller validation NLL is not a channel-error guarantee. All-pilot refit',
        'changes the fit; context and refit outcomes are in summary.json.',
        'Timing includes online selection/refit, but mixes NumPy/SciPy and PyTorch.',
        'These are single-pass development timings, not a dedicated repeated timing benchmark.',
        'The stopping control computes to 100 updates before selecting an output;',
        'its chosen iteration count is not its total processing cost.',
        '32 worlds/cell cannot establish rare-event safety. Intervals are conditional',
        'on fitted models, pointwise, and not corrected for all comparisons.',
        'Nearly noiseless SNR60 is an out-of-distribution stress test, not hardware.',
        'A failed screen stays failed; do not enlarge or tune it to pass.',
        'A successful matched comparison would close a baseline gap, not by itself',
        'establish novelty. Read NOVELTY.txt before choosing the next mechanism.',
        'Independent confirmation has not been launched.','',
        'Verification: 704 distinct screen worlds, source/input hashes checked,',
        'every observation and truth regenerated exactly, matched training masks',
        'and orders checked, selected validation scores and nine predictions replayed.',
        'Full metrics, thresholds, timings and schedules: summary.json.']
    (ROOT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
