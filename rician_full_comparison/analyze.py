"""Audit paired predictions and report fixed, conditional comparisons."""
import json
import numpy as np
import torch
from sahure_expansion.data import world
from fair_optshrink.estimator import tensors
from compact_baseline.analyze import reference_loop
from paper2_optshrink_loop.estimator import estimate as numpy_estimate
from .protocol import ROOT,DATA,CONFIG,SPECS,sha,load,save,write,initialize
from .evaluate import load_models,frozen_parent,fixed_config

@torch.no_grad()
def audit(check):
    initialize();models=load_models();parent=frozen_parent()
    frozen=json.loads((ROOT/'frozen_selection.json').read_text());digest=sha(ROOT/'frozen_selection.json')
    baseline=json.loads((ROOT/'baseline_replay.json').read_text())
    if not baseline['passed']:raise RuntimeError('baseline replay failed')
    histories={}
    for seed in CONFIG['seeds']:
        folder=ROOT/'checkpoints'/f'seed_{seed}'
        state=torch.load(folder/'allpilot_latest.pt',map_location='cpu',weights_only=False)
        best=torch.load(folder/'allpilot_best.pt',map_location='cpu',weights_only=False)
        if state['initial_parameters_sha256']!=baseline['initial_parameters_sha256'][str(seed)]:
            raise RuntimeError('matched initialization audit failed')
        if [r['epoch'] for r in state['history']]!=list(range(1,CONFIG['epochs']+1)):
            raise RuntimeError('incomplete epoch history')
        for e,row in enumerate(state['history']):
            order=torch.randperm(CONFIG['train_cases'],generator=torch.Generator().manual_seed(seed+20000+e))
            import hashlib
            expected=hashlib.sha256(order.numpy().tobytes()).hexdigest()
            if row['order_sha256']!=expected:raise RuntimeError('training order mismatch')
        scores=[state['initial_validation_score']]+[r['val_nll'] for r in state['history']]
        if best['epoch']!=int(np.argmin(scores)):raise RuntimeError('checkpoint not selected by validation minimum')
        histories[str(seed)]=dict(epochs=state['epoch'],best_epoch=best['epoch'],
            training_seconds=state['train_seconds'],orders_matched=True,initialization_matched=True)
    ids=set();replays=0;classical_replays=0;maxdiff=0.;hashes={}
    for bank,seed in enumerate(CONFIG['banks']):
        for cell,spec in enumerate(SPECS):
            check('auditing',bank=bank,cell=cell)
            folder=ROOT/'data'/f'bank{bank}'/spec['id']
            meta=json.loads((folder/'manifest.json').read_text())
            if meta['selection_sha256']!=digest or meta['generated_unix']<frozen['frozen_unix']:
                raise RuntimeError('test selection freeze violation')
            for path,value in meta['hashes'].items():
                if sha(path)!=value:raise RuntimeError('test data modified')
            observed=load(folder/'observations.npz');truth=load(folder/'truth.npz')
            path=ROOT/'predictions'/f'bank{bank}_{spec["id"]}.npz';result=load(path)
            hashes[str(path)]=sha(path)
            expected=[f'rician_full:{seed}:{i}' for i in range(cell*CONFIG['cases'],(cell+1)*CONFIG['cases'])]
            for array in (observed['trial_ids'],truth['trial_ids'],result['trial_ids']):np.testing.assert_array_equal(array,expected)
            if ids.intersection(expected):raise RuntimeError('duplicate test IDs')
            ids.update(expected)
            if str(result['selection_sha256'])!=digest:raise RuntimeError('prediction freeze mismatch')
            np.testing.assert_array_equal(result['names'],['rank_loop','opt_selected']+list(models))
            if not np.isfinite(result['estimates']).all():raise FloatingPointError('nonfinite test result')
            # One regenerated input per bank/cell; all IDs and all file hashes checked.
            w,_=world(cell*CONFIG['cases'],seed,**spec)
            for key,value in zip(('z','s','b','sigma2'),(w.Z,w.S,w.B,w.sigma2)):
                np.testing.assert_array_equal(observed[key][0],value)
            np.testing.assert_array_equal(truth['g'][0],w.G)
            if cell%8==0:
                row={k:observed[k][:1] for k in ('z','s','b','sigma2')};obs=tensors(row)
                for name,model in models.items():
                    actual=result['estimates'][result['names'].tolist().index(name),0]
                    expected_g=model(*obs.args()).numpy()[0]
                    np.testing.assert_allclose(actual,expected_g,rtol=3e-8,atol=3e-9)
                    maxdiff=max(maxdiff,float(abs(actual-expected_g).max()));replays+=1
                expected_g,scores,cfg=reference_loop(row,parent)
                np.testing.assert_allclose(result['rank_scores'][0],scores,rtol=3e-8,atol=3e-9)
                np.testing.assert_allclose(result['estimates'][0,0],expected_g,rtol=3e-8,atol=3e-9)
                np.testing.assert_array_equal(result['rank_configs'][0],[cfg['rank'],cfg['strength'],cfg['iterations']])
                cfg=fixed_config(obs,parent)
                expected_g=numpy_estimate(row['z'][0],row['s'][0],row['b'][0],float(row['sigma2'][0]),cfg)
                np.testing.assert_allclose(result['estimates'][1,0],expected_g,rtol=3e-8,atol=3e-9)
                classical_replays+=2
            timing=json.loads((ROOT/'timing'/f'{spec["id"]}.json').read_text())
            if timing['selection_sha256']!=digest or set(timing['seconds'])!=set(result['names']):raise RuntimeError('timing identity mismatch')
            if any(len(t)!=3 or not np.isfinite(t).all() or min(t)<=0 for t in timing['seconds'].values()):raise RuntimeError('invalid timings')
    record=dict(passed=True,worlds=len(ids),histories=histories,neural_replays=replays,
        independent_classical_replays=classical_replays,neural_max_abs_replay_error=maxdiff,hashes=hashes,
        scope='all IDs, file hashes, histories; one input regeneration per cell/bank; prediction replay every eighth cell/bank')
    write(ROOT/'audit.json',record);return record

def analyze(check=lambda *args,**kwargs:None):
    verification=audit(check);errors=[];powers=[];names=None
    for bank in range(len(CONFIG['banks'])):
        er=[];pw=[]
        for spec in SPECS:
            pred=load(ROOT/'predictions'/f'bank{bank}_{spec["id"]}.npz')
            names=pred['names'].tolist() if names is None else names
            np.testing.assert_array_equal(pred['names'],names)
            g=load(ROOT/'data'/f'bank{bank}'/spec['id']/'truth.npz')['g']
            er.append(abs(pred['estimates']-g[None])**2)
            pw.append(abs(g)**2)
        errors.append(np.asarray(er).sum((-2,-1)));powers.append(np.asarray(pw).sum((-2,-1)))
    errors=np.asarray(errors);power=np.asarray(powers);pc=CONFIG['primary_cells']
    if not np.isfinite(errors).all() or not np.isfinite(power).all() or (power<=0).any():raise FloatingPointError('invalid error arrays')
    save(ROOT/'all_errors.npz',squared_error=errors,channel_power=power,names=np.array(names))
    groups={arm:[names.index(f'{arm}_{seed}') for seed in CONFIG['seeds']]
        for arm in ('holdout_best','allpilot_best','holdout_final','allpilot_final')}
    groups.update({arm:[names.index(arm)] for arm in ('rank_loop','opt_selected')})
    db=10*np.log10(errors.sum((0,3))/power.sum((0,2))[:,None])
    bank_db=10*np.log10(errors.sum(-1)/power.sum(-1)[...,None])
    ratio=errors/power[:,:,None,:]
    primary={arm:float(db[:pc,idx].mean()) for arm,idx in groups.items()}
    cells={spec['id']:{arm:dict(nmse_db=float(db[c,idx].mean()),
        severe_rate=float((ratio[:,c,idx,:]>1).mean())) for arm,idx in groups.items()} for c,spec in enumerate(SPECS)}
    comparators=('holdout_best','rank_loop');rng=np.random.default_rng(CONFIG['bootstrap_seed']);draws=[]
    for start in range(0,CONFIG['bootstrap_samples'],100):
        check('bootstrap',done=start,total=CONFIG['bootstrap_samples'])
        count=min(100,CONFIG['bootstrap_samples']-start)
        weights=rng.multinomial(CONFIG['cases'],np.ones(CONFIG['cases'])/CONFIG['cases'],size=(count,len(CONFIG['banks']),pc))
        e=np.einsum('rbcn,bcmn->rcm',weights,errors[:,:pc],optimize=True)
        p=np.einsum('rbcn,bcn->rc',weights,power[:,:pc],optimize=True)
        sample_db=10*np.log10(e/p[...,None]);new=sample_db[...,groups['allpilot_best']].mean(-1)
        draws.append(np.stack([(sample_db[...,groups[other]].mean(-1)-new).mean(1) for other in comparators],-1))
    draws=np.concatenate(draws);alpha=.05/(2*len(comparators))
    contrasts={arm:dict(gain_db=primary[arm]-primary['allpilot_best'],
        simultaneous95_macro_ci=np.quantile(draws[:,i],[alpha,1-alpha]).tolist(),
        worst_cell_deterioration_db=max(0.,float(np.max(db[:pc,groups['allpilot_best']].mean(-1)-db[:pc,groups[arm]].mean(-1)))),
        improved_cells=int(np.sum(db[:pc,groups['allpilot_best']].mean(-1)<db[:pc,groups[arm]].mean(-1)))) for i,arm in enumerate(comparators)}
    timing={}
    for arm,idx in groups.items():
        values=[v for spec in SPECS[:pc] for j in idx for v in json.loads((ROOT/'timing'/f'{spec["id"]}.json').read_text())['seconds'][names[j]]]
        timing[arm]=dict(mean_ms=float(np.mean(values)*1000),median_ms=float(np.median(values)*1000))
    seeds={arm:{str(seed):float(db[:pc,names.index(f'{arm}_{seed}')].mean()) for seed in CONFIG['seeds']}
        for arm in ('holdout_best','allpilot_best','holdout_final','allpilot_final')}
    result=dict(primary=primary,contrasts=contrasts,cells=cells,timing=timing,seed_primary=seeds,
        bank_primary={arm:bank_db[:,:pc,idx].mean((1,2)).tolist() for arm,idx in groups.items()},
        severe_rate={arm:float((ratio[:,:pc,idx,:]>1).mean()) for arm,idx in groups.items()},
        pooled_db={arm:float(np.mean(10*np.log10(errors[:,:pc,idx,:].sum((0,1,3))/power[:,:pc].sum()))) for arm,idx in groups.items()},
        final_epoch_gain_db=primary['holdout_final']-primary['allpilot_final'],audit=verification,
        caveats=['Three model seeds share one training dataset; intervals condition on fitted models.',
            'Alpha1 chosen from earlier development; fresh test never tunes settings.',
            'Common21/9validation selection is primary; final20epoch sensitivity is separate.',
            'Allpilot changes training noise, geometry and objective together, not architecture.',
            'Equal example/update budgets are not equal training FLOPs; full20epochs is not convergence.',
            'No hardware, universal superiority or first-ever novelty claim.'])
    # Independent scalar-loop recalculation of headline metrics.
    for arm,idx in groups.items():
        independent=np.mean([10*np.log10(sum(float(errors[b,c,m,i]) for b in range(len(CONFIG['banks'])) for i in range(CONFIG['cases']))/
            sum(float(power[b,c,i]) for b in range(len(CONFIG['banks'])) for i in range(CONFIG['cases']))) for c in range(pc) for m in idx])
        np.testing.assert_allclose(independent,primary[arm],rtol=1e-12,atol=1e-12)
    result['independent_headline_recalculation']='passed';write(ROOT/'summary.json',result)
    lines=['FULL-BUDGET ALL-PILOT TRAINING COMPARISON','',
        '8192fresh paired worlds;8064primary and128stress. Lower NMSE is better.',
        'Same188parameter architecture,16384trainingexamples,20epochs,3matched seeds.',
        'Primary uses common validation-selected checkpoints; finalepoch sensitivity separately.','']
    for arm in groups:lines.append(f"{arm}: {primary[arm]:.4f}dB; severeNMSE>1 {result['severe_rate'][arm]:.3%}; measuredCPU {timing[arm]['mean_ms']:.3f}ms")
    lines.append('')
    for arm,v in contrasts.items():lines.append(f"Allpilot gain versus {arm}: {v['gain_db']:.4f}dB; conditional adjusted95%CI {v['simultaneous95_macro_ci']}; wins {v['improved_cells']}/126cells; worstcell deterioration {v['worst_cell_deterioration_db']:.4f}dB")
    lines.append(f"Fixed final20epoch allpilot improvement: {result['final_epoch_gain_db']:.4f}dB")
    lines+=['','Per-seed primary values: '+json.dumps(seeds),'','Stress cases (excluded from primary):']
    for spec in SPECS[pc:]:lines.append(spec['id']+': '+json.dumps(cells[spec['id']]))
    lines+=['','Interpretation limits:']+result['caveats']+['','No manuscript changes. Review all cell/seed results before any publication claim.']
    (ROOT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    return result
