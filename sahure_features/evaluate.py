"""Fresh paired evaluation. Only this module reads channel truth."""
import json
import time
import numpy as np
import torch
from sahure.data import load_observations
from sahure_expansion.train import save_checkpoint
from .model import load_model
from .protocol import ROOT, CONFIG, jobs, cells, check_sources, sha, write_json

def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream: np.savez_compressed(stream,**arrays)
    temp.replace(path)

def freeze():
    check_sources();models={}
    for job in jobs():
        out=ROOT/'checkpoints'/job['id']
        final=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if final['epoch']!=CONFIG['epochs']: raise RuntimeError('Incomplete training')
        for selection,filename in [('selected','best.pt'),('final','latest.pt')]:
            path=out/filename;model,state=load_model(path)
            models[f'{job["id"]}_{selection}']=dict(path=str(path),sha256=sha(path),arm=job['arm'],
                seed=job['seed'],selection=selection,epoch=state['epoch'],parameters=sum(p.numel() for p in model.parameters()))
    record=dict(models=models,protocol_sha256=sha(ROOT/'protocol.json'))
    path=ROOT/'frozen_selection.json'
    if path.exists():
        if json.loads(path.read_text())!=record: raise RuntimeError('Selection changed after freezing')
    else: write_json(path,record)
    return record

def prepare_cell(bank,index,cell):
    # Reuse only the generator function; its source and calibration are frozen.
    from sahure_expansion.data import world
    directory=ROOT/'data'/f'bank{bank}';op=directory/f'{cell["id"]}_observations.npz'
    tp=directory/f'{cell["id"]}_truth.npz';mp=directory/f'{cell["id"]}_manifest.json'
    seed=CONFIG['test_seed_base']+bank*100+index
    description=dict(seed=seed,cell=cell,count=CONFIG['test_cases'])
    if mp.exists():
        saved=json.loads(mp.read_text())
        assert saved['description']==description
        assert sha(op)==saved['observations_sha256'] and sha(tp)==saved['truth_sha256']
        return op,tp
    if op.exists() or tp.exists():
        raise RuntimeError('Incomplete data generation; inspect rather than overwrite')
    obs={k:[] for k in ['z','s','b','sigma2']};truth=[]
    for i in range(CONFIG['test_cases']):
        w,_=world(i,seed,**cell)
        for k,v in zip(obs,[w.Z,w.S,w.B,w.sigma2]):obs[k].append(v)
        truth.append(w.G)
    save_npz(op,**{k:np.asarray(v) for k,v in obs.items()})
    save_npz(tp,g=np.asarray(truth),trial_ids=np.arange(CONFIG['test_cases']))
    write_json(mp,dict(description=description,observations_sha256=sha(op),truth_sha256=sha(tp)))
    return op,tp

@torch.no_grad()
def run(deadline):
    frozen=freeze()
    models={n:load_model(e['path'])[0] for n,e in frozen['models'].items()}
    for bank in CONFIG['test_banks']:
        for i,cell in enumerate(cells()):
            if time.monotonic()>=deadline:return False
            op,tp=prepare_cell(bank,i,cell);data=load_observations(op,torch.float64)
            with np.load(tp) as f:truth=f['g'].copy();trial_ids=f['trial_ids'].copy()
            energy=np.sum(abs(truth)**2,axis=(-2,-1))
            for name,model in models.items():
                path=ROOT/'evaluation'/f'bank{bank}'/cell['id']/f'{name}.npz'
                if path.exists():continue
                if time.monotonic()>=deadline:return False
                write_json(ROOT/'status.json',dict(state='evaluating',bank=bank,cell=cell['id'],model=name,updated_unix=time.time()))
                tick=time.perf_counter();pred=[]
                for start in range(0,len(data),32):
                    pred.append(model(*data.batch(slice(start,start+32)).args()).numpy())
                pred=np.concatenate(pred);elapsed=time.perf_counter()-tick
                error=np.sum(abs(pred-truth)**2,axis=(-2,-1))
                if not np.isfinite(error).all():raise FloatingPointError('Nonfinite evaluation')
                save_npz(path,pred=pred,error=error,energy=energy,trial_ids=trial_ids,
                    observation_sha256=np.asarray(sha(op)),checkpoint_sha256=np.asarray(frozen['models'][name]['sha256']),
                    batched_total_seconds=np.asarray(elapsed))
            print(f'Evaluated bank {bank} {cell["id"]}',flush=True)
    return True

def summarize():
    frozen=freeze();output={};rng=np.random.default_rng(CONFIG['bootstrap_seed'])
    for bank in CONFIG['test_banks']:
        cb={};macro={s:{a:[] for a in CONFIG['arms']} for s in ['selected','final']}
        draws={a:[] for a in CONFIG['arms']}
        for cell in cells():
            scores={};errs={}
            for selection in ['selected','final']:
                scores[selection]={}
                for arm in CONFIG['arms']:
                    rows=[]
                    for seed in CONFIG['seeds']:
                        p=ROOT/'evaluation'/f'bank{bank}'/cell['id']/f'{arm}_{seed}_{selection}.npz'
                        with np.load(p) as f:rows.append(f['error'].copy());energy=f['energy'].copy()
                    e=np.stack(rows);perseed=10*np.log10(e.sum(1)/energy.sum())
                    scores[selection][arm]=dict(nmse_db=float(perseed.mean()),seed_nmse_db=perseed.tolist(),
                        severe_frequency=float((e/energy>1).mean()))
                    if cell['primary']:macro[selection][arm].append(perseed)
                    if selection=='selected':errs[arm]=e
            cb[cell['id']]=scores
            if cell['primary']:
                counts=rng.multinomial(len(energy),np.ones(len(energy))/len(energy),size=CONFIG['bootstrap_resamples'])
                for arm,e in errs.items():
                    draws[arm].append(10*np.log10((e@counts.T)/(energy@counts.T)[None,:]).mean(0))
        point={s:{a:float(np.mean(v)) for a,v in arms.items()} for s,arms in macro.items()}
        contrasts={}
        for other in ['spectral','measurement','context']:
            gain=point['selected'][other]-point['selected']['full']
            boot=np.mean(draws[other],axis=0)-np.mean(draws['full'],axis=0)
            seedg=(np.mean(macro['selected'][other],axis=0)-np.mean(macro['selected']['full'],axis=0)).tolist()
            cellg=[cb[c['id']]['selected'][other]['nmse_db']-cb[c['id']]['selected']['full']['nmse_db'] for c in cells() if c['primary']]
            ci=np.quantile(boot,[.05/6,1-.05/6]).tolist()
            contrasts[f'full_vs_{other}']=dict(gain_db=gain,adjusted_ci=ci,seed_gains_db=seedg,
                worst_primary_cell_gain_db=min(cellg),
                practical_screen_pass=bool(gain>=CONFIG['practical_gain_db'] and ci[0]>0 and min(cellg)>=-CONFIG['meaningful_cell_harm_db']))
        output[f'bank{bank}']=dict(cells=cb,macro_db=point,contrasts=contrasts)
    write_json(ROOT/'summary.json',dict(banks=output,interval_scope='Three contrasts adjusted within each bank; conditional on weights, not training-population uncertainty'))
    lines=['# Feature-access check: results','','Small-budget diagnostic: 4,096 development training worlds, 12 epochs, three matched seeds. No manuscript or old checkpoint changes.','',
        'Positive gain means the full controller has lower error. Selected checkpoints use only held-out likelihood.','',
        '| Bank | Full compared with | Gain (dB) | Adjusted conditional interval | Worst primary-cell gain |',
        '|---|---|---:|---|---:|']
    for bank,values in output.items():
        for name,d in values['contrasts'].items():
            lines.append(f'| {bank} | {name[8:]} | {d["gain_db"]:.3f} | [{d["adjusted_ci"][0]:.3f}, {d["adjusted_ci"][1]:.3f}] | {d["worst_primary_cell_gain_db"]:.3f} |')
    lines+=['','Primary averages exclude weak-reference and two-path stress cases. All cells, final checkpoints, seed-specific scores and severe errors are in summary.json.',
        'Removing controller inputs does not remove the underlying spectral operator, physical noise model or channel-dependent threshold scale. All models store 188 parameters; masked input weights have no effect.',
        'Intervals resample paired test worlds, not independent training populations. Lack of a positive contrast does not establish equivalence.',
        'These small-budget results cannot replace the broad study or prove which features are needed at full training scale. Review training curves, both banks and individual seeds before a manuscript claim.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

@torch.no_grad()
def audit():
    protocol=check_sources();frozen=freeze();files=0;max_replay=0.
    for seed in CONFIG['seeds']:
        histories=[]
        for arm in CONFIG['arms']:
            out=ROOT/'checkpoints'/f'{arm}_{seed}'
            h=json.loads((out/'history.json').read_text());assert len(h)==CONFIG['epochs']
            histories.append([(x['epoch'],x['mask_seed'],x['order_sha256']) for x in h])
            state=torch.load(out/'latest.pt',weights_only=False,map_location='cpu');assert state['label_access'] is False
            assert state['protocol_sha256']==sha(ROOT/'protocol.json')
        assert all(h==histories[0] for h in histories)
    for bank in CONFIG['test_banks']:
        for i,cell in enumerate(cells()):
            op,tp=prepare_cell(bank,i,cell);obs=load_observations(op,torch.float64)
            with np.load(tp) as f:truth=f['g'].copy();ids=f['trial_ids'].copy()
            for name,entry in frozen['models'].items():
                p=ROOT/'evaluation'/f'bank{bank}'/cell['id']/f'{name}.npz'
                with np.load(p) as f:
                    assert str(f['observation_sha256'])==sha(op) and str(f['checkpoint_sha256'])==entry['sha256']
                    np.testing.assert_array_equal(f['trial_ids'],ids)
                    np.testing.assert_allclose(f['energy'],np.sum(abs(truth)**2,axis=(-2,-1)),rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(f['error'],np.sum(abs(f['pred']-truth)**2,axis=(-2,-1)),rtol=1e-12,atol=1e-12)
                    assert np.isfinite(f['pred']).all()
                    if bank==1 and i==0:
                        model,_=load_model(entry['path']);pred=model(*obs.batch(slice(0,32)).args()).numpy()
                        max_replay=max(max_replay,float(np.max(abs(pred-f['pred'][:32]))))
                        np.testing.assert_allclose(pred,f['pred'][:32],rtol=1e-9,atol=1e-9)
                files+=1
    assert files==len(frozen['models'])*len(cells())*len(CONFIG['test_banks'])
    write_json(ROOT/'audit.json',dict(state='passed',result_files=files,models=len(frozen['models']),
        epoch_records=CONFIG['epochs']*len(jobs()),worlds=CONFIG['test_cases']*len(cells())*len(CONFIG['test_banks']),
        matched_orders=True,matched_observations=True,checkpoint_replays=24,max_replay_difference=max_replay,
        source_and_input_hashes=len(set(protocol['sources'])|set(protocol['inputs']))))
