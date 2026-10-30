"""Fresh paired evaluation; labels are confined to this evaluation module."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from sahure.data import Observations,load_observations,split_observations
from sahure.model import AdaptiveHankelEstimator,ModelConfig
from sahure.train import load_model
from sahure.classical import gradient_fit,likelihood,classical_estimate as pilot_classical_estimate
from sahure.evaluate import CLASSICAL as PILOT_CLASSICAL
from .data import generate_set
from .protocol import ROOT,CONFIG,cells,write_json,sha

CLASSICAL=tuple(PILOT_CLASSICAL)+('rician_10','amplitude_10')


def classical_estimate(name,z,s,b,sigma2,hard_rank=None):
    if name in ('rician_10','amplitude_10'):
        g,info=gradient_fit(z,s,b,sigma2,objective=name.split('_')[0],iterations=10)
        return g[0],info
    return pilot_classical_estimate(name,z,s,b,sigma2,hard_rank)


def validation_rank():
    path=ROOT/'hard_rank_validation.json'
    if path.exists(): return json.loads(path.read_text())
    data=load_observations(ROOT/'data/val_observations.npz',torch.float64)
    conditions=np.load(ROOT/'data/val_conditions.npy')
    a,b=split_observations(data,CONFIG['validation_mask_seed'])
    scores=np.empty((len(data),16));tick=time.monotonic()
    for i in range(len(data)):
        z,s,bb,v=(x[i].numpy() for x in a.args())
        g,_=gradient_fit(z,s,bb,float(v),iterations=25,ranks=np.arange(1,17))
        zz,ss,bbb,vv=(x[i].numpy() for x in b.args())
        scores[i]=likelihood(zz,g@ss+bbb,float(vv)).mean((-2,-1))
        if (i+1)%256==0: print(f'Hard-rank validation: {i+1}/{len(data)}',flush=True)
    strata=np.stack([scores[conditions==c].mean(0) for c in range(126)])
    means=strata.mean(0)
    result=dict(rank=int(means.argmin()+1),scores=means.tolist(),strata=strata.tolist(),
        seconds=time.monotonic()-tick,selection='equal-stratum held-out Rician NLL',
        updates=25,sweeps=1,transfer='use selected rank, clipped to new Hankel capacity',
        validation_sha256=sha(ROOT/'data/val_observations.npz'))
    write_json(path,result);return result


@torch.no_grad()
def constant_control(model,val,conditions):
    raw=[];values=[]
    handle=model.net.register_forward_hook(lambda mod,args,out:raw.append(out.detach()))
    try:
        for start in range(0,len(val),32):
            raw.clear();model(*val.batch(slice(start,start+32)).args())
            reltau=torch.stack([.001+torch.nn.functional.softplus(x[...,0]) for x in raw],1)
            gates=torch.stack([torch.sigmoid(x[...,1]) for x in raw],1)
            values.append(torch.stack((reltau,gates),-1).mean(2))
    finally: handle.remove()
    outputs=torch.cat(values)
    means=torch.stack([outputs[conditions==i].mean(0) for i in range(126)]).mean(0)
    control=AdaptiveHankelEstimator(ModelConfig(core=model.config.core,structure='global',stages=10)).double()
    control.steps.copy_(model.steps)
    control.structural[:,0].copy_(torch.log(torch.expm1((means[:,0]-.001).clamp_min(1e-12))))
    control.structural[:,1].copy_(torch.logit(means[:,1].clamp(1e-9,1-1e-9)))
    return control.eval(),means.tolist()


def estimator_sources():
    paths=[Path('sahure')/f for f in ['model.py','math.py','data.py','classical.py','train.py']]
    paths += [Path('sahure_expansion')/f for f in ['protocol.py','data.py','train.py','evaluate.py']]
    paths += list(Path('rydberg_sim').glob('*.py'))+[Path('codex_verification/solver.py')]
    return {str(p):sha(p) for p in sorted(paths)}


def freeze():
    path=ROOT/'frozen_selection.json'
    if path.exists():
        frozen=json.loads(path.read_text())
        if frozen['sources']!=estimator_sources(): raise RuntimeError('estimator source changed after freeze')
        for entry in frozen['models'].values():
            if sha(entry['path'])!=entry['sha256']: raise RuntimeError('checkpoint changed after freeze')
        return frozen
    models={};scores={core:[] for core in ['em','gd']}
    val=load_observations(ROOT/'data/val_observations.npz',torch.float64)
    conditions=torch.tensor(np.load(ROOT/'data/val_conditions.npy'))
    for seed in CONFIG['seeds']:
        directory=ROOT/'checkpoints'/f'seed_{seed}'
        for core in ['em','gd']:
            for structure in ['none','global','adaptive']:
                name=f'{core}_{structure}';p=directory/f'{name}_best.pt'
                latest=torch.load(directory/f'{name}_latest.pt',weights_only=False,map_location='cpu')
                if latest['epoch']!=CONFIG['epochs']: raise RuntimeError('incomplete training')
                best=torch.load(p,weights_only=False,map_location='cpu')
                models[f'{name}_s{seed}']=dict(path=str(p),sha256=sha(p),seed=seed,method=name,
                    best_validation_nll=latest['best_score'],best_epoch=best['epoch'],epochs=latest['epoch'],
                    train_seconds=latest['train_seconds'])
                if structure=='adaptive':
                    scores[core].append(latest['best_score'])
                    model,_=load_model(p);control,means=constant_control(model,val,conditions)
                    cp=directory/f'{core}_constant_control.pt'
                    torch.save(dict(model_config=asdict(control.config),model=control.state_dict()),cp)
                    models[f'{core}_constant_control_s{seed}']=dict(path=str(cp),sha256=sha(cp),
                        seed=seed,method=f'{core}_constant_control',validation_means=means)
    rank=validation_rank()
    frozen=dict(models=models,primary_core=min(scores,key=lambda c:np.mean(scores[c])),
        mean_adaptive_validation_nll={c:float(np.mean(v)) for c,v in scores.items()},
        sources=estimator_sources(),hard_rank=rank['rank'],classical=list(CLASSICAL),
        cells=cells()[0],axes=cells()[1],frozen_unix=time.time(),config=CONFIG,
        constant_control='equal-stratum validation means with all 30 pilots',
        timing='separate serial batch-one pass; parallel accuracy durations are not latency measurements')
    write_json(path,frozen);return frozen


def data_name(spec):
    # Noise mismatch changes only the variance supplied to estimators.
    if spec['noise_offset']:
        for candidate in cells()[0]:
            if all(candidate[k]==spec[k] for k in spec if k not in ['id','noise_offset']) and candidate['noise_offset']==0:
                return candidate['id']
        raise RuntimeError('missing shared world for noise mismatch')
    return spec['id']


def prepare_test():
    frozen=freeze()
    for spec in frozen['cells']:
        name=data_name(spec)
        if name!=spec['id']: continue
        seed=CONFIG['test_data_seed']+int(name.split('_')[1])
        generate_set(name,CONFIG['test_cases'],seed,spec=spec)
        print(f'Generated {name}',flush=True)


def observations_for(spec):
    data=load_observations(ROOT/'data'/f'{data_name(spec)}_observations.npz',torch.float64)
    return Observations(data.z,data.s,data.b,data.sigma2*10**(spec['noise_offset']/10))


@torch.no_grad()
def evaluate_cell(spec):
    frozen=freeze();name=spec['id'];out=ROOT/'cells'/name;out.mkdir(parents=True,exist_ok=True)
    data=observations_for(spec)
    with np.load(ROOT/'data'/f'{data_name(spec)}_truth.npz') as f: truth={k:f[k].copy() for k in f.files}
    gtrue=truth['g'];energy=np.sum(abs(gtrue)**2,axis=(-2,-1))
    def save(name,estimates,**extra):
        if not np.isfinite(estimates).all(): raise FloatingPointError(name)
        path=out/f'{name}.npz';tmp=out/f'{name}.npz.tmp'
        with tmp.open('wb') as stream:
            np.savez_compressed(stream,error=np.sum(abs(estimates-gtrue)**2,axis=(-2,-1)),energy=energy,
                estimates=estimates,**extra)
        tmp.replace(path)
    for modelname,entry in frozen['models'].items():
        if (out/f'{modelname}.npz').exists(): continue
        model,_=load_model(entry['path']);estimates=[];diagnostics={k:[] for k in ['beta','tau','correction','relative_tau']}
        adaptive=entry['method'].endswith('_adaptive')
        raw=[];handle=model.net.register_forward_hook(lambda mod,args,v:raw.append(v.detach())) if adaptive else None
        try:
            for start in range(0,len(data),32):
                raw.clear();batch=data.batch(slice(start,start+32))
                if adaptive:
                    g,trace=model(*batch.args(),diagnostics=True)
                    for key in ['beta','tau','correction']:
                        diagnostics[key].append(torch.stack([v[key] for v in trace],1).numpy())
                    diagnostics['relative_tau'].append(torch.stack([.001+torch.nn.functional.softplus(v[...,0]) for v in raw],1).numpy())
                else: g=model(*batch.args())
                estimates.append(g.numpy())
        finally:
            if handle: handle.remove()
        save(modelname,np.concatenate(estimates),**{k:np.concatenate(v) for k,v in diagnostics.items() if v})
    rows=out/'classical.jsonl';done={}
    if rows.exists():
        for line in rows.read_text().splitlines():
            row=json.loads(line);done[(row['method'],row['trial'])]=row
    with rows.open('a') as stream:
        for i in range(len(data)):
            z,s,b,v=(x[i].numpy() for x in data.args())
            for method in CLASSICAL:
                if (method,i) in done: continue
                est,info=classical_estimate(method,z,s,b,float(v),min(frozen['hard_rank'],(spec['N']+1)//2))
                error=float(np.sum(abs(est-gtrue[i])**2))
                if not np.isfinite(error): raise FloatingPointError(f'{method}: {name}/{i}')
                row=dict(method=method,trial=i,error=error,energy=float(energy[i]),info=info)
                stream.write(json.dumps(row)+'\n');stream.flush();done[(method,i)]=row
            if (i+1)%50==0: print(json.dumps(dict(cell=name,trials=i+1,total=len(data))),flush=True)
    for method in CLASSICAL:
        records=[done[(method,i)] for i in range(len(data))]
        np.savez_compressed(out/f'{method}.npz',error=[r['error'] for r in records],energy=energy)
    write_json(out/'status.json',dict(state='complete',cases=len(data),models=len(frozen['models']),classical=len(CLASSICAL),spec=spec))


@torch.no_grad()
def timing():
    """Run only after parallel workers terminate, on the declared reference cell."""
    frozen=freeze();base=next(x['cell'] for x in frozen['axes']['snr'] if x['value']==5)
    spec=next(c for c in frozen['cells'] if c['id']==base)
    data=observations_for(spec)
    models={name:load_model(entry['path'])[0] for name,entry in frozen['models'].items()}
    names=list(models)+list(CLASSICAL)
    # One warm-up for every method. All timing includes spectral initialization.
    obs=data.batch(slice(0,1));z,s,b,v=(x[0].numpy() for x in obs.args())
    def predict(name,obs,z,s,b,v):
        if name in models: return models[name](*obs.args())
        return classical_estimate(name,z,s,b,float(v),frozen['hard_rank'])
    for name in names: predict(name,obs,z,s,b,v)
    path=ROOT/'timings.jsonl';completed=set()
    if path.exists():
        completed={(x['trial'],x['method']) for x in map(json.loads,path.read_text().splitlines())}
    with path.open('a') as stream:
        for i in range(CONFIG['timing_trials']):
            obs=data.batch(slice(i,i+1));z,s,b,v=(x[0].numpy() for x in obs.args())
            for name in np.random.default_rng(602+i).permutation(names):
                if (i,str(name)) in completed: continue
                times=[]
                for _ in range(CONFIG['timing_repetitions']):
                    tick=time.perf_counter();predict(name,obs,z,s,b,v);times.append(time.perf_counter()-tick)
                stream.write(json.dumps(dict(trial=i,method=str(name),seconds=times,cell=base))+'\n');stream.flush()
            if (i+1)%10==0: print(f'Serial timing: {i+1}/{CONFIG["timing_trials"]}',flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['rank','freeze','prepare','cell','timing']);p.add_argument('--cell')
    args=p.parse_args()
    if args.phase=='rank': validation_rank()
    elif args.phase=='freeze': print(json.dumps(freeze()['mean_adaptive_validation_nll']))
    elif args.phase=='prepare': prepare_test()
    elif args.phase=='timing': timing()
    else: evaluate_cell(next(c for c in cells()[0] if c['id']==args.cell))
