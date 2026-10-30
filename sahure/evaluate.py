"""Frozen-model evaluation. This is the only estimator pipeline loading truth."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .classical import classical_estimate,select_global_hard_rank
from .data import generate,load_observations,split_observations
from .model import AdaptiveHankelEstimator,ModelConfig
from .train import load_model

CLASSICAL=('em_10','em_100','hs_gs','rician_100','rician_converged',
           'amplitude_100','amplitude_converged','rician_hard','linearized_ls')


def validation_hard_rank(out,config,deadline):
    """Independent observation-only selection, safely reusable across phases."""
    out=Path(out);path=out/'hard_rank_validation.json'
    inputs=dict(mask_seed=config['seed']+30000,updates=25,sweeps=1,
        observations_sha256=hashlib.sha256((out/'data/val_observations.npz').read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path('sahure/classical.py'),Path('sahure/data.py'),Path('codex_verification/solver.py'))})
    if path.exists():
        cached=json.loads(path.read_text())
        if cached['inputs']!=inputs: raise ValueError('validation-rank cache does not match inputs')
        return cached['rank'],cached['scores']
    val=load_observations(out/'data/val_observations.npz',torch.float64)
    a,b=split_observations(val,inputs['mask_seed'])
    tick=time.monotonic();rank,scores=select_global_hard_rank(a,b,deadline)
    path.write_text(json.dumps(dict(inputs=inputs,rank=rank,scores=scores,seconds=time.monotonic()-tick),indent=2)+'\n')
    return rank,scores


@torch.no_grad()
def constant_control(adaptive,data):
    """Replace adaptive outputs by their per-stage validation means.

    This is an intervention, not a separately optimized competitor. Threshold
    is averaged in relative units, before multiplying by channel RMS scale.
    """
    raw=[]
    handle=adaptive.net.register_forward_hook(lambda module,args,out: raw.append(out.detach()))
    values=[]
    try:
        for start in range(0,len(data),32):
            raw.clear();adaptive(*data.batch(slice(start,start+32)).args())
            tau=torch.stack([.001+torch.nn.functional.softplus(x[...,0]) for x in raw],1)
            gate=torch.stack([torch.sigmoid(x[...,1]) for x in raw],1)
            values.append(torch.stack((tau,gate),-1))
    finally: handle.remove()
    means=torch.cat(values).mean((0,2))
    cfg=ModelConfig(core=adaptive.config.core,structure='global',stages=adaptive.config.stages)
    control=AdaptiveHankelEstimator(cfg).double()
    control.steps.copy_(adaptive.steps)
    control.structural[:,0].copy_(torch.log(torch.expm1((means[:,0]-.001).clamp_min(1e-12))))
    control.structural[:,1].copy_(torch.logit(means[:,1].clamp(1e-9,1-1e-9)))
    control.eval()
    return control,means.tolist()


def freeze(out,config,deadline):
    path=out/'frozen_selection.json'
    checkpoints={}
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path('sahure')/f for f in ('__init__.py','math.py','model.py','data.py','classical.py','evaluate.py','run.py')]+[Path('codex_verification/solver.py')]}
    for core in ('em','gd'):
        for structure in ('none','global','adaptive'):
            name=f'{core}_{structure}';p=out/'checkpoints'/f'{name}_best.pt'
            latest=torch.load(out/'checkpoints'/f'{name}_latest.pt',weights_only=False,map_location='cpu')
            if latest['epoch']!=config['epochs']: raise RuntimeError(f'{name} incomplete; do not present a complete six-arm contest')
            checkpoints[name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                path=str(p),best_validation_nll=latest['best_score'],epochs=latest['epoch'],
                training_seconds=latest['train_seconds'])
    if path.exists():
        frozen=json.loads(path.read_text())
        if frozen['checkpoints']!=checkpoints: raise RuntimeError('checkpoints changed after test configuration was frozen')
        if frozen.get('estimator_source_hashes')!=sources: raise RuntimeError('estimator source changed after freezing; use a new evaluation directory')
        return frozen
    val=load_observations(out/'data/val_observations.npz',torch.float64)
    a,b=split_observations(val,config['seed']+30000)
    if time.monotonic()>=deadline: return None
    print('Selecting classical hard rank using validation observations only',flush=True)
    try: rank,scores=validation_hard_rank(out,config,deadline)
    except TimeoutError: return None
    frozen=dict(checkpoints=checkpoints,estimator_source_hashes=sources,hard_rank=rank,hard_rank_scores=scores,
        hard_rank_selection_updates=25,hard_rank_final_updates=100,hard_rank_sweeps=1,
        hs_selection_updates=25,hs_final_updates=100,hs_sweeps=4,
        classical=list(CLASSICAL),dtype='float64/complex128',timing_repetitions=3,
        fixed_condition_only=True,selection_metric='mean held-out Rician NLL',
        primary_core=min(('em','gd'),key=lambda c:checkpoints[c+'_adaptive']['best_validation_nll']))
    for core in ('em','gd'):
        model,_=load_model(out/'checkpoints'/f'{core}_adaptive_best.pt')
        # Match deployment pilot count; otherwise a constant-control gap could
        # merely reflect the context-to-full-input feature shift (21 vs 30).
        control,means=constant_control(model,val)
        torch.save(dict(model_config=asdict(control.config),model=control.state_dict()),out/'checkpoints'/f'{core}_constant_control.pt')
        frozen[f'{core}_constant_control_means']=means
    frozen['constant_control_validation_pilots']=config['P']
    path.write_text(json.dumps(frozen,indent=2)+'\n')
    return frozen


@torch.no_grad()
def evaluate(out,config,deadline):
    out=Path(out);frozen=freeze(out,config,deadline)
    if frozen is None: return False
    # No test generation, labels or error computation until selection is frozen.
    generate(out/'data','test',config['test_cases'],seed=config['test_seed'])
    observations=load_observations(out/'data/test_observations.npz',torch.float64)
    with np.load(out/'data/test_truth.npz') as f: truth={k:f[k].copy() for k in f.files}
    models={name:load_model(out/'checkpoints'/f'{name}_best.pt')[0] for name in frozen['checkpoints']}
    for core in ('em','gd'):
        models[core+'_constant_control']=load_model(out/'checkpoints'/f'{core}_constant_control.pt')[0]
    # Warm-up on validation data, not test errors.
    val=load_observations(out/'data/val_observations.npz',torch.float64).batch(slice(0,1))
    for model in models.values(): model(*val.args())
    z,s,b,v=(x[0].numpy() for x in val.args())
    for name in CLASSICAL: classical_estimate(name,z,s,b,float(v),frozen['hard_rank'])
    rows_path=out/'trials.jsonl'
    completed=set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row=json.loads(line);completed.add((row['trial'],row['method']))
    names=list(models)+list(CLASSICAL)
    with rows_path.open('a') as stream:
        for i in range(len(observations)):
            order=np.random.default_rng(config['seed']+i).permutation(names)
            obs=observations.batch(slice(i,i+1))
            z,s,b,v=(x[0].numpy() for x in obs.args())
            for name in order:
                if (i,name) in completed: continue
                if time.monotonic()>=deadline: return False
                times=[];info={}
                for repeat in range(frozen['timing_repetitions']):
                    tick=time.perf_counter()
                    if name in models: estimate=models[name](*obs.args())
                    else: estimate,info=classical_estimate(name,z,s,b,float(v),frozen['hard_rank'])
                    times.append(time.perf_counter()-tick)
                if name in models:
                    estimate=estimate[0].numpy()
                    _,trace=models[name](*obs.args(),diagnostics=True)
                    info={key:[x[key][0].tolist() for x in trace] for key in ('beta','tau','correction','nll')}
                g=truth['g'][i];error=float(np.sum(abs(estimate-g)**2));energy=float(np.sum(abs(g)**2))
                if not np.isfinite(error): raise FloatingPointError(f'invalid estimate {name} trial {i}')
                row=dict(trial=i,method=str(name),error_energy=error,true_energy=energy,
                    times_seconds=times,median_seconds=float(np.median(times)),
                    rho=truth['rho'][i].tolist(),paths=truth['paths'][i].tolist(),
                    rsr_total_realized_db=float(truth['rsr_total_realized_db'][i]),diagnostics=info)
                stream.write(json.dumps(row)+'\n');stream.flush()
            if (i+1)%10==0: print(json.dumps(dict(phase='evaluation',trials=i+1,total=len(observations))),flush=True)
    from .report import make_report
    make_report(out,config)
    return True
