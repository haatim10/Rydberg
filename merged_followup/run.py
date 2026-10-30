"""Bounded paired development runs; preserve all frozen historical studies."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import time
import traceback
import numpy as np
import torch
from sahure_expansion.data import world
from sahure.data import load_observations, split_observations
from sahure.math import nll
from sahure.train import load_model
from paper2_optshrink_loop.run import write, save, load
from paper2_optshrink_loop.estimator import estimate as opt_estimate, noise_index
from merged_review_checks.run import one_sweep
from codex_verification.solver import fit
from .estimator import context_candidates, validation_scores, estimate, refit, SCHEDULES
from .models import build

ROOT=Path('results/merged_followup')
OLD=Path('results/paper2_optshrink_loop')
ARMS=('compact','em_transformer')
SEED=202610087101
SPECS=[dict(id=f'snr_{s:+d}',snr=s,rsr=12,L=None) for s in range(-10,21,5)]
SPECS += [dict(id='weak_reference',snr=5,rsr=0,L=None),
          dict(id='two_paths',snr=5,rsr=12,L=2),
          dict(id='nearly_noiseless',snr=60,rsr=12,L=None),
          dict(id='rich_16',snr=5,rsr=12,L=16)]
MAX_SECONDS=7200


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads((OLD/'manifest.json').read_text())
    for p,d in old['sources'].items():
        if sha(p)!=d: raise RuntimeError(f'frozen source changed: {p}')
    for m in old['models'].values():
        if sha(m['path'])!=m['sha256']: raise RuntimeError('frozen model changed')
    paths=[Path('merged_followup')/n for n in ('__init__.py','run.py','estimator.py','models.py','test_checks.py')]+[Path('merged_followup/PROTOCOL.txt'),
        Path('trackD_urformer/transformer.py'),Path('merged_review_checks/run.py'),
        OLD/'frozen_selection.json',OLD/'manifest.json']
    data=Path('results/sahure_expansion/data')
    paths += [data/n for n in ('train_observations.npz','val_observations.npz','val_conditions.npy',
                              'train_generation.json','val_generation.json')]
    record=dict(sources={str(p):sha(p) for p in paths},old_sources=old['sources'],
        old_models=old['models'],config=dict(cases=32,tune_cases=16,specs=SPECS,
        epochs=12,train_cases=4096,seed=SEED,max_seconds=MAX_SECONDS),
        environment=dict(torch=torch.__version__,numpy=np.__version__,threads=1))
    path=ROOT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=record:
        raise RuntimeError('resume source/config/input mismatch')
    if not path.exists(): write(path,record)
    return record


def settings(spec): return {k:spec[k] for k in ('snr','rsr','L')}


def old_config(v):
    frozen=json.loads((OLD/'frozen_selection.json').read_text())
    return frozen['noise_aware'][str(list(range(-10,21,5))[noise_index(v)])]['loop']


def data_world(seed,i,spec):
    w,_=world(i,seed,**settings(spec))
    return w, dict(z=w.Z,s=w.S,b=w.B,variance=np.array(w.sigma2),trial_id=f'{seed}:{i}')


def freeze_release(check):
    target=ROOT/'release_selection.json'
    if target.exists():
        result=json.loads(target.read_text())
        for p,h in result['tuning_hashes'].items():
            if sha(p)!=h: raise RuntimeError('tuning record changed')
        return result
    choices={}; hashes={}
    for j,spec in enumerate(SPECS[:7]):
        rows=[]
        for i in range(16):
            check('tuning',cell=spec['id'],trial=i)
            p=ROOT/'tuning'/spec['id']/f'{i:03d}.npz'
            if not p.exists():
                _,d=data_world(202610085101+100*j,i,spec)
                cfg=old_config(float(d['variance']))
                gs=context_candidates(d['z'],d['s'],d['b'],float(d['variance']),cfg)
                scores=validation_scores(gs,d['z'],d['s'],d['b'],float(d['variance']))
                save(p,**d,scores=scores)
            rows.append(load(p)['scores']); hashes[str(p)]=sha(p)
        average=np.mean(rows,axis=0); idx=int(average.argmin())
        choices[str(spec['snr'])]=dict(schedule=SCHEDULES[idx],scores=average.tolist())
    result=dict(choices=choices,tuning_hashes=hashes,manifest_sha256=sha(ROOT/'manifest.json'),
                frozen_unix=time.time(),screen_not_generated=not (ROOT/'classical').exists())
    if not result['screen_not_generated']: raise RuntimeError('screen exists before freeze')
    write(target,result)
    return result


@torch.no_grad()
def classical(check,manifest):
    frozen=freeze_release(check)
    models={s:load_model(m['path'])[0] for s,m in manifest['old_models'].items()}
    for j,spec in enumerate(SPECS):
        for i in range(32):
            check('classical',cell=spec['id'],trial=i,total=352,completed=32*j+i)
            p=ROOT/'classical'/spec['id']/f'{i:03d}.npz'
            if p.exists(): continue
            w,d=data_world(202610086101+100*j,i,spec)
            z,s,b,v=d['z'],d['s'],d['b'],float(d['variance'])
            cfg=old_config(v); out={}; times={}; contexts={}; schedules={}
            def timed(name,fn):
                tick=time.perf_counter(); out[name]=fn(); times[name]=time.perf_counter()-tick
            timed('opt_frozen',lambda:opt_estimate(z,s,b,v,cfg))
            for mode in ('stop','em','release'):
                name='observed_'+mode
                tick=time.perf_counter();g,info=estimate(z,s,b,v,dict(**cfg,mode=mode))
                out[name]=g; times[name]=time.perf_counter()-tick
                contexts[name]=info['context']; schedules[name]=info['schedule']
            snr=str(list(range(-10,21,5))[noise_index(v)])
            fixed=tuple(frozen['choices'][snr]['schedule'])
            timed('fixed_release',lambda:refit(z,s,b,v,cfg,fixed))
            timed('em100',lambda:fit(s,z,b,v,iterations=100)[0])
            timed('hard_one_sweep',lambda:one_sweep(z,s,b,v)[0])
            for seed,model in models.items():
                timed('old_compact_'+seed,lambda m=model:m.estimate(z,s,b,v).numpy())
            names=list(out); gs=np.stack(list(out.values()))
            if not np.isfinite(gs).all(): raise FloatingPointError('nonfinite classical output')
            save(p,**d,truth=w.G,names=np.array(names),estimates=gs,
                seconds=np.array([times[n] for n in names]),
                context_names=np.array(list(contexts)),context_estimates=np.stack(list(contexts.values())),
                schedules=np.array(list(schedules.values())),fixed_schedule=fixed,
                selection_sha256=sha(ROOT/'release_selection.json'))
        print('Classical development cell complete: '+spec['id'],flush=True)


def atomic_torch(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as f:
        temp=f.name
    torch.save(value,temp);os.replace(temp,path)


@torch.no_grad()
def validation(model,data,ids):
    a,b=split_observations(data,SEED+30000)
    values=[]
    for start in range(0,len(data),32):
        aa,bb=a.batch(slice(start,start+32)),b.batch(slice(start,start+32))
        g=model(*aa.args())
        values.extend(nll(bb.z,g@bb.s+bb.b,bb.sigma2).mean((-2,-1)).tolist())
    values=np.array(values)
    if not np.isfinite(values).all(): raise FloatingPointError('validation NLL')
    return float(np.mean([values[ids==k].mean() for k in np.unique(ids)]))


def train(check):
    folder=ROOT/'learned'; folder.mkdir(parents=True,exist_ok=True)
    data=Path('results/sahure_expansion/data')
    tr=load_observations(data/'train_observations.npz').batch(slice(0,4096))
    val=load_observations(data/'val_observations.npz')
    ids=np.load(data/'val_conditions.npy')
    states={}
    for name in ARMS:
        check('training_init',model=name)
        model=build(name); opt=torch.optim.Adam(model.parameters(),lr=.001)
        path=folder/f'{name}_latest.pt'
        if path.exists():
            state=torch.load(path,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        else:
            score=validation(model,val,ids)
            state=dict(epoch=0,history=[],best=score,best_epoch=0,train_seconds=0.,
                       parameters=sum(p.numel() for p in model.parameters()),
                       model=model.state_dict(),optimizer=opt.state_dict())
            atomic_torch(path,state);atomic_torch(folder/f'{name}_best.pt',state)
        states[name]=(model,opt,state)
    for epoch in range(12):
        a,b=split_observations(tr,SEED+10000+epoch)
        order=torch.randperm(len(tr),generator=torch.Generator().manual_seed(SEED+20000+epoch))
        order_hash=hashlib.sha256(order.numpy().tobytes()).hexdigest()
        mask_hash=hashlib.sha256(a.s.numpy().tobytes()+b.s.numpy().tobytes()).hexdigest()
        for name,(model,opt,state) in states.items():
            if state['epoch']>epoch: continue
            tick=time.perf_counter(); total=0.;model.train()
            for start in range(0,len(tr),32):
                check('training',model=name,epoch=epoch+1,batch=start//32,batches=128)
                ii=order[start:start+32];aa,bb=a.batch(ii),b.batch(ii)
                opt.zero_grad(set_to_none=True);g=model(*aa.args())
                loss=nll(bb.z,g@bb.s+bb.b,bb.sigma2).mean()
                if not torch.isfinite(loss): raise FloatingPointError('training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('gradient')
                opt.step();total+=float(loss.detach())*len(ii)
            model.eval();score=validation(model,val,ids)
            elapsed=time.perf_counter()-tick;state['train_seconds']+=elapsed;state['epoch']=epoch+1
            improved=score<state['best']
            if improved:state['best']=score;state['best_epoch']=epoch+1
            state['history'].append(dict(epoch=epoch+1,train_nll=total/len(tr),val_nll=score,
                seconds=elapsed,order_hash=order_hash,mask_hash=mask_hash))
            state.update(model=model.state_dict(),optimizer=opt.state_dict())
            atomic_torch(folder/f'{name}_latest.pt',state)
            if improved:atomic_torch(folder/f'{name}_best.pt',state)
            write(folder/f'{name}_history.json',state['history'])
            print(json.dumps(dict(model=name,epoch=epoch+1,val_nll=score,best_epoch=state['best_epoch'])),flush=True)
    for key in ('order_hash','mask_hash'):
        assert [h[key] for h in states[ARMS[0]][2]['history']]==[h[key] for h in states[ARMS[1]][2]['history']]
    selection=dict(checkpoints={str(folder/f'{n}_{which}.pt'):sha(folder/f'{n}_{which}.pt')
        for n in ARMS for which in ('best','latest')},manifest_sha256=sha(ROOT/'manifest.json'),
        matched_orders=True,matched_masks=True,summary={n:{k:state[k] for k in
            ('best','best_epoch','epoch','train_seconds','parameters')} for n,(_,_,state) in states.items()})
    target=folder/'frozen_selection.json'
    if target.exists():
        if json.loads(target.read_text())!=selection:raise RuntimeError('learned selection changed')
    else:
        if (ROOT/'learned_test').exists():raise RuntimeError('learned test exists before selection')
        write(target,selection)
    return selection


@torch.no_grad()
def learned_test(check,selection):
    models={}
    for name in ARMS:
        for label in ('best','latest'):
            p=ROOT/'learned'/f'{name}_{label}.pt'
            assert sha(p)==selection['checkpoints'][str(p)]
            ck=torch.load(p,map_location='cpu',weights_only=False)
            model=build(name,dtype=torch.float64);model.load_state_dict(ck['model']);model.eval()
            models[name+'_'+label]=model
    for j,spec in enumerate(SPECS):
        for i in range(32):
            check('learned_test',cell=spec['id'],trial=i,total=352,completed=j*32+i)
            p=ROOT/'learned_test'/spec['id']/f'{i:03d}.npz'
            if p.exists():continue
            w,d=data_world(202610088101+100*j,i,spec)
            z,s,b,v=d['z'],d['s'],d['b'],float(d['variance']);out={};times={};context=[]
            zz,ss,bb,vv=torch.tensor(z[None]),torch.tensor(s[None]),torch.tensor(b[None]),torch.tensor([v],dtype=torch.float64)
            for name,model in models.items():
                tick=time.perf_counter();out[name]=model(zz,ss,bb,vv)[0].numpy();times[name]=time.perf_counter()-tick
                context.append(model(zz[:,:,:21],ss[:,:,:21],bb[:,:,:21],vv)[0].numpy())
            for name,fn in [('opt_frozen',lambda:opt_estimate(z,s,b,v,old_config(v))),
                            ('em100',lambda:fit(s,z,b,v,iterations=100)[0])]:
                tick=time.perf_counter();out[name]=fn();times[name]=time.perf_counter()-tick
            gs=np.stack(list(out.values()))
            if not np.isfinite(gs).all():raise FloatingPointError('learned output')
            save(p,**d,truth=w.G,names=np.array(list(out)),estimates=gs,seconds=np.array(list(times.values())),
                 context_names=np.array(list(models)),context_estimates=np.stack(context),
                 selection_sha256=sha(ROOT/'learned/frozen_selection.json'))
        print('Matched learned development cell complete: '+spec['id'],flush=True)


def run(phase):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=initialize();started=time.monotonic()
        previous=json.loads((ROOT/'budget.json').read_text())['seconds'] if (ROOT/'budget.json').exists() else 0.
        last=0.
        def check(state,**more):
            nonlocal last
            used=previous+time.monotonic()-started
            if used>=MAX_SECONDS:raise TimeoutError('cumulative two-hour compute cap reached')
            if time.monotonic()-last>2:
                write(ROOT/'status.json',dict(state=state,phase=phase,pid=os.getpid(),seconds=used,**more))
                last=time.monotonic()
        try:
            if phase in ('classical','all'):classical(check,manifest)
            if phase in ('learned','all'):learned_test(check,train(check))
            initialize()
            write(ROOT/'status.json',dict(state='complete',phase=phase,seconds=previous+time.monotonic()-started))
        except TimeoutError as e:
            write(ROOT/'status.json',dict(state='budget_exhausted',error=str(e),resumable=True))
        except Exception:
            write(ROOT/'status.json',dict(state='failed',error=traceback.format_exc()));raise
        finally:
            write(ROOT/'budget.json',dict(seconds=previous+time.monotonic()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['classical','learned','all'],default='all')
    run(parser.parse_args().phase)
