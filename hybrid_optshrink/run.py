"""Frozen, resumable, two-hour local hybrid pilot. Never changes older work."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
import traceback
import unittest
import numpy as np
import torch
from sahure_expansion.data import world
from sahure.data import Observations, split_observations
from sahure.math import nll
from paper2_optshrink_loop.run import write, save, load
from .model import OptController, build, OLD_SELECTION

ROOT = Path('results/hybrid_optshrink')
ARMS = ('global','adaptive','smooth')
SEEDS = (4101,4102)
SPECS = [dict(id=f'snr_{s:+d}',snr=s,rsr=12,L=None) for s in range(-10,21,5)]
SPECS += [dict(id='weak_reference',snr=5,rsr=0,L=None),dict(id='two_paths',snr=5,rsr=12,L=2),
          dict(id='rich_16',snr=5,rsr=12,L=16),dict(id='nearly_noiseless',snr=60,rsr=12,L=None)]
CONFIG = dict(train_per_cell=256,val_per_cell=32,test_per_cell=64,epochs=12,batch=32,
    lr=.001,seeds=list(SEEDS),arms=list(ARMS),stages=10,max_seconds=7200,
    train_seed=202610111001,val_seed=202610112001,test_seed=202610113001,
    timing_seed=202610114001,bootstrap_seed=202610116001,bootstrap_samples=2000,
    strengths=[0.,.25,.5,.75,1.],residual_cs=[0.,.5,1.,2.],specs=SPECS)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_torch(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as f:
        name=f.name
    torch.save(value,name);os.replace(name,path)


def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads(Path('results/paper2_optshrink_loop/manifest.json').read_text())
    for p,h in old['sources'].items():
        if sha(p)!=h: raise RuntimeError('historical source changed: '+p)
    paths=sorted(Path('hybrid_optshrink').glob('*.py'))+[Path('hybrid_optshrink/PROTOCOL.txt'),
        OLD_SELECTION,Path('sahure/data.py'),Path('sahure/math.py'),Path('sahure/model.py'),
        Path('sahure_expansion/data.py')]
    record=dict(config=CONFIG,sources={str(p):sha(p) for p in paths},old_sources=old['sources'],
        environment=dict(python=platform.python_version(),torch=torch.__version__,
                         numpy=np.__version__,system=platform.platform(),threads=1,dtype='float64'))
    path=ROOT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=record:
        raise RuntimeError('source/config/environment changed; cannot resume frozen pilot')
    if not path.exists():write(path,record)
    return record


def tensors(data):
    return Observations(*(torch.tensor(data[k],dtype=torch.complex128 if k in ('s','b') else torch.float64)
                          for k in ('z','s','b','sigma2')))


def make_cell(split,j,count,seed,check,freeze=None):
    folder=ROOT/'data'/split/SPECS[j]['id'];record=folder/'manifest.json'
    obsfile=folder/'observations.npz';truthfile=folder/'truth.npz'
    if record.exists():
        meta=json.loads(record.read_text())
        if meta['seed']!=seed or meta['count']!=count or meta['freeze']!=freeze:
            raise RuntimeError('data identity/freeze mismatch')
        for p,h in meta['hashes'].items():
            if sha(p)!=h:raise RuntimeError('data changed: '+p)
        return load(obsfile)
    if split=='test' and freeze is None:raise RuntimeError('test requires frozen choices')
    data={k:[] for k in ('z','s','b','sigma2')};truth=[]
    for i in range(count):
        check('data',split=split,cell=SPECS[j]['id'],trial=i,count=count)
        w,_=world(i,seed,**{k:SPECS[j][k] for k in ('snr','rsr','L')})
        for key,value in zip(data,(w.Z,w.S,w.B,w.sigma2)):data[key].append(value)
        if split=='test':truth.append(w.G)
    ids=np.array([f'{split}:{seed}:{i}' for i in range(count)])
    save(obsfile,**{k:np.stack(v) for k,v in data.items()},trial_ids=ids)
    paths=[obsfile]
    if split=='test':save(truthfile,g=np.stack(truth),trial_ids=ids);paths.append(truthfile)
    write(record,dict(split=split,seed=seed,count=count,spec=SPECS[j],freeze=freeze,
        generated_unix=time.time(),hashes={str(p):sha(p) for p in paths}))
    return load(obsfile)


def bank(split,count,seed,check):
    rows=[make_cell(split,j,count,seed+100*j,check) for j in range(7)]
    arrays={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    return tensors(arrays),np.repeat(np.arange(7),count),arrays['trial_ids']


@torch.no_grad()
def validation(model,contexts,heldout,ids):
    scores=[]
    model.eval()
    for start in range(0,len(contexts),32):
        a,b=contexts.batch(slice(start,start+32)),heldout.batch(slice(start,start+32))
        g=model(*a.args());scores.extend(nll(b.z,g@b.s+b.b,b.sigma2).mean((-2,-1)).tolist())
    scores=np.asarray(scores)
    if not np.isfinite(scores).all():raise FloatingPointError('validation loss')
    return float(np.mean([scores[ids==j].mean() for j in range(7)]))


@torch.no_grad()
def classical_selection(val,ids,check):
    p=ROOT/'classical_selection.json'
    if p.exists():return json.loads(p.read_text())
    a,b=split_observations(val,202610112999)
    choices=[(base,c) for base in CONFIG['strengths'] for c in CONFIG['residual_cs']]
    model=OptController('residual').double();rows=[]
    for base,c in choices:
        check('classical_tuning',base=base,residual_c=c)
        scores=[]
        for start in range(0,len(val),32):
            aa,bb=a.batch(slice(start,start+32)),b.batch(slice(start,start+32))
            g=model(*aa.args(),strength_override=base,residual_c=c)
            scores.extend(nll(bb.z,g@bb.s+bb.b,bb.sigma2).mean((-2,-1)).tolist())
        rows.append([float(np.mean(np.asarray(scores)[ids==j])) for j in range(7)])
    scores=np.asarray(rows);fixed=[i for i,(_,c) in enumerate(choices) if c==0]
    chosen={}
    for j in range(7):
        best=int(scores[:,j].argmin());best_fixed=fixed[int(scores[fixed,j].argmin())]
        chosen[str(SPECS[j]['snr'])]=dict(residual=choices[best],fixed=choices[best_fixed])
    result=dict(choices=chosen,candidates=choices,scores=scores.tolist(),
        criterion='mean held-out Rician NLL per supplied noise level',frozen_unix=time.time())
    write(p,result);return result


def train(train_data,val,ids,check):
    va,vb=split_observations(val,202610112999)
    states={}
    for seed in SEEDS:
        for name in ARMS:
            key=f'{name}_s{seed}';path=ROOT/'training'/f'{key}_latest.pt'
            model=build(name,seed);optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'])
            if path.exists():
                state=torch.load(path,map_location='cpu',weights_only=False)
                model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            else:
                check('training_init',model=key)
                score=validation(model,va,vb,ids)
                state=dict(epoch=0,history=[],best=score,best_epoch=0,seconds=0.,
                    model=copy.deepcopy(model.state_dict()),best_model=copy.deepcopy(model.state_dict()),
                    optimizer=optimizer.state_dict(),parameters=sum(p.numel() for p in model.parameters()))
                atomic_torch(path,state);atomic_torch(path.with_name(f'{key}_initial.pt'),state)
            states[key]=(model,optimizer,state)
    for epoch in range(CONFIG['epochs']):
        for seed in SEEDS:
            a,b=split_observations(train_data,202610110000+seed*100+epoch)
            order=torch.randperm(len(train_data),generator=torch.Generator().manual_seed(202611110000+seed*100+epoch))
            order_hash=hashlib.sha256(order.numpy().tobytes()).hexdigest()
            mask_hash=hashlib.sha256(a.s.numpy().tobytes()+b.s.numpy().tobytes()).hexdigest()
            for name in ARMS:
                key=f'{name}_s{seed}';model,optimizer,state=states[key]
                if state['epoch']>epoch:continue
                model.train();tick=time.perf_counter();total=0.;max_grad=0.
                for start in range(0,len(train_data),32):
                    check('training',model=key,epoch=epoch+1,epochs=CONFIG['epochs'],
                          batch=start//32+1,batches=(len(train_data)+31)//32)
                    ii=order[start:start+32];aa,bb=a.batch(ii),b.batch(ii)
                    optimizer.zero_grad(set_to_none=True)
                    g=model(*aa.args());loss=nll(bb.z,g@bb.s+bb.b,bb.sigma2).mean()
                    if not torch.isfinite(loss):raise FloatingPointError('training loss: '+key)
                    loss.backward()
                    for p in model.parameters():
                        if p.grad is not None:
                            if not torch.isfinite(p.grad).all():raise FloatingPointError('gradient: '+key)
                            max_grad=max(max_grad,float(p.grad.detach().abs().max()))
                    optimizer.step();total+=float(loss.detach())*len(ii)
                score=validation(model,va,vb,ids);elapsed=time.perf_counter()-tick
                state.update(epoch=epoch+1,seconds=state['seconds']+elapsed)
                if score<state['best']:
                    state.update(best=score,best_epoch=epoch+1,best_model=copy.deepcopy(model.state_dict()))
                state['history'].append(dict(epoch=epoch+1,train_nll=total/len(train_data),
                    val_nll=score,seconds=elapsed,max_abs_gradient=max_grad,order_hash=order_hash,mask_hash=mask_hash))
                state.update(model=copy.deepcopy(model.state_dict()),optimizer=optimizer.state_dict())
                atomic_torch(ROOT/'training'/f'{key}_latest.pt',state)
                write(ROOT/'training'/f'{key}_history.json',state['history'])
                print(json.dumps(dict(model=key,epoch=epoch+1,val_nll=score,best_epoch=state['best_epoch'],seconds=elapsed)),flush=True)
    for seed in SEEDS:
        for key in ('order_hash','mask_hash'):
            reference=[h[key] for h in states[f'global_s{seed}'][2]['history']]
            for name in ARMS[1:]:
                if reference!=[h[key] for h in states[f'{name}_s{seed}'][2]['history']]:
                    raise RuntimeError('unmatched training '+key)
    for key,(_,_,state) in states.items():
        for which in ('best','final'):
            selected=state['best_model'] if which=='best' else state['model']
            atomic_torch(ROOT/'training'/f'{key}_{which}.pt',dict(model=selected,
                epoch=state['best_epoch'] if which=='best' else state['epoch'],parameters=state['parameters']))
    return {key:{k:state[k] for k in ('epoch','best_epoch','best','seconds','parameters')}
            for key,(_,_,state) in states.items()}


def freeze(training_summary):
    data_paths=sorted((ROOT/'data').glob('*/*/manifest.json'))
    data_hashes={}
    for p in data_paths:
        if '/test/' in str(p):continue
        data_hashes[str(p)]=sha(p);data_hashes.update(json.loads(p.read_text())['hashes'])
    paths=sorted((ROOT/'training').glob('*_best.pt'))+sorted((ROOT/'training').glob('*_final.pt'))
    result=dict(manifest_sha256=sha(ROOT/'manifest.json'),data_hashes=data_hashes,
        checkpoints={str(p):sha(p) for p in paths},classical_sha256=sha(ROOT/'classical_selection.json'),
        training=training_summary,matched_orders=True,matched_masks=True)
    path=ROOT/'frozen_selection.json'
    if path.exists():
        if json.loads(path.read_text())!=result:raise RuntimeError('frozen selection mismatch')
    else:
        if (ROOT/'data/test').exists():raise RuntimeError('test generated before freeze')
        write(path,result)
    return result


def learned_models():
    result={}
    for seed in SEEDS:
        for name in ARMS:
            for which in ('best','final'):
                key=f'{name}_s{seed}_{which}'
                state=torch.load(ROOT/'training'/f'{name}_s{seed}_{which}.pt',map_location='cpu',weights_only=False)
                model=build(name,seed);model.load_state_dict(state['model']);model.eval();result[key]=model
    return result


def controls(v,selection):
    level=int(np.arange(-10,21,5)[np.argmin(abs(np.arange(-10,21,5)-10*np.log10(3/v)))])
    old=json.loads(OLD_SELECTION.read_text())['noise_aware']
    chosen=selection['choices'][str(level)]
    result={'opt_fixed10':(OptController('fixed').double(),{}),
        'opt_tuned10':(OptController('fixed').double(),dict(strength_override=chosen['fixed'][0])),
        'opt_residual10':(OptController('residual').double(),dict(strength_override=chosen['residual'][0],residual_c=chosen['residual'][1]))}
    tables=[old[str(s)]['loop'] for s in range(-10,21,5)]
    result['opt_strong']=(OptController('fixed',stages=old[str(level)]['loop']['iterations'],tables=tables).double(),{})
    result['em100']=(OptController('fixed',stages=100).double(),dict(strength_override=0.))
    return result


@torch.no_grad()
def evaluate(selection,check):
    models=learned_models();digest=sha(ROOT/'frozen_selection.json')
    for j,spec in enumerate(SPECS):
        data=make_cell('test',j,CONFIG['test_per_cell'],CONFIG['test_seed']+100*j,check,freeze=digest)
        output=ROOT/'test_outputs'/f'{spec["id"]}.npz'
        if output.exists():
            if str(load(output)['selection_sha256'])!=digest:raise RuntimeError('cached output selection mismatch')
            continue
        obs=tensors(data);control=controls(float(obs.sigma2[0]),selection)
        all_models={**{n:(m,{}) for n,m in models.items()},**control}
        preds=[];times=[];diagnostics=[];context=[];context_names=[];diag_names=[]
        ca,_=split_observations(obs,202610113999+j)
        for name,(model,kwargs) in all_models.items():
            check('evaluation',cell=spec['id'],method=name)
            rows=[];ticks=[];diag=[];cx=[]
            for start in range(0,len(obs),32):
                a=obs.batch(slice(start,start+32));tick=time.perf_counter()
                if name.startswith(('adaptive','global')):
                    g,d=model(*a.args(),diagnostics=True,**kwargs);diag.append(d.numpy())
                else:g=model(*a.args(),**kwargs)
                ticks.append(time.perf_counter()-tick);rows.append(g.numpy())
                if name.endswith('_best'):
                    cx.append(model(*ca.batch(slice(start,start+32)).args()).numpy())
            preds.append(np.concatenate(rows));times.append(sum(ticks))
            if diag:diagnostics.append(np.concatenate(diag));diag_names.append(name)
            if cx:context.append(np.concatenate(cx));context_names.append(name)
        estimates=np.stack(preds)
        if not np.isfinite(estimates).all():raise FloatingPointError('test estimates')
        save(output,names=np.array(list(all_models)),estimates=estimates,seconds=np.array(times),
            trial_ids=data['trial_ids'],selection_sha256=digest,context_names=np.array(context_names),
            context_estimates=np.stack(context),diagnostic_names=np.array(diag_names),diagnostics=np.stack(diagnostics))
        print('Test cell complete: '+spec['id'],flush=True)


@torch.no_grad()
def timing(selection,check):
    target=ROOT/'timing.json'
    if target.exists():return
    learned={n:m for n,m in learned_models().items() if n.endswith('_best')}
    result={}
    for i in range(12):
        spec=SPECS[i%7];w,_=world(i,CONFIG['timing_seed'],**{k:spec[k] for k in ('snr','rsr','L')})
        obs=tensors(dict(z=w.Z[None],s=w.S[None],b=w.B[None],sigma2=np.array([w.sigma2])))
        methods={**{n:(m,{}) for n,m in learned.items()},**controls(w.sigma2,selection)}
        for model,kwargs in methods.values():model(*obs.args(),**kwargs)
        for rep in range(3):
            names=list(methods);shift=(i+rep)%len(names);names=names[shift:]+names[:shift]
            for name in names:
                check('timing',world=i,repetition=rep,method=name)
                model,kwargs=methods[name];tick=time.perf_counter();model(*obs.args(),**kwargs)
                result.setdefault(name,[]).append(time.perf_counter()-tick)
    write(target,dict(seconds=result,worlds=12,repeats=3,backend='PyTorch CPU float64 single thread',
                      includes='initialization and all iterations; excludes loading and I/O'))


def benchmark(check):
    data=[];start=time.perf_counter()
    for i in range(32):
        w,_=world(i,202610110002,snr=[-10,5,20][i%3]);data.append(w)
    obs=tensors(dict(z=np.stack([w.Z for w in data]),s=np.stack([w.S for w in data]),
        b=np.stack([w.B for w in data]),sigma2=np.array([w.sigma2 for w in data])))
    a,b=split_observations(obs,81);record={'generation32_seconds':time.perf_counter()-start,'training_batch_seconds':{}}
    for name in ARMS:
        model=build(name);opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr']);elapsed=[]
        for i in range(4):
            check('benchmark',model=name,batch=i)
            tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss=nll(b.z,model(*a.args())@b.s+b.b,b.sigma2).mean();loss.backward();opt.step()
            elapsed.append(time.perf_counter()-tick)
        record['training_batch_seconds'][name]=float(np.mean(elapsed[1:]))
    record['estimated_training_seconds']=sum(record['training_batch_seconds'].values())*56*12*2
    record['estimated_total_range_minutes']=[record['estimated_training_seconds']*1.3/60,
                                            record['estimated_training_seconds']*2./60+5]
    record['scope']='rough throughput estimate, not convergence or guaranteed completion time'
    write(ROOT/'benchmark.json',record);print(json.dumps(record,indent=2),flush=True)


def run(phase):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initialize();start=time.monotonic();last=0.
        previous=json.loads((ROOT/'budget.json').read_text())['seconds'] if (ROOT/'budget.json').exists() else 0.
        def check(state,**more):
            nonlocal last
            used=previous+time.monotonic()-start
            if used>=CONFIG['max_seconds']:raise TimeoutError('two-hour cumulative budget exhausted')
            if time.monotonic()-last>2:
                write(ROOT/'status.json',dict(state=state,phase=phase,pid=os.getpid(),seconds=used,
                    updated_unix=time.time(),**more));write(ROOT/'budget.json',dict(seconds=used));last=time.monotonic()
        write(ROOT/'pid.json',dict(pid=os.getpid(),started_unix=time.time(),phase=phase))
        try:
            if phase=='benchmark':benchmark(check)
            else:
                suite=unittest.defaultTestLoader.loadTestsFromName('hybrid_optshrink.test_checks')
                if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise RuntimeError('preflight tests failed')
                if (ROOT/'frozen_selection.json').exists():
                    frozen=json.loads((ROOT/'frozen_selection.json').read_text())
                    if frozen['manifest_sha256']!=sha(ROOT/'manifest.json'):raise RuntimeError('frozen manifest changed')
                    for p,h in {**frozen['data_hashes'],**frozen['checkpoints']}.items():
                        if sha(p)!=h:raise RuntimeError('frozen input changed: '+p)
                    if sha(ROOT/'classical_selection.json')!=frozen['classical_sha256']:raise RuntimeError('classical selection changed')
                    selection=json.loads((ROOT/'classical_selection.json').read_text())
                else:
                    tr,_,train_ids=bank('train',CONFIG['train_per_cell'],CONFIG['train_seed'],check)
                    va,ids,val_ids=bank('val',CONFIG['val_per_cell'],CONFIG['val_seed'],check)
                    if set(train_ids)&set(val_ids):raise RuntimeError('split overlap')
                    selection=classical_selection(va,ids,check)
                    summary=train(tr,va,ids,check);freeze(summary)
                evaluate(selection,check);timing(selection,check)
                from .analyze import analyze
                check('audit_and_analysis');analyze(check)
            initialize()
            write(ROOT/'status.json',dict(state='benchmark_complete' if phase=='benchmark' else 'complete',
                phase=phase,seconds=previous+time.monotonic()-start,updated_unix=time.time()))
        except TimeoutError as e:
            write(ROOT/'status.json',dict(state='budget_exhausted',error=str(e),resumable=True));print(str(e),flush=True)
        except Exception:
            write(ROOT/'status.json',dict(state='failed',error=traceback.format_exc(),updated_unix=time.time()));raise
        finally:write(ROOT/'budget.json',dict(seconds=previous+time.monotonic()-start))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('benchmark','run'),default='run')
    run(parser.parse_args().phase)
