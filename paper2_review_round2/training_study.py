"""Bounded, matched reviewer ablations. No channel truth is loaded by training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from sahure.data import load_observations,split_observations
from sahure.train import heldout_loss
from sahure.math import nll
from sahure_expansion.data import world
from sahure_expansion.protocol import write_json
from .models import ARMS,ReviewEstimator
from .common import freeze_phase,save,load,sha,verify_sources

CONFIG=dict(version=1,train_cases=4096,val_cases=448,epochs=12,batch_size=32,lr=.001,
    data_seeds=[202610085001,202610085101,202610085201],initializations=[9201,9202,9203],
    N=32,K=3,P=30,snrs=[-10,-5,0,5,10,15,20],rsrs=[0,5,12,20],paths=[None,2],
    primary='matched depth and width under a reduced, fixed training budget; not a convergence claim',
    selection='equal condition-weighted withheld Rician NLL; fixed 21/9 validation mask',
    curriculum='first 6 epochs RSR12/20 twice, last 6 RSR0/5 twice; same total per-case exposures',
    test_seed=202610086001,test_cases_per_cell=64,invocation_cap_seconds=6600)
GRID=[dict(snr=s,rsr=r,L=l) for r in CONFIG['rsrs'] for s in CONFIG['snrs'] for l in CONFIG['paths']]
JOBS=[dict(name=f'{arm}_d0_s{seed}',arm=arm,replicate=0,seed=seed) for seed in CONFIG['initializations'] for arm in ARMS]
JOBS += [dict(name=f'baseline_d{rep}_s{seed}',arm='baseline',replicate=rep,seed=seed)
         for rep in (1,2) for seed in CONFIG['initializations'][:2]]


def atomic_torch(path,record):
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:name=f.name
    try:torch.save(record,name);os.replace(name,path)
    finally:
        if Path(name).exists():Path(name).unlink()


def make_data(directory,name,count,seed):
    obs=directory/f'{name}_observations.npz';manifest=directory/f'{name}_manifest.json'
    if manifest.exists():
        rec=json.loads(manifest.read_text())
        for p,h in rec['hashes'].items():
            if sha(p)!=h:raise RuntimeError('saved dataset changed')
        return
    # Exactly equal reference counts permit exposure-matched curriculum.
    ids=np.concatenate([np.resize(np.arange(r*14,(r+1)*14),count//4) for r in range(4)])
    ids=np.random.default_rng(seed).permutation(ids)
    arrays={k:[] for k in ('z','s','b','sigma2')}
    for i,cid in enumerate(ids):
        w,_=world(i,seed,**GRID[cid])
        for key,value in zip(arrays,(w.Z,w.S,w.B,w.sigma2)):arrays[key].append(value)
    save(obs,**{k:np.asarray(v) for k,v in arrays.items()},conditions=ids,
         rsr=np.asarray([GRID[c]['rsr'] for c in ids]),trial=np.arange(count),seed=seed)
    write_json(manifest,dict(seed=seed,count=count,grid=GRID,hashes={str(obs):sha(obs)},
                             truth='not saved in training or validation data'))


def epoch_order(data,rsr,epoch,seed,arm):
    order=torch.randperm(len(data),generator=torch.Generator().manual_seed(seed+20000+epoch))
    if arm=='curriculum':
        keep=rsr>=12 if epoch<CONFIG['epochs']//2 else rsr<12
        chosen=order[keep[order]]
        order=chosen.repeat(2)
    return order


def train_epoch(model,opt,data,rsr,epoch,seed,arm,deadline):
    a,b=split_observations(data,seed+10000+epoch)
    order=epoch_order(data,rsr,epoch,seed,arm);total=0.
    model.train()
    for start in range(0,len(order),CONFIG['batch_size']):
        if time.monotonic()>deadline:return None
        ix=order[start:start+CONFIG['batch_size']];opt.zero_grad(set_to_none=True)
        loss=heldout_loss(model,a.batch(ix),b.batch(ix))
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite gradient')
        opt.step();total+=float(loss.detach())*len(ix)
    return total/len(order),hashlib.sha256(order.numpy().tobytes()).hexdigest()


@torch.no_grad()
def validate(model,data,conditions,mask_seed):
    model.eval();a,b=split_observations(data,mask_seed);values=[]
    for start in range(0,len(data),CONFIG['batch_size']):
        ix=slice(start,start+CONFIG['batch_size']);g=model(*a.batch(ix).args());bb=b.batch(ix)
        values.append(nll(bb.z,g@bb.s+bb.b,bb.sigma2).mean((-2,-1)).numpy())
    values=np.concatenate(values)
    return float(np.mean([values[conditions==c].mean() for c in np.unique(conditions)]))


def run(cap=None):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    directory,protocol=freeze_phase('training',dict(CONFIG,jobs=JOBS,arms={a:asdict(c) for a,c in ARMS.items()}),
        [__file__,'paper2_review_round2/models.py','sahure/data.py','sahure/train.py','sahure_expansion/data.py'])
    with (directory/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic();deadline=start+(cap or CONFIG['invocation_cap_seconds'])
        if (directory/'complete.json').exists():print('Training already complete');return
        try:
            write_json(directory/'status.json',dict(state='running',phase='data',pid=os.getpid(),started=time.time()))
            data_dir=directory/'data';data_dir.mkdir(exist_ok=True)
            datasets={}
            for rep,seed in enumerate(CONFIG['data_seeds']):
                for split,count,offset in [('train',CONFIG['train_cases'],0),('val',CONFIG['val_cases'],10)]:
                    make_data(data_dir,f'{split}_d{rep}',count,seed+offset)
                tr=data_dir/f'train_d{rep}_observations.npz';va=data_dir/f'val_d{rep}_observations.npz'
                arrays=load(tr);vals=load(va)
                datasets[rep]=(load_observations(tr),torch.tensor(arrays['rsr']),load_observations(va),vals['conditions'])
            for epoch in range(CONFIG['epochs']):
                for job in JOBS:
                    out=directory/job['name'];out.mkdir(exist_ok=True);latest=out/'latest.pt'
                    torch.manual_seed(job['seed']);model=ReviewEstimator(ARMS[job['arm']])
                    opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'])
                    state=dict(review_config=asdict(ARMS[job['arm']]),job=job,epoch=0,history=[],best_score=float('inf'),
                               protocol_sha256=sha(directory/'protocol.json'))
                    if latest.exists():
                        state=torch.load(latest,map_location='cpu',weights_only=False)
                        if state['protocol_sha256']!=sha(directory/'protocol.json'):raise RuntimeError('checkpoint protocol mismatch')
                        if state['epoch']>epoch:continue
                        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
                    train,rsr,val,conditions=datasets[job['replicate']]
                    if not latest.exists():
                        state['best_score']=validate(model,val,conditions,CONFIG['data_seeds'][job['replicate']]+30000)
                        state.update(model=model.state_dict(),optimizer=opt.state_dict());atomic_torch(latest,state);atomic_torch(out/'selected.pt',state)
                    tick=time.monotonic();result=train_epoch(model,opt,train,rsr,epoch,job['seed'],job['arm'],deadline)
                    if result is None:
                        write_json(directory/'status.json',dict(state='partial',reason='time cap; resumes last complete epoch',
                                   job=job['name'],epoch=epoch,seconds=time.monotonic()-start));return
                    score=validate(model,val,conditions,CONFIG['data_seeds'][job['replicate']]+30000)
                    if not np.isfinite(score):raise FloatingPointError('nonfinite validation')
                    improved=score<state['best_score'];state['best_score']=min(score,state['best_score']);state['epoch']=epoch+1
                    state['history'].append(dict(epoch=epoch+1,train_nll=result[0],val_nll=score,order_hash=result[1],
                        mask_seed=job['seed']+10000+epoch,seconds=time.monotonic()-tick))
                    state.update(model=model.state_dict(),optimizer=opt.state_dict());atomic_torch(latest,state)
                    if improved:atomic_torch(out/'selected.pt',state)
                    write_json(out/'history.json',state['history'])
                    status=dict(state='running',phase='training',job=job['name'],epoch=epoch+1,total_epochs=12,
                        completed_epochs=sum(len(json.loads(p.read_text())) for p in directory.glob('*/history.json')),
                        expected_epochs=len(JOBS)*12,seconds=time.monotonic()-start,pid=os.getpid(),updated_unix=time.time())
                    write_json(directory/'status.json',status);print(json.dumps(status),flush=True)
            verify_sources(protocol)
            hashes={str(p):sha(p) for p in directory.glob('*/*.pt')}
            write_json(directory/'complete.json',dict(jobs=len(JOBS),epochs=len(JOBS)*12,hashes=hashes,seconds=time.monotonic()-start))
            write_json(directory/'status.json',dict(state='complete',phase='training',jobs=len(JOBS),epochs=len(JOBS)*12,
                note='Fresh test evaluation remains separate; no results claim yet',seconds=time.monotonic()-start))
        except BaseException as e:
            write_json(directory/'status.json',dict(state='failed',error=repr(e)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cap-seconds',type=float);args=parser.parse_args();run(args.cap_seconds)
