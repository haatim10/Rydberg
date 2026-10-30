"""Matched controls. Only the explicitly supervised arm reads channel labels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
from dataclasses import asdict
import hashlib
import json
import time
import numpy as np
import torch
from sahure.data import load_observations,split_observations
from sahure.model import AdaptiveHankelEstimator,ModelConfig
from sahure.train import train_epoch
from sahure_expansion.train import validate,condition_mean,save_checkpoint
from sahure_expansion.protocol import CONFIG as PREVIOUS,write_json,sha
from .protocol import ROOT,CONFIG,jobs,data_root,check_sources

def model_config(arm):
    return ModelConfig(core='em',structure=arm if arm in ('global','none') else 'adaptive',
        stages=10,final_only=arm=='final_only',gate_one=arm=='gate_one')

def load_labels(cohort,arm,split):
    if arm!='supervised': raise ValueError('Labels forbidden for this training arm')
    with np.load(data_root(cohort)/'data'/f'{split}_truth.npz') as f:
        return torch.tensor(f['g'].copy(),dtype=torch.complex64)

def channel_loss(pred,truth):
    return (pred-truth).abs().square().sum((-2,-1))/truth.abs().square().sum((-2,-1)).clamp_min(1e-20)

def checked_step(loss,model,opt):
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('nonfinite gradient')
    opt.step()

def supervised_epoch(model,opt,data,truth,epoch,seed,deadline):
    a,_=split_observations(data,seed+10000+epoch)
    order=torch.randperm(len(data),generator=torch.Generator().manual_seed(seed+20000+epoch))
    total=0.;model.train()
    for start in range(0,len(data),CONFIG['batch_size']):
        if time.monotonic()>=deadline:return None
        ids=order[start:start+CONFIG['batch_size']];opt.zero_grad(set_to_none=True)
        loss=channel_loss(model(*a.batch(ids).args()),truth[ids]).mean()
        checked_step(loss,model,opt);total+=float(loss.detach())*len(ids)
    return total/len(data)

@torch.no_grad()
def supervised_validation(model,context,truth,conditions):
    model.eval();scores=[]
    for start in range(0,len(context),32):
        ids=slice(start,start+32)
        scores.append(channel_loss(model(*context.batch(ids).args()),truth[ids]))
    return float(condition_mean(torch.cat(scores).double(),conditions)[0])

def order_hash(count,seed,epoch):
    order=torch.randperm(count,generator=torch.Generator().manual_seed(seed+20000+epoch))
    return hashlib.sha256(order.numpy().tobytes()).hexdigest()

def run(job,epochs,seconds):
    protocol=check_sources();torch.set_num_threads(1)
    out=ROOT/'checkpoints'/job['id'];out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tick=time.monotonic();deadline=tick+seconds
        directory=data_root(job['cohort'])/'data'
        data=load_observations(directory/'train_observations.npz')
        val=load_observations(directory/'val_observations.npz')
        conditions=np.load(directory/'val_conditions.npy')
        a,b=split_observations(val,PREVIOUS['validation_mask_seed'])
        truth=load_labels(job['cohort'],job['arm'],'train') if job['arm']=='supervised' else None
        vtruth=load_labels(job['cohort'],job['arm'],'val') if job['arm']=='supervised' else None
        manifest=dict(job=job,protocol_sha256=sha(ROOT/'protocol.json'),
            observations={p.name:sha(p) for p in directory.glob('*observations.npz') if p.name.startswith(('train','val'))},
            label_access=truth is not None)
        mp=out/'manifest.json'
        if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('Training inputs changed')
        write_json(mp,manifest)
        torch.manual_seed(job['seed']);mc=model_config(job['arm']);model=AdaptiveHankelEstimator(mc)
        opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr']);latest=out/'latest.pt'
        if latest.exists():
            state=torch.load(latest,weights_only=False,map_location='cpu')
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        else:
            vnll,_=validate(model,a,b,conditions)
            vnmse=supervised_validation(model,a,vtruth,conditions) if vtruth is not None else None
            state=dict(model_config=asdict(mc),model=model.state_dict(),optimizer=opt.state_dict(),
                epoch=0,best_score=vnmse if vtruth is not None else vnll,best_nll=vnll,history=[],job=job,
                initial_nll=vnll,initial_score=vnmse if vtruth is not None else vnll)
            save_checkpoint(state,out/'best.pt');save_checkpoint(state,out/'nll_best.pt');save_checkpoint(state,latest)
        for epoch in range(state['epoch'],epochs):
            if time.monotonic()>=deadline:break
            et=time.monotonic()
            if truth is None:
                loss=train_epoch(model,opt,data,epoch,job['seed'],CONFIG['batch_size'],deadline)
            else:loss=supervised_epoch(model,opt,data,truth,epoch,job['seed'],deadline)
            if loss is None:break  # Discard interrupted epoch; latest preserves its predecessor.
            vnll,_=validate(model,a,b,conditions)
            vnmse=supervised_validation(model,a,vtruth,conditions) if vtruth is not None else None
            score=vnmse if vtruth is not None else vnll
            if not np.isfinite(score):raise FloatingPointError('nonfinite validation')
            best=score<state['best_score'];best_nll=vnll<state['best_nll']
            state['best_score']=min(score,state['best_score']);state['best_nll']=min(vnll,state['best_nll'])
            row=dict(epoch=epoch+1,loss=loss,val_nll=vnll,val_nmse=vnmse,seconds=time.monotonic()-et,
                mask_seed=job['seed']+10000+epoch,order_sha256=order_hash(len(data),job['seed'],epoch))
            state['history'].append(row);state.update(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict())
            if best:save_checkpoint(state,out/'best.pt')
            if best_nll:save_checkpoint(state,out/'nll_best.pt')
            save_checkpoint(state,latest);write_json(out/'history.json',state['history'])
            write_json(out/'status.json',dict(state='training',job=job,**row))
            print(json.dumps(dict(job=job['id'],**row)),flush=True)
        complete=state['epoch']==CONFIG['epochs']
        write_json(out/'status.json',dict(state='complete' if complete else 'checkpointed',epoch=state['epoch'],job=job))
        return state['epoch']>=epochs

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seconds',type=float,default=43200);a=p.parse_args()
    try:
        done=run(next(j for j in jobs() if j['id']==a.job),a.epochs,a.seconds)
    except Exception as exc:
        write_json(ROOT/'checkpoints'/a.job/'status.json',dict(state='failed',error=repr(exc)));raise
    if not done:raise SystemExit(75)
