"""Observation-only training; equal budgets, initialization, masks and order."""
import hashlib
import time
import numpy as np
import torch
from sahure.data import load_observations, split_observations
from sahure.train import train_epoch
from sahure_expansion.train import validate, save_checkpoint
from .model import FeatureEstimator
from .protocol import ROOT, OLD, CONFIG, jobs, sha, write_json

def data():
    train=load_observations(OLD/'train_observations.npz').batch(slice(0,CONFIG['train_cases']))
    val=load_observations(OLD/'val_observations.npz')
    conditions=np.load(OLD/'val_conditions.npy')
    assert len(val)==CONFIG['val_cases'] and len(np.unique(conditions))==126
    a,b=split_observations(val,CONFIG['validation_mask_seed'])
    return train,a,b,conditions

def order_hash(seed,epoch,count):
    order=torch.randperm(count,generator=torch.Generator().manual_seed(seed+20000+epoch))
    return hashlib.sha256(order.numpy().tobytes()).hexdigest()

def completed_epochs():
    total=0
    for job in jobs():
        path=ROOT/'checkpoints'/job['id']/'history.json'
        if path.exists():
            import json
            total+=len(json.loads(path.read_text()))
    return total

def run(deadline):
    train,a,b,conditions=data()
    # Epoch-round-robin provides balanced partial progress if the cap is reached.
    for epoch in range(CONFIG['epochs']):
        for job in jobs():
            if time.monotonic()>=deadline: return False
            out=ROOT/'checkpoints'/job['id'];out.mkdir(parents=True,exist_ok=True)
            latest=out/'latest.pt'
            torch.manual_seed(job['seed'])
            model=FeatureEstimator(job['arm'])
            opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'])
            if latest.exists():
                state=torch.load(latest,map_location='cpu',weights_only=False)
                if state['epoch']>epoch: continue
                if state['epoch']!=epoch: raise RuntimeError('Missing preceding epoch')
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
            else:
                initial,_=validate(model,a,b,conditions)
                state=dict(arm=job['arm'],seed=job['seed'],epoch=0,best_score=initial,history=[],
                    model=model.state_dict(),optimizer=opt.state_dict(),label_access=False,
                    protocol_sha256=sha(ROOT/'protocol.json'))
                save_checkpoint(state,out/'best.pt');save_checkpoint(state,latest)
            started=time.monotonic()
            write_json(ROOT/'status.json',dict(state='training',job=job['id'],epoch=epoch+1,
                completed_epochs=completed_epochs(),total_epochs=len(jobs())*CONFIG['epochs'],updated_unix=time.time()))
            loss=train_epoch(model,opt,train,epoch,job['seed'],CONFIG['batch_size'],deadline)
            if loss is None: return False  # Resume the last complete epoch, never use partial updates.
            score,strata=validate(model,a,b,conditions)
            row=dict(epoch=epoch+1,train_nll=loss,val_nll=score,seconds=time.monotonic()-started,
                mask_seed=job['seed']+10000+epoch,order_sha256=order_hash(job['seed'],epoch,len(train)))
            improved=score<state['best_score']
            state['best_score']=min(score,state['best_score']);state['history'].append(row)
            state.update(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),validation_strata=strata)
            if improved: save_checkpoint(state,out/'best.pt')
            save_checkpoint(state,latest);write_json(out/'history.json',state['history'])
            print(dict(job=job['id'],**row),flush=True)
    return True
