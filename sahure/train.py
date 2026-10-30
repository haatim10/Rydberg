"""Self-supervision only. This module never loads channel truth."""
from dataclasses import asdict
from pathlib import Path
import json
import time
import torch
from .data import Observations,split_observations
from .math import nll
from .model import AdaptiveHankelEstimator,ModelConfig


def heldout_loss(model,context,heldout):
    g=model(*context.args())
    return nll(heldout.z,g@heldout.s+heldout.b,heldout.sigma2).mean()


@torch.no_grad()
def validation_score(model,data,seed,batch_size=32):
    a,b=split_observations(data,seed)
    total=0.
    for start in range(0,len(data),batch_size):
        ids=slice(start,start+batch_size)
        total+=float(heldout_loss(model,a.batch(ids),b.batch(ids)))*len(a.batch(ids))
    return total/len(data)


def choose_checkpoint(score,best_score):
    """Only the held-out likelihood determines the checkpoint decision."""
    if not torch.isfinite(torch.tensor(score)): raise FloatingPointError('nonfinite validation NLL')
    return score < best_score


def load_model(path,dtype=torch.float64):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=AdaptiveHankelEstimator(ModelConfig(**ck['model_config'])).to(dtype=dtype)
    model.load_state_dict(ck['model']);model.eval()
    return model,ck


def train_epoch(model,optimizer,data,epoch,seed,batch_size,deadline):
    # Identical masks and batch order across the six arms for each epoch.
    a,b=split_observations(data,seed+10000+epoch)
    order=torch.randperm(len(data),generator=torch.Generator().manual_seed(seed+20000+epoch))
    total=0.
    model.train()
    for start in range(0,len(data),batch_size):
        if time.monotonic()>=deadline: return None
        ids=order[start:start+batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss=heldout_loss(model,a.batch(ids),b.batch(ids))
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite gradient; no gradient substitution is permitted')
        optimizer.step()
        total+=float(loss.detach())*len(ids)
    return total/len(data)


def train_models(train,val,output,config,deadline):
    """Round-robin epochs give all arms coverage before further training.

    latest checkpoints include optimizer state. If interrupted midway through
    an epoch, the preceding complete epoch is restored on resume. At the cap,
    partial-epoch state is separately saved and not treated as validated.
    """
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    states={}
    for core in ('em','gd'):
        for structure in ('none','global','adaptive'):
            name=f'{core}_{structure}';torch.manual_seed(config['seed'])
            mc=ModelConfig(core=core,structure=structure,stages=config['stages'])
            model=AdaptiveHankelEstimator(mc)
            opt=torch.optim.Adam(model.parameters(),lr=config['lr'])
            path=output/f'{name}_latest.pt'
            state=dict(model_config=asdict(mc),epoch=0,best_score=float('inf'),history=[],train_seconds=0.)
            if path.exists():
                state=torch.load(path,map_location='cpu',weights_only=False)
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
            else:
                state['best_score']=validation_score(model,val,config['seed']+30000)
                state.update(model=model.state_dict(),optimizer=opt.state_dict())
                torch.save(state,output/f'{name}_best.pt');torch.save(state,path)
            states[name]=(model,opt,state)
    for epoch in range(config['epochs']):
        for name,(model,opt,state) in states.items():
            if state['epoch']>epoch: continue
            if time.monotonic()>=deadline: return False
            tick=time.monotonic()
            score=train_epoch(model,opt,train,epoch,config['seed'],config['batch_size'],deadline)
            if score is None:
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch_partial=epoch,
                    model_config=state['model_config']),output/f'{name}_partial.pt')
                return False
            model.eval()
            val_score=validation_score(model,val,config['seed']+30000)
            state['epoch']=epoch+1;state['train_seconds']+=time.monotonic()-tick
            state['history'].append(dict(epoch=epoch+1,train_nll=score,val_nll=val_score,seconds=state['train_seconds']))
            improved=choose_checkpoint(val_score,state['best_score'])
            if improved: state['best_score']=val_score
            state.update(model=model.state_dict(),optimizer=opt.state_dict())
            torch.save(state,output/f'{name}_latest.pt')
            if improved: torch.save(state,output/f'{name}_best.pt')
            (output/f'{name}_history.json').write_text(json.dumps(state['history'],indent=2)+'\n')
            print(json.dumps(dict(phase='training',model=name,epoch=epoch+1,val_nll=val_score,best=state['best_score'],seconds=state['train_seconds'])),flush=True)
    return True


def preflight(data,output,config):
    measurements={}
    a,b=split_observations(data.batch(slice(0,32)),config['seed'])
    for core in ('em','gd'):
        for structure in ('none','global','adaptive'):
            torch.manual_seed(config['seed'])
            model=AdaptiveHankelEstimator(ModelConfig(core=core,structure=structure,stages=config['stages']))
            opt=torch.optim.Adam(model.parameters(),lr=config['lr'])
            times=[]
            for repeat in range(4):
                tick=time.perf_counter();opt.zero_grad();loss=heldout_loss(model,a,b);loss.backward();opt.step()
                if repeat: times.append(time.perf_counter()-tick)
            seconds=sorted(times)[1]
            measurements[f'{core}_{structure}']=dict(batch_seconds=seconds,
                estimated_training_seconds=seconds*(config['train_cases']/config['batch_size'])*config['epochs'],
                parameters=sum(p.numel() for p in model.parameters()))
    Path(output).write_text(json.dumps(measurements,indent=2)+'\n')
    return measurements
