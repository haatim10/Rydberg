"""Train the single new control, without channel truth or protocol edits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import fcntl
import json
import time
import torch
import numpy as np
from sahure.data import load_observations,split_observations
from sahure.train import train_epoch,heldout_loss
from sahure_expansion.train import validate,save_checkpoint
from .model import RecurrentHankelEstimator
from .protocol import ROOT,DATA,CONFIG,write,initialize


def benchmark():
    torch.set_num_threads(1)
    data=load_observations(DATA/'train_observations.npz').batch(slice(0,32))
    a,b=split_observations(data,CONFIG['seeds'][0]+10000)
    torch.manual_seed(CONFIG['seeds'][0]);model=RecurrentHankelEstimator()
    opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr']);times=[]
    for j in range(5):
        tick=time.perf_counter();opt.zero_grad(set_to_none=True)
        loss=heldout_loss(model,a,b);loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('preflight loss/gradient not finite')
        opt.step()
        if j: times.append(time.perf_counter()-tick)
    seconds=float(np.median(times));record=dict(parameters=sum(p.numel() for p in model.parameters()),batch_seconds=seconds,
        estimated_train_seconds_per_seed=seconds*(CONFIG['train_cases']/32)*20,
        note='single-worker timing; excludes full validation, parallel workers may slow each other')
    write(ROOT/'benchmark.json',record);print(json.dumps(record),flush=True)


def run(seed):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    folder=ROOT/'checkpoints'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initialize()
        train=load_observations(DATA/'train_observations.npz')
        val=load_observations(DATA/'val_observations.npz');conditions=np.load(DATA/'val_conditions.npy')
        if len(train)!=CONFIG['train_cases'] or len(val)!=CONFIG['val_cases']: raise RuntimeError('data size changed')
        a,b=split_observations(val,CONFIG['validation_mask_seed'])
        torch.manual_seed(seed);model=RecurrentHankelEstimator();opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'])
        latest=folder/'recurrent_latest.pt';best=folder/'recurrent_best.pt'
        if latest.exists():
            state=torch.load(latest,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        else:
            initial,strata=validate(model,a,b,conditions)
            state=dict(model_config=dict(stages=10,hidden_size=4),epoch=0,best_score=initial,history=[],train_seconds=0.,seed=seed,
                model=model.state_dict(),optimizer=opt.state_dict(),validation_strata=strata)
            save_checkpoint(state,best);save_checkpoint(state,latest)
        for epoch in range(state['epoch'],CONFIG['epochs']):
            write(folder/'status.json',dict(state='training',pid=os.getpid(),seed=seed,epoch=epoch+1,completed_epochs=state['epoch'],updated_unix=time.time()))
            tick=time.monotonic();loss=train_epoch(model,opt,train,epoch,seed,32,float('inf'))
            score,strata=validate(model,a,b,conditions);seconds=time.monotonic()-tick
            improved=score<state['best_score']
            state.update(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),validation_strata=strata)
            state['train_seconds']+=seconds;state['history'].append(dict(epoch=epoch+1,train_nll=loss,val_nll=score,seconds=seconds))
            if improved: state['best_score']=score;save_checkpoint(state,best)
            save_checkpoint(state,latest);write(folder/'history.json',state['history'])
            print(json.dumps(dict(seed=seed,epoch=epoch+1,train_nll=loss,val_nll=score,best=state['best_score'],seconds=seconds)),flush=True)
        write(folder/'status.json',dict(state='complete',seed=seed,epochs=20,updated_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int);parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    if args.benchmark: benchmark()
    elif args.seed in CONFIG['seeds']: run(args.seed)
    else: raise ValueError('registered seed required')
