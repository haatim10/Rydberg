"""Mixed-condition training: observations only, equal-stratum validation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
from dataclasses import asdict
import json
import time
import numpy as np
import torch
from sahure.data import load_observations,split_observations
from sahure.math import nll
from sahure.model import AdaptiveHankelEstimator,ModelConfig
from sahure.train import train_epoch
from .protocol import ROOT,CONFIG,initialize,write_json,sha


def condition_mean(values,conditions):
    """Equal weight per observed condition, not per validation case."""
    conditions=torch.as_tensor(conditions,dtype=torch.long,device=values.device)
    ids,counts=torch.unique(conditions,sorted=True,return_counts=True)
    if not torch.equal(ids,torch.arange(len(ids),device=values.device)):
        raise ValueError('condition IDs must be contiguous and all represented')
    totals=torch.zeros(len(ids),dtype=values.dtype,device=values.device).scatter_add_(0,conditions,values)
    return (totals/counts).mean(),totals/counts


@torch.no_grad()
def validate(model,a,b,conditions):
    scores=[];model.eval()
    for start in range(0,len(a),32):
        ac=a.batch(slice(start,start+32));bc=b.batch(slice(start,start+32))
        g=model(*ac.args())
        scores.append(nll(bc.z,g@bc.s+bc.b,bc.sigma2).mean((-2,-1)))
    score,strata=condition_mean(torch.cat(scores).double(),conditions)
    if not torch.isfinite(score): raise FloatingPointError('nonfinite validation score')
    return float(score),strata.tolist()


def save_checkpoint(state,path):
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


def run(seed,epoch_limit):
    initialize();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=ROOT/'checkpoints'/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    train=load_observations(ROOT/'data/train_observations.npz')
    val=load_observations(ROOT/'data/val_observations.npz')
    conditions=np.load(ROOT/'data/val_conditions.npy')
    a,b=split_observations(val,CONFIG['validation_mask_seed'])
    source={str(p):sha(p) for p in [__file__,'sahure/model.py','sahure/math.py','sahure/train.py','sahure/data.py']}
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())['sources']!=source:
        raise RuntimeError('training code changed during campaign')
    write_json(mp,dict(seed=seed,sources=source,config=CONFIG,torch=torch.__version__))
    for epoch in range(epoch_limit):
        for core in ('em','gd'):
            for structure in ('none','global','adaptive'):
                name=f'{core}_{structure}';latest=out/f'{name}_latest.pt'
                torch.manual_seed(seed)
                mc=ModelConfig(core=core,structure=structure,stages=CONFIG['stages'])
                model=AdaptiveHankelEstimator(mc)
                opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'])
                if latest.exists():
                    state=torch.load(latest,map_location='cpu',weights_only=False)
                    if state['epoch']>epoch: continue
                    if state['epoch']!=epoch: raise RuntimeError('missing preceding epoch')
                    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
                else:
                    initial,strata=validate(model,a,b,conditions)
                    state=dict(model_config=asdict(mc),epoch=0,best_score=initial,history=[],train_seconds=0.,seed=seed,
                        model=model.state_dict(),optimizer=opt.state_dict(),validation_strata=strata)
                    save_checkpoint(state,out/f'{name}_best.pt')
                tick=time.monotonic()
                loss=train_epoch(model,opt,train,epoch,seed,CONFIG['batch_size'],float('inf'))
                score,strata=validate(model,a,b,conditions)
                seconds=time.monotonic()-tick
                improved=score<state['best_score']
                state.update(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),validation_strata=strata)
                state['train_seconds']+=seconds
                state['history'].append(dict(epoch=epoch+1,train_nll=loss,val_nll=score,seconds=seconds))
                if improved: state['best_score']=score;save_checkpoint(state,out/f'{name}_best.pt')
                save_checkpoint(state,latest)
                write_json(out/f'{name}_history.json',state['history'])
                print(json.dumps(dict(seed=seed,model=name,epoch=epoch+1,val_nll=score,best=state['best_score'],seconds=seconds)),flush=True)
    write_json(out/'status.json',dict(state='complete' if epoch_limit==CONFIG['epochs'] else 'preflight_complete',epochs=epoch_limit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=CONFIG['epochs'])
    args=p.parse_args()
    if args.seed not in CONFIG['seeds']: raise ValueError('unregistered seed')
    run(args.seed,args.epochs)
