"""Resumable paired 21/30-context training, unchanged EM-adaptive architecture."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
from dataclasses import asdict
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import nll
from sahure.model import AdaptiveHankelEstimator,ModelConfig
from .training_protocol import ROOT,CONFIG,freeze,sha,write_json,now
from .training_data import generate_set,load_set,paired_split,split_indices,mask_evidence,tensor_digest,simulate_observation,condition_grid


def model_state_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize_model(seed):
    torch.manual_seed(seed)
    model=AdaptiveHankelEstimator(ModelConfig(core='em',structure='adaptive',stages=CONFIG['stages']))
    assert sum(p.numel() for p in model.parameters())==188
    return model


def loss(model,a,b):
    g=model(*a.args())
    return nll(b.z,g@b.s+b.b,b.sigma2).mean()


@torch.no_grad()
def validate(model,context,heldout,conditions):
    model.eval();scores=[]
    for start in range(0,len(context),CONFIG['batch_size']):
        ids=slice(start,start+CONFIG['batch_size']);a=context.batch(ids);b=heldout.batch(ids)
        g=model(*a.args())
        scores.append(nll(b.z,g@b.s+b.b,b.sigma2).mean((-2,-1)))
    values=torch.cat(scores).double();ids=torch.as_tensor(conditions,dtype=torch.long)
    count=torch.bincount(ids,minlength=126)
    if torch.any(count==0):raise ValueError('missing validation stratum')
    totals=torch.zeros(126,dtype=torch.float64).scatter_add_(0,ids,values)
    strata=totals/count
    if not torch.isfinite(strata).all():raise FloatingPointError('nonfinite validation NLL')
    return float(strata.mean()),strata.tolist()


def save_checkpoint(state,path):
    path=Path(path);temp=path.with_suffix('.tmp')
    torch.save(state,temp);os.replace(temp,path)


def preflight():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    grid=condition_grid();start=time.perf_counter();rows=[]
    for i in range(128):rows.append(simulate_observation(i,CONFIG['smoke_seed'],grid[i%len(grid)]))
    generation=time.perf_counter()-start
    obs=Observations(*(torch.as_tensor(np.asarray([row[j] for row in rows]),
        dtype=torch.complex64 if j in (1,2) else torch.float32) for j in range(4)))
    a21,a30,b=paired_split(obs,CONFIG['smoke_seed']);measurements={}
    # The production loop gathers only a minibatch. Prove identical values
    # to full-bank gathering before choosing that lower-memory execution.
    c21,c30,ch=split_indices(len(obs),CONFIG['smoke_seed'])
    batch_ids=torch.tensor([5,71,22,100,3])
    subset=obs.batch(batch_ids)
    for columns,full in ((c21,a21),(c30,a30),(ch,b)):
        small=subset.columns(columns[batch_ids]);reference=full.batch(batch_ids)
        assert all(torch.equal(x,y) for x,y in zip(small.args(),reference.args()))
    for arm,a in [('context21',a21),('context30',a30)]:
        model=initialize_model(9101);opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr']);times=[]
        for rep in range(7):
            ids=slice((rep%4)*32,(rep%4+1)*32);tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            l=loss(model,a.batch(ids),b.batch(ids));l.backward();opt.step()
            if rep:times.append(time.perf_counter()-tick)
        med=float(np.median(times));measurements[arm]=dict(batch_seconds=med,observed_batch_seconds=times,
            estimated_train_seconds_per_seed=med*(CONFIG['train_cases']//32)*CONFIG['epochs'])
    result=dict(at=now(),smoke_worlds=128,generation_seconds=generation,
        estimated_full_generation_seconds=generation*(CONFIG['train_cases']+CONFIG['val_cases'])/128,
        measurements=measurements,estimated_all_six_train_seconds=sum(x['estimated_train_seconds_per_seed'] for x in measurements.values())*3,
        estimated_float32_bank_bytes=(CONFIG['train_cases']+CONFIG['val_cases'])*(32*39*4+3*39*8+32*39*8+4),
        cpu_count=os.cpu_count(),threads=1,batch_gather_bitwise_equivalent=True,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        notes='Estimates exclude validation and process contention; each worker contains both arms for one fit seed. Production gathers training columns only per minibatch.')
    write_json(ROOT/'training_preflight.json',result);print(json.dumps(result,indent=2));return result


def run(seed):
    protocol=freeze();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if seed not in CONFIG['seeds']:raise ValueError('unregistered training seed')
    train,_=load_set('train');val,conditions=load_set('val')
    v21,v30,vhold=paired_split(val,CONFIG['validation_mask_seed'])
    data_hashes={split:sha(ROOT/'training_data'/split/'manifest.json') for split in ('train','val')}
    states={};worker=ROOT/'training_workers'/f'seed_{seed}';worker.mkdir(parents=True,exist_ok=True)
    for arm in CONFIG['contexts']:
        out=ROOT/'checkpoints'/arm/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
        model=initialize_model(seed);initial_digest=model_state_digest(model)
        opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr']);latest=out/'em_adaptive_latest.pt'
        if latest.exists():
            state=torch.load(latest,map_location='cpu',weights_only=False)
            if state['protocol_sha256']!=sha(ROOT/'training_protocol.json') or state['data_manifest_hashes']!=data_hashes:
                raise RuntimeError('checkpoint protocol/data mismatch')
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        else:
            val30,strata30=validate(model,v30,vhold,conditions);val21,strata21=validate(model,v21,vhold,conditions)
            state=dict(model_config=asdict(model.config),epoch=0,best_score=val30,history=[],seed=seed,arm=arm,
                initial_state_sha256=initial_digest,model=model.state_dict(),optimizer=opt.state_dict(),
                train_seconds=0.,optimizer_updates=0,protocol_sha256=sha(ROOT/'training_protocol.json'),
                data_manifest_hashes=data_hashes,selection_context=30,val30=val30,val21=val21,
                validation_strata30=strata30,validation_strata21=strata21,
                validation_mask_evidence=mask_evidence(len(val),CONFIG['validation_mask_seed']))
            save_checkpoint(state,out/'em_adaptive_best.pt');save_checkpoint(state,latest)
        write_json(out/'manifest.json',dict(seed=seed,arm=arm,initial_state_sha256=initial_digest,
            protocol_sha256=sha(ROOT/'training_protocol.json'),data_manifest_hashes=data_hashes,parameters=188))
        states[arm]=(model,opt,state,out)
    if len({s[2]['initial_state_sha256'] for s in states.values()})!=1:raise RuntimeError('unpaired initialization')
    try:
        for epoch in range(CONFIG['epochs']):
            if all(s[2]['epoch']>epoch for s in states.values()):continue
            mask_seed=seed+10000+epoch;c21,c30,held_columns=split_indices(len(train),mask_seed)
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+20000+epoch))
            evidence=dict(mask=mask_evidence(len(train),mask_seed),order_sha256=tensor_digest(order),epoch=epoch+1)
            for arm,context_columns in [('context21',c21),('context30',c30)]:
                model,opt,state,out=states[arm]
                if state['epoch']>epoch:continue
                if state['epoch']!=epoch:raise RuntimeError('missing preceding epoch')
                tick=time.perf_counter();total=0.;model.train()
                write_json(worker/'status.json',dict(state='running',seed=seed,arm=arm,epoch=epoch+1,at=now(),
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                for start in range(0,len(train),CONFIG['batch_size']):
                    ids=order[start:start+CONFIG['batch_size']];opt.zero_grad(set_to_none=True)
                    batch=train.batch(ids)
                    a=batch.columns(context_columns[ids]);held=batch.columns(held_columns[ids])
                    l=loss(model,a,held)
                    if not torch.isfinite(l):raise FloatingPointError('nonfinite training loss')
                    l.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError('nonfinite training gradient')
                    opt.step();total+=float(l.detach())*len(ids)
                seconds=time.perf_counter()-tick
                val30,strata30=validate(model,v30,vhold,conditions);val21,strata21=validate(model,v21,vhold,conditions)
                improved=val30<state['best_score']
                state.update(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),
                    val30=val30,val21=val21,validation_strata30=strata30,validation_strata21=strata21)
                state['train_seconds']+=seconds;state['optimizer_updates']+=len(train)//CONFIG['batch_size']
                state['history'].append(dict(epoch=epoch+1,train_nll=total/len(train),val30_nll=val30,
                    val21_nll=val21,seconds=seconds,evidence=evidence))
                if improved:
                    state['best_score']=val30;save_checkpoint(state,out/'em_adaptive_best.pt')
                save_checkpoint(state,out/'em_adaptive_latest.pt')
                write_json(out/'em_adaptive_history.json',state['history'])
                print(json.dumps(dict(seed=seed,arm=arm,epoch=epoch+1,val30=val30,val21=val21,best=state['best_score'],seconds=seconds)),flush=True)
        freeze()
        outputs={arm:dict(best_sha256=sha(out/'em_adaptive_best.pt'),latest_sha256=sha(out/'em_adaptive_latest.pt'),
                         epochs=state['epoch'],optimizer_updates=state['optimizer_updates'])
                 for arm,(model,opt,state,out) in states.items()}
        write_json(worker/'status.json',dict(state='complete',seed=seed,at=now(),outputs=outputs))
    except Exception as e:
        write_json(worker/'status.json',dict(state='failed',seed=seed,at=now(),error=repr(e)));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('preflight','generate','train'),required=True)
    p.add_argument('--seed',type=int);args=p.parse_args()
    if args.phase=='preflight':preflight()
    elif args.phase=='generate':
        freeze();torch.set_num_threads(1)
        for split in ('train','val'):print(json.dumps(generate_set(split)),flush=True)
    else:run(args.seed)


if __name__=='__main__':main()
