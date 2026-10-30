"""One supervision mode per process; self-supervised workers never load labels."""
import argparse,time,hashlib
import numpy as np
import torch
from sahure.math import nll
from .common import *
from .models import build,ratio


def parameter_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def validate(m,val):
    m.eval();total=0.
    for start in range(0,len(val['z']),BATCH):
        sl=slice(start,start+BATCH);pred=m(*contexts(val,sl));z,s,b,v=heldout(val,sl)
        score=nll(z,pred@s+b,v).mean((-2,-1))
        assert torch.isfinite(score).all();total+=float(score.double().sum())
    return total/len(val['z'])


def train(mode,seed):
    frozen(check_inputs=False);manifest=json.loads((OUT/'data/manifest.json').read_text())
    # Validate only the files this worker is permitted to read. The exporter
    # and final audit validate the complete manifest separately.
    train,inputs=load_observations('train');val,more=load_observations('val');inputs+=more
    target=None
    if mode=='supervised':target,path=load_labels('train',mode);inputs.append(path)
    assert (target is not None)==(mode=='supervised')
    for p in inputs:
        name=str(Path(p).relative_to(OUT));assert sha(p)==manifest['files'][name]
    states={}
    for arch in ARCHES:
        folder=OUT/'training'/f'{arch}_{mode}'/str(seed);model=build(arch,seed)
        initsha=parameter_sha(model);opt=torch.optim.Adam(model.parameters(),lr=LR)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
        latest=folder/'latest.pt'
        if latest.exists():
            state=torch.load(latest,map_location='cpu',weights_only=False)
            assert state['protocol_sha256']==sha(OUT/'protocol.json') and state['initial_parameter_sha256']==initsha
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);schedule.load_state_dict(state['scheduler'])
        else:
            score=validate(model,val)
            state=dict(arch=arch,mode=mode,seed=seed,epoch=0,best_score=score,best_epoch=0,
                history=[],parameters=sum(p.numel() for p in model.parameters()),
                initial_parameter_sha256=initsha,protocol_sha256=sha(OUT/'protocol.json'),
                loaded_inputs=inputs,true_channel_labels_loaded=mode=='supervised',
                model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict())
            checkpoint(folder/'best.pt',state);checkpoint(latest,state)
        states[arch]=(model,opt,schedule,state,folder)
    for epoch in range(EPOCHS):
        order=torch.randperm(len(train['z']),generator=torch.Generator().manual_seed(seed+20000+epoch))
        active=[a for a in ARCHES if states[a][3]['epoch']<=epoch]
        if not active:continue
        tick=time.monotonic();totals={a:0. for a in active};grads={a:0. for a in active};seconds={a:0. for a in active}
        for a in active:states[a][0].train()
        for start in range(0,len(order),BATCH):
            ids=order[start:start+BATCH];args=contexts(train,ids);hz,hs,hb,hv=heldout(train,ids)
            for arch in active:
                m,opt,_,_,_=states[arch];at=time.monotonic();opt.zero_grad(set_to_none=True)
                pred=m(*args)
                loss=ratio(pred,target[ids]).mean() if mode=='supervised' else nll(hz,pred@hs+hb,hv).mean()
                assert torch.isfinite(loss),(arch,mode,epoch,start)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
                opt.step();totals[arch]+=float(loss.detach())*len(ids);grads[arch]=max(grads[arch],float(norm));seconds[arch]+=time.monotonic()-at
            if start%4096==0:
                for arch in active:write(states[arch][4]/'status.json',dict(state='training',mode=mode,arch=arch,seed=seed,epoch=epoch+1,completed=start,total=len(order)))
        for arch in active:
            m,opt,schedule,state,folder=states[arch];schedule.step();vt=time.monotonic();score=validate(m,val)
            entry=dict(epoch=epoch+1,training_loss=totals[arch]/len(order),validation_heldout_nll=score,
                training_seconds=seconds[arch],validation_seconds=time.monotonic()-vt,max_preclip_gradient=grads[arch],
                learning_rate=schedule.get_last_lr()[0])
            better=score<state['best_score'];state['history'].append(entry)
            state.update(epoch=epoch+1,best_score=min(score,state['best_score']),
                best_epoch=epoch+1 if better else state['best_epoch'],model=m.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict())
            checkpoint(folder/'latest.pt',state)
            if better:checkpoint(folder/'best.pt',state)
            write(folder/'history.json',dict(arch=arch,mode=mode,seed=seed,history=state['history'],best_epoch=state['best_epoch']))
            print(arch,mode,seed,'epoch',epoch+1,'valNLL',score,'trainseconds',seconds[arch],flush=True)
    for arch,(m,opt,schedule,state,folder) in states.items():
        assert state['epoch']==EPOCHS;checkpoint(folder/'final.pt',state)
        write(folder/'status.json',dict(state='complete',arch=arch,mode=mode,seed=seed,epoch=EPOCHS))
    frozen(check_inputs=False)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=MODES);p.add_argument('seed',type=int);a=p.parse_args();train(a.mode,a.seed)
