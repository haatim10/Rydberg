"""Paired global30 control training using the immutable matched-pilot banks."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import scipy
import torch
from sahure.data import Observations
from sahure.math import nll
from sahure.model import AdaptiveHankelEstimator, ModelConfig
from sahure.train import load_model
from tcom_pilot_em_20261018.training_protocol import sha, write_json, now
from tcom_pilot_em_20261018.training_data import split_indices, mask_evidence, tensor_digest, load_set
from tcom_pilot_em_20261018.training import loss, validate, save_checkpoint, model_state_digest
from .starts import OUT, REPO, SEEDS, NativeGlobalEstimator, classical_estimator, adaptive_path, global_path

OLD = REPO / 'results/tcom_pilot_em_20261018'
CONFIG = dict(version=1, model='global', parameters=30, core='em', stages=10,
    train_cases=16384, val_cases=2048, generated_P=39, context_P=30, heldout_P=9,
    deployment_P=30, epochs=20, batch_size=32, lr=.001, seeds=list(SEEDS),
    train_data_seed=2026101801, val_data_seed=2026101802, validation_mask_seed=2026101810,
    train_mask_seed='fit_seed+10000+zero_based_epoch',
    train_order_seed='fit_seed+20000+zero_based_epoch',
    initialization='same initial output as adaptive: gamma=1, relative_tau=.1, beta=.1; global constants have no random hidden layer',
    optimizer='Adam defaults except lr=.001',
    training_precision='float32/complex64', evaluation_precision='float64/complex128',
    checkpoint_selection='minimum exact heldout Rician NLL at context30/holdout9; initial epoch0 eligible; strict improvement only',
    training_weighting='equal per case, identical existing mixture',
    validation_weighting='equal mean across the identical 126 design strata',
    pairing='same immutable banks, masks, minibatch order, epochs, optimizer, initial output and validation as frozen adaptive context30',
    channel_truth_used=False, threads_per_worker=1,
    interpretation='offline learned per-stage constants; 30 trainable parameters, without observation-conditioned controller')


def initialize_model(seed):
    torch.manual_seed(seed)
    model = AdaptiveHankelEstimator(ModelConfig(core='em', structure='global', stages=10))
    assert sum(p.numel() for p in model.parameters()) == 30
    return model


def sources():
    paths = [
        Path(__file__), Path(__file__).with_name('starts.py'),
        REPO/'tcom_runtime_repair_20261020/native.py',
        REPO/'tcom_runtime_bypass_20261017/sparse_fast.py',
        REPO/'tcom_pilot_em_20261018/continuation.py',
        *[REPO/'tcom_pilot_em_20261018'/f for f in ('training.py','training_data.py','training_protocol.py')],
        *[REPO/'sahure'/f for f in ('model.py','math.py','data.py','train.py')],
    ]
    return {str(p.relative_to(REPO)): sha(p) for p in paths}


def inputs(verify=True):
    old_protocol = json.loads((OLD/'training_protocol.json').read_text())
    if verify:
        for name, digest in old_protocol['source_hashes'].items():
            if sha(REPO/name) != digest:
                raise RuntimeError(f'changed previous training source: {name}')
    paths = [OLD/'training_protocol.json']
    for split in ('train', 'val'):
        folder = OLD/'training_data'/split
        manifest = json.loads((folder/'manifest.json').read_text())
        paths.append(folder/'manifest.json')
        if set(manifest['file_hashes']) != {'z.npy','s.npy','b.npy','sigma2.npy','conditions.npy'}:
            raise RuntimeError('unexpected bank contents')
        for name, digest in manifest['file_hashes'].items():
            path = folder/name
            if verify and sha(path) != digest:
                raise RuntimeError(f'changed immutable observation bank: {path}')
            paths.append(path)
    for seed in SEEDS:
        paths.extend((adaptive_path(seed),
            adaptive_path(seed).with_name('em_adaptive_history.json')))
    return {str(p.relative_to(REPO)): sha(p) for p in paths}


def freeze():
    path = OUT/'training_protocol.json'
    record = dict(config=CONFIG, source_hashes=sources(), input_hashes=inputs(),
        environment=dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                         threads=1, training_device='cpu'))
    if path.exists():
        old = json.loads(path.read_text())
        for key in record:
            if old[key] != record[key]:
                raise RuntimeError(f'changed frozen training {key}')
        return old
    preflight = OUT/'training_preflight.json'
    checks = json.loads(preflight.read_text())
    if not checks['passed'] or checks['source_hashes'] != sources():
        raise RuntimeError('current sources require a passing preflight')
    record.update(frozen_at=now(), frozen_unix=time.time(), preflight_sha256=sha(preflight))
    write_json(path, record)
    return record


def check_pairing(seed, epoch, evidence):
    path = adaptive_path(seed).with_name('em_adaptive_history.json')
    reference = json.loads(path.read_text())[epoch]['evidence']
    if evidence != reference:
        raise RuntimeError('training masks or minibatch order differ from adaptive comparator')


@torch.no_grad()
def torch_classical(obs, config):
    """Independent canonical reference for preflight only."""
    from sahure.math import posterior_field, right_solve, spectral_initial, hankel, smooth_shrink, unhankel
    z,s,b,v = obs.args()
    gram = s@s.mH
    g = spectral_initial(z,s,b)
    traces=[]
    if config['kind']=='noise':
        inverse=torch.linalg.solve(gram,torch.eye(s.shape[1],dtype=s.dtype).expand_as(gram))
        cols=(z.shape[1]-1)//2+1;rows=z.shape[1]-cols+1
        tau=config['c']*torch.sqrt(2*v[:,None]*inverse.diagonal(dim1=-2,dim2=-1).real*max(rows,cols))
    for _ in range(10):
        dc=right_solve((posterior_field(z,g@s+b,v)-b)@s.mH,gram)
        h=hankel(dc.mT)
        if config['kind']=='noise':
            scale=(h.abs().square().sum((-2,-1))/min(h.shape[-2:])).clamp_min(1e-24).sqrt()
            relative=tau/scale
        else:relative=torch.full(dc.mT.shape[:2],config['relative_tau'],dtype=z.dtype)
        hs,_=smooth_shrink(h,relative)
        structured=unhankel(hs).mT
        g=dc+config['beta']*(structured-dc)
        traces.append(g.clone())
    return g,traces


def preflight():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    # Read only a small subset of existing observable training arrays.
    folder=OLD/'training_data/train'
    values=[torch.from_numpy(np.array(np.load(folder/f'{key}.npy',mmap_mode='r')[:128]))
            for key in ('z','s','b','sigma2')]
    obs=Observations(*values)
    _,context,held=split_indices(len(obs),9101+10000)
    a=obs.columns(context);b=obs.columns(held)
    results={}
    with torch.no_grad():
        global_model=initialize_model(9101)
        torch.manual_seed(9101)
        adaptive=AdaptiveHankelEstimator(ModelConfig(core='em',structure='adaptive',stages=10))
        x=global_model(*a.batch(slice(0,4)).args());y=adaptive(*a.batch(slice(0,4)).args())
        results['initial_global_adaptive_max_abs']=float((x-y).abs().max())
        assert torch.equal(x,y)
    # Same minibatch masks/order as every existing adaptive training epoch.
    for seed in SEEDS:
        for epoch in range(CONFIG['epochs']):
            order=torch.randperm(CONFIG['train_cases'],generator=torch.Generator().manual_seed(seed+20000+epoch))
            evidence=dict(mask=mask_evidence(CONFIG['train_cases'],seed+10000+epoch),
                          order_sha256=tensor_digest(order),epoch=epoch+1)
            check_pairing(seed,epoch,evidence)
    results['all_60_epoch_masks_and_orders_match_adaptive']=True
    model=initialize_model(9101);initial=model_state_digest(model)
    opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr']);times=[]
    for rep in range(7):
        ids=slice((rep%4)*32,(rep%4+1)*32)
        tick=time.perf_counter();opt.zero_grad(set_to_none=True)
        l=loss(model,a.batch(ids),b.batch(ids));l.backward()
        assert torch.isfinite(l) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
        if rep:times.append(time.perf_counter()-tick)
    assert initial!=model_state_digest(model)
    results['finite_training_gradients_and_changed_parameters']=True
    a64=Observations(*(v.to(torch.complex128 if v.is_complex() else torch.float64) for v in a.batch(slice(0,4)).args()))
    model=model.double().eval();native=NativeGlobalEstimator(model)
    with torch.no_grad():
        canonical,ct=model(*a64.args(),diagnostics=True)
        fast,ft=native.estimate(a64,diagnostics=True)
        errors=[float((x['g']-y['g']).abs().max()) for x,y in zip(ct,ft)]
        assert torch.allclose(canonical,fast,atol=1e-10,rtol=1e-10)
        results['native_global_max_stage_abs_error']=max(errors)
        results['native_global_scalar_batch_max_abs_error']=max(float((native.estimate(a64.batch(slice(i,i+1)))-fast[i:i+1]).abs().max()) for i in range(len(a64)))
        results['classical_checks']=[]
        for config in (dict(kind='relative',relative_tau=.01,beta=.1),
                       dict(kind='relative',relative_tau=1.,beta=1.),
                       dict(kind='noise',c=.5,beta=.1),dict(kind='noise',c=8.,beta=1.)):
            reference,stages=torch_classical(a64,config)
            value,trace=classical_estimator(config).estimate(a64,diagnostics=True)
            error=max(float((r-t['g']).abs().max()) for r,t in zip(stages,trace))
            assert torch.allclose(value,reference,atol=1e-10,rtol=1e-10)
            results['classical_checks'].append(dict(config=config,max_stage_abs_error=error))
        em=AdaptiveHankelEstimator(ModelConfig(core='em',structure='none',stages=10)).double()
        from .starts import FixedCore
        error=float((em(*a64.args())-FixedCore(np.ones(10)).estimate(a64)).abs().max())
        assert error<1e-10
        results['em10_max_abs_error']=error
    results.update(passed=True,at=now(),source_hashes=sources(),parameters=30,bank_subset_cases=128,
        batch_seconds=times,median_batch_seconds=float(np.median(times)),
        estimated_training_seconds_per_seed=float(np.median(times)*512*20),
        timing_note='Approximate training cost only; excludes validation and concurrent-worker contention.',
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(OUT/'training_preflight.json',results)
    print(json.dumps(results,indent=2),flush=True)
    return results


def run(seed):
    freeze();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if seed not in SEEDS:raise ValueError('unregistered seed')
    train,_=load_set('train');val,conditions=load_set('val')
    _,vc,vh=split_indices(len(val),CONFIG['validation_mask_seed'])
    v30=val.columns(vc);vhold=val.columns(vh)
    data_hashes={split:sha(OLD/'training_data'/split/'manifest.json') for split in ('train','val')}
    out=global_path(seed).parent;out.mkdir(parents=True,exist_ok=True)
    worker=OUT/'training_workers'/f'seed_{seed}';worker.mkdir(parents=True,exist_ok=True)
    model=initialize_model(seed);initial_digest=model_state_digest(model)
    opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'])
    latest=out/'em_global_latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['protocol_sha256']!=sha(OUT/'training_protocol.json') or state['data_manifest_hashes']!=data_hashes:
            raise RuntimeError('checkpoint protocol/data mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    else:
        score,strata=validate(model,v30,vhold,conditions)
        state=dict(model_config=asdict(model.config),epoch=0,best_score=score,history=[],seed=seed,arm='global',
            initial_state_sha256=initial_digest,model=model.state_dict(),optimizer=opt.state_dict(),
            train_seconds=0.,optimizer_updates=0,protocol_sha256=sha(OUT/'training_protocol.json'),
            data_manifest_hashes=data_hashes,selection_context=30,val30=score,validation_strata30=strata,
            initial_val30=score,validation_mask_evidence=mask_evidence(len(val),CONFIG['validation_mask_seed']))
        save_checkpoint(state,global_path(seed));save_checkpoint(state,latest)
    write_json(out/'manifest.json',dict(seed=seed,arm='global',parameters=30,initial_state_sha256=initial_digest,
        protocol_sha256=sha(OUT/'training_protocol.json'),data_manifest_hashes=data_hashes))
    try:
        for epoch in range(state['epoch'],CONFIG['epochs']):
            _,columns,held=split_indices(len(train),seed+10000+epoch)
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+20000+epoch))
            evidence=dict(mask=mask_evidence(len(train),seed+10000+epoch),
                          order_sha256=tensor_digest(order),epoch=epoch+1)
            check_pairing(seed,epoch,evidence)
            tick=time.perf_counter();total=0.;model.train()
            write_json(worker/'status.json',dict(state='running',seed=seed,epoch=epoch+1,at=now(),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            for start in range(0,len(train),CONFIG['batch_size']):
                ids=order[start:start+CONFIG['batch_size']];batch=train.batch(ids)
                opt.zero_grad(set_to_none=True)
                value=loss(model,batch.columns(columns[ids]),batch.columns(held[ids]))
                if not torch.isfinite(value):raise FloatingPointError('nonfinite training loss')
                value.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('nonfinite gradient')
                opt.step();total+=float(value.detach())*len(ids)
            seconds=time.perf_counter()-tick
            score,strata=validate(model,v30,vhold,conditions)
            improved=score<state['best_score']
            state.update(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),
                         val30=score,validation_strata30=strata)
            state['train_seconds']+=seconds;state['optimizer_updates']+=len(train)//CONFIG['batch_size']
            state['history'].append(dict(epoch=epoch+1,train_nll=total/len(train),val30_nll=score,
                                        seconds=seconds,evidence=evidence))
            if improved:
                state['best_score']=score;save_checkpoint(state,global_path(seed))
            save_checkpoint(state,latest)
            write_json(out/'em_global_history.json',state['history'])
            print(json.dumps(dict(seed=seed,epoch=epoch+1,val30=score,best=state['best_score'],seconds=seconds)),flush=True)
        freeze()
        write_json(worker/'status.json',dict(state='complete',seed=seed,at=now(),epochs=state['epoch'],
            optimizer_updates=state['optimizer_updates'],best_sha256=sha(global_path(seed)),latest_sha256=sha(latest)))
    except Exception as error:
        write_json(worker/'status.json',dict(state='failed',seed=seed,at=now(),error=repr(error)))
        raise


def summarize():
    freeze();records={}
    for seed in SEEDS:
        worker=json.loads((OUT/'training_workers'/f'seed_{seed}'/'status.json').read_text())
        if worker['state']!='complete':raise RuntimeError('training is incomplete')
        model,best=load_model(global_path(seed),dtype=torch.float64)
        latest=torch.load(global_path(seed).with_name('em_global_latest.pt'),map_location='cpu',weights_only=False)
        assert latest['epoch']==20 and latest['optimizer_updates']==10240
        scores=[latest['initial_val30']]+[x['val30_nll'] for x in latest['history']]
        assert best['best_score']==min(scores) and best['epoch']==int(np.argmin(scores))
        records[str(seed)]=dict(best_epoch=best['epoch'],best_val30=best['best_score'],
            parameters=sum(p.numel() for p in model.parameters()),best_sha256=sha(global_path(seed)),
            latest_sha256=worker['latest_sha256'],epochs=latest['epoch'],updates=latest['optimizer_updates'],
            training_seconds=latest['train_seconds'])
    result=dict(passed=True,at=now(),protocol_sha256=sha(OUT/'training_protocol.json'),fits=records)
    write_json(OUT/'training_summary.json',result);print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=('preflight','prepare','train','summary'))
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.phase=='preflight':preflight()
    elif args.phase=='prepare':print(json.dumps(freeze(),indent=2))
    elif args.phase=='summary':summarize()
    else:run(args.seed)


if __name__=='__main__':main()
