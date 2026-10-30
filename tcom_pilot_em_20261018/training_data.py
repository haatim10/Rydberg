"""Paired observation-only training banks. Channel truth is not persisted."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from sahure.data import Observations
from rydberg_sim.track_b_drivers import draw_L_k,track_b_spec
from rydberg_sim.monte_carlo import generate_channel_estimation_trial
from .training_protocol import ROOT,CONFIG,sha,write_json,now


def condition_grid():
    return [dict(snr=float(s),rsr=float(r),L=L) for s in CONFIG['snrs']
            for r in CONFIG['rsrs'] for L in [None]+CONFIG['fixed_paths']]


def allocation(count,seed):
    grid=condition_grid()
    random_ids=[i for i,c in enumerate(grid) if c['L'] is None]
    fixed_ids=[i for i,c in enumerate(grid) if c['L'] is not None]
    ids=np.concatenate([np.resize(random_ids,count//2),np.resize(fixed_ids,count-count//2)])
    return np.random.default_rng(seed).permutation(ids)


def simulate_observation(i,seed,condition):
    """The simulator needs G; only observable arrays leave this function."""
    L=condition['L'];K=CONFIG['K']
    paths=draw_L_k(i,K,master_seed=seed) if L is None else (L,)*K
    spec=track_b_spec(P=CONFIG['generated_P'],n_trials=i+1,N=CONFIG['N'],K=K,
                      L=paths,master_seed=seed,experiment='tcom_pilot_em_20261018')
    w=generate_channel_estimation_trial(spec,i,condition['snr'],condition['rsr'])
    return w.Z,w.S,w.B,w.sigma2


def generate_set(split):
    if split not in ('train','val'): raise ValueError('split must be train or val')
    count=CONFIG[f'{split}_cases'];seed=CONFIG[f'{split}_data_seed']
    out=ROOT/'training_data'/split;out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json';description=dict(split=split,count=count,seed=seed,P=CONFIG['generated_P'])
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['description']!=description: raise RuntimeError('training data configuration changed')
        for name,digest in old['file_hashes'].items():
            if sha(out/name)!=digest: raise RuntimeError(f'changed training observation data: {name}')
        return old
    N,K,P=CONFIG['N'],CONFIG['K'],CONFIG['generated_P']
    specs={'z':((count,N,P),'float32'),'s':((count,K,P),'complex64'),
           'b':((count,N,P),'complex64'),'sigma2':((count,),'float32')}
    arrays={name:np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype=dtype,shape=shape)
            for name,(shape,dtype) in specs.items()}
    ids=allocation(count,seed);grid=condition_grid();start=time.perf_counter()
    for i,cid in enumerate(ids):
        values=simulate_observation(i,seed,grid[int(cid)])
        for name,value in zip(arrays,values):arrays[name][i]=value
        if (i+1)%1024==0:
            write_json(out/'progress.json',dict(completed=i+1,count=count,seconds=time.perf_counter()-start))
    for x in arrays.values():x.flush()
    np.save(out/'conditions.npy',ids)
    hashes={p.name:sha(p) for p in sorted(out.glob('*.npy'))}
    result=dict(description=description,condition_grid=grid,
        condition_counts=np.bincount(ids,minlength=len(grid)).tolist(),file_hashes=hashes,
        generated_at=now(),generation_seconds=time.perf_counter()-start,
        contents='observations plus design stratum IDs; no G, actual paths, effective rank, or truth files')
    write_json(manifest,result);return result


def load_set(split):
    out=ROOT/'training_data'/split
    # Explicit allow-list: the estimator never loads hidden channel labels.
    values=[torch.from_numpy(np.load(out/f'{key}.npy',allow_pickle=False))
            for key in ('z','s','b','sigma2')]
    return Observations(*values),np.load(out/'conditions.npy',allow_pickle=False)


def split_indices(count,seed):
    order=torch.rand(count,CONFIG['generated_P'],generator=torch.Generator().manual_seed(seed)).argsort(-1)
    return order[:,:21],order[:,:30],order[:,30:]


def paired_split(obs,seed):
    if obs.z.shape[-1]!=CONFIG['generated_P']:raise ValueError('expected paired39-pilot observations')
    ids=split_indices(len(obs),seed)
    return tuple(obs.columns(x) for x in ids)


def tensor_digest(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def mask_evidence(count,seed):
    a,b,h=split_indices(count,seed)
    assert torch.equal(a,b[:,:21])
    assert torch.equal(torch.cat((b,h),-1).sort(-1).values,torch.arange(39).expand(count,-1))
    return dict(seed=seed,context21_sha256=tensor_digest(a),context30_sha256=tensor_digest(b),
                heldout9_sha256=tensor_digest(h),nested=True,heldout_disjoint=True)
