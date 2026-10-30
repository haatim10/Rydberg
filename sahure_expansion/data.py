"""Generation writes truth separately. Training loads only observations and strata."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rydberg_sim.track_b_drivers import draw_L_k,track_b_spec
from rydberg_sim.monte_carlo import generate_channel_estimation_trial
from rydberg_sim.forward import exact_forward
from rydberg_sim.rng import get_operating_point_rngs
from codex_verification.solver import effective_ranks
from .protocol import ROOT,CONFIG,write_json


def condition_grid():
    return [dict(snr=float(s),rsr=float(r),L=L) for s in CONFIG['snrs']
            for r in CONFIG['rsrs'] for L in [None]+CONFIG['fixed_paths']]


def allocation(count,seed):
    grid=condition_grid()
    random_ids=[i for i,c in enumerate(grid) if c['L'] is None]
    fixed_ids=[i for i,c in enumerate(grid) if c['L'] is not None]
    # Deterministic round robin within each half, then shuffle case order.
    ids=np.concatenate([np.resize(random_ids,count//2),np.resize(fixed_ids,count-count//2)])
    return np.random.default_rng(seed).permutation(ids)


def world(i,seed,*,N=32,K=3,P=30,snr=5.,rsr=12.,L=None,channel='ula',**ignored):
    paths=draw_L_k(i,K,master_seed=seed) if L is None else (L,)*K
    spec=track_b_spec(P=P,n_trials=i+1,N=N,K=K,L=paths,master_seed=seed,experiment='sahure_expansion')
    w=generate_channel_estimation_trial(spec,i,snr,rsr)
    if channel=='ula': return w,paths
    rng=np.random.default_rng(np.random.SeedSequence([seed,i,0x5A17]))
    g=np.empty((N,K),complex)
    for k in range(K):
        if channel=='clustered':
            centers=rng.uniform(-np.pi/2,np.pi/2,4)
            theta=np.clip(centers[:,None]+np.deg2rad(rng.uniform(-5,5,(4,10))),-np.pi/2,np.pi/2).ravel()
            powers=np.ones(40)/40
        else:
            theta=rng.uniform(-np.pi/2,np.pi/2,10)
            powers=np.power(.5,np.arange(10)) if channel=='unequal' else np.ones(10)
            if channel not in ('unequal','ten_path'): raise ValueError(channel)
            powers=powers/powers.sum()
        gains=(rng.standard_normal(len(theta))+1j*rng.standard_normal(len(theta)))*np.sqrt(powers/2)
        g[:,k]=(np.exp(-1j*np.arange(N)[:,None]*np.pi*np.sin(theta))*gains).sum(1)
    # Match the base world's realized total energy, as in Paper 1 alternatives.
    g*=np.linalg.norm(w.G)/np.linalg.norm(g)
    rngs=get_operating_point_rngs(seed,i,snr,rsr)
    obs=exact_forward(g,w.S,w.B,w.sigma2,rng_noise=rngs.noise)
    return SimpleNamespace(G=g,S=w.S,B=w.B,Z=obs.Z,sigma2=w.sigma2),(len(theta),)*K


def generate_set(name,count,seed,condition_ids=None,spec=None):
    out=ROOT/'data';out.mkdir(parents=True,exist_ok=True)
    manifest=out/f'{name}_generation.json'
    description=dict(count=count,seed=seed,spec=spec,mixed=condition_ids is not None)
    if manifest.exists():
        if json.loads(manifest.read_text())['description']!=description: raise RuntimeError('data mismatch')
        return
    grid=condition_grid()
    N,K,P=(32,3,30) if spec is None else (spec['N'],spec['K'],spec['P'])
    obs=dict(z=np.empty((count,N,P)),s=np.empty((count,K,P),complex),
             b=np.empty((count,N,P),complex),sigma2=np.empty(count))
    truth=dict(g=np.empty((count,N,K),complex),rho=np.empty((count,K)),
               paths=np.empty((count,K),int),rsr_total_realized_db=np.empty(count))
    for i in range(count):
        settings=grid[int(condition_ids[i])] if condition_ids is not None else spec
        w,paths=world(i,seed,**settings)
        for key,val in zip(obs,(w.Z,w.S,w.B,w.sigma2)): obs[key][i]=val
        truth['g'][i]=w.G;truth['paths'][i]=paths
        truth['rho'][i]=effective_ranks(w.G)/((N+1)//2)
        truth['rsr_total_realized_db'][i]=10*np.log10(np.mean(abs(w.B)**2)/np.mean(abs(w.G@w.S)**2))
    np.savez_compressed(out/f'{name}_observations.npz',**obs)
    np.savez_compressed(out/f'{name}_truth.npz',**truth)
    if condition_ids is not None:
        np.save(out/f'{name}_conditions.npy',condition_ids)
    write_json(manifest,dict(description=description,condition_grid=grid if condition_ids is not None else None,
        condition_counts=np.bincount(condition_ids,minlength=len(grid)).tolist() if condition_ids is not None else None))


def prepare():
    for name,count,seed in [('train',CONFIG['train_cases'],CONFIG['train_data_seed']),
                            ('val',CONFIG['val_cases'],CONFIG['val_data_seed'])]:
        print(f'Generating {name}: {count} worlds',flush=True)
        generate_set(name,count,seed,allocation(count,seed))
