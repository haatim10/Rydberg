"""Observation-only training data; truth files are evaluation-only.

Generation reuses Paper 1's physical model and calibration. Independent
master seeds separate splits without generating/discarding earlier worlds.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from rydberg_sim.track_b_drivers import draw_L_k, track_b_spec
from rydberg_sim.monte_carlo import generate_channel_estimation_trial


@dataclass(frozen=True)
class Observations:
    z: torch.Tensor
    s: torch.Tensor
    b: torch.Tensor
    sigma2: torch.Tensor

    def __len__(self): return len(self.z)

    def batch(self, ids):
        return Observations(*(getattr(self,k)[ids] for k in ('z','s','b','sigma2')))

    def args(self): return self.z,self.s,self.b,self.sigma2

    def columns(self, indices):
        def take(x): return x.gather(-1,indices[:,None].expand(-1,x.shape[1],-1))
        return Observations(take(self.z),take(self.s),take(self.b),self.sigma2)


def split_observations(data, seed, fraction=.7):
    gen = torch.Generator().manual_seed(seed)
    order = torch.rand(len(data),data.z.shape[-1],generator=gen).argsort(-1)
    n = round(fraction*data.z.shape[-1])
    if n < 2*data.s.shape[1] or n >= data.z.shape[-1]:
        raise ValueError('insufficient context or held-out pilots')
    return data.columns(order[:,:n]),data.columns(order[:,n:])


def generate(directory, split, count, *, seed, N=32,K=3,P=30,snr=5.,rsr=12.,L=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{split}_observations.npz'
    spec_dict=dict(seed=seed,N=N,K=K,P=P,snr=snr,rsr=rsr,L=L,count=count)
    manifest=directory/f'{split}_generation.json'
    if path.exists():
        if json.loads(manifest.read_text()) != spec_dict: raise ValueError('data configuration mismatch')
        return
    arrays={k:[] for k in ('z','s','b','sigma2','g','paths','rho','rsr_total_realized_db')}
    from codex_verification.solver import effective_ranks
    for i in range(count):
        paths=draw_L_k(i,K,master_seed=seed) if L is None else (L,)*K
        spec=track_b_spec(P=P,n_trials=count,N=N,K=K,L=paths,master_seed=seed,experiment='sahure')
        w=generate_channel_estimation_trial(spec,i,snr,rsr)
        for key,value in zip(('z','s','b','sigma2','g'),(w.Z,w.S,w.B,w.sigma2,w.G)):
            arrays[key].append(value)
        arrays['paths'].append(paths)
        arrays['rho'].append(effective_ranks(w.G)/((N+1)//2))
        arrays['rsr_total_realized_db'].append(10*np.log10(np.mean(abs(w.B)**2)/np.mean(abs(w.G@w.S)**2)))
    np.savez_compressed(path,**{k:np.asarray(arrays[k]) for k in ('z','s','b','sigma2')})
    np.savez_compressed(directory/f'{split}_truth.npz',**{k:np.asarray(arrays[k]) for k in ('g','paths','rho','rsr_total_realized_db')})
    manifest.write_text(json.dumps(spec_dict,indent=2)+'\n')


def load_observations(path, dtype=torch.float32):
    with np.load(path) as a:
        cdtype=torch.complex128 if dtype==torch.float64 else torch.complex64
        return Observations(*(torch.as_tensor(a[k].copy(),dtype=cdtype if k in ('s','b') else dtype)
                              for k in ('z','s','b','sigma2')))
