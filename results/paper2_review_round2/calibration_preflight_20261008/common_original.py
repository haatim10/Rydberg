"""Atomic experiment records and read-only access to the original models."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from sahure_expansion.protocol import write_json
from sahure_expansion.evaluate import estimator_sources
from sahure.train import load_model

ROOT=Path('results/paper2_review_round2')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as f:
        np.savez_compressed(f,**arrays);name=f.name
    os.replace(name,path)


def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def freeze_phase(name,config,files):
    old=json.loads(Path('results/sahure_expansion/frozen_selection.json').read_text())
    if old['sources']!=estimator_sources():raise RuntimeError('old estimator source changed')
    selected={str(s):old['models'][f'em_adaptive_s{s}'] for s in [9101,9102,9103]}
    for row in selected.values():
        if sha(row['path'])!=row['sha256']:raise RuntimeError('old checkpoint changed')
    sources={**estimator_sources(),**{str(p):sha(p) for p in [Path(__file__)]+[Path(f) for f in files]}}
    description=dict(config=config,sources=sources,models=selected)
    directory=ROOT/name;directory.mkdir(parents=True,exist_ok=True)
    p=directory/'protocol.json'
    if p.exists():
        previous=json.loads(p.read_text())
        for key,val in description.items():
            if previous[key]!=val:raise RuntimeError(f'Frozen {name} {key} mismatch')
        return directory,previous
    description['frozen_unix']=time.time();write_json(p,description)
    return directory,description


def verify_sources(protocol):
    for p,digest in protocol['sources'].items():
        if sha(p)!=digest:raise RuntimeError(f'source changed: {p}')
    for row in protocol['models'].values():
        if sha(row['path'])!=row['sha256']:raise RuntimeError('checkpoint changed')


def frozen_models(protocol):return {seed:load_model(row['path'])[0] for seed,row in protocol['models'].items()}


def progress(directory,stage,done,total,start):
    state=dict(state='running',stage=stage,completed=done,total=total,
               seconds=time.monotonic()-start,updated_unix=time.time(),pid=os.getpid())
    write_json(directory/'status.json',state)
    if done%16==0:print(json.dumps(state),flush=True)


def aggregate(errors,energy):
    errors=np.atleast_2d(errors)
    return dict(nmse_db=float(np.mean(10*np.log10(errors.sum(-1)/np.sum(energy)))),
        seed_nmse_db=(10*np.log10(errors.sum(-1)/np.sum(energy))).tolist(),
        mean_case_nmse=float(np.mean(errors/energy)),severe_fraction=float(np.mean(errors/energy>1)))
