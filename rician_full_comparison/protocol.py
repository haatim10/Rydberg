from pathlib import Path
import hashlib
import json
import os
import platform
import tempfile
import time
import numpy as np
import torch
from fair_optshrink.run import SPECS

ROOT=Path('results/rician_full_comparison')
DATA=Path('results/sahure_expansion/data')
CONFIG=dict(version=1,seeds=[9101,9102,9103],epochs=20,batch_size=32,lr=.001,
    stages=10,alpha=1.,train_cases=16384,val_cases=2048,
    validation_mask_seed=2026092203,banks=[20261014211001,20261014221001],cases=32,
    timing_seed=20261014231001,timing_repeats=3,bootstrap_seed=20261014241001,
    bootstrap_samples=4000,primary_cells=126,threads=1,workers=3,max_seconds=14400,
    specs=SPECS,selection='equal126-stratum fixed21/9 held-out Rician NLL, epoch0 eligible',
    primary='validation-selected allpilot versus validation-selected holdout and rank-loop OptShrink',
    secondary='final20epoch allpilot versus final20epoch holdout; offline-tuned OptShrink',
    metric='equal126-cell mean dB ratio-of-summed-errors to summed-channel-power, then equal model-seed mean',
    dtype_train='float32/complex64',dtype_test='float64/complex128')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');temp=f.name
    os.replace(temp,path)

def save(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
        np.savez_compressed(f,**values);temp=f.name
    os.replace(temp,path)

def load(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}

def initialize():
    paths=sorted(Path('rician_full_comparison').glob('*.py'))+[Path('rician_full_comparison/PROTOCOL.txt')]
    paths+=[Path(p) for p in ('rician_pilot_screen/noise.py','rician_pilot_screen/test_noise.py',
        'sahure/model.py','sahure/math.py','sahure/train.py','sahure/data.py',
        'sahure_expansion/data.py','sahure_expansion/protocol.py','sahure_expansion/train.py',
        'fair_optshrink/estimator.py','fair_optshrink/run.py','hybrid_optshrink/model.py',
        'compact_baseline/analyze.py','paper2_optshrink_loop/estimator.py','sahure/classical.py',
        'optshrink_rank_check/estimator.py','results/fair_optshrink/frozen_selection.json',
        'results/compact_baseline/frozen_selection.json')]
    paths.append(ROOT/'baseline_replay.json')
    paths+=sorted(Path('rydberg_sim').rglob('*.py'))
    paths+=[DATA/p for p in ('train_observations.npz','val_observations.npz','val_conditions.npy',
        'train_generation.json','val_generation.json')]
    parent=json.loads(Path('results/compact_baseline/frozen_selection.json').read_text())
    models={}
    for seed in CONFIG['seeds']:
        for label,suffix in [('best','best'),('final','latest')]:
            p=Path(f'results/sahure_expansion/checkpoints/seed_{seed}/em_adaptive_{suffix}.pt')
            digest=sha(p)
            if label=='best' and digest!=parent['models'][f'adaptive_{seed}']['sha256']:
                raise RuntimeError('historical selected model changed')
            models[f'holdout_{label}_{seed}']=dict(path=str(p),sha256=digest)
            paths.append(p)
        paths.append(Path(f'results/sahure_expansion/checkpoints/seed_{seed}/manifest.json'))
    record=dict(config=CONFIG,sources={str(p):sha(p) for p in paths},historical_models=models,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,system=platform.platform()))
    target=ROOT/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=record:raise RuntimeError('frozen sources/data/configuration changed')
    if not target.exists():write(target,record)
    return record

def freeze_test():
    record=initialize();models=dict(record['historical_models'])
    for seed in CONFIG['seeds']:
        for label,suffix in [('best','best'),('final','latest')]:
            p=ROOT/'checkpoints'/f'seed_{seed}'/f'allpilot_{suffix}.pt'
            models[f'allpilot_{label}_{seed}']=dict(path=str(p),sha256=sha(p))
    value=dict(manifest_sha256=sha(ROOT/'manifest.json'),models=models)
    path=ROOT/'frozen_selection.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix')
        if old!=value:raise RuntimeError('selected/final models changed after test freeze')
    else:
        if (ROOT/'data').exists():raise RuntimeError('test data generated before selection freeze')
        write(path,dict(**value,frozen_unix=time.time()))
    return models
