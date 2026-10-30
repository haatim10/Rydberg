"""Frozen paired context-count ablation; no edits to prior experiments."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone

ROOT = Path('results/tcom_pilot_em_20261018')
CONFIG = dict(version=1, N=32, K=3, generated_P=39, heldout_P=9,
    contexts={'context21':21,'context30':30}, deployment_P=30, stages=10,
    train_cases=16384,val_cases=2048,epochs=20,batch_size=32,lr=.001,
    seeds=[9101,9102,9103],snrs=[-10,-5,0,5,10,15,20],rsrs=[5,12,20],
    fixed_paths=[2,4,8,12,16],train_data_seed=2026101801,val_data_seed=2026101802,
    validation_mask_seed=2026101810,smoke_seed=2026101800,
    training_mixture='half random per-user 3..7; half evenly allocated fixed 2,4,8,12,16',
    training_weighting='equal weight per case, preserving previous training mixture',
    validation_weighting='equal means across 126 SNR/RSR/path-regime strata',
    checkpoint_selection='minimum held-out exact Rician NLL with 30 fitting and 9 held-out pilots for BOTH arms; 21-context score diagnostic only',
    paired_protocol='same 39 observations, same 9 held-out columns, context21 nested in context30, same initialization and epoch order per training seed',
    interpretation='fresh paired controlled retraining; context21 is not an exact reproduction of the old checkpoint-selection procedure, which validated at context21',
    training_precision='float32/complex64',evaluation_precision='float64/complex128',
    training_only_extra_observations=True,channel_truth_used_for_training=False,
    optimizer='Adam defaults except lr=.001',threads_per_worker=1)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');temp=f.name
    os.replace(temp,path)


def source_hashes():
    files=[Path(__file__),Path(__file__).with_name('training_data.py'),Path(__file__).with_name('training.py')]
    files += [Path('sahure')/name for name in ('model.py','math.py','data.py')]
    files += sorted(Path('rydberg_sim').glob('*.py'))
    return {str(p):sha(p) for p in files}


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'training_protocol.json';sources=source_hashes()
    if path.exists():
        old=json.loads(path.read_text())
        if old['config']!=CONFIG or old['source_hashes']!=sources:
            raise RuntimeError('Frozen training protocol or source differs; do not overwrite this study.')
        return old
    result=dict(config=CONFIG,source_hashes=sources,frozen_at=now())
    write_json(path,result);return result
