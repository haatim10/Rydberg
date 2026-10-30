from pathlib import Path
import hashlib
import json
import os
import tempfile

ROOT = Path('results/sahure_expansion')
CONFIG = dict(version=1,N=32,K=3,P=30,stages=10,train_cases=16384,val_cases=2048,
    test_cases=400,epochs=20,batch_size=32,lr=.001,seeds=[9101,9102,9103],
    snrs=[-10,-5,0,5,10,15,20],rsrs=[5,12,20],fixed_paths=[2,4,8,12,16],
    train_data_seed=2026092201,val_data_seed=2026092202,test_data_seed=2026092300,
    validation_mask_seed=2026092203,bootstrap_seed=2026092401,bootstrap_resamples=4000,
    validation_weighting='equal means across 126 SNR/RSR/path-regime strata',
    training_mixture='half random per-user 3..7; half evenly allocated fixed 2,4,8,12,16',
    training_seeds_share_data=True,timing_trials=100,timing_repetitions=3,
    precision_train='float32/complex64',precision_evaluate='float64/complex128')


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Several seed workers may initialize the same identical configuration.
    # Each writer needs its own temporary file before atomic replacement.
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
        temp_name=stream.name
    os.replace(temp_name,path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'config.json'
    if path.exists() and json.loads(path.read_text())!=CONFIG:
        raise RuntimeError('Do not change a running campaign configuration')
    write_json(path,CONFIG)


def cells():
    """Unique test points; duplicate reference points share the same worlds."""
    points={}; axes={}
    def add(axis,value,**changes):
        spec=dict(N=32,K=3,P=30,snr=5.,rsr=12.,L=None,channel='ula',noise_offset=0.)
        spec.update(changes)
        key=json.dumps(spec,sort_keys=True)
        if key not in points: points[key]=dict(id=f'cell_{len(points):02d}',**spec)
        axes.setdefault(axis,[]).append(dict(value=value,cell=points[key]['id']))
    for x in CONFIG['snrs']: add('snr',x,snr=float(x))
    for x in [0,5,10,12,15,20,25]: add('rsr',x,rsr=float(x))
    for x in [2,3,4,6,8,12,16]: add('paths',x,L=x)
    for x in [10,20,30]: add('pilots',x,P=x)
    for x in [8,16,32,64]: add('array',x,N=x)
    for x in [2,3,4]: add('users',x,K=x)
    for x in ['ula','clustered','ten_path','unequal']: add('channel',x,channel=x)
    for x in [-3,0,3]: add('noise_offset',x,noise_offset=float(x))
    return list(points.values()),axes
