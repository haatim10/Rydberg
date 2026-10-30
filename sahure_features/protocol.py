from pathlib import Path
import json
import time
from sahure_expansion.protocol import sha, write_json

ROOT = Path('results/sahure_features')
OLD = Path('results/sahure_expansion/data')
CONFIG = dict(version=1, arms=['full','spectral','measurement','context'],
    seeds=[9101,9102,9103], train_cases=4096, val_cases=2048, epochs=12,
    batch_size=32, lr=.001, validation_mask_seed=2026092203,
    maximum_seconds=7200, workers=1, test_banks=[1,2], test_cases=128,
    test_seed_base=202609231100, bootstrap_seed=202609232100,
    bootstrap_resamples=4000,
    primary='equal-weight average of seven per-SNR ratio-of-sums NMSE dB scores; mean of seed scores',
    contrasts=['full vs spectral','full vs measurement','full vs context'],
    selection='lowest equal-stratum held-out Rician NLL, including epoch zero',
    purpose='small-budget feature-access diagnostic, not final publication confirmation',
    inputs='first 4096 old development training worlds; all 2048 old validation worlds; no old tests',
    inference_pilots=30, training_context_pilots=21,
    practical_gain_db=.2, meaningful_cell_harm_db=.25)

def jobs():
    return [dict(id=f'{arm}_{seed}', arm=arm, seed=seed) for seed in CONFIG['seeds'] for arm in CONFIG['arms']]

def cells():
    base = dict(N=32,K=3,P=30,snr=5.,rsr=12.,L=None,channel='ula')
    return [dict(base,id=f'snr_{s}',snr=float(s),primary=True) for s in [-10,-5,0,5,10,15,20]] + [
        dict(base,id='weak_reference',rsr=0.,primary=False),dict(base,id='two_paths',L=2,primary=False)]

def protected_records():
    old = json.loads(Path('results/sahure_checks/protocol.json').read_text())
    return {**old['sources'], **old['previous_sha256']}

def source_records():
    paths = list(Path('sahure_features').glob('*.py'))
    return {**protected_records(), **{str(p):sha(p) for p in sorted(paths)}}

def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    for p,h in protected_records().items():
        if sha(p)!=h: raise RuntimeError(f'Protected input changed: {p}')
    inputs = [OLD/'train_observations.npz', OLD/'val_observations.npz', OLD/'val_conditions.npy']
    record = dict(config=CONFIG,jobs=jobs(),cells=cells(),sources=source_records(),
                  inputs={str(p):sha(p) for p in inputs})
    path = ROOT/'protocol.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if any(saved[k]!=v for k,v in record.items()): raise RuntimeError('Frozen feature protocol changed')
        return saved
    record['declared_unix']=time.time()
    write_json(path,record)
    return record

def check_sources():
    saved=json.loads((ROOT/'protocol.json').read_text())
    for p,h in {**saved['sources'],**saved['inputs']}.items():
        if sha(p)!=h: raise RuntimeError(f'Frozen input changed: {p}')
    return saved
