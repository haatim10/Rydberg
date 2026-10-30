from pathlib import Path
import json
import time
from fair_optshrink.run import SPECS, sha, save, load, write

ROOT = Path('results/compact_baseline')
DATA = Path('results/sahure_expansion/data')
CONFIG = dict(version=1, seeds=[9101,9102,9103], stages=10, hidden_size=4,
    epochs=20, batch_size=32, lr=.001, train_cases=16384, val_cases=2048,
    validation_mask_seed=2026092203, banks=[20261012110001,20261012120001], cases=16,
    timing_seed=20261012130001, bootstrap_seed=20261012140001, bootstrap_samples=4000,
    threads=1, dtype_train='float32/complex64', dtype_test='float64/complex128',
    validation_criterion='equal126-stratum withheld Rician NLL, including epoch0',
    primary_criterion='equal126-cell dB ratio-of-sums NMSE, then equal training-seed average',
    primary_contrasts=['adaptive_vs_none','adaptive_vs_global','adaptive_vs_recurrent'],
    test_cases=4096, stress_cells_excluded_from_primary=2, architecture_search=False,
    timing_repeats=3, max_seconds=10800, specs=SPECS)


def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    paths = sorted(Path('compact_baseline').glob('*.py'))+[Path('compact_baseline/PROTOCOL.txt'),
        Path('sahure/model.py'),Path('sahure/math.py'),Path('sahure/train.py'),Path('sahure/data.py'),
        Path('sahure_expansion/data.py'),Path('sahure_expansion/train.py'),
        Path('fair_optshrink/estimator.py'),Path('optshrink_rank_check/estimator.py'),
        Path('results/fair_optshrink/frozen_selection.json'),Path('results/fair_optshrink/manifest.json')]
    paths += [DATA/p for p in ('train_observations.npz','val_observations.npz','val_conditions.npy')]
    original = {}
    for seed in CONFIG['seeds']:
        for arm in ('none','global','adaptive'):
            p=Path('results/sahure_expansion/checkpoints')/f'seed_{seed}'/f'em_{arm}_best.pt'
            original[f'{arm}_{seed}']=dict(path=str(p),sha256=sha(p))
            paths.append(p.with_name(f'em_{arm}_latest.pt'))
    parent=json.loads(Path('results/fair_optshrink/manifest.json').read_text())
    for path,digest in parent['sources'].items():
        if sha(path)!=digest: raise RuntimeError('historical source changed: '+path)
    for seed,entry in parent['models'].items():
        if entry['sha256']!=original['adaptive_'+seed]['sha256']:
            raise RuntimeError('not the original frozen188-parameter checkpoint')
    record=dict(config=CONFIG,sources={str(p):sha(p) for p in paths},models=original)
    target=ROOT/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=record:
        raise RuntimeError('compact protocol/source/data changed after freeze')
    if not target.exists(): write(target,record)
    return record


def freeze_test():
    record=initialize()
    models=dict(record['models'])
    for seed in CONFIG['seeds']:
        p=ROOT/'checkpoints'/f'seed_{seed}'/'recurrent_best.pt'
        models[f'recurrent_{seed}']=dict(path=str(p),sha256=sha(p))
    target=ROOT/'frozen_selection.json'
    frozen=dict(manifest_sha256=sha(ROOT/'manifest.json'),models=models)
    if target.exists():
        old=json.loads(target.read_text());old.pop('frozen_unix')
        if old!=frozen: raise RuntimeError('test freeze changed')
    else:
        if (ROOT/'data').exists(): raise RuntimeError('test observations generated before freeze')
        write(target,dict(**frozen,frozen_unix=time.time()))
