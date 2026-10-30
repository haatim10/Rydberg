"""Freeze one development-selected noise rule before fresh paired evaluation."""
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from fair_optshrink.run import SPECS as ORIGINAL_SPECS
from rician_full_comparison.protocol import sha, write, load, save

ROOT = Path('results/rician_selector_screen')
DATA = Path('results/sahure_expansion/data')
PARENT = Path('results/rician_full_comparison')
SPECS = list(ORIGINAL_SPECS) + [
    dict(id=f'boundary_{"minus" if snr < 0 else "plus"}2p5_rsr{rsr}',
         snr=snr, rsr=float(rsr), L=None)
    for snr in (-2.5, 2.5) for rsr in (5, 12)
]
CONFIG = dict(
    version=1, seed=9101, seeds=[9101], nominal_power=3.,
    thresholds=[-100., -7.5, -2.5, 2.5, 7.5, 12.5, 17.5, 100.],
    validation_mask_seed=2026092203, validation_cases=2048,
    banks=[20261015151001], cases=32, primary_cells=126,
    noise_offsets=[0., -3., 3.], timing_seed=20261015161001,
    timing_repeats=3, bootstrap_seed=20261015171001, bootstrap_samples=2000,
    threads=1, max_seconds=3600, specs=SPECS,
    selection='minimum equal126-stratum 21/9 validation Rician NLL; tie prefers original',
    screen=dict(min_gain_db=.1, max_severe_excess=.002, max_runtime_ratio=1.05),
    scope='one-seed routing control, not a new estimator or publication confirmation',
)


def historical_models():
    parent = json.loads((PARENT / 'frozen_selection.json').read_text())
    models = {name: parent['models'][f'{name}_{CONFIG["seed"]}']
              for name in ('holdout_best', 'allpilot_best')}
    for name, entry in models.items():
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('historical checkpoint changed: ' + name)
    return models


def initialize():
    prior = json.loads((PARENT / 'manifest.json').read_text())
    if json.loads((PARENT / 'status.json').read_text())['state'] != 'complete':
        raise RuntimeError('original/all-pilot comparison must be complete')
    for path, digest in prior['sources'].items():
        if sha(path) != digest:
            raise RuntimeError('frozen source/data changed: ' + path)
    if prior['environment']['torch'] != torch.__version__:
        raise RuntimeError('PyTorch environment changed')
    models = historical_models()
    paths = sorted(Path('rician_selector_screen').glob('*.py'))
    paths += [Path('rician_selector_screen/PROTOCOL.txt'),
              PARENT / 'manifest.json', PARENT / 'frozen_selection.json']
    paths += [Path(x['path']) for x in models.values()]
    record = dict(config=CONFIG, models=models,
                  sources={**prior['sources'], **{str(p): sha(p) for p in paths}},
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   torch=torch.__version__, system=platform.platform()))
    path = ROOT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise RuntimeError('frozen selector protocol/source/data changed')
    if not path.exists():
        write(path, record)
    return record


def freeze_selection():
    record = initialize()
    path = ROOT / 'frozen_selection.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['manifest_sha256'] != sha(ROOT / 'manifest.json') or old['models'] != record['models']:
            raise RuntimeError('router freeze changed')
        return old
    if (ROOT / 'data').exists():
        raise RuntimeError('fresh cases generated before selecting the routing rule')
    from sahure.train import load_model
    from .selector import fit_rule, THRESHOLDS
    if list(THRESHOLDS) != CONFIG['thresholds']:
        raise RuntimeError('selector candidate list differs from frozen protocol')
    models = {name: load_model(value['path'], dtype=torch.float64)[0]
              for name, value in record['models'].items()}
    choice = fit_rule(models)
    if choice['threshold_db'] not in CONFIG['thresholds']:
        raise RuntimeError('unregistered selected threshold')
    value = dict(manifest_sha256=sha(ROOT / 'manifest.json'), models=record['models'],
                 threshold_db=choice['threshold_db'], development=choice,
                 frozen_unix=time.time(),
                 storage_parameters=376, active_parameters_per_block=188)
    write(path, value)
    return value
