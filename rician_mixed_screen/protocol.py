"""Fixed settings and immutable provenance for the one-seed screen."""
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from fair_optshrink.run import SPECS
from rician_full_comparison.protocol import sha, write, save, load

ROOT = Path('results/rician_mixed_screen')
DATA = Path('results/sahure_expansion/data')
PARENT = Path('results/rician_full_comparison')
CONFIG = dict(
    version=1, seed=9101, seeds=[9101], epochs=20, batch_size=32, lr=.001,
    stages=10, alpha=1., train_cases=16384, val_cases=2048,
    validation_mask_seed=2026092203, banks=[20261015111001], cases=32,
    primary_cells=126, timing_seed=20261015121001, timing_repeats=3,
    bootstrap_seed=20261015131001, bootstrap_samples=2000,
    threads=1, workers=1, max_seconds=7200, specs=SPECS,
    branch_rule='(zero-based batch number + zero-based epoch) modulo 2; 0 holdout, 1 allpilot',
    selection='equal126-stratum fixed21/9 held-out Rician NLL; epoch0 eligible',
    primary='selected mixed versus selected original and allpilot, seed9101',
    secondary='final20-epoch sensitivity and frozen classical comparators',
    metric='equal126-cell mean dB ratio-of-summed-errors to summed-channel-power',
    screen=dict(min_gain_db=.2, max_severe_excess_over_original=.002,
                max_runtime_ratio=1.05, low_snr=[-10., -5.]),
    scope='one-seed development screen, not independent publication confirmation',
)


def historical_models():
    parent = json.loads((PARENT / 'frozen_selection.json').read_text())
    models = {}
    for arm in ('holdout_best', 'holdout_final', 'allpilot_best', 'allpilot_final'):
        entry = parent['models'][f'{arm}_{CONFIG["seed"]}']
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('historical checkpoint changed: ' + arm)
        models[arm] = entry
    return models


def initialize():
    if json.loads((PARENT / 'status.json').read_text())['state'] != 'complete':
        raise RuntimeError('parent comparison must be complete')
    prior = json.loads((PARENT / 'manifest.json').read_text())
    replay = json.loads((PARENT / 'baseline_replay.json').read_text())
    if not replay['passed']:
        raise RuntimeError('historical baseline replay did not pass')
    # Reuse, but revalidate, the successful full baseline replay. No old writes.
    for path, digest in prior['sources'].items():
        if sha(path) != digest:
            raise RuntimeError('frozen parent source or data changed: ' + path)
    if prior['environment']['torch'] != torch.__version__:
        raise RuntimeError('PyTorch environment differs from matched comparison')
    for key in ('epochs', 'batch_size', 'lr', 'stages', 'train_cases', 'val_cases',
                'validation_mask_seed'):
        if CONFIG[key] != prior['config'][key]:
            raise RuntimeError('matched setting changed: ' + key)
    models = historical_models()
    paths = sorted(Path('rician_mixed_screen').glob('*.py'))
    paths += [Path('rician_mixed_screen/PROTOCOL.txt'),
              PARENT / 'manifest.json', PARENT / 'baseline_replay.json',
              PARENT / 'frozen_selection.json',
              Path('results/fair_optshrink/frozen_selection.json')]
    paths += [Path(x['path']) for x in models.values()]
    record = dict(config=CONFIG, sources={**prior['sources'], **{str(p): sha(p) for p in paths}},
                  historical_models=models,
                  initial_parameters_sha256=replay['initial_parameters_sha256'][str(CONFIG['seed'])],
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   torch=torch.__version__, system=platform.platform()))
    path = ROOT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise RuntimeError('mixed-screen sources, data, environment or settings changed')
    if not path.exists():
        write(path, record)
    return record


def freeze_selection():
    record = initialize()
    models = dict(record['historical_models'])
    folder = ROOT / 'checkpoints' / f'seed_{CONFIG["seed"]}'
    state = torch.load(folder / 'mixed_latest.pt', map_location='cpu', weights_only=False)
    if state['epoch'] != CONFIG['epochs']:
        raise RuntimeError('all twenty epochs must finish before freezing evaluation')
    for label, suffix in (('best', 'best'), ('final', 'latest')):
        path = folder / f'mixed_{suffix}.pt'
        models[f'mixed_{label}'] = dict(path=str(path), sha256=sha(path))
    value = dict(manifest_sha256=sha(ROOT / 'manifest.json'), models=models)
    path = ROOT / 'frozen_selection.json'
    if path.exists():
        old = json.loads(path.read_text())
        old.pop('frozen_unix')
        if old != value:
            raise RuntimeError('models changed after evaluation freeze')
    else:
        if (ROOT / 'data').exists():
            raise RuntimeError('development data exists before selection freeze')
        write(path, dict(**value, frozen_unix=time.time()))
    return models
