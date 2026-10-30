"""Fresh, observation-only matched inference; no new training or test tuning."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from fair_optshrink.estimator import estimate as opt_estimate, lookup_key, tensors
from optshrink_rank_check.estimator import fit_candidates, select_and_estimate as old_select
from sahure_expansion.data import world
from . import estimator, protocol

ROOT, CONFIG, SPECS = protocol.ROOT, protocol.CONFIG, protocol.SPECS
sha, write, save, load = protocol.sha, protocol.write, protocol.save, protocol.load
PARENT = Path('results/fair_optshrink/frozen_selection.json')
GENERATOR = Path('sahure_expansion/data.py')
MODEL_NAMES = ('original_9101', 'original_9102', 'original_9103')
NAMES = MODEL_NAMES + ('rank_old_opt', 'offline_old_opt', 'matched_onepass',
                       'lrhd_loop', 'lrhd_final', 'lrhd_offline', 'em100')


def frozen():
    path = ROOT / 'frozen_selection.json'
    if not path.exists():
        raise RuntimeError('selection must be frozen before fresh test generation')
    record = json.loads(path.read_text())
    if record['manifest_sha256'] != sha(ROOT / 'manifest.json'):
        raise RuntimeError('frozen manifest changed')
    return record


def load_models():
    models = protocol.models()
    if set(models) != set(MODEL_NAMES):
        raise RuntimeError('unexpected historical model names')
    for model in models.values():
        model.eval()
        parameter = next(model.parameters())
        if parameter.dtype != torch.float64 or parameter.device.type != 'cpu':
            raise RuntimeError('all estimators require CPU float64')
        if sum(p.numel() for p in model.parameters()) != 188:
            raise RuntimeError('original architecture changed')
    return models


def configuration(obs, selected, method, placement):
    """Known normalized variance; ties choose the lower development SNR."""
    if len(obs) != 1:
        raise ValueError('one observation required')
    snr = float(10 * torch.log10(3 / obs.sigma2[0]))
    grid = (-10, 0, 10, 20)
    key = str(min(grid, key=lambda value: (abs(snr-value), value)))
    config = selected['choices'][method][key][placement]
    if config['placement'] != placement:
        raise RuntimeError('wrong frozen filter placement')
    return config


@torch.no_grad()
def infer(name, obs, models, selected, parent):
    if name in MODEL_NAMES:
        return models[name](*obs.args()), {}
    if name == 'rank_old_opt':
        return old_select(obs, parent, 'loop')
    if name == 'offline_old_opt':
        config = parent['choices'][lookup_key(obs)[0]]['selected']
        return opt_estimate(obs, config), dict(config=config)
    if name == 'matched_onepass':
        config, scores = fit_candidates(obs, configuration(obs, selected, 'onepass', 'loop'))
        return opt_estimate(obs, config), dict(config=config, scores=scores.tolist())
    if name in ('lrhd_loop', 'lrhd_final'):
        placement = name.removeprefix('lrhd_')
        return estimator.select_and_estimate(obs, configuration(obs, selected, 'lrhd', placement),
            tolerance=CONFIG['tolerance'], max_sweeps=CONFIG['max_sweeps'])
    if name == 'lrhd_offline':
        config = configuration(obs, selected, 'lrhd', 'loop')
        value, diagnostics = estimator.estimate(obs, [dict(rank=config['rank'], strength=config['strength'])],
            iterations=config['iterations'], placement=config['placement'],
            tolerance=CONFIG['tolerance'], max_sweeps=CONFIG['max_sweeps'])
        return value, dict(config=config, refit=diagnostics)
    if name == 'em100':
        config = dict(rank=0, strength=0., iterations=100, placement='loop')
        return opt_estimate(obs, config), dict(config=config)
    raise ValueError(name)


def data(bank, cell):
    selection = frozen()
    digest, generator = sha(ROOT / 'frozen_selection.json'), sha(GENERATOR)
    folder = ROOT / 'data' / f'bank{bank}' / SPECS[cell]['id']
    meta_path = folder / 'manifest.json'
    seed = CONFIG['banks'][bank]
    indices = list(range(cell*CONFIG['cases'], (cell+1)*CONFIG['cases']))
    ids = np.array([f'lrhd_screen:{seed}:{i}' for i in indices])
    identity = dict(seed=seed, bank=bank, indices=indices, spec=SPECS[cell],
                    selection_sha256=digest, generator_sha256=generator)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if any(meta[k] != value for k, value in identity.items()):
            raise RuntimeError('fresh data identity changed')
        if meta['generated_unix'] < selection['frozen_unix']:
            raise RuntimeError('fresh data generated before freeze')
        for path, value in meta['hashes'].items():
            if sha(path) != value:
                raise RuntimeError('data hash changed: ' + path)
        observations = load(folder / 'observations.npz')
        np.testing.assert_array_equal(observations['trial_ids'], ids)
        return observations
    arrays = {key: [] for key in ('z', 's', 'b', 'sigma2')}
    truths = []
    for index in indices:
        item, _ = world(index, seed, **SPECS[cell])
        for key, value in zip(arrays, (item.Z, item.S, item.B, item.sigma2)):
            arrays[key].append(value)
        truths.append(item.G)
    observations = {key: np.stack(values) for key, values in arrays.items()}
    if not all(np.isfinite(value).all() for value in observations.values()):
        raise FloatingPointError('nonfinite generated observations')
    paths = (folder / 'observations.npz', folder / 'truth.npz')
    save(paths[0], **observations, trial_ids=ids)
    save(paths[1], g=np.stack(truths), trial_ids=ids)
    write(meta_path, dict(**identity, generated_unix=time.time(),
                         hashes={str(path): sha(path) for path in paths}))
    return observations | dict(trial_ids=ids)


def paths(bank, spec, case):
    base = ROOT / 'predictions' / f'bank{bank}' / spec['id'] / f'case{case:03d}'
    return base.with_suffix('.npz'), base.with_suffix('.json')


def cached(bank, cell, case, trial_id):
    path, commit = paths(bank, SPECS[cell], case)
    if not commit.exists():
        return None
    info = json.loads(commit.read_text())
    if (info['selection_sha256'] != sha(ROOT / 'frozen_selection.json')
            or info['parent_selection_sha256'] != sha(PARENT)
            or info['trial_id'] != str(trial_id)
            or info['prediction_sha256'] != sha(path)):
        raise RuntimeError('cached prediction identity changed')
    value = load(path)
    np.testing.assert_array_equal(value['names'], NAMES)
    if str(value['trial_id']) != str(trial_id):
        raise RuntimeError('prediction pairing changed')
    if value['estimates'].shape != (len(NAMES), 32, 3) or not np.isfinite(value['estimates']).all():
        raise FloatingPointError('invalid cached prediction')
    return value, info


@torch.no_grad()
def evaluate(models, selected, parent, check):
    for bank in range(len(CONFIG['banks'])):
        for cell, spec in enumerate(SPECS):
            check('testing', bank=bank, cell=cell, case=0)
            observations = data(bank, cell)
            obs = tensors(observations)
            for case, trial_id in enumerate(observations['trial_ids']):
                if cached(bank, cell, case, trial_id) is not None:
                    continue
                row = obs.batch(slice(case, case+1))
                estimates, diagnostics, seconds = [], {}, {}
                for name in NAMES:
                    check('testing', bank=bank, cell=cell, case=case, method=name)
                    start = time.perf_counter()
                    value, diagnostics[name] = infer(name, row, models, selected, parent)
                    seconds[name] = time.perf_counter()-start
                    if value.shape != (1, 32, 3) or not torch.isfinite(value).all():
                        raise FloatingPointError('invalid estimate: ' + name)
                    estimates.append(value.numpy()[0])
                path, commit = paths(bank, spec, case)
                save(path, estimates=np.stack(estimates), names=np.array(NAMES), trial_id=trial_id)
                write(commit, dict(trial_id=str(trial_id), diagnostics=diagnostics,
                    seconds=seconds, prediction_sha256=sha(path),
                    selection_sha256=sha(ROOT / 'frozen_selection.json'),
                    parent_selection_sha256=sha(PARENT)))
            print(f'Fresh bank {bank+1}/{len(CONFIG["banks"])} cell {cell+1}/{len(SPECS)} '
                  f'complete: {spec["id"]}', flush=True)


@torch.no_grad()
def timing(models, selected, parent, check):
    digest, parent_digest = sha(ROOT / 'frozen_selection.json'), sha(PARENT)
    for cell, spec in enumerate(SPECS):
        path = ROOT / 'timing' / f'{spec["id"]}.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if (previous['selection_sha256'] != digest or previous['parent_selection_sha256'] != parent_digest
                    or set(previous['seconds']) != set(NAMES)):
                raise RuntimeError('timing identity changed')
            continue
        item, _ = world(cell, CONFIG['timing_seed'], **spec)
        obs = tensors(dict(z=item.Z[None], s=item.S[None], b=item.B[None], sigma2=np.array([item.sigma2])))
        for name in NAMES:
            check('timing_warmup', cell=cell, method=name)
            value, _ = infer(name, obs, models, selected, parent)
            if not torch.isfinite(value).all():
                raise FloatingPointError('nonfinite timed estimator')
        seconds = {name: [] for name in NAMES}
        rng = np.random.default_rng(CONFIG['timing_seed']+cell)
        order = list(rng.permutation(NAMES))
        for repeat in range(CONFIG['timing_repeats']):
            rotated = order[repeat:]+order[:repeat]
            for name in rotated:
                check('timing', cell=cell, method=str(name), repeat=repeat)
                start = time.perf_counter()
                infer(str(name), obs, models, selected, parent)
                seconds[str(name)].append(time.perf_counter()-start)
        write(path, dict(seconds=seconds, selection_sha256=digest, parent_selection_sha256=parent_digest,
            seed=CONFIG['timing_seed'], index=cell, spec=spec,
            backend='PyTorch CPU float64 one thread batch1',
            includes='initialization, every EM/inner sweep, rank selection and refit; '
                     'excludes model loading, training, offline tuning and I/O'))


def run(check=lambda *args, **kwargs: None):
    if torch.get_num_threads() != 1:
        raise RuntimeError('one CPU thread required')
    protocol.initialize()
    selected = frozen()
    parent = json.loads(PARENT.read_text())
    models = load_models()
    evaluate(models, selected, parent, check)
    timing(models, selected, parent, check)
