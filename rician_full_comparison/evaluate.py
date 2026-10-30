"""Paired fresh tests with frozen checkpoints and unchanged OptShrink selection.

The two learned training objectives use the same all-pilot inference interface.
Channel truth is saved separately and is never passed to an estimator here.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from fair_optshrink.estimator import estimate, lookup_key, tensors
from optshrink_rank_check.estimator import fit_candidates, select_and_estimate
from sahure.train import load_model
from sahure_expansion.data import world

from .protocol import CONFIG, ROOT, SPECS, load, save, sha, write


PARENT_SELECTION = Path('results/fair_optshrink/frozen_selection.json')
GENERATOR_SOURCE = Path('sahure_expansion/data.py')
ARMS = ('holdout_best', 'allpilot_best', 'holdout_final', 'allpilot_final')


def expected_models():
    return [f'{arm}_{seed}' for arm in ARMS for seed in CONFIG['seeds']]


def frozen_record():
    path = ROOT / 'frozen_selection.json'
    if not path.exists():
        raise RuntimeError('freeze checkpoints and settings before fresh evaluation')
    record = json.loads(path.read_text())
    if record['manifest_sha256'] != sha(ROOT / 'manifest.json'):
        raise RuntimeError('frozen experiment manifest changed')
    return record


def frozen_parent():
    frozen_record()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    expected = manifest['sources'][str(PARENT_SELECTION)]
    if sha(PARENT_SELECTION) != expected:
        raise RuntimeError('historical OptShrink selection changed')
    return json.loads(PARENT_SELECTION.read_text())


def load_models():
    entries = frozen_record()['models']
    if set(entries) != set(expected_models()):
        raise RuntimeError('unexpected frozen comparison model names')
    models = {}
    for name in expected_models():
        entry = entries[name]
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('frozen checkpoint changed: ' + name)
        model, _ = load_model(entry['path'], dtype=torch.float64)
        if sum(p.numel() for p in model.parameters()) != 188:
            raise RuntimeError('comparison requires the 188-parameter architecture')
        models[name] = model.eval()
    return models


def fixed_config(obs, parent):
    keys = lookup_key(obs)
    if len(set(keys)) != 1:
        raise ValueError('offline configuration lookup requires a homogeneous cell')
    return parent['choices'][keys[0]]['selected']


def data(bank, cell):
    """Fixed bank seeds and unique indices permit exact world regeneration."""
    frozen_record()
    if not 0 <= bank < len(CONFIG['banks']) or not 0 <= cell < len(SPECS):
        raise IndexError('unknown fresh-test bank or condition')
    digest = sha(ROOT / 'frozen_selection.json')
    generator_digest = sha(GENERATOR_SOURCE)
    folder = ROOT / 'data' / f'bank{bank}' / SPECS[cell]['id']
    manifest_path = folder / 'manifest.json'
    seed = CONFIG['banks'][bank]
    indices = np.arange(cell * CONFIG['cases'], (cell + 1) * CONFIG['cases'])
    ids = np.array([f'rician_full:{seed}:{index}' for index in indices])
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        identity = (meta['selection_sha256'] == digest
                    and meta['generator_sha256'] == generator_digest
                    and meta['seed'] == seed and meta['indices'] == indices.tolist()
                    and meta['spec'] == SPECS[cell])
        if not identity:
            raise RuntimeError('fresh-test data identity or source changed')
        for path, value in meta['hashes'].items():
            if sha(path) != value:
                raise RuntimeError('fresh-test data hash mismatch: ' + path)
        observations = load(folder / 'observations.npz')
        np.testing.assert_array_equal(observations['trial_ids'], ids)
        return observations
    arrays = {key: [] for key in ('z', 's', 'b', 'sigma2')}
    truth = []
    for index in indices:
        item, _ = world(int(index), seed, **SPECS[cell])
        for key, value in zip(arrays, (item.Z, item.S, item.B, item.sigma2)):
            arrays[key].append(value)
        truth.append(item.G)
    observations = {key: np.stack(values) for key, values in arrays.items()}
    if not all(np.isfinite(value).all() for value in observations.values()):
        raise FloatingPointError('nonfinite fresh observation')
    paths = [folder / 'observations.npz', folder / 'truth.npz']
    save(paths[0], **observations, trial_ids=ids)
    save(paths[1], g=np.stack(truth), trial_ids=ids)
    # The manifest is the commit marker; interrupted partial cells regenerate.
    write(manifest_path, dict(seed=seed, indices=indices.tolist(), count=len(truth),
          bank=bank, spec=SPECS[cell], selection_sha256=digest,
          generator_source=str(GENERATOR_SOURCE), generator_sha256=generator_digest,
          generated_unix=time.time(), hashes={str(path): sha(path) for path in paths}))
    return load(paths[0])


@torch.no_grad()
def evaluate(parent, models, check):
    digest = sha(ROOT / 'frozen_selection.json')
    names = ['rank_loop', 'opt_selected'] + expected_models()
    parent_digest = sha(PARENT_SELECTION)
    for bank in range(len(CONFIG['banks'])):
        for cell, spec in enumerate(SPECS):
            observed = data(bank, cell)
            path = ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz'
            if path.exists():
                previous = load(path)
                if (str(previous['selection_sha256']) != digest
                        or str(previous['parent_selection_sha256']) != parent_digest):
                    raise RuntimeError('prediction selection hash mismatch')
                np.testing.assert_array_equal(previous['trial_ids'], observed['trial_ids'])
                np.testing.assert_array_equal(previous['names'], names)
                if not np.isfinite(previous['estimates']).all():
                    raise FloatingPointError('nonfinite cached channel estimate')
                continue
            obs = tensors(observed)
            if obs.s.shape[-1] != 30:
                raise ValueError('rank selection requires the frozen 21/9 pilot split')
            rank_outputs, configs, scores = [], [], []
            tick = time.perf_counter()
            for case in range(len(obs)):
                check('testing', bank=bank, cell=cell, cells=len(SPECS), case=case)
                row = obs.batch(slice(case, case + 1))
                entry = parent['choices'][lookup_key(row)[0]]
                config, score = fit_candidates(row, entry['loop'])
                if config['placement'] != 'loop':
                    raise RuntimeError('rank control must use in-loop OptShrink')
                rank_outputs.append(estimate(row, config).numpy()[0])
                configs.append([config['rank'], config['strength'], config['iterations']])
                scores.append(score.numpy())
            outputs = dict(rank_loop=np.stack(rank_outputs))
            costs = dict(rank_loop=time.perf_counter() - tick)
            check('testing', bank=bank, cell=cell, cells=len(SPECS), method='opt_selected')
            tick = time.perf_counter()
            outputs['opt_selected'] = estimate(obs, fixed_config(obs, parent)).numpy()
            costs['opt_selected'] = time.perf_counter() - tick
            for name in expected_models():
                check('learned_testing', bank=bank, cell=cell, method=name)
                tick = time.perf_counter()
                outputs[name] = models[name](*obs.args()).numpy()
                costs[name] = time.perf_counter() - tick
            values = np.stack([outputs[name] for name in names])
            expected_shape = (len(names), len(obs), obs.z.shape[1], obs.s.shape[1])
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise FloatingPointError('invalid or nonfinite channel estimate')
            save(path, estimates=values, names=np.array(names),
                 trial_ids=observed['trial_ids'], rank_configs=np.array(configs),
                 rank_scores=np.array(scores), selection_sha256=digest,
                 parent_selection_sha256=parent_digest,
                 batch_evaluation_seconds=np.array([costs[name] for name in names]))
            print(f'Fresh bank {bank + 1}/{len(CONFIG["banks"])} cell '
                  f'{cell + 1}/{len(SPECS)} complete: {spec["id"]}', flush=True)


@torch.no_grad()
def timing(parent, models, check):
    """Batch-one timing includes initialization, the 21/9 search, and 30-pilot refit."""
    digest = sha(ROOT / 'frozen_selection.json')
    parent_digest = sha(PARENT_SELECTION)
    expected_names = ['rank_loop', 'opt_selected'] + expected_models()
    for cell, spec in enumerate(SPECS):
        path = ROOT / 'timing' / f'{spec["id"]}.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if (previous['selection_sha256'] != digest
                    or previous['parent_selection_sha256'] != parent_digest
                    or set(previous['seconds']) != set(expected_names)):
                raise RuntimeError('timing identity changed')
            continue
        item, _ = world(cell, CONFIG['timing_seed'], **spec)
        obs = tensors(dict(z=item.Z[None], s=item.S[None], b=item.B[None],
                           sigma2=np.array([item.sigma2])))
        methods = dict(rank_loop=lambda: select_and_estimate(obs, parent, 'loop')[0],
                       opt_selected=lambda: estimate(obs, fixed_config(obs, parent)))
        methods.update({name: (lambda model=models[name]: model(*obs.args()))
                        for name in expected_models()})
        for name, fn in methods.items():
            check('timing_warmup', cell=cell, method=name)
            value = fn()
            if not torch.isfinite(value).all():
                raise FloatingPointError('nonfinite timing estimate')
        seconds = {name: [] for name in methods}
        rng = np.random.default_rng(CONFIG['timing_seed'] + cell)
        for repeat in range(CONFIG['timing_repeats']):
            for name in rng.permutation(list(methods)):
                check('timing', cell=cell, method=str(name), repeat=repeat)
                start = time.perf_counter()
                methods[name]()
                seconds[name].append(time.perf_counter() - start)
        write(path, dict(seconds=seconds, spec=spec, selection_sha256=digest,
              parent_selection_sha256=parent_digest,
              backend='PyTorch CPU float64 one thread batch1',
              includes='initialization, controller, all iterations, rank search and refit; '
                       'excludes training, offline tuning, model loading and I/O'))


def run(check=lambda *args, **kwargs: None, models=None):
    if torch.get_num_threads() != 1:
        raise RuntimeError('comparison requires one CPU thread')
    parent = frozen_parent()
    models = load_models() if models is None else models
    if set(models) != set(expected_models()):
        raise RuntimeError('unexpected comparison models')
    for model in models.values():
        model.eval()
        if (next(model.parameters()).dtype != torch.float64
                or next(model.parameters()).device.type != 'cpu'):
            raise RuntimeError('all comparison models must use float64 on CPU')
    evaluate(parent, models, check)
    timing(parent, models, check)
