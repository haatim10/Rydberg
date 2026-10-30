"""Fresh paired evaluation; no test-dependent setting or checkpoint selection."""
import json
import time

import numpy as np
import torch

from fair_optshrink.estimator import estimate, lookup_key, tensors
from optshrink_rank_check.estimator import fit_candidates, select_and_estimate
from sahure_expansion.data import world

from .models import load_models
from .protocol import CONFIG, ROOT, SPECS, load, save, sha, write


PARENT = ROOT.parent / 'fair_optshrink'


def frozen_parent():
    return json.loads((PARENT / 'frozen_selection.json').read_text())


def fixed_config(obs, parent):
    # This lookup observes only the known variance and reference energy.
    return parent['choices'][lookup_key(obs)[0]]['selected']


def data(bank, cell):
    """Keep each bank seed fixed; unique case indices distinguish conditions."""
    freeze = ROOT / 'frozen_selection.json'
    if not freeze.exists():
        raise RuntimeError('selection must be frozen before generating test data')
    digest = sha(freeze)
    folder = ROOT / 'data' / f'bank{bank}' / SPECS[cell]['id']
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        if meta['selection_sha256'] != digest:
            raise RuntimeError('data selection hash mismatch')
        for path, value in meta['hashes'].items():
            if sha(path) != value:
                raise RuntimeError('data hash mismatch: ' + path)
        return load(folder / 'observations.npz')
    seed = CONFIG['banks'][bank]
    indices = np.arange(cell * CONFIG['cases'], (cell + 1) * CONFIG['cases'])
    arrays = {key: [] for key in ('z', 's', 'b', 'sigma2')}
    truth = []
    for index in indices:
        w, _ = world(int(index), seed, **SPECS[cell])
        for key, value in zip(arrays, (w.Z, w.S, w.B, w.sigma2)):
            arrays[key].append(value)
        truth.append(w.G)
    ids = np.array([f'compact:{seed}:{index}' for index in indices])
    paths = [folder / 'observations.npz', folder / 'truth.npz']
    save(paths[0], **{key: np.stack(values) for key, values in arrays.items()}, trial_ids=ids)
    save(paths[1], g=np.stack(truth), trial_ids=ids)
    write(manifest_path, dict(seed=seed, indices=indices.tolist(), count=len(truth),
          spec=SPECS[cell], selection_sha256=digest, generated_unix=time.time(),
          hashes={str(path): sha(path) for path in paths}))
    return load(paths[0])


@torch.no_grad()
def evaluate(parent, models, check):
    digest = sha(ROOT / 'frozen_selection.json')
    expected_names = ['rank_loop', 'opt_selected'] + list(models)
    for bank in range(len(CONFIG['banks'])):
        for cell, spec in enumerate(SPECS):
            observed = data(bank, cell)
            path = ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz'
            if path.exists():
                previous = load(path)
                if str(previous['selection_sha256']) != digest:
                    raise RuntimeError('prediction selection hash mismatch')
                np.testing.assert_array_equal(previous['trial_ids'], observed['trial_ids'])
                np.testing.assert_array_equal(previous['names'], expected_names)
                continue
            obs = tensors(observed)
            rank_outputs, configs, scores = [], [], []
            for case in range(len(obs)):
                check('testing', bank=bank, cell=cell, cells=len(SPECS), case=case)
                row = obs.batch(slice(case, case + 1))
                entry = parent['choices'][lookup_key(row)[0]]
                config, score = fit_candidates(row, entry['loop'])
                rank_outputs.append(estimate(row, config).numpy()[0])
                configs.append([config['rank'], config['strength'], config['iterations']])
                scores.append(score.numpy())
            outputs = dict(rank_loop=np.stack(rank_outputs),
                           opt_selected=estimate(obs, fixed_config(obs, parent)).numpy())
            for name, model in models.items():
                check('learned_testing', bank=bank, cell=cell, method=name)
                outputs[name] = model(*obs.args()).numpy()
            values = np.stack(list(outputs.values()))
            if not np.isfinite(values).all():
                raise FloatingPointError('nonfinite channel estimate')
            save(path, estimates=values, names=np.array(list(outputs)),
                 trial_ids=observed['trial_ids'], rank_configs=np.array(configs),
                 rank_scores=np.array(scores), selection_sha256=digest)
            print(f'Fresh bank {bank + 1}/{len(CONFIG["banks"])} cell '
                  f'{cell + 1}/{len(SPECS)} complete: {spec["id"]}', flush=True)


@torch.no_grad()
def timing(parent, models, check):
    """Single-world, serial inference includes every estimator's initialization."""
    digest = sha(ROOT / 'frozen_selection.json')
    for cell, spec in enumerate(SPECS):
        path = ROOT / 'timing' / f'{spec["id"]}.json'
        if path.exists():
            if json.loads(path.read_text())['selection_sha256'] != digest:
                raise RuntimeError('timing selection hash mismatch')
            continue
        w, _ = world(cell, CONFIG['timing_seed'], **spec)
        obs = tensors(dict(z=w.Z[None], s=w.S[None], b=w.B[None],
                           sigma2=np.array([w.sigma2])))
        methods = dict(rank_loop=lambda: select_and_estimate(obs, parent, 'loop')[0],
                       opt_selected=lambda: estimate(obs, fixed_config(obs, parent)))
        methods.update({name: (lambda model=model: model(*obs.args()))
                        for name, model in models.items()})
        for fn in methods.values():
            fn()
        seconds = {name: [] for name in methods}
        rng = np.random.default_rng(CONFIG['timing_seed'] + cell)
        for repeat in range(3):
            for name in rng.permutation(list(methods)):
                check('timing', cell=cell, method=str(name), repeat=repeat)
                start = time.perf_counter()
                methods[name]()
                seconds[name].append(time.perf_counter() - start)
        write(path, dict(seconds=seconds, spec=spec, selection_sha256=digest,
              backend='PyTorch CPU float64 one thread batch1',
              includes='initialization, controller, all iterations, rank search and refit; excludes model loading and I/O'))


def run(check=lambda *args, **kwargs: None):
    if torch.get_num_threads() != 1:
        raise RuntimeError('focused comparison requires one CPU thread')
    if not (ROOT / 'frozen_selection.json').exists():
        raise RuntimeError('freeze checkpoints and settings before test evaluation')
    parent, models = frozen_parent(), load_models()
    expected = {f'{arm}_{seed}' for arm in ('none', 'global', 'adaptive', 'recurrent')
                for seed in CONFIG['seeds']}
    if set(models) != expected:
        raise RuntimeError('unexpected comparison models')
    for model in models.values():
        model.eval()
        if next(model.parameters()).dtype != torch.float64:
            raise RuntimeError('all comparison models must use float64 at test time')
    evaluate(parent, models, check)
    timing(parent, models, check)

