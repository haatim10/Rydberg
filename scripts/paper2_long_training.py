"""Matched conventional-loss budget experiment; CPU, resumable, isolated outputs.

Run with the project .venv-paper2 Python, from the repository root:
  PYTHONPATH=. python scripts/paper2_long_training.py smoke
  PYTHONPATH=. python scripts/paper2_long_training.py launch
  PYTHONPATH=. python scripts/paper2_long_training.py status
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t
import torch
from torch.utils.data import DataLoader, Dataset

from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset, collate
from trackD_urformer.stage1 import build_model
from trackD_urformer.stage2 import select_epoch, validation_per_trial
from trackD_urformer.train import nmse_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paper2_long_training'
SEEDS = {1: 20260827, 2: 20262827, 3: 20263827}
BUDGETS = (13, 26, 52)
COUNTS = {'train': 80000, 'val': 2000, 'test': 2000}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save_torch(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def cfg_for(seed, hankel, budget):
    base = TrackDConfig()
    return replace(base,
        data=replace(base.data, n_train=80000, n_val=2000, n_test=2000),
        train=replace(base.train, seed=SEEDS[seed], init='spectral', epochs=budget),
        model=replace(base.model, filter_init='random', use_transformer=True,
                      use_hankel=hankel, hankel_rank=7, hankel_mode='fixed',
                      hankel_gate='none'))


def dataset(cfg, split):
    return TrackDDataset(split, sysc=cfg.system, datac=cfg.data,
                        numeric=cfg.numeric, init='spectral', cache=False)


class CachedData(Dataset):
    """Disk-backed float32 copies of the original deterministic dataset items."""
    def __init__(self, split):
        folder = OUT / 'data' / split
        self.meta = json.loads((folder / 'complete.json').read_text())
        # Copy-on-write mapping: torch can safely wrap without read-only warning.
        self.arrays = {key: np.load(folder / f'{key}.npy', mmap_mode='c')
                       for key in self.meta['keys']}

    def __len__(self):
        return self.meta['count']

    def __getitem__(self, index):
        return {key: torch.as_tensor(array[index]) for key, array in self.arrays.items()}


def cache_data():
    cfg = cfg_for(1, False, 13)
    for split, count in COUNTS.items():
        folder = OUT / 'data' / split
        if (folder / 'complete.json').exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        original = dataset(cfg, split)
        first = original[0]
        arrays = {key: np.lib.format.open_memmap(folder / f'{key}.npy', mode='w+',
                  dtype=value.numpy().dtype, shape=(count, *value.shape))
                  for key, value in first.items()}
        start = time.monotonic()
        for index in range(count):
            item = original[index]
            for key, value in item.items():
                arrays[key][index] = value.numpy()
            original._g0_cache.clear()
            if (index + 1) % 1000 == 0:
                for array in arrays.values():
                    array.flush()
                progress = {'state': 'generating_data', 'split': split,
                            'samples': index + 1, 'total': count,
                            'seconds': time.monotonic() - start}
                save_json(OUT / 'status.json', progress)
                print(json.dumps(progress), flush=True)
        for array in arrays.values():
            array.flush()
        save_json(folder / 'complete.json', {'count': count, 'keys': list(arrays),
                   'config': cfg.to_dict(), 'sha256': {
                     key: hash_file(folder / f'{key}.npy') for key in arrays}})
        cached = CachedData(split)
        for index in [0, count // 2, count - 1]:
            for key, value in original[index].items():
                assert torch.equal(value, cached[index][key]), (split, index, key)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def train_step(model, optimizer, batch, cfg):
    optimizer.zero_grad(set_to_none=True)
    estimate = model(batch['G0'], batch['Z'], batch['S'], batch['B'], batch['sigma2'])
    loss = nmse_loss(estimate, batch['G_true'])
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss; refusing to continue.')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip,
                                         error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


def smoke():
    torch.set_num_threads(1)
    cfg = cfg_for(1, False, 13)
    ds = dataset(cfg, 'train')
    batches = [collate([ds[i] for i in range(start, start + 32)])
               for start in (0, 32, 64)]
    models = [build_model(cfg_for(1, h, 13), 'arm1b_full_random')[0]
              for h in (False, True)]
    assert all(torch.equal(a, b) for a, b in zip(models[0].parameters(),
                                                models[1].parameters()))
    timing = {}
    for h, model in zip((False, True), models):
        opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        train_step(model, opt, batches[0], cfg)
        start = time.monotonic()
        values = [train_step(model, opt, b, cfg)[0] for b in batches[1:]]
        timing['H' if h else 'U'] = {'seconds_per_batch': (time.monotonic()-start)/2,
                                    'losses': values}
    # Resume must reproduce the very next update exactly, including Adam state.
    import copy
    model = models[0]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    train_step(model, opt, batches[0], cfg)
    saved_model, saved_opt = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    rng = torch.get_rng_state()
    train_step(model, opt, batches[1], cfg)
    expected = copy.deepcopy(model.state_dict())
    model.load_state_dict(saved_model)
    opt.load_state_dict(saved_opt)
    torch.set_rng_state(rng)
    train_step(model, opt, batches[1], cfg)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in expected.items())
    result = {'matched_initial_parameters': True, 'exact_resume_update': True,
              'timing': timing, 'torch': torch.__version__, 'numpy': np.__version__,
              'scipy': scipy.__version__, 'platform': platform.platform(),
              'estimated_training_hours_excluding_validation_data_generation':
                  3 * sum(BUDGETS) * 2500 * sum(v['seconds_per_batch'] for v in timing.values()) / 3600}
    save_json(OUT / 'smoke.json', result)
    print(json.dumps(result, indent=2), flush=True)


@torch.no_grad()
def test_model(model, loader):
    model.eval()
    records = {key: [] for key in ['numerator', 'denominator', 'snr_db', 'trial']}
    for batch in loader:
        estimate = model(batch['G0'], batch['Z'], batch['S'], batch['B'], batch['sigma2'])
        records['numerator'].extend(torch.abs(estimate - batch['G_true']).square().sum((1, 2)).double().tolist())
        records['denominator'].extend(torch.abs(batch['G_true']).square().sum((1, 2)).double().tolist())
        for key in ['snr_db', 'trial']:
            records[key].extend(batch[key].tolist())
    return {key: np.asarray(value) for key, value in records.items()}


def verify_original():
    """Re-evaluate the saved conventional pairs against the committed scores."""
    torch.set_num_threads(1)
    reference_path = ROOT / 'reports/p19/p29_score.json'
    reference = json.loads(reference_path.read_text())
    cfg = cfg_for(1, False, 13)
    original = dataset(cfg, 'test')
    items = [original[i] for i in range(2000)]
    loader = DataLoader(items, batch_size=32, shuffle=False, collate_fn=collate)
    results = {}
    for seed in SEEDS:
        records = []
        for hankel in [False, True]:
            if seed == 1:
                path = ROOT / ('results/track_d/stage3/H1_hs_urformer_80k/best.pt'
                               if hankel else 'results/track_d/stage2/B3_80k_13ep/best.pt')
            else:
                path = ROOT / f'results/p17/tier1/C2_{"H1" if hankel else "U1"}_seed{seed}/best.pt'
            blob = torch.load(path, map_location='cpu', weights_only=False)
            assert blob['config']['train']['init'] == 'spectral'
            assert blob['config']['train']['seed'] == SEEDS[seed]
            model, _ = build_model(cfg_for(seed, hankel, 13), 'arm1b_full_random')
            model.load_state_dict(blob['model'])
            records.append(test_model(model, loader))
        u, h = records
        delta = 10*np.log10(u['numerator']/h['numerator'])
        measured = float(np.median(delta[u['snr_db'] >= 5]))
        expected = reference['per_seed'][f'seed{seed}']['ge5_median_db']
        per_bin = [float(np.median(delta[(u['snr_db'] >= lo) & (u['snr_db'] < lo+5)]))
                   for lo in range(-10, 20, 5)]
        expected_bins = reference['per_seed'][f'seed{seed}']['per_bin_db']
        error = max(abs(measured-expected), max(abs(a-b) for a,b in zip(per_bin, expected_bins)))
        results[f'seed{seed}'] = {'measured_ge5_db': measured, 'reference_ge5_db': expected,
             'per_bin_db': per_bin, 'max_error_db': error, 'tolerance_db': .005,
             'passed': error <= .005}
        print(json.dumps(results[f'seed{seed}']), flush=True)
    save_json(OUT / 'original_reproduction.json', results)
    assert all(r['passed'] for r in results.values()), 'Original checkpoint reproduction failed'


def validate_pipeline():
    """Exercise caching, training, selection and scoring on disposable tiny data."""
    import tempfile
    global OUT, COUNTS, BUDGETS
    saved = OUT, COUNTS, BUDGETS
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory(prefix='pipeline-check-', dir=OUT) as folder:
            OUT = Path(folder)
            COUNTS = {'train': 64, 'val': 64, 'test': 64}
            BUDGETS = (2, 13, 26, 52)
            cache_data()
            data = {split: CachedData(split) for split in COUNTS}
            for hankel in [False, True]:
                train_one(1, hankel, 2, data)
                target = OUT / f'seed1_{"H" if hankel else "U"}_2ep'
                before = hash_file(target / 'checkpoint.pt')
                train_one(1, hankel, 2, data)
                assert before == hash_file(target / 'checkpoint.pt')
                # Remove only the tiny result marker, then re-enter via checkpoint.
                (target / 'result.json').unlink()
                train_one(1, hankel, 2, data)
                assert before == hash_file(target / 'checkpoint.pt')
            summary = json.loads((OUT / 'summary.json').read_text())
            assert summary['2ep_selected']['n_seeds'] == 1
            assert summary['2ep_final']['n_seeds'] == 1
    finally:
        OUT, COUNTS, BUDGETS = saved
    save_json(OUT / 'pipeline_check.json', {'passed': True, 'checks': [
        'cached tensors equal original generator', 'structured/unstructured batch order matches',
        'finite forward/backward and validation', 'validation-only epoch selection',
        'selected and final test scoring', 'completed run skips safely',
        'checkpoint reload and result recovery']})
    print('Pipeline checks passed; tiny diagnostic results discarded.', flush=True)


def train_one(seed, hankel, budget, data):
    name = f'seed{seed}_{"H" if hankel else "U"}_{budget}ep'
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    if (folder / 'result.json').exists():
        return
    cfg = cfg_for(seed, hankel, budget)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    train_loader = DataLoader(data['train'], batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(data['val'], batch_size=32, shuffle=False, collate_fn=collate)
    model, meta = build_model(cfg, 'arm1b_full_random')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget)
    checkpoint = folder / 'checkpoint.pt'
    start, history, validations = 0, [], []
    if checkpoint.exists():
        blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert blob['config'] == cfg.to_dict()
        model.load_state_dict(blob['model'])
        optimizer.load_state_dict(blob['optimizer'])
        scheduler.load_state_dict(blob['scheduler'])
        start, history, validations = blob['epoch']+1, blob['history'], blob['validations']
        torch.set_rng_state(blob['torch_rng'])
        np.random.set_state(blob['numpy_rng'])
    for epoch in range(start, budget):
        model.train()
        t0, total = time.monotonic(), 0.0
        order_hash = hashlib.sha256()
        for batch_index, batch in enumerate(train_loader):
            order_hash.update(batch['trial'].numpy().tobytes())
            loss, grad_norm = train_step(model, optimizer, batch, cfg)
            total += loss
            if (batch_index + 1) % 100 == 0:
                progress = {'state': 'training', 'run': name, 'epoch': epoch+1,
                            'budget': budget, 'batch': batch_index+1,
                            'batches': len(train_loader), 'epoch_seconds': time.monotonic()-t0,
                            'loss_so_far': total/(batch_index+1)}
                save_json(OUT / 'status.json', progress)
                print(json.dumps(progress), flush=True)
        scheduler.step()
        v = validation_per_trial(model, val_loader, cfg)
        if not np.isfinite(v).all():
            raise RuntimeError('Non-finite validation error.')
        validations.append(v)
        rec = {'epoch': epoch+1, 'train_loss': total/len(train_loader),
               'validation_db': float(10*np.log10(v.mean())),
               'lr': scheduler.get_last_lr()[0], 'seconds': time.monotonic()-t0,
               'data_order_sha256': order_hash.hexdigest()}
        history.append(rec)
        save_torch(folder / f'epoch_{epoch+1:03d}.pt', {'model': model.state_dict()})
        save_torch(checkpoint, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'epoch': epoch, 'history': history,
             'validations': validations, 'config': cfg.to_dict(),
             'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state()})
        save_json(folder / 'history.json', history)
        print(name, json.dumps(rec), flush=True)
    curve = np.asarray([r['validation_db'] for r in history])
    chosen, selection = select_epoch(curve, validations[int(curve.argmin())], 'one_se')
    test_loader = DataLoader(data['test'], batch_size=32, shuffle=False, collate_fn=collate)
    for label, ep in [('selected', chosen+1), ('final', budget)]:
        model.load_state_dict(torch.load(folder / f'epoch_{ep:03d}.pt', weights_only=True)['model'])
        records = test_model(model, test_loader)
        np.savez(folder / f'test_{label}.npz', **records)
    save_json(folder / 'result.json', {'run': name, 'seed': seed, 'hankel': hankel,
        'budget': budget, 'chosen_epoch_1based': chosen+1, 'selection': selection,
        'config': cfg.to_dict(), 'history': history, 'model_meta': meta})
    summarize()


def summarize():
    results = {}
    for budget in BUDGETS:
        for label in ['selected', 'final']:
            pairs = []
            for seed in SEEDS:
                dirs = [OUT / f'seed{seed}_{arm}_{budget}ep' for arm in ['U', 'H']]
                if not all((d / 'result.json').exists() for d in dirs):
                    continue
                run_meta = [json.loads((d / 'result.json').read_text()) for d in dirs]
                assert [x['data_order_sha256'] for x in run_meta[0]['history']] == [
                    x['data_order_sha256'] for x in run_meta[1]['history']]
                u, h = [np.load(d / f'test_{label}.npz') for d in dirs]
                assert np.array_equal(u['trial'], h['trial'])
                assert np.array_equal(u['denominator'], h['denominator'])
                bins = {}
                masks = {'snr_ge5': u['snr_db'] >= 5, 'all': np.ones(len(u['trial']), dtype=bool)}
                for lo, hi in zip(range(-10, 20, 5), range(-5, 25, 5)):
                    masks[f'[{lo},{hi})'] = (u['snr_db'] >= lo) & (u['snr_db'] < hi)
                for key, mask in masks.items():
                    delta = 10*np.log10(u['numerator'][mask]/h['numerator'][mask])
                    bins[key] = {'n': int(mask.sum()), 'paired_median_db': float(np.median(delta)),
                       'ratio_of_squared_error_sums_db': float(10*np.log10(u['numerator'][mask].sum()/h['numerator'][mask].sum())),
                       'ratio_of_normalized_error_sums_db': float(10*np.log10((u['numerator'][mask]/u['denominator'][mask]).sum()/(h['numerator'][mask]/h['denominator'][mask]).sum()))}
                pairs.append({'seed': seed, 'bins': bins})
            vals = [p['bins']['snr_ge5']['paired_median_db'] for p in pairs]
            record = {'pairs': pairs, 'n_seeds': len(vals)}
            if vals:
                record['mean_gain_db'] = float(np.mean(vals))
            if len(vals) > 1:
                sd = float(np.std(vals, ddof=1))
                margin = float(t.ppf(.975, len(vals)-1)*sd/np.sqrt(len(vals)))
                record.update(sd_db=sd, seed_ci95_db=[float(np.mean(vals)-margin), float(np.mean(vals)+margin)])
            results[f'{budget}ep_{label}'] = record
    for budget in (26, 52):
        for label in ['selected', 'final']:
            earlier = {p['seed']: p['bins']['snr_ge5']['paired_median_db'] for p in results[f'13ep_{label}']['pairs']}
            change = [p['bins']['snr_ge5']['paired_median_db']-earlier[p['seed']]
                      for p in results[f'{budget}ep_{label}']['pairs'] if p['seed'] in earlier]
            if change:
                results[f'{budget}ep_{label}']['paired_changes_from_13ep_db'] = change
                results[f'{budget}ep_{label}']['mean_paired_change_db'] = float(np.mean(change))
                if len(change) > 1:
                    margin = float(t.ppf(.975, len(change)-1)*np.std(change, ddof=1)/np.sqrt(len(change)))
                    results[f'{budget}ep_{label}']['paired_change_seed_ci95_db'] = [float(np.mean(change)-margin), float(np.mean(change)+margin)]
    save_json(OUT / 'summary.json', results)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        reproduction = json.loads((OUT / 'original_reproduction.json').read_text())
        assert len(reproduction) == 3 and all(r['passed'] for r in reproduction.values())
        manifest = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'pid': os.getpid(), 'budgets': BUDGETS, 'seeds': SEEDS,
            'config': cfg_for(1, False, 13).to_dict(), 'torch': torch.__version__,
            'numpy': np.__version__, 'scipy': scipy.__version__, 'python': sys.version,
            'platform': platform.platform(), 'script_sha256': hash_file(__file__),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
        if (OUT / 'manifest.json').exists():
            old = json.loads((OUT / 'manifest.json').read_text())
            for key in ['script_sha256', 'torch', 'numpy', 'scipy', 'git_commit']:
                assert old[key] == manifest[key], f'Resume provenance changed: {key}'
        else:
            save_json(OUT / 'manifest.json', manifest)
        cache_data()
        data = {split: CachedData(split) for split in COUNTS}
        # A complete U/H pair at each budget makes progress interpretable.
        for seed in SEEDS:
            for budget in BUDGETS:
                for hankel in [False, True]:
                    train_one(seed, hankel, budget, data)
        save_json(OUT / 'status.json', {'state': 'complete', 'runs': 18})
    except Exception:
        save_json(OUT / 'status.json', {'state': 'failed', 'error': traceback.format_exc()})
        raise


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['smoke', 'verify-original', 'validate-pipeline', 'run', 'launch', 'status'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == 'smoke':
        smoke()
    elif args.command == 'verify-original':
        verify_original()
    elif args.command == 'validate-pipeline':
        validate_pipeline()
    elif args.command == 'run':
        run()
    elif args.command == 'launch':
        if (OUT / 'pid.json').exists():
            pid = json.loads((OUT / 'pid.json').read_text())['pid']
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'Process {pid} is already running.')
        with (OUT / 'run.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, env={**os.environ, 'PYTHONPATH': str(ROOT),
                    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        save_json(OUT / 'pid.json', {'pid': child.pid})
        print(f'Started PID {child.pid}; progress: {OUT / "status.json"}', flush=True)
    else:
        for name in ['pid.json', 'status.json', 'smoke.json']:
            if (OUT / name).exists():
                print(name, (OUT / name).read_text())


if __name__ == '__main__':
    main()
