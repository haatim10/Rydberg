"""Small in-memory preflight; never writes historical models or test data."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import io
import json
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from sahure.data import load_observations, split_observations
from rician_pilot_screen.noise import make_views
from . import train
from .protocol import ROOT, DATA, CONFIG, PARENT, initialize, sha, write


def assert_models_equal(first, second):
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def audit_training():
    """Validate the complete matched budget and validation-only selection."""
    manifest = initialize()
    folder = ROOT / 'checkpoints' / f'seed_{CONFIG["seed"]}'
    state = torch.load(folder / 'mixed_latest.pt', map_location='cpu', weights_only=False)
    best = torch.load(folder / 'mixed_best.pt', map_location='cpu', weights_only=False)
    if state['initial_parameters_sha256'] != manifest['initial_parameters_sha256']:
        raise RuntimeError('initialization no longer matches the historical baseline')
    if state['epoch'] != CONFIG['epochs']:
        raise RuntimeError('incomplete production training budget')
    history = state['history']
    if history != json.loads((folder / 'history.json').read_text()):
        raise RuntimeError('checkpoint and readable epoch histories disagree')
    if [row['epoch'] for row in history] != list(range(1, CONFIG['epochs'] + 1)):
        raise RuntimeError('missing or duplicated training epochs')
    batches = CONFIG['train_cases'] // CONFIG['batch_size']
    for epoch, row in enumerate(history):
        order_seed, augmentation_seed = CONFIG['seed'] + 20000 + epoch, CONFIG['seed'] + 10000 + epoch
        order = torch.randperm(CONFIG['train_cases'], generator=torch.Generator().manual_seed(order_seed))
        branches = (torch.arange(batches, dtype=torch.int64) + epoch) % 2
        if (row['order_sha256'] != train.tensor_hash(order)
                or row['branches_sha256'] != train.tensor_hash(branches)
                or row['order_seed'] != order_seed or row['augmentation_seed'] != augmentation_seed):
            raise RuntimeError('production data order or branch schedule mismatch')
        for arm in ('holdout', 'allpilot'):
            if row[arm + '_batches'] != batches // 2 or row[arm + '_examples'] != CONFIG['train_cases'] // 2:
                raise RuntimeError('production mixture is not exactly 50/50')
        if not np.isfinite([row['val_nll'], row['holdout_train_nll'], row['allpilot_train_nll']]).all():
            raise FloatingPointError('nonfinite completed-epoch loss')
    scores = [state['initial_validation_score']] + [row['val_nll'] for row in history]
    expected_best = int(np.argmin(scores))
    if (best['epoch'] != expected_best or state['best_epoch'] != expected_best
            or best['best_score'] != scores[expected_best]):
        raise RuntimeError('selected checkpoint is not the validation minimum')
    expected_steps = batches * CONFIG['epochs']
    if not state['optimizer']['state']:
        raise RuntimeError('optimizer state missing')
    for values in state['optimizer']['state'].values():
        if int(values['step']) != expected_steps:
            raise RuntimeError('incorrect total optimizer updates')
        for value in values.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError('nonfinite saved optimizer state')
    if sum(v.numel() for v in state['model'].values()) != 188:
        raise RuntimeError('architecture changed')
    if any(not torch.isfinite(v).all() for v in state['model'].values()):
        raise FloatingPointError('nonfinite saved model')
    report = dict(passed=True, epochs=len(history), selected_epoch=expected_best,
                  updates=expected_steps, batches_per_branch_per_epoch=batches // 2,
                  matched_initialization=True, matched_data_orders=True,
                  exact_branch_counts=True, validation_only_selection=True,
                  train_seconds=state['train_seconds'])
    write(ROOT / 'training_audit.json', report)
    return report


def run():
    started = time.monotonic()
    train.configure_cpu()
    suite = unittest.defaultTestLoader.loadTestsFromName('rician_pilot_screen.test_noise')
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise RuntimeError('Rician view tests failed')
    data = load_observations(DATA / 'train_observations.npz').batch(slice(0, 128))
    seed = CONFIG['seed']
    model, _ = train.new_model(seed)
    expected_hash = json.loads((PARENT / 'baseline_replay.json').read_text())['initial_parameters_sha256'][str(seed)]
    if train.parameters_hash(model) != expected_hash:
        raise RuntimeError('initial parameters do not match historical seed9101')
    records = []
    for epoch in (0, 1):
        split = split_observations(data, seed + 10000 + epoch)
        augmented = make_views(data, 1., seed=seed + 10000 + epoch)
        for view in augmented:
            torch.testing.assert_close(view.sigma2, 2 * data.sigma2, rtol=0, atol=0)
            if view.s.shape[-1] != 30:
                raise RuntimeError('all-pilot branch lost pilot columns')
        if (split[0].s.shape[-1], split[1].s.shape[-1]) != (21, 9):
            raise RuntimeError('holdout mask lengths changed')
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed + 20000 + epoch))
        calls = []
        def inspect_step(_model, _optimizer, context, target):
            batch = len(calls)
            branch = (batch + epoch) % 2
            expected_views = split if branch == 0 else augmented
            ids = order[batch * 32:(batch + 1) * 32]
            for actual, expected in zip((context, target), expected_views):
                for x, y in zip(actual.args(), expected.batch(ids).args()):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
            calls.append(branch)
            return 1.25 if branch == 0 else 2.5
        with patch.object(train, 'step', side_effect=inspect_step):
            row = train.train_epoch(model, None, data, epoch, seed)
        if calls != [(b + epoch) % 2 for b in range(4)]:
            raise RuntimeError('wrong branch order or wrong update count')
        if row['holdout_train_nll'] != 1.25 or row['allpilot_train_nll'] != 2.5:
            raise RuntimeError('losses not separately averaged')
        if row['order_sha256'] != train.tensor_hash(order):
            raise RuntimeError('data-order hash mismatch')
        records.append(dict(epoch=epoch + 1, branches=calls, order_sha256=row['order_sha256']))
    # Exercise actual finite gradients for both branches and exact optimizer resume.
    small = data.batch(slice(0, 64))
    continuous, _ = train.new_model(seed)
    optimizer = torch.optim.Adam(continuous.parameters(), lr=CONFIG['lr'])
    first = train.train_epoch(continuous, optimizer, small, 0, seed)
    stream = io.BytesIO()
    torch.save(dict(model=continuous.state_dict(), optimizer=optimizer.state_dict()), stream)
    stream.seek(0)
    state = torch.load(stream, map_location='cpu', weights_only=False)
    resumed, _ = train.new_model(seed)
    resumed.load_state_dict(state['model'])
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=CONFIG['lr'])
    resumed_optimizer.load_state_dict(state['optimizer'])
    second = train.train_epoch(continuous, optimizer, small, 1, seed)
    replayed = train.train_epoch(resumed, resumed_optimizer, small, 1, seed)
    assert_models_equal(continuous, resumed)
    if second != replayed:
        raise RuntimeError('resumed branch losses or hashes differ')
    for a, b in zip(optimizer.state.values(), resumed_optimizer.state.values()):
        for key in a:
            if torch.is_tensor(a[key]):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            elif a[key] != b[key]:
                raise RuntimeError('resumed optimizer state differs')
    for epoch in range(CONFIG['epochs']):
        branches = (np.arange(CONFIG['train_cases'] // CONFIG['batch_size']) + epoch) % 2
        if np.bincount(branches).tolist() != [256, 256]:
            raise RuntimeError('production schedule is not exactly half/half')
    # Only now freeze the fully implemented sources, after all tests have passed.
    record = initialize()
    if record['initial_parameters_sha256'] != expected_hash:
        raise RuntimeError('manifest initial-parameter hash mismatch')
    report = dict(passed=True, seconds=time.monotonic()-started,
                  manifest_sha256=sha(ROOT / 'manifest.json'),
                  initial_parameters_sha256=expected_hash,
                  branch_checks=records, production_batches_per_branch=256,
                  exact_optimizer_resume=True, finite_gradient_checks=True,
                  actual_gradient_epoch_records=[first, second],
                  historical_hashes_checked=len(record['sources']),
                  note='No new production training or fresh evaluation cases generated by preflight.')
    write(ROOT / 'preflight.json', report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    run()
