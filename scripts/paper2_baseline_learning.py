"""Isolated paired baseline learning checks. Never reads a final test set."""
import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import paper2_long_training as engine
import paper2_reproduction as repro
from paper2_original_channel import PaperChannelData, draw_channel, rng_for
from rydberg_sim.forward import exact_forward
from trackD_urformer.dataset import collate
from trackD_urformer.stage2 import select_epoch
from trackD_urformer.torch_forward import em_gs_layer

OUT = ROOT/'results/paper2_baseline_learning'
SEED = 2026092401
ARMS = ('physical', 'normalized')
STUDIES = {
    'memorization': {'train': 128, 'val': 256, 'epochs': 100, 'start': 20000000, 'snr': 60.},
    'development': {'train': 4096, 'val': 512, 'epochs': 12, 'start': 22000000, 'snr': None},
}


def cfg():
    base = repro.cfg_for(1)
    return replace(base, train=replace(base.train, seed=SEED))


class LearningData(PaperChannelData):
    def __init__(self, study, split, spec=None):
        if split not in ('train', 'val'):
            raise ValueError('Development train/val only; no final test allowed')
        self.spec = STUDIES[study] if spec is None else spec
        super().__init__(split, 'literal_independent', self.spec[split])
        self.start = self.spec['start']+(1000000 if split == 'val' else 0)

    def raw(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        trial = self.start+index
        snr = self.spec['snr']
        if snr is None:
            snr = float(np.round(rng_for(trial, 5).uniform(-10, 20), 3))
        variance = self.signal_power/10**(snr/10)
        g = draw_channel(trial, self.mode)
        return trial, snr, variance, g, exact_forward(g, self.S, self.B, variance, rng_noise=rng_for(trial, 6))


class Tensors(Dataset):
    def __init__(self, arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays['trial'])

    def __getitem__(self, index):
        return {k:v[index] for k,v in self.arrays.items()}


def normalize_inputs(g, z, s, b, variance, scale):
    return g/scale, z/scale, s, b/scale, variance/scale**2


class Units(torch.nn.Module):
    def __init__(self, model, scale):
        super().__init__()
        self.model, self.scale = model, scale
        for layer in model.layers:
            layer.eps /= scale

    def forward(self, g, z, s, b, variance):
        return self.scale*self.model(*normalize_inputs(g, z, s, b, variance, self.scale))


def build(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    base, meta = repro.build_model(cfg())
    return Units(base, 1. if arm == 'physical' else np.sqrt(32.)), meta


def weight_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def loader(data, epoch=None):
    return DataLoader(data, batch_size=32, shuffle=epoch is not None, collate_fn=collate,
        generator=None if epoch is None else torch.Generator().manual_seed(SEED+epoch))


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    values = []
    for b in loader(data):
        pred = model(*(b[k] for k in ('G0', 'Z', 'S', 'B', 'sigma2')))
        values.extend((abs(pred-b['G_true']).square().sum((1, 2))/abs(b['G_true']).square().sum((1, 2))).tolist())
    values = np.asarray(values)
    assert np.isfinite(values).all() and (values >= 0).all()
    return values


def cache(study, spec):
    folder = OUT/study/'data'
    folder.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for split in ('train', 'val'):
        path = folder/f'{split}.pt'
        meta_path = folder/f'{split}.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert meta['spec'] == spec and engine.hash_file(path) == meta['sha256']
            arrays = torch.load(path, weights_only=True)
        else:
            original = LearningData(study, split, spec)
            arrays = collate([original[i] for i in range(len(original))])
            engine.save_torch(path, arrays)
            engine.save_json(meta_path, {'spec':spec, 'sha256':engine.hash_file(path), 'count':len(original)})
        datasets[split] = Tensors(arrays)
    assert not np.intersect1d(datasets['train'].arrays['trial'], datasets['val'].arrays['trial']).size
    return datasets


def train_arm(study, arm, spec, data, stop_after=None):
    folder = OUT/study/arm
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'result.json').exists():
        return json.loads((folder/'result.json').read_text())
    model, meta = build(arm)
    initial_hash = weight_hash(model)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spec['epochs'])
    checkpoint = folder/'checkpoint.pt'
    start, history, validations = 0, [], []
    initial = None
    if checkpoint.exists():
        b = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert b['spec'] == spec and b['arm'] == arm and b['initial_hash'] == initial_hash
        model.load_state_dict(b['model']); opt.load_state_dict(b['optimizer']); sched.load_state_dict(b['scheduler'])
        torch.set_rng_state(b['torch_rng'])
        start, history, validations, initial = b['epoch']+1, b['history'], b['validations'], b['initial']
    else:
        initial = {s:float(10*np.log10(evaluate(model, d).mean())) for s,d in data.items()}
    for epoch in range(start, spec['epochs']):
        begin = time.monotonic(); model.train(); total = 0.; order = hashlib.sha256(); max_grad = 0.
        batches = loader(data['train'], epoch)
        for index, batch in enumerate(batches):
            order.update(batch['trial'].numpy().tobytes())
            loss, grad = repro.train_step(model, opt, batch, cfg())
            total += loss; max_grad = max(max_grad, grad)
            if (index+1)%32 == 0:
                engine.save_json(OUT/'status.json', {'state':'training', 'study':study, 'arm':arm,
                    'epoch':epoch+1, 'epochs':spec['epochs'], 'batch':index+1, 'batches':len(batches)})
        sched.step()
        val = evaluate(model, data['val']); validations.append(val)
        rec = {'epoch':epoch+1, 'train_online_nmse':total/len(batches),
            'validation_nmse_db':float(10*np.log10(val.mean())), 'lr':sched.get_last_lr()[0],
            'max_gradient_norm':max_grad, 'data_order_sha256':order.hexdigest(), 'seconds':time.monotonic()-begin}
        if study == 'memorization':
            rec['training_replay_nmse_db'] = float(10*np.log10(evaluate(model, data['train']).mean()))
        rec['seconds'] = time.monotonic()-begin
        history.append(rec)
        engine.save_torch(folder/f'epoch_{epoch+1:03d}.pt', {'model':model.state_dict()})
        engine.save_torch(checkpoint, {'model':model.state_dict(), 'optimizer':opt.state_dict(),
            'scheduler':sched.state_dict(), 'torch_rng':torch.get_rng_state(), 'epoch':epoch,
            'spec':spec, 'arm':arm, 'initial_hash':initial_hash, 'initial':initial,
            'history':history, 'validations':validations})
        engine.save_json(folder/'history.json', history)
        engine.save_json(OUT/'status.json', {'state':'training', 'study':study, 'arm':arm,
            'epoch':epoch+1, 'epochs':spec['epochs'], 'epoch_complete':True, **rec})
        print(json.dumps({'study':study, 'arm':arm, **rec}), flush=True)
        with (OUT/'run.log').open('a') as log:
            log.write(json.dumps({'study':study, 'arm':arm, **rec})+'\n')
        if stop_after is not None and epoch+1 >= stop_after:
            return None
    curve = np.array([r['validation_nmse_db'] for r in history])
    chosen, selection = select_epoch(curve, validations[int(curve.argmin())], 'one_se')
    final_training = float(10*np.log10(evaluate(model, data['train']).mean()))
    result = {'study':study, 'arm':arm, 'spec':spec, 'initial_hash':initial_hash, 'initial_nmse_db':initial,
        'final_training_nmse_db':final_training, 'final_validation_nmse_db':float(curve[-1]),
        'chosen_epoch':chosen+1, 'selection':selection, 'history':history, 'model_meta':meta,
        'validation_only':True, 'seed':SEED}
    np.savez(folder/'validation_errors.npz', errors=np.asarray(validations),
        trial=data['val'].arrays['trial'].numpy(), snr_db=data['val'].arrays['snr_db'].numpy())
    engine.save_json(folder/'result.json', result)
    return result


class Classical(torch.nn.Module):
    def __init__(self, steps):
        super().__init__(); self.steps = steps

    def forward(self, g, z, s, b, sigma):
        for _ in range(self.steps):
            g = em_gs_layer(g, z, s, b, sigma, eps=1e-8)
        return g


def summarize(study, data):
    folder = OUT/study
    records = [json.loads((folder/a/'result.json').read_text()) for a in ARMS]
    assert records[0]['initial_hash'] == records[1]['initial_hash']
    assert [r['data_order_sha256'] for r in records[0]['history']] == [r['data_order_sha256'] for r in records[1]['history']]
    files = [np.load(folder/a/'validation_errors.npz') for a in ARMS]
    for k in ('trial', 'snr_db'):
        np.testing.assert_array_equal(files[0][k], files[1][k])
    masks = {'all':np.ones(len(data['val']), dtype=bool)}
    snr = files[0]['snr_db']
    if study == 'development':
        masks.update({f'[{lo},{lo+5})':(snr >= lo)&(snr < lo+5) for lo in range(-10, 20, 5)})
    scores = {}
    for label in ('final', 'selected'):
        errors = [f['errors'][-1 if label == 'final' else r['chosen_epoch']-1] for f,r in zip(files, records)]
        scores[label] = {group:{'n':int(m.sum()), 'physical_db':float(10*np.log10(errors[0][m].mean())),
            'normalized_db':float(10*np.log10(errors[1][m].mean())),
            'normalization_gain_db':float(10*np.log10(errors[0][m].mean()/errors[1][m].mean()))} for group,m in masks.items()}
    classical = {}
    for steps in (10, 100):
        values = evaluate(Classical(steps), data['val'])
        np.savez(folder/f'emgs_{steps}_validation.npz', errors=values, trial=files[0]['trial'], snr_db=snr)
        classical[str(steps)] = {group:float(10*np.log10(values[m].mean())) for group,m in masks.items()}
    result = {'study':study, 'matched_initial_weights':True, 'matched_data_order':True,
        'scores':scores, 'classical_validation_nmse_db':classical,
        'training_diagnostics':{r['arm']:{'initial_db':r['initial_nmse_db']['train'],
            'final_db':r['final_training_nmse_db'], 'improvement_db':r['initial_nmse_db']['train']-r['final_training_nmse_db']} for r in records},
        'limitations':'Single-seed small-data development diagnostic. No final test. Not a published reproduction.'}
    engine.save_json(folder/'summary.json', result)
    return result


def provenance():
    paths = [Path(__file__), ROOT/'reports/paper2_baseline_learning_plan.md',
             ROOT/'scripts/paper2_reproduction.py', ROOT/'scripts/paper2_original_channel.py', ROOT/'scripts/paper2_long_training.py']
    paths += sorted((ROOT/'trackD_urformer').glob('*.py'))+sorted((ROOT/'rydberg_sim').glob('*.py'))
    return {str(p.relative_to(ROOT)):engine.hash_file(p) for p in paths}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1)
        manifest = {'source_hashes':provenance(), 'studies':STUDIES, 'arms':ARMS,
            'seed':SEED, 'torch':torch.__version__, 'numpy':np.__version__}
        canonical = json.loads(json.dumps(manifest))
        if (OUT/'manifest.json').exists():
            assert json.loads((OUT/'manifest.json').read_text()) == canonical
        else:
            engine.save_json(OUT/'manifest.json', manifest)
        engine.save_json(OUT/'pid.json', {'pid':os.getpid()})
        try:
            summaries = []
            for study, spec in STUDIES.items():
                engine.save_json(OUT/'status.json', {'state':'preparing_data', 'study':study})
                data = cache(study, spec)
                for arm in ARMS:
                    train_arm(study, arm, spec, data)
                summaries.append(summarize(study, data))
            engine.save_json(OUT/'summary.json', {'studies':summaries, 'validation_only':True})
            engine.save_json(OUT/'status.json', {'state':'complete', 'runs':4, 'validation_only':True})
        except Exception:
            engine.save_json(OUT/'status.json', {'state':'failed', 'error':traceback.format_exc()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run'])
    parser.parse_args()
    run()
