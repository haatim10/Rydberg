"""Read-only, validation-only interventions on the frozen reconstructed baseline.

No training, test-set scoring, or mutation of original outputs. Interventions
are diagnostic probes of existing weights, not proposed/reproduced estimators.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import paper2_reproduction as repro
from trackD_urformer.dataset import collate
from trackD_urformer.torch_forward import em_gs_layer, gs_layer, least_squares_G

ROOT = Path(__file__).resolve().parents[1]


class Validation(Dataset):
    def __init__(self):
        p = ROOT / 'results/paper2_reproduction/data/val'
        self.arrays = {f.stem: np.load(f, mmap_mode='c') for f in p.glob('*.npy')}

    def __len__(self):
        return len(self.arrays['trial'])

    def __getitem__(self, index):
        return {k: torch.as_tensor(v[index]) for k, v in self.arrays.items()}


def score(errors, snr):
    masks = {'all': np.ones(len(snr), dtype=bool)}
    masks.update({f'[{lo},{lo+5})': (snr >= lo) & (snr < lo+5) for lo in range(-10, 20, 5)})
    return {k: {'n': int(m.sum()), 'nmse_db': float(10*np.log10(np.mean(errors[m])))} for k, m in masks.items()}


@torch.no_grad()
def evaluate(model, data):
    errors = []
    for b in DataLoader(data, batch_size=32, collate_fn=collate):
        pred = model(b['G0'], b['Z'], b['S'], b['B'], b['sigma2'])
        errors.extend((abs(pred-b['G_true']).square().sum((1, 2))/abs(b['G_true']).square().sum((1, 2))).tolist())
    e = np.asarray(errors)
    assert np.isfinite(e).all()
    return score(e, data.arrays['snr_db'])


class Classic(torch.nn.Module):
    def __init__(self, count, em):
        super().__init__()
        self.count, self.em = count, em

    def forward(self, g, z, s, b, sigma):
        for _ in range(self.count):
            g = em_gs_layer(g, z, s, b, sigma, eps=1e-8) if self.em else gs_layer(g, z, s, b, eps=1e-8)
        return g


def run(seeds):
    torch.set_num_threads(1)
    data = Validation()
    result = {'scope': 'Existing validation data only. Frozen-weight interventions; not independently trained methods.',
              'classical': {}, 'neural': [], 'algebra': {}}
    # Check exact LS against independent pseudoinverse, and physically equivalent units.
    b = collate([data[i] for i in range(32)])
    g, z, s, ref, sigma = [b[k] for k in ('G0', 'Z', 'S', 'B', 'sigma2')]
    r = (g@s+ref)*.37-ref
    err = float(abs(least_squares_G(r, s)-r@torch.linalg.pinv(s)).max())
    assert err < 1e-4
    scale = np.sqrt(32)
    x = em_gs_layer(g, z, s, ref, sigma, eps=1e-8)
    y = scale*em_gs_layer(g/scale, z/scale, s, ref/scale, sigma/32, eps=1e-8/scale)
    scale_err = float(abs(x-y).max())
    assert scale_err < 1e-4
    true = b['G_true']
    noiseless = abs(true@s+ref)
    identity_err = float(abs(gs_layer(true, noiseless, s, ref, eps=1e-8)-true).max())
    assert identity_err < 1e-4
    result['algebra'] = {'LS_max_abs_error': err, 'EMGS_unit_rescaling_max_abs_error': scale_err,
                         'noiseless_true_channel_GS_fixed_point_max_error': identity_err}
    for name, count, em in [('GS10', 10, False), ('EMGS10', 10, True), ('GS100', 100, False), ('EMGS100', 100, True)]:
        result['classical'][name] = evaluate(Classic(count, em), data)
        print(json.dumps({'classical': name, 'scores': result['classical'][name]}), flush=True)
    for seed in seeds:
        folder = ROOT / f'results/paper2_reproduction/seed{seed}_U_50ep'
        record = json.loads((folder/'result.json').read_text())
        for label, epoch in [('selected', record['chosen_epoch_1based']), ('final', 50)]:
            for intervention in ('original', 'GS_filter', 'exact_Bessel_learned_gate', 'no_Transformer', 'exact_EMGS_no_Transformer'):
                model, _ = repro.build_model(repro.cfg_for(seed))
                model.load_state_dict(torch.load(folder/f'epoch_{epoch:03d}.pt', map_location='cpu', weights_only=True)['model'])
                model.eval()
                for layer in model.layers:
                    if intervention == 'GS_filter':
                        layer._override_alpha = 0.
                    if intervention in ('exact_Bessel_learned_gate', 'exact_EMGS_no_Transformer'):
                        layer._override_filter = 'exact_bessel'
                    if intervention in ('no_Transformer', 'exact_EMGS_no_Transformer'):
                        layer._disable_residual = True
                    if intervention == 'exact_EMGS_no_Transformer':
                        layer._override_alpha = 1.
                scores = evaluate(model, data)
                if intervention == 'original':
                    np.testing.assert_allclose(scores['all']['nmse_db'], record['history'][epoch-1]['validation_db'], atol=1e-7, rtol=0)
                if intervention == 'exact_EMGS_no_Transformer':
                    for k in scores:
                        np.testing.assert_allclose(scores[k]['nmse_db'], result['classical']['EMGS10'][k]['nmse_db'], atol=1e-6, rtol=0)
                row = {'seed': seed, 'checkpoint': label, 'epoch': epoch, 'intervention': intervention, 'scores': scores}
                result['neural'].append(row)
                print(json.dumps(row), flush=True)
    print('FINAL_RESULT='+json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[1])
    run(parser.parse_args().seeds)
