"""Independent read-only replay; never changes training or primary results."""
import argparse, platform, sys
import numpy as np
import torch
from matched_supervision_20261023.common import OUT, OLD, pilots, power, sha, write, frozen
from matched_supervision_20261023.models import frontend


@torch.no_grad()
def data_audit():
    frozen()
    records = []
    for split in ('train', 'val'):
        folder = OUT / 'data' / split
        z = np.load(folder / 'observations/z.npy', mmap_mode='r')
        v = np.load(folder / 'observations/v.npy', mmap_mode='r')
        cached = np.load(folder / 'observations/initial.npy', mmap_mode='r')
        labels = np.load(folder / 'labels/h.npy', mmap_mode='r')
        with np.load(OLD / 'data' / f'{split}.npz') as old:
            assert np.array_equal(labels, old['h'])
            old_h, old_w, old_snr = old['h'], old['w'][..., :29], old['snr']
        s = torch.tensor(pilots(29), dtype=torch.complex64)
        vv = (power(20) / 10 ** (torch.from_numpy(old_snr).double()/10)).float()
        expected_z = (torch.from_numpy(old_h) @ s + np.sqrt(10*power(20))
                      + vv[:, None, None].sqrt()*torch.from_numpy(old_w)).abs().numpy()
        np.testing.assert_allclose(z, expected_z, atol=2e-5, rtol=2e-6)
        np.testing.assert_array_equal(v, vv.numpy())
        maximum = 0.
        for start in range(0, len(z), 64):
            sl = slice(start, start+64)
            context = torch.from_numpy(np.array(z[sl, ..., :20]))
            pp = s[None, :, :20].expand(len(context), -1, -1)
            bb = torch.full_like(context, np.sqrt(10*power(20))).to(torch.complex64)
            estimate = frontend(context, pp, bb, vv[sl])[0].numpy()
            maximum = max(maximum, float(np.max(np.abs(estimate-cached[sl]))))
            np.testing.assert_allclose(estimate, cached[sl], atol=1e-5, rtol=1e-5)
        records.append(dict(split=split, examples=len(z), label_order_exact=True,
                            observations_replayed=True, initializer_context_only=True,
                            initializer_max_absolute_error=maximum))
    import scipy
    result = dict(passed=True, records=records, protocol_sha256=sha(OUT/'protocol.json'),
                  environment=dict(python=sys.version, platform=platform.platform(),
                                   machine=platform.machine(), numpy=np.__version__,
                                   torch=torch.__version__, scipy=scipy.__version__))
    write(OUT/'additional_data_audit.json', result)
    print(result, flush=True)


def score_audit():
    frozen()
    with np.load(OUT/'data/test_raw.npz') as raw:
        h = raw['h'].astype(np.complex128)
    energy = np.sum(np.abs(h)**2, axis=(-2,-1))
    records = []
    for path in sorted((OUT/'test').glob('*.npz')):
        maximum = 0.
        with np.load(path) as scores, np.load(OUT/'predictions'/path.name) as predictions:
            np.testing.assert_allclose(scores['truth_energy'], energy, atol=1e-9, rtol=1e-13)
            for name in predictions.files:
                independent = np.sum(np.abs(predictions[name]-h)**2, axis=(-2,-1))/energy
                maximum = max(maximum, float(np.max(np.abs(independent-scores[name]))))
                np.testing.assert_allclose(independent, scores[name], atol=1e-12, rtol=1e-12)
            records.append(dict(cell=path.name, methods=len(predictions.files),
                                max_per_channel_ratio_error=maximum))
    assert len(records)==12
    write(OUT/'additional_score_audit.json', dict(passed=True, cells=records,
          scoring='Independent NumPy recomputation from every saved prediction; truth used only for scoring.'))
    print('Independent NMSE replay passed for all 12 cells.', flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('phase', choices=('data','scores'))
    phase=parser.parse_args().phase
    (data_audit if phase=='data' else score_audit)()
