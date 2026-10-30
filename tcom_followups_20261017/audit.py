"""Cross-study pairing, archived-output consistency, and frozen-input audit."""
import json
import numpy as np
from .common import ROOT, REPO, sha, write_json


def main():
    inputs = json.loads((ROOT / 'inputs.json').read_text())
    for path, digest in inputs['hashes'].items():
        if sha(REPO / path) != digest:
            raise RuntimeError('Previously frozen input changed: ' + path)
    checked, overlaps = 0, 0
    maximum = 0.
    for regime, sparse_regime in [('L2', 'L2'), ('L4', 'L4'), ('random3to7', 'random3-7')]:
        for snr in (0, 5, 10, 15, 20):
            with np.load(ROOT / f'sparse/test_snr{snr}_{sparse_regime}.npz') as s, \
                 np.load(ROOT / f'ber/cells/{regime}_snr{snr}.npz') as b, \
                 np.load(ROOT / f'ber/truth/{regime}_snr{snr}.npz') as truth, \
                 np.load(ROOT / f'ber/observations/{regime}_snr{snr}_pilots.npz') as obs:
                np.testing.assert_array_equal(s['truth_g'], truth['g'])
                for key in ('z','s','b','sigma2'):
                    np.testing.assert_array_equal(s[key], obs[key])
                for name in ('learned_9101', 'learned_9102', 'learned_9103', 'em100', 'optshrink'):
                    idx = list(b['names']).index(name)
                    delta = float(np.max(np.abs(s[name + '_g'] - b['estimates'][idx])))
                    maximum = max(maximum, delta)
                    np.testing.assert_allclose(s[name + '_g'], b['estimates'][idx], rtol=1e-12, atol=1e-12)
                    e = np.abs(s[name + '_g'] - truth['g']) ** 2
                    np.testing.assert_allclose(e.sum((-2,-1)), b['squared_channel_error'][idx], rtol=1e-12, atol=1e-12)
                checked += 1
                if snr == 10:
                    with np.load(ROOT / f'high_snr/anchors/{regime}_rsr12.npz') as h:
                        np.testing.assert_array_equal(h['g'], truth['g'])
                        np.testing.assert_array_equal(h['anchor_z'], obs['z'])
                        overlaps += len(truth['g'])
    write_json(ROOT / 'audit.json', dict(passed=True, original_input_hashes_unchanged=True,
        sparse_BER_identical_observations_and_truth_cells=checked,
        sparse_BER_identical_frozen_estimates_max_difference=maximum,
        overlap_with_high_snr_anchor_worlds=overlaps,
        total_distinct_pilot_worlds=15 * 64 + 9 * 64 - overlaps,
        meaning='Agreement checks compare independently executed pipelines. Sparse and BER share960 worlds; '
                'the high-SNR sweep uses576 coupled anchors,192 of which overlap at10dB/RSR12. '
                'Repeated SNRs and methods are not independent Monte Carlo worlds.'))
    print('Cross-study pairing and unchanged frozen inputs verified.', flush=True)


if __name__ == '__main__':
    main()
