"""Fixed full-alphabet BER for the fresh reliability-check comparison.

This module only evaluates cached channels. Checkpoint, policy, sparse-control,
scope and sampling decisions must be frozen before any test cell is generated.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from tcom_followups_20261017.ber_detector import detect, hypotheses, verify_detector
import tempfile
from . import study

REPO = Path(__file__).resolve().parents[1]
SOURCE = study.OUT
OUT = SOURCE / 'ber'
SNRS = tuple(study.SNRS)
RSRS = tuple(study.RSRS)
REGIMES = tuple(r for r, _ in study.REGIMES)
SEEDS = tuple(study.SEEDS)
CASES = study.TEST_COUNT
TEST_SEED = study.TEST_SEED
DATA_SEED = study.DATA_SEED
BOOTSTRAPS = 20000
BOOTSTRAP_SEED = study.BOOT_SEED
VECTORS = 64
BITS_PER_CASE = 3 * 2 * VECTORS
USED = tuple(study.NAMES)
NAMES = ('perfect_csi',) + USED
GROUPS = {name: tuple(members) for name, members in study.GROUPS.items()
          if all(member in USED for member in members)}
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        temporary = stream.name
    os.replace(temporary, path)


def cells():
    return [(r, ref, s) for r in REGIMES for ref in RSRS for s in SNRS]


def label(regime, rsr, snr):
    return f'{regime}_rsr{rsr:g}_snr{snr:g}'


def load(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files}


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def protocol_spec():
    expected = tuple(f'{family}_{seed}' for family in
                     ('receiver', 'prior', 'latest', 'adaptive40', 'global40') for seed in SEEDS)
    expected += ('classical40', 'em100', 'compact_sparse', 'algebraic20')
    assert USED == expected and len(GROUPS) == 5
    assert CASES == 128 and TEST_SEED == 2026102104 and DATA_SEED == 2026102105
    assert BOOTSTRAP_SEED == 2026102108 and len(cells()) == 153
    assert SNRS == (0., 5., 10., 15., 20., 22.5, 25., 27.5, 30., 32.5, 35., 37.5, 40., 42.5, 45., 50., 60.)
    assert RSRS == (5., 12., 20.) and REGIMES == ('L2', 'L4', 'random3to7')
    return dict(
        name='Fresh reliability-check BER', N=32, K=3, P=30,
        methods=USED, groups=GROUPS, cases_per_cell=CASES, cells=153,
        test_seed=TEST_SEED, data_seed=DATA_SEED, bootstrap_seed=BOOTSTRAP_SEED,
        sampling='All64 joint unit-power Gray-QPSK vectors once per channel;384bits/case,49152bits/cell/fit. Fixed sampling, no adaptive extra bits.',
        data='Fresh proper complex Gaussian data noise independent of fitting observations. SeedSequence([DATA_SEED,regime_index,reference_index,local_index]); standardized noise shared across SNR then scaled by sqrt(sigma2).',
        detector='Same exact64-hypothesis Rician magnitude-domain ML detector and data/reference/variance for every channel estimate; no coherent observations.',
        truth='Channel truth only generates shared data and the labeled perfect-CSI diagnostic. No truth input to estimation, checks, trigger selection or runtime routing.',
        duplicates='Only bitwise-identical entire channel arrays may reuse counts. Every reuse is recorded and audited. Estimator timings never reuse outputs.',
        primary='Receiver versus prior selective-EM and versus latest, all153conditions. NMSE and BER are four outcomes; Bonferroni4 intervals only there.',
        bootstrap='20000 multinomial128-anchor draws independently within each(regime,reference)block, in regime/reference order, shared across SNR/method/fit. Freshworldseeds domain-separate reference pathcounts too:1152anchorclusters. Fits average metrics, not estimates.',
        confirmation='The frozen A-selected receiver is evaluated even if C fails, labeled unconfirmed; the operational recommendation then remains the prior. Test results never change this decision.',
        zero_errors='Zero errors/collapsed empirical intervals do not establish zero BER, rare-event bounds, equivalence or noninferiority.',
        statistical_scope='Conditional on frozen trained fits/policy and simulation grid. Other references/scopes are descriptive.')


def selection_hash():
    _, digest = study.check_selection()
    return digest


def standard_noise(regime, rsr, local):
    rng = np.random.default_rng(np.random.SeedSequence([DATA_SEED, REGIMES.index(regime), RSRS.index(rsr), int(local)]))
    return (rng.standard_normal((32, VECTORS)) + 1j * rng.standard_normal((32, VECTORS))) / np.sqrt(2)


def data_for(channel, reference, sigma2, regime, rsr, local):
    x, bits = hypotheses(3)
    z = np.abs(channel @ x.T + np.asarray(reference).reshape(32, 1) + np.sqrt(sigma2) * standard_noise(regime, rsr, local))
    return z, bits


def evaluate(digest, available_only=False, reference=None):
    selected = selection_hash()
    complete = True
    if reference is not None and reference not in RSRS:
        raise ValueError('reference must be one of the frozen reference levels')
    selected_cells = [cell for cell in cells() if reference is None or cell[1] == reference]
    assert len(selected_cells) == (153 if reference is None else 51)
    for regime, rsr, snr in selected_cells:
        name = label(regime, rsr, snr)
        source = SOURCE / 'test/cells' / (name + '.npz')
        destination = OUT / 'cells' / (name + '.npz')
        if not source.exists() and available_only:
            complete = False
            continue
        if destination.exists():
            old = load(destination)
            assert str(old['protocol_sha256']) == digest and str(old['source_sha256']) == sha(source)
            assert str(old['selection_sha256']) == selected
            print('BER cached', name, flush=True)
            continue
        a = load(source)
        assert str(a['selection_sha256']) == selected and int(a['seed']) == TEST_SEED
        assert str(a['protocol_sha256']) == sha(SOURCE / 'protocol.json')
        assert float(a['snr']) == snr and float(a['rsr']) == rsr and str(a['regime']) == regime
        assert tuple(map(str, a['method_names'])) == USED
        assert a['z'].shape == (CASES, 32, 30) and a['s'].shape == (CASES, 3, 30)
        offset = {'L2': 100000, 'L4': 200000, 'random3to7': 0}[regime]
        np.testing.assert_array_equal(a['indices'], np.arange(CASES) + offset)
        np.testing.assert_array_equal(a['local_indices'], np.arange(CASES))
        truth = a['truth_g']
        assert truth.shape == (CASES, 32, 3)
        channels = np.concatenate((truth[None], a['estimates']))
        assert channels.shape == (len(NAMES), CASES, 32, 3) and np.isfinite(channels).all()
        reference, variance = a['b'][..., :1], a['sigma2']
        assert np.array_equal(a['b'], np.broadcast_to(reference, a['b'].shape))
        assert variance.shape == (CASES,) and np.all(variance > 0)
        assert np.isfinite(reference).all() and np.isfinite(variance).all()
        data = np.stack([data_for(truth[i], reference[i], float(variance[i]), regime, rsr, i)[0] for i in range(CASES)])
        bits = hypotheses(3)[1]
        datapath = OUT / 'data' / (name + '.npz')
        save(datapath, z=data, b=reference, sigma2=variance, indices=a['indices'], local_indices=a['local_indices'],
             transmitted_bits=bits, hypothesis_symbols=hypotheses(3)[0], protocol_sha256=digest)
        errors = np.zeros((len(NAMES), CASES, 3), dtype=np.int64)
        seconds = np.zeros((len(NAMES), CASES))
        reuse, seen = [], {}
        for mi, method in enumerate(NAMES):
            fingerprint = hashlib.sha256(channels[mi].tobytes()).hexdigest()
            previous = seen.get(fingerprint)
            if previous is not None:
                assert np.array_equal(channels[mi], channels[previous])
                errors[mi] = errors[previous]
                reuse.append(NAMES[previous])
                continue
            seen[fingerprint] = mi
            reuse.append('')
            for i in range(CASES):
                start = time.perf_counter()
                decoded = detect(data[i], channels[mi, i], reference[i], float(variance[i]))
                seconds[mi, i] = time.perf_counter() - start
                errors[mi, i] = np.count_nonzero(decoded != bits, axis=(0, 2))
        save(destination, method_names=np.array(NAMES), errors_per_user=errors,
             bits_per_user=np.full((CASES, 3), 2 * VECTORS, dtype=np.int64), duplicate_of=np.array(reuse),
             detector_wall_seconds=seconds, indices=a['indices'], local_indices=a['local_indices'], source_sha256=sha(source),
             data_sha256=sha(datapath), protocol_sha256=digest, selection_sha256=selected)
        print(json.dumps(dict(cell=name, unique_detector_methods=len(seen), methods=len(NAMES),
                              total_errors={method: int(errors[mi].sum()) for mi, method in enumerate(NAMES)})), flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=float, choices=RSRS)
    parser.add_argument('--available-only', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = protocol_spec()
    if args.preflight:
        print(json.dumps(dict(passed=True, protocol=spec, detector_checks=verify_detector(),
                              protocol_written=False, data_generated=False)), flush=True)
        return
    policy, selected = study.check_selection()
    digest = policy['protocol_sha256']
    assert digest == sha(SOURCE / 'protocol.json')
    complete = evaluate(digest, args.available_only, args.reference)
    suffix = '' if args.reference is None else f'_rsr{args.reference:g}'
    write_json(OUT / f'status{suffix}.json', dict(complete=complete, reference=args.reference,
               protocol_sha256=digest, selection_sha256=selected))


if __name__ == '__main__':
    main()
