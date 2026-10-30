"""Exploratory B-proposal BER, using the exact frozen parent data."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import json
import time
import numpy as np
import torch
from tcom_followups_20261017.ber_detector import detect, hypotheses, verify_detector
from tcom_followups_20261017.common import sha, write_json
from . import study

OUT = study.OUT / 'ber'
SOURCE = study.PARENT_OUT
NAMES = tuple(study.NAMES)
COUNT = study.TEST_COUNT
SNRS, RSRS, REGIMES = study.SNRS, study.RSRS, study.REGIMES
VECTORS = 64
BITS_PER_CASE = 384


def protocol_spec():
    assert NAMES == ('proposal_9101', 'proposal_9102', 'proposal_9103')
    assert COUNT == 128 and study.DATA_SEED == 2026102055 and study.BOOT_SEED == 2026102058
    assert len(SNRS) * len(RSRS) * len(REGIMES) == 153
    return dict(
        methods=NAMES, cases_per_cell=COUNT, cells=153, vectors_per_channel=64,
        bits_per_user_per_case=128, primary=False, new_selection=False,
        policy='Exactly the already-frozen B proposed_steps, with the same A crossover. '
               'All 21 nonzero proposals failed C confirmation. This is exploratory.',
        data='Read the exact parent ber/data archive; verify parent cell/data hashes and '
             'known reference/variance against the parent channel archive. Generate no data.',
        detector='Unchanged exact 64-hypothesis magnitude-only Rician ML detector. '
                 'Decode all three proposal fits independently; no prediction/count reuse.',
        intervals='Descriptive pointwise coupled-channel bootstrap only. No new primary '
                  'or multiplicity-adjusted claims; no equivalence from zero errors.',
        parent_primary='Unchanged C-confirmed hybrid, with zero extra EM everywhere.')


def cells(reference=None):
    assert reference is None or reference in RSRS
    return [(r, ref, snr) for r, _ in REGIMES for ref in RSRS for snr in SNRS
            if reference is None or ref == reference]


def evaluate(reference=None, available_only=False):
    record, digest = study.check_selection()
    complete = True
    for regime, ref, snr in cells(reference):
        name = study.cellname(regime, ref, snr)
        source_path = study.OUT / 'test/cells' / f'{name}.npz'
        parent_path = SOURCE / 'test/cells' / f'{name}.npz'
        parent_ber = SOURCE / 'ber/cells' / f'{name}.npz'
        data_path = SOURCE / 'ber/data' / f'{name}.npz'
        destination = OUT / 'cells' / f'{name}.npz'
        paths = (source_path, parent_path, parent_ber, data_path)
        if available_only and not all(p.exists() for p in paths):
            complete = False
            continue
        hashes = dict(source_sha256=sha(source_path), parent_source_sha256=sha(parent_path),
                      parent_ber_sha256=sha(parent_ber), parent_data_sha256=sha(data_path))
        if destination.exists():
            old = study.load(destination)
            for key, value in hashes.items():
                assert str(old[key]) == value
            assert str(old['selection_sha256']) == str(old['protocol_sha256']) == digest
            assert str(old['parent_selection_sha256']) == record['parent_selection_sha256']
            continue
        proposal, parent, baseline, data = map(study.load, paths)
        assert tuple(map(str, proposal['method_names'])) == NAMES
        assert str(proposal['selection_sha256']) == str(proposal['protocol_sha256']) == digest
        assert str(proposal['parent_selection_sha256']) == record['parent_selection_sha256']
        for key, value in (('parent_source_sha256', hashes['parent_source_sha256']),
                           ('parent_ber_cell_sha256', hashes['parent_ber_sha256']),
                           ('parent_ber_data_sha256', hashes['parent_data_sha256'])):
            assert str(proposal[key]) == value
        assert str(parent['selection_sha256']) == str(baseline['selection_sha256']) == record['parent_selection_sha256']
        assert str(baseline['source_sha256']) == hashes['parent_source_sha256']
        assert str(baseline['data_sha256']) == hashes['parent_data_sha256']
        assert float(parent['snr']) == float(proposal['snr']) == snr
        assert float(parent['rsr']) == float(proposal['rsr']) == ref
        assert str(parent['regime']) == str(proposal['regime']) == regime
        for key in ('indices', 'local_indices'):
            for item in (proposal, baseline, data):
                np.testing.assert_array_equal(item[key], parent[key])
        np.testing.assert_array_equal(parent['local_indices'], np.arange(COUNT))
        np.testing.assert_array_equal(data['b'], parent['b'][..., :1])
        np.testing.assert_array_equal(data['sigma2'], parent['sigma2'])
        symbols, bits = hypotheses(3)
        np.testing.assert_array_equal(data['hypothesis_symbols'], symbols)
        np.testing.assert_array_equal(data['transmitted_bits'], bits)
        assert data['z'].shape == (COUNT, 32, 64) and np.isfinite(data['z']).all()
        estimates = proposal['estimates']
        assert estimates.shape == (3, COUNT, 32, 3) and np.isfinite(estimates).all()
        errors = np.zeros((3, COUNT, 3), dtype=np.int64)
        seconds = np.zeros((3, COUNT))
        for mi, method in enumerate(NAMES):
            for case in range(COUNT):
                start = time.perf_counter()
                decoded = detect(data['z'][case], estimates[mi, case], data['b'][case],
                                 float(data['sigma2'][case]))
                seconds[mi, case] = time.perf_counter() - start
                errors[mi, case] = np.count_nonzero(decoded != bits, axis=(0, 2))
        study.save(destination, method_names=np.array(NAMES), errors_per_user=errors,
                   bits_per_user=np.full((COUNT, 3), 128, dtype=np.int64),
                   detector_wall_seconds=seconds, indices=parent['indices'],
                   local_indices=parent['local_indices'], **hashes,
                   selection_sha256=np.array(digest), protocol_sha256=np.array(digest),
                   parent_selection_sha256=np.array(record['parent_selection_sha256']))
        print(json.dumps(dict(cell=name, exploratory=True,
                              errors={n: int(errors[i].sum()) for i, n in enumerate(NAMES)})), flush=True)
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
    complete = evaluate(args.reference, args.available_only)
    suffix = '' if args.reference is None else f'_rsr{args.reference:g}'
    write_json(OUT / f'status{suffix}.json', dict(complete=complete, reference=args.reference))


if __name__ == '__main__':
    main()
