"""Execution-only reference partitions around the immutable BER evaluator.

The scientific constants, data streams, detector, sampling and analysis remain
in ber.py. Only evaluate() sees a temporary disjoint subset of cells().
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/tcom_filter_continuation_20261019/ber'
REFERENCES = (5., 12., 20.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def worker(reference):
    import numpy as np
    from . import ber
    ber.torch.set_num_threads(1)
    ber.torch.set_num_interop_threads(1)
    protocol = ber.freeze()  # Full-grid validation before any partition.
    original = ber.cells
    full = original()
    assert len(full) == 153 and ber.RSRS == REFERENCES
    subset = [cell for cell in full if cell[1] == reference]
    assert len(subset) == 51
    constants = (ber.SNRS, ber.RSRS, ber.REGIMES, ber.DATA_SEED, ber.TEST_SEED, ber.BOOTSTRAP_SEED)
    checks = []
    try:
        ber.cells = lambda: list(subset)
        # Compare a full data cell after partitioning with its already saved
        # serial or earlier-pass version; RNG indices remain global.
        for cell in subset:
            name = ber.label(*cell)
            datapath = OUT / 'data' / (name + '.npz')
            cellpath = OUT / 'cells' / (name + '.npz')
            if not (datapath.exists() and cellpath.exists()):
                continue
            source = ber.load(ber.SOURCE / 'test/cells' / (name + '.npz'))
            data = ber.load(datapath)
            regenerated = np.stack([ber.data_for(source['truth_g'][i], source['b'][i, :, :1],
                                     float(source['sigma2'][i]), cell[0], reference, i)[0] for i in range(ber.CASES)])
            np.testing.assert_array_equal(regenerated, data['z'])
            checks.append(dict(cell=name, cases=ber.CASES, data_sha256=digest(datapath)))
            break
        complete = ber.evaluate(protocol, available_only=True)
    finally:
        ber.cells = original
    assert ber.cells() == full
    assert constants == (ber.SNRS, ber.RSRS, ber.REGIMES, ber.DATA_SEED, ber.TEST_SEED, ber.BOOTSTRAP_SEED)
    write(OUT / f'partition_check_rsr{reference:g}.json', dict(
        passed=True, reference=reference, partition_cells=51, protocol_sha256=protocol,
        scheduler_sha256=digest(__file__), restored_full_grid=True,
        scientific_constants_unchanged=True, deterministic_data_checks=checks,
        all_reference_cells_completed=complete, completed_unix=time.time()))


def schedule():
    from . import ber
    ber.torch.set_num_threads(1)
    ber.torch.set_num_interop_threads(1)
    protocol = ber.freeze()
    policy, selection = ber.study.check_selection()
    full = ber.cells()
    partitions = {f'{r:g}': [ber.label(*c) for c in full if c[1] == r] for r in REFERENCES}
    flat = [c for values in partitions.values() for c in values]
    assert len(flat) == len(set(flat)) == 153
    provenance = dict(scheduler_sha256=digest(__file__), protocol_sha256=protocol,
        selection_sha256=selection, source_hashes=policy['ber_source_hashes'],
        reference_partitions=partitions, cells_per_partition=51, full_cells=153,
        rule='Disjoint reference ownership; freeze validates the full scientific grid before temporary cells() filtering only during evaluate(). Constants and RNG indexing are unchanged. Full-grid analysis runs once after all workers exit.',
        prior_serial_workers_exited_before_start=True,
        handoff='The serial coordinator was terminated; its supervised child also exited. Completed archives are atomic and preserved; any unfinished cell is recomputed from frozen inputs.',
        initial_cell_hashes={name: digest(OUT / 'cells' / (name + '.npz')) for name in flat
                             if (OUT / 'cells' / (name + '.npz')).exists()},
        started_unix=time.time())
    write(OUT / 'execution_partition.json', provenance)
    while True:
        jobs = []
        for reference in REFERENCES:
            log = (OUT / f'partition_rsr{reference:g}.log').open('a', buffering=1)
            process = subprocess.Popen([sys.executable, '-u', '-m',
                'tcom_filter_continuation_20261019.ber_schedule', '--reference', f'{reference:g}'],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((reference, process, log))
        failures = []
        for reference, process, log in jobs:
            code = process.wait()
            log.close()
            if code:
                failures.append((reference, code))
        if failures:
            raise RuntimeError(f'BER workers failed: {failures}; inspect partition logs')
        done = len(list((OUT / 'cells').glob('*.npz')))
        print(json.dumps(dict(event='Partition pass complete', ber_cells=done)), flush=True)
        if done == 153:
            break
        time.sleep(60)
    subprocess.run([sys.executable, '-u', '-m', 'tcom_filter_continuation_20261019.ber', '--analyze-only'],
                   cwd=REPO, check=True)
    for name, expected in provenance['initial_cell_hashes'].items():
        assert digest(OUT / 'cells' / (name + '.npz')) == expected
    provenance.update(completed_unix=time.time(), completed_cells=153, all_workers_exited=True,
                      full_grid_analysis_completed=True, serial_completed_cells_unchanged=True)
    write(OUT / 'execution_partition.json', provenance)
    print('All reference workers and full-grid analysis completed.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=float, choices=REFERENCES)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reference is None:
        schedule()
    else:
        worker(args.reference)


if __name__ == '__main__':
    main()
