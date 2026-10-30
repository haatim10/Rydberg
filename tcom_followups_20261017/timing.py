"""Run only after other study jobs finish: warmed batch-one CPU latency."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import json
import platform
import time
import numpy as np
import torch
from .common import ROOT, load_estimators, world_batch, write_json, sha, frozen_inputs
from .sparse import estimate as sparse_estimate


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    selection_path = ROOT / 'sparse/frozen_selection.json'
    selection = json.loads(selection_path.read_text())['selections']
    estimators = load_estimators()
    names = list(estimators) + ['sparse']
    rng = np.random.default_rng(2026101732)
    measurements = []
    for snr in (0, 5, 10, 15, 20):
        config = selection[str(snr)]
        current = dict(estimators, sparse=lambda obs: sparse_estimate(obs, config))
        for L in (2, 4, None):
            for case in range(3):
                obs, _, _ = world_batch(2026101731, [case], snr, L=L)
                # Warm every complete estimator, including spectral initializer.
                for name in rng.permutation(names):
                    current[name](obs)
                for repeat in range(3):
                    for name in rng.permutation(names):
                        started = time.perf_counter_ns()
                        g = current[name](obs)
                        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                        if not torch.isfinite(g).all():
                            raise FloatingPointError(name)
                        measurements.append(dict(method=name, snr_db=snr, L=L, case=case,
                                                 repeat=repeat, milliseconds=elapsed_ms))
        print(f'Isolated timing completed SNR {snr}', flush=True)
    aggregate = {}
    for name in names:
        values = np.array([r['milliseconds'] for r in measurements if r['method'] == name])
        aggregate[name] = dict(mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                               p10_ms=float(np.quantile(values, .1)), p90_ms=float(np.quantile(values, .9)),
                               measurements=len(values))
    learned = np.array([r['milliseconds'] for r in measurements if r['method'].startswith('learned_')])
    aggregate['learned_seed_mean'] = dict(mean_ms=float(learned.mean()), median_ms=float(np.median(learned)))
    write_json(ROOT / 'timing.json', dict(aggregate=aggregate, measurements=measurements,
        protocol='Separate run after all three experimental computations completed. Float64/complex128, '
                 'CPU one thread, batch1, 45 fresh worlds covering all15 ordinary cells, '
                 'three random-order repeats after warming every method on every world. '
                 'Includes initializer and all estimation operations, excludes model/file loading, '
                 'observation generation and detector. Sparse uses frozen validation-selected settings.',
        device=platform.platform(), machine=platform.machine(), inputs=frozen_inputs(),
        selection_sha256=sha(selection_path), source_sha256=sha(__file__)))
    print(json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
