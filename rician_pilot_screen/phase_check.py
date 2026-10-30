"""Oracle phase-error decomposition; never an implementable phase correction.

Run from the repository root with ``python -m rician_pilot_screen.phase_check``.
All estimators receive only observations. Truth is loaded after prediction for
the diagnostic evaluator, which is the only place that aligns phase.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import torch

from fair_optshrink.estimator import tensors, trajectory
from sahure.train import load_model
from sahure_expansion.data import world


MASTER = 20261014031001
SEEDS = (9101, 9102, 9103)
CELLS = [dict(N=32, K=3, P=30, snr=float(s), rsr=float(r), L=None,
              channel='ula') for s in (-5, 5, 15) for r in (0, 5, 12)]
OUT = Path('results/rician_pilot_screen/phase')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def phase_errors(estimate, truth):
    """Squared errors before and after oracle unit-modulus phase alignment."""
    cross_row = np.sum(estimate.conj() * truth, axis=-1)
    row_phase = np.ones_like(cross_row)
    np.divide(cross_row, np.abs(cross_row), out=row_phase,
              where=np.abs(cross_row) > 0)
    cross_global = cross_row.sum(axis=-1)
    global_phase = np.ones_like(cross_global)
    np.divide(cross_global, np.abs(cross_global), out=global_phase,
              where=np.abs(cross_global) > 0)
    raw = np.sum(np.abs(estimate - truth) ** 2, axis=(-2, -1))
    glob = np.sum(np.abs(estimate * global_phase[..., None, None] - truth) ** 2,
                  axis=(-2, -1))
    row = np.sum(np.abs(estimate * row_phase[..., None] - truth) ** 2,
                 axis=(-2, -1))
    tol = 1e-10 * np.maximum(raw, 1)
    if np.any(glob > raw + tol) or np.any(row > glob + tol):
        raise AssertionError('oracle alignment must never increase squared error')
    return np.stack((raw, glob, row), axis=-1)


def algebra_checks():
    rng = np.random.default_rng(MASTER + 1)
    truth = rng.normal(size=(4, 5, 3)) + 1j * rng.normal(size=(4, 5, 3))
    row_angles = rng.uniform(-np.pi, np.pi, (4, 5, 1))
    err = phase_errors(truth * np.exp(1j * row_angles), truth)
    np.testing.assert_allclose(err[..., 2], 0, atol=1e-25)
    global_angles = row_angles[:, :1]
    err = phase_errors(truth * np.exp(1j * global_angles), truth)
    np.testing.assert_allclose(err[..., 1:], 0, atol=1e-25)
    zeros = phase_errors(np.zeros_like(truth), truth)
    np.testing.assert_allclose(zeros, np.repeat(zeros[:, :1], 3, axis=-1))
    # Check the analytic minimum against explicit aligned residuals.
    estimate = truth + rng.normal(size=truth.shape) + 1j * rng.normal(size=truth.shape)
    err = phase_errors(estimate, truth)
    norms = np.sum(abs(estimate) ** 2 + abs(truth) ** 2, axis=(-2, -1))
    cross = np.sum(estimate.conj() * truth, axis=-1)
    np.testing.assert_allclose(err[:, 2], norms - 2 * np.abs(cross).sum(-1))
    np.testing.assert_allclose(err[:, 1], norms - 2 * np.abs(cross.sum(-1)))
    # Independent row phases preserve noiseless magnitudes only at B=0.
    pilots = rng.normal(size=(4, 3, 9)) + 1j * rng.normal(size=(4, 3, 9))
    shifted = truth * np.exp(1j * row_angles)
    np.testing.assert_allclose(abs(truth @ pilots), abs(shifted @ pilots), atol=1e-12)
    ref = rng.normal(size=(4, 5, 9)) + 1j * rng.normal(size=(4, 5, 9))
    change = abs(abs(truth @ pilots + ref) - abs(shifted @ pilots + ref))
    if np.any(change > 2 * abs(ref) + 1e-12):
        raise AssertionError('reference bound violated')
    return dict(passed=True, checks=['row_phase_exact', 'global_phase_exact',
        'zero_estimate_finite', 'analytic_minimum', 'ordered_errors',
        'zero_reference_row_invariance', 'reference_magnitude_bound'], seed=MASTER + 1)


def setup(cases, output):
    checkpoints = {}
    for seed in SEEDS:
        path = Path(f'results/sahure_expansion/checkpoints/seed_{seed}/em_adaptive_best.pt')
        checkpoints[str(seed)] = dict(path=str(path), sha256=digest(path))
    frozen = json.loads(Path('results/compact_baseline/frozen_selection.json').read_text())
    for seed, entry in checkpoints.items():
        if entry['sha256'] != frozen['models']['adaptive_' + seed]['sha256']:
            raise RuntimeError('original frozen checkpoint hash mismatch')
    sources = [Path(__file__), Path('sahure/model.py'), Path('sahure/math.py'),
               Path('sahure/train.py'), Path('sahure_expansion/data.py'),
               Path('fair_optshrink/estimator.py'), Path('rydberg_sim/rng.py'),
               Path('rydberg_sim/track_b_drivers.py'), Path('rydberg_sim/monte_carlo.py')]
    protocol = dict(version=1, master_seed=MASTER, cases_per_cell=cases, cells=CELLS,
                    training_seeds=list(SEEDS), checkpoints=checkpoints,
                    sources={str(p): digest(p) for p in sources},
                    geometry='N32 K3 P30, independent random path counts 3..7 per user',
                    backend='PyTorch CPU one thread float64/complex128',
                    purpose='diagnostic only: truth-based global and antenna-row phase alignment',
                    metrics=['raw', 'oracle_global_phase', 'oracle_row_phase'],
                    uncertainty='three fits share test cases; they are not independent channels')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError('diagnostic protocol has changed')
    else:
        write(path, protocol)
    return protocol


def generate(cases, output):
    op, tp, mp = [output / p for p in ('observations.npz', 'truth.npz', 'generation.json')]
    if mp.exists():
        for p, expected in json.loads(mp.read_text())['hashes'].items():
            if digest(p) != expected:
                raise RuntimeError('saved fresh diagnostic data changed')
        return
    observed = {key: [] for key in ('z', 's', 'b', 'sigma2')}
    truths, paths, cell_ids, trial_ids, indices = [], [], [], [], []
    for cell, spec in enumerate(CELLS):
        for case in range(cases):
            index = cell * cases + case
            w, ray_counts = world(index, MASTER, **spec)
            for key, val in zip(observed, (w.Z, w.S, w.B, w.sigma2)):
                observed[key].append(val)
            truths.append(w.G)
            paths.append(ray_counts)
            cell_ids.append(cell)
            indices.append(index)
            trial_ids.append(f'phase:{MASTER}:{index}')
    np.savez_compressed(op, **{k: np.stack(v) for k, v in observed.items()},
                        cell_ids=np.array(cell_ids), indices=np.array(indices),
                        trial_ids=np.array(trial_ids))
    np.savez_compressed(tp, g=np.stack(truths), paths=np.array(paths), trial_ids=np.array(trial_ids))
    write(mp, dict(master_seed=MASTER, indices=indices, total_cases=len(indices),
                   hashes={str(p): digest(p) for p in (op, tp)},
                   all_randomness='world(index, master_seed, condition) deterministic generator; no inference randomness'))


@torch.no_grad()
def predict(cases, protocol, output):
    observed = dict(np.load(output / 'observations.npz', allow_pickle=False))
    models = {}
    for seed, entry in protocol['checkpoints'].items():
        model, _ = load_model(entry['path'])
        if sum(p.numel() for p in model.parameters()) != 188:
            raise RuntimeError('wrong adaptive model')
        models['adaptive_' + seed] = model
    methods = list(models) + ['em10', 'em100']
    predictions, timing = [], []
    for cell, spec in enumerate(CELLS):
        dest = output / f'predictions_cell{cell}.npz'
        if dest.exists():
            saved = dict(np.load(dest, allow_pickle=False))
            np.testing.assert_array_equal(saved['names'], methods)
            predictions.append(saved['estimates'])
            timing.append(saved['seconds'].tolist())
            continue
        ids = slice(cell * cases, (cell + 1) * cases)
        obs = tensors({k: observed[k][ids] for k in ('z', 's', 'b', 'sigma2')})
        values, seconds = [], []
        for name, model in models.items():
            start = time.perf_counter()
            values.append(model(*obs.args()).numpy())
            seconds.append(time.perf_counter() - start)
        start = time.perf_counter()
        classical = trajectory(obs, [dict(rank=0, strength=0.)] * len(obs), (10, 100))
        classical_time = time.perf_counter() - start
        values += [classical[10].numpy(), classical[100].numpy()]
        # Shared EM trajectory: use one combined elapsed time, not artificial per-method timing.
        seconds += [0., classical_time]
        values = np.stack(values)
        if not np.isfinite(values).all():
            raise FloatingPointError('nonfinite diagnostic estimate')
        np.savez_compressed(dest, estimates=values, names=np.array(methods),
                            trial_ids=observed['trial_ids'][ids], seconds=np.array(seconds))
        predictions.append(values)
        timing.append(seconds)
        print(f'Phase diagnostic {cell + 1}/{len(CELLS)}: SNR={spec["snr"]:g}, RSR={spec["rsr"]:g}', flush=True)
    return np.concatenate(predictions, axis=1), methods, timing


def metric_record(errors, truth_power):
    # Errors may include fitted-model axis: ratios are calculated separately per seed.
    ratios = errors.sum(axis=-2) / truth_power.sum()
    db = 10 * np.log10(np.maximum(ratios, 1e-300))
    severe = (errors / truth_power[..., None] > 1).mean(axis=-2)
    fraction = 1 - errors[..., 2].sum(axis=-1) / np.maximum(errors[..., 0].sum(axis=-1), 1e-300)
    return dict(nmse_ratio_db=np.mean(db, axis=0).tolist() if db.ndim > 1 else db.tolist(),
                mean_case_nmse_db=(10 * np.log10((errors / truth_power[..., None]).mean(axis=-2))).mean(axis=0).tolist()
                if errors.ndim == 3 else (10 * np.log10((errors / truth_power[..., None]).mean(axis=-2))).tolist(),
                severe_error_fraction=np.mean(severe, axis=0).tolist() if severe.ndim > 1 else severe.tolist(),
                row_phase_removable_squared_error_fraction=float(np.mean(fraction)),
                improvement_row_db=float(np.mean(db[..., 0] - db[..., 2])),
                improvement_global_db=float(np.mean(db[..., 0] - db[..., 1])))


def analyze(predictions, methods, cases, timing, output, elapsed, checks):
    # Ground truth enters only this evaluator, after all estimates have been saved.
    truth = dict(np.load(output / 'truth.npz', allow_pickle=False))
    observed = dict(np.load(output / 'observations.npz', allow_pickle=False))
    np.testing.assert_array_equal(truth['trial_ids'], observed['trial_ids'])
    errors = phase_errors(predictions, truth['g'][None])
    power = np.sum(abs(truth['g']) ** 2, axis=(-2, -1))
    np.savez_compressed(output / 'errors.npz', squared_errors=errors, truth_power=power,
                        names=np.array(methods), alignment=np.array(['raw', 'oracle_global', 'oracle_row']),
                        cell_ids=observed['cell_ids'], trial_ids=truth['trial_ids'])
    cells = []
    for cell, spec in enumerate(CELLS):
        ids = slice(cell * cases, (cell + 1) * cases)
        records = {name: metric_record(errors[j, ids], power[ids]) for j, name in enumerate(methods)}
        records['adaptive_seed_average'] = metric_record(errors[:3, ids], power[ids])
        cells.append(dict(spec=spec, count=cases, methods=records))
    summary = dict(status='complete', purpose='Oracle phase-error diagnostic, not a realizable estimator improvement',
                   total_cases=len(power), cases_per_cell=cases, cells=cells,
                   macro_db={name: np.mean([x['methods'][name]['nmse_ratio_db'] for x in cells], axis=0).tolist()
                             for name in methods + ['adaptive_seed_average']},
                   algebra_checks=checks, runtime_seconds=elapsed,
                   prediction_seconds_by_cell=timing,
                   timing_note='batch64 exploratory wall times; EM10 and EM100 share a trajectory, stored jointly under EM100',
                   seed_average_note='average of three per-seed dB ratios; test cases shared, not 1728 independent observations',
                   metrics_order=['raw', 'oracle_global_phase', 'oracle_row_phase'],
                   caveats=['Oracle alignment uses true channels and is not available to the receiver.',
                            'Row alignment can remove noise as well as phase ambiguity; it does not prove a distinct local minimum.',
                            'Three training seeds share one historical training dataset; no new training is run.',
                            'No inference method, checkpoint or setting is chosen from these diagnostics.'],
                   artifact_hashes={str(p): digest(p) for p in output.glob('*.npz')},
                   environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                    cpu=platform.machine(), threads=torch.get_num_threads()))
    write(output / 'summary.json', summary)
    lines = ['PHASE-ERROR DIAGNOSTIC', '',
             'All phase-aligned numbers use the true channel. They describe error; they are not achievable receiver results.',
             '', '188-parameter estimator: average of three separately evaluated fitted seeds.',
             'SNR  RSR | raw dB | global-oracle dB | row-oracle dB | removable squared error | raw severe % | row severe %']
    for cell in cells:
        spec, val = cell['spec'], cell['methods']['adaptive_seed_average']
        a, b, c = val['nmse_ratio_db']
        severe = val['severe_error_fraction']
        lines.append(f'{spec["snr"]:3.0f}  {spec["rsr"]:3.0f} | {a:7.3f} | {b:16.3f} | {c:13.3f} | '
                     f'{100*val["row_phase_removable_squared_error_fraction"]:8.2f}% | {100*severe[0]:8.2f} | {100*severe[2]:8.2f}')
    lines += ['', 'Equal-cell average dB NMSE [raw, global oracle, row oracle]:']
    lines += [name + ': ' + ', '.join(f'{x:.3f}' for x in val) for name, val in summary['macro_db'].items()]
    lines += ['', f'Total elapsed seconds: {elapsed:.2f}', '', *summary['caveats']]
    (output / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=64)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    start = time.perf_counter()
    torch.set_num_threads(1)
    checks = algebra_checks()
    protocol = setup(args.cases, args.output)
    generate(args.cases, args.output)
    values, methods, timing = predict(args.cases, protocol, args.output)
    summary = analyze(values, methods, args.cases, timing, args.output, time.perf_counter() - start, checks)
    print(json.dumps(dict(status=summary['status'], macro_db=summary['macro_db'],
                          elapsed=summary['runtime_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    main()
