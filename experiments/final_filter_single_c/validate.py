"""Step 3: validation draws, cached pre-filter estimates, label-free selection.

  .venv/bin/python -m experiments.final_filter_single_c.validate run --workers 4
  .venv/bin/python -m experiments.final_filter_single_c.validate select

`run` caches per-cell held-out NLL and squared error for every grid constant.
`select` chooses constants from held-out NLL only and freezes them.
"""
import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy
import torch

from .common import (CACHE, RESULTS, HERE, REPO, PREREG, SEEDS, REGIMES, GRID_SNRS, GRID_RSRS, COUNT, SEED,
                     NOISE_C, POSTERIOR_C, HIGH_ROUTE_DB, cellname, make_block, engine, receiver_cache,
                     table_c, noise_filter, posterior_filter, posterior_tau, heldout_nll, squared_error,
                     observed, fit_holdout, sha)

OUT = CACHE/'validation'
_EST = None


def _init():
    global _EST
    torch.set_num_threads(1)
    _EST = engine()


def run_block(job):
    regime, rsr = job
    a = make_block(SEED['validation'], regime, rsr, GRID_SNRS)
    rows = []
    for snr in GRID_SNRS:
        path = OUT/'cells'/f'{cellname(regime, rsr, snr)}.npz'
        if path.exists():
            continue
        obs, hold = fit_holdout(observed(a, snr, P=39))
        c = receiver_cache(obs, _EST)
        tc = table_c(obs, c['ordinary'])
        shape = (len(SEEDS), COUNT)
        n_nll, n_err = np.empty((len(NOISE_C),) + shape), np.empty((len(NOISE_C),) + shape)
        p_nll, p_err = np.empty((len(POSTERIOR_C),) + shape), np.empty((len(POSTERIOR_C),) + shape)
        a_nll, a_err = np.empty(shape), np.empty(shape)
        parity = 0.
        for i in range(len(SEEDS)):
            g = c['g_pre'][i]
            arm_a = noise_filter(obs, g, tc)
            if not np.array_equal(arm_a, c['frozen'][i]):
                raise AssertionError(f'arm A differs from frozen output in {path.stem}')
            parity = max(parity, float(np.max(np.abs(arm_a - c['frozen'][i]))))
            a_nll[i], a_err[i] = heldout_nll(arm_a, hold), squared_error(arm_a, a['g'])
            for j, cc in enumerate(NOISE_C):
                f = noise_filter(obs, g, cc)
                n_nll[j, i], n_err[j, i] = heldout_nll(f, hold), squared_error(f, a['g'])
            unit = posterior_tau(obs, g)
            for j, cc in enumerate(POSTERIOR_C):
                f = posterior_filter(obs, g, cc, unit)
                p_nll[j, i], p_err[j, i] = heldout_nll(f, hold), squared_error(f, a['g'])
        np.savez_compressed(path, noise_nll=n_nll, noise_err=n_err, posterior_nll=p_nll, posterior_err=p_err,
            a_nll=a_nll, a_err=a_err, energy=np.sum(np.abs(a['g'])**2, axis=(-2, -1)),
            g_pre=c['g_pre'], accepted=c['accepted'], triggered=c['triggered'], table_c=np.array(tc),
            family=np.array(c['family']), steps=np.array(c['steps']), ordinary=np.array(c['ordinary']),
            nominal_snr=c['nominal_snr'], snr=np.array(snr), rsr=np.array(rsr), regime=np.array(regime),
            parity_max_abs=np.array(parity))
        rows.append((path.stem, parity))
    return rows


def run(workers):
    (OUT/'cells').mkdir(parents=True, exist_ok=True)
    jobs = [(r, x) for r in REGIMES for x in GRID_RSRS]
    tick = time.time()
    with ProcessPoolExecutor(workers, initializer=_init) as pool:
        for rows in pool.map(run_block, jobs):
            for name, parity in rows:
                print(json.dumps(dict(cell=name, parity=parity, seconds=round(time.time() - tick, 1))), flush=True)


def load_cells():
    cells = []
    for regime in REGIMES:
        for rsr in GRID_RSRS:
            for snr in GRID_SNRS:
                path = OUT/'cells'/f'{cellname(regime, rsr, snr)}.npz'
                with np.load(path) as d:
                    cells.append(dict({k: d[k] for k in d.files if k != 'g_pre'}, path=path))
    assert len(cells) == 153
    return cells


def nmse_db(err, energy):
    """Cell NMSE dB per fit (ratio of sums), averaged over fits."""
    return float(np.mean(10*np.log10(err.sum(-1)/energy.sum())))


def argmin_c(values, grid):
    values = np.asarray(values)
    best = values.min()
    return float(min(c for c, v in zip(grid, values) if v <= best + 1e-12))


def select():
    frozen = HERE/'configs/frozen_choices.json'
    assert not frozen.exists(), 'choices already frozen'
    cells = load_cells()
    high = np.array([float(np.mean(c['nominal_snr'])) > HIGH_ROUTE_DB + 1e-10 for c in cells])
    low30 = np.array([float(c['snr']) <= 30. for c in cells])
    # Held-out NLL per cell and constant (mean over fits and channels).
    c1_nll = np.array([[c['noise_nll'][j].mean() for c in cells] for j in range(len(NOISE_C))])   # [grid, cell]
    c0 = NOISE_C.index(0.)
    c2_nll = np.where(high[None], c1_nll, c1_nll[c0][None])
    d_nll = np.array([[c['posterior_nll'][j].mean() for c in cells] for j in range(len(POSTERIOR_C))])
    a_nll = np.array([c['a_nll'].mean() for c in cells])
    c1_db = np.array([[nmse_db(c['noise_err'][j], c['energy']) for c in cells] for j in range(len(NOISE_C))])
    c2_db = np.where(high[None], c1_db, c1_db[c0][None])
    d_db = np.array([[nmse_db(c['posterior_err'][j], c['energy']) for c in cells] for j in range(len(POSTERIOR_C))])
    a_db = np.array([nmse_db(c['a_err'], c['energy']) for c in cells])

    choice = dict(C1=argmin_c(c1_nll.mean(1), NOISE_C), C2=argmin_c(c2_nll.mean(1), NOISE_C))
    j1, j2 = NOISE_C.index(choice['C1']), NOISE_C.index(choice['C2'])
    low_c1, low_c2 = float(c1_nll[j1, low30].mean()), float(c2_nll[j2, low30].mean())
    run_d = low_c1 < low_c2 - 1e-12
    if run_d:
        choice['D'] = argmin_c(d_nll.mean(1), POSTERIOR_C)
    diagnostic = dict(C1=argmin_c(c1_db.mean(1), NOISE_C), C2=argmin_c(c2_db.mean(1), NOISE_C),
                      D=argmin_c(d_db.mean(1), POSTERIOR_C))
    curves = dict(
        C1=[dict(c=c, heldout_nll=float(c1_nll[j].mean()), nmse_db=float(c1_db[j].mean())) for j, c in enumerate(NOISE_C)],
        C2=[dict(c=c, heldout_nll=float(c2_nll[j].mean()), nmse_db=float(c2_db[j].mean())) for j, c in enumerate(NOISE_C)],
        D=[dict(c=c, heldout_nll=float(d_nll[j].mean()), nmse_db=float(d_db[j].mean())) for j, c in enumerate(POSTERIOR_C)],
        A=dict(heldout_nll=float(a_nll.mean()), nmse_db=float(a_db.mean())))
    hashes = {str(c['path'].relative_to(HERE)): sha(c['path']) for c in cells}
    record = dict(
        selected_constants=choice,
        selection_rule=PREREG['selection']['rule'], selection_score=PREREG['selection']['score'],
        d_gate=dict(run=bool(run_d), c1_low30_heldout_nll=low_c1, c2_low30_heldout_nll=low_c2,
                    rule=PREREG['selection']['d_gate'],
                    reason=None if run_d else 'C1 did not beat C2 (no filter) below 30 dB on validation held-out loss; above 20 dB u_k equals the noise scale / sqrt(2) to within 0.3%, so D adds nothing where the filter acts.'),
        true_nmse_choice_diagnostic=diagnostic, validation_curves=curves,
        high_route_cells=int(high.sum()), parity_max_abs=max(float(c['parity_max_abs']) for c in cells),
        recovery_accepted=int(sum(c['accepted'].sum() for c in cells)),
        validation_cell_sha256=hashes, validation_seed=SEED['validation'],
        preregistration_sha256=sha(HERE/'configs/preregistration.json'),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         torch=torch.__version__, machine=platform.machine()),
        frozen_unix=time.time())
    frozen.write_text(json.dumps(record, indent=2) + '\n')
    (HERE/'configs/frozen_choices.sha256').write_text(f'{sha(frozen)}  configs/frozen_choices.json\n')
    print(json.dumps(dict(selected=choice, d_gate=record['d_gate'], diagnostic=diagnostic, A=curves['A']), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('run', 'select'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.workers) if args.phase == 'run' else select()


if __name__ == '__main__':
    main()
