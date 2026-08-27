"""Track D evaluation - paired Monte Carlo over a shared test set.

Every method sees the **exact same** realization of ``G_true, S, B, W, Z``.
One dataset, interleaved evaluation, never a separate Monte Carlo per
algorithm (PROMPT 2 sec. 11). Trained models obviously differ from one another;
the pairing constraint applies to the test data.

Per-trial rows are stored, not just means, so any pooling (ratio-of-sums, mean
of ratios, median, bootstrap over any subset) can be reconstructed later
without rerunning anything. Aggregation follows the repository convention:
the reported NMSE is **ratio-of-sums** (metrics.py:386).

Both RSR conventions are stored on every row::

    rsr_ours_dB          - Cui single-user denominator (what we run)
    rsr_paper_equiv_dB   - rsr_ours_dB + 10*log10(K), the paper's multi-user one

so no figure caption ever has to carry the correction by hand.

Not executed in the build phase.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import nmse_parts, run_em_gs, run_gs, run_linearised_ls
from .config import TrackDConfig
from .dataset import TrackDDataset

__all__ = ["ROW_COLUMNS", "evaluate_paired", "aggregate_rows"]

ROW_COLUMNS = (
    "trial_id", "seed", "N", "K", "P", "snr_db",
    "rsr_ours_dB", "rsr_paper_equiv_dB",
    "initializer", "estimator",
    "nmse_error_energy", "nmse_true_energy", "nmse_linear", "nmse_db",
    "runtime_s", "converged", "outlier",
)

# An outlier is a trial whose NMSE exceeds 0 dB: the estimate is worse than
# returning zero. Declared here, before any run, so the criterion cannot be
# chosen after seeing results.
OUTLIER_NMSE_LINEAR = 1.0


def _row(sample, *, estimator, initializer, G_hat, runtime_s, K):
    err, den = nmse_parts(G_hat, sample.G_true)
    lin = err / den
    return {
        "trial_id": sample.trial,
        "seed": sample.trial,
        "N": sample.N, "K": sample.K, "P": sample.P,
        "snr_db": sample.snr_db,
        "rsr_ours_dB": sample.rsr_db,
        "rsr_paper_equiv_dB": sample.rsr_db + 10.0 * math.log10(K),
        "initializer": initializer,
        "estimator": estimator,
        "nmse_error_energy": err,
        "nmse_true_energy": den,
        "nmse_linear": lin,
        "nmse_db": 10.0 * math.log10(max(lin, 1e-30)),
        "runtime_s": runtime_s,
        "converged": bool(np.all(np.isfinite(G_hat))),
        "outlier": bool(lin > OUTLIER_NMSE_LINEAR),
    }


@torch.no_grad()
def evaluate_paired(
    cfg: TrackDConfig,
    *,
    models: dict[str, object] | None = None,
    N: int | None = None,
    P: int | None = None,
    snr_db: float | None = None,
    n_trials: int | None = None,
    initializers: tuple[str, ...] = ("random", "spectral", "linearized_ls"),
) -> list[dict]:
    """Run every method on the same test realizations. Returns per-trial rows.

    Parameters
    ----------
    models
        ``{initializer: trained URformer}``. Omit to evaluate classical
        baselines only.
    """
    ds = TrackDDataset("test", sysc=cfg.system, datac=cfg.data,
                       numeric=cfg.numeric, N=N, P=P, snr_db=snr_db)
    n = int(n_trials if n_trials is not None else cfg.data.n_test)
    K = cfg.system.K
    cd, rd = cfg.numeric.complex_dtype, cfg.numeric.real_dtype
    rows: list[dict] = []

    for i in range(n):
        s = ds.sample(i)          # the ONE shared realization for this trial

        for init in initializers:
            t0 = time.time()
            g = run_gs(s, max_iter=cfg.baseline.T_GS, init=init, seed=s.trial,
                       ridge=cfg.baseline.ridge)
            rows.append(_row(s, estimator="gs", initializer=init, G_hat=g,
                             runtime_s=time.time() - t0, K=K))

            t0 = time.time()
            g = run_em_gs(s, max_iter=cfg.baseline.T_GS, init=init, seed=s.trial,
                          ridge=cfg.baseline.ridge)
            rows.append(_row(s, estimator="em_gs", initializer=init, G_hat=g,
                             runtime_s=time.time() - t0, K=K))

        t0 = time.time()
        g = run_linearised_ls(s, ridge=cfg.baseline.ridge)
        rows.append(_row(s, estimator="linearised_ls", initializer="closed_form",
                         G_hat=g, runtime_s=time.time() - t0, K=K))

        if models:
            Z = torch.as_tensor(np.array(s.Z, copy=True), dtype=rd)[None]
            S = torch.as_tensor(np.array(s.S, copy=True), dtype=cd)[None]
            B = torch.as_tensor(np.array(s.B, copy=True), dtype=cd)[None]
            s2 = torch.tensor([s.sigma2], dtype=rd)
            for init, model in models.items():
                from .baselines import make_initial_G
                G0 = torch.as_tensor(
                    make_initial_G(init, S=s.S, Z=s.Z, B=s.B, seed=s.trial),
                    dtype=cd)[None]
                t0 = time.time()
                out = model(G0, Z, S, B, s2)[0].detach().cpu().numpy()
                rows.append(_row(s, estimator="urformer", initializer=init,
                                 G_hat=out, runtime_s=time.time() - t0, K=K))
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Pool per-trial rows by (estimator, initializer, N, P, snr_db).

    Reports the repository's ratio-of-sums NMSE as primary, plus the mean of
    per-trial ratios, the median, a standard error, and the outlier rate.
    """
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["estimator"], r["initializer"], r["N"], r["P"],
                round(r["snr_db"], 3))].append(r)

    out = []
    for (est, init, N, P, snr), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        err = np.array([r["nmse_error_energy"] for r in rs])
        den = np.array([r["nmse_true_energy"] for r in rs])
        lin = np.array([r["nmse_linear"] for r in rs])
        ros = float(err.sum() / den.sum())
        # SE of the ratio-of-sums via the delta method on the per-trial ratios.
        se = float(lin.std(ddof=1) / np.sqrt(len(lin))) if len(lin) > 1 else 0.0
        out.append({
            "estimator": est, "initializer": init, "N": N, "P": P, "snr_db": snr,
            "n_trials": len(rs),
            "nmse_ratio_of_sums": ros,
            "nmse_db": 10.0 * math.log10(max(ros, 1e-30)),
            "nmse_mean_of_ratios": float(lin.mean()),
            "nmse_median": float(np.median(lin)),
            "nmse_se_linear": se,
            "ci95_low_linear": float(lin.mean() - 1.96 * se),
            "ci95_high_linear": float(lin.mean() + 1.96 * se),
            "outlier_rate": float(np.mean([r["outlier"] for r in rs])),
            "outlier_criterion": f"nmse_linear > {OUTLIER_NMSE_LINEAR}",
            "mean_runtime_s": float(np.mean([r["runtime_s"] for r in rs])),
            "rsr_ours_dB": rs[0]["rsr_ours_dB"],
            "rsr_paper_equiv_dB": rs[0]["rsr_paper_equiv_dB"],
        })
    return out


def write_rows(rows: list[dict], path: str | Path) -> None:
    """Persist per-trial rows as CSV, matching the repository's flat format."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROW_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ROW_COLUMNS})
