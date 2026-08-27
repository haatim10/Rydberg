"""Track D experiment drivers D1-D3 - SCAFFOLD ONLY, NOT EXECUTED.

PROMPT 2 sec. 11 requires these to be built and left unrun. Every entry point
below refuses to run unless ``--i-have-approval`` is passed, so an accidental
invocation cannot start a multi-hour job.

    D1  NMSE vs SNR          P=20, RSR=10 dB (ours), SNR over D1_SNR_GRID_DB
    D2  NMSE vs pilots       SNR=5 dB, RSR=10 dB (ours), P over D2_P_GRID
    D3  NMSE vs array size   N over D3_N_GRID -- ONE TRAINED MODEL PER N

Shape-locking (see transformer.py): the user-token Transformer is locked to
``(N, K)``, so D3 needs a separate trained model per ``N``. ``P`` and SNR are
architecture-free and need no retraining, but D2's training distribution must
cover its sweep.

Methods: GS, EM-GS, URformer, each with all three initializers, plus the
linearised closed-form LS. No Hankel anywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    D1_SNR_GRID_DB, D2_P_GRID, D3_N_GRID, INITIALIZERS, TrackDConfig,
)

RESULTS = Path("results") / "track_d"

__all__ = ["training_matrix", "d1_plan", "d2_plan", "d3_plan", "main"]


def training_matrix(cfg: TrackDConfig) -> dict:
    """How many distinct trainings each experiment needs.

    Rows = experiments, columns = trainings per method, with the initializer
    variants multiplied in. URformer-random and URformer-spectral are two
    SEPARATELY TRAINED models, not one model with a runtime flag.
    """
    n_init = len(INITIALIZERS)
    return {
        "note": (
            "GS / EM-GS / linearised LS require no training. Only URformer "
            "does. URformer-<init> are separately trained models."
        ),
        "D1_nmse_vs_snr": {
            "urformer_trainings": n_init,
            "detail": f"N={cfg.system.N}, P={cfg.system.P}, SNR range covers the "
                      f"sweep; {n_init} initializers",
            "classical_trainings": 0,
        },
        "D2_nmse_vs_pilots": {
            "urformer_trainings": n_init,
            "detail": f"N={cfg.system.N}, P ~ U{D2_P_GRID} during training "
                      f"(P is architecture-free); {n_init} initializers",
            "classical_trainings": 0,
        },
        "D3_nmse_vs_array_size": {
            "urformer_trainings": n_init * (len(D3_N_GRID) - 1),
            "detail": f"one model per N in {D3_N_GRID} x {n_init} initializers; "
                      f"the N={cfg.system.N} column REUSES D1's models "
                      "(same config), so only the other N values are new",
            "classical_trainings": 0,
        },
        "total_urformer_trainings": n_init + n_init + n_init * (len(D3_N_GRID) - 1),
    }


def d1_plan(cfg: TrackDConfig) -> dict:
    return {"experiment": "D1_nmse_vs_snr", "N": cfg.system.N, "P": cfg.system.P,
            "rsr_ours_dB": cfg.system.rsr_db,
            "rsr_paper_equiv_dB": cfg.system.rsr_paper_equiv_db,
            "snr_grid_db": list(D1_SNR_GRID_DB),
            "methods": ["gs", "em_gs", "linearised_ls", "urformer"],
            "initializers": list(INITIALIZERS),
            "n_test": cfg.data.n_test, "paired": True}


def d2_plan(cfg: TrackDConfig) -> dict:
    return {"experiment": "D2_nmse_vs_pilots", "N": cfg.system.N,
            "snr_db": cfg.data.snr_fixed_db,
            "rsr_ours_dB": cfg.system.rsr_db,
            "rsr_paper_equiv_dB": cfg.system.rsr_paper_equiv_db,
            "p_grid": list(D2_P_GRID),
            "methods": ["gs", "em_gs", "linearised_ls", "urformer"],
            "initializers": list(INITIALIZERS),
            "n_test": cfg.data.n_test, "paired": True,
            "note": "P is architecture-free: one model trained with P ~ U(p_grid)"}


def d3_plan(cfg: TrackDConfig) -> dict:
    return {"experiment": "D3_nmse_vs_array_size", "P": cfg.system.P,
            "rsr_ours_dB": cfg.system.rsr_db,
            "rsr_paper_equiv_dB": cfg.system.rsr_paper_equiv_db,
            "n_grid": list(D3_N_GRID),
            "methods": ["gs", "em_gs", "linearised_ls", "urformer"],
            "initializers": list(INITIALIZERS),
            "n_test": cfg.data.n_test, "paired": True,
            "note": "SHAPE-LOCKED: one trained URformer per N. No Hankel."}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track D experiment driver")
    ap.add_argument("experiment", choices=["plan", "d1", "d2", "d3"])
    ap.add_argument("--i-have-approval", action="store_true",
                    help="required to actually run; scaffolding refuses otherwise")
    args = ap.parse_args(argv)

    cfg = TrackDConfig()
    plans = {"d1": d1_plan, "d2": d2_plan, "d3": d3_plan}

    if args.experiment == "plan":
        payload = {"training_matrix": training_matrix(cfg),
                   "D1": d1_plan(cfg), "D2": d2_plan(cfg), "D3": d3_plan(cfg)}
        print(json.dumps(payload, indent=2))
        return 0

    if not args.i_have_approval:
        print(f"REFUSING to run {args.experiment}: this is scaffolding only "
              "(PROMPT 2 sec. 11). Training and the D-experiments start only on "
              "explicit approval. Re-run with --i-have-approval once given.")
        print(json.dumps(plans[args.experiment](cfg), indent=2))
        return 2

    raise NotImplementedError(
        f"{args.experiment} execution is deliberately not wired up in the build "
        "phase. Implement the run loop only after approval, using "
        "evaluate.evaluate_paired for the shared test set."
    )


if __name__ == "__main__":
    raise SystemExit(main())
