"""Cui Fig. 7 (BER vs SNR) and Fig. 8 (BER vs RSR) drivers.

Everything here is taken from the paper; nothing is guessed where Cui is
explicit. This module defines **no** mathematics: the observation model,
SNR/RSR calibration, biased GS, EM-GS, ZF-known-phase, the exhaustive LS/ML
searches, the QAM alphabet and the BER counter are all the already-validated
Track-A code paths.

Paper sources
-------------
§VI-C: "To evaluate the BER performance, the constellation demapping step is
introduced to project the recovered signals s̃_k for each user to the nearest
constellation point ŝ_k and then to 0-1 bits."

Fig. 7 — BER vs SNR, RSR fixed at 12 dB (§VI-C):
  (a) N × K = 36 × 3,  4-QAM   ("small-scale configuration")
  (b) N × K = 100 × 6, 16-QAM  ("large-scale configuration")
SNR sweep −5 … 12 dB (§VI-A: "the SNR is varying from −5 dB to 12 dB";
the plotted axes run −5 to 12 in both panels).

Fig. 8 — BER vs RSR, SNR fixed at 3 dB, RSR 0 … 25 dB (§VI-A: "RSR grows
from 0 dB to 25 dB"; the plotted axis runs 0 to 25).

.. warning::
   **The paper is self-contradictory about Fig. 8's modulation.** The caption
   says "for a 16-QAM modulator under 3 dB SNR"; the body text says "The SNR
   is fixed as 3 dB and a 4-QAM modulator is adopted." The plotted BER levels
   settle it in favour of the body text: Fig. 8 at RSR = 12 dB shows EM-GS at
   ≈5e-3, matching Fig. 7(a) (4-QAM, the same N × K, RSR = 12 dB) at
   SNR = 3 dB ≈ 4e-3. A 16-QAM curve at 3 dB SNR would sit near 1e-1 — two
   orders of magnitude away. :data:`FIG8_QAM` therefore follows the body
   text, and :data:`FIG8_QAM_CAPTION_CLAIM` records the caption for the
   documented discrepancy.

Algorithm sets follow Cui's own benchmark list, minus CM-ZF:

  §VI-A describes CM-ZF only as "extend[ing] this approximation to the biased
  PR problem" from reference [39]. That is not a specification, so
  ``baselines.py`` deliberately does not implement it and it is excluded here
  rather than invented.

Fig. 7(b) drops both exhaustive searches, exactly as Cui does: "For the
large-scale configuration depicted in Fig. 7(b), the computation of
exhaustive search method is prohibitive, so it is excluded from this
comparison." (16**6 = 16 777 216 candidates per trial.)

Fig. 8 plots no ZF-known-phase curve; it is still evaluated here because it
costs almost nothing and is a useful reference, and it is kept out of the
figure to match Cui.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .channel_cui import CHANNEL_MODEL_CUI, CuiChannelParams
from .config import SimulationConfig
from .monte_carlo import ExperimentSpec
from .track_a_fig5 import FIG5_MASTER_SEED, FIG5_T0

# --- Cui §VI-A / §VI-C, verbatim -------------------------------------------
FIG7_RSR_DB: float = 12.0
FIG7_SNR_DB: tuple[float, ...] = tuple(float(x) for x in range(-5, 13))

FIG7A_N, FIG7A_K, FIG7A_QAM = 36, 3, 4
FIG7B_N, FIG7B_K, FIG7B_QAM = 100, 6, 16

FIG8_SNR_DB: float = 3.0
FIG8_RSR_DB: tuple[float, ...] = tuple(float(x) for x in range(0, 26))
FIG8_N, FIG8_K = 36, 3
FIG8_QAM: int = 4          # body text (§VI-C); see the module warning
FIG8_QAM_CAPTION_CLAIM: int = 16   # Fig. 8 caption, contradicts the text

# Cui's benchmark list minus CM-ZF (unspecified — see module docstring).
FIG7A_ALGORITHMS: tuple[str, ...] = (
    "biased_gs", "em_gs", "exhaustive_ls", "exhaustive_ml", "genie_zf",
)
FIG7B_ALGORITHMS: tuple[str, ...] = ("biased_gs", "em_gs", "genie_zf")
FIG8_ALGORITHMS: tuple[str, ...] = FIG7A_ALGORITHMS

# monte_carlo writes the BER row under this metric name.
BER_METRIC = "ber"

FIG7A_EXPERIMENT = "cui_fig7a"
FIG7B_EXPERIMENT = "cui_fig7b"
FIG8_EXPERIMENT = "cui_fig8"

# Curves reach ~5e-5, so the ladder is sized by bit count, not trial count.
FIG7_CHECKPOINTS: tuple[int, ...] = (2_000, 10_000, 40_000)
FIG8_CHECKPOINTS: tuple[int, ...] = (2_000, 10_000, 40_000)


def _ber_spec(
    *,
    experiment: str,
    n_trials: int,
    N: int,
    K: int,
    qam_M: int,
    algorithms: Sequence[str],
    snr_db_grid: Sequence[float],
    rsr_db_grid: Sequence[float],
    master_seed: int = FIG5_MASTER_SEED,
    cui_params: CuiChannelParams | None = None,
) -> ExperimentSpec:
    """A Track-A detection spec with BER writing enabled.

    Identical in every other respect to the Fig. 5/6 configuration: same
    channel model, same Table-I parameters, same ``t0 = 50`` iteration count,
    same master seed and therefore the same common-random-number structure.
    """
    cfg = SimulationConfig.create(
        N=N, K=K, L=1, beta=1.0, master_seed=master_seed, c=1.0
    )
    return ExperimentSpec(
        experiment=experiment,
        track="A",
        cfg=cfg,
        P=1,
        vartheta=0.0,
        snr_db_grid=tuple(float(x) for x in snr_db_grid),
        rsr_db_grid=tuple(float(x) for x in rsr_db_grid),
        n_trials=int(n_trials),
        algorithms=tuple(algorithms),
        max_iter=FIG5_T0,
        qam_M=int(qam_M),
        channel_model=CHANNEL_MODEL_CUI,
        cui_params=cui_params if cui_params is not None else CuiChannelParams(),
        write_ber=True,
    )


def track_a_fig7a_spec(*, n_trials: int, **kw) -> ExperimentSpec:
    """Fig. 7(a): N × K = 36 × 3, 4-QAM, RSR = 12 dB, SNR −5…12 dB."""
    kw.setdefault("experiment", FIG7A_EXPERIMENT)
    kw.setdefault("snr_db_grid", FIG7_SNR_DB)
    kw.setdefault("rsr_db_grid", (FIG7_RSR_DB,))
    return _ber_spec(
        n_trials=n_trials, N=FIG7A_N, K=FIG7A_K, qam_M=FIG7A_QAM,
        algorithms=FIG7A_ALGORITHMS, **kw,
    )


def track_a_fig7b_spec(*, n_trials: int, **kw) -> ExperimentSpec:
    """Fig. 7(b): N × K = 100 × 6, 16-QAM, RSR = 12 dB, SNR −5…12 dB."""
    kw.setdefault("experiment", FIG7B_EXPERIMENT)
    kw.setdefault("snr_db_grid", FIG7_SNR_DB)
    kw.setdefault("rsr_db_grid", (FIG7_RSR_DB,))
    return _ber_spec(
        n_trials=n_trials, N=FIG7B_N, K=FIG7B_K, qam_M=FIG7B_QAM,
        algorithms=FIG7B_ALGORITHMS, **kw,
    )


def track_a_fig8_spec(*, n_trials: int, qam_M: int = FIG8_QAM, **kw) -> ExperimentSpec:
    """Fig. 8: N × K = 36 × 3, SNR = 3 dB, RSR 0…25 dB.

    ``qam_M`` defaults to the body text's 4-QAM; pass
    :data:`FIG8_QAM_CAPTION_CLAIM` to run the caption's claim instead.
    """
    kw.setdefault("experiment", FIG8_EXPERIMENT)
    kw.setdefault("snr_db_grid", (FIG8_SNR_DB,))
    kw.setdefault("rsr_db_grid", FIG8_RSR_DB)
    return _ber_spec(
        n_trials=n_trials, N=FIG8_N, K=FIG8_K, qam_M=qam_M,
        algorithms=FIG8_ALGORITHMS, **kw,
    )


# ---------------------------------------------------------------------------
# BER aggregation
# ---------------------------------------------------------------------------


def aggregate_ber(
    rows: Sequence[dict], *, sweep_key: str = "snr_db"
) -> list[dict]:
    """Pool ``metric == "ber"`` rows into per-(algorithm, sweep point) BER.

    BER is a **ratio of sums** — total bit errors over total bits — never the
    mean of per-trial BERs (``metrics.py``: "Global BER is
    ``total_bit_errors / total_bit_count``, not the mean of per-trial BERs").

    ``ber == 0`` is reported as ``0.0`` with ``ber_is_zero`` set, so plotting
    can drop the point rather than take ``log10(0)``. The 95 % interval uses
    the Wilson score, which stays valid for very small counts and for zero.
    """
    buckets: dict[tuple[str, float], list[int]] = {}
    for r in rows:
        if r.get("metric") != BER_METRIC or r.get("status", "ok") != "ok":
            continue
        key = (str(r["algorithm"]), float(r[sweep_key]))
        acc = buckets.setdefault(key, [0, 0, 0])
        acc[0] += int(r["bit_errors"])
        acc[1] += int(r["bit_count"])
        acc[2] += 1

    out: list[dict] = []
    for (alg, x), (errs, bits, n) in sorted(buckets.items(), key=lambda kv: kv[0]):
        ber = (errs / bits) if bits else float("nan")
        lo, hi = _wilson(errs, bits)
        out.append({
            "algorithm": alg,
            sweep_key: x,
            "bit_errors": errs,
            "bit_count": bits,
            "n_trials": n,
            "ber": ber,
            "ber_ci95_low": lo,
            "ber_ci95_high": hi,
            "ber_is_zero": errs == 0,
        })
    return out


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval; valid at k = 0 and for tiny n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def ber_snr_gap_db(
    agg: Sequence[dict], alg_a: str, alg_b: str, target_ber: float,
    *, sweep_key: str = "snr_db",
) -> float | None:
    """SNR gap at a fixed BER, by log-linear interpolation on each curve.

    Cui reports this for Fig. 7(b): "the SNR gap between the EM-GS algorithm
    and the ZF method with known phase for realizing the same BER level is
    between 3 ∼ 4 dB".
    """
    xa = _snr_at_ber(agg, alg_a, target_ber, sweep_key)
    xb = _snr_at_ber(agg, alg_b, target_ber, sweep_key)
    if xa is None or xb is None:
        return None
    return float(xa - xb)


def _snr_at_ber(
    agg: Sequence[dict], alg: str, target: float, sweep_key: str
) -> float | None:
    pts = sorted(
        ((float(r[sweep_key]), float(r["ber"])) for r in agg
         if r["algorithm"] == alg and r["ber"] > 0.0),
        key=lambda t: t[0],
    )
    for (x0, b0), (x1, b1) in zip(pts, pts[1:]):
        if (b0 - target) * (b1 - target) <= 0.0 and b0 != b1:
            t = (np.log10(target) - np.log10(b0)) / (np.log10(b1) - np.log10(b0))
            return float(x0 + t * (x1 - x0))
    return None
