"""Aggregate a Track-B per-trial store into pooled NMSE, CIs, gains, win rates.

Everything is recomputed from the stored per-trial numerators
``||Ghat-G||_F^2`` and denominators ``||G||_F^2``; nothing is read from a
pre-reduced summary. The pooled metric is the ratio of sums

    NMSE_G = 10 log10( Σ_t ||Ĝ_t - G_t||_F^2 / Σ_t ||G_t||_F^2 ),

never a mean of per-trial dB values. The bootstrap resamples *trials*
(paired across estimators, preserving common random numbers) and
recomputes the same ratio of sums on each resample.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ESTIMATORS = ("biased_gs", "em_gs", "hs_gs")
NBOOT = 2000
BOOT_SEED = 987654321


def pooled_db(num: np.ndarray, den: np.ndarray) -> float:
    return float(10 * np.log10(num.sum() / den.sum()))


def summarize(path: Path, n_boot: int = NBOOT) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files if k != "fingerprint"}
        fp = str(z["fingerprint"])
    den = d["denom"]
    n = den.size
    num = {e: d[f"num_{e}"] for e in ESTIMATORS}

    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(n_boot, n))          # paired resample
    den_b = den[idx].sum(1)
    boot = {e: 10 * np.log10(num[e][idx].sum(1) / den_b) for e in ESTIMATORS}
    gain_b = 10 * np.log10(num["em_gs"][idx].sum(1) / num["hs_gs"][idx].sum(1))

    out = {
        "file": path.name, "fingerprint": fp, "n_trials": int(n),
        "pooled_db": {e: pooled_db(num[e], den) for e in ESTIMATORS},
        "ci95_db": {e: [float(np.percentile(boot[e], 2.5)),
                        float(np.percentile(boot[e], 97.5))]
                    for e in ESTIMATORS},
        # diagnostic only -- never a replacement for the pooled metric
        "median_per_trial_db": {e: float(10 * np.log10(np.median(num[e] / den)))
                                for e in ESTIMATORS},
        "gain_hs_vs_em_db": float(10 * np.log10(num["em_gs"].sum()
                                                / num["hs_gs"].sum())),
        "gain_ci95_db": [float(np.percentile(gain_b, 2.5)),
                         float(np.percentile(gain_b, 97.5))],
        "gain_median_per_trial_db": float(
            10 * np.log10(np.median(num["em_gs"] / num["hs_gs"]))),
        "win_rate_vs_em": float(np.mean(num["hs_gs"] < num["em_gs"])),
        # HS-GS == EM-GS bit-for-bit whenever the order selector picks
        # L_hat >= rank cap, which makes the projection a no-op. Those
        # trials are exact ties, and they dominate the median at N = 8.
        "tie_frac": float(np.mean(num["hs_gs"] == num["em_gs"])),
        "gain_median_active_db": (
            float(10 * np.log10(np.median(
                (num["em_gs"] / num["hs_gs"])[d["active"]])))
            if d["active"].any() else float("nan")),
        "mean_sum_L_true": float(d["L_true"].sum(1).mean()),
        "mean_L_hat": float(d["L_hat"].mean()),
        "constraint_active_frac": float(d["active"].mean()),
        # is the pooled gain an artefact of a few catastrophic EM-GS trials?
        "em_worst_share": float(num["em_gs"].max() / num["em_gs"].sum()),
        "hs_worst_share": float(num["hs_gs"].max() / num["hs_gs"].sum()),
    }
    # pooled gain with the 5% most extreme EM-GS trials removed (test F)
    keep = num["em_gs"] / den <= np.percentile(num["em_gs"] / den, 95)
    out["gain_trimmed95_db"] = float(
        10 * np.log10(num["em_gs"][keep].sum() / num["hs_gs"][keep].sum()))
    return out


def parse(name: str):
    body = name.replace(".npz", "")
    parts = body.split("_")
    N, P, snr = int(parts[0][1:]), int(parts[1][1:]), float(parts[2][3:])
    rsr = float(parts[3][3:]) if len(parts) > 3 else None
    return N, P, snr, rsr


def main() -> None:
    store = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results/track_b/b3"
    rows = []
    for f in sorted(store.glob("N*.npz")):
        N, P, snr, rsr = parse(f.name)
        r = summarize(f)
        r.update(N=N, P=P, snr_db=snr)
        if rsr is not None:
            r["rsr_db"] = rsr
        rows.append(r)
    rows.sort(key=lambda r: (r["N"], r["P"], r.get("rsr_db", 0.0), r["snr_db"]))
    (store / "summary.json").write_text(json.dumps(rows, indent=2))

    hdr = (f"{'N':>3}{'P':>4}{'SNR':>6} {'n':>5} | {'GS':>7} {'EM-GS':>7} "
           f"{'HS-GS':>7} | {'gain':>6} {'gain 95% CI':>16} {'win':>5} "
           f"{'act':>5} {'Lhat':>5} {'sumL':>5} | {'medGS':>7} {'medEM':>7} "
           f"{'medHS':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        p, m = r["pooled_db"], r["median_per_trial_db"]
        lo, hi = r["gain_ci95_db"]
        x = r.get("rsr_db", r["snr_db"])
        print(f"{r['N']:3d}{r['P']:4d}{x:6.1f} {r['n_trials']:5d} | "
              f"{p['biased_gs']:7.2f} {p['em_gs']:7.2f} {p['hs_gs']:7.2f} | "
              f"{r['gain_hs_vs_em_db']:+6.2f} [{lo:+6.2f},{hi:+6.2f}] "
              f"{r['win_rate_vs_em']:5.0%} {r['constraint_active_frac']:5.0%} "
              f"{r['mean_L_hat']:5.2f} {r['mean_sum_L_true']:5.2f} | "
              f"{m['biased_gs']:7.2f} {m['em_gs']:7.2f} {m['hs_gs']:7.2f}")
    print(f"\nwrote {store/'summary.json'}")


if __name__ == "__main__":
    main()
