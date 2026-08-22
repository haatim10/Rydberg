"""FIX 10 — re-measure achieved SNR and RSR with a large sample and CIs.

The report quoted 2.82 dB against a 3.00 dB target as a passing check, with
no sample size and no interval. A 0.18 dB gap is a 4% power error, and Sec. 3
is built on the calibration being exact -- so it needs either a CI containing
the target or a diagnosed bias.

Both quantities are ratios of expectations (Sec. 5.1's rule), never means of
per-realization ratios.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rydberg_sim.track_b_drivers import TRACK_B_K, track_b_world

N_REAL = 120_000 // 30          # realizations; each gives N*P field samples
BOOT = 2000


def main():
    snr_db, rsr_db, P, N = 3.0, 12.0, 30, 8
    sig_num, noise_num, ref_num, per_user_num = [], [], [], []
    for t in range(N_REAL):
        w = track_b_world(t, P, snr_db, rsr_db=rsr_db, N=N)
        field = w.G @ w.S                       # N x P, all K users summed
        sig_num.append(float(np.mean(np.abs(field) ** 2)))
        noise_num.append(float(np.mean(np.abs(w.W) ** 2)))
        ref_num.append(float(np.mean(np.abs(w.B) ** 2)))
        # single-user contribution: a_{n,k} s_k for one k, per Cui eq. (37)
        one = np.outer(w.G[:, 0], w.S[0])
        per_user_num.append(float(np.mean(np.abs(one) ** 2)))
    sig = np.array(sig_num); noi = np.array(noise_num)
    ref = np.array(ref_num); usr = np.array(per_user_num)

    rng = np.random.default_rng(20250820)
    idx = rng.integers(0, len(sig), size=(BOOT, len(sig)))

    def ratio_db(a, b, ix=None):
        if ix is None:
            return 10 * np.log10(a.sum() / b.sum())
        return 10 * np.log10(a[ix].sum(1) / b[ix].sum(1))

    snr_meas = ratio_db(sig, noi)
    snr_boot = ratio_db(sig, noi, idx)
    rsr_meas = ratio_db(ref, usr)
    rsr_boot = ratio_db(ref, usr, idx)

    out = {
        "n_realizations": int(N_REAL), "n_field_samples": int(N_REAL * N * P),
        "snr": {"target_db": snr_db, "measured_db": float(snr_meas),
                "ci95_db": [float(np.percentile(snr_boot, 2.5)),
                            float(np.percentile(snr_boot, 97.5))]},
        "rsr": {"target_db": rsr_db, "measured_db": float(rsr_meas),
                "ci95_db": [float(np.percentile(rsr_boot, 2.5)),
                            float(np.percentile(rsr_boot, 97.5))]},
        "method": "ratio of summed energies (Sec. 5.1 rule), bootstrap over "
                  "realizations, 2000 resamples",
    }
    for k in ("snr", "rsr"):
        v = out[k]
        v["target_inside_ci"] = bool(v["ci95_db"][0] <= v["target_db"] <= v["ci95_db"][1])
        v["bias_db"] = v["measured_db"] - v["target_db"]

    # diagnostic (a) of Fix 10: is the numerator exactly K per realization?
    num_per_real = []
    for t in range(200):
        w = track_b_world(t, P, snr_db, rsr_db=rsr_db, N=N)
        num_per_real.append(float(np.mean(np.abs(w.G @ w.S) ** 2)))
    out["diagnostic_numerator"] = {
        "expected_exactly_K": TRACK_B_K,
        "mean_over_200_realizations": float(np.mean(num_per_real)),
        "std_over_realizations": float(np.std(num_per_real)),
        "note": "if row normalization were per-realization this would be "
                "exactly K with zero spread; a non-zero spread means the "
                "normalization is in expectation, so the measured SNR carries "
                "genuine Monte Carlo error and a small-sample bias",
    }
    (REPO / "results/track_b/calibration.json").write_text(json.dumps(out, indent=2))

    print("=" * 70)
    print(f"CALIBRATION — {out['n_realizations']} realizations, "
          f"{out['n_field_samples']:,} field samples")
    print("=" * 70)
    for k, name in (("snr", "SNR"), ("rsr", "RSR")):
        v = out[k]
        mark = "target INSIDE CI" if v["target_inside_ci"] else "target OUTSIDE CI"
        print(f"  {name}: {v['measured_db']:.3f} dB "
              f"[{v['ci95_db'][0]:.3f}, {v['ci95_db'][1]:.3f}]  "
              f"target {v['target_db']:.2f}  bias {v['bias_db']:+.3f} dB  -> {mark}")
    d = out["diagnostic_numerator"]
    print(f"\n  numerator E|a^H s|^2: mean {d['mean_over_200_realizations']:.4f} "
          f"(expected K = {d['expected_exactly_K']}), "
          f"per-realization std {d['std_over_realizations']:.4f}")
    print(f"\nwrote {REPO/'results/track_b/calibration.json'}")


if __name__ == "__main__":
    main()
    print()
    track_a_calibration()


def track_a_calibration(n_real: int = 4000, snr_db: float = 3.0,
                        rsr_db: float = 12.0, N: int = 36, K: int = 3):
    """Same measurement on the Track-A 38.901 channel (read-only).

    Generating worlds from the shared library does not touch the frozen
    Track-A tree; nothing is written there and nothing is rerun.
    """
    from rydberg_sim.config import SimulationConfig
    from rydberg_sim.monte_carlo import ExperimentSpec, generate_detection_trial

    cfg = SimulationConfig.create(N=N, K=K, L=(1,) * K, beta=1.0,
                                  master_seed=20250820, c=1.0)
    spec = ExperimentSpec(experiment="calib", track="A", cfg=cfg, P=1,
                          vartheta=0.0, snr_db_grid=(snr_db,),
                          rsr_db_grid=(rsr_db,), n_trials=n_real,
                          algorithms=("biased_gs",), max_iter=50, qam_M=16,
                          channel_model="cui_38901", write_ber=False)
    sig, noi, ref, usr = [], [], [], []
    for t in range(n_real):
        d = generate_detection_trial(spec, t, snr_db, rsr_db)
        sig.append(float(np.mean(np.abs(d.A.conj().T @ d.s) ** 2)))
        noi.append(float(np.mean(np.abs(d.w) ** 2)))
        ref.append(float(np.mean(np.abs(d.b) ** 2)))
        usr.append(float(np.mean(np.abs(np.conjugate(d.A[0]) * d.s[0]) ** 2)))
    sig, noi = np.array(sig), np.array(noi)
    ref, usr = np.array(ref), np.array(usr)
    rng = np.random.default_rng(7)
    ix = rng.integers(0, n_real, size=(BOOT, n_real))
    def rdb(a, b, i=None):
        return (10 * np.log10(a.sum() / b.sum()) if i is None
                else 10 * np.log10(a[i].sum(1) / b[i].sum(1)))
    out = {}
    for name, (a, b, tgt) in {"snr": (sig, noi, snr_db),
                              "rsr": (ref, usr, rsr_db)}.items():
        m, bt = rdb(a, b), rdb(a, b, ix)
        lo, hi = float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))
        out[name] = {"target_db": tgt, "measured_db": float(m),
                     "ci95_db": [lo, hi], "bias_db": float(m) - tgt,
                     "target_inside_ci": bool(lo <= tgt <= hi)}
    out["n_realizations"] = n_real
    out["channel"] = "Track-A 3GPP TR 38.901 (read-only; nothing rerun)"
    p = REPO / "results/track_b/calibration_track_a.json"
    p.write_text(json.dumps(out, indent=2))
    print("=" * 70)
    print(f"TRACK-A CHANNEL CALIBRATION — {n_real} realizations, N={N}, K={K}")
    print("=" * 70)
    for k, nm in (("snr", "SNR"), ("rsr", "RSR")):
        v = out[k]
        print(f"  {nm}: {v['measured_db']:.3f} dB "
              f"[{v['ci95_db'][0]:.3f}, {v['ci95_db'][1]:.3f}]  "
              f"target {v['target_db']:.2f}  bias {v['bias_db']:+.3f} dB  -> "
              f"{'INSIDE' if v['target_inside_ci'] else 'OUTSIDE'}")
    print(f"wrote {p}")
    return out
