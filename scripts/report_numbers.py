"""Emit every number the report quotes, straight from the stored data.

Fixes 3, 5, 16 and 17 were all the same failure: a number in prose that no
longer matched the data behind it. This is the single source of truth; the
report is checked against the JSON it writes.

Read-only with respect to Track A: it opens Track A's stored aggregates but
never writes to that tree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TRACK_A = Path("/home/user/Rydberg")
sys.path.insert(0, str(REPO))
from rydberg_sim.track_b_drivers import TRACK_B_K
from rydberg_sim.track_b_proposed import hankel_rank_cap

B3 = REPO / "results/track_b/b3"
B4 = REPO / "results/track_b/b4"
B6 = REPO / "results/track_b/b6"
EST = ("biased_gs", "em_gs", "hs_gs")


def load(p: Path) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "fingerprint"}


def parse(name: str):
    parts = name.replace(".npz", "").split("_")
    return int(parts[0][1:]), int(parts[1][1:]), float(parts[2][3:])


def counts(store: Path) -> dict:
    per = {}
    for f in sorted(store.glob("N*.npz")):
        d = load(f)
        t = d["trial"]
        assert len(np.unique(t)) == len(t), f"duplicate trial index in {f.name}"
        per[f.name] = int(len(t))
    return per


def main() -> None:
    out: dict = {}

    # ---- Fix 3: trial counts reconstructed from the checkpoints ----------
    c3, c4 = counts(B3), counts(B4)
    c6 = counts(B6) if B6.exists() else {}
    # B4 reuses two B3 points verbatim (identical CRN worlds, copied not rerun)
    copied = [n for n in c4 if n in c3 and c4[n] == c3[n]]
    b4_new = {k: v for k, v in c4.items() if k not in copied}
    out["trials"] = {
        "b3_per_point": c3, "b4_per_point": c4,
        "b3_total": sum(c3.values()),
        "b4_total": sum(c4.values()),
        "b4_copied_points": sorted(copied),
        "b4_copied_trials": sum(c4[n] for n in copied),
        "b4_new_only": sum(b4_new.values()),
        "b6_per_point": c6, "b6_total": sum(c6.values()),
        "n_points_b6": len(c6),
        "grand_total_unique": (sum(c3.values()) + sum(b4_new.values())
                               + sum(c6.values())),
        "n_points_b3": len(c3), "n_points_b4": len(c4),
        "b3_at_400": sum(1 for v in c3.values() if v == 400),
        "b3_at_1200": sum(1 for v in c3.values() if v == 1200),
    }
    assert (out["trials"]["grand_total_unique"]
            == out["trials"]["b3_total"] + out["trials"]["b4_new_only"]
            + out["trials"]["b6_total"])

    # ---- Fix 5: E[L] measured on the ACTUAL B3 trials --------------------
    tot_sum, tot_n, per_N = 0.0, 0, {}
    for f in sorted(B3.glob("N*.npz")):
        N, P, s = parse(f.name)
        d = load(f)
        sl = d["L_true"].sum(1)                       # sum_k L_k per trial
        per_N.setdefault(N, []).append(sl)
        tot_sum += float(sl.sum()); tot_n += len(sl)
    mean_sumL = tot_sum / tot_n                       # E[sum_k L_k]
    out["L"] = {
        "mean_sum_L_over_b3": mean_sumL,
        "mean_L_per_user": mean_sumL / TRACK_B_K,
        "n_trials_used": tot_n,
        "exact_uniform_mean": 5.0,
        "note": "measured on the B3 trials themselves, not a fresh sample",
    }

    # ---- rho(N) recomputed from that same E[L] ---------------------------
    out["rho"] = {}
    for N in sorted(per_N):
        sl = np.concatenate(per_N[N])
        mN = float(sl.mean())
        out["rho"][str(N)] = {
            "rank_cap": hankel_rank_cap(N),
            "dof_unstructured": 2 * N * TRACK_B_K,
            "mean_sum_L": mN,
            "dof_geometric": 3 * mN,
            "rho": 2 * N * TRACK_B_K / (3 * mN),
        }

    # ---- Fix 17 + Fix 13: B5 aggregates from per-trial data --------------
    summ = json.loads((B3 / "summary.json").read_text())
    b5, em_by = {}, {}
    for N in sorted({r["N"] for r in summ}):
        sub = [r for r in summ if r["N"] == N]
        # win rate weighted by trials, and unweighted, so drift is visible
        wr_w = sum(r["win_rate_vs_em"] * r["n_trials"] for r in sub) / sum(
            r["n_trials"] for r in sub)
        b5[str(N)] = {
            "mean_gain_db": float(np.mean([r["gain_hs_vs_em_db"] for r in sub])),
            "gain_by_P": {str(P): float(np.mean(
                [r["gain_hs_vs_em_db"] for r in sub if r["P"] == P]))
                for P in sorted({r["P"] for r in sub})},
            "mean_win_rate_unweighted": float(np.mean(
                [r["win_rate_vs_em"] for r in sub])),
            "mean_win_rate_trial_weighted": float(wr_w),
            "mean_constraint_active": float(np.mean(
                [r["constraint_active_frac"] for r in sub])),
        }
        for r in sub:
            em_by.setdefault((r["P"], r["snr_db"]), {})[N] = r["pooled_db"]["em_gs"]
    out["b5"] = b5

    # ---- Fix 13 (test I): is the EM-GS baseline N-dependent? -------------
    spreads = {f"P={P},SNR={s:+.0f}": max(v.values()) - min(v.values())
               for (P, s), v in sorted(em_by.items())}
    out["test_I"] = {
        "em_gs_spread_across_N_db": spreads,
        "max_spread_db": max(spreads.values()),
        "detail": {f"P={P},SNR={s:+.0f}": v for (P, s), v in sorted(em_by.items())},
    }

    # ---- Fix 2: vacuity vs inactivity, kept distinct ---------------------
    Ls = np.arange(3, 8)
    out["vacuity"] = {}
    for N in sorted(per_N):
        cap = hankel_rank_cap(N)
        out["vacuity"][str(N)] = {
            "rank_cap": cap,
            "p_representationally_vacuous": float(np.mean(Ls >= cap)),
            "p_informative": float(np.mean(Ls < cap)),
            "vacuous_when_L_ge": cap,
        }

    # ---- Fix 16: Track A Fig. 7(a) per-point trial counts (read-only) ----
    fa = TRACK_A / "results/track_a/fig7a/aggregate.json"
    if fa.exists():
        d = json.loads(fa.read_text())
        rows = d["aggregate"] if isinstance(d, dict) and "aggregate" in d else d
        alg = sorted({r["algorithm"] for r in rows})[0]
        pts = {r["snr_db"]: int(r.get("n_trials", 0))
               for r in rows if r["algorithm"] == alg}
        bits = {s: n * 6 for s, n in pts.items()}     # 4-QAM, K=3 -> 6 bits/trial
        out["fig7a"] = {
            "per_point_trials": {f"{k:+.0f}": v for k, v in sorted(pts.items())},
            "total_trials": sum(pts.values()),
            "total_bits": sum(bits.values()),
            "bits_per_trial": 6,
            "min_point_bits": min(bits.values()),
            "max_point_bits": max(bits.values()),
            "distinct_trial_counts": sorted(set(pts.values())),
        }

    (REPO / "results/track_b/report_numbers.json").write_text(
        json.dumps(out, indent=2, default=float))

    # ------------------------------ print ---------------------------------
    t = out["trials"]
    print("=" * 74)
    print("TRIAL COUNTS (reconstructed from checkpoints, unique indices only)")
    print("=" * 74)
    print(f"  B3: {t['n_points_b3']} points, {t['b3_at_400']} at 400 + "
          f"{t['b3_at_1200']} at 1200 = {t['b3_total']}")
    print(f"  B4: {t['n_points_b4']} points, {t['b4_total']} total")
    print(f"      of which copied from B3: {t['b4_copied_trials']} "
          f"({', '.join(t['b4_copied_points'])})")
    print(f"      new work only: {t['b4_new_only']}")
    print(f"  B6: {t['n_points_b6']} points, {t['b6_total']} total")
    print(f"  GRAND TOTAL (deduplicated): {t['grand_total_unique']}")
    print()
    print("=" * 74)
    print("E[L] AND rho(N)  (measured on the B3 trials themselves)")
    print("=" * 74)
    print(f"  E[sum_k L_k] = {out['L']['mean_sum_L_over_b3']:.4f} over "
          f"{out['L']['n_trials_used']} trials "
          f"(E[L_k] = {out['L']['mean_L_per_user']:.4f})")
    for N, v in out["rho"].items():
        chk = v["rho"] * v["dof_geometric"]
        print(f"  N={N:>2}: 2NK={v['dof_unstructured']:>3}  "
              f"3E[sumL]={v['dof_geometric']:.2f}  rho={v['rho']:.3f}x  "
              f"[check rho*3E[sumL] = {chk:.2f} vs 2NK = {v['dof_unstructured']}]")
    print()
    print("=" * 74)
    print("VACUITY (true L_k vs cap) vs INACTIVITY (selected L_hat vs cap)")
    print("=" * 74)
    for N, v in out["vacuity"].items():
        print(f"  N={N:>2}: cap={v['rank_cap']:>2}, constraint vacuous when "
              f"L_k >= {v['vacuous_when_L_ge']} -> "
              f"{v['p_representationally_vacuous']:.0%} of the prior; "
              f"informative {v['p_informative']:.0%}")
    print()
    print("=" * 74)
    print("B5 WIN RATES (Fix 17)")
    print("=" * 74)
    for N, v in out["b5"].items():
        print(f"  N={N:>2}: gain {v['mean_gain_db']:+.3f} dB, "
              f"win rate unweighted {v['mean_win_rate_unweighted']:.2%}, "
              f"trial-weighted {v['mean_win_rate_trial_weighted']:.2%}")
    print()
    print("=" * 74)
    print("TEST I — IS THE EM-GS BASELINE N-DEPENDENT? (Fix 13)")
    print("=" * 74)
    for k, v in out["test_I"]["em_gs_spread_across_N_db"].items():
        det = out["test_I"]["detail"][k]
        print(f"  {k:>14}: " + "  ".join(f"N={n}:{x:7.2f}" for n, x in det.items())
              + f"   spread {v:.3f} dB")
    print(f"  MAX SPREAD ACROSS ALL POINTS: "
          f"{out['test_I']['max_spread_db']:.3f} dB")
    if "fig7a" in out:
        print()
        print("=" * 74)
        print("TRACK A FIG. 7(a) BIT ACCOUNTING (Fix 16, read-only)")
        print("=" * 74)
        f = out["fig7a"]
        print(f"  per-point trials: {f['per_point_trials']}")
        print(f"  total {f['total_trials']} trials x {f['bits_per_trial']} bits "
              f"= {f['total_bits']} bits")
        print(f"  distinct per-point trial counts: {f['distinct_trial_counts']}")
        print(f"  bits at the smallest point: {f['min_point_bits']}, "
              f"largest: {f['max_point_bits']}")
    print(f"\nwrote {REPO/'results/track_b/report_numbers.json'}")


if __name__ == "__main__":
    main()
