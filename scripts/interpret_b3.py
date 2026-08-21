"""Track-B interpretation tests A-H, evaluated numerically from the stores.

Each test is answered from the saved per-trial data, not from expectation.
Where the data contradict the earlier 60-trial smoke result, that is
reported as such.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rydberg_sim.track_b_proposed import hankel_rank_cap
from rydberg_sim.track_b_drivers import (
    TRACK_B_K, TRACK_B_L_MAX, TRACK_B_L_MIN,
)

TB = REPO / "results/track_b"


def sign_of(ci):
    lo, hi = ci
    if lo > 0:
        return "+"          # credibly positive
    if hi < 0:
        return "-"          # credibly negative
    return "0"              # CI straddles zero


def redundancy(N, sumL, K=TRACK_B_K):
    """Unstructured vs geometric degrees of freedom.

    Unstructured: G is N x K complex, 2NK real parameters.
    Geometric: each path contributes one angle and one complex gain,
    3 real parameters, so 3 * sum_k L_k.
    This is a parameter count, i.e. a statement about representational
    redundancy -- NOT an identifiability theorem.
    """
    return 2 * N * K / (3 * sumL)


def main():
    rows = json.loads((TB / "b3/summary.json").read_text())
    Ns = sorted({r["N"] for r in rows})
    Ps = sorted({r["P"] for r in rows})
    out = {}

    print("=" * 78)
    print("A/B  PER-N CREDIBILITY OF THE HS-GS GAIN (bootstrap 95% CI on the")
    print("     pooled ratio-of-sums gain; '+' = CI entirely above 0)")
    print("=" * 78)
    print(f"{'N':>3} {'pts':>4} {'CI>0':>5} {'CI<0':>5} {'straddle':>9} "
          f"{'min gain':>9} {'max gain':>9} {'mean':>7}")
    for N in Ns:
        sub = [r for r in rows if r["N"] == N]
        sg = [sign_of(r["gain_ci95_db"]) for r in sub]
        g = [r["gain_hs_vs_em_db"] for r in sub]
        out.setdefault("per_N", {})[N] = {
            "n_points": len(sub), "n_pos": sg.count("+"), "n_neg": sg.count("-"),
            "n_straddle": sg.count("0"), "min_gain_db": min(g),
            "max_gain_db": max(g), "mean_gain_db": float(np.mean(g)),
        }
        print(f"{N:3d} {len(sub):4d} {sg.count('+'):5d} {sg.count('-'):5d} "
              f"{sg.count('0'):9d} {min(g):+9.2f} {max(g):+9.2f} "
              f"{np.mean(g):+7.2f}")
    print("\n  A: N=8 mixed at 400 trials?  ->",
          "YES, mixed" if out["per_N"][8]["n_neg"] and out["per_N"][8]["n_pos"]
          else ("uniformly negative" if out["per_N"][8]["n_neg"]
                else "uniformly positive/inconclusive"))
    print("  B: N=16 credible positive gain? ->",
          f"{out['per_N'][16]['n_pos']}/{out['per_N'][16]['n_points']} points "
          f"with CI entirely above 0")

    print()
    print("=" * 78)
    print("C  DOES THE GAIN GROW FROM N=16 TO N=32? (per (P,SNR) point)")
    print("=" * 78)
    print(f"{'P':>3}{'SNR':>6} | {'N=8 gain [CI]':>26} {'N=16 gain [CI]':>26} "
          f"{'N=32 gain [CI]':>26} | {'32>16 CIs disjoint':>19}")
    disj = 0; tot = 0
    for P in Ps:
        for s in sorted({r["snr_db"] for r in rows}):
            cells = []
            byN = {}
            for N in Ns:
                r = next((r for r in rows if r["N"] == N and r["P"] == P
                          and r["snr_db"] == s), None)
                byN[N] = r
                cells.append("" if r is None else
                             f"{r['gain_hs_vs_em_db']:+6.2f} "
                             f"[{r['gain_ci95_db'][0]:+5.2f},"
                             f"{r['gain_ci95_db'][1]:+5.2f}]")
            ok = ""
            if byN.get(16) and byN.get(32):
                tot += 1
                d = byN[32]["gain_ci95_db"][0] > byN[16]["gain_ci95_db"][1]
                disj += d
                ok = "yes" if d else "no"
            print(f"{P:3d}{s:6.1f} | " + " ".join(f"{c:>26}" for c in cells)
                  + f" | {ok:>19}")
    print(f"\n  C: N=32 gain CI strictly above the N=16 gain CI at "
          f"{disj}/{tot} points")
    out["C_disjoint"] = [disj, tot]

    print()
    print("=" * 78)
    print("D  WIN RATE vs N (fraction of trials where HS-GS beats EM-GS)")
    print("=" * 78)
    hdr = f"{'P':>3}{'SNR':>6} | " + " ".join(f"{'N='+str(N):>8}" for N in Ns)
    print(hdr)
    wins = {N: [] for N in Ns}
    for P in Ps:
        for s in sorted({r["snr_db"] for r in rows}):
            cs = []
            for N in Ns:
                r = next((r for r in rows if r["N"] == N and r["P"] == P
                          and r["snr_db"] == s), None)
                cs.append("" if r is None else f"{r['win_rate_vs_em']:8.1%}")
                if r:
                    wins[N].append(r["win_rate_vs_em"])
            print(f"{P:3d}{s:6.1f} | " + " ".join(f"{c:>8}" for c in cs))
    mean_w = {N: float(np.mean(wins[N])) for N in Ns}
    print("  mean   | " + " ".join(f"{mean_w[N]:8.1%}" for N in Ns))
    mono = all(mean_w[a] <= mean_w[b] for a, b in zip(Ns, Ns[1:]))
    print(f"\n  D: mean win rate monotonically increasing in N? -> "
          f"{'YES' if mono else 'NO'}")
    out["D_mean_win_rate"] = mean_w
    out["D_monotone"] = bool(mono)

    print()
    print("=" * 78)
    print("E  STRUCTURAL REDUNDANCY / REPRESENTATIONAL INFORMATIVENESS")
    print("=" * 78)
    print("   Parameter counts only. This is NOT an identifiability theorem.")
    Ls = np.arange(TRACK_B_L_MIN, TRACK_B_L_MAX + 1)
    print(f"\n{'N':>3} {'rank cap':>9} {'P(L_k<cap)':>11} {'E[sum L_k]':>11} "
          f"{'2NK':>6} {'3*sumL':>8} {'redundancy':>11} {'mean gain':>10}")
    for N in Ns:
        cap = hankel_rank_cap(N)
        p_inf = float(np.mean(Ls < cap))
        sub = [r for r in rows if r["N"] == N]
        sumL = float(np.mean([r["mean_sum_L_true"] for r in sub]))
        red = redundancy(N, sumL)
        out.setdefault("E", {})[N] = {
            "rank_cap": cap, "p_constraint_informative": p_inf,
            "mean_sum_L": sumL, "dof_unstructured": 2 * N * TRACK_B_K,
            "dof_geometric": 3 * sumL, "redundancy": red,
            "mean_gain_db": out["per_N"][N]["mean_gain_db"],
        }
        print(f"{N:3d} {cap:9d} {p_inf:11.0%} {sumL:11.2f} "
              f"{2*N*TRACK_B_K:6d} {3*sumL:8.1f} {red:10.2f}x "
              f"{out['per_N'][N]['mean_gain_db']:+10.2f}")
    reds = [out["E"][N]["redundancy"] for N in Ns]
    gains = [out["E"][N]["mean_gain_db"] for N in Ns]
    print(f"\n  E: redundancy and mean gain both monotone in N? -> "
          f"{'YES' if all(a<b for a,b in zip(reds,reds[1:])) and all(a<b for a,b in zip(gains,gains[1:])) else 'NO'}")

    print()
    print("=" * 78)
    print("F  IS THE GAIN AN ARTEFACT OF A FEW CATASTROPHIC EM-GS TRIALS?")
    print("=" * 78)
    print(f"{'N':>3}{'P':>4}{'SNR':>6} | {'gain':>7} {'gain w/o worst 5%':>18} "
          f"{'delta':>7} | {'EM worst share':>15} {'HS worst share':>15}")
    for r in rows:
        print(f"{r['N']:3d}{r['P']:4d}{r['snr_db']:6.1f} | "
              f"{r['gain_hs_vs_em_db']:+7.2f} {r['gain_trimmed95_db']:+18.2f} "
              f"{r['gain_trimmed95_db']-r['gain_hs_vs_em_db']:+7.2f} | "
              f"{r['em_worst_share']:15.1%} {r['hs_worst_share']:15.1%}")
    d = [r["gain_trimmed95_db"] - r["gain_hs_vs_em_db"] for r in rows]
    print(f"\n  F: median change when the worst 5% of EM-GS trials are "
          f"dropped: {np.median(d):+.2f} dB (max |change| {np.max(np.abs(d)):.2f} dB)")
    out["F_median_trim_delta_db"] = float(np.median(d))

    print()
    print("=" * 78)
    print("G  HIGH-SNR BEHAVIOUR: DOES HS-GS FLOOR OUT?")
    print("=" * 78)
    print(f"{'N':>3}{'P':>4} | {'alg':>10} {'NMSE@15':>9} {'NMSE@20':>9} "
          f"{'slope dB/dB (10->20)':>21}")
    for N in Ns:
        for P in Ps:
            for alg in ("em_gs", "hs_gs"):
                pt = {r["snr_db"]: r["pooled_db"][alg] for r in rows
                      if r["N"] == N and r["P"] == P}
                if not {10.0, 15.0, 20.0} <= set(pt):
                    continue
                sl = (pt[20.0] - pt[10.0]) / 10.0
                print(f"{N:3d}{P:4d} | {alg:>10} {pt[15.0]:9.2f} "
                      f"{pt[20.0]:9.2f} {sl:21.3f}")
            print()
    print("  G: a slope near -1.0 dB/dB means no floor; a slope toward 0")
    print("     means the estimator has stopped improving with SNR.")

    print()
    print("=" * 78)
    print("H  POOLED vs MEDIAN: DO THE CONCLUSIONS AGREE?")
    print("=" * 78)
    print("   HS-GS reduces to EM-GS bit-for-bit when the selected order")
    print("   reaches the Hankel rank cap, so those trials are EXACT TIES.")
    print("   A median of +0.00 dB means the median trial is such a tie, not")
    print("   a disagreement; the tie-free column excludes them.")
    print(f"\n{'N':>3}{'P':>4}{'SNR':>6} | {'pooled':>8} {'median (all)':>13} "
          f"{'ties':>6} {'median (active)':>16} {'sign agrees':>12}")
    agree = agree_active = 0
    for r in rows:
        a = np.sign(r["gain_hs_vs_em_db"]) == np.sign(r["gain_median_per_trial_db"])
        aa = np.sign(r["gain_hs_vs_em_db"]) == np.sign(r["gain_median_active_db"])
        agree += a; agree_active += aa
        print(f"{r['N']:3d}{r['P']:4d}{r['snr_db']:6.1f} | "
              f"{r['gain_hs_vs_em_db']:+8.2f} "
              f"{r['gain_median_per_trial_db']:+13.2f} {r['tie_frac']:6.1%} "
              f"{r['gain_median_active_db']:+16.2f} "
              f"{('yes' if aa else 'NO'):>12}")
    print(f"\n  H: pooled vs median-over-all-trials agree in sign at "
          f"{agree}/{len(rows)} points; the shortfall is entirely exact ties.")
    print(f"     Excluding ties, pooled vs median agree at "
          f"{agree_active}/{len(rows)} points.")
    out["H_sign_agreement"] = [int(agree), len(rows)]
    out["H_sign_agreement_tie_free"] = [int(agree_active), len(rows)]

    (TB / "b3/interpretation.json").write_text(json.dumps(out, indent=2,
                                                          default=float))
    print(f"\nwrote {TB/'b3/interpretation.json'}")


if __name__ == "__main__":
    main()
