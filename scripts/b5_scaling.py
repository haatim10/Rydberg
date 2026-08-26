"""B5: how the HS-GS gain scales with the array dimension N.

Terminology, deliberately careful
---------------------------------
What follows is a *degrees-of-freedom / structural-redundancy* argument:
a comparison of parameter counts, plus the algebraic fact that the Hankel
rank constraint is vacuous above the pencil bound. It is NOT an
identifiability theorem. Nothing here proves that the constrained problem
has a unique solution, that the alternating projection converges to it, or
that the estimator attains any bound. Those would need proofs we do not
have. The claim supported by the data is narrower: the measured advantage
of the structured estimator grows with N in the way a redundancy argument
would lead one to expect.

The counts
----------
Unstructured channel: G is N x K complex, so 2NK real parameters.
Geometric channel: each path carries one angle psi_lk and one complex gain
alpha_lk, so 3 real parameters per path, 3 * sum_k L_k in total.
Redundancy ratio rho(N) = 2NK / (3 sum_k L_k). Only N moves with the array;
the path budget does not, so rho grows linearly in N.

Where the constraint bites
-------------------------
A length-N sequence that is a sum of L complex exponentials has a Hankel
lifting of rank at most L: each path contributes one rank-one outer product
v_l w_l^T. The converse does NOT hold for the physical model -- rank <= L
also admits exponentials with poles off the unit circle, |z_l| != 1, which
are not ULA steering responses. Low Hankel rank is therefore a NECESSARY
structural property of the sparse geometric ULA channel, exploited here as a
relaxation of the geometric manifold rather than an exact characterisation of
it. A length-N Hankel matrix
has rank at most max_p min(N-p, p+1) = ceil(N/2). So for L_k >= ceil(N/2)
the true channel already saturates the achievable rank and the constraint
carries no information about that user. This is a property of the
configuration, not of the algorithm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rydberg_sim.track_b_drivers import (
    TRACK_B_K, TRACK_B_L_MAX, TRACK_B_L_MIN,
)
from rydberg_sim.track_b_proposed import hankel_rank_cap

TB = REPO / "results/track_b"


def main():
    rows = json.loads((TB / "b3/summary.json").read_text())
    Ns = sorted({r["N"] for r in rows})
    Ps = sorted({r["P"] for r in rows})
    Ls = np.arange(TRACK_B_L_MIN, TRACK_B_L_MAX + 1)
    K = TRACK_B_K
    out = {"caveat": "structural redundancy / representational "
                     "informativeness only; NOT an identifiability theorem"}

    print("=" * 92)
    print("B5  SCALING OF THE HS-GS ADVANTAGE WITH ARRAY DIMENSION N")
    print("=" * 92)
    print("    Structural redundancy argument -- parameter counts and the")
    print("    Hankel pencil bound. NOT an identifiability theorem.\n")
    hdr = (f"{'N':>3} {'cap=ceil(N/2)':>14} {'P(L_k<cap)':>11} "
           f"{'2NK':>5} {'3*E[sumL]':>10} {'rho(N)':>8} | "
           f"{'mean gain':>10} {'P=10':>7} {'P=30':>7} {'win rate':>9} "
           f"{'act':>6}")
    print(hdr); print("-" * len(hdr))
    table = []
    for N in Ns:
        cap = hankel_rank_cap(N)
        sub = [r for r in rows if r["N"] == N]
        sumL = float(np.mean([r["mean_sum_L_true"] for r in sub]))
        rho = 2 * N * K / (3 * sumL)
        g = float(np.mean([r["gain_hs_vs_em_db"] for r in sub]))
        gp = {P: float(np.mean([r["gain_hs_vs_em_db"] for r in sub
                                if r["P"] == P])) for P in Ps}
        wr = float(np.mean([r["win_rate_vs_em"] for r in sub]))
        act = float(np.mean([r["constraint_active_frac"] for r in sub]))
        row = dict(N=N, rank_cap=cap, p_informative=float(np.mean(Ls < cap)),
                   dof_unstructured=2 * N * K, dof_geometric=3 * sumL,
                   redundancy=rho, mean_gain_db=g, gain_by_P=gp,
                   mean_win_rate=wr, mean_constraint_active=act)
        table.append(row)
        print(f"{N:3d} {cap:14d} {row['p_informative']:11.0%} "
              f"{2*N*K:5d} {3*sumL:10.1f} {rho:7.2f}x | {g:+10.2f} "
              f"{gp[Ps[0]]:+7.2f} {gp[Ps[1]]:+7.2f} {wr:9.1%} {act:6.0%}")
    out["table"] = table

    print("\n  Increment per doubling of N (mean pooled gain):")
    for a, b in zip(table, table[1:]):
        print(f"    N = {a['N']:2d} -> {b['N']:2d}:  "
              f"{a['mean_gain_db']:+.2f} -> {b['mean_gain_db']:+.2f} dB   "
              f"(change {b['mean_gain_db']-a['mean_gain_db']:+.2f} dB), "
              f"rho {a['redundancy']:.2f}x -> {b['redundancy']:.2f}x")
    out["increments"] = [
        {"from": a["N"], "to": b["N"],
         "delta_gain_db": b["mean_gain_db"] - a["mean_gain_db"],
         "delta_rho": b["redundancy"] - a["redundancy"]}
        for a, b in zip(table, table[1:])]

    print("\n  What the three N values do and do not support:")
    print("    - The measured gain is monotone in N and the sign of the")
    print("      advantage flips from negative to positive between N = 8 and")
    print("      N = 16, which is where the rank cap first exceeds max L_k.")
    print("    - Three values of N cannot identify a functional form. No")
    print("      growth law is fitted here and none should be quoted.")
    print("    - rho(N) is a parameter count. It bounds nothing on its own.")

    (TB / "b5_scaling.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {TB/'b5_scaling.json'}")


if __name__ == "__main__":
    main()
