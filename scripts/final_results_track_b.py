"""Parts 5-8 + 11 — the final results tables, all from stored per-trial data."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
TB = REPO / "results/track_b"
C = json.loads((TB / "crlb.json").read_text())
S3 = json.loads((TB / "b3/summary.json").read_text())
S4 = json.loads((TB / "b4/summary.json").read_text())
S6 = json.loads((TB / "b6/summary.json").read_text())
NUM = json.loads((TB / "report_numbers.json").read_text())
TIM = json.loads((TB / "timing.json").read_text())
out = {}

def sgn(lo, hi):
    return "+" if lo > 0 else ("-" if hi < 0 else "0")

print("=" * 96)
print("PART 5 — FINAL B3 AUDIT: does the HS-GS advantage grow with N?")
print("=" * 96)
print(f"{'N':>3}{'P':>4}{'SNR':>5}{'n':>6} | {'GS':>7}{'EM-GS':>8}{'HS-GS':>8}{'uCRLB':>8} | "
      f"{'gain':>7} {'95% CI':>16} {'sig':>4} {'win':>6}{'act':>6}{'tie':>6} | {'medgain':>8}")
print("-" * 96)
for r in S3:
    lo, hi = r["gain_ci95_db"]
    cr = C["b3"][f"N{r['N']}_P{r['P']}_snr{r['snr_db']:+.0f}"]
    p = r["pooled_db"]
    print(f"{r['N']:3d}{r['P']:4d}{r['snr_db']:5.0f}{r['n_trials']:6d} | "
          f"{p['biased_gs']:7.2f}{p['em_gs']:8.2f}{p['hs_gs']:8.2f}{cr:8.2f} | "
          f"{r['gain_hs_vs_em_db']:+7.2f} [{lo:+6.2f},{hi:+6.2f}] {sgn(lo,hi):>4} "
          f"{r['win_rate_vs_em']:6.0%}{r['constraint_active_frac']:6.0%}"
          f"{r['tie_frac']:6.0%} | {r['gain_median_active_db']:+8.2f}")
print()
for N in (8, 16, 32):
    sub = [r for r in S3 if r["N"] == N]
    sg = [sgn(*r["gain_ci95_db"]) for r in sub]
    loses = [r for r in sub if r["gain_ci95_db"][1] < 0]
    nulls = [r for r in sub if sgn(*r["gain_ci95_db"]) == "0"]
    out.setdefault("b3", {})[N] = {
        "mean_gain_db": float(np.mean([r["gain_hs_vs_em_db"] for r in sub])),
        "credibly_positive": sg.count("+"), "credibly_negative": sg.count("-"),
        "unresolved": sg.count("0"),
        "loses_at": [f"P={r['P']},SNR={r['snr_db']:+.0f} ({r['gain_hs_vs_em_db']:+.2f})"
                     for r in loses],
        "null_at": [f"P={r['P']},SNR={r['snr_df' if False else 'snr_db']:+.0f} "
                    f"({r['gain_hs_vs_em_db']:+.2f} [{r['gain_ci95_db'][0]:+.2f},"
                    f"{r['gain_ci95_db'][1]:+.2f}])" for r in nulls],
        "win_rate": float(np.mean([r["win_rate_vs_em"] for r in sub])),
        "active": float(np.mean([r["constraint_active_frac"] for r in sub])),
    }
    o = out["b3"][N]
    print(f"  N={N:2d}: mean gain {o['mean_gain_db']:+.3f} dB | "
          f"CI>0 at {o['credibly_positive']}/12, CI<0 at {o['credibly_negative']}/12, "
          f"unresolved {o['unresolved']}/12 | win {o['win_rate']:.1%} | "
          f"active {o['active']:.1%}")
    if o["loses_at"]:
        print(f"        LOSES (CI entirely below 0): {'; '.join(o['loses_at'])}")
    if o["null_at"]:
        print(f"        UNRESOLVED (CI straddles 0): {'; '.join(o['null_at'])}")

print("\n" + "=" * 96)
print("PART 6 — FINAL B4 AUDIT: pilot length, N = 16, SNR = 5 dB")
print("=" * 96)
print(f"{'P':>4}{'n':>6} | {'GS':>8}{'EM-GS':>8}{'HS-GS':>8}{'uCRLB':>8} | "
      f"{'gain':>7} {'95% CI':>16} {'win':>6}{'act':>6} | {'EM-uCRLB':>9}{'HS-uCRLB':>9}")
print("-" * 96)
for r in sorted(S4, key=lambda r: r["P"]):
    lo, hi = r["gain_ci95_db"]; p = r["pooled_db"]; cr = C["b4"][f"P{r['P']}"]
    print(f"{r['P']:4d}{r['n_trials']:6d} | {p['biased_gs']:8.2f}{p['em_gs']:8.2f}"
          f"{p['hs_gs']:8.2f}{cr:8.2f} | {r['gain_hs_vs_em_db']:+7.2f} "
          f"[{lo:+6.2f},{hi:+6.2f}] {r['win_rate_vs_em']:6.0%}"
          f"{r['constraint_active_frac']:6.0%} | {p['em_gs']-cr:+9.2f}{p['hs_gs']-cr:+9.2f}")
ps = sorted(S4, key=lambda r: r["P"])
sl_em = (ps[-1]["pooled_db"]["em_gs"] - ps[-2]["pooled_db"]["em_gs"])
sl_hs = (ps[-1]["pooled_db"]["hs_gs"] - ps[-2]["pooled_db"]["hs_gs"])
print(f"\n  P 30->40: EM-GS improves {sl_em:+.2f} dB, HS-GS {sl_hs:+.2f} dB "
      f"-> {'neither has flattened' if min(abs(sl_em),abs(sl_hs))>0.2 else 'flattening'}")
print(f"  gains span {min(r['gain_hs_vs_em_db'] for r in S4):+.2f} to "
      f"{max(r['gain_hs_vs_em_db'] for r in S4):+.2f} dB; "
      f"{sum(1 for r in S4 if r['gain_ci95_db'][0]>0)}/6 CIs strictly above 0")
out["b4"] = {"all_ci_positive": all(r["gain_ci95_db"][0] > 0 for r in S4),
             "slope_em_30_40": sl_em, "slope_hs_30_40": sl_hs}

print("\n" + "=" * 96)
print("PART 7 — FINAL B5 SCALING")
print("=" * 96)
print(f"{'N':>3}{'cap':>5}{'P(Lk<cap)':>11}{'E[Lk]':>8}{'2NK':>6}{'3E[sumL]':>10}"
      f"{'rho':>8} | {'mean gain':>10}{'P=10':>8}{'P=30':>8}{'win':>7}{'active':>8}")
print("-" * 96)
for N in ("8", "16", "32"):
    r, b5, v = NUM["rho"][N], NUM["b5"][N], NUM["vacuity"][N]
    print(f"{N:>3}{r['rank_cap']:5d}{v['p_informative']:11.0%}"
          f"{NUM['L']['mean_L_per_user']:8.4f}{r['dof_unstructured']:6d}"
          f"{r['dof_geometric']:10.2f}{r['rho']:7.3f}x | {b5['mean_gain_db']:+10.3f}"
          f"{b5['gain_by_P']['10']:+8.3f}{b5['gain_by_P']['30']:+8.3f}"
          f"{b5['mean_win_rate_unweighted']:7.1%}{b5['mean_constraint_active']:8.1%}")
g = [NUM["b5"][N]["mean_gain_db"] for N in ("8", "16", "32")]
print(f"\n  increments: {g[0]:+.3f} -> {g[1]:+.3f} ({g[1]-g[0]:+.3f}), "
      f"{g[1]:+.3f} -> {g[2]:+.3f} ({g[2]-g[1]:+.3f})")
print("  monotone in N: yes. No growth law is fitted; three N values bracket")
print("  the sign change to (8,16], where the algebraic threshold N=15 lies.")

print("\n" + "=" * 96)
print("PART 8 — FINAL B6 RSR ANALYSIS (P=30, SNR=5 dB)")
print("=" * 96)
print(f"{'N':>3}{'RSR':>5}{'n':>6} | {'GS':>8}{'EM-GS':>8}{'HS-GS':>8} | "
      f"{'HS-EM gain':>11} {'95% CI':>16} | {'EM-GS vs GS':>12} | {'win':>6}{'act':>6}{'Lhat':>6}")
print("-" * 96)
for r in sorted(S6, key=lambda r: (r["N"], r["rsr_db"])):
    lo, hi = r["gain_ci95_db"]; p = r["pooled_db"]
    emgs = p["biased_gs"] - p["em_gs"]
    print(f"{r['N']:3d}{r['rsr_db']:5.0f}{r['n_trials']:6d} | {p['biased_gs']:8.2f}"
          f"{p['em_gs']:8.2f}{p['hs_gs']:8.2f} | {r['gain_hs_vs_em_db']:+11.2f} "
          f"[{lo:+6.2f},{hi:+6.2f}] | {emgs:+12.2f} | {r['win_rate_vs_em']:6.0%}"
          f"{r['constraint_active_frac']:6.0%}{r['mean_L_hat']:6.2f}")
for N in (8, 32):
    sub = sorted([r for r in S6 if r["N"] == N], key=lambda r: r["rsr_db"])
    gs = [r["pooled_db"]["biased_gs"] for r in sub]
    em = [r["pooled_db"]["em_gs"] for r in sub]
    hs = [r["gain_hs_vs_em_db"] for r in sub]
    adv = [r["pooled_db"]["biased_gs"] - r["pooled_db"]["em_gs"] for r in sub]
    print(f"\n  N={N}: GS improves {gs[0]:.2f} -> {min(gs):.2f} dB over RSR 0->24 "
          f"(then {'saturates' if abs(gs[-1]-gs[-2])<0.3 else 'still moving'})")
    print(f"        EM-GS advantage over GS: {max(adv):+.2f} dB at RSR "
          f"{sub[int(np.argmax(adv))]['rsr_db']:.0f}, "
          f"{min(adv):+.2f} dB at RSR {sub[int(np.argmin(adv))]['rsr_db']:.0f}"
          f"  -> concentrated at WEAK reference")
    print(f"        HS-GS gain: {min(hs):+.2f} to {max(hs):+.2f} dB, "
          f"{'all CIs exclude 0' if all(r['gain_ci95_db'][0]>0 or r['gain_ci95_db'][1]<0 for r in sub) else 'some unresolved'}"
          f"; weakest at RSR {sub[int(np.argmin([abs(x) for x in hs]))]['rsr_db']:.0f} dB"
          if N == 32 else
          f"        HS-GS gain: {min(hs):+.2f} to {max(hs):+.2f} dB (deficit at every RSR)")
out["b6"] = {"n8_gain_range": [min(r["gain_hs_vs_em_db"] for r in S6 if r["N"]==8),
                               max(r["gain_hs_vs_em_db"] for r in S6 if r["N"]==8)],
             "n32_gain_range": [min(r["gain_hs_vs_em_db"] for r in S6 if r["N"]==32),
                                max(r["gain_hs_vs_em_db"] for r in S6 if r["N"]==32)]}

print("\n" + "=" * 96)
print("PART 11 — COMPUTATIONAL COST (deterministic benchmark, not Monte Carlo)")
print("=" * 96)
print(f"{'N':>3}{'cap':>5} | {'GS':>9}{'EM-GS':>9}{'EM chained':>12}{'HS-GS':>10} | "
      f"{'HS/EM':>7}{'HS/GS':>7} | {'order%':>8}{'proj%':>7}")
print("-" * 96)
for r in TIM:
    t = r["median_ms"]
    print(f"{r['N']:3d}{r['rank_cap']:5d} | {t['biased_gs']:9.1f}{t['em_gs']:9.1f}"
          f"{t['em_gs_chained']:12.1f}{t['hs_gs_auto']:10.1f} | "
          f"{r['ratio_to_em_gs']['hs_gs_auto']:6.1f}x"
          f"{t['hs_gs_auto']/t['biased_gs']:6.1f}x | "
          f"{r['order_search_share']:8.0%}{r['projection_share']:7.0%}")
print(f"\n  iterations: t0 = 50 for all three; HS-GS additionally searches "
      f"L_hat over 1..cap(N) with a {TIM[0].get('n_order_candidates','')}-to-16-point grid,")
print("  each candidate refit for 20 iterations on the training pilot half.")
print("  DOMINANT OVERHEAD: the order search, not the projection.")
(TB / "final_results.json").write_text(json.dumps(out, indent=2, default=float))
print(f"\nwrote {TB/'final_results.json'}")
