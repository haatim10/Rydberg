"""Track D stage-3 figures (PROMPT 6 Part B).

Palette: validated dataviz categorical slots. Repo conventions kept.

Primary statistic in every panel: the paired per-trial median. The one
ratio-of-sums panel is labelled as such, and it is there because Q4 asks about
the low-SNR tail, which is the only question ratio-of-sums answers better.

Run:  PYTHONPATH=. python3 scratch/trackD_stage3_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("results/track_d/stage3")
D3 = json.load(open("reports/trackD_stage3_results.json"))
TEST, CON = D3["test"], D3["test"]["contrasts"]
PER = {k: np.asarray(v) for k, v in TEST["per_trial_nmse"].items()}
SNR = np.asarray(TEST["snr_db"])
LK = np.asarray(TEST["L_k"])

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948")
INK, MUTED = "#0b0b0b", "#52514e"

db = lambda x: 10.0 * np.log10(x)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT/name}.png")


def _boot_median_ci(d, n_boot=4000, seed=20260830):
    rng = np.random.default_rng(seed)
    b = np.array([np.median(rng.choice(d, d.size, replace=True))
                  for _ in range(n_boot)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ---------------------------------------------------------------------------
# 1. THE headline: where every arm lands, and the Delta_H that decides Part C
# ---------------------------------------------------------------------------
order = ["U0_em_gs", "H0_hs_em_gs", "X1_emgs_plus_former", "U1_urformer_80k",
         "U1_plus_post", "H1_hs_urformer_80k", "oracle_phase"]
label = {"U0_em_gs": "U0  EM-GS", "H0_hs_em_gs": "H0  HS-EM-GS",
         "X1_emgs_plus_former": "X1  EM-GS + 1 former",
         "U1_urformer_80k": "U1  URformer 80k",
         "U1_plus_post": "U1+post  (post-hoc)",
         "H1_hs_urformer_80k": "H1  HS-URformer",
         "oracle_phase": "unstructured-LS oracle"}
col = {"U0_em_gs": MUTED, "H0_hs_em_gs": AQUA, "X1_emgs_plus_former": YELLOW,
       "U1_urformer_80k": BLUE, "U1_plus_post": ORANGE,
       "H1_hs_urformer_80k": MAGENTA, "oracle_phase": INK}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1),
                              gridspec_kw={"width_ratios": [1.35, 1]})
meds = [db(np.median(PER[k])) for k in order]
ci = [tuple(db(np.array(_boot_median_ci(PER[k])))) for k in order]
y = np.arange(len(order))[::-1]
for yi, k, m, (lo, hi) in zip(y, order, meds, ci):
    ax.barh(yi, m, color=col[k], alpha=0.85, height=0.62)
    ax.plot([lo, hi], [yi, yi], color=INK, lw=1.2)
    ax.annotate(f"{m:.2f}", (m, yi), textcoords="offset points",
                xytext=(-4 if m < 0 else 4, 0), ha="right" if m < 0 else "left",
                va="center", fontsize=6.5, fontweight="bold", color=INK)
ax.set_yticks(y)
ax.set_yticklabels([label[k] for k in order])
ax.set_xlabel("test NMSE, per-trial MEDIAN (dB)   lower is better")
ax.set_title("a. where each arm lands", loc="left", fontweight="bold")
ax.invert_xaxis()

# The decision panel: Delta_H against the pre-registered threshold.
dH = CON["delta_H"]
names = ["delta_H", "delta_H_posthoc", "internal_vs_posthoc",
         "delta_H_classical"]
nice = {"delta_H": "$\\Delta_H$  U1 → H1\n(internal, the decision)",
        "delta_H_posthoc": "U1 → U1+post\n(post-hoc)",
        "internal_vs_posthoc": "U1+post → H1\n(integration)",
        "delta_H_classical": "U0 → H0\n(classical, Track B)"}
cc = [MAGENTA, ORANGE, BLUE, AQUA]
yy = np.arange(len(names))[::-1]
ax2.axvspan(0.0, 0.3, color=YELLOW, alpha=0.13)
ax2.axvline(0.0, color=INK, lw=1.0)
ax2.axvline(0.3, color=RED, ls="--", lw=1.4)
for yi, nm, c in zip(yy, names, cc):
    v = CON[nm]
    lo, hi = v["boot_ci95_median"]
    ax2.plot([lo, hi], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
    ax2.plot(v["median_diff_db"], yi, "o", color=c, ms=7,
             markeredgecolor="white", markeredgewidth=0.8)
    ax2.annotate(f"{v['median_diff_db']:+.3f}", (v["median_diff_db"], yi),
                 textcoords="offset points", xytext=(0, 8), ha="center",
                 fontsize=6.5, fontweight="bold", color=c)
ax2.set_yticks(yy)
ax2.set_yticklabels([nice[n] for n in names], fontsize=6.5)
ax2.set_xlabel("paired median $\\Delta$ (dB), + = Hankel helps")
ax2.set_title("b. the pre-registered decision", loc="left", fontweight="bold")
ax2.annotate("go\nthreshold", (0.3, yy[0] + 0.42), color=RED, fontsize=6,
             ha="center", fontweight="bold")
fig.suptitle(f"HS-URformer at 80k — verdict: {TEST['verdict']['decision']}",
             fontsize=9, fontweight="bold", y=1.02)
_save(fig, "fig1_headline")


# ---------------------------------------------------------------------------
# 2. Q4 -- does the Hankel prior help the hard low-SNR tail disproportionately?
# ---------------------------------------------------------------------------
d_med = db(PER["U1_urformer_80k"]) - db(PER["H1_hs_urformer_80k"])
edges = np.array([-10, -5, 0, 5, 10, 15, 20])
mid = 0.5 * (edges[:-1] + edges[1:])
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
m, elo, ehi = [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    s = (SNR >= a) & (SNR < b)
    m.append(np.median(d_med[s]))
    lo, hi = _boot_median_ci(d_med[s])
    elo.append(m[-1] - lo)
    ehi.append(hi - m[-1])
ax.axhline(0, color=INK, lw=1.0)
ax.errorbar(mid, m, yerr=[elo, ehi], color=MAGENTA, lw=2, marker="o", ms=5,
            capsize=3)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("$\\Delta_H$, paired median (dB)")
ax.set_title("a. $\\Delta_H$ by SNR (PRIMARY: median)", loc="left",
             fontweight="bold")

# The one place ratio-of-sums earns its keep: it is dominated by the worst
# trials, which is exactly the tail Q4 asks about. Labelled, not smuggled.
ros = [db(PER["U1_urformer_80k"][(SNR >= a) & (SNR < b)].sum()
          / PER["H1_hs_urformer_80k"][(SNR >= a) & (SNR < b)].sum())
       for a, b in zip(edges[:-1], edges[1:])]
ax2.axhline(0, color=INK, lw=1.0)
ax2.plot(mid, m, color=MAGENTA, lw=2, marker="o", ms=5, label="median (primary)")
ax2.plot(mid, ros, color=ORANGE, lw=2, marker="s", ms=5, ls="--",
         label="ratio-of-sums (SECONDARY)")
ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel("$\\Delta_H$ (dB)")
ax2.legend(frameon=False)
ax2.set_title("b. Q4: the two statistics disagree where the tail is",
              loc="left", fontweight="bold")
_save(fig, "fig2_delta_by_snr")


# ---------------------------------------------------------------------------
# 3. How hard was the prior actually imposed? (the n_iter caveat, measured)
# ---------------------------------------------------------------------------
iters = [1, 2, 4, 8]
fig, ax = plt.subplots(figsize=(3.7, 2.9))
vals = [CON[f"posthoc_n_iter_{k}"]["median_diff_db"] for k in iters]
er = np.array([[v - CON[f"posthoc_n_iter_{k}"]["boot_ci95_median"][0],
                CON[f"posthoc_n_iter_{k}"]["boot_ci95_median"][1] - v]
               for v, k in zip(vals, iters)]).T
ax.axhline(0, color=INK, lw=1.0)
ax.errorbar(iters, vals, yerr=er, color=ORANGE, lw=2, marker="o", ms=6,
            capsize=3, label="U1+post gain")
ax.axhline(CON["delta_H"]["median_diff_db"], color=MAGENTA, ls="--", lw=1.5)
ax.annotate("$\\Delta_H$ (H1, internal, n_iter=1)", (1.05, CON["delta_H"]["median_diff_db"]),
            textcoords="offset points", xytext=(0, 4), color=MAGENTA,
            fontsize=6.5, fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(iters)
ax.set_xticklabels(iters)
ax.set_xlabel("Cadzow sweeps (n_iter)")
ax.set_ylabel("paired median gain (dB)")
ax.set_title("One sweep is not a projection:\ndoes imposing it harder help?",
             loc="left", fontweight="bold", fontsize=7.5)
_save(fig, "fig3_niter_sensitivity")


# ---------------------------------------------------------------------------
# 4. X1 -- the control that asks whether the unrolling is doing anything
# ---------------------------------------------------------------------------
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
arms = ["U0_em_gs", "X1_emgs_plus_former", "U1_urformer_80k"]
lbl = ["EM-GS\n(0 params)", "X1: EM-GS + 1 former\n(158k, no unrolling)",
       "U1: URformer\n(1.59M, 10 unrolled)"]
vals = [db(np.median(PER[k])) for k in arms]
bars = ax.bar(range(3), vals, color=[MUTED, YELLOW, BLUE], width=0.6)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                xytext=(0, -12 if v < 0 else 4), ha="center", fontsize=7,
                fontweight="bold", color="white" if v < 0 else INK)
ax.set_xticks(range(3))
ax.set_xticklabels(lbl, fontsize=6.3)
ax.set_ylabel("test NMSE, median (dB)")
ax.set_title("a. does unrolling earn its 10x?", loc="left", fontweight="bold")

gap = CON["X1_vs_U1"]
ax2.axvline(0, color=INK, lw=1.0)
for i, (nm, key, c) in enumerate([
        ("X1 vs U1\n(is unrolling needed?)", "X1_vs_U1", YELLOW),
        ("U1 vs U0\n(URformer gain)", "U1_vs_U0", BLUE),
        ("X1 vs U0\n(post-processor gain)", "X1_vs_U0", ORANGE)]):
    v = CON[key]
    lo, hi = v["boot_ci95_median"]
    ax2.plot([lo, hi], [2 - i, 2 - i], color=c, lw=2.2, solid_capstyle="round")
    ax2.plot(v["median_diff_db"], 2 - i, "o", color=c, ms=7,
             markeredgecolor="white", markeredgewidth=0.8)
    ax2.annotate(f"{v['median_diff_db']:+.2f}", (v["median_diff_db"], 2 - i),
                 textcoords="offset points", xytext=(0, 8), ha="center",
                 fontsize=6.5, fontweight="bold", color=c)
    ax2.text(0, 0, "")
ax2.set_yticks([2, 1, 0])
ax2.set_yticklabels(["X1 vs U1\n(is unrolling needed?)", "U1 vs U0\n(URformer gain)",
                     "X1 vs U0\n(post-processor gain)"], fontsize=6.3)
ax2.set_xlabel("paired median $\\Delta$ (dB), + = second arm better")
ax2.set_title("b. paired, per trial", loc="left", fontweight="bold")
_save(fig, "fig4_X1_control")


# ---------------------------------------------------------------------------
# 5. Does the prior help more when the channel is actually low-rank?
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.7, 2.9))
Lmax = LK.max(axis=1)
ks = sorted(set(Lmax.tolist()))
m, er = [], []
for k in ks:
    s = Lmax == k
    m.append(np.median(d_med[s]))
    lo, hi = _boot_median_ci(d_med[s])
    er.append([m[-1] - lo, hi - m[-1]])
ax.axhline(0, color=INK, lw=1.0)
ax.errorbar(ks, m, yerr=np.array(er).T, color=AQUA, lw=2, marker="o", ms=6,
            capsize=3)
ax.set_xlabel("$\\max_k L_k$ in the trial")
ax.set_ylabel("$\\Delta_H$, paired median (dB)")
ax.set_title("r=7 is slack when $L_k$ is small.\nDoes that show up?", loc="left",
             fontweight="bold", fontsize=7.5)
_save(fig, "fig5_delta_by_rank")

# ---------------------------------------------------------------------------
# 6. Did the straight-through estimator train cleanly?
#
# The STE makes H1's backward pass the gradient of a network WITHOUT the
# projection, while the forward pass has it. That is standard practice but it
# is a real approximation, and it is the first thing to suspect if H1 trains
# worse than U1 rather than differently. Same seed, same init, same data order,
# same schedule -- the curves are directly comparable.
# ---------------------------------------------------------------------------
def _curve(path):
    import csv as _csv
    with open(path) as fh:
        rows = list(_csv.DictReader(fh))
    return (np.array([float(r["epoch"]) for r in rows]),
            np.array([float(r["train_loss_db"]) for r in rows]),
            np.array([float(r["val_nmse_db"]) for r in rows]))


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
eu, tu, vu = _curve("results/track_d/stage2/B3_80k_13ep/curves.csv")
eh, th, vh = _curve(OUT / "H1_hs_urformer_80k" / "curves.csv")
ax.plot(eu, tu, color=BLUE, lw=2, marker="o", ms=3.5, label="U1 URformer")
ax.plot(eh, th, color=MAGENTA, lw=2, marker="s", ms=3.5, label="H1 HS-URformer")
ax.set_xlabel("epoch")
ax.set_ylabel("train NMSE (dB)")
ax.legend(frameon=False)
ax.set_title("a. training loss — did the STE train cleanly?", loc="left",
             fontweight="bold", fontsize=7.5)

ax2.plot(eu, vu, color=BLUE, lw=2, marker="o", ms=3.5, label="U1 URformer")
ax2.plot(eh, vh, color=MAGENTA, lw=2, marker="s", ms=3.5, label="H1 HS-URformer")
cu = D3.get("_u1_chosen_epoch", 6)
ch = D3["runs"]["H1_hs_urformer_80k"]["chosen_epoch"]
ax2.axvline(cu, color=BLUE, ls=":", lw=1.2)
ax2.axvline(ch, color=MAGENTA, ls=":", lw=1.2)
ax2.annotate(f"U1 sel. ep{cu}", (cu, vu.max()), fontsize=6, color=BLUE,
             ha="center", textcoords="offset points", xytext=(0, 3))
ax2.annotate(f"H1 sel. ep{ch}", (ch, vh.max()), fontsize=6, color=MAGENTA,
             ha="center", textcoords="offset points", xytext=(0, -9))
ax2.set_xlabel("epoch")
ax2.set_ylabel("val NMSE (dB)")
ax2.legend(frameon=False)
ax2.set_title("b. validation, with the one-SE selected epochs", loc="left",
              fontweight="bold", fontsize=7.5)
_save(fig, "fig6_training_curves")


print("\nstage-3 figures written to", OUT)
