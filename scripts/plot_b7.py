"""Figure B3 -- HS-GS gain vs controlled path count L (experiment B7).

Plain style by instruction: white background, default lines, readable markers,
units on both axes, concise legend. Nothing here is hand-entered; every value
is recomputed from results/track_b/b7/L*.npz.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
STORE = REPO / "results" / "track_b" / "b7"
OUT = REPO / "results" / "track_b" / "artifact"
CAP = 16          # cap(N=32) = ceil(32/2)
NBOOT, BOOT_SEED = 2000, 987654321


def load():
    rows = []
    for f in sorted(glob.glob(str(STORE / "L*.npz"))):
        d = np.load(f)
        L = int(Path(f).stem[1:])
        e, h, den = d["num_em_gs"], d["num_hs_gs"], d["denom"]
        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, den.size, size=(NBOOT, den.size))
        gb = 10 * np.log10(e[idx].sum(1) / h[idx].sum(1))
        rows.append(dict(
            L=L, n=int(den.size),
            em=10 * np.log10(e.sum() / den.sum()),
            hs=10 * np.log10(h.sum() / den.sum()),
            gs=10 * np.log10(d["num_biased_gs"].sum() / den.sum()),
            gain=10 * np.log10(e.sum() / h.sum()),
            lo=float(np.percentile(gb, 2.5)), hi=float(np.percentile(gb, 97.5)),
            win=float((h < e).mean()), active=float(d["active"].mean()),
            Lhat=float(d["L_hat"].mean())))
    return rows


def main():
    r = load()
    L = [x["L"] for x in r]
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(L, [x["gs"] for x in r], "s--", color="0.45", label="biased GS")
    ax[0].plot(L, [x["em"] for x in r], "o-", color="C0", label="EM-GS")
    ax[0].plot(L, [x["hs"] for x in r], "^-", color="C3", label="HS-GS (proposed)")
    ax[0].set_xlabel("paths per user  $L$  (fixed, identical for all users)")
    ax[0].set_ylabel("channel NMSE  (dB)")
    ax[0].set_title("(a) NMSE vs path count")

    ax[1].axhline(0.0, color="0.6", lw=0.9)
    ax[1].plot(L, [x["gain"] for x in r], "^-", color="C3",
               label="HS-GS gain over EM-GS")
    ax[1].fill_between(L, [x["lo"] for x in r], [x["hi"] for x in r],
                       color="C3", alpha=0.15, lw=0, label="95% paired bootstrap CI")
    ax[1].set_xlabel("paths per user  $L$")
    ax[1].set_ylabel("gain over EM-GS  (dB)")
    ax[1].set_title("(b) gain vanishes as $L$ reaches the rank cap")

    for a in ax:
        a.axvline(CAP, color="0.3", ls=":", lw=1.2)
        a.annotate("Hankel rank cap\n$\\lceil N/2\\rceil = 16$", xy=(CAP, 0.02),
                   xycoords=("data", "axes fraction"), xytext=(-6, 6),
                   textcoords="offset points", ha="right", fontsize=8.5, color="0.3")
        a.set_xticks(L)
        a.grid(True, alpha=0.3, lw=0.6)
        a.legend(fontsize=9, framealpha=1.0)

    fig.suptitle("Figure B3 — controlled path count.  $N$ = 32, $P$ = 30, "
                 "SNR = 5 dB, RSR = 12 dB, exact model $Z=|GS+B+W|$",
                 fontsize=10.5, y=1.005)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"B3_gain_vs_pathcount.{ext}", dpi=150,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (OUT / "table_B7_pathcount.csv").write_text(
        "L,trials,GS_dB,EMGS_dB,HSGS_dB,gain_dB,gain_ci_lo,gain_ci_hi,"
        "win_rate,active_frac,mean_Lhat\n" +
        "".join("%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f\n" % (
            x["L"], x["n"], x["gs"], x["em"], x["hs"], x["gain"], x["lo"],
            x["hi"], x["win"], x["active"], x["Lhat"]) for x in r))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
