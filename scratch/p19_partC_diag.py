"""PROMPT 19 Part C -- evaluation-only diagnostics. No training, no refitting.

C1a  Why do the paired median and the ratio of sums disagree in sign?

     Paper 1's answer at N = 8 is abstention: the selector declines on most
     trials, those trials are bit-identical to the baseline, the median is
     pinned at zero, and the active minority dominates a sum of squared errors.
     The learned arm has no abstention rule, so either something plays the same
     role -- a gate switching the projection off on most trials -- or the
     mechanism is different and the resemblance is coincidental.

     Measured directly: the per-trial, per-layer relative change the projection
     makes, ||proj(G) - G||_F / ||G||_F, by wrapping hankel.project_G. A trial
     whose every layer is near zero is effectively unprojected and is the
     learned analogue of an abstention.

C1b  Does the per-bin sign pattern replicate across initialisers, training
     designs and seeds? Assembled from files already in the repository; no
     number is recomputed here, and the invalid arms are labelled invalid.

Run:  PYTHONPATH=. python3 scratch/p19_partC_diag.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset
from trackD_urformer.stage4 import HIGH_SNR
import trackD_urformer.hankel as hk

from scratch.p17_score_c1 import ARMS, checkpoint_provenance, load

OUT = Path("reports/p19/partC_diagnostic.json")
N_TEST = 2000

# A trial is called "effectively unprojected" if the projection moves the
# estimate by less than this relative Frobenius norm in EVERY layer. Fixed
# here before looking at the distribution.
UNPROJECTED_REL = 1e-3
NEARLY_UNPROJ_REL = 1e-2

# Paper 1's reference point, for the comparison C1a exists to make.
PAPER1_ABSTENTION_FRACTION_N8 = 0.42


class Recorder:
    """Wraps hankel.project_G and records the relative change per call.

    project_G is imported inside URformerLayer.forward, so patching the module
    attribute takes effect on the next call -- no model surgery needed.
    """

    def __init__(self):
        self.orig = hk.project_G
        self.rel: list[float] = []

    def __enter__(self):
        def wrapped(G, *a, **kw):
            out = self.orig(G, *a, **kw)
            d = (out - G).reshape(G.shape[0], -1)
            g = G.reshape(G.shape[0], -1)
            num = torch.linalg.norm(d, dim=1).to(torch.float64)
            den = torch.linalg.norm(g, dim=1).to(torch.float64)
            self.rel.extend((num / (den + 1e-30)).detach().cpu().numpy().tolist())
            return out
        hk.project_G = wrapped
        return self

    def __exit__(self, *exc):
        hk.project_G = self.orig
        return False


@torch.no_grad()
def projection_activity(cfg, path: str, n_layers: int) -> dict:
    """One pass over the focused test set, recording projection activity."""
    ecfg = replace(cfg, data=replace(cfg.data, snr_range_db=HIGH_SNR,
                                     n_test=N_TEST))
    ds = TrackDDataset("test", sysc=ecfg.system, datac=ecfg.data,
                       numeric=ecfg.numeric, init=ecfg.train.init)
    cd, rd = ecfg.numeric.complex_dtype, ecfg.numeric.real_dtype
    m = load(cfg, path, hankel=True)
    T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)

    with Recorder() as rec:
        for i in range(N_TEST):
            s = ds.sample(i)
            m(T(ds.g0(i), cd), T(s.Z, rd), T(s.S, cd), T(s.B, cd),
              torch.tensor([s.sigma2], dtype=rd))
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{N_TEST}", flush=True)

    rel = np.asarray(rec.rel, dtype=np.float64)
    if rel.size != N_TEST * n_layers:
        raise RuntimeError(f"expected {N_TEST * n_layers} projection calls, "
                           f"recorded {rel.size} -- the layer count or the "
                           f"call site changed and this diagnostic is void")
    per_trial_layer = rel.reshape(N_TEST, n_layers)
    mx = per_trial_layer.max(axis=1)
    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {
        "n_test": N_TEST, "n_layers": n_layers,
        "per_layer_median_rel": [round(float(x), 6)
                                 for x in np.median(per_trial_layer, axis=0)],
        "per_layer_mean_rel": [round(float(x), 6)
                               for x in per_trial_layer.mean(axis=0)],
        "all_calls_quantiles": {f"p{p}": round(float(np.percentile(rel, p)), 6)
                                for p in q},
        "per_trial_max_quantiles": {f"p{p}": round(float(np.percentile(mx, p)), 6)
                                    for p in q},
        "frac_trials_every_layer_below_1e-3":
            round(float((mx < UNPROJECTED_REL).mean()), 4),
        "frac_trials_every_layer_below_1e-2":
            round(float((mx < NEARLY_UNPROJ_REL).mean()), 4),
        "frac_calls_below_1e-3": round(float((rel < UNPROJECTED_REL).mean()), 4),
    }


# --- C1b: assembled from files, nothing recomputed --------------------------
# Sign convention throughout: U1 - H1 (or uniform - balanced for the loss
# contrast), positive = the arm carrying the prior / the balanced loss is
# better. Bins are [-10,-5) [-5,0) [0,5) [5,10) [10,15) [15,20).

SIGN_TABLE = [
    {"arm": "mixed-SNR Delta_H, seed 1",
     "design": "mixed [-10,20]", "init": "spectral", "valid": True,
     "source": "reports/p12/PART_A.md A2, from trackD_stage3_results.json",
     "per_bin": [-0.111, -0.506, -0.451, +0.398, +1.305, +2.226]},
    {"arm": "focused Delta_H, seed 1",
     "design": "focused [5,20]", "init": "spectral", "valid": True,
     "source": "reports/p17/p28_score.json",
     "per_bin": [None, None, None, -0.0757, +0.1297, +0.2322]},
    {"arm": "focused Delta_H, seed 2",
     "design": "focused [5,20]", "init": "spectral", "valid": True,
     "source": "reports/p17/p28_score.json",
     "per_bin": [None, None, None, -0.1422, +0.0864, +0.3265]},
    {"arm": "focused Delta_H, seed 3",
     "design": "focused [5,20]", "init": "spectral", "valid": True,
     "source": "reports/p17/p28_score.json",
     "per_bin": [None, None, None, -0.3009, -0.0160, +0.2177]},
    {"arm": "Delta_H under the balanced loss (P20)",
     "design": "mixed [-10,20]", "init": "random", "valid": False,
     "source": "reports/p15/partC_scores.json P20_delta_h_balanced",
     "per_bin": [-0.1875, -0.4736, -0.6699, -0.3212, +0.1289, +0.9186]},
]


def sign_analysis(rows: list[dict]) -> dict:
    out = []
    for r in rows:
        signs = [None if v is None else ("+" if v > 0 else "-")
                 for v in r["per_bin"]]
        obs = [(i, s) for i, s in enumerate(signs) if s is not None]
        # index of the last negative bin; the flip sits between it and the next
        neg = [i for i, s in obs if s == "-"]
        pos = [i for i, s in obs if s == "+"]
        monotone_block = bool(neg and pos and max(neg) < min(pos)) or not neg \
            or not pos
        out.append({**{k: r[k] for k in ("arm", "design", "init", "valid",
                                         "source")},
                    "per_bin": r["per_bin"],
                    "signs": signs,
                    "all_negatives_precede_all_positives": monotone_block,
                    "flip_between_bins": (None if not (neg and pos)
                                          else [max(neg), min(pos)])})
    return {
        "bins": [[-10, -5], [-5, 0], [0, 5], [5, 10], [10, 15], [15, 20]],
        "convention": ("U1 - H1, positive = the arm carrying the structural "
                       "prior is better"),
        "rows": out,
        "every_arm_negatives_precede_positives":
            all(r["all_negatives_precede_all_positives"] for r in out),
        "flip_positions": sorted({tuple(r["flip_between_bins"]) for r in out
                                  if r["flip_between_bins"]}),
    }


# --- C1c: what DOES explain the pooling disagreement? ----------------------
# Runs on the stored stage-4 per-trial rows, so it needs no model pass at all.
# The paired median weights every trial equally; the ratio of sums weights each
# trial by its own squared error. So the two disagree exactly when the trials
# carrying the error mass differ systematically from the typical trial.
STAGE4 = Path("reports/trackD_stage4_results.json")


def pooling_account() -> dict:
    from scipy.stats import spearmanr

    pc = json.loads(STAGE4.read_text())["part_c"]
    u = np.asarray(pc["per_trial_nmse"]["C_U1_snr5_20"], dtype=np.float64)
    h = np.asarray(pc["per_trial_nmse"]["C_H1_snr5_20"], dtype=np.float64)
    snr = np.asarray(pc["snr_db"], dtype=np.float64)
    dl = 10.0 * np.log10(u / h)          # per-trial Delta_H in dB

    order = np.argsort(u)                 # ascending baseline NMSE
    k = len(u) // 10
    deciles = []
    for j in range(10):
        idx = order[j * k:(j + 1) * k]
        deciles.append({
            "decile": j + 1,
            "median_baseline_nmse": round(float(np.median(u[idx])), 6),
            "share_of_sum_u": round(float(u[idx].sum() / u.sum()), 4),
            "median_delta_db": round(float(np.median(dl[idx])), 4),
            "frac_delta_positive": round(float((dl[idx] > 0).mean()), 3)})

    top, rest = order[-2 * k:], order[:-2 * k]
    # Does baseline conditioning matter BEYOND SNR? If the decile trend is just
    # SNR re-expressed, the correlation must vanish inside a bin.
    within = []
    for lo, hi in ((5, 10), (10, 15), (15, 20)):
        m = (snr >= lo) & (snr < hi)
        r, p = spearmanr(u[m], dl[m])
        within.append({"bin": [lo, hi], "n": int(m.sum()),
                       "spearman_baseline_nmse_vs_delta": round(float(r), 4),
                       "p_value": float(p)})
    r_all, p_all = spearmanr(u, dl)
    r_snr, p_snr = spearmanr(snr, dl)

    return {
        "source": str(STAGE4) + " part_c per-trial rows (seed 1 only)",
        "median_delta_db": round(float(np.median(dl)), 4),
        "ratio_of_sums_db": round(float(10 * np.log10(u.sum() / h.sum())), 4),
        "deciles_by_baseline_nmse": deciles,
        "top_two_deciles": {
            "share_of_sum_u": round(float(u[top].sum() / u.sum()), 4),
            "ratio_of_sums_db": round(
                float(10 * np.log10(u[top].sum() / h[top].sum())), 4)},
        "bottom_eight_deciles": {
            "share_of_sum_u": round(float(u[rest].sum() / u.sum()), 4),
            "ratio_of_sums_db": round(
                float(10 * np.log10(u[rest].sum() / h[rest].sum())), 4)},
        "overall_spearman_baseline_nmse_vs_delta": round(float(r_all), 4),
        "overall_spearman_snr_vs_delta": round(float(r_snr), 4),
        "within_bin_spearman": within,
        "reading": ("if the within-bin correlations are ~0 while the overall "
                    "one is not, baseline conditioning adds nothing beyond "
                    "SNR and the pooling disagreement IS the bin structure"),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1c-only", action="store_true",
                    help="recompute only C1c and merge into the existing "
                         "report; needs no model pass")
    a = ap.parse_args()
    if a.c1c_only:
        cur = json.loads(OUT.read_text()) if OUT.exists() else {}
        cur["C1c"] = pooling_account()
        OUT.write_text(json.dumps(cur, indent=1) + "\n", encoding="utf-8")
        c = cur["C1c"]
        print("=== C1c: what explains the pooling disagreement ===")
        print(f"  median {c['median_delta_db']:+.4f} vs ratio-of-sums "
              f"{c['ratio_of_sums_db']:+.4f}")
        print(f"  {'decile':>7} {'share sum u':>12} {'median dl':>11} {'frac>0':>7}")
        for d in c["deciles_by_baseline_nmse"]:
            print(f"  {d['decile']:>7} {d['share_of_sum_u']:>12.4f} "
                  f"{d['median_delta_db']:>+11.4f} "
                  f"{d['frac_delta_positive']:>7.3f}")
        print(f"  top two deciles: {c['top_two_deciles']['share_of_sum_u']:.3f}"
              f" of the error mass, ratio-of-sums "
              f"{c['top_two_deciles']['ratio_of_sums_db']:+.4f} dB")
        print(f"  overall spearman(baseline NMSE, dl) = "
              f"{c['overall_spearman_baseline_nmse_vs_delta']:+.3f}; "
              f"spearman(SNR, dl) = {c['overall_spearman_snr_vs_delta']:+.3f}")
        for w in c["within_bin_spearman"]:
            print(f"    within {w['bin']}: "
                  f"{w['spearman_baseline_nmse_vs_delta']:+.3f} "
                  f"(p={w['p_value']:.2g}, n={w['n']})")
        print("  wrote", OUT)
        return 0

    torch.set_num_threads(1)
    base = TrackDConfig()
    cfg = replace(base, train=replace(base.train, init="spectral"))
    n_layers = int(cfg.model.T_UR)

    arms = {tag: ARMS[tag][1] for tag in ("seed1", "seed2", "seed3")
            if Path(ARMS[tag][1]).exists()}
    prov = {t: checkpoint_provenance(p) for t, p in arms.items()}

    # The gate question, settled from the checkpoints rather than assumed.
    gates = {}
    for t, p in arms.items():
        b = torch.load(p, map_location="cpu", weights_only=False)
        md = (b.get("config") or {}).get("model", {})
        gates[t] = {"hankel_gate": md.get("hankel_gate"),
                    "hankel_rank": md.get("hankel_rank"),
                    "hankel_mode": md.get("hankel_mode"),
                    "hankel_iters": md.get("hankel_iters")}

    activity = {}
    for t, p in arms.items():
        print(f"  [{t}] projection activity over {N_TEST} worlds", flush=True)
        activity[t] = projection_activity(cfg, p, n_layers)

    out = {
        "C1a": {
            "question": ("does something in the learned arm play the role "
                         "abstention plays in Paper 1 at N=8?"),
            "paper1_reference_abstention_fraction_N8":
                PAPER1_ABSTENTION_FRACTION_N8,
            "checkpoint_provenance": prov,
            "gate_configuration": gates,
            "gate_note": ("hankel_gate='none' means URformerLayer takes the "
                          "G_lin = proj branch unconditionally: there is no "
                          "learned gate, no sigma^2 dependence and no "
                          "per-trial switch. Any abstention-like behaviour "
                          "would have to come from the projection itself being "
                          "inactive, which is what the activity numbers "
                          "measure."),
            "unprojected_threshold_rel": UNPROJECTED_REL,
            "activity": activity,
        },
        "C1b": sign_analysis(SIGN_TABLE),
        "C1c": pooling_account(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print("\n=== C1a: projection activity ===")
    for t, a in activity.items():
        print(f"  {t}: gate={gates[t]['hankel_gate']}  "
              f"median rel change per call "
              f"{a['all_calls_quantiles']['p50']:.4f}")
        print(f"       trials with EVERY layer below 1e-3: "
              f"{a['frac_trials_every_layer_below_1e-3']:.4f}   "
              f"below 1e-2: {a['frac_trials_every_layer_below_1e-2']:.4f}")
        print(f"       per-trial max, p1 {a['per_trial_max_quantiles']['p1']:.4f}"
              f"  p50 {a['per_trial_max_quantiles']['p50']:.4f}"
              f"  p99 {a['per_trial_max_quantiles']['p99']:.4f}")
    print("\n=== C1b: per-bin sign pattern ===")
    print(f"  {'arm':<38} {'init':<9} {'valid':<6} signs")
    for r in out["C1b"]["rows"]:
        s = "".join(x if x else "." for x in r["signs"])
        print(f"  {r['arm']:<38} {r['init']:<9} "
              f"{'yes' if r['valid'] else 'NO':<6} {s}  "
              f"flip {r['flip_between_bins']}")
    print(f"\n  every arm: negatives precede positives = "
          f"{out['C1b']['every_arm_negatives_precede_positives']}")
    print(f"  distinct flip positions: {out['C1b']['flip_positions']}")
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
