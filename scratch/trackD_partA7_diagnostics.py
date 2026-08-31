"""PROMPT 7 Part A -- A3 (STE gradient fidelity) and A4 (loss by SNR).

A3 asks whether the straight-through estimator is a good approximation to the
exact gradient through the SVD, separately at low and high SNR. If fidelity is
high in both regimes the STE is not the explanation for H1's low-SNR loss and
Part C can be skipped.

The exact SVD gradient carries ``1/(sigma_i - sigma_j)`` terms, so it is only
trustworthy when the singular values are well separated. Every exact-gradient
number below is reported WITH the gap that produced it, and discarded if the
gap makes it meaningless.

A4 tests whether per-sample normalized NMSE lets low-SNR trials dominate the
training signal -- which would mean H1 was optimized mainly for the regime its
own constraint damages.

Run:  PYTHONPATH=. python3 scratch/trackD_partA7_diagnostics.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from trackD_urformer.config import NumericConfig, TrackDConfig
from trackD_urformer.dataset import TrackDDataset, collate
from trackD_urformer.hankel import singular_gap_stats
from trackD_urformer.stage1 import build_model
from trackD_urformer.train import make_initial_batch, nmse_loss

OUT = Path("reports/trackD_partA7_diagnostics.json")
STAGE2 = Path("results/track_d/stage2/B3_80k_13ep/best.pt")
STAGE3 = Path("results/track_d/stage3/H1_hs_urformer_80k/best.pt")
BINS = [(-10, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]
LOW, HIGH = (-10.0, 0.0), (10.0, 20.0)


def _cfg() -> TrackDConfig:
    """float64 END TO END -- data included, not just the model.

    A3 compares an exact SVD gradient against the STE's. In float32 the
    difference between the two would be contaminated by rounding at roughly the
    size of the effect being measured, so the dataset must yield float64 too;
    promoting only the model is what raised the first dtype error here.
    Training itself ran in float32 (the repository default) and is untouched.
    """
    c = TrackDConfig()
    return replace(c, train=replace(c.train, init="spectral"),
                   numeric=NumericConfig(dtype="float64"))


def load_arm(cfg, *, hankel: bool):
    """U1 (hankel=False, stage 2) or H1 (hankel=True, stage 3), float64."""
    rcfg = replace(cfg, model=replace(cfg.model, filter_init="random",
                                      use_transformer=True, use_hankel=hankel,
                                      hankel_rank=7, hankel_mode="fixed"))
    m, _ = build_model(rcfg, "arm1b_full_random")   # already .double() via numeric
    blob = torch.load(STAGE3 if hankel else STAGE2, map_location="cpu",
                      weights_only=False)
    # The checkpoints are float32; load_state_dict casts on copy. The weights
    # are bit-preserved under float32 -> float64, so this is exact.
    m.load_state_dict(blob["model"])
    return m.double().eval(), rcfg


def batch_in_snr_range(cfg, lo, hi, n, split="val"):
    """Collate the first ``n`` samples of ``split`` whose SNR lies in [lo, hi)."""
    ds = TrackDDataset(split, sysc=cfg.system, datac=cfg.data,
                       numeric=cfg.numeric, init=cfg.train.init)
    picked = []
    for i in range(len(ds)):
        if lo <= ds.sample(i).snr_db < hi:
            picked.append(ds[i])
            if len(picked) == n:
                break
    return collate(picked), len(picked)


def grads(model, batch, cfg, *, exact: bool):
    model.zero_grad(set_to_none=True)
    model._set_test_mode(exact_hankel_grad=exact)
    G0 = make_initial_batch(batch, cfg.train.init, cfg)
    nmse_loss(model(G0, batch["Z"], batch["S"], batch["B"], batch["sigma2"]),
              batch["G_true"]).backward()
    out = {n: (p.grad.detach().clone() if p.grad is not None
               else torch.zeros_like(p)) for n, p in model.named_parameters()}
    model._set_test_mode(exact_hankel_grad=False)
    return out


def group_of(name: str) -> str:
    if "filter_net" in name:
        return "filter_net"
    if name.endswith(".gate"):
        return "gate"
    return "transformer"


def compare(ge, gs) -> dict:
    """Per (layer, group) cosine similarity and norm ratio, exact vs STE."""
    acc: dict = {}
    for n in ge:
        layer = int(n.split(".")[1])
        key = (layer, group_of(n))
        a, b = acc.setdefault(key, ([], []))
        a.append(ge[n].reshape(-1))
        b.append(gs[n].reshape(-1))
    rows = []
    for (layer, grp), (a, b) in sorted(acc.items()):
        e, s = torch.cat(a), torch.cat(b)
        ne, ns = float(e.norm()), float(s.norm())
        cos = (float(torch.dot(e, s) / (e.norm() * s.norm()))
               if ne > 0 and ns > 0 else float("nan"))
        rows.append({"layer": layer, "group": grp, "cosine": cos,
                     "norm_exact": ne, "norm_ste": ns,
                     "norm_ratio_ste_over_exact": (ns / ne) if ne > 0
                     else float("nan")})
    return rows


def a3(cfg) -> dict:
    """Exact-vs-STE gradient fidelity, low SNR and high SNR separately."""
    H1, rcfg = load_arm(cfg, hankel=True)
    res = {}
    for tag, (lo, hi) in (("low_snr_-10_0", LOW), ("high_snr_10_20", HIGH)):
        batch, n = batch_in_snr_range(cfg, lo, hi, 32)
        # The gaps that make or break the exact gradient. Measured on the LS
        # estimate the projection actually sees in layer 0.
        G0 = make_initial_batch(batch, rcfg.train.init, rcfg)
        gaps = singular_gap_stats(G0, rank=7)
        ge = grads(H1, batch, rcfg, exact=True)
        gs = grads(H1, batch, rcfg, exact=False)
        rows = compare(ge, gs)
        cos = np.array([r["cosine"] for r in rows if np.isfinite(r["cosine"])])
        res[tag] = {
            "n_samples": n, "snr_range": [lo, hi],
            "singular_gaps_layer0_input": gaps,
            "per_layer_group": rows,
            "cosine_min": float(cos.min()), "cosine_median": float(np.median(cos)),
            "cosine_mean": float(cos.mean()),
            "frac_cosine_above_0.9": float(np.mean(cos > 0.9)),
            "frac_cosine_above_0.99": float(np.mean(cos > 0.99)),
        }
    lo_c, hi_c = (res["low_snr_-10_0"]["cosine_median"],
                  res["high_snr_10_20"]["cosine_median"])
    res["verdict"] = {
        "cosine_median_low": lo_c, "cosine_median_high": hi_c,
        "degrades_at_low_snr": bool(lo_c < hi_c - 0.05),
        "high_fidelity_both": bool(min(lo_c, hi_c) > 0.9),
    }
    return res


def a4(cfg) -> dict:
    """Share of training loss and of gradient norm, per SNR bin.

    ``nmse_loss`` is the MEAN over the batch of per-sample ``||dG||^2/||G||^2``,
    so every trial contributes with equal weight to the mean but with wildly
    unequal magnitude: a low-SNR trial's normalized error is far larger.
    """
    out = {}
    for arm, hankel in (("U1_urformer", False), ("H1_hs_urformer", True)):
        model, rcfg = load_arm(cfg, hankel=hankel)
        rows = []
        for lo, hi in BINS:
            batch, n = batch_in_snr_range(cfg, lo, hi, 48, split="train")
            model.zero_grad(set_to_none=True)
            G0 = make_initial_batch(batch, rcfg.train.init, rcfg)
            est = model(G0, batch["Z"], batch["S"], batch["B"], batch["sigma2"])
            num = torch.sum(torch.abs(est - batch["G_true"]) ** 2, dim=(1, 2))
            den = torch.sum(torch.abs(batch["G_true"]) ** 2, dim=(1, 2))
            per = (num / den)
            per.mean().backward()
            gn = float(torch.sqrt(sum((p.grad ** 2).sum()
                                      for p in model.parameters()
                                      if p.grad is not None)))
            rows.append({"bin": [lo, hi], "n": n,
                         "mean_per_sample_nmse": float(per.mean()),
                         "mean_nmse_db": float(10 * np.log10(float(per.mean()))),
                         "grad_norm": gn})
        tot_l = sum(r["mean_per_sample_nmse"] for r in rows)
        tot_g = sum(r["grad_norm"] for r in rows)
        for r in rows:
            r["loss_share"] = r["mean_per_sample_nmse"] / tot_l
            r["grad_share"] = r["grad_norm"] / tot_g
        out[arm] = {"bins": rows,
                    "loss_share_below_5dB": sum(r["loss_share"] for r in rows
                                                if r["bin"][1] <= 5),
                    "grad_share_below_5dB": sum(r["grad_share"] for r in rows
                                                if r["bin"][1] <= 5)}
    return out


def main() -> int:
    torch.set_num_threads(4)
    cfg = _cfg()
    print("=== A3: STE vs exact gradient ===")
    r3 = a3(cfg)
    for tag in ("low_snr_-10_0", "high_snr_10_20"):
        d = r3[tag]
        g = d["singular_gaps_layer0_input"]
        print(f"\n{tag}  (n={d['n_samples']})")
        print(f"  min relative gap at truncation : "
              f"{g['min_relative_gap_at_truncation']:.3e}")
        print(f"  min absolute gap at truncation : "
              f"{g['min_gap_at_truncation']:.3e}  "
              f"(sigma_max median {g['sigma_max_median']:.3e})")
        print(f"  cosine  median {d['cosine_median']:.4f}  min {d['cosine_min']:.4f}"
              f"   >0.9: {d['frac_cosine_above_0.9']:.2f}"
              f"   >0.99: {d['frac_cosine_above_0.99']:.2f}")
    print(f"\n  verdict: {r3['verdict']}")

    print("\n=== A4: loss and gradient share by SNR ===")
    r4 = a4(cfg)
    for arm, d in r4.items():
        print(f"\n{arm}")
        print(f"  {'bin':>12} {'mean NMSE dB':>13} {'loss share':>11} {'grad share':>11}")
        for r in d["bins"]:
            print(f"  [{r['bin'][0]:+3d},{r['bin'][1]:+3d})  {r['mean_nmse_db']:12.2f} "
                  f"{r['loss_share']:11.3f} {r['grad_share']:11.3f}")
        print(f"  share below 5 dB:  loss {d['loss_share_below_5dB']:.3f}   "
              f"grad {d['grad_share_below_5dB']:.3f}")

    OUT.write_text(json.dumps({"A3_ste_fidelity": r3, "A4_loss_by_snr": r4},
                              indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
