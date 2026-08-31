"""Track D stage 4 - gated Hankel (PROMPT 7 Part B) and the [5,20] control (Part C).

Part B, both at 80k / 13 epochs, matched to stage 3's H1 and stage 2's U1 on
seed, initialization, data order and schedule:

    G1   beta_t = sigmoid(g_t)                 one learned scalar per layer
    G2   beta_t = sigmoid(MLP_t(log sigma^2))  conditioned on the noise level

Part C, the mechanism probe:

    C_U1   URformer,    trained on SNR in [5, 20] ONLY
    C_H1   HS-URformer, trained on SNR in [5, 20] ONLY

**Part C is not a performance claim.** Per PROMPT 7 standing rule 2, a narrower
SNR range chosen after seeing the answer is exactly what pre-registration
prevents, and these numbers are never HS-URformer's headline. They exist to
separate two causes of H1's low-SNR damage that A3 and A4 left entangled: loss
weighting (A4: 86% of H1's gradient norm comes from below 5 dB) versus the STE
or the constraint itself (A3: gradient cosine 0.630 low / 0.821 high, poor in
both). Training on [5, 20] removes the loss-weighting confound. Its arms are
trained AND tested on their own [5, 20] worlds, so they are comparable only to
each other, never to the stage-3 table.

Primary presentation everywhere: **Delta(SNR) per bin, paired per-trial
median.** A pooled scalar is reported for continuity and carries no decision
weight.

Run:  PYTHONPATH=. python3 -m trackD_urformer.stage4 --i-have-approval
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrackDConfig
from .dataset import TrackDDataset, collate
from .stage1 import build_model, db
from .stage2 import SELECTION_RULE, select_epoch
from .stage3 import paired
from .train import make_initial_batch, nmse_loss

RESULTS = Path("results") / "track_d" / "stage4"
REPORTS = Path("reports")
STAGE2_U1 = Path("results/track_d/stage2/B3_80k_13ep/best.pt")
STAGE3_H1 = Path("results/track_d/stage3/H1_hs_urformer_80k/best.pt")

N_TRAIN, EPOCHS, HANKEL_RANK = 80_000, 13, 7
FULL_SNR = (-10.0, 20.0)
HIGH_SNR = (5.0, 20.0)
BINS = [(-10, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]

RUNS: dict[str, dict] = {
    "G1_gate_scalar":  {"hankel": True,  "gate": "scalar", "snr": FULL_SNR},
    "G2_gate_snr":     {"hankel": True,  "gate": "snr",    "snr": FULL_SNR},
    "C_U1_snr5_20":    {"hankel": False, "gate": "none",   "snr": HIGH_SNR},
    "C_H1_snr5_20":    {"hankel": True,  "gate": "none",   "snr": HIGH_SNR},
}
PART_B = ["G1_gate_scalar", "G2_gate_snr"]
PART_C = ["C_U1_snr5_20", "C_H1_snr5_20"]


def run_cfg(cfg: TrackDConfig, spec: dict) -> TrackDConfig:
    return replace(
        cfg,
        data=replace(cfg.data, n_train=N_TRAIN, snr_range_db=spec["snr"]),
        model=replace(cfg.model, filter_init="random", use_transformer=True,
                      use_hankel=spec["hankel"], hankel_rank=HANKEL_RANK,
                      hankel_mode="fixed", hankel_gate=spec["gate"]))


def build_arm(cfg: TrackDConfig, name: str):
    m, meta = build_model(run_cfg(cfg, RUNS[name]), "arm1b_full_random")
    return m, {**meta, "arm": name, "spec": RUNS[name]}


@torch.no_grad()
def validation_per_trial(model, loader, cfg) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        G0 = make_initial_batch(batch, cfg.train.init, cfg)
        est = model(G0, batch["Z"], batch["S"], batch["B"], batch["sigma2"])
        num = torch.sum(torch.abs(est - batch["G_true"]) ** 2, dim=(1, 2))
        den = torch.sum(torch.abs(batch["G_true"]) ** 2, dim=(1, 2))
        out.extend((num / den).tolist())
    model.train()
    return np.asarray(out, dtype=np.float64)


def beta_curve(model, cfg) -> dict:
    """The learned ``beta_t(sigma^2)`` per layer -- P8's deliverable.

    Probed on the sigma^2 values the SNR grid actually produces, so the curve
    is reported over the range the gate was trained on and not extrapolated.
    """
    snrs = np.array([-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0,
                     12.5, 15.0, 17.5, 20.0])
    # sigma^2 as the generator sets it: measured pairs from a probe dataset.
    ds = TrackDDataset("val", sysc=cfg.system, datac=cfg.data,
                       numeric=cfg.numeric, cache=False)
    obs = {}
    for i in range(min(400, len(ds))):
        s = ds.sample(i)
        obs.setdefault(round(s.snr_db), float(s.sigma2))
    fit = np.polyfit(list(obs), np.log(list(obs.values())), 1)
    sig2 = np.exp(np.polyval(fit, snrs))
    t = torch.as_tensor(sig2, dtype=cfg.numeric.real_dtype)
    out = {"snr_db": snrs.tolist(), "sigma2": sig2.tolist(), "per_layer": []}
    with torch.no_grad():
        for li, lay in enumerate(model.layers):
            if lay.hankel_gate is None:
                out["per_layer"].append({"layer": li, "beta": None})
            else:
                b = lay.hankel_gate(t).reshape(-1).double().tolist()
                out["per_layer"].append({"layer": li, "beta": b})
    return out


def train_run(cfg: TrackDConfig, name: str, out_dir: Path) -> dict:
    spec = RUNS[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.train.num_threads)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    rcfg = run_cfg(cfg, spec)

    train_ds = TrackDDataset("train", sysc=rcfg.system, datac=rcfg.data,
                             numeric=rcfg.numeric, init=rcfg.train.init)
    val_ds = TrackDDataset("val", sysc=rcfg.system, datac=rcfg.data,
                           numeric=rcfg.numeric, init=rcfg.train.init)
    assert len(val_ds) == cfg.data.n_val
    train_ld = DataLoader(train_ds, batch_size=rcfg.train.batch_size,
                          shuffle=True, collate_fn=collate, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=rcfg.train.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model, meta = build_arm(cfg, name)
    opt = torch.optim.Adam(model.parameters(), lr=rcfg.train.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    probe = torch.tensor([0.2], dtype=rcfg.numeric.real_dtype)
    meta["initial_betas"] = model.initial_betas(probe)

    ckpt, csv_path = out_dir / "checkpoint.pt", out_dir / "curves.csv"
    vpt_path = out_dir / "val_per_trial.npy"
    start, history, vpt = 0, [], []
    if ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        sched.load_state_dict(blob["scheduler"])
        start, history = blob["epoch"] + 1, blob.get("history", [])
        if vpt_path.exists():
            vpt = list(np.load(vpt_path))
        print(f"  [{name}] resumed at epoch {start}")
    if start == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["epoch", "train_loss_db", "val_nmse_db",
                                     "lr", "seconds", "mean_beta"])

    for epoch in range(start, EPOCHS):
        t0, run_loss, nb = time.time(), 0.0, 0
        for batch in train_ld:
            opt.zero_grad()
            G0 = make_initial_batch(batch, rcfg.train.init, rcfg)
            loss = nmse_loss(model(G0, batch["Z"], batch["S"], batch["B"],
                                   batch["sigma2"]), batch["G_true"])
            loss.backward()
            if rcfg.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               rcfg.train.grad_clip)
            opt.step()
            run_loss += float(loss.detach())
            nb += 1
        sched.step()

        v = validation_per_trial(model, val_ld, rcfg)
        vpt.append(v)
        np.save(vpt_path, np.asarray(vpt))
        mb = float(np.mean(model.initial_betas(probe)))
        rec = {"epoch": epoch, "train_loss_db": db(run_loss / max(nb, 1)),
               "val_nmse_db": db(v.mean()), "lr": sched.get_last_lr()[0],
               "seconds": round(time.time() - t0, 1), "mean_beta": mb}
        history.append(rec)
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([rec[k] for k in
                                     ("epoch", "train_loss_db", "val_nmse_db",
                                      "lr", "seconds", "mean_beta")])
        torch.save({"model": model.state_dict()}, out_dir / f"ep{epoch:03d}.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "history": history, "config": rcfg.to_dict(),
                    "run": name}, ckpt)
        print(f"  [{name}] epoch {epoch:3d}  train {rec['train_loss_db']:7.2f} dB"
              f"  val {rec['val_nmse_db']:7.2f} dB  {rec['seconds']:6.1f}s"
              f"  beta {mb:.3f}", flush=True)

    val_curve = np.array([r["val_nmse_db"] for r in history])
    vpt_arr = np.asarray(vpt)
    chosen, sel = select_epoch(val_curve, vpt_arr[int(np.argmin(val_curve))],
                               SELECTION_RULE)
    blob = torch.load(out_dir / f"ep{chosen:03d}.pt", map_location="cpu",
                      weights_only=False)
    torch.save({"model": blob["model"], "epoch": chosen,
                "config": rcfg.to_dict()}, out_dir / "best.pt")
    model.load_state_dict(blob["model"])
    return {**meta, "run": name, "n_train": N_TRAIN, "epochs": EPOCHS,
            "history": history, "selection": sel, "chosen_epoch": chosen,
            "chosen_val_db": float(val_curve[chosen]),
            "best_val_db": float(val_curve.min()),
            "best_path": str(out_dir / "best.pt"),
            "beta_curve": beta_curve(model, rcfg)}


# ------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(cfg: TrackDConfig, names: list[str], snr_range, n_test: int,
             include_stage3: bool) -> dict:
    """One pass over the test set for this SNR range; all arms, identical worlds."""
    from .baselines import nmse_parts

    ecfg = replace(cfg, data=replace(cfg.data, snr_range_db=snr_range))
    ds = TrackDDataset("test", sysc=ecfg.system, datac=ecfg.data,
                       numeric=ecfg.numeric, init=ecfg.train.init)
    cd, rd = ecfg.numeric.complex_dtype, ecfg.numeric.real_dtype

    models = {}
    for nm in names:
        m, _ = build_arm(cfg, nm)
        m.load_state_dict(torch.load(RESULTS / nm / "best.pt",
                                     map_location="cpu",
                                     weights_only=False)["model"])
        models[nm] = m.eval()
    if include_stage3:
        for nm, path, hk in (("U1_urformer_80k", STAGE2_U1, False),
                             ("H1_hs_urformer_80k", STAGE3_H1, True)):
            rc = replace(cfg, model=replace(cfg.model, filter_init="random",
                                            use_transformer=True,
                                            use_hankel=hk, hankel_rank=7,
                                            hankel_mode="fixed",
                                            hankel_gate="none"))
            m, _ = build_model(rc, "arm1b_full_random")
            m.load_state_dict(torch.load(path, map_location="cpu",
                                         weights_only=False)["model"])
            models[nm] = m.eval()

    per = {k: [] for k in models}
    snr = []
    T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)
    t0 = time.time()
    for i in range(n_test):
        s = ds.sample(i)
        snr.append(s.snr_db)
        G0, Z = T(ds.g0(i), cd), T(s.Z, rd)
        S, B = T(s.S, cd), T(s.B, cd)
        s2 = torch.tensor([s.sigma2], dtype=rd)
        for nm, m in models.items():
            gh = m(G0, Z, S, B, s2)[0].detach().numpy()
            e, d_ = nmse_parts(gh, s.G_true)
            per[nm].append(e / d_)
        if (i + 1) % 400 == 0:
            el = time.time() - t0
            print(f"  eval {i+1}/{n_test}  {el/60:.1f} min, "
                  f"{el/(i+1)*(n_test-i-1)/60:.1f} min left", flush=True)

    return {"n_test": n_test, "snr_range": list(snr_range), "snr_db": snr,
            "per_trial_nmse": {k: v for k, v in per.items()},
            "methods": {k: {"nmse_median_db": db(np.median(np.array(v))),
                            "nmse_ratio_of_sums_db": db(np.mean(np.array(v)))}
                        for k, v in per.items()}}


def by_bin(per: dict, a: str, b: str, snr: np.ndarray) -> dict:
    """``Delta(SNR)`` per bin plus the two success-criterion aggregates.

    PRIMARY presentation. The pooled value is carried for continuity with
    stage 3 and is explicitly labelled as sampling-design-dependent.
    """
    out = {"contrast": f"{a} - {b}", "bins": []}
    for lo, hi in BINS:
        m = (snr >= lo) & (snr < hi)
        if m.sum() < 20:
            continue
        sub = {k: list(np.asarray(per[k])[m]) for k in (a, b)}
        r = paired(sub, a, b)
        out["bins"].append({"bin": [lo, hi], "n": int(m.sum()),
                            "median_diff_db": r["median_diff_db"],
                            "boot_ci95_median": r["boot_ci95_median"],
                            "ci_excludes_zero": r["ci_excludes_zero"]})
    for tag, m in (("high_snr_ge5", snr >= 5), ("low_snr_lt5", snr < 5)):
        if m.sum() < 20:
            continue
        sub = {k: list(np.asarray(per[k])[m]) for k in (a, b)}
        out[tag] = paired(sub, a, b)
    out["pooled_SAMPLING_DESIGN_DEPENDENT"] = paired(per, a, b)
    return out


def criterion(agg: dict) -> dict:
    """B4's two-part success criterion, each condition reported separately."""
    hi = agg.get("high_snr_ge5")
    lo = agg.get("low_snr_lt5")
    if hi is None or lo is None:
        return {"applicable": False,
                "reason": "criterion needs both SNR halves; this evaluation "
                          "does not span them"}
    c1 = hi["median_diff_db"] >= 0.85
    c2 = lo["median_diff_db"] >= -0.05
    return {
        "applicable": True,
        "condition_i_high_snr": {
            "threshold_db": 0.85, "measured_db": hi["median_diff_db"],
            "ci95": hi["boot_ci95_median"], "pass": bool(c1),
            "pct_of_stage3_high_snr_gain": hi["median_diff_db"] / 1.209},
        "condition_ii_low_snr": {
            "threshold_db": -0.05, "measured_db": lo["median_diff_db"],
            "ci95": lo["boot_ci95_median"], "pass": bool(c2)},
        "success": bool(c1 and c2),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track D stage 4 (PROMPT 7)")
    ap.add_argument("--i-have-approval", action="store_true")
    ap.add_argument("--runs", type=str, default=",".join(RUNS))
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.i_have_approval:
        print("REFUSING: stage 4 requires --i-have-approval (PROMPT 7).")
        return 2
    a.eval = a.eval or a.eval_only

    cfg = TrackDConfig()
    cfg = replace(cfg, train=replace(cfg.train, init="spectral"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "trackD_stage4_results.json"

    def merge_write(**up):
        cur = (json.loads(path.read_text()) if path.exists()
               else {"config": cfg.to_dict(), "runs": {}})
        for k, v in up.items():
            if k == "runs":
                cur.setdefault("runs", {}).update(v)
            else:
                cur[k] = v
        path.write_text(json.dumps(cur, indent=2) + "\n", encoding="utf-8")
        return cur

    summary = merge_write()
    if not a.eval_only:
        for nm in [r.strip() for r in a.runs.split(",") if r.strip()]:
            print(f"\n=== {nm} ===", flush=True)
            t0 = time.time()
            info = train_run(cfg, nm, RESULTS / nm)
            info["train_seconds"] = round(time.time() - t0, 1)
            summary = merge_write(runs={nm: info})
            print(f"  [{nm}] done in {info['train_seconds']}s, epoch "
                  f"{info['chosen_epoch']}, val {info['chosen_val_db']:.3f} dB")

    if a.eval:
        if set(PART_B) <= set(summary["runs"]):
            print("\n=== Part B evaluation (full SNR test set) ===", flush=True)
            tb = evaluate(cfg, PART_B, FULL_SNR, cfg.data.n_test, True)
            snr = np.asarray(tb["snr_db"])
            tb["contrasts"] = {
                f"{g}_vs_U1": by_bin(tb["per_trial_nmse"], "U1_urformer_80k",
                                     g, snr) for g in PART_B}
            tb["contrasts"]["H1_vs_U1_recheck"] = by_bin(
                tb["per_trial_nmse"], "U1_urformer_80k",
                "H1_hs_urformer_80k", snr)
            tb["criterion"] = {g: criterion(tb["contrasts"][f"{g}_vs_U1"])
                               for g in PART_B}
            summary = merge_write(part_b=tb)
            for g in PART_B:
                c = tb["criterion"][g]
                print(f"\n  {g}:")
                for r in tb["contrasts"][f"{g}_vs_U1"]["bins"]:
                    print(f"    [{r['bin'][0]:+3d},{r['bin'][1]:+3d})  "
                          f"{r['median_diff_db']:+7.3f} "
                          f"[{r['boot_ci95_median'][0]:+.3f},"
                          f"{r['boot_ci95_median'][1]:+.3f}]")
                print(f"    (i)  SNR>=5 {c['condition_i_high_snr']['measured_db']:+.3f} "
                      f">= +0.85 ? {c['condition_i_high_snr']['pass']}")
                print(f"    (ii) SNR<5  {c['condition_ii_low_snr']['measured_db']:+.3f} "
                      f">= -0.05 ? {c['condition_ii_low_snr']['pass']}")
                print(f"    SUCCESS: {c['success']}")

        if set(PART_C) <= set(summary["runs"]):
            print("\n=== Part C evaluation ([5,20] test set) ===", flush=True)
            tc = evaluate(cfg, PART_C, HIGH_SNR, cfg.data.n_test, False)
            snr = np.asarray(tc["snr_db"])
            tc["contrasts"] = {"C_H1_vs_C_U1": by_bin(
                tc["per_trial_nmse"], "C_U1_snr5_20", "C_H1_snr5_20", snr)}
            tc["note"] = (
                "MECHANISM PROBE ONLY. Trained and tested on SNR in [5,20]; "
                "comparable only to each other, never to the stage-3 table. "
                "Registered operating range remains [-10,20].")
            summary = merge_write(part_c=tc)
            ct = tc["contrasts"]["C_H1_vs_C_U1"]
            for r in ct["bins"]:
                print(f"    [{r['bin'][0]:+3d},{r['bin'][1]:+3d})  "
                      f"{r['median_diff_db']:+7.3f} "
                      f"[{r['boot_ci95_median'][0]:+.3f},"
                      f"{r['boot_ci95_median'][1]:+.3f}]")
            p = ct["pooled_SAMPLING_DESIGN_DEPENDENT"]
            print(f"    over [5,20]: {p['median_diff_db']:+.3f} "
                  f"[{p['boot_ci95_median'][0]:+.3f},"
                  f"{p['boot_ci95_median'][1]:+.3f}]   "
                  f"(stage 3 H1 on the same range: +1.209)")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
