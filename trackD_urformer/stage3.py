"""Track D stage 3 - HS-URformer, the two gate runs (PROMPT 6 Part B).

Two trained arms, matched to the stage-2 80k URformer in every respect except
the one thing under test:

    H1   HS-URformer, 80k / 13 ep, internal Hankel projection, r = 7 fixed
    X1   converged EM-GS (T_GS=100) -> ONE Transformer post-processor,
         no unrolling, 158,592 parameters

and four arms that cost no training at all:

    U0        EM-GS-spectral, T_GS = 100                     (classical floor)
    H0        HS-EM-GS from Track B, L_hat = 7, T = 100       (classical + prior)
    U1        the stage-2 80k URformer, reloaded              (the comparison)
    U1+post   U1's output, projected ONCE at the end          (post-hoc control)

Primary statistic, held everywhere: **the paired per-trial median.**
Ratio-of-sums is computed and reported too, but it is labelled secondary and
carries an argument only for Q4, where the tail is the question. Reading a
difference between the two as a difference between conditions is exactly the
mistake that produced the PROMPT 5 "reversal", and it is not repeated here.

    Delta_H = NMSE_U1 - NMSE_H1   in dB, paired per trial.  POSITIVE = Hankel helps.

Go / no-go, fixed before the runs (reports/trackD_hankel_prereg.md):

    Delta_H >= +0.3 dB, CI excluding zero  ->  Part C
    0 < Delta_H < +0.3 dB                  ->  stop and report (marginal)
    Delta_H <= 0                           ->  stop and report

Why H1 can be compared to a checkpoint trained in stage 2
---------------------------------------------------------
Gate A4 established that the stage-3 train/val/test splits are byte-identical
to stage 2's: same seed ranges, same fixed pilot matrix, same SNR draw, same
channel realizations. U1 is therefore the *same* experiment as H1 with the
projection switched off, and re-training it would only add sampling noise.
:func:`assert_dataset_identity` re-checks this at startup rather than trusting
the earlier gate.

Run:  PYTHONPATH=. python3 -m trackD_urformer.stage3 --i-have-approval
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import TrackDConfig
from .dataset import TrackDDataset, collate
from .stage1 import build_model, db
from .stage2 import SELECTION_RULE, select_epoch
from .train import make_initial_batch, nmse_loss
from .transformer import ChannelTransformer

RESULTS = Path("results") / "track_d" / "stage3"
REPORTS = Path("reports")
STAGE2 = Path("results") / "track_d" / "stage2"

N_TRAIN = 80_000
EPOCHS = 13
T_GS = 100
HANKEL_RANK = 7          # L_max: a system design assumption, NOT oracle info

RUNS: dict[str, dict] = {
    "H1_hs_urformer_80k": {"kind": "urformer", "use_hankel": True},
    "X1_emgs_plus_former": {"kind": "postproc", "use_hankel": False},
}


# ---------------------------------------------------------------- X1 model
class PostProcessor(nn.Module):
    """One Transformer residual on a fixed classical estimate. That is all.

    ``zero_init_out=True`` means at initialization this returns its input
    unchanged, so X1 starts *exactly* as converged EM-GS -- the same
    degeneration property gate F asserts for the URformer layer, and it makes
    "X1 beat EM-GS" a statement about the learned residual alone.
    """

    def __init__(self, N: int, K: int, mcfg) -> None:
        super().__init__()
        self.former = ChannelTransformer(
            N=N, K=K, d_model=mcfg.d_model, L_enc=mcfg.L_enc,
            n_heads=mcfg.n_heads, ffn_mult=mcfg.ffn_mult, dropout=mcfg.dropout,
            zero_init_out=True)

    def forward(self, G_in: torch.Tensor) -> torch.Tensor:
        return G_in + self.former(G_in)


class EMGSFeatureDataset(Dataset):
    """``TrackDDataset`` plus the cached converged-EM-GS estimate as ``G_in``.

    The underlying worlds are untouched; this only attaches a precomputed
    column that :mod:`emgs_cache` proved bit-identical to computing it inline.
    """

    def __init__(self, base: TrackDDataset, feats: np.ndarray) -> None:
        if len(base) != feats.shape[0]:
            raise ValueError(
                f"EM-GS cache has {feats.shape[0]} rows but the split has "
                f"{len(base)}; the cache was built for a different budget")
        self.base, self.feats = base, feats

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.base[idx]
        item["G_in"] = torch.as_tensor(np.array(self.feats[idx], copy=True),
                                       dtype=self.base.numeric.complex_dtype)
        return item


# ------------------------------------------------------------ shared pieces
def estimate(model, batch, cfg, kind: str) -> torch.Tensor:
    """One forward pass, whichever arm this is."""
    if kind == "postproc":
        return model(batch["G_in"])
    G0 = make_initial_batch(batch, cfg.train.init, cfg)
    return model(G0, batch["Z"], batch["S"], batch["B"], batch["sigma2"])


@torch.no_grad()
def validation_per_trial(model, loader, cfg, kind: str) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        est = estimate(model, batch, cfg, kind)
        num = torch.sum(torch.abs(est - batch["G_true"]) ** 2, dim=(1, 2))
        den = torch.sum(torch.abs(batch["G_true"]) ** 2, dim=(1, 2))
        out.extend((num / den).tolist())
    model.train()
    return np.asarray(out, dtype=np.float64)


def assert_dataset_identity(cfg: TrackDConfig) -> dict:
    """Re-derive gate A4 rather than trusting it: are the splits stage 2's?"""
    s2 = json.loads((REPORTS / "trackD_stage2_results.json").read_text())["config"]
    now = cfg.to_dict()
    fields = ["val_seed_range", "test_seed_range", "train_seed_range", "n_val",
              "n_test", "pilot_mode", "fixed_S_seed", "snr_mode",
              "snr_range_db", "rsr_train_mode"]
    diffs = {}
    # JSON turns tuples into lists; normalise both sides before comparing, or
    # every tuple-valued field reports a spurious difference -- the A4 false
    # positive from PROMPT 6 Part A.
    norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v
    for f in fields:
        a, b = norm(s2["data"][f]), norm(now["data"][f])
        if a != b:
            diffs[f] = {"stage2": a, "stage3": b}
    for f in ["master_seed", "N", "K", "P", "L_min", "L_max", "rsr_db"]:
        if s2["system"][f] != now["system"][f]:
            diffs[f] = {"stage2": s2["system"][f], "stage3": now["system"][f]}
    if diffs:
        raise AssertionError(
            f"stage 3 splits differ from stage 2: {diffs}. U1 cannot be reused "
            "as the comparison arm; retrain it or fix the config.")
    return {"identical_to_stage2": True, "fields_checked": fields}


def build_arm(cfg: TrackDConfig, name: str):
    """Construct the model for one stage-3 arm, plus its metadata."""
    spec = RUNS[name]
    if spec["kind"] == "postproc":
        m = PostProcessor(cfg.system.N, cfg.system.K, cfg.model)
        n_par = sum(p.numel() for p in m.parameters())
        return m, {"arm": name, "kind": "postproc", "total_params": n_par,
                   "unrolled_layers": 0, "front_end": f"EM-GS T_GS={T_GS}"}
    rcfg = replace(cfg, model=replace(
        cfg.model, filter_init="random", use_transformer=True,
        use_hankel=spec["use_hankel"], hankel_rank=HANKEL_RANK,
        hankel_mode="fixed"))
    m, meta = build_model(rcfg, "arm1b_full_random")
    meta = {**meta, "arm": name, "kind": "urformer",
            "use_hankel": spec["use_hankel"], "hankel_rank": HANKEL_RANK}
    return m, meta


def train_run(cfg: TrackDConfig, name: str, out_dir: Path) -> dict:
    """Train one stage-3 arm. Matched to stage-2 B3: 80k, 13 epochs, same seed."""
    spec = RUNS[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.train.num_threads)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    rcfg = replace(cfg, data=replace(cfg.data, n_train=N_TRAIN),
                   model=replace(cfg.model, filter_init="random",
                                 use_transformer=True,
                                 use_hankel=spec["use_hankel"],
                                 hankel_rank=HANKEL_RANK, hankel_mode="fixed"))
    kind = spec["kind"]
    need_init = None if kind == "postproc" else rcfg.train.init

    train_ds = TrackDDataset("train", sysc=rcfg.system, datac=rcfg.data,
                             numeric=rcfg.numeric, init=need_init)
    val_ds = TrackDDataset("val", sysc=rcfg.system, datac=rcfg.data,
                           numeric=rcfg.numeric, init=need_init)
    assert len(val_ds) == cfg.data.n_val
    assert val_ds.sample(0).trial == cfg.data.val_seed_range[0]

    if kind == "postproc":
        from .emgs_cache import load_cache
        train_ds = EMGSFeatureDataset(
            train_ds, load_cache("train", N_TRAIN, T_GS, n_shards=3))
        val_ds = EMGSFeatureDataset(
            val_ds, load_cache("val", cfg.data.n_val, T_GS, n_shards=1))

    train_ld = DataLoader(train_ds, batch_size=rcfg.train.batch_size,
                          shuffle=True, collate_fn=collate, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=rcfg.train.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model, meta = build_arm(cfg, name)
    opt = torch.optim.Adam(model.parameters(), lr=rcfg.train.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

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
                                     "lr", "seconds"])

    for epoch in range(start, EPOCHS):
        t0, run_loss, nb = time.time(), 0.0, 0
        for batch in train_ld:
            opt.zero_grad()
            loss = nmse_loss(estimate(model, batch, rcfg, kind),
                             batch["G_true"])
            loss.backward()
            if rcfg.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               rcfg.train.grad_clip)
            opt.step()
            run_loss += float(loss.detach())
            nb += 1
        sched.step()

        v = validation_per_trial(model, val_ld, rcfg, kind)
        vpt.append(v)
        np.save(vpt_path, np.asarray(vpt))
        rec = {"epoch": epoch, "train_loss_db": db(run_loss / max(nb, 1)),
               "val_nmse_db": db(v.mean()), "lr": sched.get_last_lr()[0],
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([rec[k] for k in ("epoch", "train_loss_db",
                                                      "val_nmse_db", "lr",
                                                      "seconds")])
        torch.save({"model": model.state_dict()}, out_dir / f"ep{epoch:03d}.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "history": history, "config": rcfg.to_dict(),
                    "run": name}, ckpt)
        print(f"  [{name}] epoch {epoch:3d}  train {rec['train_loss_db']:7.2f} dB"
              f"  val {rec['val_nmse_db']:7.2f} dB  {rec['seconds']:6.1f}s",
              flush=True)

    val_curve = np.array([r["val_nmse_db"] for r in history])
    vpt_arr = np.asarray(vpt)
    chosen, sel = select_epoch(val_curve, vpt_arr[int(np.argmin(val_curve))],
                               SELECTION_RULE)
    blob = torch.load(out_dir / f"ep{chosen:03d}.pt", map_location="cpu",
                      weights_only=False)
    torch.save({"model": blob["model"], "epoch": chosen,
                "config": rcfg.to_dict()}, out_dir / "best.pt")
    return {**meta, "run": name, "n_train": N_TRAIN, "epochs": EPOCHS,
            "sample_passes": N_TRAIN * EPOCHS, "history": history,
            "selection": sel, "chosen_epoch": chosen,
            "chosen_val_db": float(val_curve[chosen]),
            "best_val_db": float(val_curve.min()),
            "best_path": str(out_dir / "best.pt")}


# ------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_stage3(cfg: TrackDConfig, n_test: int, arms: dict) -> dict:
    """ONE pass over the FIXED test set; every arm sees identical worlds."""
    from rydberg_sim.forward import exact_forward
    from rydberg_sim.rng import get_operating_point_rngs
    from rydberg_sim.track_b_proposed import hs_gs

    from .baselines import nmse_parts, run_em_gs
    from .hankel import project_G
    from .torch_forward import least_squares_G

    ds = TrackDDataset("test", sysc=cfg.system, datac=cfg.data,
                       numeric=cfg.numeric, init=cfg.train.init)
    cd, rd = cfg.numeric.complex_dtype, cfg.numeric.real_dtype

    # U1: the stage-2 80k URformer, reloaded, Hankel OFF.
    u1cfg = replace(cfg, model=replace(cfg.model, filter_init="random",
                                       use_transformer=True, use_hankel=False))
    U1, _ = build_model(u1cfg, "arm1b_full_random")
    U1.load_state_dict(torch.load(STAGE2 / "B3_80k_13ep" / "best.pt",
                                  map_location="cpu",
                                  weights_only=False)["model"])
    U1.eval()

    trained = {"U1_urformer_80k": ("urformer", U1)}
    for name, info in arms.items():
        m, _ = build_arm(cfg, name)
        m.load_state_dict(torch.load(info["best_path"], map_location="cpu",
                                     weights_only=False)["model"])
        m.eval()
        trained[name] = (RUNS[name]["kind"], m)

    # A single sweep of H^-1.Pi_r.H is not a projection: it leaves ~4% of the
    # column's Hankel energy above rank 7 (see hankel.py). H1 runs at
    # hankel_iters=1 because that is the specified operator, so a small
    # Delta_H could mean "the prior does not help" OR "the prior was barely
    # imposed". Sweeping n_iter on the training-free post-hoc arm separates
    # those two readings at no training cost.
    POST_ITERS = (1, 2, 4, 8)
    post_keys = [f"U1_plus_post_it{k}" for k in POST_ITERS]

    keys = ["U0_em_gs", "H0_hs_em_gs", "oracle_phase", "U1_urformer_80k",
            "U1_plus_post", "H1_hs_urformer_80k", "X1_emgs_plus_former",
            *post_keys]
    per = {k: [] for k in keys}
    en = {k: [0.0, 0.0] for k in keys}
    snr, Lk = [], []

    T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)
    t0 = time.time()
    for i in range(n_test):
        s = ds.sample(i)
        snr.append(s.snr_db)
        Lk.append(s.L_k)
        est = {}
        est["U0_em_gs"] = run_em_gs(s, max_iter=T_GS, init="spectral",
                                    seed=s.trial)
        est["H0_hs_em_gs"] = hs_gs(s.S, s.Z, s.B, s.sigma2, L_hat=HANKEL_RANK,
                                   exact_step="em_gs", max_iter=T_GS).G_hat

        rngs = get_operating_point_rngs(cfg.system.master_seed, s.trial,
                                        s.snr_db, s.rsr_db)
        ex = exact_forward(s.G_true, s.S, s.B, s.sigma2, rng_noise=rngs.noise)
        of = s.Z * np.exp(1j * np.angle(np.asarray(ex.E)))
        est["oracle_phase"] = least_squares_G(
            T(of, torch.complex128) - T(s.B, torch.complex128),
            T(s.S, torch.complex128))[0].numpy()

        G0, Z = T(ds.g0(i), cd), T(s.Z, rd)
        S, B = T(s.S, cd), T(s.B, cd)
        s2 = torch.tensor([s.sigma2], dtype=rd)
        batch = {"G0": G0, "Z": Z, "S": S, "B": B, "sigma2": s2,
                 "G_in": T(est["U0_em_gs"], cd)}
        for name, (kind, m) in trained.items():
            est[name] = estimate(m, batch, cfg, kind)[0].detach().numpy()

        # U1+post: the SAME U1 output with the projection applied at the end.
        # The post-hoc control for Q6; identical operator, different placement.
        u1t = T(est["U1_urformer_80k"], torch.complex128)
        for k in POST_ITERS:
            est[f"U1_plus_post_it{k}"] = project_G(
                u1t, rank=HANKEL_RANK, mode="fixed", n_iter=k)[0].numpy()
        # The headline post-hoc arm is iteration-matched to H1 (n_iter=1).
        est["U1_plus_post"] = est["U1_plus_post_it1"]

        for k in keys:
            e, d_ = nmse_parts(est[k], s.G_true)
            en[k][0] += e
            en[k][1] += d_
            per[k].append(e / d_)
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  eval {i+1}/{n_test}  {el/60:.1f} min, "
                  f"{el/(i+1)*(n_test-i-1)/60:.1f} min left", flush=True)

    out = {"n_test": n_test, "methods": {}, "snr_db": snr,
           "L_k": [list(x) for x in Lk], "per_trial_nmse": {k: per[k]
                                                            for k in keys}}
    for k in keys:
        a = np.array(per[k])
        out["methods"][k] = {
            "nmse_median_db": db(np.median(a)),                    # PRIMARY
            "nmse_ratio_of_sums_db": db(en[k][0] / en[k][1]),      # secondary
            "nmse_mean_of_ratios_db": db(a.mean()),
        }
    return out


def paired(per: dict, a: str, b: str, seed: int = 20260830, n_boot: int = 4000
           ) -> dict:
    """``NMSE_a - NMSE_b`` in dB per trial. Positive => b is better than a.

    With ``a='U1...'`` and ``b='H1...'`` this is exactly ``Delta_H`` as
    pre-registered: positive means the Hankel projection helps.
    """
    d = 10 * np.log10(np.array(per[a])) - 10 * np.log10(np.array(per[b]))
    rng = np.random.default_rng(seed)
    boot = np.array([np.median(rng.choice(d, d.size, replace=True))
                     for _ in range(n_boot)])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    # Ratio-of-sums version of the same contrast. SECONDARY, and only argued
    # for Q4, where the low-SNR tail is the question being asked.
    ros = 10 * np.log10(np.sum(per[a]) / np.sum(per[b]))
    return {"contrast": f"{a} - {b}", "median_diff_db": float(np.median(d)),
            "mean_diff_db": float(d.mean()),
            "std_diff_db": float(d.std(ddof=1)),
            "boot_ci95_median": [lo, hi],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "win_rate_b": float(np.mean(d > 0)), "n": int(d.size),
            "ratio_of_sums_diff_db_SECONDARY": float(ros),
            "percentiles_db": {p: float(np.percentile(d, p))
                               for p in (5, 25, 50, 75, 95)}}


def verdict(delta: dict) -> dict:
    """The pre-registered go/no-go rule, applied verbatim."""
    m, lo, hi = (delta["median_diff_db"], *delta["boot_ci95_median"])
    excl = lo > 0 or hi < 0
    if m >= 0.3 and excl:
        d = ("GO", "Delta_H >= +0.3 dB with a CI excluding zero -> Part C "
                   "(budget sweep) is authorised by the pre-registered rule.")
    elif m > 0:
        d = ("STOP-MARGINAL", f"0 < Delta_H = {m:+.3f} dB < +0.3 dB -> stop "
                              "and report. Part C is NOT launched.")
    else:
        d = ("STOP-NULL", f"Delta_H = {m:+.3f} dB <= 0 -> stop and report; the "
                          "explicit prior does not help on top of the "
                          "implicit one.")
    return {"decision": d[0], "reason": d[1], "delta_h_db": m,
            "ci95": [lo, hi], "ci_excludes_zero": bool(excl),
            "threshold_db": 0.3,
            "rule_source": "reports/trackD_hankel_prereg.md (committed 7775265)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track D stage 3 (PROMPT 6 Part B)")
    ap.add_argument("--i-have-approval", action="store_true")
    ap.add_argument("--runs", type=str, default=",".join(RUNS))
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_have_approval:
        print("REFUSING: stage 3 requires --i-have-approval (PROMPT 6 Part B).")
        return 2

    cfg = TrackDConfig()
    cfg = replace(cfg, train=replace(cfg.train, init="spectral"))
    ident = assert_dataset_identity(cfg)
    print(f"dataset identity vs stage 2: {ident['identical_to_stage2']}")
    print(f"selection rule: {SELECTION_RULE}   Hankel rank r={HANKEL_RANK} "
          f"(fixed, = L_max, not oracle)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "trackD_stage3_results.json"
    summary = json.loads(path.read_text()) if path.exists() else {
        "config": cfg.to_dict(), "dataset_identity": ident, "runs": {}}

    if not args.eval_only:
        for name in [r.strip() for r in args.runs.split(",") if r.strip()]:
            print(f"\n=== {name} ===")
            t0 = time.time()
            info = train_run(cfg, name, RESULTS / name)
            info["train_seconds"] = round(time.time() - t0, 1)
            summary["runs"][name] = info
            path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
            print(f"  [{name}] done in {info['train_seconds']}s, epoch "
                  f"{info['chosen_epoch']}, val {info['chosen_val_db']:.3f} dB")

    if set(RUNS) <= set(summary["runs"]):
        print("\n=== single test evaluation ===")
        test = evaluate_stage3(cfg, cfg.data.n_test, summary["runs"])
        per = test["per_trial_nmse"]
        test["contrasts"] = {
            "delta_H": paired(per, "U1_urformer_80k", "H1_hs_urformer_80k"),
            "delta_H_posthoc": paired(per, "U1_urformer_80k", "U1_plus_post"),
            "internal_vs_posthoc": paired(per, "U1_plus_post",
                                          "H1_hs_urformer_80k"),
            "delta_H_classical": paired(per, "U0_em_gs", "H0_hs_em_gs"),
            "X1_vs_U1": paired(per, "X1_emgs_plus_former", "U1_urformer_80k"),
            "U1_vs_U0": paired(per, "U0_em_gs", "U1_urformer_80k"),
            "H1_vs_U0": paired(per, "U0_em_gs", "H1_hs_urformer_80k"),
            "X1_vs_U0": paired(per, "U0_em_gs", "X1_emgs_plus_former"),
            **{f"posthoc_n_iter_{k}": paired(per, "U1_urformer_80k",
                                             f"U1_plus_post_it{k}")
               for k in (1, 2, 4, 8)},
        }
        test["verdict"] = verdict(test["contrasts"]["delta_H"])
        summary["test"] = test
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        print("\n  per-arm test NMSE (PRIMARY = median):")
        for k, v in test["methods"].items():
            print(f"    {k:24s} median {v['nmse_median_db']:8.3f} dB   "
                  f"(ros {v['nmse_ratio_of_sums_db']:8.3f})")
        print("\n  paired contrasts (positive = second arm better):")
        for k, v in test["contrasts"].items():
            print(f"    {k:22s} {v['median_diff_db']:+7.3f} dB  "
                  f"CI [{v['boot_ci95_median'][0]:+.3f}, "
                  f"{v['boot_ci95_median'][1]:+.3f}]  "
                  f"excl0={v['ci_excludes_zero']}")
        print(f"\n  VERDICT: {test['verdict']['decision']}")
        print(f"  {test['verdict']['reason']}")
    else:
        print(f"\nskipping evaluation: have {sorted(summary['runs'])}, "
              f"need {sorted(RUNS)}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
