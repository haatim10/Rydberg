"""Track D timing probe - measure CPU throughput, do not guess it.

PROMPT 2 sec. 0: run ~200 samples x 3 epochs at the reference config and
extrapolate; benchmark torch.set_num_threads(1) against (4). With tensors this
small, op-dispatch overhead usually dominates and fewer threads is often
faster -- the measurement decides.

Writes reports/trackD_timing.json. Throwaway; delete after the phase.

Run:  PYTHONPATH=. python3 scratch/trackD_timing_probe.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset, collate
from trackD_urformer.train import make_initial_batch, nmse_loss
from trackD_urformer.urformer import URformer, count_parameters

OUT = Path("reports/trackD_timing.json")
N_SAMPLES = 200
N_EPOCHS = 3


def bench(n_threads: int, cfg: TrackDConfig, *, N: int | None = None,
          P: int | None = None) -> dict:
    torch.set_num_threads(n_threads)
    torch.manual_seed(0)
    np.random.seed(0)

    N = int(cfg.system.N if N is None else N)
    P = int(cfg.system.P if P is None else P)
    ds = TrackDDataset("train", sysc=cfg.system, datac=cfg.data,
                       numeric=cfg.numeric, N=N, P=P)
    ds.n_items = N_SAMPLES
    # Pre-generate so dataset synthesis is not counted as training time.
    t0 = time.time()
    for i in range(N_SAMPLES):
        ds.sample(i)
    gen_s = time.time() - t0

    ld = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=0)
    model = URformer(N, cfg.system.K, cfg.model, cfg.numeric)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # one warm-up epoch, not timed (lazy kernel selection / allocator warm-up)
    for batch in ld:
        opt.zero_grad()
        G0 = make_initial_batch(batch, "random", cfg)
        loss = nmse_loss(model(G0, batch["Z"], batch["S"], batch["B"],
                               batch["sigma2"]), batch["G_true"])
        loss.backward()
        opt.step()
        break

    epoch_times = []
    for _ in range(N_EPOCHS):
        t0 = time.time()
        for batch in ld:
            opt.zero_grad()
            G0 = make_initial_batch(batch, "random", cfg)
            loss = nmse_loss(model(G0, batch["Z"], batch["S"], batch["B"],
                                   batch["sigma2"]), batch["G_true"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        epoch_times.append(time.time() - t0)

    med = float(np.median(epoch_times))
    return {
        "n_threads": n_threads,
        "N": N,
        "P": P,
        "n_samples": N_SAMPLES,
        "epoch_seconds": [round(t, 3) for t in epoch_times],
        "median_epoch_seconds": round(med, 3),
        "samples_per_second": round(N_SAMPLES / med, 2),
        "dataset_gen_seconds_for_200": round(gen_s, 2),
        "dataset_gen_per_sample_ms": round(1000 * gen_s / N_SAMPLES, 2),
    }


def main() -> None:
    cfg = TrackDConfig()
    info = {
        "config": cfg.to_dict(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "reference_config": {
            "N": cfg.system.N, "K": cfg.system.K, "P": cfg.system.P,
            "T_UR": cfg.model.T_UR, "d_model": cfg.model.d_model,
            "L_enc": cfg.model.L_enc, "batch_size": cfg.train.batch_size,
            "dtype": cfg.numeric.dtype,
        },
    }
    torch.manual_seed(0)
    m = URformer(cfg.system.N, cfg.system.K, cfg.model, cfg.numeric)
    info["params"] = count_parameters(m)

    runs = {}
    for nt in (1, 2, 4):
        print(f"benchmarking num_threads={nt} ...")
        runs[str(nt)] = bench(nt, cfg)
        print("   ", runs[str(nt)]["median_epoch_seconds"], "s/epoch @",
              runs[str(nt)]["samples_per_second"], "samples/s")
    info["runs"] = runs

    best = min(runs.values(), key=lambda r: r["median_epoch_seconds"])
    info["chosen_num_threads"] = best["n_threads"]

    sps = best["samples_per_second"]
    gen_ms = best["dataset_gen_per_sample_ms"]
    nt = best["n_threads"]
    full_train_s = 20000 / sps * 50
    gen_s = 20000 * gen_ms / 1000
    info["extrapolation"] = {
        "basis": f"num_threads={nt}, {sps} samples/s at N={cfg.system.N}",
        "one_training_20000x50_seconds": round(full_train_s, 1),
        "one_training_20000x50_hours": round(full_train_s / 3600, 2),
        "dataset_generation_20000_seconds": round(gen_s, 1),
        "dataset_generation_20000_hours": round(gen_s / 3600, 3),
        "total_one_training_hours": round((full_train_s + gen_s) / 3600, 2),
    }

    # Per-N throughput, so the training-matrix estimate is MEASURED not guessed.
    per_N = {}
    for N in (8, 16, 32):
        print(f"benchmarking N={N} at num_threads={nt} ...")
        r = bench(nt, cfg, N=N)
        hrs = (20000 / r["samples_per_second"] * 50) / 3600
        r["one_training_hours"] = round(hrs, 2)
        per_N[str(N)] = r
        print(f"    N={N}: {r['samples_per_second']} samples/s "
              f"-> {hrs:.2f} h per training")
    info["per_N"] = per_N

    h = {n: per_N[n]["one_training_hours"] for n in per_N}
    # Training matrix (PROMPT 2 sec. 6): 3 initializers, each a separate model.
    # D1 (N=32, P=20) and D3's N=32 column are the same config -> reuse.
    matrix = {
        "D1_nmse_vs_snr": {"trainings": 3, "detail": "N=32, P=20, 3 initializers",
                           "hours": round(3 * h["32"], 2)},
        "D2_nmse_vs_pilots": {"trainings": 3,
                              "detail": "N=32, P~U{sweep}, 3 initializers",
                              "hours": round(3 * h["32"], 2)},
        "D3_nmse_vs_array_size": {
            "trainings": 6,
            "detail": "N in {8,16} x 3 initializers; the N=32 column reuses D1",
            "hours": round(3 * h["8"] + 3 * h["16"], 2)},
    }
    matrix["total"] = {
        "trainings": sum(v["trainings"] for v in matrix.values()),
        "hours": round(sum(v["hours"] for v in matrix.values()), 2),
    }
    info["training_matrix"] = matrix
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info["extrapolation"], indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
