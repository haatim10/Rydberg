"""PROMPT 15 A1 -- which epoch's weights produced the stored rows?

Every per-epoch checkpoint survives on disk (`ep000.pt` .. `ep012.pt`), and
`best.pt` is a copy of the 1-SE-chosen epoch. If the stored per-trial NMSE
matches some epoch OTHER than the one `best.pt` holds, the stored row was
evaluated against different weights than `best.pt` carries today, and the
discrepancy is a stale results file rather than nondeterministic evaluation.

Run:  PYTHONPATH=. python3 scratch/p15_a1_which_epoch.py [n_trials]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from trackD_urformer.baselines import nmse_parts
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset
from trackD_urformer.stage4 import HIGH_SNR, PART_C, build_arm
from dataclasses import replace

N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
S4 = Path("results/track_d/stage4")


def main():
    st = json.load(open("reports/trackD_stage4_results.json"))["part_c"]
    cfg = TrackDConfig()
    cfg = replace(cfg, train=replace(cfg.train, init="spectral"))
    ecfg = replace(cfg, data=replace(cfg.data, snr_range_db=HIGH_SNR))
    ds = TrackDDataset("test", sysc=ecfg.system, datac=ecfg.data,
                       numeric=ecfg.numeric, init=ecfg.train.init)
    cd, rd = ecfg.numeric.complex_dtype, ecfg.numeric.real_dtype

    worlds = []
    T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)
    for i in range(N):
        s = ds.sample(i)
        worlds.append((T(ds.g0(i), cd), T(s.Z, rd), T(s.S, cd), T(s.B, cd),
                       torch.tensor([s.sigma2], dtype=rd), s.G_true))

    for nm in PART_C:
        target = np.asarray(st["per_trial_nmse"][nm][:N], dtype=float)
        bp = torch.load(S4 / nm / "best.pt", map_location="cpu",
                        weights_only=False)
        print(f"\n{nm}: best.pt records epoch {bp['epoch']}; "
              f"stored median {10*np.log10(np.median(target)):+.4f} dB")
        model, _ = build_arm(cfg, nm)
        for ep in range(13):
            f = S4 / nm / f"ep{ep:03d}.pt"
            if not f.exists():
                continue
            model.load_state_dict(torch.load(f, map_location="cpu",
                                             weights_only=False)["model"])
            model.eval()
            got = []
            with torch.no_grad():
                for G0, Z, S, B, s2, G in worlds:
                    gh = model(G0, Z, S, B, s2)[0].numpy()
                    got.append(np.divide(*nmse_parts(gh, G)))
            got = np.asarray(got, dtype=float)
            d = np.abs(got - target)
            mark = "  <== MATCH" if np.array_equal(got, target) else ""
            print(f"   ep{ep:03d}  max|rel|={np.max(d/np.abs(target)):.3e}  "
                  f"median={10*np.log10(np.median(got)):+.4f} dB{mark}")


if __name__ == "__main__":
    main()
