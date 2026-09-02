"""Matched-pilot points at the sweep's own operating point (SNR = 5 dB).

`reports/trackD_stage5_eval.json` evaluates the matched-trained models over a
UNIFORM SNR draw, while the pilot sweep in `results/track_d/sweeps` sits at a
fixed 5 dB (the point Xiao et al. use for their Fig. 4). Those two axes cannot
be placed on one figure without re-measuring, so this script evaluates the
matched-trained models on the SAME worlds the sweep used: same split, same
`n = 400` trial indices, same `snr_db = 5.0`, same `P` grid entries.

That makes the three-way pilot figure honest -- classical efficiency, learned
pilot-count GENERALIZATION (one P=20 model evaluated everywhere) and learned
pilot EFFICIENCY (a model trained at each P) all measured on identical trials.

Run:  PYTHONPATH=. python3 scratch/trackD_matched_pilot_points.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from trackD_urformer.baselines import nmse_parts
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset
from trackD_urformer.stage1 import build_model

OUT = Path("results/track_d/sweeps/matched_pilot_points.json")
N_TRIALS = 400
PILOT_SNR = 5.0

# P -> (name, checkpoint, hankel, gate). P=20's matched model IS U1, which the
# sweep already carries; it is re-measured here so all three points come from
# one code path.
MATCHED = {
    10: ("C2_P10_matched", "results/track_d/stage5/C2_urformer_P10/best.pt", False, "none"),
    20: ("U1_P20_uniform", "results/track_d/stage2/B3_80k_13ep/best.pt", False, "none"),
    35: ("C3_P35_matched", "results/track_d/stage5/C3_urformer_P35/best.pt", False, "none"),
}
# G1 at P=10 is the P15 learned-half arm; carried so the figure can show it.
EXTRA = {10: ("C4_G1_P10", "results/track_d/stage5/C4_g1_P10/best.pt", True, "scalar")}


def load(cfg, path, hk, gate):
    rc = replace(cfg, model=replace(cfg.model, filter_init="random",
                                    use_transformer=True, use_hankel=hk,
                                    hankel_rank=7, hankel_mode="fixed",
                                    hankel_gate=gate))
    m, _ = build_model(rc, "arm1b_full_random")
    m.load_state_dict(torch.load(path, map_location="cpu",
                                 weights_only=False)["model"])
    return m.eval()


@torch.no_grad()
def main() -> int:
    torch.set_num_threads(1)
    cfg = replace(TrackDConfig(),
                  train=replace(TrackDConfig().train, init="spectral"))
    cd, rd = cfg.numeric.complex_dtype, cfg.numeric.real_dtype
    T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)
    out = {}
    for P, spec in MATCHED.items():
        specs = [spec] + ([EXTRA[P]] if P in EXTRA else [])
        models = {s[0]: load(cfg, s[1], s[2], s[3]) for s in specs}
        ds = TrackDDataset("test", sysc=cfg.system,
                           datac=replace(cfg.data, n_test=N_TRIALS),
                           numeric=cfg.numeric, P=P, snr_db=PILOT_SNR,
                           init=cfg.train.init)
        num = {k: [] for k in models}
        den = {k: [] for k in models}
        for i in range(N_TRIALS):
            s = ds.sample(i)
            G0, Z = T(ds.g0(i), cd), T(s.Z, rd)
            S, B = T(s.S, cd), T(s.B, cd)
            s2 = torch.tensor([s.sigma2], dtype=rd)
            for k, m in models.items():
                a, b = nmse_parts(m(G0, Z, S, B, s2)[0].numpy(), s.G_true)
                num[k].append(a)
                den[k].append(b)
        out[str(P)] = {"P": P, "snr_db": PILOT_SNR, "n": N_TRIALS,
                       "num": num, "den": den,
                       "median_db": {k: float(10 * np.log10(np.median(
                           np.asarray(num[k]) / np.asarray(den[k]))))
                           for k in models}}
        print(f"P={P:3d}  " + "  ".join(
            f"{k} {v:+.3f}" for k, v in out[str(P)]["median_db"].items()),
            flush=True)
    OUT.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
