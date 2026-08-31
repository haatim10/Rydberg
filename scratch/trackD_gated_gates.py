"""PROMPT 7 B2 -- the GK1/GK2 degeneration gates for the gated Hankel arm.

    GK1   beta = 0  =>  gated model is EXACTLY U1 (plain URformer)
    GK2   beta = 1  =>  gated model is EXACTLY H1 (HS-URformer)

Together these prove the gated architecture is a strict interpolation between
the two arms already measured in stage 3, so any difference in the results can
come only from the gate. They are the counterpart of HK5 (identity at full
rank) and gate D (``alpha=0`` gives classical GS).

Weight matching, and why seeding is not enough
----------------------------------------------
Constructing the gate modules CONSUMES RNG draws, so a gated model and an
ungated one built under the same ``torch.manual_seed`` do NOT share Transformer
weights -- the stream has shifted by the time the later layers are built. Every
gate below therefore COPIES the shared parameters by name and asserts the copy
was total, rather than trusting a seed. Getting this wrong would make GK1/GK2
fail for a reason that has nothing to do with the gate.

Run:  PYTHONPATH=. python3 scratch/trackD_gated_gates.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from trackD_urformer.config import ModelConfig, NumericConfig, TrackDConfig
from trackD_urformer.urformer import URformer, count_parameters

OUT = Path("reports/trackD_gated_gates.json")
C128, F64 = torch.complex128, torch.float64


def copy_shared(src: URformer, dst: URformer) -> int:
    """Copy every parameter/buffer whose name exists in both. Returns the count."""
    ssd, dsd = src.state_dict(), dst.state_dict()
    shared = {k: v for k, v in ssd.items() if k in dsd and dsd[k].shape == v.shape}
    dsd.update(shared)
    dst.load_state_dict(dsd)
    missing = [k for k in dsd if k not in ssd]
    if missing:
        raise AssertionError(f"dst has parameters src lacks: {missing}")
    return len(shared)


def world(N=16, K=3, P=20, b=6, seed=4242):
    rng = np.random.default_rng(seed)
    cx = lambda *s: torch.as_tensor(
        rng.standard_normal(s) + 1j * rng.standard_normal(s), dtype=C128)
    return dict(
        G0=cx(b, N, K), S=cx(b, K, P), B=cx(b, N, P),
        Z=torch.as_tensor(np.abs(rng.standard_normal((b, N, P))), dtype=F64),
        # A spread of noise levels, so the SNR-conditioned gate genuinely varies
        # across the batch rather than being tested at one operating point.
        sigma2=torch.tensor([0.02, 0.1, 0.5, 1.0, 4.0, 20.0], dtype=F64))


def rel(a, b) -> float:
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def build(N, K, *, hankel: bool, gate: str, mcfg_base: ModelConfig):
    torch.manual_seed(20260827)
    m = ModelConfig(**{**mcfg_base.__dict__, "use_hankel": hankel,
                       "hankel_gate": gate})
    return URformer(N, K, m, NumericConfig(dtype="float64")).double().eval()


def main() -> int:
    N, K = 16, 3
    base = ModelConfig(T_UR=4, use_hankel=False, hankel_rank=5,
                       hankel_gate="none")
    w = world(N, K)
    res = {"tolerance": 1e-12, "gates": []}

    for gate_mode in ("scalar", "snr"):
        gated = build(N, K, hankel=True, gate=gate_mode, mcfg_base=base)
        plain = build(N, K, hankel=False, gate="none", mcfg_base=base)
        hsur = build(N, K, hankel=True, gate="none", mcfg_base=base)
        n1, n2 = copy_shared(gated, plain), copy_shared(gated, hsur)

        # Make the shared weights non-trivial: a zero-init Transformer would
        # make GK1/GK2 pass even if the gate were wired wrong, because every
        # branch would collapse to the same fixed update.
        with torch.no_grad():
            for mdl in (gated, plain, hsur):
                for lay in mdl.layers:
                    torch.manual_seed(99)
                    torch.nn.init.normal_(lay.former.out_proj.weight, std=0.05)
                    torch.nn.init.normal_(lay.former.out_proj.bias, std=0.05)

        with torch.no_grad():
            gated._set_test_mode(beta=0.0)
            g0 = gated(w["G0"], w["Z"], w["S"], w["B"], w["sigma2"])
            gated._set_test_mode(beta=1.0)
            g1 = gated(w["G0"], w["Z"], w["S"], w["B"], w["sigma2"])
            gated._clear_test_mode()
            u = plain(w["G0"], w["Z"], w["S"], w["B"], w["sigma2"])
            h = hsur(w["G0"], w["Z"], w["S"], w["B"], w["sigma2"])
            # Sanity: U1 and H1 must actually DIFFER, or the gates are vacuous.
            sep = rel(u, h)

        r1, r2 = rel(g0, u), rel(g1, h)
        res["gates"] += [
            {"gate": "GK1", "mode": gate_mode, "condition": "beta=0 == URformer",
             "rel": r1, "pass": bool(r1 < 1e-12), "shared_params_copied": n1},
            {"gate": "GK2", "mode": gate_mode, "condition": "beta=1 == HS-URformer",
             "rel": r2, "pass": bool(r2 < 1e-12), "shared_params_copied": n2},
            {"gate": "GK0", "mode": gate_mode,
             "condition": "URformer != HS-URformer (gates are non-vacuous)",
             "rel": sep, "pass": bool(sep > 1e-6)},
        ]
        print(f"[{gate_mode}] GK1 beta=0 vs URformer     rel {r1:.3e}  "
              f"{'PASS' if r1 < 1e-12 else 'FAIL'}")
        print(f"[{gate_mode}] GK2 beta=1 vs HS-URformer  rel {r2:.3e}  "
              f"{'PASS' if r2 < 1e-12 else 'FAIL'}")
        print(f"[{gate_mode}] GK0 the two arms differ    rel {sep:.3e}  "
              f"{'PASS' if sep > 1e-6 else 'FAIL (gates vacuous!)'}")

    # Initial betas at the real config, reported per layer as B2 requires.
    cfg = TrackDConfig()
    for gate_mode in ("scalar", "snr"):
        torch.manual_seed(cfg.train.seed)
        m = URformer(cfg.system.N, cfg.system.K,
                     replace(cfg.model, use_hankel=True, hankel_rank=7,
                             hankel_gate=gate_mode), cfg.numeric)
        s2 = torch.tensor([0.02, 0.2, 2.0, 20.0], dtype=torch.float32)
        res[f"initial_betas_{gate_mode}"] = {
            "per_layer_at_sigma2_0.2": m.initial_betas(s2[1:2]),
            "across_sigma2_layer0": [
                float(m.layers[0].hankel_gate(s2[i:i + 1]).reshape(-1)[0])
                for i in range(4)],
            "sigma2_probes": s2.tolist(),
            "total_params": count_parameters(m)["totals"]["all_parameters"],
        }
        b = res[f"initial_betas_{gate_mode}"]
        print(f"\n[{gate_mode}] initial beta per layer: "
              f"{[round(x, 4) for x in b['per_layer_at_sigma2_0.2']]}")
        print(f"[{gate_mode}] beta across sigma^2 {s2.tolist()}: "
              f"{[round(x, 4) for x in b['across_sigma2_layer0']]}")
        print(f"[{gate_mode}] total params: {b['total_params']}")

    res["all_pass"] = all(g["pass"] for g in res["gates"])
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nall gates pass: {res['all_pass']}\nwrote {OUT}")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
