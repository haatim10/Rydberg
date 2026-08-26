"""Re-stamp existing result stores for the widened fingerprint -- but only
after PROVING each store still reproduces under the current code.

Context. Before the audit, ``runner.fingerprint()`` hashed only the four
wrapper modules in this package. Every number actually comes from
``rydberg_sim/`` (channel generator, pilots, reference, forward model, EM-GS
solver, Hankel lifting, Cadzow, rank selector, RNG), none of which was
hashed, so a change there left every stored result looking valid. The
fingerprint now covers those modules too.

Widening the hash changes it, which would ordinarily force a full rerun of
10,800 paired trials. That is only acceptable if the stores are actually
stale. This script decides that empirically instead of assuming it: for each
store it re-runs a random sample of the ALREADY-STORED trial indices under
the current code and requires the recomputed NMSE numerator, the Hankel
numerator and the shared denominator to agree with what is on disk. A store
that reproduces is re-stamped; one that does not is left untouched and
reported, because then the stored numbers really were produced by different
code and must be recomputed rather than relabelled.

    python migrate_stores.py            # verify + re-stamp (10 trials/store)
    python migrate_stores.py --check    # verify only, change nothing
    SAMPLE=25 python migrate_stores.py  # deeper sample
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

import config as cfg
import em_gs
import hankel_em_gs as hem
import runner
from system_model import channel_nmse_parts, make_world

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TOL = 1e-9


def meta_from_path(path: Path) -> dict:
    """Recover the operating point from the store's location and name."""
    stem = path.stem
    if path.parent.name == "pathcount":
        return runner.meta_of(N=cfg.EXP_C_N, P=cfg.P_DEFAULT,
                              snr_db=cfg.EXP_C_SNR, rsr_db=cfg.RSR_DB,
                              L=int(stem[1:]))
    N = int(stem[1:3])
    P = int(stem.split("_P")[1].split("_")[0])
    snr = float(stem.split("snr")[1])
    return runner.meta_of(N=N, P=P, snr_db=snr, rsr_db=cfg.RSR_DB, L=None)


def verify_store(path: Path, meta: dict, sample: int, rng) -> tuple[bool, str]:
    """Re-run a sample of the stored trials and compare against disk."""
    d = np.load(path)
    trials = d["trial"]
    if trials.size == 0:
        return False, "store holds no completed trials"
    idx = rng.choice(trials.size, size=min(sample, trials.size), replace=False)
    L = None if meta["L"] < 0 else meta["L"]
    worst = 0.0
    for i in idx:
        t = int(trials[i])
        w = make_world(t, N=meta["N"], P=meta["P"], snr_db=meta["snr_db"], L=L)
        G_base = em_gs.em_gs(w.S, w.Z, w.B, w.sigma2, max_iter=cfg.GS_MAX_ITER)
        res = hem.hankel_em_gs(w.S, w.Z, w.B, w.sigma2, max_iter=cfg.GS_MAX_ITER)
        e_base, den = channel_nmse_parts(G_base, w.G)
        e_hank, _ = channel_nmse_parts(res.G_hat, w.G)
        worst = max(worst,
                    abs(e_base - float(d["num_em_gs"][i])),
                    abs(e_hank - float(d["num_hankel_em_gs"][i])),
                    abs(den - float(d["denom"][i])))
        if int(res.L_hat) != int(d["L_hat"][i]):
            return False, f"trial {t}: L_hat {res.L_hat} != stored {d['L_hat'][i]}"
    return worst < TOL, f"{len(idx)} trials, max |recomputed - stored| = {worst:.2e}"


def main() -> int:
    check_only = "--check" in sys.argv
    sample = int(os.environ.get("SAMPLE", "10"))
    rng = np.random.default_rng(20250820)

    stores = sorted(glob.glob(str(RES / "grid" / "*.npz")) +
                    glob.glob(str(RES / "pathcount" / "*.npz")))
    stores = [Path(s) for s in stores if not s.endswith(".tmp.npz")]
    if not stores:
        print("No stores found.")
        return 1

    print(f"target fingerprint: {runner.FP}")
    print(f"{len(stores)} stores, sampling {sample} trials each\n")
    ok_n = skip_n = fail_n = 0
    failures = []
    for path in stores:
        meta = meta_from_path(path)
        d = np.load(path)
        if str(d["fingerprint"]) == runner.FP and "meta" in d.files:
            print(f"  [current] {path.parent.name}/{path.name}")
            skip_n += 1
            continue
        good, detail = verify_store(path, meta, sample, rng)
        if not good:
            print(f"  [STALE  ] {path.parent.name}/{path.name}: {detail}")
            failures.append(str(path))
            fail_n += 1
            continue
        print(f"  [ok     ] {path.parent.name}/{path.name}: {detail}")
        ok_n += 1
        if not check_only:
            payload = {k: d[k] for k in runner.FIELDS}
            runner._save(path, payload, meta)

    print(f"\n{ok_n} verified{'' if check_only else ' and re-stamped'}, "
          f"{skip_n} already current, {fail_n} stale")
    if failures:
        print("\nSTALE stores -- their numbers were NOT produced by the current "
              "code. Delete and recompute them; do not relabel:")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
