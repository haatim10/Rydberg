"""Resumable Cui Fig. 7(a) / 7(b) / 8 BER runs.

BER curves span four decades, so trials are allocated **per sweep point**
(more where the BER is small) instead of uniformly. Each sweep point is its
own ``run_experiment`` call into a per-chunk store; ``config_fingerprint``
excludes the sweep grids, so all chunks share one fingerprint and merge
cleanly.

Usage:
    python3 scripts/run_fig78.py <fig7a|fig7b|fig8> <chunk> <n_chunks>
    python3 scripts/run_fig78.py <fig7a|fig7b|fig8> merge
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rydberg_sim.monte_carlo import (
    RESULTS_NAME,
    config_fingerprint,
    load_result_table,
    run_experiment,
)
from rydberg_sim.track_a_fig78 import (
    FIG8_QAM_CAPTION_CLAIM,
    aggregate_ber,
    track_a_fig7a_spec,
    track_a_fig7b_spec,
    track_a_fig8_spec,
)

REPO = Path(__file__).resolve().parent.parent


def _alloc(points, bands):
    """bands: list of (predicate, n_trials); first match wins."""
    out = {}
    for x in points:
        for pred, n in bands:
            if pred(x):
                out[x] = n
                break
    return out


FIG7_SNR = [float(v) for v in range(-5, 13)]
FIG8_RSR = [float(v) for v in range(0, 26)]
# Small diagnostic sweep documenting the Fig. 8 caption/body contradiction:
# the caption claims 16-QAM, the body text says 4-QAM. A handful of RSR
# points is enough to show which one the published BER levels belong to.
FIG8_16QAM_RSR = [0.0, 6.0, 12.0, 18.0, 25.0]


def _fig8_16qam_spec(*, n_trials: int, **kw):
    """Fig. 8 exactly as the caption claims: 16-QAM instead of 4-QAM."""
    kw.setdefault("experiment", "cui_fig8_16qam_caption")
    return track_a_fig8_spec(
        n_trials=n_trials, qam_M=FIG8_QAM_CAPTION_CLAIM, **kw
    )

JOBS = {
    # name: (spec builder, sweep key, fixed other-axis value, allocation)
    "fig7a": (
        track_a_fig7a_spec, "snr_db", 12.0,
        _alloc(FIG7_SNR, [(lambda x: x <= 0, 3_000),
                          (lambda x: x <= 4, 10_000),
                          (lambda x: x <= 7, 25_000),
                          (lambda x: True, 40_000)]),
    ),
    "fig7b": (
        track_a_fig7b_spec, "snr_db", 12.0,
        _alloc(FIG7_SNR, [(lambda x: x <= 0, 2_000),
                          (lambda x: x <= 5, 5_000),
                          (lambda x: x <= 9, 10_000),
                          (lambda x: True, 15_000)]),
    ),
    "fig8": (
        track_a_fig8_spec, "rsr_db", 3.0,
        _alloc(FIG8_RSR, [(lambda x: x <= 6, 3_000),
                          (lambda x: x <= 14, 8_000),
                          (lambda x: True, 14_000)]),
    ),
    # 16-QAM at 3 dB SNR sits near 1e-1, so 3k trials (= 36k bits) is ample.
    "fig8_16qam": (
        _fig8_16qam_spec, "rsr_db", 3.0,
        _alloc(FIG8_16QAM_RSR, [(lambda x: True, 3_000)]),
    ),
}


def store(name: str) -> Path:
    return REPO / "results" / "track_a" / name


def run_chunk(name: str, chunk: int, n_chunks: int) -> None:
    builder, sweep_key, _, alloc = JOBS[name]
    points = sorted(alloc)
    mine = [p for i, p in enumerate(points) if i % n_chunks == chunk]
    out = store(name) / f"chunk{chunk}"
    out.mkdir(parents=True, exist_ok=True)
    for p in mine:
        n = alloc[p]
        grid = {"snr_db_grid" if sweep_key == "snr_db" else "rsr_db_grid": (p,)}
        spec = builder(n_trials=n, **grid)
        t0 = time.time()
        run_experiment(spec, out, resume=True)
        print(f"[{name} c{chunk}] {sweep_key}={p:+.1f} n={n} "
              f"({time.time()-t0:.0f}s)", flush=True)


def merge(name: str) -> None:
    builder, sweep_key, _, alloc = JOBS[name]
    out = store(name)
    rows: list[dict] = []
    for d in sorted(out.glob("chunk*")):
        f = d / RESULTS_NAME
        if f.exists():
            rows.extend(load_result_table(f))
    agg = aggregate_ber(rows, sweep_key=sweep_key)
    (out / "aggregate.json").write_text(json.dumps(
        {"figure": name, "sweep_key": sweep_key,
         "trials_per_point": {str(k): v for k, v in alloc.items()},
         "fingerprint": config_fingerprint(builder(n_trials=alloc[max(alloc)])),
         "aggregate": agg}, indent=2))
    hdr = ["algorithm", sweep_key, "ber", "ber_ci95_low", "ber_ci95_high",
           "bit_errors", "bit_count", "n_trials"]
    lines = [",".join(hdr)]
    for r in agg:
        lines.append(",".join(str(r[h]) for h in hdr))
    (out / "aggregate.csv").write_text("\n".join(lines) + "\n")
    print(f"[{name}] merged {len(rows)} rows -> {len(agg)} aggregate points")
    for r in agg:
        print(f"  {r['algorithm']:14s} {sweep_key}={r[sweep_key]:+6.1f} "
              f"BER={r['ber']:.4e}  ({r['bit_errors']:>7d}/{r['bit_count']:>9d})")


if __name__ == "__main__":
    name = sys.argv[1]
    if sys.argv[2] == "merge":
        merge(name)
    else:
        run_chunk(name, int(sys.argv[2]), int(sys.argv[3]))
