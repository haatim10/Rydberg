"""Write per-figure metadata and copy the aggregate CSVs into final_figures/."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TA = REPO / "results/track_a"
TB = Path("/home/user/rydberg-trackb/results/track_b")
FINAL = REPO / "results/final_figures"

SPECS = {
    "fig5_clean": dict(
        store=TA / "fig5_final", src="results/track_a/fig5_final",
        title="Cui Fig. 5 — detection NMSE vs SNR",
        config="N x K = 36 x 3, 16-QAM, RSR = 12 dB, t0 = 50, SNR -5..12 dB (18 pts)",
        trials="2 000 trials/SNR",
        algs="biased GS, EM-GS, ZF w/ known phase, Cui CRLB",
        omitted="CM-ZF — not specified in the paper (see baselines.py); "
                "exhaustive search — Cui does not plot it in an NMSE figure",
        metric="Detection NMSE, ratio of sums, 10 log10",
        note="Retains the documented ~2 dB channel-conditioning offset vs Cui.",
    ),
    "fig6_clean": dict(
        store=TA / "fig6", src="results/track_a/fig6",
        title="Cui Fig. 6 — detection NMSE vs RSR",
        config="N x K = 36 x 3, 16-QAM, SNR = 3 dB, t0 = 50, RSR 0..25 dB (26 pts)",
        trials="500 trials/RSR",
        algs="biased GS, EM-GS, ZF w/ known phase, Cui CRLB",
        omitted="CM-ZF — unspecified in the paper",
        metric="Detection NMSE, ratio of sums, 10 log10",
        note="ZF is flat in RSR (fitted slope -0.0026 dB/dB, not significant), "
             "as required since the genie ZF error does not depend on b.",
    ),
    "fig7a_clean": dict(
        store=TA / "fig7a", src="results/track_a/fig7a",
        title="Cui Fig. 7(a) — BER vs SNR, small-scale configuration",
        config="N x K = 36 x 3, 4-QAM, RSR = 12 dB, t0 = 50, SNR -5..12 dB (18 pts)",
        trials="3 000 (-5..0), 10 000 (1..4), 25 000 (5..7), 40 000 (8..12) "
               "= 333 000 trials; 1 998 000 bits per algorithm",
        algs="biased GS, EM-GS, exhaustive search (LS), exhaustive search (ML), "
             "ZF w/ known phase",
        omitted="CM-ZF — described only as extending ref. [39]; not a specification",
        metric="BER = global bit errors / global bit count",
        note="Zero-error points are omitted from the log axis rather than "
             "plotted at zero; their one-sided 95% Wilson upper bounds "
             "(1.1e-5 at 240 000 bits) are in aggregate.csv.",
    ),
    "fig7b_clean": dict(
        store=TA / "fig7b", src="results/track_a/fig7b",
        title="Cui Fig. 7(b) — BER vs SNR, large-scale configuration",
        config="N x K = 100 x 6, 16-QAM, RSR = 12 dB, t0 = 50, SNR -5..12 dB (18 pts)",
        trials="2 000 (-5..0), 5 000 (1..5), 10 000 (6..9), 15 000 (10..12) "
               "= 122 000 trials; 2 928 000 bits per algorithm",
        algs="biased GS, EM-GS, ZF w/ known phase",
        omitted="exhaustive search — excluded exactly as Cui does (16^6 = 16.7M "
                "candidates per trial); CM-ZF — unspecified",
        metric="BER = global bit errors / global bit count",
        note="Measured EM-GS/ZF SNR gap 3.79-3.97 dB; Cui states 'between 3 ~ 4 dB'.",
    ),
    "fig8_clean": dict(
        store=TA / "fig8", src="results/track_a/fig8",
        title="Cui Fig. 8 — BER vs RSR (4-QAM, body-text interpretation)",
        config="N x K = 36 x 3, 4-QAM, SNR = 3 dB, t0 = 50, RSR 0..25 dB (26 pts)",
        trials="3 000 (0..6), 8 000 (7..14), 14 000 (15..25) = 239 000 trials; "
               "1 434 000 bits per algorithm",
        algs="biased GS, EM-GS, exhaustive search (LS), exhaustive search (ML)",
        omitted="ZF w/ known phase — evaluated and present in aggregate.csv, but "
                "Cui plots no ZF curve in Fig. 8; CM-ZF — unspecified",
        metric="BER = global bit errors / global bit count",
        note="Cui's caption says 16-QAM but the body text says 4-QAM; the plotted "
             "BER levels support 4-QAM (see fig8_16qam_diagnostic).",
    ),
    "fig8_16qam_diagnostic": dict(
        store=TA / "fig8_16qam", src="results/track_a/fig8_16qam",
        title="DIAGNOSTIC — Fig. 8 run at 16-QAM as the caption claims",
        config="N x K = 36 x 3, 16-QAM, SNR = 3 dB, RSR {0,6,12,18,25} dB (5 pts)",
        trials="3 000 trials/RSR = 15 000 trials; 180 000 bits per algorithm",
        algs="biased GS, EM-GS, exhaustive search (LS), exhaustive search (ML)",
        omitted="ZF w/ known phase — kept out to match Fig. 8's curve set",
        metric="BER = global bit errors / global bit count",
        note="DIAGNOSTIC ONLY, not a reproduction of Fig. 8. Median BER ratio to "
             "Cui: 4-QAM 0.82 vs 16-QAM 24.12, so the body text is correct and "
             "the caption is in error.",
    ),
    "b1_clean": dict(
        store=TB, src="rydberg-trackb:results/track_b",
        title="Track B, B1 — channel NMSE vs SNR (EXACT nonlinear model)",
        config="Geometric ULA, N = 8, K = 3, L_k ~ U{3..7} per realization, "
               "RSR = 12 dB, t0 = 50, P in {10, 30}",
        trials="400 trials/point",
        algs="Cui biased GS, Cui EM-GS — both on the EXACT model Z = |GS+B+W|",
        omitted="Xu linearized LS and the HS-GS structural prototype — the frozen "
                "baseline plots only the two unstructured exact-model estimators",
        metric="Channel NMSE_G = sum||Ghat-G||_F^2 / sum||G||_F^2, 10 log10",
        note="NO linearization anywhere on this path. Bootstrap 95% CIs are in "
             "baseline_preliminary.json.",
    ),
    "b2_clean": dict(
        store=TB, src="rydberg-trackb:results/track_b",
        title="Track B, B2 — channel NMSE vs pilot length P (EXACT nonlinear model)",
        config="Geometric ULA, N = 8, K = 3, L_k ~ U{3..7}, RSR = 12 dB, "
               "SNR = 5 dB, P in {6,10,14,20,30,40}",
        trials="400 trials/point",
        algs="Cui biased GS, Cui EM-GS — both on the EXACT model Z = |GS+B+W|",
        omitted="as B1",
        metric="Channel NMSE_G, ratio of sums, 10 log10",
        note="P = 2K marked. NO linearization anywhere on this path.",
    ),
}


def fingerprint_of(store: Path) -> str:
    for cand in (store / "aggregate.json", store / "config.json"):
        if cand.exists():
            d = json.loads(cand.read_text())
            if isinstance(d, dict):
                for k in ("fingerprint", "config_fingerprint"):
                    if k in d:
                        return str(d[k])
    cfg = store / "config.json"
    if cfg.exists():
        d = json.loads(cfg.read_text())
        return str(d.get("fingerprint", "n/a"))
    return "n/a"


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    lines = ["# Final figures\n",
             "Publication-style regeneration of every completed figure.\n",
             "All plots are generated **only** from the saved aggregate stores; "
             "no Monte Carlo was rerun.\n",
             "\n**Uncertainty is deliberately not drawn.** Confidence intervals "
             "(bootstrap for NMSE, Wilson for BER) remain in each aggregate "
             "CSV/JSON and in the notes below.\n",
             "\nStyle: linewidth 1.3, markersize 4.5, open markers, thin spines, "
             "subtle grid, no titles (captions carry them), 300 dpi PNG + vector PDF.\n"]
    for stem, s in SPECS.items():
        store = s["store"]
        fp = fingerprint_of(store)
        # copy the aggregate CSV next to the figure
        for cand in ("aggregate.csv", "baseline_preliminary.json"):
            src = store / cand
            if src.exists():
                shutil.copy2(src, FINAL / f"{stem}__{cand}")
        lines += [
            f"\n---\n\n## `{stem}`\n",
            f"**{s['title']}**\n",
            f"- source store: `{s['src']}`",
            f"- configuration: {s['config']}",
            f"- trials: {s['trials']}",
            f"- metric: {s['metric']}",
            f"- algorithms plotted: {s['algs']}",
            f"- omitted: {s['omitted']}",
            f"- config fingerprint: `{fp}`",
            "- error bars: **intentionally omitted from the visual** "
            "(values retained in the aggregate files)",
            f"- note: {s['note']}",
        ]
    (FINAL / "README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {FINAL/'README.md'}")
    for f in sorted(FINAL.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
