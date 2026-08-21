# Final figures

Publication-style regeneration of every completed figure.

All plots are generated **only** from the saved aggregate stores; no Monte Carlo was rerun.


**Uncertainty is deliberately not drawn.** Confidence intervals (bootstrap for NMSE, Wilson for BER) remain in each aggregate CSV/JSON and in the notes below.


Style: linewidth 1.3, markersize 4.5, open markers, thin spines, subtle grid, no titles (captions carry them), 300 dpi PNG + vector PDF.


---

## `fig5_clean`

**Cui Fig. 5 — detection NMSE vs SNR**

- source store: `results/track_a/fig5_final`
- configuration: N x K = 36 x 3, 16-QAM, RSR = 12 dB, t0 = 50, SNR -5..12 dB (18 pts)
- trials: 2 000 trials/SNR
- metric: Detection NMSE, ratio of sums, 10 log10
- algorithms plotted: biased GS, EM-GS, ZF w/ known phase, Cui CRLB
- omitted: CM-ZF — not specified in the paper (see baselines.py); exhaustive search — Cui does not plot it in an NMSE figure
- config fingerprint: `925f2ab8975505e255f5c37af74d115be38bc302991267f95f440e5f61ce4472`
- error bars: **intentionally omitted from the visual** (values retained in the aggregate files)
- note: Retains the documented ~2 dB channel-conditioning offset vs Cui.

---

## `fig6_clean`

**Cui Fig. 6 — detection NMSE vs RSR**

- source store: `results/track_a/fig6`
- configuration: N x K = 36 x 3, 16-QAM, SNR = 3 dB, t0 = 50, RSR 0..25 dB (26 pts)
- trials: 500 trials/RSR
- metric: Detection NMSE, ratio of sums, 10 log10
- algorithms plotted: biased GS, EM-GS, ZF w/ known phase, Cui CRLB
- omitted: CM-ZF — unspecified in the paper
- config fingerprint: `925f2ab8975505e255f5c37af74d115be38bc302991267f95f440e5f61ce4472`
- error bars: **intentionally omitted from the visual** (values retained in the aggregate files)
- note: ZF is flat in RSR (fitted slope -0.0026 dB/dB, not significant), as required since the genie ZF error does not depend on b.

---

## `fig7a_clean`

**Cui Fig. 7(a) — BER vs SNR, small-scale configuration**

- source store: `results/track_a/fig7a`
- configuration: N x K = 36 x 3, 4-QAM, RSR = 12 dB, t0 = 50, SNR -5..12 dB (18 pts)
- trials: 3 000 (-5..0), 10 000 (1..4), 25 000 (5..7), 40 000 (8..12) = 333 000 trials; 1 998 000 bits per algorithm
- metric: BER = global bit errors / global bit count
- algorithms plotted: biased GS, EM-GS, exhaustive search (LS), exhaustive search (ML), ZF w/ known phase
- omitted: CM-ZF — described only as extending ref. [39]; not a specification
- config fingerprint: `87f2337c9a5ef7c26ed2d749f5d6f20b0d4a59883fce0283f934203fcb9117f1`
- error bars: **intentionally omitted from the visual** (values retained in the aggregate files)
- note: Zero-error points are omitted from the log axis rather than plotted at zero; their one-sided 95% Wilson upper bounds (1.1e-5 at 240 000 bits) are in aggregate.csv.

---

## `fig7b_clean`

**Cui Fig. 7(b) — BER vs SNR, large-scale configuration**

- source store: `results/track_a/fig7b`
- configuration: N x K = 100 x 6, 16-QAM, RSR = 12 dB, t0 = 50, SNR -5..12 dB (18 pts)
- trials: 2 000 (-5..0), 5 000 (1..5), 10 000 (6..9), 15 000 (10..12) = 122 000 trials; 2 928 000 bits per algorithm
- metric: BER = global bit errors / global bit count
- algorithms plotted: biased GS, EM-GS, ZF w/ known phase
- omitted: exhaustive search — excluded exactly as Cui does (16^6 = 16.7M candidates per trial); CM-ZF — unspecified
- config fingerprint: `17aef19fda93aa00cbde97937e027b4cb976edba33f38a8ace62e5c825eeb18f`
- error bars: **intentionally omitted from the visual** (values retained in the aggregate files)
- note: Measured EM-GS/ZF SNR gap 3.79-3.97 dB; Cui states 'between 3 ~ 4 dB'.

---

## `fig8_clean`

**Cui Fig. 8 — BER vs RSR (4-QAM, body-text interpretation)**

- source store: `results/track_a/fig8`
- configuration: N x K = 36 x 3, 4-QAM, SNR = 3 dB, t0 = 50, RSR 0..25 dB (26 pts)
- trials: 3 000 (0..6), 8 000 (7..14), 14 000 (15..25) = 239 000 trials; 1 434 000 bits per algorithm
- metric: BER = global bit errors / global bit count
- algorithms plotted: biased GS, EM-GS, exhaustive search (LS), exhaustive search (ML)
- omitted: ZF w/ known phase — evaluated and present in aggregate.csv, but Cui plots no ZF curve in Fig. 8; CM-ZF — unspecified
- config fingerprint: `87f2337c9a5ef7c26ed2d749f5d6f20b0d4a59883fce0283f934203fcb9117f1`
- error bars: **intentionally omitted from the visual** (values retained in the aggregate files)
- note: Cui's caption says 16-QAM but the body text says 4-QAM; the plotted BER levels support 4-QAM (see fig8_16qam_diagnostic).

---

## `fig8_16qam_diagnostic`

**DIAGNOSTIC — Fig. 8 run at 16-QAM as the caption claims**

- source store: `results/track_a/fig8_16qam`
- configuration: N x K = 36 x 3, 16-QAM, SNR = 3 dB, RSR {0,6,12,18,25} dB (5 pts)
- trials: 3 000 trials/RSR = 15 000 trials; 180 000 bits per algorithm
- metric: BER = global bit errors / global bit count
- algorithms plotted: biased GS, EM-GS, exhaustive search (LS), exhaustive search (ML)
- omitted: ZF w/ known phase — kept out to match Fig. 8's curve set
- config fingerprint: `925f2ab8975505e255f5c37af74d115be38bc302991267f95f440e5f61ce4472`
- error bars: **intentionally omitted from the visual** (values retained in the aggregate files)
- note: DIAGNOSTIC ONLY, not a reproduction of Fig. 8. Median BER ratio to Cui: 4-QAM 0.82 vs 16-QAM 24.12, so the body text is correct and the caption is in error.

---

## `b1_clean`

**Track B, B1 — channel NMSE vs SNR (EXACT nonlinear model)**

- source store: `rydberg-trackb:results/track_b`
- configuration: Geometric ULA, N = 8, K = 3, L_k ~ U{3..7} per realization, RSR = 12 dB, t0 = 50, P in {10, 30}
- trials: 400 trials/point
- metric: Channel NMSE_G = sum||Ghat-G||_F^2 / sum||G||_F^2, 10 log10
- algorithms plotted: Cui biased GS, Cui EM-GS — both on the EXACT model Z = |GS+B+W|
- omitted: Xu linearized LS and the HS-GS structural prototype — the frozen baseline plots only the two unstructured exact-model estimators
- config fingerprint: `n/a`
- error bars: **intentionally omitted from the visual** (values retained in the aggregate files)
- note: NO linearization anywhere on this path. Bootstrap 95% CIs are in baseline_preliminary.json.

---

## `b2_clean`

**Track B, B2 — channel NMSE vs pilot length P (EXACT nonlinear model)**

- source store: `rydberg-trackb:results/track_b`
- configuration: Geometric ULA, N = 8, K = 3, L_k ~ U{3..7}, RSR = 12 dB, SNR = 5 dB, P in {6,10,14,20,30,40}
- trials: 400 trials/point
- metric: Channel NMSE_G, ratio of sums, 10 log10
- algorithms plotted: Cui biased GS, Cui EM-GS — both on the EXACT model Z = |GS+B+W|
- omitted: as B1
- config fingerprint: `n/a`
- error bars: **intentionally omitted from the visual** (values retained in the aggregate files)
- note: P = 2K marked. NO linearization anywhere on this path.
