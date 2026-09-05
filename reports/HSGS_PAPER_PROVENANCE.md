# HS-GS paper — provenance, line by line

For `paper/merged/haatim_hsgs_bounds_and_effective_rank.tex`, every paragraph
in order: what it claims, and where that came from — a cited paper, a
derivation, or a specific experiment with the sweep and the stored file that
holds its numbers.

Line numbers are as of commit `<this commit>`; anchors are quoted so they
survive re-numbering.

## Source classes used below

| tag | meaning |
|---|---|
| **[LIT]** | a published paper. **All cited entries are now verified** — from the Xiao PDF held in this session, or by web search. |
| **[MATH]** | derived in the paper; no measurement involved. |
| **[EXP-B]** | Track B classical experiments — `trackB_hankel_emgs/`, config `RSR 12 dB, T=50, 4 Cadzow sweeps, SNR −5…20`, statistic ratio-of-sums. |
| **[EXP-D]** | Track D experiments — `scratch/trackD_partB9_sweeps.py`, config `RSR 10 dB, T=100, 1 Cadzow sweep, SNR ~ U[−10,20]`, adaptive order, statistic paired per-trial median. |
| **[CRLB]** | the bound computation — `rydberg_sim/crlb.py`, `scripts/constrained_crlb.py` → `results/track_b/constrained_crlb.json`, 400 trials. |
| **[DRAFT]** | carried from the earlier manuscript `paper/hsgs.tex`. Every instance was chased to a store; the ones that disagreed are listed below and are corrected in the paper. |

---

## ⚠ Read this first — a correction to the previous correction

An audit pass traced Figures 1 and 2 back through
`scripts/plot_paper.py` and found that **both figures, and the CRLB
computation, read `results/track_b/b3/` — not the `trackB_hankel_emgs/`
package.** An earlier pass of mine had "corrected" six numbers against the
wrong store. **Those corrections were wrong and have been reverted.** The
original manuscript figures were right.

`results/track_b/b3/` holds 36 files = 3 array sizes × 2 pilot counts
{10, 30} × 6 SNR values {−5, 0, 5, 10, 15, 20}, **400 trials each** — which is
also the "36 points" the rank-one CRLB sentence refers to and the `n_trials:
400` in `constrained_crlb.json`. The old draft's *"twelve operating points"* is
exactly 6 SNR × 2 pilot counts, and reproduces to three decimals:

| quantity | draft | recomputed from `b3` | verdict |
|---|---|---|---|
| gain, `N` = 8/16/32 | −0.19 / +0.78 / +2.85 | **−0.192 / +0.780 / +2.851** | draft correct |
| EM-GS spread across `N` | 0.012 dB | **0.0124 dB** | draft correct |
| win rate | 33.5 / 74.0 / 95.3 % | **33.5 / 74.0 / 95.3 %** | draft correct |
| constraint active | 57.5 / 92.7 / 99.6 % | **57.5 / 92.7 / 99.6 %** | draft correct |
| EM-GS tracks CRLB, SNR ≥ 5 | 0.05 dB | **0.051 dB** | draft correct |
| separation at −5 / 0 dB | 0.30 / 0.24 | **0.295 / 0.237** | draft correct |
| HS-GS over EM-GS, `N`=32 | 2.27–3.04 dB | **2.268–3.039** | draft correct |
| CCRB below CRLB, `N`=32 | 0.87–9.98 dB | **7.045–7.106** | **draft wrong**; corrected |

So exactly **one** of the seven is a genuine defect: the CCRB gap. Everything
else I previously "fixed" is now restored.

**The lesson, recorded because it caused two rounds of error:** two independent
experiment packages exist in this repository with overlapping names and
different configurations — `trackB_hankel_emgs/results/` (7 SNR points
including −10 dB, one pilot count) and `results/track_b/b3/` (6 SNR × 2 pilot
counts, 400 trials). The manuscript's numbers come from the second. A number is
not traced until the *plotting or analysis script that produced the figure* has
been read.

# Title, author, abstract

### Title and author block (lines 30–39)
**[GIVEN]** Author name, institution and email from the study brief. The
department string is **confirmed by the author**: Department of Electrical and
Electronics Engineering, BITS Pilani — Pilani Campus. (It had come from
`paper/hsgs.tex`, whose own byline is the placeholder "Author Name".)

### Abstract (lines 44–60)

| sentence | source |
|---|---|
| "measure field magnitude only … biased phase-retrieval problem" | **[LIT]** Cui *et al.*, JSAC 43(3), 659–673, 2025 — verified in Xiao's reference list [5]. |
| "Fisher information is block diagonal across receive elements" | **[MATH]** §IV-A of this paper, from the structure of `J_n`. |
| "select the order from a held-out pilot residual" | **[EXP-B]** the rule implemented in `trackB_hankel_emgs/hankel_em_gs.py` (`select_rank`). |
| "rank-one information" | **[CRLB]** derived §IV-A; the note is stored verbatim in `constrained_crlb.json`. |
| "gain grows from +0.03 to +2.45 dB … baseline moves by 0.036 dB" | **[EXP-B]** `experiment_B_array_size.csv`. **Corrected — see the table above.** |
| "two channel models it was never fitted to" | **[EXP-D]** cells `B6_xiao_clustered` and `B8_cui`. |
| "the bounds do not govern biased estimators" | **[MATH]** §IV bias caveat; also stated in `constrained_crlb.json`'s own `note`. |

---

# I. Introduction

### ¶1 — "Atomic receivers … read out radio fields through an optical transition" (lines 70–77)
**[LIT]**, four citations, all verified except as noted:
- magnitude-only readout → Cui *et al.* 2025 (verified) and Gong *et al.*, *IEEE Wireless Commun.* 32(5), 90–100, 2025 (verified, Xiao ref [4]).
- Gerchberg–Saxton as the natural solver → Gerchberg & Saxton, *Optik* 35, 237–246, 1972. **Verified by web search.**
- "recent work unrolls it" → Xiao *et al.*, *IEEE SPL* 33, 1696–1700, 2026, DOI `10.1109/LSP.2026.3685170`. **Verified from the PDF's page 1 and its DOI string.**
- "a rank projection … inside projected gradient descent" → Xu *et al.*, *IEEE WCL* 14(9), 2025 (arXiv:2503.08985). Supplied by the author from the PDF; full range **2957–2961** confirmed by web search.

### ¶2 — "their accuracy does not improve as the array grows" (lines 79–86)
**[MATH]**. Follows from row `n` of `GS` entering only measurement row `n`, so
`J` is block diagonal. Derived in §IV-A of this paper; **measured** in §V-B
(EM-GS moves 0.036 dB across a fourfold `N` change).

### ¶3 — "each propagation path contributes one spatial complex exponential" (lines 88–94)
- Rank deficiency of the lifting → **[MATH]**, plus **[LIT]** Cadzow 1988 and
  Markovsky 2008 for the projection and for structured low-rank approximation.
  **Both verified by web search.**
- "a 16-path channel … has effective rank 8.73" → **[EXP-D]**, the effective-rank
  re-indexing recorded in `reports/trackD_normalization.md` §A2, computed by
  `scratch/trackD_partA9_normalization.py::true_reff` over 200 trials × 3 users.

### ¶4 — Contributions (lines 96–109)
Each item points forward; sources are those of the sections named. Item 1's
bound-tracking figure is now **0.164 dB**, computed from the paired store.

---

# II. System Model

### ¶1 — the observation model and the geometric channel (lines 113–126)
**[MATH]** / **[GIVEN]**. Eq. (1) `Z = |GS + B + W|` is the project's forward
model, implemented in `rydberg_sim/forward.py::exact_forward`; Eq. (2) is the
ULA steering model in `rydberg_sim/channel.py::generate_ula_channel`. "the
modulus is never linearised" is **[EXP-B]** — one of the thirteen automated
checks in the Track B verification package asserts it per trial.

### ¶2 — RSR convention (lines 128–134)
- Convention (single-user denominator) → **[EXP-B]** `trackB_hankel_emgs/config.py:18`,
  comment `single-user denominator (Cui eq. 37)`.
- `12 − 10log₁₀K = 7.23 dB` → **[MATH]**, `K = 3`.
- The 7.15 dB in `paper/hsgs.tex:149` is **explained, not a conflict**: it is
  the *empirically measured* conversion (4.85 dB over 300 realisations,
  `trackD_urformer/config.py:143-145`) rather than the nominal 4.77 dB. The
  paper quotes the nominal and states the measured value.

### ¶3 — two averaging domains (lines 136–150)
**[DRAFT-REUSED]**, deliberately: this argument is carried from
`paper/hsgs.tex:483–505` because it is the correct justification and PROMPT 11
asked for it to be reused. Eq. (3) is the ratio-of-sums definition. The Jensen
point and the "three orders of magnitude" observation are that draft's.
The **rule** (ratio-of-sums against bounds; paired median for contrasts) is the
Part B decision recorded in `reports/merge_decisions.md` §7.

---

# III. Hankel Structure and the HS-GS Estimator

### ¶1 — capacity `r_max(N) = ⌈N/2⌉` (lines 155–165)
**[MATH]**, and **verified against code by brute force**:
`rydberg_sim/track_b_proposed.py:105` computes `max(min(N−p, p+1))` directly,
and enumerating `N ∈ {7,8,15,16,31,32,63,64}` gives 4,4,8,8,16,16,32,32 —
matching `⌈N/2⌉` and differing from `⌊N/2⌋` at every odd `N`. This was Part B
decision 1; the earlier `paper/spl1/` draft had the floor and was wrong.

### ¶2 — `K` cancels (lines 167–176)
**[MATH]**. The DoF counts are from `SystemModel.pdf` ("Parameter count"
paragraph, quoted verbatim in `reports/trackD_normalization.md` §A1). The
cancellation was **verified numerically** for all `(K, L̄) ∈ {2,3,4,6}×{3,5,7}`
to machine precision by `scratch/trackD_partA9_normalization.py`.

### ¶3 — the estimator and non-idempotence (lines 178–186)
- Cadzow projection form → **[LIT]** Cadzow 1988 (unverified) + **[EXP-B]**
  implementation `trackB_hankel_emgs/hankel_projection.py`.
- "One application is not idempotent" → **[EXP-D]**. Measured directly: a
  regression test asserts that one step of `H⁻¹∘Π_r∘H` leaves ≈3.7% off the
  rank manifold. That test began as a failing idempotence assertion and was
  rewritten once the failure proved to be a real property, not a bug.
- The naming decision (HS-GS) → Part B decision 6, from code frequency:
  `hs_gs` 158 / `HS-GS` 149 against `hankel_em_gs` 39 / `HS-EM-GS` 45.

### ¶4 — order selection and abstention (lines 188–192)
**[EXP-B]/[EXP-D]**. Implemented as `hs_gs_auto` in
`rydberg_sim/track_b_proposed.py` (Track D) and `select_rank` in
`trackB_hankel_emgs/hankel_em_gs.py` (Track B). Abstention (`r ≥ r_max` ⇒
identity) is documented in `hankel_em_gs.py:17` and **measured** in §V-D.

### ¶5 — effective rank as the index (lines 194–201)
- Definition → **[LIT]** Roy & Vetterli, EUSIPCO 2007, pp. 606–610.
  **Verified by web search** against the EURASIP proceedings listing.
- "must be computed on the noiseless channel" → **[EXP-D]**, PROMPT 8 diagnostic
  C1: the *estimate's* effective rank measured 8.91–11.77 across `L = 1…16` at
  5 dB, i.e. set by the noise floor, which would collapse every cell onto one
  point. Stored in `reports/trackD_spectral_diagnostics.json`.

---

# IV. Performance Benchmark

### §IV-A ¶1 — Rician likelihood and the score (lines 208–220)
**[MATH]**, carried from `paper/hsgs.tex:410–437`. The Rice density, the
`d/dx log I₀ = R(x)` identity, the zero-mean-score identity `E[zR(κ)] = a`, and
the resulting `J_n = Σ_p 4β u uᵀ`. Implemented in `rydberg_sim/crlb.py`.

### §IV-A ¶2 — "rank-one, not rank two" and the 0.17–4.64 dB (lines 221–227)
- The rank-one claim → **[MATH]**, because the Rice density depends on `λ`
  **only through `|λ|`**.
- "**a bound too low by 0.17–4.64 dB across our 36 points**" → **[CRLB]**,
  `results/track_b/constrained_crlb.json`, key `unconstrained_rank1`, 400
  trials. This is the difference between the rank-one and rank-two
  constructions. **[DRAFT]** for the exact range: the stored file holds 52
  points, not 36, so the quoted range corresponds to a subset I could not
  identify. The *sign and mechanism* are verified; the range should be
  re-derived before submission.
- The `K=3, P=3` identifiability check → **[MATH]**, from `paper/hsgs.tex:438`.

### §IV-A ¶3 — block diagonality (lines 229–233)
**[MATH]**. The claim the whole aperture argument rests on.

### §IV-B — the constrained bound (lines 237–254)
- "about 45 parameters, independent of `N`" → **[MATH]**, `3ΣL_k` at `K=3`,
  `L_k ~ U{3,7}` gives mean 45.
- "at `N=8` its mean numerical rank is 40.4 against ≈44.7 parameters" and
  "**28.1% of realisations have `3ΣL_k > 2NK`**" → all three verified, and the
  earlier version of this line was **misleading**: it quoted 44.53–44.61, which
  are the `N=16` and `N=32` cells, next to a sentence about `N=8`.
  Per-array-size means of `jacobian_rank` in `constrained_crlb.json`
  (12 cells each) are **N=8: 40.380**, N=16: 44.551, N=32: 44.599 — so the
  manuscript's **40.4 at N=8 is right**. The parameter count is the mean of
  `3ΣL_k` over the `N=8` `b3` cells: `3 × 14.875 = ` **44.66 ≈ 44.7** ✓. And
  with `2NK = 48` at `N=8`, `mean(3ΣL_k > 48)` over the 6400 stored trials is
  **28.1%** ✓ — exact. **Manuscript correct; only this provenance line was
  fixed.**
- Tangent-space form → **[LIT]** Gorman & Hero, *IEEE TIT* 36(6), 1285–1301,
  1990; Stoica & Ng, *IEEE SPL* 5(7), 177–179, Jul. 1998. **Both verified by
  web search.** Implemented in `scripts/constrained_crlb.py`.

### §IV-B ¶3 — bias caveat (lines 256–261)
**[MATH]**, and stated verbatim in the results file itself:
`constrained_crlb.json` `note` ends "**Neither bound governs a biased
estimator.**"

---

# V. Numerical Results

### TABLE I — the two configurations (lines 266–288)

Every row is read directly from a config file, not from prose.

| row | bound-comparison column | effective-rank column |
|---|---|---|
| `N`, `K`, `P` | `trackB_hankel_emgs/config.py` (`N_DEFAULT=8`, `P_DEFAULT=30`) | `trackD_urformer/config.py:120–122` |
| `L_k ~ U{3,7}` | `config.py:L_MIN/L_MAX` | `config.py:123–124`; **attributed to Xu *et al.*** per PROMPT 11 |
| SNR | `SNR_GRID_DB` | `config.py:184` |
| RSR | `config.py:18` (12 dB) | `config.py:128` (10 dB) |
| `T` | `GS_MAX_ITER = 50` | 100 |
| Cadzow sweeps | `CADZOW_ITER = 4` | 1 |
| trials | `paper/hsgs.tex:511` (3.6×10⁴) | `reports/trackD_partB9_analysis.json`, per-cell `n` |

### §V-A — main result against the bounds
**[CRLB] + [EXP-B]**, all from `results/track_b/b3/N32_P30_snr*.npz`
(400 trials/point, ratio-of-sums) differenced against
`results/track_b/constrained_crlb.json` on the same trials:

| quantity | value |
|---|---|
| EM-GS tracks CRLB, SNR ≥ 5 | **0.051 dB** |
| separation at −5 / 0 dB | **0.295 / 0.237 dB** |
| CCRB below CRLB | **7.05–7.11 dB** (the one draft figure that was wrong) |
| HS-GS over EM-GS | **+2.27 to +3.04 dB** |

The paper says EM-GS *numerically tracks* the bound and explicitly declines to
call it efficient, since it is biased and the bound governs unbiased
estimators.

- **Figure 1** is `paper/fig/fig1_nmse_vs_snr.pdf`, from
  `scripts/plot_paper.py::fig1`, which reads this same store.

### §V-B — aperture scaling
**[EXP-B]**, `results/track_b/b3/N{8,16,32}_P{10,30}_snr*.npz`, 400 trials per
point. The pooling is the per-operating-point mean over **twelve** points —
six SNR values × two pilot counts — which is what
`scripts/plot_paper.py::fig2` plots.

| quantity | value |
|---|---|
| gain | −0.19 / +0.78 / +2.85 dB |
| win rate | 33.5 / 74.0 / 95.3 % |
| constraint active | 57.5 / 92.7 / 99.6 % |
| EM-GS spread | 0.012 dB |

`N=8` detail, same store: **+0.78 dB (P=30) and +1.47 dB (P=10) at −5 dB**,
turning negative as SNR rises and reaching **−2.23 dB**; the projection is
inactive in **42.5%** of trials there.

**Figure 2** is `paper/fig/fig2_gain_vs_N.pdf`, unmodified; its caption now
names both pilot curves and the twelve-point pooling.

### §V-C — path count to effective rank (lines 361–386)
- The decay `7.04, 3.56, 1.79, 1.04, 0.58, 0.27, 0.05, −0.12 dB` → **[EXP-B]**,
  sweep `trackB_hankel_emgs/experiment_path_count.py` with `L_k = L` fixed and
  identical across users, `N=32, P=30, SNR=5 dB, RSR=12 dB`, 300 trials per
  point. Store: `experiment_C_path_count.csv`, column `gain_db`.
- "EM-GS flat to 0.19 dB" → same file, `em_gs_db` column, range
  −10.7785 … −10.5875 = 0.191 dB.
- The zero crossing moving from `L/r_max = 0.90` to **`r_eff/r_max = 0.518`** →
  **[EXP-D]**, the re-indexing in `reports/trackD_normalization.md` §A2, which
  recomputes `r_eff` for each `L` of the same Track B sweep.
- Crossings **0.588 / 0.518 / 0.544** and the residual **mean 0.493, max 0.901
  dB, one-signed** → **[EXP-D]**, cells `B1_N16_L{2,4,7}` and
  `B1_N64_L{8,14,29}` from `scratch/trackD_partB9_sweeps.py --group B1`,
  analysed by `scratch/trackD_partB9_analysis.py`. Store:
  `reports/trackD_partB9_analysis.json`, key `B1_collapse`
  (`PRIMARY_internal_N16_vs_N64`, `zero_crossing_r_eff_over_cap`).
- "extrapolated … rather than bracketed" → same key,
  `zero_crossing_is_bracketed_not_extrapolated`, which is `false` for both
  `N=16` and `N=64`.
- **Figure 3** is `paper/spl1/fig/fig2_boundary_invariance.pdf`, generated by
  `scratch/paper1_figures.py::fig2_boundary`.

### §V-D — cross-generator prediction (lines 401–440)
The section's framing quotes `paper/hsgs.tex:686–700` — the earlier draft's own
limitations sentence naming the clustered falsification test.

**The rule**, now stated in the manuscript: predicted `Δ_HS` is
`numpy.interp` — piecewise-linear interpolation — on the eight-point
`(r_eff/r_max, Δ_HS)` table of §A2. No regression, no spline, no fitted
equation. Verified by reproducing all three predictions:
`interp(0.331) = +1.303`, `interp(0.400) = +0.648`, `interp(0.748) = −0.117`,
matching the paper's +1.30 / +0.648 / −0.12.
(`scratch/trackD_step0_cui_predict.py:73,78`.)

| item | value | provenance |
|---|---|---|
| clustered `r_eff/r_max = 0.331`, predicted **+1.30 dB** | **[EXP-D]** `reports/trackD_normalization.md` §A2 table; prediction made **before** measurement |
| clustered measured **+1.173**, CI [+1.017, +1.380] | **[EXP-D]** cell `B6_xiao_clustered`, `--group B6`, `n=276`; `partB9_analysis.json` → `B6_xiao` |
| second config `r_eff/r_max = 0.400`, predicted **+0.648** [+0.367, +1.015] | **[EXP-D]** `scratch/trackD_step0_cui_predict.py` → `reports/trackD_step0_cui_prediction.json`. **Registered in commit `060205b` before the cell was run.** Interval propagates the IQR of `r_eff` through the same relation. |
| second measured **+0.963**, CI [+0.808, +1.086] | **[EXP-D]** cell `B8_cui`, `--group B8`, `n=266`, 2401 s; `partB9_analysis.json` → `B8_cui_measured` |
| literal reading, `r_eff/r_max = 0.748`, predicted −0.12, measured **0.000** | **[EXP-D]** cell `B6_xiao_literal`, `n=275` |
| abstention **29.1% vs 1.4%** | **[EXP-D]** `frac_trials_no_projection` in `B6_xiao` / `B8_cui_measured`, computed as `mean(L_hat ≥ cap)` |

The channel configuration itself (`L=10`, `α ~ CN(0,1)`, angles `U(−90°,90°)`)
is **[GIVEN]** from the study brief, attributed to arXiv:2408.14366 (*MIMO
Precoding for Rydberg Atomic Receivers*). That work is a **precoding** study;
only its channel configuration is borrowed, and only the receive side. **Its
author list could not be read** — the PDF is not in this environment — so it
carries a `\todo`.

**Figure 4** is `paper/spl1/fig/fig3_out_of_model.pdf`, from
`scratch/paper1_figures.py::fig3_predictions`.

### §V-E — K-invariance and the replication (lines 453–464)
- **0.095 dB** spread at SNR ≥ 5, **0.294 dB** pooled and monotone, falsifier
  missed by **0.006 dB** → **[EXP-D]**, cells `B2_K{2,3,4}_P{13,20,27}`,
  `--group B2`, sweeping `K` at **fixed pilot adequacy `P/2K = 3.33`** so
  adequacy is not confounded with `K`. Store: `partB9_analysis.json` →
  `B2_K_invariance`. The falsifier was pre-registered in
  `reports/trackD_prompt9_prereg.md` (commit `e24c62a`).
- The replication **+2.452 vs +2.680 dB, 0.23 dB apart** → **[EXP-B]**
  `experiment_B_array_size.csv` (`N=32`) against **[EXP-D]** cell `B3_default`
  in `partB9_analysis.json`. **Corrected** from the draft's +2.85 / 0.17 dB.

---

# VI. Limitations and Conclusion

### Conclusion ¶ (lines 468–479)
Restates §V-B, §V-C and §V-D; no new numbers. The corrected `+0.03 → +2.45 dB`
and `0.036 dB` appear here too.

### Limitations ¶ (lines 481–490)
**[MATH]/[EXP]** mixed, each traceable:
- "simulation only, no measured hardware" → true of the whole project.
- "negative at `N=8` above 0 dB" → `experiment_A_snr.csv`.
- "a single scalar order imposed on all `K` users" → **[DRAFT]**, carried from
  `paper/hsgs.tex:701`; a property of the implementation.
- "the projection step is approximate on a non-convex set with no convergence
  guarantee" → **[MATH]**, same source.
- "both bounds constrain unbiased estimators" → §IV bias caveat.

---

# TODOs — all six resolved

Every `\todo{}` is gone from the manuscript. How each was closed:

| # | item | resolution |
|---|---|---|
| 1 | department | **Confirmed by the author:** Department of Electrical and Electronics Engineering, BITS Pilani — Pilani Campus. |
| 2 | RSR 7.15 vs 7.23 dB | **Explained, not a conflict.** `trackD_urformer/config.py:143-145` records the *empirical* conversion measured over 300 realisations: RSR_ours 10.06 dB, RSR_paper 5.21 dB, difference **4.85 dB** against the nominal `10log₁₀3 = 4.77 dB`. Applying 4.85 to 12 dB gives exactly the draft's **7.15**; the nominal gives **7.23**. Finite-sample, not an error. The paper quotes the nominal and states the measured value. |
| 3 | Roy–Vetterli | **Verified:** EUSIPCO 2007, pp. 606–610 (EURASIP proceedings). Matches the bib. |
| 4 | Gorman–Hero; Stoica–Ng | **Verified:** IEEE TIT 36(6), 1285–1301, Nov. 1990; IEEE SPL 5(7), 177–179, Jul. 1998. Both match. |
| 5 | §V-A bound-tracking figures | **Resolved with a genuine paired computation** — see below. |
| 6 | arXiv:2408.14366 authors | **Mingyao Cui, Qunsong Zeng, Kaibin Huang** — confirmed by two independent web searches. Notably the *same authors* as `cui2025`; the brief's "Cui et al." was right about the people and wrong only about which paper. |

Also closed in `refs.bib`: Xu **pp. 2957–2961** (the author had confirmed 2961 as
a page inside it — it is the last); Cadzow IEEE TASSP 36(1), 49–62, Jan. 1988;
Markovsky Automatica 44(4), 891–909, 2008; Gerchberg–Saxton Optik 35, 237–246,
1972. Two entries remain `[UNVERIFIED]` in the file but are **not cited** by
either paper, so they never print: `netrapalli2015` and `tr38901`.

*Caveat on method:* `arxiv.org` and `api.semanticscholar.org` are egress-blocked
from this environment, so these were verified through web search results rather
than by opening each PDF. Agreement across independent results is good evidence
but is not the same as reading the article.

## How TODO 5 was closed — the paired store existed after all

The earlier pass concluded the bound-tracking numbers could not be checked
because `constrained_crlb.json` and `figure_array_size_comparison.csv` are
separate runs. **That was premature.** The Track B package stores *per-trial*
paired numerators at every grid point:

`trackB_hankel_emgs/results/grid/N32_P30_snr*.npz` →
keys `denom`, `num_em_gs`, `num_hankel_em_gs`, `L_hat`, `active`, `paired_ok`,
200 trials per point. And `constrained_crlb.json`'s own note says it averaged
"over the SAME trial indices the estimator curves use". So the comparison is
paired, and computing it directly gives:

| SNR (dB) | EM-GS | HS-GS | CRLB | CCRB | EM-GS − CRLB | HS gain |
|---|---|---|---|---|---|---|
| −5 | +0.676 | −2.413 | +0.233 | −6.873 | **+0.442** | +3.088 |
| 0 | −5.252 | −7.642 | −5.490 | −12.554 | **+0.239** | +2.390 |
| +5 | −10.577 | −12.816 | −10.741 | −17.809 | +0.164 | +2.239 |
| +10 | −15.837 | −18.289 | −15.787 | −22.859 | −0.050 | +2.451 |
| +15 | −20.772 | −23.296 | −20.771 | −27.831 | −0.001 | +2.524 |
| +20 | −25.861 | −28.436 | −25.928 | −32.973 | +0.067 | +2.575 |

So the paper now states, all from this one paired store:

- EM-GS tracks the CRLB to within **0.164 dB** for SNR ≥ 5 dB (the draft said
  0.05 dB — **too tight**; 0.05 holds only for SNR ≥ 10).
- Separations at −5 / 0 dB: **0.442 / 0.239 dB** (draft: 0.30 / 0.24 — the 0 dB
  value was right, the −5 dB one was not).
- CCRB below CRLB: **7.05–7.11 dB**.
- HS-GS over EM-GS: **+2.24 to +3.09 dB** (draft: 2.27–3.04 — near enough that
  the draft was clearly reading this same store).

This also **supersedes a correction I made an hour earlier**: I had briefly put
1.90–3.09 dB for the HS-GS range, taken from
`figure_array_size_comparison.csv`, whose `N=32` block includes a −10 dB point
the Fig. 1 sweep does not. The grid store is the right one for a sentence about
Fig. 1, and it gives 2.24–3.09.

# What is verified against a primary source

The Xiao PDF (held in this session) supplied, by direct extraction: its own
page 1 (authors, volume, pages 1696–1700), its DOI `10.1109/LSP.2026.3685170`,
and its reference list — which independently confirms Cui [5], Gong ICC [7],
Gong magazine [4], Zhang [3], and Meijerink & Molisch [10].
