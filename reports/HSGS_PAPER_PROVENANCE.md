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
| **[LIT]** | a published paper. Verified ones are marked; the rest carry `\todo{VERIFY CITATION}` in the manuscript. |
| **[MATH]** | derived in the paper; no measurement involved. |
| **[EXP-B]** | Track B classical experiments — `trackB_hankel_emgs/`, config `RSR 12 dB, T=50, 4 Cadzow sweeps, SNR −5…20`, statistic ratio-of-sums. |
| **[EXP-D]** | Track D experiments — `scratch/trackD_partB9_sweeps.py`, config `RSR 10 dB, T=100, 1 Cadzow sweep, SNR ~ U[−10,20]`, adaptive order, statistic paired per-trial median. |
| **[CRLB]** | the bound computation — `rydberg_sim/crlb.py`, `scripts/constrained_crlb.py` → `results/track_b/constrained_crlb.json`, 400 trials. |
| **[DRAFT]** | carried from the earlier manuscript `paper/hsgs.tex` and **not** reproducible from a committed store. Every instance is flagged. |

---

## ⚠ Read this first — corrections made while writing this document

Producing this file did what it was supposed to do: tracing each number to a
store surfaced **five figures I had carried from the old manuscript's prose
without checking them against the data.** All five are now corrected in the
manuscript. My earlier statement that "no number was untraceable" was wrong,
and this is the correction.

| claim | was (from `hsgs.tex` prose) | now (from the store) | store |
|---|---|---|---|
| aperture gains at `N`=8/16/32 | −0.19 / +0.78 / **+2.85** dB | **+0.029 / +0.812 / +2.452** dB | `experiment_B_array_size.csv`, `mean_gain_db` |
| EM-GS spread across `N` | 0.012 dB | **0.036 dB** | same file, `em_gs_db_mean_over_points` |
| win rates | 33.5 / 74.0 / 95.3 % | **33.4 / 78.0 / 96.3 %** | same file, `win_rate` |
| constraint active | 57.5 / 92.7 / 99.6 % | **54.1 / 92.7 / 99.4 %** | same file, `active_frac` |
| CCRB below CRLB at `N`=32,`P`=30 | 0.87–9.98 dB | **7.05–7.11 dB** | `constrained_crlb.json`, `b3` `N32_P30` |
| HS-GS over EM-GS at `N`=32 | 2.27–3.04 dB | **1.90–3.09 dB** | `figure_array_size_comparison.csv`, `N=32` |
| the replication gap | 2.85 vs 2.680 → 0.17 dB | **2.452 vs 2.680 → 0.23 dB** | both of the above |

**Why the old numbers exist.** The draft says these were averaged over
"twelve operating points" per array size. **No committed store contains a
twelve-point aperture sweep** — `summary.json` `experiment_B` has seven SNR
points per `N`, and so does `figure_array_size_comparison.csv`. That run either
predates the stores or was never committed. The seven-point numbers are what
can be defended, so those are what the paper now says.

Note this *weakens* the headline slightly (`N=8` becomes +0.03 rather than
−0.19) but does not change any conclusion: the `N=8` mixed-result reading rests
on the per-SNR values, which are traceable and unchanged.

**One claim could not be resolved either way and now carries a `\todo`:** the
"EM-GS within 0.05 dB of the CRLB" and "0.30 / 0.24 dB separation at −5/0 dB"
figures. Differencing the two relevant files gives 0.164 / −0.050 / −0.001 /
0.067 dB at SNR = 5/10/15/20 and 0.44 / 0.24 dB at −5/0 dB — but those files
are **separate runs**, so differencing them is not the paired comparison the
sentence claims. Re-derive from one paired store before submission.

---

# Title, author, abstract

### Title and author block (lines 30–39)
**[GIVEN]** Author name, institution and email from the study brief. The
department string is **[DRAFT]** — it comes from `paper/hsgs.tex`, whose own
byline is the placeholder "Author Name", so it is not an independent source.
Carries `\todo{CONFIRM DEPARTMENT}`.

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
- Gerchberg–Saxton as the natural solver → Gerchberg & Saxton, *Optik* 35, 237–246, 1972. **Unverified** — page numbers to confirm.
- "recent work unrolls it" → Xiao *et al.*, *IEEE SPL* 33, 1696–1700, 2026, DOI `10.1109/LSP.2026.3685170`. **Verified from the PDF's page 1 and its DOI string.**
- "a rank projection … inside projected gradient descent" → Xu *et al.*, *IEEE WCL* 14(9), 2025 (arXiv:2503.08985). Supplied by the author from the PDF; start page 2961 confirmed, **full page range to verify.**

### ¶2 — "their accuracy does not improve as the array grows" (lines 79–86)
**[MATH]**. Follows from row `n` of `GS` entering only measurement row `n`, so
`J` is block diagonal. Derived in §IV-A of this paper; **measured** in §V-B
(EM-GS moves 0.036 dB across a fourfold `N` change).

### ¶3 — "each propagation path contributes one spatial complex exponential" (lines 88–94)
- Rank deficiency of the lifting → **[MATH]**, plus **[LIT]** Cadzow 1988 and
  Markovsky 2008 for the projection and for structured low-rank approximation.
  **Both unverified.**
- "a 16-path channel … has effective rank 8.73" → **[EXP-D]**, the effective-rank
  re-indexing recorded in `reports/trackD_normalization.md` §A2, computed by
  `scratch/trackD_partA9_normalization.py::true_reff` over 200 trials × 3 users.

### ¶4 — Contributions (lines 96–109)
Each item points forward; sources are those of the sections named. Item 1's
"within 0.05 dB" is **[DRAFT]** and is the unresolved claim flagged above.

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
- **The `\todo`** records that `paper/hsgs.tex:149` states 7.15 dB. The 0.08 dB
  difference **could not be reconciled from any committed result**; it is
  reported rather than silently harmonised.

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
- Definition → **[LIT]** Roy & Vetterli, EUSIPCO 2007. **Unverified**, carries
  `\todo`.
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
- "at `N=8` its mean numerical rank is 40.4 against ≈44.7" and "**28.1% of
  realisations have `3ΣL_k > 2NK`**" → **[CRLB]**, `constrained_crlb.json`,
  key `jacobian_rank` (stored values 44.53–44.61 across cells).
- Tangent-space form → **[LIT]** Gorman & Hero, *IEEE TIT* 36(6), 1285–1301,
  1990; Stoica & Ng, *IEEE SPL* 5(7), 177–179, 1998. **Both unverified**,
  carry a joint `\todo`. Implemented in `scripts/constrained_crlb.py`.

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

### §V-A — main result against the bounds (lines 297–314)
**[CRLB] + [EXP-B]**, and the section with the most surviving uncertainty.
- CCRB **7.05–7.11 dB** below the unconstrained bound at `N=32, P=30`:
  computed here by differencing `unconstrained_rank1` and `constrained` for the
  six `b3 N32_P30` entries of `constrained_crlb.json`. **Corrected** from the
  draft's 0.87–9.98 dB, which corresponds to no subset I could reconstruct
  (all 52 stored points span 0.889–49.863 dB).
- HS-GS over EM-GS **1.90–3.09 dB**: `figure_array_size_comparison.csv`, the
  seven `N=32` rows, `gain_db`. **Corrected** from 2.27–3.04.
- The "0.05 dB" / "0.30 and 0.24 dB" claims are the **unresolved [DRAFT]**
  items, now replaced by qualitative wording plus a `\todo` giving the numbers
  the stores do yield and why they are not a paired comparison.
- **Figure 1** is `paper/fig/fig1_nmse_vs_snr.pdf`, reused unmodified from the
  earlier manuscript (commit `b44bf2a`, re-sourced at `19d0a5f`).

### §V-B — aperture scaling (lines 328–347)
**[EXP-B]**, sweep: `trackB_hankel_emgs/experiment_array_size.py`, seven SNR
points per `N`, 4200 / 2800 / 1400 trials at `N` = 8 / 16 / 32.
Store: `trackB_hankel_emgs/results/experiment_B_array_size.csv` (and the same
numbers under `summary.json` → `experiment_B`).

| quantity | value | CSV column |
|---|---|---|
| gain | +0.029 / +0.812 / +2.452 dB | `mean_gain_db` |
| win rate | 33.4 / 78.0 / 96.3 % | `win_rate` |
| constraint active | 54.1 / 92.7 / 99.4 % | `active_frac` |
| EM-GS spread | 0.036 dB | `em_gs_db_mean_over_points`, max − min |

The `N=8` per-SNR reading (**+0.744 dB at −10 dB**, losses of
**0.190–0.403 dB above 0 dB**) is **[EXP-B]** from
`experiment_A_snr.csv` — which is the `N = 8`, `P = 30` sweep, 600 trials per
SNR point (`config.py:16`, `N_DEFAULT = 8`). These were traceable and are
unchanged. The `r_max(8) = 4` versus `L_k ∈ {3,…,7}` explanation is **[MATH]**
from Eq. (4).

**Figure 2** is `paper/fig/fig2_gain_vs_N.pdf`, reused unmodified.

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

### §V-D — out-of-model prediction (lines 401–440)
The section's framing quotes `paper/hsgs.tex:686–700` — the earlier draft's own
limitations sentence naming the clustered falsification test.

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

# Every open `\todo` in this paper, with why

| # | location | what is unresolved |
|---|---|---|
| 1 | author block | department string, from `hsgs.tex` not an independent source |
| 2 | §II RSR | 7.15 dB (draft) vs 7.23 dB (arithmetic), 0.08 dB unreconciled |
| 3 | §III effective rank | Roy–Vetterli citation unverified |
| 4 | §IV-B | Gorman–Hero and Stoica–Ng citations unverified |
| 5 | §V-A | the "0.05 dB" / "0.30, 0.24 dB" bound-tracking figures — not reproducible as a paired comparison from committed stores |
| 6 | §V-D | arXiv:2408.14366 author list, PDF unavailable here |

Plus, in `refs.bib`: Xu's full page range, and page numbers for Cadzow,
Markovsky, Gerchberg–Saxton and Netrapalli.

# What is verified against a primary source

The Xiao PDF (held in this session) supplied, by direct extraction: its own
page 1 (authors, volume, pages 1696–1700), its DOI `10.1109/LSP.2026.3685170`,
and its reference list — which independently confirms Cui [5], Gong ICC [7],
Gong magazine [4], Zhang [3], and Meijerink & Molisch [10].
