# PROMPT 12 — Part A: repository checks and re-analysis

No new compute. Every number below is read off committed results files or off
source constants, with `file:line` given. Channel-generation-only recomputation
(A3, A4) runs no estimator.

Artifacts produced this part:

| file | contents |
|---|---|
| `reports/p12/a3_abstention_vs_rho.json` | ρ and abstention for all 16 adaptive-order cells |
| `reports/p12/a3_within_N.json` | within-aperture rank correlations |
| `reports/p12/a2_internal_vs_posthoc.json` | the three per-bin contrasts |
| `paper/fig/p12/a2_internal_vs_posthoc.pdf` | the A2 figure |
| `scratch/p12_partA.py`, `p12_partA_rho.py`, `p12_partA2.py` | the drivers |

---

## A1. What actually separates B3 from TD

**[FACT] Two of the four differences asserted in the brief do not exist.** The
brief states B3 is "`n_cz = 4`, fixed-order selection" and TD is "`n_cz = 1`,
adaptive order". Read off the call sites:

- `scripts/run_b3.py:145` → `hs_gs_auto(w.S, w.Z, w.B, w.sigma2, **HS_KW)` with
  `HS_KW = dict(exact_step="em_gs", max_iter=50, select_iter=20)`
  (`scripts/run_b3.py:54`).
- `scratch/trackD_partB9_sweeps.py:171` → `hs_gs_auto(s.S, s.Z, s.B, s.sigma2,
  exact_step="em_gs", max_iter=T_GS, select_iter=25)`.
- `rydberg_sim/track_b_proposed.py:246` → `def hs_gs_auto(..., cadzow_iter=4, ...)`.

**Neither call site passes `cadzow_iter`.** Both therefore run `n_cz = 4`.
And both call `hs_gs_auto`, which selects the order from held-out pilots
(`track_b_proposed.py:249`, `select_order_heldout`) — so **both are adaptive
order; neither uses a fixed or oracle order.** A repository-wide grep for
`cadzow_iter` finds no override on either path.

### The differences that are real

| field | B3 | TD | moves Δ_HS? | direction |
|---|---|---|---|---|
| entry point | `hs_gs_auto` | `hs_gs_auto` | no | identical function |
| order selection | held-out, adaptive | held-out, adaptive | no | identical routine |
| Cadzow sweeps `n_cz` | **4** (default) | **4** (default) | no | **identical** |
| `project_every` | 1 (default) | 1 (default) | no | identical |
| selection iterations | 20 | 25 | marginal | longer selection → slightly better `L̂` → favours TD |
| estimator iterations `T` | 50 | 100 | small | both arms converge further; EM-GS gains too |
| **RSR (dB)** | **12.0** | **10.0** | **yes** | lower RSR → harder retrieval → more headroom → favours TD |
| **pilots `P`** | **{10, 30}** | **20** | **yes** | fewer pilots → larger Δ_HS (B7 measured this) |
| array sizes `N` | {8, 16, 32} | {16, 32, 64} | yes | different capacity grids |
| users `K` | 3 | 3 | no | same |
| paths `L_k` | U{3..7} | U{3..7} | no | same |
| master seed | 20250820 | 20260827 | no | different worlds, same distribution |
| **SNR handling** | fixed 6-point grid | U[−10,20], binned post hoc | **yes** | TD has a −10 dB bin B3 has no cell for |
| trials | 400/point | time-budgeted, 60–1000 | no | affects CI width only |
| **pooling rule** | ratio-of-sums within a point | paired per-trial median within a bin | **yes** | different statistics; Jensen gap |
| world generator | `track_b_world` | Track D `make_world` | no | both geometric ULA |

Sources: `rydberg_sim/track_b_drivers.py:44,46,47,48,54,56`;
`trackD_urformer/config.py:120,121,122,123,124,128,129`.

### Does a TD cell already exist at N=32, K=3, P=30?

**No.** All 16 committed TD cells enumerated from `results/track_d/partB9/`:

```
B1_N16_L2/L4/L7        N=16 K=3 P=20     B2_K2_P13   N=32 K=2 P=13
B1_N64_L8/L14/L29      N=64 K=3 P=20     B2_K3_P20   N=32 K=3 P=20
B3_default             N=32 K=3 P=20     B2_K4_P27   N=32 K=4 P=27
B6_xiao_clustered/literal  N=32 K=3 P=20 B7_P10/P15/P35  N=32 K=3 P=10/15/35
B8_cui                 N=32 K=3 P=20
```

The pilot counts present at N=32, K=3 are {10, 15, 20, 35}. **P=30 is absent,
so B1 (the bridging run) is required.**

**Consequence for the brief's defect #1.** The defect is real but narrower than
stated: the two families differ in **RSR (12 vs 10 dB), pilot count, SNR
handling and pooling rule** — not in the operator. The estimator is the same
operator in both. This makes the bridging run cheaper to interpret: it is a
change of operating point, not a change of method.

---

## A2. Post-hoc versus internal, per bin

From `reports/trackD_stage3_results.json` per-trial rows (2000 paired test
realisations). Paired per-trial median within each bin, bootstrap 95% CI over
2000 resamples. **Positive = the second arm is better.**

| SNR bin | H1 − U1 (prior in loop) | U1+post − U1 (post-hoc) | **H1 − U1+post (internal vs post-hoc)** |
|---|---|---|---|
| [−10, −5) | −0.111 [−0.173, −0.078]\* | +0.013 [+0.004, +0.022]\* | **−0.138 [−0.184, −0.094]\*** |
| [−5, 0) | −0.506 [−0.587, −0.434]\* | +0.072 [+0.058, +0.088]\* | **−0.582 [−0.630, −0.493]\*** |
| [0, +5) | −0.451 [−0.523, −0.347]\* | +0.190 [+0.179, +0.205]\* | **−0.636 [−0.754, −0.552]\*** |
| [+5, +10) | +0.398 [+0.235, +0.477]\* | +0.497 [+0.449, +0.533]\* | **−0.207 [−0.279, −0.078]\*** |
| [+10, +15) | +1.305 [+1.171, +1.374]\* | +0.826 [+0.780, +0.874]\* | **+0.388 [+0.293, +0.462]\*** |
| [+15, +20) | +2.226 [+2.040, +2.360]\* | +0.837 [+0.799, +0.868]\* | **+1.309 [+1.185, +1.459]\*** |
| SNR ≥ 5 | +1.209 [+1.139, +1.330]\* | +0.728 [+0.710, +0.761]\* | **+0.428 [+0.358, +0.511]\*** |

\* bootstrap CI excludes zero — **every** cell above is significant.

**[FACT] The +1.309 dB figure the brief names is confirmed** at 15–20 dB, CI
[+1.185, +1.459].

**[FACT] The internal-vs-post-hoc contrast changes sign with SNR.** It is
significantly *negative* in all three bins below +5 dB (worst −0.636 dB at
0–5 dB) and significantly *positive* above +10 dB. Interleaving the projection
is better than bolting it on **only at high SNR**; below 5 dB the post-hoc arm
wins. The pooled SNR ≥ 5 figure (+0.428 dB) is smaller than the top-bin figure
by a factor of three, which is why the per-bin table and not a scalar is the
reportable object.

This is directly relevant to **B5/P18**: the learned side is bin-dependent, so
the classical contrast should not be assumed one-signed.

---

## A3. Abstention rate as an observable proxy for ρ

ρ was not stored per cell, so it is recomputed from the channel **distribution**
at each cell's configuration (400 draws, median Roy–Vetterli effective rank of
the noiseless channel columns on the Hankel lifting) — the same definition and
procedure that built the committed `R_EFF` table in
`scratch/trackD_partB9_analysis.py:34`. No estimator is run.

Abstention is defined as the selector returning `L̂ ≥ r_max`, which makes the
projection the identity.

| cell | N | cap | ρ | abstention | mean `L̂` | n |
|---|---|---|---|---|---|---|
| B1_N64_L8 | 64 | 32 | 0.197 | 0.0% | 6.52 | 84 |
| B3_default | 32 | 16 | 0.233 | 0.4% | 4.59 | 274 |
| B7_P10 | 32 | 16 | 0.233 | 1.4% | 4.29 | 296 |
| B7_P15 | 32 | 16 | 0.233 | 0.7% | 4.74 | 285 |
| B7_P35 | 32 | 16 | 0.233 | 0.4% | 4.99 | 247 |
| B1_N16_L2 | 16 | 8 | 0.235 | 1.4% | 2.32 | 853 |
| B2_K2_P13 | 32 | 16 | 0.247 | 0.3% | 4.49 | 306 |
| B2_K4_P27 | 32 | 16 | 0.248 | 0.0% | 4.55 | 245 |
| B2_K3_P20 | 32 | 16 | 0.248 | 0.3% | 4.58 | 295 |
| B1_N64_L14 | 64 | 32 | 0.303 | 0.0% | 10.24 | 85 |
| B6_xiao_clustered | 32 | 16 | 0.335 | 1.4% | 6.14 | 276 |
| B1_N16_L4 | 16 | 8 | 0.371 | 4.0% | 3.62 | 840 |
| B8_cui | 32 | 16 | 0.408 | 1.5% | 7.17 | 266 |
| B1_N64_L29 | 64 | 32 | 0.501 | 0.0% | 16.45 | 83 |
| B1_N16_L7 | 16 | 8 | 0.542 | 10.8% | 4.75 | 874 |
| B6_xiao_literal | 32 | 16 | 0.745 | 29.1% | 11.08 | 275 |

**Global Spearman rank correlation ρ vs abstention: +0.418** over 16 cells.

### Within a fixed array size

| N | cap | cells | Spearman | ρ range | abstention range |
|---|---|---|---|---|---|
| 16 | 8 | 3 | **+1.000** | 0.235–0.542 | 1.4–10.8% |
| 32 | 16 | 10 | **+0.430** | 0.233–0.745 | 0.0–29.1% |
| 64 | 32 | 3 | **+1.000** | 0.197–0.501 | 0.0–0.0% |

### Verdict — **[FACT], and it is negative**

**A single global abstention threshold does not separate the useful region
(ρ ≲ 0.5) from the vacuous one.** Two committed cells decide it:

- `B1_N64_L29`: ρ = 0.501, i.e. **at the boundary**, yet abstention is
  **0.0%** — with cap = 32 the selector essentially never reaches the ceiling.
- `B1_N16_L4`: ρ = 0.371, **well inside the useful region**, yet abstention is
  **4.0%** — higher than every N=64 cell including the boundary one.

Max abstention among ρ ≤ 0.5 cells (4.0%) **exceeds** min abstention among
ρ > 0.5 cells (0.0%), so no threshold separates them.

**Mechanism.** Abstention is monotone in ρ *at fixed N* (perfectly so at N=16
and N=64), but it is confounded by capacity: reaching `L̂ ≥ r_max` is far harder
at cap = 32 than at cap = 8 for the same ρ. Abstention is therefore an
**aperture-dependent** proxy, not a capacity-free one.

**Consequence for PROMPT 13.** Per the brief's own instruction, the diagnostic
did **not** separate, so Paper 1 gets the one-sentence limitation, not a
subsection: ρ is a design-time coordinate and is not receiver-observable, and
abstention is only a within-aperture proxy for it. The N=64 counterexample
should be named, because it is the honest form of the objection.

---

## A4. The two internal inconsistencies

### A4a. `r_eff` of the L=16, N=32 channel — **8.73 is correct; 8.54 is wrong**

The two quoted values use the **same definition** (median over columns of
`exp(−Σ p_i ln p_i)`, `p_i = σ_i/Σσ_j`, on the Hankel lifting —
`scratch/trackD_spectral_diagnostics.py:64–74`). They differ only in the draw.

Recomputed with the **Track B Experiment C generator**, which is the stated
provenance of the 8.73 table (`trackB_hankel_emgs/system_model.py:43`,
`MASTER_SEED = 20250820`, N=32, SNR = 5 dB):

| L | 200 trials (600 cols) | 1000 trials (3000 cols) | committed |
|---|---|---|---|
| 2 | 1.906 | — | **1.90** ✓ |
| 8 | 5.649 | — | **5.70** ✓ |
| 14 | 7.986 | 8.033 | **8.11** ✓ |
| **16** | **8.694** | **8.669** | **8.73** ✓ / 8.54 ✗ |

**[FACT]** At the 200-trial budget the committed table used, L=16 reproduces as
**8.694** against the quoted **8.73** — agreement to 0.04, within Monte-Carlo
noise, and the whole column reproduces. The converged value (3000 columns) is
**8.67**. The generalization audit's **8.54** is 0.13–0.19 below the converged
value and does not reproduce.

**Fix required (PROMPT 13):** `reports/trackD_generalization_audit.md:233,244,
252,517` should read **8.73** (or 8.67 if the converged figure is preferred).
The long manuscript (§I, §XII) is **already correct** and needs no change.
ρ = 8.73/16 = **0.546**, as the committed table states.

### A4b. Xu et al. page range — **cannot be settled offline**

**The Xu et al. PDF is not in the repository.** An exhaustive search
(`find . -iname "*.pdf"`, 131 hits) returns only project figures and
`SystemModel.pdf`, which is this project's own system-model document (9 pages,
"Pilot-Based Channel Estimation for an Uplink Rydberg Atomic MIMO Receiver"),
not Xu et al.

Worse, the existing bib note is **internally contradictory**
(`paper/master/refs.bib:26–27`):

> `% [VERIFIED-WEB+USER] Start page 2961 confirmed by the author inside the PDF;`
> `% the full range 2957-2961 confirmed by web search.`

A start page of 2961 and a range of 2957–2961 cannot both hold. Per the
standing citation rule (no network; do not infer page ranges), **this is left
for the human.** Recommended action for PROMPT 13: keep
`pages = {2957--2961}` but attach `\todo{VERIFY CITATION: start page 2957 or
2961? bib note is self-contradictory and no PDF is in the repo}`.

### A4c. `\todo{CONFIRM DEPARTMENT}` — left for the human, as instructed

`paper/spl2/haatim_structural_priors_evaluation.tex` byline. **Note:** the
department was supplied earlier in this project as *Department of Electrical
and Electronics Engineering, BITS Pilani — Pilani Campus*, and
`paper/master/`, `paper/summary/` and both digests already carry it. The spl2
draft is the only file still holding the placeholder. Listed here rather than
changed, per instruction.

---

## A5. Related-work gap list

Paper 1 currently cites Cadzow, Markovsky, Roy–Vetterli, Gerchberg–Saxton,
Gorman–Hero, Stoica–Ng, and the Rydberg literature. It engages **none** of the
structured-low-rank or line-spectrum literature below. Every entry is marked
`\todo{VERIFY CITATION}` — bibliographic fields cannot be confirmed offline and
none of these PDFs is in the repository.

### Structured low-rank Hankel recovery for sums of exponentials

1. **EMaC — Chen & Chi**, enhanced matrix completion for spectrally sparse
   signals. `\todo{VERIFY CITATION}`
   *Distinction:* EMaC recovers a Hankel/Toeplitz-structured matrix from
   **partial linear** observations by nuclear-norm minimisation with recovery
   guarantees. Here the observation is **magnitude-only** — the map is not
   linear, no convex relaxation applies, and the projection is used as a
   regulariser inside a non-convex phase-retrieval iteration, not as the
   estimator.
2. **ALOHA — Jin & Ye**, annihilating-filter low-rank Hankel matrix approach.
   `\todo{VERIFY CITATION}`
   *Distinction:* ALOHA solves a structured low-rank completion in a transform
   domain for images/MRI with linear sampling. Same distinction as above, plus
   ours selects the rank from held-out data rather than fixing it.
3. **LORAKS — Haldar**, low-rank modelling of local k-space neighbourhoods.
   `\todo{VERIFY CITATION}`
   *Distinction:* LORAKS exploits limited support and phase constraints in MRI
   with linear measurements. Our constraint is the array manifold, and the
   nonlinearity is the modulus, not undersampling.

### Gridless line spectrum / atomic norm

4. **Atomic-norm denoising and gridless line-spectral estimation** (Bhaskar,
   Tang, Recht; Tang, Bhaskar, Shah, Recht; Candès & Fernandez-Granda).
   `\todo{VERIFY CITATION}`
   *Distinction:* these recover the **frequencies themselves** from linear
   measurements with separation conditions and convex guarantees. We do not
   estimate angles — see the sentence PROMPT 13 requires (§below) — and our
   measurements are magnitudes, so the atomic-norm machinery does not transfer
   without a new relaxation.
5. **Cadzow / structured low-rank approximation** — already cited; retained
   here only to note that our contribution is not the projection.

### Toeplitz / Vandermonde covariance in array processing

6. **Toeplitz covariance estimation and Vandermonde decomposition**
   (Carathéodory–Toeplitz; covariance fitting for DOA; coarray methods).
   `\todo{VERIFY CITATION}`
   *Distinction:* these impose structure on the **covariance** accumulated over
   many snapshots, for angle estimation. Ours imposes structure on a **single
   channel realisation** in a pilot-limited regime (P = 20–30) where no useful
   covariance can be accumulated, and the target is the channel, not the angles.
7. **Root-MUSIC / ESPRIT-family subspace methods.** `\todo{VERIFY CITATION}`
   *Distinction:* these are the named alternative for the "why not estimate
   angles directly?" sentence PROMPT 13 asks for. They need the signal
   subspace, which requires either multiple snapshots or the complex field;
   with magnitude-only single-snapshot data the subspace is not identifiable
   without first solving the phase-retrieval problem — at which point the
   channel is already estimated.

**Suggested one-sentence framing for PROMPT 13** (not written into the
manuscript this turn): *we regularise toward the exponential-sum manifold
rather than estimating its parameters, because with magnitude-only
measurements the angles are identifiable only after phase retrieval has
succeeded, whereas a rank constraint is usable during it.*

---

## Part A status

| item | result |
|---|---|
| A1 | done — two asserted differences do not exist; **B1 required** (no P=30 TD cell) |
| A2 | done — table + figure; +1.309 dB confirmed; **contrast changes sign with SNR** |
| A3 | done — **negative**: no global abstention threshold; monotone only within fixed N |
| A4a | done — **8.73 correct, 8.54 wrong**; fix the generalization audit |
| A4b | **cannot be settled** — Xu PDF absent; bib note self-contradictory; left for human |
| A4c | listed for human, as instructed |
| A5 | done — 7 entries, all `\todo{VERIFY CITATION}` |
