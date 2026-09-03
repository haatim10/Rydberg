# PROMPT 11 — Part A inventory and Part B decisions

**Written and committed BEFORE any merged text was drafted**, so the record of
what was decided precedes what was written.

No new experiments, no retraining, no new sweeps were run for this merge.

---

## 0. A labelling correction that governs the whole task

**The brief's `spl1`/`spl2` are swapped relative to the repository.** Acting on
the labels rather than the content would apply every fix to the wrong paper.

| directory | actual title | brief calls it |
|---|---|---|
| `paper/spl1/` | *Effective-Rank Scaling of Hankel-Structured Channel Estimation…* | "spl2" |
| `paper/spl2/` | *On the Evaluation of Structural Priors in Unrolled Channel Estimation…* | "spl1" |

So: **Part C merges `paper/hsgs.tex` with `paper/spl1/`** (the effective-rank
paper, which carries the out-of-model predictions the brief describes as the
headline). **Part E's three fixes apply to `paper/spl2/`** (the evaluation
paper, whose Sec. II carries the Saleh–Valenzuela error). Everything below
follows the content.

---

## Part A — inventory, with branch and commit

Everything was found on the **current branch**
(`claude/new-paper-implementation-fs4ako`); nothing had to be recovered from
another branch. The manuscript's own history predates the Track D work.

### 1. The HS-GS manuscript

| artifact | path | first added |
|---|---|---|
| source | `paper/hsgs.tex` (776 lines) | `b44bf2a` |
| compiled | `paper/hsgs.pdf` (6 pp.) | `b44bf2a` |
| figures | `paper/fig/fig1…fig6` (`.pdf`+`.png`) | `b44bf2a`, `7c0cf14` |

Subsequent commits: `7c0cf14` (absolute-NMSE figures, four algorithm floats),
`19d0a5f` (re-sourced Fig. 3, fitted to six pages), `97721d9` (legends),
`dda6074` (**geometric CCRB Jacobian fix, θ/ψ; Rician FIM verified**),
`54c88d5` (CCRB regeneration finalised). Its bibliography is inline
(`\begin{thebibliography}`), not a `.bib` file.

### 2. **CRLB / CCRB derivation code — FOUND. The gate passes.**

This was the stop condition, so it is reported first and in full:

| file | size | role |
|---|---|---|
| `rydberg_sim/crlb.py` | 15,985 B | Rician FIM, rank-one construction |
| `scripts/constrained_crlb.py` | 12,305 B | Gorman–Hero / Stoica–Ng tangent-space CCRB |
| `scripts/constrained_crlb_fast.py` | 9,265 B | fast path |
| `scripts/compare_ccrb.py` | 4,309 B | comparison driver |

Committed results: `results/track_b/crlb.json`,
`results/track_b/constrained_crlb.json` (400 trials), plus figures
`results/track_b/final/b1_*`, `b3_with_constrained_crlb_P{10,30}`,
`b4_*`, `b6_*`, and `results/track_b/artifact/C3_gap_to_constrained_crlb.*`.
Introduced at `8b7e885` ("Constrained CRLB, and a correction to the
unconstrained one").

The rank-one claim is recorded in the results file itself, verbatim:

> "unconstrained_rank1: each magnitude measurement contributes ONE real
> constraint (rank-1 Fisher in real coordinates). constrained: Gorman-Hero /
> Stoica-Ng bound on the 3*sum(L_k)-parameter geometric manifold. **Neither
> bound governs a biased estimator.**"

`jacobian_rank` ≈ 44.53–44.61 against ≈44.7 parameters, confirming the
rank-deficiency that forces the tangent-space form.

### 3. Track B experiment stores and driver

| artifact | path |
|---|---|
| path-count sweep | `trackB_hankel_emgs/results/experiment_C_path_count.csv` |
| array-size sweep | `trackB_hankel_emgs/results/experiment_B_array_size.csv` |
| SNR sweep | `trackB_hankel_emgs/results/experiment_A_snr.csv` |
| ablations | `trackB_hankel_emgs/results/ablations.json` |
| N×SNR grid | `trackB_hankel_emgs/results/figure_array_size_comparison.csv` |
| driver | `trackB_hankel_emgs/{runner,experiment_*,hankel_em_gs,config}.py` |
| config | `trackB_hankel_emgs/config.py` |

### 4. PROMPT 10 output

`paper/spl1/` and `paper/spl2/` (`main.tex`, `refs.bib`, `fig/`, `main.pdf`),
at `e869128`, `8d81214`, `859cd4b`, `32a8d76`. Track D results under
`reports/trackD_*.json|md` and `results/track_d/`.

---

## Part B — the seven conflicts, decided

### 1. Capacity formula — **`⌈N/2⌉`. The old draft is right; `paper/spl1/` is wrong.**

Verified by brute force against the code
(`rydberg_sim/track_b_proposed.py:105`, which computes
`max(min(N-p, p+1))` directly):

| `N` | brute force | `⌈N/2⌉` | `⌊N/2⌋` |
|---|---|---|---|
| 7 | 4 | **4** | 3 ✗ |
| 15 | 8 | **8** | 7 ✗ |
| 31 | 16 | **16** | 15 ✗ |
| 8, 16, 32, 64 | 4, 8, 16, 32 | ✓ | ✓ (coincide) |

**No measured number changes**, because every experiment in either draft used
even `N` (8, 16, 32, 64), where the two agree. The error is confined to the
stated formula and to any odd-`N` statement. Downstream check: `paper/spl1/`
uses `cap` in the derivation, in `r_eff/cap`, and in the `cap(8)=4` sentence
about Xu et al.'s array — the last is 4 under both conventions, so it stands.

### 2. RSR — **state the single-user convention once, and give the conversion.**

Both drafts define RSR against a *single* user
(`trackB_hankel_emgs/config.py:18`). The merged paper states that once and
converts explicitly. Old configuration 12 dB; Track D 10 dB.

**A discrepancy I could not resolve.** The old draft states a nominal 12 dB is
"7.15 dB against the K-user total" (`paper/hsgs.tex:149`). The arithmetic
for `K = 3` gives `12 − 10log₁₀3 = 7.229 dB`, not 7.15. I cannot reconcile the
0.08 dB from any committed result, so the merged paper reports **7.23 dB** as
the nominal conversion and carries a `\todo` recording that the earlier draft
said 7.15. Under the Track D convention, 10 dB single-user is 5.23 dB.

### 3. Cadzow sweeps — **4 (old) and 1 (new) are different operators; stated per experiment.**

`trackB_hankel_emgs/config.py:24` sets `CADZOW_ITER = 4`; Track D applies one
step. One step of `H⁻¹∘Π_r∘H` is **not** idempotent — a property this project
measured directly (a single step leaves ≈3.7% off-manifold), so these are not
the same operator with a different budget. Every figure names its own setting;
no figure mixes them.

### 4–5. SNR range (−5…20 vs −10…20) and iterations (`T` = 50 vs 100)

Not harmonised, and nothing re-run. **The old configuration is canonical for
the bound comparisons** (it is what the CRLB/CCRB were computed against, over
3.6×10⁴ trials); effective-rank and out-of-model results are reported at their
own configuration. Every figure caption names its configuration, and each
figure is internally single-configuration.

### 6. Name — **HS-GS.**

Chosen to match the artifact: the code says `hs_gs` 158 times and `HS-GS` 149,
against `hankel_em_gs` 39 and `HS-EM-GS` 45. The paper notes once that the
exact step is the EM-GS update either way, so the two names denote the same
estimator.

### 7. Primary statistic — **both, under an explicit rule.**

- **Ratio-of-sums** (`paper/hsgs.tex:483`) wherever a curve is compared to a
  bound, because that is the ensemble quantity a CRLB bounds and a
  mean-of-decibels is smaller by Jensen.
- **Paired per-trial median with bootstrap CI** for paired contrasts.

The old draft's "two averaging domains, kept distinct" paragraph carries this
argument already and is reused rather than rewritten.

### The replication, stated as an asset

Two independent implementations, at two different configurations, give
**+2.85 dB** (Track B: RSR 12 dB, `T`=50, 4 Cadzow sweeps, ratio-of-sums over
12 operating points) and **+2.680 dB** (Track D: RSR 10 dB, `T`=100, 1 sweep,
adaptive rank, paired median over a uniform SNR draw) for the same contrast at
`N = 32`. Different code, different settings, different pooling, 0.17 dB apart.
The merged paper says so in one sentence.

---

## Part D — page budget, decided in advance

Target 5 pages; the old draft alone is 6. **The CCRB is protected
unconditionally.** If 5 pages cannot be reached with it intact, the merged
paper is delivered at 6 and that is reported, per the brief.
