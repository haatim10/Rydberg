# Generalization audit — how specific is the rank-7 Hankel result?

PROMPT 8. Every claim tagged **[FACT]** (verified from the repository, with
`file:line`), **[MATH]** (derived from a [FACT], derivation shown), or **[HYP]**
(needs an experiment, which is named).

No training was run. No estimator or config was modified. Part C was executed;
Part D is costed and **not** launched.

---

## Part A — provenance

### A1. Where `L_k ∈ [3,7]` came from

**[FACT]** `rydberg_sim/track_b_drivers.py:47-48`:

```python
TRACK_B_L_MIN = 3
TRACK_B_L_MAX = 7
```

drawn i.i.d. per user per realization (`:95-107`, `rng.integers(L_min, L_max+1, size=K)`).

**[FACT]** Introduced by commit `683fd6f` (2026-08-20), "Track B baseline: audit
against frozen model, L_k ~ U{3..7}, B1/B2 for GS/EM-GS". The commit body states:

> "The frozen document lists `L_k` only as 'user-dependent' and gives **NO
> distribution**. `L_k ~ U{3,…,7}` comes from the ULA implementation plan
> (Part 5, Fig. B1), not from the system model."

**[FACT]** I verified this against the PDF rather than trusting the message.
`SystemModel.pdf` (extracted, 9 pp.) lists in its symbol table: "`L_k` — number
of resolvable paths of user k — **user-dependent**". A regex sweep for any
numeric bound on `L_k` (`L_k ≤ …`, `L_k ∈ …`) returns **nothing**. The document
constrains the *distribution of path gains* (`α_{ℓ,k} ~ CN(0, β_k/L_k)`) and of
*angles* (`θ ~ U[−π/2, π/2]`), but never the path count.

**[FACT]** The Cui, Gong, Xu and Xiao PDFs are **not in the repository** (only
`SystemModel.pdf` and the project's own `paper/hsgs.pdf`). Nor is the "ULA
implementation plan". So I cannot check those four sources this turn. **That is
an open gap, not a clearance** — I am not entitled to say "no source supports
3–7"; I can only say no source *in the repository* does, and the one document
that does specify the model is silent.

**Verdict, plainly: `L_k ∈ [3,7]` is an undocumented project design choice.** It
traces to a planning document not under version control, with no cited physical
justification anywhere I can reach. I am not constructing a post-hoc rationale.

### A2. What `r = 7` is

**[FACT]** A **simulation design parameter**. `trackD_urformer/hankel.py:66`:

> "`r = 7` fixed (`L_max`)  PRIMARY. A system design assumption, the same one the
> channel generator uses. NOT oracle information."

It is not an **oracle** — it uses no per-realization knowledge, and the code has
a separate `mode="oracle"` path for that, flagged "DIAGNOSTIC UPPER BOUND ONLY"
(`hankel.py:70-71`). It is not **physical** — it equals `L_max`, and A1 shows
`L_max` has no physical basis.

**A paper draft cannot present `r = 7` as anything stronger than a matched
sparse-path benchmark.** It is `L_max` because we chose `L_max`.

**[FACT] — and this materially narrows the problem:** `r = 7` is a **Track D
artifact**, not the classical method's design. `paper/hsgs.pdf` states "the
model order comes from a held-out pilot residual, **no oracle path count is
used**", and scopes its result to "the tested sparse geometric multipath model
with i.i.d. uniform angles of arrival and equal per-path power", adding "the
benefit is conditional rather than general". The classical line already uses
adaptive rank (`hs_gs_auto`, `track_b_proposed.py:244`) and already scopes
itself. Fixed `r = 7` was introduced for the URformer comparison
(`stage3.py`, `HANKEL_RANK = 7`).

### A3. The exact embedding

**[FACT]** `rydberg_sim/track_b_structure.py:153-161`:

```python
def hankel_matrix(g, pencil=None):
    g = np.asarray(g, dtype=np.complex128).ravel()
    N = g.size
    p = int(pencil) if pencil is not None else N // 2
    if not (1 <= p <= N - 1):
        raise ValueError(...)
    rows = N - p
    return np.lib.stride_tricks.sliding_window_view(g, p + 1)[:rows]
```

Convention: **`rows = N − p`, `cols = p + 1`**; default pencil `p = N // 2`; at
`N = 32` that is a **16 × 17** matrix. Inverse is anti-diagonal averaging
(`:164-175`).

**[FACT]** One inconsistency worth recording: `cadzow_project`
(`track_b_proposed.py:114-135`) defaults to `best_pencil(N)`, not `N//2`. For
even `N` these coincide (both maximize the cap), so nothing measured is
affected; for odd `N` they could differ.

**[FACT]** `cadzow_project:126-127` contains the endpoint guard:
`if r >= cap: return g` — "constraint inactive; do not perturb".

---

## Part B — what is guaranteed and what is not

### B1. Maximum representable rank

**[MATH]** `H(g)` is `(N−p) × (p+1)`, so `rank ≤ min(N−p, p+1)`. Maximizing over
the pencil:

```
cap(M) = max_p min(M − p, p + 1) = floor(M / 2)
```

**[FACT]** Verified numerically for M ∈ {8, 16, 32, 64, 128} — `cap == floor(M/2)`
in every case, matching `hankel_rank_cap` (`track_b_proposed.py:101`).

| M | best pencil | rows × cols | cap | floor(M/2) |
|---|---|---|---|---|
| 8 | 3 | 5 × 4 | 4 | 4 |
| 16 | 7 | 9 × 8 | 8 | 8 |
| 32 | 15 | 17 × 16 | **16** | 16 |
| 64 | 31 | 33 × 32 | 32 | 32 |

Normalized condition: **`L_eff ≪ floor(M/2)`**.

**This reprices the setup as the prompt anticipated.** At `M = 32`, `L_k = 7`
sits at **7/16 = 44%** of the ceiling. `L_k ≥ 13` approaches vacuity, and
`r = 16` at `M = 32` is exactly the identity — the same failure mode as the N=8
vacuity found in stage 3 (`hankel.py` HK7), on a different axis.

**Two thresholds, and they differ:**

- **Algebraic [MATH]:** Hankel rank is exactly `L` iff `min(rows, cols) ≥ L`,
  i.e. `L ≤ floor(M/2)`.
- **Statistical [HYP]:** subspace separation needs a noise subspace, so useful
  estimation wants `L` comfortably below the cap. **My rule of thumb:
  `L/cap ≤ 0.5` for a gain above 1 dB.** This is not derived — but it is not
  invented either: it is read off Track B's already-completed Experiment C
  (below), which measures the decay directly at `M = 32`. Settled generally
  only by repeating that sweep at another `M` (D3 axis 3).

### B2. Assumption inventory

**Exact vs approximate low rank — [FACT], and the answer is "approximate".**
`cadzow_project` truncates to rank `r` and averages back; nothing requires the
input to be exactly rank-`r`. Part C below shows the operator is useful on
channels that are *nowhere near* exactly low rank. **The method requires
spectral compressibility, not exact sparsity.**

**Relationship between `L_k`, `r`, `M` — [MATH]:** the constraint bites only when
`r < cap(M) = floor(M/2)`; it is informative only when the channel's energy
concentrates in fewer than `r` components.

**`L_k > r` (under-modeling) — [FACT], degradation is graceful, not cliff-edged.**
Track B Experiment C (`results/experiment_C_path_count.csv`, 300 trials/cell,
M=32, SNR=5, adaptive rank) shows monotone decay with no cliff:

| L | L/cap | gain (dB) | CI95 | win rate | mean L̂ | L̂/L |
|---|---|---|---|---|---|---|
| 2 | 0.125 | **+7.043** | [+6.73, +7.34] | 1.00 | 2.13 | 1.07 |
| 4 | 0.250 | +3.556 | [+3.38, +3.73] | 0.99 | 4.02 | 1.00 |
| 6 | 0.375 | +1.792 | [+1.64, +1.96] | 0.94 | 5.67 | 0.94 |
| 8 | 0.500 | +1.038 | [+0.93, +1.15] | 0.90 | 7.30 | 0.91 |
| 10 | 0.625 | +0.577 | [+0.45, +0.69] | 0.79 | 8.44 | 0.84 |
| 12 | 0.750 | +0.266 | [+0.19, +0.34] | 0.73 | 9.81 | 0.82 |
| 14 | 0.875 | +0.046 | [−0.05, +0.14] | 0.60 | 10.55 | 0.75 |
| 16 | 1.000 | **−0.117** | [−0.21, −0.04] | 0.45 | 11.63 | 0.73 |

**[FACT] The gain decays monotonically in `L/cap` and crosses zero at ≈0.90.**
The `floor(M/2)` vacuity prediction is not merely derivable — it is already
measured, and the experiment carried a pre-registered prediction
(`experiment_path_count.py:12-18`) that it confirmed.

**`L_k → floor(M/2)` — [FACT]:** gain reaches `−0.117 dB` at `L = cap`. Slightly
*negative*, not zero, because rank selection under-estimates there (L̂/L = 0.73).

**`r > L_k` (over-modeling) — [MATH]:** degrades **toward** EM-GS, never below
it, because `cadzow_project` returns `g` unchanged once `r ≥ cap`. Between
`L_k` and `cap` it is a soft version of the same thing. So over-modeling is
safe; under-modeling is what costs.

**Does the method need to know `L_k`? — [FACT]: neither, as implemented.**
`select_order_heldout` (`track_b_proposed.py:212-243`) picks `L̂` by held-out
pilot residual, and its docstring is explicit: "Uses no ground truth… every
quantity involved (`S`, `Z`, `B`) is observable." It also documents why the
in-sample residual cannot be used. **But `hs_gs` itself takes `L_hat` as a
required argument, and Track D's H0 passes the constant 7** (`stage3.py`).
So: *the classical method needs neither; the Track D arm was given `L_max`.*

**[FACT]** `results/ablations.json` (M=32, P=30, SNR=5, L=4, 120 trials) already
prices that choice:

| variant | NMSE (dB) | gain (dB) |
|---|---|---|
| EM-GS baseline | −10.646 | 0 |
| Hankel, interleaved, **adaptive** L̂ (mean 3.95) | −14.355 | **+3.709** |
| Hankel, post-hoc | −14.067 | +3.421 |
| Hankel, **ORACLE** rank L̂ = L | −14.723 | **+4.077** |
| Hankel, 1 Cadzow sweep | −14.647 | +4.001 |
| Hankel, 8 Cadzow sweeps | −14.319 | +3.673 |

**Oracle-free selection recovers 3.709 / 4.077 = 91% of the oracle-rank gain.**
And 1 Cadzow sweep beats both 4 and 8 — independently reproducing the PROMPT 6
`n_iter` finding on the classical side.

---

## Part C — spectral diagnostics (run)

`reports/trackD_spectral_diagnostics.json`,
`results/track_d/spectral/fig_hankel_spectrum.png`. Metrics per column on
`H(g)`, pencil `p = N//2`: cumulative energy `E(q)`; effective rank
`exp(−Σ p_i ln p_i)` with `p_i = σ_i/Σσ_j` (Roy–Vetterli, column `erank`) and
the energy form `p_i = σ_i²/Σσ_j²` (`erank_E`); stable rank `Σσ_i²/σ_1²`; tail
energy `1 − E(r)`.

*Note on scope:* the prompt's permitted list calls Part C estimator-free, but C1
explicitly asks for the spectrum "on the noisy EM-GS estimate". I ran EM-GS as a
**data source** for the spectrum. Nothing was trained or tuned.

### C1 — `L` sweep at `M = 32` (cap 16)

| configuration | erank | erank_E | srank | tail@3 % | tail@7 % | tail@13 % |
|---|---|---|---|---|---|---|
| true, L=1 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| true, L=2 | 1.91 | 1.70 | 1.29 | 0.00 | 0.00 | 0.00 |
| true, L=3 | 2.68 | 2.16 | 1.55 | 0.00 | 0.00 | 0.00 |
| true, L=5 | 3.91 | 3.06 | 1.92 | 5.57 | 0.00 | 0.00 |
| true, L=7 | 5.11 | 3.77 | 2.16 | 12.42 | 0.00 | 0.00 |
| true, L=10 | 6.52 | 4.58 | 2.45 | 19.44 | 0.19 | 0.00 |
| true, L=13 | 7.65 | 5.26 | 2.61 | 24.22 | 1.55 | 0.00 |
| true, L=16 | **8.54** | 5.81 | 2.75 | 27.38 | **3.13** | 0.00 |
| est. 5 dB, L=1 | **8.91** | 2.13 | 1.17 | 9.31 | 3.57 | 0.28 |
| est. 5 dB, L=3 | 9.60 | 3.72 | 1.74 | 10.86 | 3.63 | 0.23 |
| est. 5 dB, L=5 | 10.13 | 4.62 | 2.23 | 16.52 | 4.01 | 0.29 |
| est. 5 dB, L=7 | 10.65 | 5.49 | 2.45 | 21.91 | 4.64 | 0.33 |
| est. 5 dB, L=16 | **11.77** | 7.13 | 3.06 | 33.04 | 7.77 | 0.45 |

**Two findings here, both consequential.**

**[FACT] The true channel's effective rank is far below `L`.** At `L = 16` — the
algebraic ceiling, where the Hankel matrix is numerically *full rank* — the
effective rank is only **8.54**, and rank-7 truncation costs just **3.13%** of
the energy. **[MATH]** The cause is the generator's own gain model: `α ~ CN(0,
β/L)` i.i.d. means the `L` components have random magnitudes, so the spectrum
decays even when the rank is full. **Exact sparsity was never what the operator
exploited; spectral decay was.**

**[FACT] The estimator does not see `L`.** At 5 dB the estimate's effective rank
runs **8.91 → 11.77** across `L = 1 … 16` — a span of 2.9 — while the true
effective rank runs 1.00 → 8.54, a span of 7.5. **The noise floor, not the path
count, sets what the estimator sees.** This is the direct measurement behind
D2's predicted failure mode for threshold and gap rules.

### C2 — `M` sweep at `L = 5` (noiseless)

| configuration | erank | srank | tail@7 % | L_eff/cap |
|---|---|---|---|---|
| M=16 (cap 8) | 3.47 | 1.71 | 0.00 | 0.43 |
| M=32 (cap 16) | 3.91 | 1.92 | 0.00 | 0.24 |
| M=64 (cap 32) | 4.25 | 2.08 | 0.00 | 0.13 |

**[FACT] Effective rank grows sub-linearly in `M` while the cap doubles**, so
`L_eff/cap` **falls** with aperture: 0.43 → 0.24 → 0.13. **[MATH]** The prior
should therefore get *more* useful as `M` grows. **[FACT]** That is what Track B
measured independently — `paper/hsgs.pdf` reports the HS-GS advantage widening
"0.03 → 0.81 → 2.45 dB at N = 8/16/32". Two independent routes to the same
scaling; this is E1 requirement #4 already partly satisfied.

### C3 — SNR sweep on the estimate (`L = 5`, `M = 32`)

| SNR (dB) | erank | srank | tail@3 % | tail@7 % | tail@13 % |
|---|---|---|---|---|---|
| −10 | **13.77** | 4.93 | 51.72 | **19.43** | 1.45 |
| −5 | 13.44 | 4.21 | 47.24 | 17.38 | 1.19 |
| 0 | 12.29 | 2.77 | 31.74 | 10.26 | 0.71 |
| +5 | 10.13 | 2.23 | 16.52 | 4.01 | 0.29 |
| +10 | 8.17 | 2.02 | 9.94 | 1.48 | 0.10 |
| +15 | 6.73 | 1.94 | 6.67 | 0.47 | 0.03 |
| +20 | **5.73** | 1.92 | 5.97 | **0.15** | 0.01 |

**[FACT] SNR moves the estimate's effective rank far more than `L` does.** At
fixed `L = 5`, erank spans 13.77 → 5.73 across SNR (Δ = 8.0). At fixed SNR = 5,
erank spans 8.91 → 11.77 across `L = 1…16` (Δ = 2.9). **SNR is the first-order
axis of the spectrum the estimator actually sees; `L` is second-order.**

**This has a direct design consequence for D1:** an `(L, r)` grid run at a single
SNR would be sweeping the *weaker* of the two axes. The grid must carry SNR.

### C4 — Xiao's own channel, `M = 32`, 4 clusters × 10 rays

Xiao Table I gives `θ_{l,c} ~ U(−π/2, π/2)` subscripted per *ray*, which reads as
40 independent DoAs with no clustering. But a Saleh–Valenzuela channel normally
concentrates rays about a cluster centre, and Table I gives no intra-cluster
spread. **This is the same ambiguity the repository already resolved for Cui's
Table I** ("incident angles Uniform(−90,90)", "max cluster AS Uniform(−5,5)"),
handled in `channel_cui.py:62-67` as uniform centres with rays offset within
±5°. Both readings are reported; neither is chosen for me.

| reading | erank | erank_E | srank | tail@3 % | tail@7 % | tail@13 % | smallest r with tail < 1% |
|---|---|---|---|---|---|---|---|
| **clustered** (±5°, Cui precedent) | **5.36** | 3.54 | 2.03 | 10.10 | **0.01** | 0.00 | **5** |
| **literal** (40 indep. DoAs) | 12.00 | 7.96 | 3.36 | 38.58 | **10.64** | 0.30 | 12 |
| *reference:* sparse L=7 | 5.11 | 3.77 | 2.16 | 12.42 | 0.00 | 0.00 | 6 |

**[FACT] Under the clustered reading, Xiao's channel is as compressible as our
sparse `L = 7` channel** — effective rank 5.36 vs 5.11, and `r = 7` costs
**0.01%** of the energy. The rank-7 prior transfers to the paper's own channel
model **with no retuning**.

**[FACT] Under the literal reading it is still compressible, just at higher `r`**
— effective rank 12.00, and `r = 13` captures 99.7%.

**Correction to something I said earlier this session.** I previously wrote that
under Table I as printed the channel is "full Hankel rank at N=32, so the HS-*
arms would have no low-rank structure to exploit". The first half is right about
*numerical* rank; **the second half was too strong.** The effective rank is
12.00, not 16, and `r = 13` leaves 0.30% tail. Even the pessimistic reading is
spectrally compressible — the prior would need a larger `r`, not abandonment.
That changes the conclusion, so it is worth stating plainly rather than leaving
the looser phrasing to stand.

### Which metric best supports a generalization claim

**Tail energy at `r`, `1 − E(r)`.** Effective rank is a good *descriptor* but a
poor *criterion*: it is a single summary of the whole spectrum and, per C1, it
is badly inflated by noise on the estimate (8.91 at `L = 1`). Stable rank is
worse still — it barely moves (1.00 → 2.75 across the entire `L` range) because
`σ_1²` dominates.

Tail energy is the right metric because **it is the quantity the operator
actually destroys**. It is directly interpretable ("rank-7 truncation discards
0.01% of this channel"), it is defined per candidate `r` so it maps onto the
design choice, and it is what makes C4's claim concrete and falsifiable.

---

## Part D — proposed experiments (NOT run)

### D1. The classical (L, r) study — substantially cheaper than drafted

**The L-axis is already done.** Track B Experiment C swept `L ∈ {2,…,16}` at
`M=32`, 300 trials/cell, with adaptive rank and a pre-registered prediction. It
should be *cited*, not repeated.

**What is genuinely missing** is the other two axes:
1. **fixed-`r` mismatch** — Track B used adaptive `L̂` throughout, so no cell
   measures `r < L` or `r > L` at a *held* `r`;
2. **SNR** — Track B fixed SNR = 5 dB, and C3 shows SNR is the stronger axis.

**Corrected grid, in normalized terms.** Cells with `L/cap ≥ 0.9` are known
vacuous (Track B measures −0.117 dB at 1.00), and `r/cap = 1` is the identity by
construction. So:

- `L/cap ∈ {0.19, 0.31, 0.50}` → at M=32: **L ∈ {3, 5, 8}**
- `r/cap ∈ {0.19, 0.31, 0.44, 0.63}` → at M=32: **r ∈ {3, 5, 7, 10}**

This 3 × 4 spans `r < L`, `r = L`, `r > L` without spending a cell on vacuity.
Add the two free endpoint checks: `r = 16` must reproduce EM-GS exactly (guard
at `cadzow_project:126`), and adaptive `L̂` as the reference column.

**Cost, honestly.** Measured HS-EM-GS throughput this session: **742 ms/trial**
at M=32, T=100 (PROMPT 6 timing probe). Per the prompt: 1000 trials/cell, SNR
drawn per trial across `[−10, 20]` and **binned post hoc**, so each cell yields a
whole `Δ_HS(SNR)` curve rather than one scalar.

| stage | cells | trials | core-hours | wall (4 cores) |
|---|---|---|---|---|
| **3 × 3 pilot** (L ∈ {3,5,8}, r ∈ {3,5,10}) | 9 | 9,000 | 1.9 | **~30 min** |
| full 3 × 4 refinement | 12 | 12,000 | 2.5 | ~40 min |
| adaptive-`L̂` reference column | 3 | 3,000 | 1.5* | ~25 min |

*the adaptive column costs ~2.4× per trial because `select_order_heldout` reruns
the estimator once per candidate rank.

**Total ≈ 1.5 h wall**, against the 17 h the draft grid implied. The saving comes
from dropping the vacuous cells, citing Track B for the L-axis, and binning SNR
post hoc instead of running a grid per SNR.

Paired SE at 1000 trials ≈ 0.01–0.03 dB against effects of order 1 dB — ample.

**Reporting: `Δ_HS` per SNR bin. A pooled scalar is never a headline and never a
decision rule** (standing rule from stage 3).

### D2. Adaptive rank — analysis

| | physical reading | robustness to unknown `L` | noise sensitivity | cost | preserves EM-GS structure | publishably stronger than `r=7`? |
|---|---|---|---|---|---|---|
| **A. fixed `r = L_max`** | none — a design constant | none: fails when `L > r` | insensitive (no decision) | 1× | yes | **no** — A2 says it can only be a matched benchmark |
| **B. energy threshold `η`** | "keep 1−η of the energy" | good if `η` well set | **poor** — reads a noise-inflated spectrum | ~1× | yes | yes, if `η` transfers across SNR |
| **C. spectral gap** | "signal/noise subspace split" | good at high SNR | **worst** — needs a gap that C3 shows closes at low SNR | ~1× | yes | yes, but fragile |
| **D. soft shrinkage `max(σ−λ,0)`** | "shrink toward structure" | **best** — no hard decision to get wrong | mild, degrades smoothly | 1× | yes | **most likely** |
| *(existing)* **held-out residual** | "pick the order that predicts unseen pilots" | good — measured L̂/L = 0.73–1.07 | moderate | **~2.4×** | yes | **already 91% of oracle [FACT]** |

**Named failure modes.**

- **B and C read a noise-inflated spectrum**, and C1 quantifies exactly how bad
  that is: at 5 dB the estimate's effective rank is **8.91 even when `L = 1`**.
  Both rules will over-estimate rank at low SNR — the regime stages 3–4 found
  most fragile. **[FACT]**, not speculation: this is measured in C1.
- **C additionally needs a gap to exist.** C3 shows the estimate's spectrum
  flattening as SNR drops (erank 13.77 at −10 dB out of a cap of 16). At low SNR
  there is no gap to find.
- **A fails silently** when `L > r`: Track B's L=16 cell shows it going negative.
- **D has no hard rank decision at all**, which is its main advantage — the
  failure mode is a mis-set `λ`, which degrades smoothly rather than switching
  to a wrong integer.

**Is D compatible with the existing operator? [MATH] — yes, and it is a two-line
change.** `cadzow_project` already forms `U, s, Vh` and reconstructs
`(U[:, :r] * s[:r]) @ Vh[:r]`. Soft shrinkage replaces the hard index cut with
`s_shrunk = np.maximum(s - lam, 0.0)` and reconstructs from all components. It
needs **no rework**, keeps the alternating-projection structure, and keeps the
`r ≥ cap` no-op guard's spirit (`λ = 0` is the identity). **[HYP]** that it beats
fixed `r = 7`: settled by adding one column to the D1 grid, ~25 min.

**Recommendation:** carry **D (soft shrinkage)** as the candidate improvement and
**held-out residual** as the already-validated oracle-free reference. Drop C
(gap) — C3 shows its precondition fails where it is needed. Keep B only as a
cheap ablation.

### D3. Generalization axes, ranked by (reviewer value)/(cost)

| rank | axis | value / cost | one-line justification |
|---|---|---|---|
| 1 | **`(L, r)` mismatch** | high / **1.5 h** | The whole claim is "works under unknown order"; nothing yet measures held-`r` mismatch. |
| 2 | **approximate vs exact low rank** | high / **already done** | C4 delivers it at zero cost on a second, independently specified channel. |
| 3 | **`M` (array size)** | high / ~2 h | Theory predicts `L_eff/cap` falls with `M` (C2) and Track B already measured the trend; one more `M` confirms the scaling law. |
| 4 | **SNR distribution** | high / **free** | Already carried by binning D1 post hoc. Costs nothing extra and C3 says it is the strong axis. |
| 5 | angle-distribution mismatch | medium / ~1 h | C4's clustered-vs-literal contrast is already a version of this. |
| 6 | pilot count `P` | medium / ~1 h | Already swept for the learned arms; classical sweep is cheap. |
| 7 | train/test path-count mismatch | medium / expensive | Only meaningful for learned arms; needs retraining (see D4). |
| 8 | `K` (users) | low / ~2 h | The operator is per-column; `K` enters only through the LS step. Nuisance robustness. |
| 9 | gain distribution | low / ~1 h | Equal-power is the scoped assumption; worth one ablation, not a study. |
| 10 | bias / RSR level | low / ~1 h | Affects the phase-retrieval difficulty, not the Hankel structure. |

**Smallest set that would satisfy a skeptical reviewer:** axes **1 + 2 + 3**,
with 4 free. I agree with your prior, with one amendment: **axis 2 is already
delivered by C4**, so the remaining spend is axis 1 (~1.5 h) and axis 3 (~2 h).
Everything below rank 5 is nuisance robustness and belongs in an appendix table
if anywhere.

### D4. Learned-model follow-up — stage 4 is binding

**Distinguish three claims that are routinely conflated:**

- **fixed-model OOD evaluation** — cheap (minutes), tests whether *this trained
  model* survives a shift. Supports a **robustness** claim only.
- **retraining under each condition** — expensive (~2.2 h/arm), required for any
  **method** claim ("HS helps under X").
- **matched-condition performance** — what the method achieves when trained *for*
  the condition. Different from robustness and not interchangeable with it.

**Stage 4's finding governs everything here [FACT]:** a +1.209 dB structural
gain at high SNR collapsed **15× to +0.078 dB** once U1 was trained on the
operating range. The gain was a **training-adequacy proxy**, not an information
advantage. So **any learned "HS helps under condition X" claim must control for
training adequacy under X, or it is uninterpretable.**

**Recommendation — and the honest answer is uncomfortable:**

- **OOD evaluation suffices** for: pilot count `P`, `K`, RSR, angle
  distribution. These are robustness statements and should be labelled as such.
- **Retraining is unavoidable** for any claim about `(L, r)` mismatch or `M` in
  the learned arms — because those change what the network *can* learn, which is
  exactly the confound stage 4 exposed.
- **Stated plainly: the learned Hankel line does not currently support a method
  claim.** Stage 3 gave STOP-MARGINAL; stage 4 showed the surviving high-SNR gain
  was a training artifact; PROMPT 7 G1/G2 failed the low-SNR half of the success
  criterion with CIs excluding the threshold. To support a method claim it would
  need matched-adequacy retraining at every condition — roughly 4 arms × 3
  conditions × 2.2 h ≈ **26 h** — and stage 4 gives a strong prior that the
  result would be ≈ 0. **I do not recommend spending it.**

---

## Part E — framing and verdict

### E1. What would justify the general claim

Your four requirements, sharpened, with current status:

1. **`Δ_HS > 0` across a region of `(L, r)`, including `r < L` and `r > L`.**
   **Not yet measured** (Track B used adaptive `r`). This is D1. *Amendment:* the
   `r > L` half is already **[MATH]**-guaranteed to degrade toward EM-GS, not
   below it (`cadzow_project` no-op guard), so the *experiment* is really about
   `r < L`, which cuts the interesting grid in half.
2. **`Δ_HS > 0` on a channel with no exact sparsity, with untuned `r`.**
   **Delivered by C4** under the clustered reading: effective rank 5.36, `r = 7`
   costs 0.01%. *Amendment:* the spectral evidence is delivered; the **NMSE**
   evidence still needs one HS-EM-GS run on SV channels (~25 min, not in D1's
   budget above — add it).
3. **An oracle-free rank rule recovering most of matched-`r` performance.**
   **Largely delivered [FACT]:** held-out residual gets **91%** of oracle
   (+3.709 / +4.077 dB). Needs replication across SNR, which D1 gives free.
4. **The useful `r` region tracking `floor(M/2)`.**
   **Partly delivered:** C2 shows `L_eff/cap` falling with `M` and Track B
   measured the matching NMSE trend (0.03 → 0.81 → 2.45 dB). One more `M` closes it.

**I would add a fifth:** the claim should state the **SNR regime**, because C3
shows the estimate's compressibility is SNR-dominated. "Effective under unknown
multipath order" without an SNR qualifier repeats the stage-3 pooling error at
the level of the abstract.

### E2. What we must not claim

- **Anything presenting `r = 7` as physically motivated.** A1 finds no source in
  the repository; `SystemModel.pdf` explicitly leaves `L_k` unconstrained.
- **Anything generalizing from the matched `L ≤ r = 7` setting** — that is the
  diagonal of a grid we have not run.
- **Any learned-model structural gain stated without a training-adequacy
  control.** Stage 4 makes this concrete: the honest version of "+1.209 dB" is
  "+0.078 dB once the baseline is trained adequately".
- **Any pooled `Δ` as a headline or a decision rule** — stage 3's standing rule.
- **"The channel is low rank"** as if exact. C1 shows even `L = cap` has
  effective rank 8.54; the operator exploits **spectral decay**, and the claim
  should say so.
- **"Vacuous on Xiao's channel"** — my own earlier over-statement, corrected in C4.
- **`L_k ~ U{3..7}` as a physically representative channel.** It is a design
  choice; the paper draft already scopes it correctly and should keep doing so.

### E3. Final output

**1. How over-specific is `L_k ∈ [3,7]`, `r = 7`?**
Moderately, and less than it first appears — but for a reason that *strengthens*
the work rather than excusing it. The **parameters** are arbitrary (A1: no
source; A2: `r = L_max` by construction). But C1 shows the mechanism was never
exact sparsity — it is spectral decay, which is generic — and C4 shows the
`r = 7` operator transfers untuned to an independently specified channel. **The
setup is over-specific; the mechanism is not.**

**2. The single most important generalization risk.**
**Not `L`. It is SNR.** C3: the estimate's effective rank moves 13.77 → 5.73
across SNR at fixed `L`, versus 8.91 → 11.77 across `L = 1…16` at fixed SNR —
**SNR is ~3× the stronger axis.** Every rank-selection rule reads that spectrum,
so every one of them is SNR-fragile, and stages 3–4 already found the low-SNR
regime where the prior misbehaves in the learned arms. A study that sweeps
`(L, r)` at one SNR would measure the weaker axis and report it as the answer.

**3. Minimum classical plan, costed.** ~**1.5 h wall** on 4 cores: a 3×3 pilot
(30 min) → 3×4 refinement (40 min) → adaptive-`L̂` reference column (25 min), SNR
drawn per trial and binned post hoc, 1000 trials/cell. Plus **25 min** for
HS-EM-GS on SV channels to convert C4's spectral evidence into NMSE evidence.
Cite Track B Experiment C for the L-axis rather than repeating it.

**4. Minimum learned follow-up.** OOD evaluation only (minutes), reported as
**robustness**, not as a method claim. Do not fund matched-adequacy retraining
(~26 h) on a stage-4 prior of ≈ 0.

**5. Is adaptive rank / soft shrinkage worth developing?**
**Soft shrinkage: yes** — two lines in the existing operator, no hard decision to
get wrong, and it addresses the SNR fragility that is risk #1. **Threshold and
gap rules: no** — C1/C3 show they read a spectrum that is noise-dominated
exactly where they are needed. **Held-out residual: already validated at 91% of
oracle**; keep it as the reference.

**6. Recommended method hierarchy for the paper.**
1. EM-GS (baseline)
2. HS-EM-GS, **adaptive `L̂`** via held-out residual — *the headline method*
3. HS-EM-GS, soft shrinkage — *the proposed refinement, if D2 confirms*
4. HS-EM-GS, oracle rank — *diagnostic upper bound, labelled as such*
5. fixed `r = L_max` — *demoted to an ablation*, since A2 says it cannot carry a
   claim
The learned arms do not belong in this hierarchy on current evidence.

**7. Proposed table and figure set.**
- **T1** assumption inventory (B2), with the `L̂`-free property stated
- **T2** `Δ_HS(L, r)` per SNR bin — the D1 deliverable
- **T3** spectral summary (C1–C4), tail energy at `r` as the criterion column
- **F1** `Δ_HS` vs SNR, curves indexed by `L/cap` — *the primary figure*
- **F2** cumulative Hankel energy (already produced: `fig_hankel_spectrum.png`)
- **F3** gain vs `L/cap` — Track B Experiment C, renormalized, already measured
- **F4** rank-rule comparison: fixed / adaptive / shrinkage / oracle vs SNR

**8. Verdict: GO — for the classical line, MODIFY the framing, STOP the learned
line.**

- **GO (classical).** The reasoning is the one you flagged: `H0 − U0` is positive
  in **every** SNR bin, +0.809 dB at the worst, monotone, **no training and
  therefore no adequacy confound**. It is the most robust Hankel finding in the
  project. C4 now adds that it is not an artifact of our sparse channel, and the
  91%-of-oracle rank rule shows it is deployable. Remaining cost to a defensible
  general claim: **~2 h**.
- **MODIFY (framing).** Recast from "low-rank/sparse-path prior" to
  **"spectrally compressible spatial structure"**, and attach an SNR qualifier.
  C1 and C4 support the broader claim better than the narrow one, and the narrow
  one is the version A1 cannot justify.
- **STOP (learned).** Stage 3 STOP-MARGINAL, stage 4's 15× collapse under
  training-adequacy control, PROMPT 7's failed low-SNR criterion. Three
  independent results point the same way. Further spend needs ~26 h to reach a
  claim the existing evidence predicts will be ≈ 0.

---

## Files

- `reports/trackD_spectral_diagnostics.json` — all Part C metrics
- `results/track_d/spectral/fig_hankel_spectrum.png` — the Part C figure
- `scratch/trackD_spectral_diagnostics.py`, `scratch/trackD_spectral_plots.py`
- Cited, not re-run: `trackB_hankel_emgs/results/experiment_C_path_count.csv`,
  `trackB_hankel_emgs/results/ablations.json`
