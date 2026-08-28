# Track D — clarifications and stage-1 training plan

Rendered from `reports/trackD_clarify.json`, `reports/trackD_verify.json` and
`reports/trackD_timing.json`. **No training has been launched.**

---

## 1. RSR convention — locked

**RSR = 10 dB in our repository / system-model definition** (Cui single-user
denominator). No conversion to 14.77 dB.

Config changes made:

- `DataConfig.rsr_train_mode: Literal["fixed","range"] = "fixed"` — new field,
  so no later reader can assume a range was sampled.
- `SystemConfig.rsr_db = 10.0` (unchanged), `rsr_paper_equiv_db = 14.7712`
  (derived property).
- Every trial row already stores both `rsr_ours_dB` and `rsr_paper_equiv_dB`
  (`evaluate.ROW_COLUMNS`); unchanged.

### Consequence, booked now

Training at **fixed** RSR means every stage-1 checkpoint is specialized to
`RSR = 10 dB (ours)`. The reference level enters the network through `B`, which
appears in `Y = ĜS + B` and hence in both the phase factor and `κ` at every one
of the 10 unrolled layers — it is not a nuisance parameter the model can ignore.

So the later Xiao-comparability experiment at `rsr_paper_equiv_dB = 14.77 dB`
**requires a retraining, not a re-evaluation.** A fixed-RSR model is
off-distribution at any other reference level, and evaluating it there would
measure distribution shift, not estimator quality.

**Cost booked: +1 training per arm per initializer at the comparability RSR**
(≈1.1 h each at `N=32`). This is recorded in `config.py` next to
`rsr_train_mode` so it cannot be forgotten.

---

## 2. Correction — the untrained network is **not** a classical estimator

**My phase-2 claim was wrong.** I wrote that the untrained URformer "*is* the
classical algorithm." It is not. Here is the algebra and the measurement.

### The algebra

Per layer, with the residual zero:

```
Y_rec = α·(Y_direct ⊙ R_learned) + (1−α)·Y_direct
      = Y_direct ⊙ [ α·R_learned + (1−α) ]
```

Write the effective elementwise multiplier `m = α·R_learned + (1−α)`. Then:

- the layer equals **GS** iff `m ≡ 1`, i.e. `α = 0` **or** `R_learned ≡ 1`
- the layer equals **EM-GS** iff `m ≡ R_exact`, i.e. `α = 1` **and**
  `R_learned ≡ R_exact = I₁(κ)/I₀(κ)`

The defaults are `α = σ(−2) = 0.11920292` and `filter_init = "random"`. Neither
condition holds, so `m` is strictly between the two.

### The measurements (`reports/trackD_clarify.json`)

Answering each numbered item:

**1. FilterNet initialization at `t=0`** — PyTorch default `nn.Linear` init
(Kaiming-uniform weights, uniform bias), i.e. **random**. It is *not* seeded to
anything Bessel-like.

**2. What `R_learned(κ)` produces before training** — a nearly constant value
near 0.4, essentially independent of `κ`, because a randomly-initialized
sigmoid-output MLP is flat. Over five seeds, at the reference config:

| seed | `R_learned` min | max | mean | rel err vs Bessel |
|---|---|---|---|---|
| 0 | 0.4495 | 0.5029 | 0.4630 | 0.525 |
| 1 | 0.2860 | 0.4868 | 0.3231 | 0.669 |
| 2 | 0.4125 | 0.4246 | 0.4169 | 0.572 |
| 3 | 0.3586 | 0.4207 | 0.4142 | 0.575 |
| 4 | 0.2916 | 0.4159 | 0.3061 | 0.686 |

The exact Bessel ratio at the same operating point has mean **0.9723** (range
0.1785–0.9906). So `R_learned` is wrong by **53–69% relative** at
initialization. It is nowhere near `R_exact`.

**3. Warm-started or random?** — **Random**, and worse than that: until this
turn `ModelConfig.filter_init` was **declared but never read**. `URformerLayer`
constructed a plain `FilterNet` and never consulted the field;
`warmstart_filternet()` existed in `filter_net.py` and was never called. It was
a dead config field. **This is now wired** (`URformer.apply_filter_warmstart`),
with the default left at `"random"` so no reported behaviour changed.

**4. Initial `α` for every layer** — `0.11920292` for all 10 layers,
identically (`gate_init="near_gs"`, `g=−2.0`, untied but identically
initialized).

**5. Initial Transformer residual** — **exactly 0.0** (`out_proj` weight and
bias zero-initialized). Verified to the bit, not to a tolerance.

**6. Which classical estimator does the untrained URformer equal?**
**None.** Measured against one classical step on the same realization:

| comparison | relative error |
|---|---|
| untrained URformer vs **one GS step** | **0.1230** |
| untrained URformer vs **one EM-GS step** | **0.0947** |
| effective multiplier `m` | min 0.9344, max 0.9407, mean 0.9360 |
| max deviation of `m` from 1 | **0.0656** |

It is ~12% away from GS and ~9% away from EM-GS. Not equal to either.

### The four claims, kept strictly separate

| | Claim | Verdict | Evidence |
|---|---|---|---|
| (i) | Transformer residual is initially exactly zero | **TRUE** | max abs residual `0.0`, exact |
| (ii) | Architecture **can be forced** to exact GS | **TRUE** | `α=0`, residual off → rel err **6.02e-13** |
| (iii) | Architecture **can be forced** to exact EM-GS | **TRUE** | exact Bessel, `α=1`, residual off → rel err **5.84e-13** |
| (iv) | **Default untrained** URformer equals some classical estimator | **FALSE** | 0.1230 from GS, 0.0947 from EM-GS |

My phase-2 report asserted (iv) on the strength of (i) and (ii). That was an
unjustified leap: a zero residual makes the *Transformer* inert, but the gated
filter still multiplies `Y_direct` by `m ≈ 0.936`, and a random `R_learned`
is not `R_exact`. The corrected statement is:

> The untrained URformer is a **gated, mis-filtered GS variant** — structurally
> a single classical update per layer, but with a learned multiplier that at
> initialization is neither 1 nor the Bessel ratio. It is *close to* GS
> (within ~12%) because `α` is small, not equal to it.

`reports/trackD_phase2.md` has been corrected, and
`test_default_untrained_urformer_is_NOT_a_classical_estimator` now pins the
corrected claim so it cannot silently regress.

**No default was changed to rescue the claim.** `gate_init` is still
`"near_gs"`, `α` is still 0.1192, `filter_init` is still `"random"`.

### Would a warm start fix it? No.

Even with a *perfect* Bessel warm start, `m = α·R_exact + (1−α)` ranges
**0.9021–0.9989**, still neither 1 nor `R_exact`. Getting exact EM-GS needs
`α = 1` **and** a perfect warm start together. Recorded so this is not
mistaken for a warm-start bug.

### Prerequisites the prompt asked for

**κ measured over the actual training dataset** (512 realizations, full SNR
range, at `G=0` where `κ` is largest):

| statistic | value |
|---|---|
| min | 0.1181 |
| p0.1 | 1.3223 |
| p1 | 3.7581 |
| p50 | 66.196 |
| p99 | 866.16 |
| p99.9 | 1148.5 |
| **max** | **1628.4** |

**This materially revises the phase-2 statement that "κ is small (max ≈ 116)."**
That 116 came from the *audit's* probe configuration (`N=8, K=2, P=8`, single
SNR). At the Track D reference configuration (`N=32, K=3, P=20`, SNR ∈ [0,20] dB)
κ reaches **1628** — 14× larger. Scaled Bessels were mandatory anyway; the point
is that the warm-start grid had to be *measured*, and a grid calibrated to
κ_max ≈ 116 would have been badly wrong.

**Warm-start grid actually used:** `logspace(log10(1.3223), log10(6513.5), 4096)`
— lower bound = measured p0.1, upper = 4× measured max.

**Achieved warm-start fit MSE: 9.896e-05**, against the `< 1e-4` threshold.
Converged in 140 Adam steps.

⚠️ **The MSE threshold is passing but weak.** Over the same grid the warm-started
net's **max absolute error is 0.0560** (mean 0.0058). A 2-layer, 32-hidden-unit
MLP on `log1p(κ)` cannot represent the Bessel ratio's knee tightly. So
"warm-started" means "within ~6% worst-case of EM-GS", not "equal to EM-GS".
If we want a genuinely EM-GS-equivalent start I would raise `filter_hidden` to
128 and tighten the criterion to max-abs-error `< 1e-3` — flagged as a proposal,
**not changed**, since `filter_init` defaults to `"random"` and stage 1 does not
depend on it.

---

## 3. The training matrix

### 3a. Checkpoint reuse — D1 and D2 **can** share a model

`P` does not enter any weight shape: `Z`, `κ` and `Y` are elementwise, and the
LS step forms `S S^H` per sample. Only the Transformer is shape-locked, and it
is locked to `(N, K)` only. So one `N=32` model trained over both an SNR range
and a `P` distribution can serve D1 and D2.

Judged against the stated priority order:

1. **Scientifically fair** — yes, with the caveat in 3c recorded.
2. **No leakage** — unaffected; splits are by trial index, not by `P`.
3. **Clear interpretation** — acceptable *provided* the `P` training
   distribution is stated on every D2 figure.
4. **Efficiency** — a bonus, not the reason.

**Decision: merge D1 and D2 into one checkpoint set per initializer.** This cuts
the matrix from 12 to **9** trainings. I am *not* merging D3 — that is a genuine
shape constraint, not an efficiency question.

### Full table of proposed trained models (post-merge)

| # | checkpoint | N | K | train `P` | train SNR | RSR | init | pilots | FilterNet init | purpose | reused by |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `d12_N32_random` | 32 | 3 | `U{6,10,15,20,30}` | `U[0,20] dB` | 10 dB ours | random | fixed_S | random | paper-faithful arm | D1, D2 |
| 2 | `d12_N32_spectral` | 32 | 3 | `U{6,10,15,20,30}` | `U[0,20] dB` | 10 dB ours | spectral | fixed_S | random | honest control; **stage-1 primary** | D1, D2 |
| 3 | `d12_N32_linls` | 32 | 3 | `U{6,10,15,20,30}` | `U[0,20] dB` | 10 dB ours | linearized_ls | fixed_S | random | third control | D1, D2 |
| 4 | `d3_N8_random` | 8 | 3 | 20 | `U[0,20] dB` | 10 dB ours | random | fixed_S | random | array-size sweep | D3 |
| 5 | `d3_N8_spectral` | 8 | 3 | 20 | `U[0,20] dB` | 10 dB ours | spectral | fixed_S | random | array-size sweep | D3 |
| 6 | `d3_N8_linls` | 8 | 3 | 20 | `U[0,20] dB` | 10 dB ours | linearized_ls | fixed_S | random | array-size sweep | D3 |
| 7 | `d3_N16_random` | 16 | 3 | 20 | `U[0,20] dB` | 10 dB ours | random | fixed_S | random | array-size sweep | D3 |
| 8 | `d3_N16_spectral` | 16 | 3 | 20 | `U[0,20] dB` | 10 dB ours | spectral | fixed_S | random | array-size sweep | D3 |
| 9 | `d3_N16_linls` | 16 | 3 | 20 | `U[0,20] dB` | 10 dB ours | linearized_ls | fixed_S | random | array-size sweep | D3 |

Plus, for stage 1 only, two arm-2 companions at `N=32` (see §5b):

| # | checkpoint | N | params | purpose |
|---|---|---|---|---|
| S1a | `stage1_N32_spectral_full` | 32 | 1,586,900 | = model 2 above, trained at fixed `P=20` |
| S1b | `stage1_N32_spectral_filteronly` | 32 | **980** | attribution ablation |

⚠️ **D3's `N=32` column does NOT reuse models 1–3** under the merge. Models 1–3
are trained on a `P` *distribution*; D3 runs at fixed `P=20`. Mixing a
`P`-range-trained model into an otherwise `P`-specific sweep would confound the
`N` trend with a training-distribution difference. D3's `N=32` column therefore
needs either its own fixed-`P` model or an explicit note. **I recommend
training three more fixed-`P` `N=32` models for D3** (+3.3 h), bringing the
total to **12** — the same count as before, but now for a *stated scientific
reason* rather than by oversight. Efficiency was priority 4; this is priority 1.

### 3b. Variable-`P` batching — **fixed `P` per batch, varying across batches**

Chosen scheme: **bucketing by `P`.** Each batch is drawn from a single `P`
value; `P` varies across batches within an epoch. No padding, no mask.

Why this and not padding:

- **No mask exists for the network to exploit.** With padding, a mask tensor (or
  the padding pattern itself) is an input feature perfectly correlated with `P`,
  and the model can condition on it. Bucketing removes that channel entirely.
- **`κ` statistics are untouched.** `κ = 2Z⊙|Y|/σ²` is elementwise over the real
  `N×P` grid. Padded entries would either be zeros (dragging `κ` statistics and
  any normalization toward zero) or garbage; neither occurs here.
- **The LS step stays exact.** `S S^H` is formed from the true `K×P` pilots with
  no padded columns, so the M-step is the same operator at every `P`.

Cost: batches are homogeneous in `P`, so shuffling is *within* bucket plus a
shuffled bucket order. That is a mild reduction in gradient-batch diversity,
which I will record. It is not needed for stage 1 (fixed `P=20`).

**Confirmation: no `P` information leaks into the estimate.** The network sees
only `Z, S, B, σ²`; `P` is expressed solely as the second dimension of those
tensors, exactly as it is at test time.

### 3c. Fairness caveat — recorded before running

A `P`-range-trained model is generically **worse** at any single `P` than a
`P`-specific model. Therefore, pre-committed:

- If URformer **wins** at a given `P`, the claim is **conservative** — the
  figure caption and report must say so explicitly.
- If URformer **loses** at some `P`, the result is **ambiguous** between
  "estimator is worse" and "training distribution was diluted". **Any losing
  point must be re-checked with a `P`-specific model before it appears in a
  figure.** Budget for up to 5 such re-checks (one per `P` in the D2 grid).

### 3d. Initializer checkpoints — **confirmed, genuinely separate**

Separate checkpoints are required. The initializer is **not** a runtime switch:

`Ĝ⁽⁰⁾` is the input to layer 1, and every subsequent layer's input is a
deterministic function of it. The learned parameters therefore adapt to the
*statistics of the initialization*, which differ sharply between the three:
`random` has zero mean and no dependence on the data at all, while `spectral`
and `linearized_ls` are already data-dependent estimates with structured error.
A network trained to correct one is solving a different regression problem from
one trained to correct another. Concretely, `spectral` starts far closer to the
truth, so the residual the Transformer must learn is smaller and differently
distributed.

Empirically the three starting points are not interchangeable: on the stage-1
test set the classical estimators alone differ by initializer, and gate-I
showed the network readily specializes to whatever it is fed.

So: **3 initializers ⇒ 3 checkpoints per (N, P-regime)**, and
`URformer-random` / `URformer-spectral` are two separately trained models. This
is what multiplies the matrix, and it is the correct cost.

### 3e. Seed ledger — asserted programmatically

Declared ranges (disjoint by construction, enforced in `DataConfig.__post_init__`):

| split | declared range | actually used |
|---|---|---|
| train | `[0, 1000000)` | `[0, 20000)` |
| val | `[1000000, 2000000)` | `[1000000, 1002000)` |
| test | `[2000000, 3000000)` | `[2000000, 2002000)` |

Assertion output (`reports/trackD_clarify.json → seed_ledger`):

```
train_vs_val   declared_overlap_count = 0   used_overlap_count = 0
train_vs_test  declared_overlap_count = 0   used_overlap_count = 0
val_vs_test    declared_overlap_count = 0   used_overlap_count = 0
all_disjoint = true
```

All three `assert` statements passed. Also pinned as a unit test
(`test_seed_ranges_used_by_defaults_are_disjoint`) so it runs on every suite
execution. Every checkpoint in the table above uses these same ranges.

---

## 4. Gate J — exact conditions and what it does **not** establish

Conditions actually used (`verify.gate_J`, `N=16, K=3, P=20`):

| | |
|---|---|
| `W` | **exactly zero** — `exact_forward(G, S, B, 0.0)`, no RNG drawn |
| `sigma2` | swept `1e-6, 1e-9, 1e-12` (the *filter's* `σ²`, decoupled from the noise, which is zero) |
| `alpha` | **not applicable** — `em_gs_layer` is the fixed classical Torch layer, not the URformer |
| Bessel | **fixed exact** `i1e/i0e`. No FilterNet involved |
| Transformer | **absent** — no network is instantiated in this gate |
| initialization | `Ĝ⁽⁰⁾ = G_true`, i.e. **started at the truth** |
| oracle phase | **not supplied explicitly**, but implied: with `W=0` and `Ĝ⁽⁰⁾=G`, `Y = GS+B` is the exact noiseless field, so the phase is exact by construction |
| iterations | **one** |

### What the gate establishes

That the exact noiseless solution is a **fixed point** of one classical EM-GS
update as `σ²→0`: starting *at* the truth, one iteration returns the truth to
`3.28e-14`. Together with the companion sweep it also shows EM-GS → GS in the
`σ²→0` limit (`3.06e-14`), since `R(κ)→1`.

### What the gate does **NOT** establish

Stated plainly, because the inference is tempting and wrong:

- **It does not show that EM-GS converges from an arbitrary initialization.**
  The gate starts at the answer. It is a self-consistency check, not a
  convergence result. Nothing here bounds the basin of attraction, and biased
  phase retrieval is non-convex with known local minima.
- **It does not show anything about the URformer.** No learned module is
  instantiated. It validates the *classical* Torch layer only.
- **It does not say `W=0` makes the problem easy.** Even noiseless, from
  `Ĝ⁽⁰⁾=0` the problem is a non-convex phase-retrieval instance.
- **It is not a statement about `σ²>0`.** The residual scales linearly in `σ²`
  (the exact `1−R(κ)` bias); at realistic `σ²` the truth is *not* a fixed point,
  which is correct EM-GS behaviour and not an error.

"`W=0` ⇒ arbitrary-init EM-GS converges exactly" is **not** inferable from gate
J, and this paragraph exists so it is not accidentally inferred.

---

## 5. Stage-1 plan

**Question:** does URformer improve over our validated GS and EM-GS baselines on
**unseen test channels**? Nominal array size only, `N = 32`.

### 5a. Pre-registered success criterion — fixed before any numbers exist

> **Success** = `URformer-spectral` beats `EM-GS-spectral` by **≥ 2 dB median
> NMSE** on the held-out test set, evaluated on **paired** trials, with the
> bootstrap CI (2000 resamples) of the **paired per-trial difference** excluding
> zero.
>
> The paired-difference *distribution* is reported, not only the two means.

**I endorse 2 dB.** Reasoning from the measured baseline spread (200 held-out
test trials, spectral init, `reports/trackD_clarify.json → baseline_spread`):

| quantity | value |
|---|---|
| EM-GS-spectral NMSE (ratio-of-sums) | **−9.14 dB** |
| GS-spectral NMSE (ratio-of-sums) | −8.75 dB |
| per-trial NMSE spread (EM-GS) | median −12.94 dB, **std 6.27 dB**, p5..p95 = −21.99..−3.34 |
| **paired** GS−EM-GS difference | median **0.068 dB**, std **0.217 dB**, boot CI95 **[0.044, 0.100]** |

The per-trial NMSE spread is huge (6.27 dB std), but the **paired** difference
between two classical estimators is tight — median 0.068 dB with a CI of width
0.056 dB. Pairing removes the channel-realization variance almost entirely,
which is exactly why the criterion is specified on paired differences.

So 2 dB is:

- **~9× the paired noise scale** (0.217 dB) ⇒ not a statistical-power question;
  with 2000 test trials the CI half-width will be ~0.01 dB.
- **~30× the real GS-vs-EM-GS algorithmic effect** (0.068 dB) ⇒ comfortably
  above "a genuine but minor algorithmic improvement."
- **Well below what the paper claims** ⇒ not an unreachable bar if URformer
  works as advertised.

It is therefore a **practical-significance** threshold, not a detectability one,
which is the right kind of pre-registration here. One caveat recorded in
advance: the URformer-vs-EM-GS paired difference will have a larger spread than
the classical-vs-classical 0.217 dB, because the two are different estimator
families; the CI is computed on the actual paired differences, not assumed.

### 5b. Three arms

| arm | model | parameters | purpose |
|---|---|---|---|
| 1 | **URformer** (full) | 1,586,900 | the paper's architecture as implemented |
| 2 | **URformer-filteronly** | **980** | FilterNet + gate + LS; Transformer **not constructed** |
| 3 | classical: GS, EM-GS, linearised LS | 0 | matched initializers, all labeled |

Arm 2 is now implemented (`ModelConfig.use_transformer=False`), verified to
build 980 parameters with `former is None` in every layer, and to still
degenerate to exact GS at `α=0` (`test_filteronly_still_runs_and_equals_gs_when_alpha_zero`).

**Why arm 2 is the highest-information run:** the Transformer is 99.94% of the
parameters. Without it we cannot attribute any gain between "unrolling the
physics helped" and "a 1.57M-parameter learned denoiser helped." Pre-committed
interpretation:

- full wins, filter-only does not ⇒ **the gain is the denoiser**, and the report
  says so plainly.
- filter-only captures most of the gain ⇒ a **stronger and more interesting**
  result than the paper's: 980 parameters buying the improvement.
- neither wins ⇒ URformer does not transfer to our channel model, reported as a
  negative result.

Not in scope: weight-tied ablation, antenna-token variant, closed-box
Transformer. All still deferred.

### 5c. Specification

| | |
|---|---|
| training samples | **20,000** (paper Table I), seeds `[0, 20000)` |
| validation samples | **2,000**, seeds `[1000000, 1002000)` |
| test trials | **2,000**, seeds `[2000000, 2002000)` |
| SNR training distribution | `U[0, 20] dB`, quantized to the millidB grid, sampled per-sample from a dedicated substream |
| `P` training distribution | **fixed `P = 20`** for stage 1 (no bucketing needed; scheme 3b applies to D2 later) |
| RSR | **10 dB, our convention** (`rsr_paper_equiv_dB = 14.77`), `rsr_train_mode = "fixed"` |
| **K** | **3** — our repository's value (Track B frozen `K=3`), **not** the paper's `K=4` |
| **L_k regime** | **`L_k ~ U{3..7}` i.i.d. per user per realization**, matching Track B exactly via `draw_L_k` |
| pilots | `fixed_S` (one matrix reused across the dataset), seed `777000001` |
| initializer | **`spectral`** for the primary comparison (both arms). `random` and `linearized_ls` deferred to the full matrix. |
| FilterNet init | `random` (default; warm start now wired but not used) |
| gate init | `near_gs`, `α₀ = 0.1192` |
| epochs | **50**, Adam, `lr = 1e-3`, cosine annealing to zero, batch 32, grad-clip 1.0 |
| early stopping | **none** — fixed 50-epoch budget, so both arms get identical compute and the comparison is not confounded by different stopping points |
| checkpoint selection | **best validation ratio-of-sums NMSE**, `best.pt`. **Never test.** Test is touched exactly once, at the end, for the final numbers. |
| dtype | `float32` (training precision floor ~1e-7 relative, per gate B/C) |
| threads | 1 (determinism; measured to cost ≤20%) |

**Plots generated** (all via `plot_results.py` conventions, `.png` + `.pdf`):

1. `stage1_training_curves` — train and validation NMSE vs epoch, both arms.
2. `stage1_nmse_vs_snr` — test NMSE vs SNR: GS, EM-GS, linearised LS,
   URformer-full, URformer-filteronly, all spectral-init, all labeled.
3. `stage1_paired_difference` — histogram + ECDF of the **paired per-trial**
   `URformer − EM-GS` difference in dB, with the median and bootstrap CI marked,
   and the 2 dB decision line drawn. This is the figure the criterion is read off.
4. `stage1_attribution` — full vs filter-only vs EM-GS, the arm-2 attribution.

**Estimated runtime:**

| item | hours |
|---|---|
| arm 1 (full, `N=32`, 20k×50) | 1.11 |
| arm 2 (filter-only — far cheaper, 980 params) | ~0.35 (est.) |
| classical baselines on 2,000 test trials × 3 initializers | ~0.15 |
| evaluation + plots | ~0.05 |
| **total stage 1** | **≈ 1.7 h** |

Arm 2's estimate is extrapolated from the parameter reduction, not measured; the
LS and forward passes dominate and are unchanged, so 0.35 h is a lower bound and
it may be closer to arm 1. I will report the measured figure.

### What stage 1 decides

- **Success** ⇒ proceed to the full matrix (12 trainings, ~11.7 h) and the D1–D3
  figures.
- **Failure with filter-only ≈ full** ⇒ the Transformer is not earning its
  99.94%; report that and reconsider the architecture before spending 11.7 h.
- **Failure of both** ⇒ URformer does not transfer to our geometric ULA model at
  `N=32`; report as a negative result and stop rather than sweeping `N`.

---

## Code changes made this turn

Additive and behaviour-neutral; all defaults unchanged.

| change | file | why |
|---|---|---|
| `rsr_train_mode` field | `config.py` | item 1, explicitly requested |
| `use_transformer` flag | `config.py`, `urformer.py` | arm 2 of stage 1 |
| `filter_warmstart_cache` + `apply_filter_warmstart()` | `config.py`, `urformer.py` | wires the **dead** `filter_init` field found in item 2 |
| 5 new tests | `tests/test_trackD_urformer.py` | pin the corrected claim, arm 2, seed ledger, `rsr_train_mode` |

Verified unchanged: `filter_init="random"`, `gate_init="near_gs"` (`α=0.1192`),
full-model parameter count 1,586,900. Filter-only builds 980.

**Tests: 383 passed, 1 skipped** (was 378+1; +5 new). **Verification gates:
15/15 pass.**

**No training launched.**
