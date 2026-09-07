# Structure versus training: how the two manuscripts fit together

This project produced two manuscripts that a reviewer holding both will read as
contradictory. One says a Hankel structural prior is worth roughly 3 dB. The
other says it is worth roughly nothing. This document states why both are true.

**No new numbers appear here.** Every figure is carried from a committed
results file and is tagged with the experiment family it came from.

---

## 1. The unified statement

> **A structural prior supplies information only where learning cannot.**

Concretely, and in a form either abstract can carry:

The Hankel low-rank prior is worth roughly **3 dB inside a classical iteration
with no trainable capacity**, and **close to nothing inside a trained unrolled
network at N = 32 once the training regime is adequate**. What survives the
control on the trained side is **robustness under distribution shift**
(+0.48 dB degradation against +0.73 dB), not in-distribution accuracy.

The two results are not in tension because they answer different questions.
EM-GS has no free parameters with which to discover the array manifold, so
telling it about the manifold adds information. A trained unrolled network
with 1.59 M parameters can discover the manifold itself, given enough gradient
in the regime that matters — and once it has, being told the same thing again
adds nothing. The apparent structural gain on the trained side was the network's
high-SNR underfitting wearing the prior's clothes.

---

## 2. Every headline number, mapped to its family and configuration

Numbers from different families are **not** comparable as measurements of the
same estimator. Nothing in either manuscript may be read across rows.

### The classical families (Paper 1)

| # | family | configuration | statistic |
|---|---|---|---|
| B3 | `results/track_b/b3/` | N ∈ {8,16,32}, P ∈ {10,30}, 6-point SNR grid, RSR 12 dB, T = 50, `select_iter` 20, `n_cz` 4, 400 trials/point | ratio-of-sums within a point |
| EXT | `trackB_hankel_emgs/results/` | P = 30 only, 7 SNR points incl. −10 dB, RSR 12 dB, T = 50, `n_cz` 4, 600/400/200 trials | ratio-of-sums, mean of per-SNR dB |
| TD | `results/track_d/partB9/` | N ∈ {16,32,64}, P = 20, SNR ~ U[−10,20] binned post hoc, RSR 10 dB, T = 100, `select_iter` 25, `n_cz` 4 | paired per-trial median per bin |

**Correction carried from PROMPT 12 Part A.** Earlier drafts described B3 as
using a *fixed* order and `n_cz = 4`, and TD as using an *adaptive* order and
`n_cz = 1`. Both descriptions were wrong on two fields. Read off the call sites
(`scripts/run_b3.py:145`, `scratch/trackD_partB9_sweeps.py:171`,
`rydberg_sim/track_b_proposed.py:246`): **both families call `hs_gs_auto`,
both select the order from held-out pilots, and neither passes `cadzow_iter`,
so both run `n_cz = 4`.** The families differ in **operating point and
statistic**, not in operator.

| headline number | value | family | where |
|---|---|---|---|
| HS-GS over EM-GS, N = 32, P = 30 | +2.27 to +3.04 dB | **B3** | Paper 1 §III-A |
| Aperture gain, N = 8/16/32 | −0.19 / +0.78 / +2.85 dB | **B3** | Paper 1 §III-B |
| Win rate, N = 8/16/32 | 33.5 / 74.0 / 95.3 % | **B3** | Paper 1 §III-B |
| Constraint active, N = 8/16/32 | 57.5 / 92.7 / 99.6 % | **B3** | Paper 1 §III-B |
| EM-GS vs unconstrained CRLB, SNR ≥ 5 | within 0.051 dB | **B3** | Paper 1 §III-A |
| CCRB below unconstrained CRLB | 7.05–7.11 dB | **B3** | Paper 1 §III-A |
| Aperture gain (separate experiment) | +0.03 / +0.81 / +2.45 dB | **EXT** | Paper 1 §III-B note |
| Path-count decay, L = 2 → 16 | +7.04 → −0.12 dB | **EXT** | Paper 1 §III-C |
| EM-GS flatness in N / in L | 0.012 dB (B3) / 0.191 dB (EXT) | **B3 / EXT** | Paper 1 §III-C |
| Effective-rank zero crossings, N = 16/32/64 | 0.588 / 0.518 / 0.544 | **TD** | Paper 1 §III-D |
| N=16 vs N=64 magnitude residual | mean 0.493, max 0.901 dB | **TD** | Paper 1 §III-D |
| Prospective prediction errors | 0.13 dB and 0.315 dB | **TD** | Paper 1 §III-E |
| K-invariance, SNR ≥ 5 / pooled | 0.094 / 0.294 dB | **TD** | Paper 1 §IV |
| Order-selection share of runtime | 57.7–85.5 % | `results/track_b/timing.json` (B3 configuration) | Paper 1 §IV |
| r_eff(L = 16, N = 32) | **8.73**, ρ = 0.546 | EXT generator, Experiment C re-index | Paper 1 §I, §XII |

The B3 and EXT aperture rows measure the same quantity at different operating
points and **disagree**. Both are correct for their own configuration. Neither
may be quoted for the other.

### The learned family (Paper 2)

Single family throughout: the Track D URformer study. N = 32, K = 3, P = 20
unless stated, SNR ~ U[−10,20] dB, RSR 10 dB, 10 unrolled layers,
1,586,900 parameters, 80k training samples / 13 epochs, 2000 paired test
realisations.

| headline number | value | store |
|---|---|---|
| URformer over EM-GS | +3.345 dB | `trackD_stage2_report.md` |
| — learned filter share (980 params) | 0.147 dB | `trackD_stage2_report.md` |
| — attention share (1,585,920 params) | 3.198 dB, **96 %** | `trackD_stage2_report.md` |
| Unrolling vs post-hoc Transformer (X1→U1) | +0.920 dB, CI [+0.888, +0.988] | `trackD_stage3_report.md` |
| Internal vs post-hoc projection, 15–20 dB | **+1.309 dB**, CI [+1.185, +1.459] | `p12/a2_internal_vs_posthoc.json` |
| Internal vs post-hoc projection, 0–5 dB | **−0.636 dB**, CI [−0.754, −0.552] | `p12/a2_internal_vs_posthoc.json` |
| Gradient share below 5 dB | 89.7 % | `trackD_stage5_results.json` |
| Per-bin gradient-share span | **30** (0.4653 / 0.0155) | `trackD_stage5_results.json` |
| Δ_H, mixed-SNR training, SNR ≥ 5 | +1.209 dB | `trackD_stage4_report.md` |
| Δ_H, focused [5,20] training, SNR ≥ 5 | +0.078 dB, CI [+0.050, +0.112] | `trackD_stage4_report.md` |
| Focused training gains: unstructured / structured | +2.653 / +1.629 dB | `trackD_stage4_report.md` |
| SNR-balanced loss vs unbalanced, SNR ≥ 5 | +1.902 dB, CI [+1.823, +1.986] | `trackD_stage5_eval.json` |
| — low-SNR change (pre-registered as a cost) | **+0.135 dB**, CI [+0.101, +0.172] | `trackD_stage5_eval.json` |
| Path-richness OOD degradation, structured / unstructured | **0.48 / 0.73 dB** | `trackD_stage5_eval.json` |
| Pilot efficiency vs generalisation, P = 10 | +2.231 dB pooled | `trackD_stage5_eval.json` |

**Two corrections carried from PROMPT 12.**

1. The gradient-share span is **30**, not 31. The store gives
   0.4653 / 0.0155 = 30.03 (`trackD_stage5_results.json`,
   `C1_snr_balanced_P20.unweighted_shares.grad_share`).
2. The K-invariance high-SNR spread is **0.094 dB**, not 0.095. Full precision
   is 0.09444; the 0.095 came from differencing already-rounded per-cell
   medians.

### The one number that must never be crossed between the two papers

Paper 1's Δ_HS (≈ +2.3 to +3.0 dB, classical) and Paper 2's Δ_H (+1.209 or
+0.078 dB, learned) are **both** "the value of the Hankel prior", and they are
**not** the same measurement. Paper 1's is EM-GS versus HS-GS at RSR 12 dB with
no trained component anywhere. Paper 2's is a trained unrolled network with and
without the same prior inside it, at RSR 10 dB and P = 20. A reader who
subtracts one from the other has made an error, and both manuscripts should be
written so that subtraction never looks inviting.

---

## 3. Explicit scope of the negative result

Paper 2's negative result — that the structural prior's in-distribution value
collapses under matched training — holds **only** under these conditions:

- **N = 32**, K = 3, L_k ~ U{3..7}, geometric ULA channels;
- **P = 20**, and the two pilot counts otherwise tested (10 and 35);
- **one data budget**: 80k training samples, 13 epochs;
- RSR 10 dB, SNR ~ U[−10,20] dB;
- the specific unrolled architecture used here (10 layers, 1.59 M parameters).

It is **not** a general claim that structural priors cannot help unrolled
estimators. Three things bound it from the other side, all from the same study:

1. **The same prior applied to the classical estimator gives a large, stable
   gain** — that is Paper 1's entire result, and no training adequacy is at
   stake there.
2. **Under distribution shift the structured network degrades least**
   (0.48 dB against 0.73 dB). The prior buys robustness even where it buys no
   in-distribution accuracy.
3. **At smaller aperture the classical prior turns actively harmful**
   (−0.19 dB mean at N = 8, worst −2.23 dB), so "structure helps classically"
   is itself conditional. The condition is the same one in both papers: how much
   of the available structural capacity the channel occupies.

Pending measurements that could narrow or widen this scope are pre-registered
as P19–P21 (`reports/p12/PREREG_P19_P21.md`): seed spread on the collapse,
Δ_H under the recommended balanced loss, and whether a log-domain loss
reproduces the balanced-weighting result.

---

## 4. The sentence for both discussions

> A structural prior supplies information only where learning cannot: inside a
> classical iteration with no trainable capacity it is worth roughly 3 dB,
> while inside a trained unrolled network at this aperture and data budget it
> is worth close to nothing once the training regime is adequate — what
> survives there is robustness under distribution shift rather than
> in-distribution accuracy.
