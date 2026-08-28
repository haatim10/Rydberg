# Track D stage 1 — results (PROMPT 4 Part C)

Rendered from `reports/trackD_stage1_results.json`, `trackD_partA.json`,
`trackD_partA4.json`, `trackD_verify.json`. Predictions from
`trackD_stage1_preregistration.md`, committed at `54655f2` **before** any run.

## Verdict, plainly

> **The pre-registered success criterion is NOT MET.**
>
> Best arm beats EM-GS-spectral by **1.434 dB** median paired NMSE. The bar was
> **≥ 2 dB**. The bootstrap CI excludes zero, so the improvement is real — it is
> simply below the threshold fixed in advance.

**All three pre-registered predictions held.**

---

## 1. Part A go/no-go

| # | Condition | Result | Verdict |
|---|---|---|---|
| 1 | every config field live | 23 tests; a **second** dead field found (`rsr_train_mode="range"`) and fixed | PASS |
| 2 | `EM-GS − oracle ≥ 3 dB` | **4.34–5.31 dB** across the sweep | PASS |
| 3 | warm start max abs `< 0.01` | **0.009999** at hidden=32, variant `R` | PASS |
| 4 | SNR and κ coverage | failed first, **widened** (below) | PASS |

**A2 — classical vs the oracle-phase line** (400 paired trials per point, N=32):

| SNR | GS | EM-GS | linLS | oracle | headroom |
|---:|---:|---:|---:|---:|---:|
| −10 | 9.63 | 6.80 | 7.15 | 2.46 | 4.34 |
| −5 | 4.12 | 2.73 | 2.04 | −2.58 | 5.31 |
| 0 | −1.76 | −2.26 | −2.19 | −7.47 | 5.21 |
| 5 | −7.65 | −7.92 | −5.80 | −12.60 | 4.68 |
| 10 | −12.96 | −13.06 | −8.14 | −17.56 | 4.51 |
| 15 | −18.10 | −18.18 | −9.16 | −22.69 | 4.51 |
| 20 | −22.88 | −22.90 | −9.77 | −27.53 | 4.63 |

Linearised LS **floors at −9.77 dB** while the others keep improving: the
strong-reference linearisation stops being valid above ~5 dB.

**A4 failed on both checks and was fixed, not waived:**

- SNR: training `[0,20]` vs evaluation `[−10,20]` — would have **extrapolated**
  at the two lowest points, the regime where the paper claims its largest
  gains. Widened to `[−10,20]`.
- κ: grid `[1.32, 6513]` vs eval `[0.0090, 2131]` — the evaluation minimum sat
  **147× below** the fitted grid, because `grid_lo` was a *training percentile*
  and training percentiles do not bound evaluation. Now anchored on
  `0.1 × observed min`; grid `[0.00086, 8525]` covers both.

**A3:** the original 0.056 max error was the **stopping criterion**, not
capacity — width 32 reaches 0.009999 once max-abs-error is the target. Arm 2
stays at **980 parameters**. All six width × variant combinations pass.

## 2. Pre-registered predictions, as written before the runs

**P1** — arm 2 (~980 params) lands within 0.2 dB of EM-GS, because the measured
paired GS−EM-GS gap is only 0.068 dB, so the classical filter is nearly inert
at RSR 10 dB.

**P2** — consequently essentially all of any full-URformer gain is the
Transformer residual, not the unrolled filter.

**P3** — *my own call:* arm 1a does **not** cleanly clear 2 dB; expected
**0.5–1.5 dB**. Mechanism: the user-token scheme gives the Transformer only
`K=3` tokens, so attention is nearly trivial, while the structure Track B showed
worth +2.85 dB at `N=32` lives across the **32 antennas** — inside the token
dimension, reachable only by dense projections, never by attention.

## 3. Training, per arm

| arm | params | best val | best epoch | runtime | final train/val gap |
|---|---:|---:|---:|---:|---:|
| 1a full, warm start | 1,586,900 | −5.571 dB | **8** / 50 | 69 min | 3.5 dB |
| 1b full, random | 1,586,900 | −5.458 dB | **20** / 50 | 72 min | 4.0 dB |
| 2 filter-only | **980** | −4.869 dB | **44** / 50 | 30 min | **none** |

- No early stopping; fixed 50-epoch budget for every arm.
- Checkpoint selection on **validation ratio-of-sums NMSE only**.
- Test set touched **exactly once**, after all training.

**Both full arms overfit badly.** Arm 1a peaked at epoch 8 and ended ~0.9 dB
*worse* on validation than its epoch-0 model; 42 of 50 epochs were wasted. Arm
1b peaked at 20 and ended with a 4.0 dB gap. **Arm 2 did not overfit at all** —
it finished with val (−4.87) *better* than train (−4.78), which is what 980
parameters on 20,000 samples should do, and confirms the full arms' overfitting
is a capacity problem, not a data-generation bug.

This was foreseen and not acted on: `trackD_phase2.md` §17 called the ~1.2:1
sample-to-parameter ratio "thin", said "overfitting is a live risk, not a
hypothetical", and recommended a 40,000-sample control that was never run.

**Gate behaviour is mechanistically coherent:**

| arm | α final | reading |
|---|---:|---|
| 1a (correct Bessel filter) | 0.214 | leans *on* the filter |
| 1b (random filter) | 0.115 | *suppresses* it, staying near plain GS |
| 2 (no Transformer) | **0.521** | leans hardest — it is the only tool left |

## 4. Test set — 2000 paired trials, every method on identical realizations

| method | ratio-of-sums | median |
|---|---:|---:|
| GS (spectral) | +1.454 | −7.193 |
| **EM-GS (spectral)** | −0.604 | **−7.448** |
| linearised LS | −1.986 | −6.891 |
| oracle phase | −5.247 | −11.582 |
| arm 1a full, warm start | −5.500 | −8.416 |
| **arm 1b full, random** | −5.347 | **−8.951** |
| arm 2 filter-only | −4.824 | −7.665 |

## 5. Paired difference vs EM-GS-spectral (negative = better)

| method | median | CI95 | win rate | p5 | p95 |
|---|---:|---|---:|---:|---:|
| arm 1a | −0.979 | [−1.103, −0.799] | 66.2% | −6.86 | +3.19 |
| **arm 1b** | **−1.434** | **[−1.658, −1.273]** | **71.5%** | −6.40 | +2.89 |
| arm 2 | −0.193 | [−0.310, −0.073] | 53.9% | −6.42 | +5.17 |
| GS | +0.230 | [+0.187, +0.256] | 9.2% | −0.01 | +2.65 |
| linearised LS | +0.545 | [+0.360, +0.690] | 43.2% | −2.34 | +9.09 |
| oracle phase | −4.183 | [−4.218, −4.135] | 100.0% | −5.56 | −3.02 |

Every CI excludes zero. The URformer arms are **genuinely better than EM-GS** —
just not by 2 dB.

Note the spread: arm 1b's p5–p95 runs −6.40 to +2.89 dB. It wins 71.5% of
trials and **loses on 28.5%**. The oracle, by contrast, wins 100% with a tight
[−5.56, −3.02] band. The learned arms are high-variance; the classical
comparison is not.

## 6. Did the predictions hold?

| | prediction | measured | held? |
|---|---|---|---|
| **P1** | arm 2 within 0.2 dB of EM-GS | **0.193 dB** | **HELD** (by 0.007 dB) |
| **P2** | gain is the Transformer, not the filter | **80%** is the Transformer | **HELD** |
| **P3** | arm 1a misses 2 dB; 0.5–1.5 dB expected | **0.979 dB** | **HELD**, inside the range |

P1 held by 7 thousandths of a dB. I am not going to dress that up as a
successful forecast of the *value* — the mechanism (the classical filter has
almost no headroom at RSR 10 dB, measured as a 0.068 dB paired GS−EM-GS gap)
was right, and the number landing at 0.193 against a 0.2 threshold is partly
luck.

## 7. Attribution — what 1.586M parameters bought

```
gated filter alone (980 params)     0.193 dB
+ Transformer (+1,585,920 params)   0.786 dB
                                    ─────────
full URformer                       0.979 dB      (of 4.183 dB available)
```

**80% of the gain is the Transformer**, exactly as P2 predicted. But the whole
model captures only **23% of the oracle-phase headroom**, and the marginal
return on the Transformer is **0.786 dB for 1,619× the parameters**.

## 8. The result I did not predict: random init beats the warm start

| | validation | test (paired median) |
|---|---:|---:|
| arm 1a warm start | **−5.571** (better) | −0.979 |
| arm 1b random | −5.458 | **−1.434** (better) |

**The ranking reverses between validation and test**, and arm 1b — the
*paper-faithful control*, framed in B1 as the weaker arm — beats arm 1a, the
"method's best shot", by **0.455 dB**.

I will not over-explain this from one run. Two candidate mechanisms, both
testable, neither established:

1. **Selection noise.** Arm 1a's checkpoint was chosen at epoch 8 from a
   2,000-sample validation set on a curve that was already turning over. A
   validation optimum that early is poorly determined.
2. **The warm start is a worse basin.** Seeding FilterNet at the exact Bessel
   ratio may anchor the layer near classical EM-GS and reduce the diversity the
   Transformer can exploit — arm 1a's α rose to 0.214 while arm 1b's fell to
   0.115, i.e. the two found genuinely different solutions.

What it does establish: **the newly-wired `filter_init` flag materially changes
the trained result** — which is what B1 said this pair was for. It just changed
it in the opposite direction to the one the naming assumed.

## 9. Where this leaves the paper's claim

The paper reports ≈ −20 dB at `P=15, SNR=5`. Ours: EM-GS **−5.59 dB**, with the
**oracle-phase bound at −10.98 dB** at that operating point.

**−20 dB is below the ceiling for any perfect-phase-plus-LS estimator in our
configuration**, so the paper cannot be running our operating point. The most
likely cause is reference power: our RSR is 10 dB in the single-user convention
= **5.23 dB** in the paper's multi-user one, and weaker reference means worse
phase recovery. This is not a defect on either side; the configurations differ.

### A scope correction to the A2 bound

On **ratio-of-sums**, arm 1a (−5.500 dB) edges *past* the oracle line
(−5.247 dB). That is not a contradiction. The oracle quantity is
`G + LS(W,S)` — the ceiling for perfect phase recovery **followed by
unstructured LS**. A learned estimator can beat it by exploiting the `L_k ≤ 7`
structural prior that LS discards, and in the aggregate it does, because
ratio-of-sums is dominated by low-SNR trials where LS noise amplification is
worst and a prior helps most.

On the **median** trial the oracle remains far ahead (−11.58 vs −8.42), and on
the pre-registered *paired median* it wins 100% of trials. So A2's line bounds
the classical pipeline, not every estimator, and the prompt's description of it
as "the exact ceiling for any magnitude-only estimator" is too strong. The
criterion was specified on paired medians, where the bound holds.

## 10. Runtime and the revised matrix

| | measured |
|---|---:|
| arm 1a / 1b (1.59M params) | 69 / 72 min |
| arm 2 (980 params) | 30 min |
| test evaluation, 2000 paired trials | ~4 min |
| **stage 1 total** | **~2.9 h** |

Revised full-matrix estimate: **~12 trainings ≈ 13–14 h** at ~70 min each for
full-size arms, against the 11.7 h projected in phase 2 — the earlier figure
under-counted because it assumed 1.11 h from a 200-sample extrapolation.

## 11. Recommendation

**Do not proceed to the full 12-model matrix as specified.** Two reasons:

1. **The reference arm is data-limited, not architecture-limited.** Both full
   arms overfit; their reported checkpoints come from epochs 8 and 20 of 50.
   Sweeping `N` with an under-trained model measures the training budget, not
   the architecture.
2. **The measured effect is 1.4 dB against a 2 dB bar, with 28.5% of trials
   lost.** That is not a foundation for a 13-hour sweep.

Cheapest decisive next steps, in order:

| | experiment | cost | what it settles |
|---|---|---:|---|
| 1 | arm 1b at **40,000–80,000** samples | ~2–4 h | separates "architecture doesn't transfer" from "20k is too few". Data costs 1.27 ms/sample — this is the cheapest information available. |
| 2 | **antenna-token** variant at `N=32` | ~1.2 h | tests P3's mechanism directly. If attention over 32 antennas beats attention over 3 users, the diagnosis is confirmed and HS-URformer is well-motivated. |
| 3 | `d_model`/`L_enc` reduction | ~1 h | if the Transformer is overfitting, a smaller one may generalize better for the same 0.79 dB. |

Only after those does the `N` sweep answer a question worth 13 hours.

**This also sharpens the HS-URformer case, exactly as pre-registered.** The
Transformer bought 0.79 dB attending over 3 user tokens; the Hankel projection
acts along the 32-antenna axis the architecture cannot see, and Track B measured
+2.85 dB there at this same `N`. That reasoning was recorded at `54655f2` before
any of these numbers existed.

---

## Appendix — reproduction

```
Part A       PYTHONPATH=. python3 scratch/trackD_partA_probe.py
             PYTHONPATH=. python3 scratch/trackD_partA4_refit.py
gates        PYTHONPATH=. python3 -m trackD_urformer.verify
stage 1      PYTHONPATH=. python3 -m trackD_urformer.stage1 --i-have-approval
figures      PYTHONPATH=. python3 scratch/trackD_stage1_plots.py
```

Config: `N=32`, `K=3` (ours, not the paper's 4), `L_k ~ U{3..7}`, `P=20`,
RSR 10 dB ours = 5.23 dB paper, spectral init throughout, SNR `U[−10,20]`,
20k/2k/2k on disjoint seed ranges, 50 epochs, Adam + cosine, float32.

Tests **410 passed, 1 skipped**. Gates **15/15**.

**Not done, by instruction:** D3, the full matrix, weight-tied ablation,
antenna-token variant, closed-box Transformer, anything Hankel, and no change
to the RSR decision.
