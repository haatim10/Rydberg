# PROMPT 21 Part B — batch C4 scored against P31

**Every verdict in this document is a RE-REGISTRATION.** The invalid
`random`-init arm already reported `+0.343` (log − balanced), `+2.228`
(log − unbalanced), a span of `3.32` and a share below 5 dB of `0.359`, and the
predictions in `reports/p21/PREREG_P31.md` (committed `c4cbc56`, standing
alone, before C4 started) were written knowing all four.

Source: `reports/p21/p31_score.json`, produced by `scratch/p21_score_c4.py`
(committed `ed218b4`, **before batch C4 finished**).

---

## B4.0 The runs

| run | selected epoch | val at selection | seconds |
|---|---|---|---|
| `C4_U1log_seed1` | 7 | −6.627 dB | 10267 |
| `C4_U1log_seed2` | 8 | −6.728 dB | 9153 |
| `C4_U1log_seed3` | 7 | −6.787 dB | 9581 |

Against bin-reweighted partners `C1_seed{1,2,3}` at epochs 8, 8, 6. Neither
side is stopped systematically earlier. All nine arms in the scoring pass
report `init='spectral'`. [FACT]

## B4.1 Path check

Neither target contrast has a published value on valid arms, so the check is
indirect: the pass reproduces the **published P27 contrast** `U1 − C1`, which
shares the worlds, the path, the bin edges and **both** comparison arms with
the targets.

| seed | published | measured | error |
|---|---|---|---|
| 1 | +1.9024 | +1.9024 | 3.0e−5 |
| 2 | +2.2427 | +2.2427 | 4.7e−5 |
| 3 | +2.3691 | +2.3691 | 3.5e−5 |

**Reproduced.** [FACT]

---

## B4.2 The result

`*` marks a bootstrap CI over test realisations excluding zero.

### log − balanced (positive = the log loss is better)

| seed | SNR ≥ 5 | CI over **test realisations** | ratio-of-sums | pooled median |
|---|---|---|---|---|
| 1 | **+0.2278** * | [+0.183, +0.261] | +0.096 | −0.014 |
| 2 | **−0.0252** | [−0.065, +0.016] | −0.125 | −0.101 |
| 3 | **+0.1595** * | [+0.117, +0.230] | +0.135 | +0.032 |

| across-seed quantity | value |
|---|---|
| three-seed mean | **+0.1207** dB |
| SD **over seeds** | **0.1309** dB |
| `mean − 2·SD` | **−0.1410** dB |

### log − unbalanced (positive = the log loss is better)

| seed | SNR ≥ 5 | ratio-of-sums |
|---|---|---|
| 1 | +2.1425 | +1.708 |
| 2 | +2.1882 | +1.803 |
| 3 | +2.5391 | +2.073 |
| **mean** | **+2.2899** | |

Both fixes beat the conventional loss by a large, stable margin. Against each
other they are a wash.

### Verdicts

**P31a(i) HELD** (re-registration). Across-seed SD `0.1309` against a falsifier
of `0.35`; point estimate `0.15`.

**P31a(ii) HELD** (re-registration). Three-seed mean `+0.1207` against an
equivalence margin of `±0.50`; point estimate `+0.15`. Both parts of P31a
landed close to their point estimates, which is worth stating because P30b's
point estimate missed by `0.43` dB and was reported as such.

**P31b HELD** (re-registration). The log loss's gradient-share span is
`3.589` — it does **not** fall below the bin-reweighted arm's `1.9`. Point
estimate was `2.8`; measured `3.589`.

**P31c′ FAILED** (re-registration), **and it failed backwards.** Predicted the
log-over-balanced margin larger at low SNR than at high. Measured
`low − high` = `−0.640`, `−0.201`, `−0.299` — negative on every seed. The
advantage sits at *high* SNR, the opposite end from the prediction.

**P31c UNSCOREABLE.** No prior-carrying log arm exists; see
`PREREG_P31` §D1.

---

## B4.3 Why P31c′ failed: the premise was wrong, and the data says how

P31c′ reasoned that bin reweighting **over-corrects** — realised share below
5 dB `0.427` against an ideal `0.5` — so a log loss that divides the scale out
exactly should be relatively better at low SNR.

The realised shares say the log loss over-corrects **further in the same
direction**, not less:

| loss | share below 5 dB | span (max/min) |
|---|---|---|
| ideal | 0.500 | 1.0 |
| conventional | 0.897 | 30.0 |
| bin-reweighted | 0.427 | **1.9** |
| **log-domain** (3 seeds) | **0.317 – 0.329** | **3.37 – 3.87** |

The log loss allocates about **32%** of its gradient below 5 dB where the ideal
is 50% and bin reweighting manages 43%. It over-serves the high-SNR end more
than bin reweighting does, which is exactly why its advantage shows up there.
One wrong premise explains both P31b's margin and P31c′'s sign. [FACT]

### A control worth keeping

The same log-trained models, evaluated under the **conventional** loss, give a
share below 5 dB of `0.906 – 0.911` and a span of `40.2 – 42.2`.

So the imbalance is a property of the **loss**, not of the trained weights: a
model trained to be well-balanced still shows 91% of gradient below 5 dB the
moment it is scored with the conventional objective. That is the cleanest
statement of the mechanism in the project, and it cost one extra pass. [FACT]

---

## B4.4 Per-bin, log − balanced

| bin (dB) | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| [−10, −5) | −0.115 * | −0.064 * | −0.040 * |
| [−5, 0) | −0.163 * | −0.185 * | −0.056 * |
| [0, 5) | −0.151 * | −0.238 * | +0.002 |
| [5, 10) | −0.046 | −0.187 * | +0.091 * |
| [10, 15) | +0.240 * | −0.020 | +0.200 * |
| [15, 20) | +0.525 * | +0.137 * | +0.259 * |
| **signs** | `- - - - + +` | `- - - - - +` | `- - + + + +` |

Three different sign sequences across three seeds of one design — the loosest
per-bin agreement any batch has produced. Every one still places its negatives
before its positives.

**This is a different quantity from Δ_H** and must not be folded into the
universal-ordering claim, which is about the structural prior. It is an
independent instance of the same shape on a loss-design contrast, and is worth
one sentence, not a section.

---

## B4.5 The decision branch, as fixed before the run

`|m| = 0.121 ≤ 0.50` and `m − 2s = −0.141 ≤ 0` → **EQUIVALENT**.

> The two are equivalent within the registered margin. Paper 2 reports bin
> reweighting as adequate and the log loss as **a simpler alternative that is
> no better** — in those words — and the title claims an attribution finding
> with a replication attached.

**Paper 2 does not get a fix that improves on published work.** The log loss
needs no bin edges, no pre-training measurement pass and no weight estimation,
and it lands within `0.12` dB of the scheme that needs all three. That is worth
reporting as a practical simplification; it is not a delta on
`wiesmayr2023bler` and the title must not imply one.

---

## What this settles for the manuscript

The five stubbed sections are now writable. What they must say:

- the title claims an attribution finding **with a replication attached**;
- the log loss is *a simpler alternative that is no better*, in those words;
- both fixes beat the conventional loss by ~2.2–2.3 dB, and each other by
  nothing;
- the loss-imbalance mechanism is `wiesmayr2023bler`'s and is cited, with the
  91%-under-conventional-scoring control as this project's own demonstration
  that the imbalance lives in the loss rather than the weights;
- P31c is not answered, and the manuscript must not imply a fourth design
  tested the sign sequence.
