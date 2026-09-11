# PROMPT 19 Part B — batch C2 scored against P29

Source: `reports/p19/p29_score.json`, produced by `scratch/p19_score_c2.py`
(committed `d8a4b11`, **before batch C2 finished**) with every threshold and
every outcome trigger transcribed from `reports/p17/PREREG_P29.md` (committed
`157eca6`, standing alone, before C2 started).

---

## B2.0 The runs

| run | selected epoch | val at selection | seconds |
|---|---|---|---|
| `C2_U1_seed2` | 6 | −6.530 dB | 9526 |
| `C2_U1_seed3` | 6 | −6.557 dB | 10606 |
| `C2_H1_seed2` | 6 | −6.352 dB | 12363 |
| `C2_H1_seed3` | 6 | −6.410 dB | 12191 |

Seed 1 is the existing stage-2 / stage-3 pair and was not retrained. All six
checkpoints report `init='spectral'` from their own configs; the scorer refuses
to produce a verdict otherwise. [FACT]

All four selected epoch 6, as did both published seed-1 arms. The mixed-SNR
validation curve is flat: seven epochs sat within one standard error of the
best, against four in the focused design.

---

## B2.1 Scorer correctness check

**Seed 1 reproduces `+1.2088` dB against the published `+1.209`, absolute error
`2.2e-4`.** Its pooled-over-everything figure reproduces `+0.1289` exactly, and
its six per-bin values reproduce to the published precision. [FACT]

All 2000 test-world SNR draws were checked element-wise against the stored
stage-3 draws before the batch launched and are identical. Everything below is
read through the path that produced the published number.

---

## B2.2 The three seeds

Contrast is `U1` (no prior) − `H1` (Hankel prior, rank 7); **positive means the
structural prior is better**. `*` marks a bootstrap CI over test realisations
excluding zero.

| seed | Δ_H at SNR ≥ 5 | CI over **test realisations** | ratio-of-sums, SNR ≥ 5 | pooled median | pooled ratio-of-sums |
|---|---|---|---|---|---|
| 1 | **+1.2088** * | [+1.141, +1.328] | **+0.723** | +0.1289 | −0.214 |
| 2 | **+1.3514** * | [+1.237, +1.454] | **+0.856** | +0.2247 | −0.146 |
| 3 | **+1.2984** * | [+1.221, +1.412] | **+0.901** | +0.1956 | −0.141 |

| across-seed quantity | value |
|---|---|
| three-seed mean | **+1.2862** dB |
| SD **over seeds** | **0.0721** dB |
| range **over seeds** | **0.1426** dB |
| `mean − 2·SD` | **+1.1420** dB |

Per-seed intervals are bootstrap CIs over 2000 paired test realisations; the SD
and range are over seeds, on two degrees of freedom.

### Verdicts

**P29a HELD.** Across-seed SD `0.0721` against a falsifier of `0.45`; the
pre-registered point estimate was `0.20`. The spread is a quarter of the
predicted value — this contrast is far more reproducible than either anchor
suggested.

**P29b FAILED — and it failed in the direction that strengthens the result.**
`+1.209` is the **smallest** of the three, not the middle. The published
single-seed number was a conservative draw, not a favourable one. [FACT]

**P29c HELD.** All three seeds positive, minimum `+1.2088`, every CI excluding
zero.

**P29d HELD, on all three seeds independently.** Every seed's per-bin sign
sequence is exactly `- - - + + +`: the sign flips once, between `[0,5)` and
`[5,10)`, and **all eighteen per-bin CIs exclude zero**. [FACT]

| bin (dB) | seed 1 | seed 2 | seed 3 | 3-seed mean | SD over seeds |
|---|---|---|---|---|---|
| [−10, −5) | −0.111 * | −0.160 * | −0.090 * | **−0.120** | 0.036 |
| [−5, 0) | −0.506 * | −0.300 * | −0.394 * | **−0.400** | 0.103 |
| [0, 5) | −0.451 * | −0.138 * | −0.291 * | **−0.293** | 0.157 |
| [5, 10) | +0.398 * | +0.481 * | +0.493 * | **+0.457** | 0.052 |
| [10, 15) | +1.305 * | +1.425 * | +1.479 * | **+1.403** | 0.089 |
| [15, 20) | +2.226 * | +2.383 * | +2.262 * | **+2.290** | 0.082 |

---

## B2.3 Which outcome fired

**Outcome (a)**, on the rule fixed before the run and evaluated in its stated
order. The three `(b)` tests are checked first and none fires: the minimum seed
is `+1.2088 > 0`; `mean − 2·SD = +1.142 > 0`; `SD = 0.072` against
`0.40 × mean = 0.514`. Then `(a)` fires because `mean = 1.286 ≥ 0.90`.

> **(a)** The collapse claim survives in corrected and stronger form: from a
> stable, substantial gain to a quantity with no stable sign. This is the best
> outcome and the easiest to write.

It is worth being precise about how much stronger, because the retired claim
was wrong about *what kind of thing* it had measured, not only about its size.

| | mixed-SNR training | matched focused training |
|---|---|---|
| three-seed mean at SNR ≥ 5 | **+1.286** dB | **+0.020** dB |
| SD over seeds | 0.072 | 0.083 |
| `mean − 2·SD` | **+1.142** | **−0.147** |
| sign, per seed | +, +, + | +, +, **−** |
| sign under ratio-of-sums at SNR ≥ 5 | +, +, + | **−, −, −** |
| per-bin CIs excluding zero | 18 of 18 | 8 of 9 |

**The mixed-SNR gain is real, stable and positive under both pooling rules.**
The focused gain is none of those things. The paper's claim is therefore not
that a number shrank by a factor, but that **the same contrast changes category
when both arms are trained adequately in the regime of interest**: from an
effect that survives every check to one whose sign depends on which seed was
drawn and which statistic was used.

---

## B2.4 Two things the numbers settle that were previously arguable

**1. The published `+1.209` was conservative.** P29b's failure is the evidence:
seed 1 is the minimum of its three. Nothing in the repository was inflated by a
favourable draw; if anything the headline understated the effect by `0.077` dB.
[FACT]

**2. The C1c account of the pooling disagreement is confirmed on a second
design.** At SNR ≥ 5 the ratio of sums is positive on all three mixed seeds
(`+0.723`, `+0.856`, `+0.901`); pooled over the whole range it is negative on
all three (`−0.214`, `−0.146`, `−0.141`). The only difference between those two
columns is whether the three negative low-SNR bins are included. That is exactly
the mechanism C1c measured on the focused design: the sum of squared errors is
dominated by low-SNR realisations, which is where the prior costs accuracy.
[FACT]

The disagreement is therefore not a property of the focused design, nor of any
one seed. It is a property of the per-bin sign structure, which is now
established on six arms across two designs and four seeds.

---

## Gate

`PREREG_P29` gates C3 on this report. The gate is now discharged: **C2 has
reported, and C3 may launch once P30 is committed standing alone.**

---

## What this changes for the manuscript

The structural-prior section, the abstract and the title were stubbed under
PROMPT 19 E2 precisely because the three outcomes were three different papers.
Outcome (a) selects the first: the paper keeps structural priors as its
subject, and its claim is that their apparent in-distribution value is
contingent on training adequacy — supported now by three seeds on each side
rather than one, and by a per-bin pattern that replicates exactly.

Those sections still do not get written from this report alone. C3 tests
whether the per-bin ordering survives a third training design, and PROMPT 19
E2's stubs name C3 as well as C2.
