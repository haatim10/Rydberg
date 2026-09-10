# PROMPT 17 Part C — batch C1 scored against P28

Source: `reports/p17/p28_score.json`, produced by `scratch/p17_score_c1.py`
(committed `b9025cc`, **before batch C1 finished**) with every threshold
transcribed from `reports/p16/PREREG_P28.md` (committed `e95c7ec`, standing
alone, before C1 started).

---

## C1.0 The runs

Four runs, one driver, 80k / 13 epochs, `N = 32`, `K = 3`, `L_k ~ U{3..7}`,
`P = 20`, RSR 10 dB, training and test SNR both in `[5, 20]`.

| run | selected epoch | val at selection | seconds |
|---|---|---|---|
| `C1_U1_seed2` | 9 | −17.146 dB | 5066 |
| `C1_U1_seed3` | 9 | −17.208 dB | 5072 |
| `C1_H1_seed2` | 8 | −17.098 dB | 6623 |
| `C1_H1_seed3` | 8 | −17.016 dB | 6706 |

Seed 1 is the existing stage-4 pair and was not retrained. Every one of the six
checkpoints reports `init='spectral'` from its own config, read rather than
assumed; the scorer refuses to produce a verdict otherwise. [FACT]

---

## C1.1 Scorer correctness check

**Seed 1 reproduces `+0.0778` dB against the published `+0.0778`, absolute
error `4e-6`.** [FACT]

The evaluation path mirrors `stage4.py`'s Part C: test SNR `[5, 20]`,
`n_test = 2000`, `P = 20`, spectral initialiser, `by_bin` with the same edges
and sign convention. All 2000 test-world SNR draws were checked element-wise
against the stored stage-4 draws before the batch launched and are identical.

Everything below is therefore read through the same path that produced the
published number, and the verdicts stand or fall on their own.

---

## C1.2 The three seeds

Contrast is `U1` (no prior) − `H1` (Hankel prior, rank 7); **positive means the
structural prior is better**. `*` marks a bootstrap CI over test realisations
that excludes zero.

| seed | Δ_H at SNR ≥ 5 (dB) | CI over **test realisations** | ratio-of-sums (secondary) |
|---|---|---|---|
| 1 | **+0.0778** * | [+0.050, +0.112] | −0.059 |
| 2 | **+0.0568** * | [+0.015, +0.097] | −0.077 |
| 3 | **−0.0755** * | [−0.112, −0.034] | −0.234 |

| across-seed quantity | value |
|---|---|
| three-seed mean | **+0.0197** dB |
| SD **over seeds** | **0.0831** dB |
| range **over seeds** | **0.1533** dB |
| `mean − 2·SD` | **−0.1465** dB |

**These are two different quantities and are never pooled.** The per-seed
intervals are bootstrap CIs over the 2000 paired test realisations; the SD and
range are over seeds. Three seeds give two degrees of freedom, so the SD
carries large sampling uncertainty and any statement made from it must say it
was evaluated on three seeds.

### Verdicts

**P28a HELD.** Across-seed SD `0.0831` against a falsifier of `0.35`; the
pre-registered point estimate was `0.18`. The seed spread is *smaller* than
predicted, not larger.

**P28b FAILED.** `+0.078` is the **largest** of the three seed values, not the
middle. Under exchangeability that outcome had probability 1/3, and it was
registered in advance as the expected-to-fail test precisely because a
single-seed number quoted as a headline is a claim that it is typical. It is
not typical: it is the maximum of its own small sample. [FACT]

**P28c FAILED.** Seed 3 gives **−0.0755 dB**, with a CI over test realisations
that excludes zero. The sign of Δ_H is not stable across seeds. [FACT]

### Decision rule — as fixed before the run

`mean − 2·SD = −0.1465 ≤ 0`. Under the rule in `PREREG_P28.md`:

> Paper 2 must then say that the focused-training advantage is **within seed
> variation and cannot be distinguished from zero at this sample size** — not
> that it is zero, which three seeds cannot establish either.

That is the sentence Paper 2 gets. Note also which branch did *not* fire: P28a
held, so seed variance does **not** dominate the contrast, and the clause that
would have forced every single-seed Δ_H in the repository — `+1.209` included —
to be reported as a single draw is not triggered by this batch. Whether
`+1.209` is a single draw is what C2 measures, and this batch says nothing
about it.

### Gate on C2–C4

**PROCEED.** The gate stops the remaining batches if P28a fails or the
three-seed range exceeds 0.5 dB. The SD is 0.083 and the range is 0.153, so
neither condition is met. C2 may launch once P29 is committed standing alone.

---

## C1.3 The finding: the prior has no stable sign, and it is not a noise problem

Per-bin contrast, `U1 − H1`, positive = the prior is better. `*` marks a
bootstrap CI over test realisations that excludes zero.

| bin (dB) | seed 1 | seed 2 | seed 3 | 3-seed mean | SD over seeds |
|---|---|---|---|---|---|
| [5, 10) | −0.076 * | −0.142 * | −0.301 * | **−0.173** | 0.116 |
| [10, 15) | +0.130 * | +0.086 * | −0.016 | **+0.067** | 0.075 |
| [15, 20) | +0.232 * | +0.327 * | +0.218 * | **+0.259** | 0.059 |

Three things follow, and the third is the one that matters.

**1. The prior hurts at 5–10 dB on every seed.** Three of three negative, all
three CIs excluding zero. This is not a wash and not noise; it is a consistent
cost. [FACT]

**2. The prior helps at 15–20 dB on every seed.** Three of three positive, all
three CIs excluding zero. [FACT]

**3. The pooled figure over `[5, 20]` is therefore a weighted average of a
consistent penalty and a consistent benefit, and its sign is decided by which
of the two the pooling and the seed happen to favour.** Under the paired
per-trial median it is positive on two seeds and negative on the third; under
ratio-of-sums it is **negative on all three** (−0.059, −0.077, −0.234). [FACT]

So `+0.078` was never a small positive effect. It was the residue of two larger
opposing effects, and the residue does not keep its sign under either a change
of seed or a change of pooling rule. The seed spread being *small* (P28a held
comfortably) is what makes this a statement about the effect rather than about
the noise: these are tight, mutually inconsistent measurements, not a wide
distribution straddling zero. [FACT]

**Offered as a reading, not established here:** the Hankel projection buys
denoising, which is worth most where the noise dominates the error — and costs
model bias, which is worth least where the residual error is already
model-limited. That would put the benefit at high SNR and the cost at low SNR,
which is the opposite of the ordering measured. The measured ordering instead
suggests the projection's rank-7 truncation is discarding signal that matters
most when the observation is *good* enough to have resolved it — but nothing in
this batch tests that, and it should not be written into a paper without an
experiment that does. [HYP]

---

## What is settled and what is not

**Settled by this batch.** The scorer is correct (C1.1). The across-seed SD of
Δ_H under focused training is `0.083` dB, so the contrast is not seed-noisy
(P28a). `+0.078` is the maximum of its three seeds, not a typical value
(P28b). The sign of Δ_H is not stable: one of three seeds is significantly
negative (P28c). The prior consistently hurts below 10 dB and consistently
helps above 15 dB, on every seed.

**Not settled.** Anything about the mixed-SNR `+1.209` dB contrast — that is
C2, and no number here bears on it. The mechanism behind the bin ordering. And
whether the focused design is the right one to draw conclusions from at all,
given that its own note in `reports/trackD_stage4_results.json` calls it a
mechanism probe comparable only to itself.
