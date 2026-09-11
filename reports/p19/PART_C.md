# PROMPT 19 Part C — evaluation-only diagnostics

Source: `reports/p19/partC_diagnostic.json`, produced by
`scratch/p19_partC_diag.py`. No training, no refitting, no new arm. C1a runs
the three focused `H1` checkpoints over the 2000 shared test worlds; C1b reads
files already in the repository; C1c reads the stored stage-4 per-trial rows
and needs no model pass at all.

---

## C1a — there is no abstention analogue. The two papers do not share a mechanism.

**Settled from the checkpoints before anything was measured:** every `H1` arm
carries `hankel_gate='none'`. `URformerLayer.forward` then takes the
`G_lin = proj` branch **unconditionally** — no learned gate, no `σ²`
dependence, no per-trial switch. So an abstention analogue could only come from
the projection itself being inactive on most trials. [FACT]

It is not. Measured as the relative change the projection makes,
`‖proj(G) − G‖_F / ‖G‖_F`, on every one of the 10 unrolled layers of every one
of 2000 trials:

| seed | median change per call | per-trial max, p1 | p50 | p99 | trials with **every** layer < 10⁻² |
|---|---|---|---|---|---|
| 1 | 0.0693 | 0.0928 | 0.1597 | 0.2452 | **0.0000** |
| 2 | 0.0708 | 0.1012 | 0.1618 | 0.2496 | **0.0000** |
| 3 | 0.0721 | 0.1017 | 0.1615 | 0.2492 | **0.0000** |

Not one trial in 2000, on any seed, is even approximately unprojected. The
least-projected trial in the bottom percentile still has a layer moving the
estimate by more than 9%. [FACT]

Paper 1's `N = 8` result is driven by the selector abstaining on **42%** of
trials, which pins the median at zero and leaves the active minority to
dominate a sum of squared errors. Nothing of that kind happens here. **The
resemblance between the two statistic-dependences is coincidental at the level
of mechanism**, and Paper 2 may not borrow Paper 1's explanation. [FACT]

---

## C1c — the disagreement does have an account, and it is the bin structure

The brief anticipated that if C1a came back negative the paper would have to
say it has no account. It does not have to: the paired median weights every
trial equally, the ratio of sums weights each trial by its own squared error,
so the two disagree exactly when the trials carrying the error mass differ
systematically from the typical trial. That is directly measurable on the
stored seed-1 rows.

Trials sorted by baseline (no-prior) NMSE, ascending:

| decile | share of Σ error | median Δ_H (dB) | fraction Δ_H > 0 |
|---|---|---|---|
| 1 | 0.017 | +0.089 | 0.575 |
| 2 | 0.024 | +0.231 | 0.625 |
| 3 | 0.033 | +0.226 | 0.625 |
| 4 | 0.042 | +0.275 | 0.675 |
| 5 | 0.057 | +0.098 | 0.575 |
| 6 | 0.076 | +0.146 | 0.580 |
| 7 | 0.102 | +0.033 | 0.530 |
| 8 | 0.138 | **−0.077** | 0.470 |
| 9 | 0.195 | **−0.055** | 0.470 |
| 10 | 0.317 | **−0.126** | 0.410 |

The **top two deciles carry 51.2% of the total error** and contribute
**−0.134 dB**; the other eight carry 48.8% and contribute essentially nothing.
The ratio of sums is negative because the trials it weights most are the trials
where the prior hurts. [FACT]

**Is that conditioning, or is it SNR re-expressed?** The discriminating test is
whether baseline NMSE still predicts Δ_H *inside* a bin:

| | Spearman(baseline NMSE, Δ_H) |
|---|---|
| overall | **−0.161** (p = 5e−13) |
| overall, against SNR instead | **+0.198** (p = 3e−19) |
| within [5, 10) | −0.009 (p = 0.82) |
| within [10, 15) | −0.046 (p = 0.24) |
| within [15, 20) | +0.147 (p = 0.0002) |

Inside a bin the correlation vanishes, and the one bin where it is significant
carries the **opposite** sign to the conditioning story. So baseline
conditioning adds nothing beyond SNR. [FACT]

**The account, then:** the sum of squared errors is dominated by low-SNR
trials, which is precisely where the prior hurts; the paired median weights all
trials equally and most of them sit at SNRs where the prior helps. The two
statistics disagree because they weight the SNR axis differently, and the
per-bin table already shows that axis flipping sign. This is the same object
seen twice, not two findings — and it is one more reason the reportable object
is the per-bin table rather than any scalar.

---

## C1b — the ordering replicates; the crossing SNR does not

Convention throughout: `U1 − H1`, positive = the arm carrying the structural
prior is better. Bins `[−10,−5) [−5,0) [0,5) [5,10) [10,15) [15,20)`; `.` means
the design does not cover that bin.

| arm | design | init | valid | signs | flip between |
|---|---|---|---|---|---|
| mixed-SNR Δ_H, seed 1 | mixed [−10,20] | spectral | yes | `---+++` | [0,5) → [5,10) |
| focused Δ_H, seed 1 | focused [5,20] | spectral | yes | `...-++` | [5,10) → [10,15) |
| focused Δ_H, seed 2 | focused [5,20] | spectral | yes | `...-++` | [5,10) → [10,15) |
| focused Δ_H, seed 3 | focused [5,20] | spectral | yes | `...--+` | [10,15) → [15,20) |
| Δ_H under the balanced loss (P20) | mixed [−10,20] | **random** | **NO** | `----++` | [5,10) → [10,15) |

**On all five arms every negative bin precedes every positive bin.** That holds
across two initialisers, two training designs, two loss functions and four
seeds — including the invalid `random`-init arm, which is carried here with its
invalid label rather than dropped, because an invalid arm can still witness a
qualitative pattern even when its numbers are unusable. [FACT]

**But the crossing moves across three different bin boundaries.** [FACT]

So the robust object is the *ordering* — the prior costs accuracy at low SNR
and buys it at high SNR, monotonically — and **not** the SNR at which the cost
turns into a benefit, which moves by up to two bins between seeds of the same
design.

**[HYP] Paper 2 should report the per-bin pattern in place of a single number.**
The pattern has survived every arm available; the scalar has survived nothing —
it changes sign with the seed (P28c) and with the pooling rule (C1c). C3 is the
test: if the balanced-loss arms reproduce the ordering under a valid
initialiser, the pattern is established on a third design and the claim is
safe. Until then it is a hypothesis with five supporting observations, one of
which comes from an arm we have declared invalid.

---

## What this changes

1. Paper 2 **may not** explain its median/ratio-of-sums disagreement by analogy
   to Paper 1's abstention. That explanation is measurably wrong here.
2. Paper 2 **can** explain it, by the SNR weighting of the two statistics, and
   the explanation is measured rather than asserted.
3. The reportable object for the structural prior is the per-bin sign pattern,
   pending C3. No scalar in this family has survived a change of seed or a
   change of pooling.
