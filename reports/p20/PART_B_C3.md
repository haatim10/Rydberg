# PROMPT 20 Part B — batch C3 scored against P30

**Every verdict in this document is a RE-REGISTRATION.** The invalid
`random`-init arm already reported `+0.174` for this quantity with sign
sequence `- - - - + +`, and the predictions in `reports/p20/PREREG_P30.md`
(committed `da632bd`, standing alone, before C3 started) were written knowing
that. The test is correspondingly weaker than a blind one and must be quoted
as such.

Source: `reports/p20/p30_score.json`, produced by `scratch/p20_score_c3.py`
(committed `fc0d7d0`, **before batch C3 finished**).

---

## B3.0 The runs

| run | selected epoch | val at selection | seconds |
|---|---|---|---|
| `C3_H1bal_seed1` | 8 | −6.378 dB | 11533 |
| `C3_H1bal_seed2` | 7 | −6.364 dB | 12542 |
| `C3_H1bal_seed3` | 7 | −6.249 dB | 11595 |

Paired against the balanced-loss no-prior arms `results/p16/tier05/C1_seed{1,2,3}`,
which selected epochs 8, 8 and 6. Neither side of the contrast is stopped
systematically earlier. All six checkpoints report `init='spectral'`. [FACT]

## B3.1 Path check

This contrast has no published value, so the check is indirect: the same pass
reproduces the **published P27 contrast** `U1 − C1`, which shares the worlds,
the path, the bin edges and the no-prior arms with the target.

| seed | published | measured | error |
|---|---|---|---|
| 1 | +1.9024 | +1.9024 | 3.0e−5 |
| 2 | +2.2427 | +2.2427 | 4.7e−5 |
| 3 | +2.3691 | +2.3691 | 3.5e−5 |

**Reproduced.** The verdicts below stand on their own. [FACT]

---

## B3.2 The result

Contrast is `C1` (balanced loss, no prior) − `H1bal` (balanced loss, prior);
**positive means the structural prior is better**. `*` marks a bootstrap CI
over test realisations excluding zero.

| seed | Δ_H at SNR ≥ 5 | CI over **test realisations** | ratio-of-sums | pooled median |
|---|---|---|---|---|
| 1 | **−0.0818** * | [−0.168, −0.004] | **−0.388** | −0.326 |
| 2 | **−0.0985** * | [−0.198, −0.005] | **−0.531** | −0.421 |
| 3 | **−0.3496** * | [−0.414, −0.287] | **−0.722** | −0.530 |

| across-seed quantity | value |
|---|---|
| three-seed mean | **−0.1766** dB |
| SD **over seeds** | **0.1500** dB |
| range **over seeds** | 0.2679 dB |
| `mean − 2·SD` | **−0.4767** dB |

**Under the SNR-balanced loss the structural prior is not merely worthless. It
is harmful — on every seed, under both pooling rules, with every confidence
interval excluding zero.** [FACT]

### Verdicts

**P30a HELD** (re-registration). Across-seed SD `0.150` against a falsifier of
`0.35`; point estimate was `0.12`.

**P30b HELD** (re-registration) — **the thesis test**. Three-seed mean
`−0.177` against a falsifier of `+0.60`. It held, and it held in a direction I
did not predict: the point estimate was `+0.25`, corrected *upward* from the
invalid arm's `+0.174` on the amended rule's reasoning that trained arms do
better than instinct allows. That correction was wrong by `0.43` dB, in the
direction of the prior being worse than instinct allowed. The bound was
one-sided so the prediction stands, but the point estimate missed badly and
this document says so.

**P30c FAILED** (re-registration). Predicted `- - - + + +` on every seed.
Measured `- - - - - +` on every seed — and **not** the invalid arm's
`- - - - + +` either. A third distinct pattern, with the crossing one bin
higher again.

---

## B3.3 Three designs, one monotone progression

| design | mean Δ_H, SNR ≥ 5 | SD over seeds | `mean − 2·SD` | sign per seed | ratio-of-sums | per-bin signs |
|---|---|---|---|---|---|---|
| mixed SNR, conventional loss | **+1.286** | 0.072 | +1.142 | +, +, + | +, +, + | `- - - + + +` |
| matched focused `[5,20]` | **+0.020** | 0.083 | −0.147 | +, +, **−** | **−, −, −** | (3 bins only) |
| mixed SNR, **balanced loss** | **−0.177** | 0.150 | −0.477 | **−, −, −** | **−, −, −** | `- - - - - +` |

**This is the paper's central table.** The same prior, the same architecture,
the same data budget and the same test set yield `+1.286`, `+0.020` and
`−0.177` dB depending only on how the two arms were trained. Two independent
ways of removing the training imbalance — narrowing the training range, and
reweighting the loss without narrowing it — both dissolve the advantage, and
the second reverses it. [FACT]

The attribution argument therefore holds in its **strong** form. It is not
specific to the focused design, and it is not an artefact of restricting the
operating range: the balanced arm trains and is tested on the full
`[−10, 20]` dB range, exactly as the conventional arm does.

## B3.4 Per bin, under the balanced loss

| bin (dB) | seed 1 | seed 2 | seed 3 | 3-seed mean | SD over seeds |
|---|---|---|---|---|---|
| [−10, −5) | −0.207 * | −0.247 * | −0.247 * | **−0.233** | 0.023 |
| [−5, 0) | −0.541 * | −0.711 * | −0.858 * | **−0.703** | 0.158 |
| [0, 5) | −0.784 * | −0.970 * | −1.076 * | **−0.944** | 0.148 |
| [5, 10) | −0.488 * | −0.728 * | −0.859 * | **−0.692** | 0.188 |
| [10, 15) | −0.090 * | −0.194 * | −0.437 * | **−0.241** | 0.178 |
| [15, 20) | +0.527 * | +0.759 * | +0.415 * | **+0.567** | 0.175 |

**All eighteen cells have CIs excluding zero, and fifteen of the eighteen are
negative.** The prior helps only in the top bin. [FACT]

## B3.5 What P30c's failure actually says

The prediction that failed was the *specific sequence*. The weaker claim it was
standing in for did not fail:

**Every negative bin still precedes every positive bin.** `- - - - - +`
satisfies that, as did `- - - + + +` and `- - - - + +`. That ordering now holds
on **every arm this project has measured** — three training designs, two
losses, two initialisers, four seeds — without exception. [FACT]

What moves is the crossing:

| arm | crossing sits between |
|---|---|
| mixed SNR, conventional loss (3 seeds) | `[0,5)` and `[5,10)` |
| focused (seeds 1, 2) | `[5,10)` and `[10,15)` |
| focused (seed 3) | `[10,15)` and `[15,20)` |
| balanced loss, invalid `random` init | `[5,10)` and `[10,15)` |
| balanced loss, valid (3 seeds) | `[10,15)` and `[15,20)` |

**[HYP] The crossing moves upward as training adequacy improves.** The
conventional arm is the worst-trained and crosses lowest; the balanced arm is
the best-trained and crosses highest. If that is the mechanism, the prior's
benefit is confined to the regime the network has least data to learn from, and
it retreats as training improves. This is a hypothesis with five supporting
arms and no test; nothing here was designed to measure it, and the manuscript
must not state it as a finding.

---

## What this changes

**The structural-prior section, the abstract and the title are now unblocked.**
They were stubbed pending C3, and C3 has reported. What they must say:

- the contrast is `+1.286` / `+0.020` / `−0.177` dB across three training
  designs, three seeds each, all `init='spectral'`;
- the claim is that the prior's apparent value is a property of the training
  design, demonstrated on two independent ways of fixing the imbalance;
- the ordering replicates universally and the crossing does not, so the
  reportable object is the per-bin pattern;
- every one of these is a re-registration where it descends from P30, and the
  invalid arm's `+0.174` must be mentioned as prior information rather than
  hidden.

**What it does not license.** The retired fifteenfold collapse stays retired.
`−0.177` is not "the prior is harmful"; it is "the prior is harmful **under
this loss, at this operating point, on three seeds**". §IV's scope paragraph
already restricts the negative result to `N = 32`, `K = 3`, this data budget
and these pilot counts, and that restriction now carries more weight, not less.

C4 is next, and P31 must be committed standing alone before it.
