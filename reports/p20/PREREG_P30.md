# PROMPT 20 Part B — pre-registration P30 (a **RE-REGISTRATION**)

**Committed standing alone, before Tier 1 batch C3 starts.**

## This is a re-registration and is correspondingly weaker

The original P20 was scored on arms trained with `init='random'` while the arms
they were compared against were trained with `init='spectral'`. Those arms are
invalid and their numbers may not be used as evidence
(`reports/p16/validity.csv`). **But they are not unseen.** The invalid run
reported, for exactly the quantity C3 measures:

| quantity | invalid `random`-init value |
|---|---|
| Δ_H under the balanced loss, SNR ≥ 5 | **+0.174** dB |
| per-bin | −0.187, −0.474, −0.670, −0.321, +0.129, +0.919 |
| per-bin sign sequence | **`- - - - + +`** |

A prediction written now is made **knowing those numbers**. It is therefore a
weaker test than the original was, and nothing below may be reported as though
it were a blind prediction. Recorded here so that no later reading can credit
it with more than it earned.

The one thing that does mitigate this: the `random`-init arms have already been
shown to be badly predictive of their `spectral` counterparts. On the focused
contrast the invalid arms gave `+0.244` and `+0.632` where the valid ones give
`+0.057` and `−0.076`. So `+0.174` is weak prior information, not a preview.

## The runs

Batch **C3**: `H1` under the balanced loss, three seeds, three runs.

| arm | prior | loss | training SNR | seeds |
|---|---|---|---|---|
| `C3_H1bal_seed{1,2,3}` | Hankel, rank 7 | SNR-balanced | `[-10, 20]` | 1, 2, 3 |

Paired against the balanced-loss **no-prior** arms already in hand,
`results/p16/tier05/C1_seed{1,2,3}`. No valid seed 1 exists for the prior side,
so all three are trained here.

### The pairing is across two drivers, and that was checked rather than assumed

The pairing arms were trained by `scratch/p16_tier05.py`; the C3 arms will be
trained by `scratch/p17_tier1.py`. A driver difference confounded with the arm
under test is exactly what cost PROMPT 15 a turn, so it was settled before this
document was written:

- the two drivers' configs for a balanced no-prior arm at the same seed label
  are **identical in every field** (`to_dict()` comparison, no differences);
- their initial weights are **bitwise identical**, `max|diff| = 0.000e+00`
  across all 480 tensors;
- the C3 arm's config differs from its pairing in **exactly one field**,
  `model.use_hankel`. [FACT]

## What the contrast is for

This is the question PROMPT 15 was built around. The paper's thesis is that a
structural prior's apparent value is contingent on training adequacy. Two
designs now bracket it:

| design | three-seed mean Δ_H at SNR ≥ 5 | SD over seeds |
|---|---|---|
| mixed SNR, conventional loss | **+1.286** dB | 0.072 |
| focused training `[5,20]` | **+0.020** dB | 0.083 |

The balanced loss is the *third* way of removing the training-adequacy defect,
and unlike focused training it removes it without narrowing the operating
range. If the thesis is right, Δ_H under the balanced loss should sit near the
focused value, not the conventional one.

## Predictions

### P30a — across-seed SD of Δ_H under the balanced loss

Three valid anchors now exist and the SD does not scale with the effect:

| contrast | magnitude | SD over seeds |
|---|---|---|
| Tier 0.5 loss design | +2.17 | 0.241 |
| C2 mixed Δ_H | +1.29 | 0.072 |
| C1 focused Δ_H | +0.02 | 0.083 |

**Predicted: the across-seed SD is at most 0.35 dB.** Point estimate
**0.12 dB**. Hedged per the amended standing rule; the band is roughly three
times the largest of the two Δ_H anchors.

**Falsifier:** P30a fails if the across-seed SD exceeds **0.35 dB**.

### P30b — the thesis test

**Predicted: the three-seed mean Δ_H at SNR ≥ 5 under the balanced loss is at
most +0.60 dB.** Point estimate **+0.25 dB**.

`+0.60` is slightly under half of the conventional-loss value `+1.286`. The
prediction is that fixing the loss removes *most* of the prior's apparent
advantage, in the same direction focused training did, without requiring the
operating range to be narrowed. Corrected upward from the invalid arm's
`+0.174` per the amended rule's first clause — trained arms historically do
better than instinct allows — and the threshold is deliberately loose enough
that a genuine residual advantage of a few tenths of a dB still passes.

**Falsifier:** P30b fails if the three-seed mean is **greater than +0.60 dB**.
That outcome would say the balanced loss does *not* dissolve the prior's
advantage, and the paper's attribution argument would have to be restated as
specific to focused training rather than to training adequacy in general.

### P30c — the per-bin sign sequence, which is the sharper test

`- - - + + +` has now held on **every seed of two training designs** — three
focused seeds over their three available bins, three mixed seeds over all six —
with **every** CI over test realisations excluding zero.

The invalid balanced-loss arm instead gave `- - - - + +`, with the crossing one
bin higher. So this prediction has a real tension to resolve, and it is
recorded before the runs rather than after.

**Predicted: all three seeds give `- - - + + +`** — negative in the three bins
below +5 dB, positive in the three at or above it, the sign flipping exactly
once between `[0,5)` and `[+5,+10)`.

**Falsifier:** P30c fails if any seed's sign sequence differs. A bin whose CI
covers zero is scored by its point estimate, and the CI is reported alongside.

**If it fails specifically by reproducing `- - - - + +` on all three seeds**,
that is not noise: it says the crossing SNR moves with the loss design, which
is itself reportable and is the outcome the invalid arm points to.

## Decision rule — fixed before the run

Let `m` be the three-seed mean at SNR ≥ 5 and `s` the across-seed SD.

- **If `m ≤ 0.60` (P30b holds)**, the paper states that removing the training
  imbalance dissolves most of the prior's advantage, and that this holds under
  two independent ways of removing it — narrowing the training range, and
  reweighting the loss without narrowing it. This is the strong form of the
  attribution argument.
- **If `m > 0.60`**, the attribution argument is restated as specific to the
  focused design. The paper then says the prior retains substantial value under
  the balanced loss and that the focused result does not generalise to every
  way of fixing the imbalance. This is a materially weaker paper and it is
  registered here as a live possibility.
- **If `m − 2s > 0` and P30b holds**, the residual advantage is small but real
  and is quoted with its seed spread.
- **If `m − 2s ≤ 0`**, the balanced-loss Δ_H is not separable from seed noise,
  exactly as the focused Δ_H was not.
- **If P30a fails**, seed variance dominates this contrast and every statement
  about it is made on three seeds with the spread quoted.

## Recording rule

P30a, P30b and P30c are scored **held / failed** in the Part B report, stated
plainly either way, and every one of them is labelled a re-registration
wherever it is quoted. Per-seed values are reported **per bin and under both
pooling rules**. Every interval names the quantity it is over: test
realisations or seeds. Three seeds give two degrees of freedom.

## Gate

C4 does not launch until C3 has been scored and P31 committed standing alone.
