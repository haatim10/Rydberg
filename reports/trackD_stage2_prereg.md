# Track D stage 2 — pre-registered predictions

**Written before any stage-2 training run**, committed separately so git history
establishes the timestamp. Not editable once the runs start; outcomes go in
`reports/trackD_stage2_results.json` and the Part C report.

Design (matched compute, ~1M sample-passes each):

| run | samples | epochs | sample-passes | model |
|---|---:|---:|---:|---|
| B1 | 20,000 | 50 | 1,000,000 | *reused: stage-1 arm 1b* |
| B2 | 40,000 | 25 | 1,000,000 | full URformer |
| B3 | 80,000 | 13 | **1,040,000** | full URformer |
| B3-filteronly | 80,000 | 13 | 1,040,000 | filter-only, 980 params |

`filter_init="random"` throughout (the stage-1 test winner). Validation and test
sets **fixed and identical** across all runs — only `n_train` changes, and the
val/test seed ranges are independent of it. Asserted at startup.

**The bar does not move:** ≥ 2 dB median paired improvement over
EM-GS-spectral, bootstrap CI excluding zero.

---

## P4 — test paired improvement increases monotonically with sample count

**Predicted: HOLDS, but weakly**, and I expect the 40k→80k step to be smaller
than the 20k→40k step.

Both full arms overfit at 20k (train/val gaps 3.5 and 4.0 dB; best epochs 8 and
20 of 50), so there is recoverable generalization loss. More data recovers it.
But the recovery is bounded by whatever architectural ceiling exists, so the
increments should compress. If P4 fails outright — a *decrease* with more data
— the most likely cause is the compressed epoch budget (13 epochs at 80k) not
being enough to converge, not data hurting.

## P5 — the best epoch, as a fraction of total epochs, moves later as data grows

**Predicted: HOLDS.**

Overfitting onset is governed by how many times the model has seen each sample,
not by epoch index. At 20k the best was epoch 20/50 = **40%**. At matched
sample-passes the equal-passes point at 80k is epoch 5/13 = 38%; with four times
the data the model should tolerate more passes before memorizing, pushing the
optimum past that, so I expect **> 40%** at both 40k and 80k.

## P6 — does B3 (80k) clear the 2 dB bar?

**My prediction: NO. I expect 1.6–2.0 dB, most likely ~1.8 dB.** Confidence:
moderate (~60%). This is deliberately close to the bar — I think it lands just
under.

The arithmetic first. Stage-1 arm 1b reached **1.434 dB** paired median. Only
**~0.57 dB** more is needed. In absolute terms arm 1b's test median was
−8.951 dB against EM-GS at −7.448 dB and the unstructured-LS oracle at
−11.582 dB, so the model sits 2.63 dB above the oracle line with room to move.

Three forces, in the order I weight them:

1. **Recoverable overfitting loss (helps).** A 4.0 dB train/val gap at 20k is
   large. Quadrupling the data should recover a meaningful part of it — on a
   typical `n^{-α}` generalization scaling with α ≈ 0.3–0.5, a 4× data increase
   buys roughly 0.5–0.9 dB of the excess error. That alone is about what is
   needed, which is why this is close.

2. **Compressed optimization (hurts).** Matched compute means 13 epochs with a
   cosine schedule annealed over 13, not 50. Arm 1b needed 20 epochs — 400k
   sample-passes — to reach its best; at 80k that is 5 epochs, leaving only 8
   more for the anneal. Fewer, coarser LR stages on a non-convex unrolled
   objective is a real cost, and it partly cancels force 1.

3. **The architectural ceiling (binds).** This is why I predict a miss. Arm 2
   showed the gated filter contributes 0.193 dB *regardless* of anything, so all
   improvement must come from the Transformer — and P3's mechanism says the
   Transformer attends over **K=3 user tokens** while the exploitable structure
   lives across the **32 antennas**. Data can recover generalization loss; it
   cannot make attention see an axis it is not pointed at. The
   `L_k ≤ 7`-in-`N=32` structure that Track B monetised for +2.85 dB is reachable
   here only through the dense per-token projections.

So: force 1 gets us to roughly 1.9–2.3 dB, force 2 takes some back, force 3
caps it. I land at **~1.8 dB, just under the bar**.

**What each outcome means:**

| outcome | reading |
|---|---|
| B3 ≥ 2 dB | P6 wrong, and the stage-1 verdict was a **data artifact**. "Not met" becomes "not met at 20k". Reopens the matrix. |
| 1.6–2.0 dB | predicted. Data helps but does not rescue it; the ceiling is architectural. Strongest case for the antenna-token variant. |
| < 1.5 dB, or non-monotone | P4 fails. Either 13 epochs is too few (test by re-running B3 at 50 epochs, ~4.5 h) or the 20k result was optimistic. |

**Falsifier for the mechanism, stated now:** if B3 clears 2 dB, force 3 is wrong
and my P3 mechanism from stage 1 is undermined — the dense projections *can*
reach the antenna structure given enough data. I will say so rather than
retreating to "the ceiling is just higher".

## Selection rule — pre-registered, stage 2 only

**One-standard-error rule:** select the **earliest** epoch whose validation
metric is within 1 SE of the best, where SE is the bootstrap standard error of
the validation metric at the best epoch (2000 resamples).

Implemented as `stage2.SELECTION_RULE = "one_se"` before any run.

**Not retro-applied to stage 1** — that would be tuning against runs already
seen. Stage 1's numbers stand as reported under plain best-validation selection.

Rationale: it is the standard guard against selecting a point on a flat curve
that happened to win by noise, and it biases toward earlier (less overfit)
checkpoints, which is the right direction given stage 1's behaviour. Adopting it
is conditional on A4 showing a non-trivial indistinguishable plateau; if A4
shows the selected epoch is cleanly separated from its neighbours, the rule
degenerates to plain best-validation selection anyway and costs nothing.

## A4 limitation carried forward

Stage 1 retained only `best.pt` and the final `checkpoint.pt`, so the exact A4
test could not be computed from its artifacts. **Stage 2 stores the per-trial
validation NMSE at every epoch** (2000 float64 = 16 KB/epoch) plus per-epoch
weights, making the epoch-vs-epoch paired test exact without guesswork.
