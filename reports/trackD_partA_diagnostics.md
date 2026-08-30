# Track D — stage-1 diagnostics (PROMPT 5 Part A)

Rendered from `reports/trackD_partA_diag.json`. No training. All three stage-1
checkpoints re-evaluated on the **validation** set in exactly the test form.

## A5 — the verdict, first

> **The reversal is (1): an artifact of incommensurable metrics.** It is not a
> genuine effect, not selection noise, and not non-exchangeable splits.

The evidence is unambiguous — the ranking is **consistent across sets and flips
across statistics**:

| statistic | VAL 1a | VAL 1b | winner | TEST 1a | TEST 1b | winner |
|---|---:|---:|:---:|---:|---:|:---:|
| ratio-of-sums (dB) | **−5.571** | −5.458 | **1a** | **−5.500** | −5.347 | **1a** |
| median (dB) | −8.194 | **−8.680** | **1b** | −8.416 | **−8.951** | **1b** |
| paired vs EM-GS (dB) | −1.007 | **−1.536** | **1b** | −0.979 | **−1.434** | **1b** |

Each statistic gives the *same* answer on validation as on test. There is no
val→test reversal to explain.

**What I actually did in the stage-1 report** was quote validation as
**ratio-of-sums** (−5.571 vs −5.458, 1a ahead) and test as **paired median
improvement** (0.979 vs 1.434, 1b ahead), then read the difference as a
val/test reversal. It was a ratio-of-sums-vs-median reversal, visible *within*
either set on its own. My "two candidate mechanisms" in the stage-1 report were
both answers to a question that does not arise.

### The real, and more interesting, finding underneath

The two arms genuinely differ, and the difference has a consistent direction:

- **arm 1a (warm start) wins on ratio-of-sums**, which pools energy and is
  therefore dominated by the hardest, lowest-SNR trials.
- **arm 1b (random) wins on the median**, i.e. on the typical trial.

Both are real: A2's paired per-trial difference excludes zero on **both** sets
with the **same sign** (val +0.285 dB CI [+0.252, +0.316]; test +0.228 dB CI
[+0.200, +0.267], positive = 1a worse), and arm 1a is better on only 34.7% /
36.4% of trials.

So the Bessel warm start buys accuracy on the hard tail at the cost of the
typical trial. That is a coherent mechanism — a filter seeded at the exact
`I₁/I₀` ratio is most useful where `κ` is small and the filter actually bites,
which is the low-SNR tail — and it is a sharper statement than the one I made
before.

**The pre-registered criterion is on the paired median**, so arm 1b remains the
correct arm to carry into stage 2, and the stage-1 verdict is unchanged.

## A1 — validation and test in matched units

Paired improvement vs EM-GS-spectral (negative = better), 2000 trials each:

| arm | VAL median | VAL CI95 | TEST median | TEST CI95 |
|---|---:|---|---:|---|
| 1a full, warm start | −1.007 | [−1.182, −0.874] | −0.979 | [−1.107, −0.797] |
| **1b full, random** | **−1.536** | [−1.698, −1.389] | **−1.434** | [−1.639, −1.286] |
| 2 filter-only (980p) | −0.218 | [−0.326, −0.123] | −0.193 | [−0.310, −0.068] |

Every arm's validation and test CIs overlap heavily. In matched units the two
splits agree to within ~0.1 dB for all three arms.

## A2 — paired arm 1a − arm 1b

| set | median | CI95 | excludes 0 | 1a better on |
|---|---:|---|:---:|---:|
| validation | +0.285 dB | [+0.252, +0.316] | **yes** | 34.7% |
| test | +0.228 dB | [+0.200, +0.267] | **yes** | 36.4% |

**Both CIs exclude zero with the same sign.** Per the prompt's decision rule
this is a genuine effect, not a tie — but it is genuine in the *same direction*
on both sets, so it is not a reversal. Mechanism given above.

## A3 — exchangeability

Untrained classical estimators, absolute median NMSE:

| method | val | test | val − test |
|---|---:|---:|---:|
| GS (spectral) | −6.834 | −7.193 | **+0.360** |
| EM-GS (spectral) | −7.113 | −7.448 | **+0.336** |
| linearised LS | −6.614 | −6.891 | **+0.277** |
| oracle phase | −11.178 | −11.582 | **+0.404** |

**The validation set is uniformly ~0.28–0.40 dB harder than test**, and the
offset is close to constant across four methods that share no code path beyond
the data. So the splits are **not perfectly exchangeable**.

It is not an SNR effect: mean SNR is 4.785 dB (val) vs 4.746 dB (test), a
0.039 dB difference. The offset comes from the channel realizations themselves
— the `L_k` draws, angles and conditioning — at 2000 samples per split.

**Consequence:** any *absolute* val-vs-test comparison carries a ~0.34 dB
handicap against validation. This does **not** explain the apparent reversal
(which is visible within a single set), but it is a second reason to compare
paired-within-set, and it is why the A1 table above is the honest presentation.

## A4 — checkpoint-selection noise

**Limitation, stated plainly.** Stage 1 retained only `best.pt` (selected epoch)
and `checkpoint.pt` (final epoch). Per-epoch weights were not kept, so the exact
requested test — paired validation difference between *every* epoch and the
selected one — **is not computable from existing artifacts.** That is a gap in
my stage-1 harness. Stage 2 now stores per-epoch weights *and* the per-trial
validation NMSE at every epoch (16 KB/epoch), making it exact next time.

What the retained artifacts do support:

| arm | best epoch | selected vs **final**, paired | CI95 | excl 0 | marginal val SE | epochs within 1 SE |
|---|---:|---:|---|:---:|---:|---:|
| 1a warm start | 8 / 50 | −0.183 dB | [−0.217, −0.140] | yes | 0.102 dB | **8 / 50** |
| 1b random | 20 / 50 | −0.397 dB | [−0.433, −0.364] | yes | 0.112 dB | **23 / 50** |

Two readings:

1. **Selection was not merely picking noise at the endpoint** — the selected
   checkpoint is significantly better than the final one in a genuine paired
   test, for both arms.
2. **But the plateau is wide for arm 1b: up to 23 of 50 epochs sit within one
   marginal standard error of the best.** That is an *upper* bound — a paired
   epoch-vs-epoch test would cancel realization variance and give a narrower
   band — but 23/50 is large enough to justify a guard.

**This is why the one-standard-error rule is adopted for stage 2** (earliest
epoch within 1 SE of the best), pre-registered in
`reports/trackD_stage2_prereg.md` before any stage-2 run, and **not**
retro-applied to stage 1.

## A6 — oracle language, corrected

The Prompt-4 wording and my stage-1 report both need fixing. The A2 quantity is:

> **the unstructured-LS oracle** — `G + LS(W,S)`, the ceiling for perfect phase
> recovery **followed by an unstructured least-squares solve**.

It is **not** "the exact ceiling for any magnitude-only estimator." An estimator
that carries a prior over `G` is not bound by it.

### The positive reading — this is a result, not a caveat

**A learned estimator exceeding the unstructured-LS oracle is direct evidence
that the network has recovered the geometric prior that unstructured LS
discards.**

On ratio-of-sums, arm 1a reaches **−5.500 dB** against the oracle's **−5.247 dB**
— it is *past* the bound. Unstructured LS treats the `N×K` channel as `2NK` free
real parameters; the truth has only `3·Σ L_k ≤ 3·21 = 63` degrees of freedom in
an `N=32, K=3` array, because `L_k ~ U{3..7}`. Every parameter LS spends beyond
those is spent fitting noise, and that is exactly the waste a learned estimator
can avoid by having internalised the path structure.

The effect appears in the energy-pooled statistic and not the median precisely
because that is where it should: ratio-of-sums is dominated by the lowest-SNR
trials, where LS noise amplification is worst and a prior is worth the most.

**This is the strongest result in stage 1 for pursuing structure, and it is the
direct line to HS-URformer.** The network is already recovering some of the
`L_k ≤ 7` geometry implicitly, through dense projections that were never aimed
at it. A Hankel projection imposes that same structure *explicitly*, along the
antenna axis — the axis the user-token Transformer cannot attend over — and
Track B measured **+2.85 dB** for doing so at this exact `N=32`. The implicit
recovery observed here is evidence the structure is both present and
exploitable; making it explicit is the obvious next lever.
