# Track D stage 2 — matched-compute data scaling (PROMPT 5 Part C)

Rendered from `reports/trackD_stage2_results.json`, `trackD_partA_diag.json`,
`trackD_stage1_results.json`. Predictions from `trackD_stage2_prereg.md`,
committed at `1feeb06` **before** any stage-2 run.

## Verdict

> **The bar is CLEARED — at 40,000 samples and again at 80,000.**
>
> | budget | median gain vs EM-GS | CI95 | verdict |
> |---|---:|---|---|
> | 20,000 (stage 1) | 1.434 dB | [−1.639, −1.286] | not met |
> | **40,000** | **2.796 dB** | [−3.038, −2.601] | **MET** |
> | **80,000** | **3.345 dB** | [−3.524, −3.101] | **MET** |
>
> The bar never moved. The data budget did.

## I was wrong, and I said in advance what that would mean

**P6 predicted B3 would NOT clear 2 dB, at ~1.8 dB.** Measured: **3.345 dB**.
Wrong on the verdict and badly wrong on the magnitude — I under-predicted by
1.5 dB, nearly double.

The pre-registration (`1feeb06`) named the falsifier explicitly:

> "if B3 clears 2 dB, force 3 is wrong and my P3 mechanism from stage 1 is
> undermined — the dense projections *can* reach the antenna structure given
> enough data. I will say so rather than retreating to 'the ceiling is just
> higher'."

**So: force 3 is wrong. The P3 mechanism is undermined.** My stage-1 story —
that the user-token scheme puts attention on `K=3` tokens while the exploitable
structure lives across 32 antennas, imposing an architectural ceiling — does
not survive. The dense input/output projections evidently *do* reach the
antenna-axis structure, given enough data to learn it. The stage-1 limitation
was **data, not architecture**, and I attributed it to architecture with a
mechanism that sounded right and was not.

What I got right was narrower: **the direction of P4 and P5**, and the
observation that the 20k regime was data-limited — which I flagged in the
stage-1 report and then failed to carry into my own P6 reasoning. The evidence
for "data-limited" was in front of me (4.0 dB train/val gap, best epoch 20 of
50) and I still weighted the architectural story above it.

## Did the predictions hold?

| | prediction | measured | held? |
|---|---|---|---|
| **P4** | paired improvement rises monotonically with data; 40k→80k step smaller than 20k→40k | 1.434 → **2.796** → **3.345**; steps **+1.362** then **+0.549** | **HELD**, both parts |
| **P5** | best-epoch *fraction* moves later with more data | 20k: **0.40** → 40k: **0.48** → 80k: **0.69** | **HELD** |
| **P6** | B3 misses the bar, ~1.8 dB | **3.345 dB, cleared** | **FAILED** |

P4's second clause is the interesting one: the increments do compress
(+1.362 then +0.549), so returns are diminishing — but from a *higher* base
than I expected, and not yet flat.

## Per-run detail

| run | samples | epochs | passes | best ep | **chosen ep** (1-SE) | val (dB) | final train−val gap | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 (stage 1) | 20,000 | 50 | 1.00M | 20 | 20 *(plain best)* | −5.458 | **4.00 dB** | 72 min |
| B2 | 40,000 | 25 | 1.00M | 12 | **10** | −6.317 | **1.80 dB** | 100 min |
| B3 | 80,000 | 13 | 1.04M | 9 | **6** | −6.622 | **0.65 dB** | 110 min |
| B3 filter-only | 80,000 | 13 | 1.04M | 12 | **6** | −4.596 | 0.07 dB | 45 min |

**Overfitting collapses with data: 4.00 → 1.80 → 0.65 dB.** That is the
mechanism behind the whole result, and it is exactly what the stage-1 diagnosis
predicted would happen — I simply did not believe it would be worth 1.9 dB.

**The one-SE rule bound in every run**, selecting epochs 10, 6 and 6 against
plain-best 12, 9 and 12 — earlier and cheaper checkpoints, costing 0.011,
0.085 and 0.085 dB of validation. With 7–8 epochs within 1 SE at each budget,
A4's concern was real and the guard did its job. Note it was applied to stage 2
only, as pre-registered; stage 1's numbers stand under plain best-validation
selection.

## Absolute test NMSE (2000 paired trials, identical realizations)

| method | ratio-of-sums | median |
|---|---:|---:|
| GS (spectral) | +1.454 | −7.193 |
| EM-GS (spectral) | −0.604 | −7.448 |
| linearised LS | −1.986 | −6.891 |
| **unstructured-LS oracle** | **−5.247** | **−11.582** |
| B2 40k | −6.301 | −10.357 |
| **B3 80k** | **−6.583** | **−10.831** |
| B3 filter-only | −4.718 | −7.621 |

## The structural result, now much stronger

At 20k, arm 1a edged past the unstructured-LS oracle on ratio-of-sums by
0.25 dB. At 80k, **B3 beats it by 1.34 dB** (−6.583 vs −5.247), and on the
median it closes to within **0.75 dB** (−10.831 vs −11.582).

The A6 reading holds and strengthens: unstructured LS spends `2NK = 192` free
real parameters where the truth has at most `3·Σ L_k = 63` in an `N=32, K=3`
array with `L_k ~ U{3..7}`. **A learned estimator exceeding that bound is direct
evidence it has recovered the geometric prior LS discards** — and the margin
grows with data, which is what "learning the prior" should look like.

B3 now captures **80% of the 4.183 dB oracle headroom**, against 23% at 20k.

## Attribution at 80k

```
gated filter alone (980 params)      0.147 dB      (was 0.193 at 20k)
+ Transformer (+1,585,920 params)    3.198 dB      (was 0.786 at 20k)
                                     ─────────
full URformer                        3.345 dB      (of 4.183 available)
```

**The Transformer's share rises from 80% to 96%.** Filter-only does not improve
with data — it goes slightly *down*, 0.193 → 0.147 dB — which is the expected
behaviour of a 980-parameter model that was never capacity-limited. Its final
train−val gap is 0.07 dB at 80k.

So P1/P2 from stage 1 are confirmed and sharpened: the gated filter is inert at
this operating point regardless of budget, and essentially all of the gain is
the Transformer. The difference is that the Transformer is now doing far more
with the same parameters, because it finally has the data to fit them.

## Runtime

| | measured |
|---|---:|
| B2 40k / 25 ep | 100 min |
| B3 80k / 13 ep | 110 min |
| B3 filter-only | 45 min |
| test evaluation | ~5 min |
| **total** | **~4.3 h** |

Matched-compute held reasonably: 100 vs 110 min for the two full runs at equal
sample-passes. The 10% excess at 80k is the larger per-epoch `G0` cache
population and validation pass, not extra gradient work.

## Revised recommendation

**Structure — but the case for it has changed, and so has the priority order.**

1. **Push data further before anything else** (~2 h). The curve has not
   flattened: +0.549 dB from 40k→80k. A 160k point at matched compute settles
   whether we are near an asymptote or still climbing, and it is the cheapest
   remaining experiment. Until that is known, every architectural comparison is
   confounded by where each variant sits on its own data curve — which is
   precisely the error I made in stage 1.

2. **Antenna-token variant** (~2 h) — *demoted, and the reason matters*. It was
   my top recommendation when I believed an architectural ceiling existed. That
   belief is now falsified: the user-token model reaches 80% of the oracle
   headroom. The variant is still worth running, but as a *comparison at matched
   data*, not as a fix for a diagnosed defect. Run it only after step 1, at the
   best budget.

3. **HS-URformer** — still the most interesting direction, and the structural
   evidence is *stronger* than in stage 1 (1.34 dB past the unstructured-LS
   oracle, growing with data). But the honest framing has changed: the network
   is already recovering much of the geometry implicitly, so the Hankel
   projection is now a candidate for **efficiency** — reaching the same place
   with less data or fewer parameters — rather than for breaking a ceiling that
   does not exist. That is a weaker claim than the one I made at `54655f2`, and
   it is the one the data supports.

4. **Do not reopen the 12-model matrix yet.** The `N` sweep is now meaningful
   in principle, since the reference arm is no longer data-limited at 80k, but
   it costs ~13 h and would be run at a budget we have not yet shown to be
   converged.

**What I would not do:** claim the paper's method is validated. We cleared *our*
pre-registered bar on *our* channel model at a data budget four times the
paper's stated 20,000. The paper's own configuration remains unreproduced — its
−20 dB at `P=15, SNR=5` still sits below our unstructured-LS oracle of
−10.98 dB at that point, so its operating point differs from ours in a way we
have not identified.

---

## Appendix

```
Part A   PYTHONPATH=. python3 scratch/trackD_partA_diagnostics.py
stage 2  PYTHONPATH=. python3 -m trackD_urformer.stage2 --i-have-approval
figures  PYTHONPATH=. python3 scratch/trackD_stage2_plots.py
```

`N=32`, `K=3`, `L_k ~ U{3..7}`, `P=20`, RSR 10 dB ours = 5.23 dB paper,
spectral init, SNR `U[−10,20]`, val/test fixed at 2000 each on disjoint seed
ranges, `filter_init="random"`, Adam + cosine, float32, one-SE selection.

**Not done, by instruction:** antenna-token variant, weight-tied ablation,
closed-box Transformer, D3, the full matrix, anything Hankel.
