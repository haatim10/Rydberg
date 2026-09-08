# PROMPT 16 Part B — pre-registration P27

**Committed standing alone, before any Tier 0.5 run starts.**

## The runs

Four runs, nothing else. Two additional seeds on each arm of the `+1.902` dB
contrast:

| arm | loss | P | seeds to add |
|---|---|---|---|
| `U1_P20_uniform` | per-sample normalised NMSE | 20 | 2, 3 |
| `C1_P20_balanced` | same, per-bin reweighted | 20 | 2, 3 |

80k samples / 13 epochs, `N = 32`, `K = 3`, `L_k ~ U{3..7}`, `P = 20`,
RSR 10 dB, `init="spectral"`, SNR range `[-10, 20]`, matched on data order,
schedule, optimiser and selection rule. Only the seed and the loss change.

**Seed 1 is the existing pair**: `results/track_d/stage2/B3_80k_13ep`
(unbalanced) and `results/track_d/stage5/C1_snr_balanced_P20` (balanced), both
`init='spectral'`, both `seed=20260827`, verified in
`reports/p16/validity.csv`.

**One caveat recorded now rather than discovered later.** Seed 1's unbalanced
arm was trained by `stage2.py` and its balanced arm by `stage5.py`; seeds 2
and 3 will use one driver for both arms. The two drivers use the same
optimiser (Adam, lr 1e-3), the same cosine schedule over 13 epochs, the same
gradient clipping (1.0), the same one-SE epoch selection (`stage5.py:53`
imports it from `stage2.py`), and `weighted_nmse_loss(w=None)` is exactly
`nmse_loss`. They are the same procedure. But if seed 1 turns out to be the
outlier of the three, the driver difference is a candidate explanation and must
be named as one rather than attributed to seed alone.

## Cost

Measured: `C1_snr_balanced_P20` took **9294 s** (`train_seconds`,
`reports/trackD_stage5_results.json`). Four runs ≈ **10.3 CPU-hours**, run four
ways in parallel at `num_threads=1` ≈ **2.6 h elapsed**.

---

## P27a — across-seed SD of the balanced-versus-unbalanced gain at SNR ≥ 5 dB

### The claim at risk

Paper 2's headline is `+1.902` dB, CI `[+1.823, +1.986]` over test
realisations, n = 983 (`reports/trackD_stage5_eval.json`,
`P13_balanced_vs_uniform_P20`). That is one seed. Whether it means anything
depends on a seed spread we have never measured.

### Prediction, hedged

**We have no valid seed-variance estimate for anything.** The only figure we
had, `0.244` dB, came from arms trained with the wrong initialiser and is
withdrawn. The amended standing rule says to hedge predictions about noise,
spread and reproducibility rather than to sharpen them, and the reason it says
so is that P19 was lost by predicting `SD <= 0.12` and measuring `0.244`.

**Predicted: the across-seed SD of the SNR >= 5 dB gain is at most 0.45 dB.**
Point estimate **0.25 dB**.

The band is set from the one thing we do know — the within-run bootstrap CI on
the gain is 0.163 dB wide — and then widened by roughly a factor of three
rather than the "small multiple" instinct that lost P19. It is deliberately
loose. A prediction that cannot fail is worthless, so the falsifier below is
what makes this a claim.

### Falsifier

P27a fails if the across-seed SD exceeds **0.45 dB**.

---

## P27b — does `+1.902` survive three seeds?

### Prediction

1. **The three-seed mean gain at SNR >= 5 dB is at least `+1.40` dB.**
   Point estimate `+1.85` dB. Corrected *upward* per the amended rule's first
   clause: this is the magnitude of a learned effect, and the historical
   pattern is that the trained arms do better than my instinct allows.
2. **Every individual seed's gain at SNR >= 5 dB exceeds `+1.00` dB.**
3. **The monotone per-bin rise survives.** The three-seed mean per bin is
   non-decreasing across the six bins, and the top bin `[15,20)` is at least
   `+2.00` dB. Seed 1's per-bin values are
   `+0.039, +0.044, +0.511, +1.353, +1.974, +2.628`.
4. **No low-SNR cost.** The three-seed mean in each of the two lowest bins is
   at least `-0.10` dB.

### Falsifier

P27b fails if **any** of:

- the three-seed mean at SNR >= 5 dB falls below `+1.40` dB;
- any single seed's gain at SNR >= 5 dB falls below `+1.00` dB;
- the three-seed per-bin means are not monotone non-decreasing;
- the top-bin three-seed mean falls below `+2.00` dB;
- either of the two lowest bins has a three-seed mean below `-0.10` dB.

### Decision rule — fixed before the run

**Separation criterion:** the headline is clearly separated from no-effect if
`mean - 2 x SD > 0` across the three seeds.

- **If separation holds**, Paper 2 may state `+1.902` dB as its headline with
  the three-seed mean and the seed spread reported alongside it, and every
  interval labelled with the quantity it is over.
- **If separation fails** — that is, if the seed spread is large enough that
  the mean is within two standard deviations of zero — **the headline is a
  single-seed result and the paper must say so in those words**, in the
  abstract and at the point of first quotation. It may not be reported as a
  property of the loss correction.
- **If the monotone rise fails but the aggregate holds**, the per-bin claim is
  withdrawn and only the aggregate is reported.

## Recording rule

P27a and P27b are scored **held / failed** in the Part B report, stated plainly
either way, including if that costs the headline. Every confidence interval in
that report names the quantity it is over: **test realisations** or **seeds**.
Conflating the two is the error that produced P19.
