# PROMPT 19 Part B — pre-registration P29

**Committed standing alone, before Tier 1 batch C2 starts.**

## The runs

Batch **C2**: two additional seeds on each arm of the **mixed-SNR** Δ_H
contrast, four runs.

| arm | prior | training SNR | seeds to add |
|---|---|---|---|
| `C2_U1_seed{2,3}` | none | `[-10, 20]` | 2, 3 |
| `C2_H1_seed{2,3}` | Hankel, rank 7 | `[-10, 20]` | 2, 3 |

80k / 13 epochs, `N = 32`, `K = 3`, `L_k ~ U{3..7}`, `P = 20`, RSR 10 dB,
`init="spectral"`, one driver for both arms, matched on data order and
schedule.

## The anchor, read from disk rather than from memory

Seed 1 is the existing stage-2 / stage-3 pair:
`results/track_d/stage2/B3_80k_13ep/best.pt` and
`results/track_d/stage3/H1_hs_urformer_80k/best.pt`. Both checkpoints report
`init='spectral'`, `train_seed=20260827`, selected epoch **6**. [FACT]

The headline `+1.209` dB is the **SNR ≥ 5 restriction** of that contrast, CI
over test realisations `[+1.139, +1.330]`
(`reports/p12/PART_A.md` §A2, from the stage-3 per-trial rows). Its per-bin
values, all six CIs excluding zero:

| bin (dB) | Δ_H, seed 1 |
|---|---|
| [−10, −5) | **−0.111** |
| [−5, 0) | **−0.506** |
| [0, +5) | **−0.451** |
| [+5, +10) | +0.398 |
| [+10, +15) | +1.305 |
| [+15, +20) | +2.226 |

Two things about the anchor that must be said before any new number lands:

1. **The pooled-over-everything figure in `reports/trackD_stage3_results.json`
   is `+0.129` dB, not `+1.209`.** The difference is entirely the SNR ≥ 5
   restriction, which discards the three negative bins. This is not an error in
   either number; it is the reason a scalar was never the right object.
2. **The ratio-of-sums pooling over the full range is `−0.214` dB** — negative,
   already in the repository, already seen. The ratio-of-sums figure
   *restricted to SNR ≥ 5* is **not** in the repository and has never been
   computed; it is unseen and the C2 scorer will produce it for every seed.

## Predictions

### P29a — across-seed SD of mixed-SNR Δ_H at SNR ≥ 5

Two valid anchors now exist, and they do not scale together:

| contrast | magnitude | across-seed SD | SD / magnitude |
|---|---|---|---|
| Tier 0.5 loss design | ≈ +2.17 dB | 0.241 | 0.11 |
| C1 focused Δ_H | ≈ +0.02 dB mean | 0.083 | ≫ 1 |

The SD is not proportional to the effect. Between a 0.07 dB effect with
SD 0.083 and a 2.17 dB effect with SD 0.241, a 1.2 dB effect interpolates to
roughly 0.15–0.20.

Hedged per the amended standing rule, which exists because P19 was lost by
predicting `SD <= 0.12` and measuring `0.244`:

**Predicted: the across-seed SD of mixed-SNR Δ_H at SNR ≥ 5 is at most
0.45 dB.** Point estimate **0.20 dB**.

**Falsifier:** P29a fails if the across-seed SD exceeds **0.45 dB**.

### P29b — is `+1.209` representative of its own distribution?

The literal "does `+1.209` lie inside the three-seed range" is vacuous: seed 1
**is** `+1.209`, so it lies in `[min, max]` by construction. Recorded so the
test is not later credited with passing something it cannot fail. This is the
same trap P28b named, and P28b then failed on the non-vacuous version.

**Predicted: `+1.209` is the middle of the three seed values, not the minimum
or the maximum.** Probability 1/3 under exchangeability, so this is expected to
fail two times in three, and that asymmetry is the point: a single-seed number
quoted as a headline is a claim that it is typical.

**Falsifier:** P29b fails if `+1.209` is the smallest or the largest of the
three.

### P29c — sign stability

**Predicted: all three seeds give mixed-SNR Δ_H > 0 at SNR ≥ 5.** Corrected
*upward* per the amended rule's first clause — the trained arms historically do
better than instinct allows — and distinguished from P28c, which failed on a
contrast whose magnitude was comparable to its own spread. Here the anchor is
fifteen times its predicted spread.

**Falsifier:** any seed gives Δ_H ≤ 0 at SNR ≥ 5.

### P29d — the per-bin sign pattern replicates

**This prediction is an addition beyond the brief**, recorded as such. The
brief asks only P30 to predict a per-bin sign pattern; registering it here too
makes C2 a sharper test than the scalar alone, and it costs nothing because the
same evaluation produces it.

Every focused seed in C1 gave a negative `[5,10)` bin and a positive `[15,20)`
bin. Seed 1 of the mixed design gives negatives in all three bins below +5 dB
and positives in all three at or above +5 dB.

**Predicted: on all three mixed seeds, every bin below +5 dB is negative and
every bin at or above +5 dB is positive** — that is, the sign flips exactly
once, between `[0,+5)` and `[+5,+10)`, on each of the three seeds
independently.

**Falsifier:** P29d fails if any seed's per-bin sign sequence is not three
negatives followed by three positives. A bin whose CI over test realisations
covers zero is scored by its point estimate, and the CI is reported alongside.

## Which of the three outcomes fires — fixed before scoring

Let `m` be the three-seed mean at SNR ≥ 5, `s` the across-seed SD, `r` the
range, `v_min` the smallest seed value. Evaluated **in this order**, so the
branches are exhaustive and mutually exclusive:

1. **Outcome (b) — no training design yields a stable Δ_H** if any of:
   `v_min ≤ 0`, or `m − 2s ≤ 0`, or `s ≥ 0.40·m`.
   The last clause is the brief's "spread comparable to its magnitude" made
   numeric; at `m ≈ 1.2` it triggers at `s ≥ 0.48`.
   *Consequence:* the paper's finding becomes that Δ_H is not measurable at the
   resolution the literature reports it to. Larger claim, harder to state,
   still publishable. The structural-prior section is written around the
   per-bin pattern, not a scalar.

2. **Outcome (a) — the collapse claim survives in corrected, stronger form** if
   branch 1 did not fire and `m ≥ 0.90`.
   `0.90` is 75% of `+1.209`; a drop past a quarter of the headline is a
   material change to what may be quoted.
   *Consequence:* the paper says the structural prior gives a stable,
   substantial gain under mixed-SNR training and a quantity with no stable sign
   under focused training. This is the best outcome and the easiest to write.

3. **Outcome (c) — `+1.209` was a favourable seed** if neither of the above,
   i.e. branch 1 did not fire and `m < 0.90`.
   *Consequence:* the paper reports the three-seed mean plainly, says `+1.209`
   was the favourable draw, and never quotes it as a point value again.

**The outcome is read off these rules mechanically.** No branch may be argued
for after the numbers are seen.

## Gate

**C2 reports before C3 launches.** Batches C3 and C4 do not start until this
pre-registration has been scored and the outcome recorded.

## Recording rule

P29a, P29b, P29c and P29d are scored **held / failed** in the Part B report,
stated plainly either way. Per-seed values are reported **per bin and under
both pooling rules** — paired per-trial median and ratio of sums — because C1
showed the two disagree in sign and reporting only one would hide it. Every
interval names the quantity it is over: test realisations or seeds. Three seeds
give two degrees of freedom, so the SD carries large sampling uncertainty and
any statement Paper 2 makes from it must say it was evaluated on three seeds.
