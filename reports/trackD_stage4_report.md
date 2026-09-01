# Gated Hankel and the [5,20] control — PROMPT 7

**Headline: gating works and the criterion still fails — and Part C shows the
gain it preserves was never the prior's to begin with.**

Primary presentation throughout: `Δ(SNR)` per bin, paired per-trial median,
`n_test = 2000`. Pooled scalars appear only under an explicit
sampling-design-dependent label and carry no decision weight.

---

## 1. Part B — `Δ(SNR)` for the gated arms

| SNR bin | H1 (ungated) | G1 (scalar gate) | G2 (SNR gate) |
|---|---|---|---|
| −10 … −5 | −0.111 | −0.036 | −0.022 |
| −5 … 0 | −0.506 | −0.161 | −0.197 |
| 0 … +5 | −0.451 | −0.085 | −0.117 |
| +5 … +10 | +0.398 | +0.555 | +0.506 |
| +10 … +15 | +1.305 | +1.249 | +1.426 |
| +15 … +20 | +2.226 | +1.883 | +1.984 |

Success criterion, both conditions evaluated separately with bootstrap CIs:

| arm | (i) SNR ≥ 5, bar +0.85 | (ii) SNR < 5, bar −0.05 | success |
|---|---|---|---|
| H1 | +1.209 | −0.333 | — |
| **G1** | **+1.178** [+1.108, +1.233] ✓ (97% of H1) | **−0.095** [−0.127, −0.065] ✗ | **FAIL** |
| **G2** | **+1.324** [+1.245, +1.375] ✓ (110% of H1) | **−0.110** [−0.143, −0.073] ✗ | **FAIL** |

**Both arms pass (i) and fail (ii).** The failure is unambiguous, not a
borderline call: both CIs for condition (ii) exclude −0.05 entirely. I am not
arguing a 0.045 dB miss over the line.

What gating *did* achieve is nonetheless substantial: low-SNR damage falls from
H1's −0.333 to −0.095/−0.110, a **3× reduction**, while the high-SNR gain is
fully preserved and, for G2, slightly improved (+1.324 vs +1.209). Gating is
the right architecture for this problem. It just does not reach a bar that
demanded essentially zero low-SNR cost.

**G2 beats G1, but barely.** Paired on SNR ≥ 5: **+0.084 dB**, CI
[+0.047, +0.118], excluding zero. At SNR < 5 they are indistinguishable
(−0.009, CI [−0.045, +0.026]). So G1 — ten scalars, no SNR information —
captures **89%** of G2's high-SNR gain. PROMPT 7 flagged this outcome
explicitly: a fixed partial strength is most of what matters, and the SNR
conditioning is a real but small refinement rather than the mechanism.

Sanity check: re-evaluating H1 on this pass reproduces stage 3 exactly
(+1.209 / −0.333 / +0.129), same checkpoints and same test set.

## 2. What the gate learned (P8)

G2's `β` rises monotonically with SNR **in every layer**, mean 0.33 at −10 dB
to 0.98 at +20 dB, a span of 0.65. The sharpest conditioning sits in the middle
layers — layer 6 runs 0.00 → 1.00 with a knee at 0 dB.

Per-layer mean `β`, layer 0 → 9:

    G2   0.995  0.935  0.925  0.879  0.815  0.834  0.664  0.805  0.799  0.626
    G1   0.382  0.404  0.524  0.665  0.915  0.746  0.859  0.896  0.902  0.424

**Early layers project hardest.** Both arms agree that the last layer should
project least. The reading: early in the unrolled stack the estimate is rough
and truncation is a strong denoiser; by the last layer the estimate is refined
and hard truncation would discard detail the Transformer has recovered.

## 3. Part C — the result that reframes the project

`C_U1` and `C_H1`, trained *and* tested on SNR ∈ [5, 20] only, matched compute.
**Mechanism probe only** — never a headline, per standing rule 2; the
registered operating range remains [−10, 20].

| SNR bin | H1 vs U1, trained [−10,20] | H1 vs U1, trained [5,20] |
|---|---|---|
| +5 … +10 | +0.398 | **−0.076** [−0.126, −0.020] |
| +10 … +15 | +1.305 | **+0.130** [+0.080, +0.192] |
| +15 … +20 | +2.226 | **+0.232** [+0.172, +0.287] |
| over [5,20] | +1.209 | **+0.078** [+0.050, +0.112] |

**Train on the operating range and the Hankel advantage collapses by 15×.**

The mechanism, from absolute medians on SNR ≥ 5:

| arm | trained [−10,20] | trained [5,20] | gained |
|---|---|---|---|
| U1 (no prior) | −16.314 | −18.967 | **+2.653 dB** |
| H1 (Hankel prior) | −17.476 | −19.105 | +1.629 dB |

**U1 gains far more from focused training than H1 does.** So H1's high-SNR
advantage in stage 3 was not the prior supplying information the network lacked.
It was the prior **partially compensating for U1's underfitting of the high-SNR
regime** — underfitting caused by exactly the loss weighting A4 measured (79–86%
of gradient norm from trials below 5 dB). Give U1 training data concentrated on
the operating range and it learns the geometry itself, leaving the explicit
projection with +0.078 dB to add.

**A4's mechanism is real but runs opposite to my registered reading.** I
predicted loss weighting *damaged H1*, and that removing it would let H1 win by
≥ +1.5 dB. In fact loss weighting *handicapped U1*, and the Hankel prior's
apparent value was a proxy for that handicap. My prediction was falsified, and
falsified in the opposite direction.

**The honest characterisation of the Hankel prior at N=32:** it is a partial
substitute for adequate high-SNR training, not an independent source of
information. Given training focused on the operating range, it is worth
+0.078 dB.

One limit on this claim: the two training distributions produce different
worlds, so the absolute-median comparison across them is distributionally valid
(uniform[−10,20] conditioned on ≥5 *is* uniform[5,20], and the realized means
match at 12.43 vs 12.37) but not paired. The load-bearing number, `C_H1` vs
`C_U1` at +0.078, **is** paired.

## 4. Predictions — one direction right, everything else wrong

**P8 — direction CONFIRMED, magnitude and layer-ordering WRONG.** `β` does
increase with SNR; the named falsifier (flat span < 0.05, or decreasing) did not
fire. But I registered a span of 0.2–0.45 with levels 0.03–0.10 low and
0.25–0.50 high; the truth is a span of 0.65 with levels 0.33 and 0.98. And I
predicted *later* layers would run higher `β`, reasoning from the post-hoc arm.
The opposite holds, monotonically.

**P9 — FALSIFIED, and exactly backwards.** I predicted G2 would pass (ii) and
miss (i) at ≈ +0.70, on the reasoning that gradient starvation would stop the
gate finding the high-SNR branch. G2 missed (ii) and passed (i) at +1.324 —
nearly double my number, and 110% of H1's gain. The gate found the high-SNR
branch without difficulty. My mechanism was wrong even about which half was
hard.

**P10 — ordering CORRECT, reasoning WRONG.** G2 does beat G1 on (i)
(+0.084, CI excluding zero). But I predicted G1 would be near-inert with `β`
drifting *below* its 0.119 initialization; instead `β` climbed to 0.67 mean and
G1 delivered +1.178. A fixed partial projection strength is something the model
actively wants.

**Part C — FALSIFIED, dramatically and in the opposite direction.** Registered:
H1 ≥ +1.5 dB when the loss-weighting confound is removed. Measured: +0.078 dB.

## 5. What this establishes

1. **Gating is the right architecture and still fails the bar.** 3× less
   low-SNR damage, high-SNR gain preserved or improved, but condition (ii)
   fails with CIs excluding the threshold.
2. **Conditioning is not the mechanism.** G1's ten scalars get 89% of G2's
   high-SNR gain. Fixed partial strength is what matters; SNR conditioning adds
   +0.084 dB.
3. **The Hankel prior's value at N=32 is largely a proxy for a training-data
   allocation problem.** Train on the operating range and it is worth +0.078 dB,
   not +1.209.
4. **Early layers want the projection most**, in both gated arms — the opposite
   of the post-hoc intuition.
5. **The STE is a poor gradient approximation**, degrading with depth (cosine
   1.000 at layer 9 → −0.011 at layer 0) rather than with SNR.

## 6. What this does NOT establish

- **That the STE is or is not responsible for the residual low-SNR cost.**
  Part C removed the loss-weighting confound and the gated arms reduced the
  damage 3×, but the residual −0.10 dB is still unattributed between the
  constraint and the STE. An exact-gradient training run would settle it and
  was not authorised.
- **Anything at `N ≠ 32`.** No array-size sweep was run.
- **Anything about sample efficiency.** No budget sweep was run.
- **That [5,20] is the right operating range.** Part C is a mechanism probe.
  Choosing that range as a claim would require arguing it from the system model
  and registering it beforehand.
- **That adaptive rank would behave the same.** Only fixed `r = 7` was trained.

## 7. Files

- `reports/trackD_stage4_results.json` — per-trial rows, per-bin contrasts, β curves
- `reports/trackD_partA7_diagnostics.json` — A3 gradient fidelity, A4 loss decomposition
- `reports/trackD_gated_gates.json` — GK0/GK1/GK2
- `reports/trackD_gated_prereg.md` — P8–P10, committed `f718446` before any run
- `results/track_d/stage4/fig1_gated_delta_by_snr.png` — the Part B result
- `results/track_d/stage4/fig2_learned_beta.png` — what the gate learned
- `results/track_d/stage4/fig3_partC_reframing.png` — **the Part C collapse**
- `results/track_d/stage4/fig4_ste_fidelity.png` — A3 by depth
