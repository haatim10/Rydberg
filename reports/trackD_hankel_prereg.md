# HS-URformer — pre-registered predictions (PROMPT 6 Part B)

**Written before any HS-URformer training run.** Committed separately so git
history establishes the timestamp.

Primary statistic, held everywhere: **the paired per-trial median.** Ratio-of-sums
appears only where labelled, and only for Q4 where the tail is the question.

`Δ_H = NMSE_UR − NMSE_HUR` in dB, paired per trial. **Positive means Hankel
helps.**

---

## Q1 — accuracy: is `Δ_H > 0` at 80k?

## P7 — my prediction for `Δ_H` at 80k

**`Δ_H ≈ +0.1 to +0.4 dB`, most likely ~+0.25 dB, and I expect the CI to
exclude zero but land BELOW the +0.3 dB go threshold.** Confidence: moderate
(~55%). So my prediction is that Part B lands in the *marginal* band and Part C
should **not** be launched on this evidence alone.

Reasoning, and it is deliberately shaped by having been wrong before:

**Why some gain is likely.** The stage-2 structural result is real: at 80k the
URformer already beats the unstructured-LS oracle by 1.34 dB on ratio-of-sums,
which is direct evidence it has internalised part of the `L_k ≤ 7` geometry.
Imposing that structure exactly should help at least a little, particularly at
low SNR where the estimate is noisiest and the rank-7 constraint removes the
most noise-fitting freedom.

**Why I expect it to be small.** This is the lesson from stage 2. Track B
measured HS-GS beating EM-GS by **+2.85 dB at N=32** — but that was against a
classical estimator with *no* mechanism for learning structure. The URformer at
80k has already climbed to 80% of the oracle headroom and captures much of the
same geometry implicitly. **The Hankel projection is not adding a prior to a
model that lacks one; it is replacing an implicit prior with an explicit one.**
The marginal value of that is the *difference* between the two, not the full
+2.85 dB, and the stage-2 evidence says the implicit version is already good.

**Why I am not predicting zero or negative.** The explicit constraint is exact
where the implicit one is approximate, and it costs no parameters. At the
lowest SNRs, where the network has the least signal to learn from, an exact
constraint should still win.

**Falsifier, stated now:** if `Δ_H ≥ +1.0 dB`, my "the network already has the
prior" reading of stage 2 is wrong — the implicit recovery would have to be far
weaker than the oracle-crossing suggested, and the ratio-of-sums crossing would
need a different explanation. I will say so rather than absorbing it.

**Conversely, if `Δ_H ≤ 0`:** explicit structure is actively harmful on top of
a model that has learned it implicitly, most likely because the rank-7 hard
truncation destroys genuine signal the network was using — and the honest
conclusion is that HS-URformer is not worth pursuing at `N=32`.

## Q2 — sample efficiency, as a horizontal shift

Reported as "HS-URformer reaches equal NMSE with X× less data", from
interpolating the two budget curves, with a CI. **Prediction: `X ≈ 1.2–1.6×`,**
i.e. a real but modest shift. Rationale: an explicit prior should help most
where data is scarcest, so I expect `Δ_H` to be *larger at 20k than at 80k* —
which is the sample-efficiency claim, and is a different question from Q1.

**This is the prediction I hold most confidently**, and it is the one that
would justify Part C even if Q1 is marginal. A prior that buys little at 80k but
meaningfully more at 20k is exactly what "sample-efficiency prior" means.

## Q3 — scaling with `N`

`Δ_H ≤ 0` at `N=8`, positive and growing for `N ≥ 16`. **Gate HK7 has already
settled the `N=8` half, and more sharply than predicted:** at `N=8, p=4` the
embedding is 4×5, so *every* length-8 vector has Hankel rank ≤ 4 and a rank-7
request truncates to `min(7,4)=4` — no truncation at all. Measured: the
projection returns an unstructured random vector unchanged (rel `4.97e-16`),
while genuinely constraining one at `N=32` (rel `3.32e-01`).

So at `N=8` the operator is **vacuous, not lossy**, and HS-URformer is *exactly*
URformer. `Δ_H = 0` there is not a prediction but an algebraic identity, up to
training noise.

**The `N ≥ 2L − 1` condition and why it deserves attention.** A length-`N` sum
of `L` exponentials has Hankel rank exactly `L` only when the embedding can
carry it, which needs `N ≥ 2L − 1`; at `L_max = 7` that is `N ≥ 13`.
`SystemModel.pdf` §10 requires only `N > max_k L_k` for the angles to be
identifiable — `N ≥ 8`. **The structural estimator therefore needs roughly twice
the array that bare identifiability needs.** Between `N = 8` and `N = 13` the
channel is identifiable but the Hankel prior cannot be imposed, which is a real
gap and the reason Q3 predicts a sign change rather than a monotone trend.

## Q4 — the tail (ratio-of-sums, labelled)

**Prediction: yes, Hankel helps the hard low-SNR tail disproportionately.** This
is the one question where ratio-of-sums is the right statistic, because it is
dominated by exactly those trials. I expect `Δ_H` measured on ratio-of-sums to
exceed `Δ_H` on the median, plausibly by 2×.

## Q6 — internal vs post-hoc

**Prediction: internal beats post-hoc, but by less than half the total effect.**
Applying the projection once at the end (`U1+post`) captures the constraint but
not its interaction with the ten unrolled updates. If they tie, structural
*integration* is not demonstrated — only that the prior helps somewhere — and I
will report it that way.

## X1 — the control that reframes everything

EM-GS (converged, `T_GS=100`) followed by a **single** Transformer
post-processor (~158k params, not ten). **Prediction: X1 lands within 0.5 dB of
the full URformer.**

With 96% of the stage-2 gain in the Transformer, this is the load-bearing
control. If a non-unrolled denoiser on classical output matches a ten-layer
unrolled network, then Track D's honest description is **"denoise the classical
estimate"**, and the unrolling — the paper's central claim — is doing little.
I think that is more likely than not, and it would be the most consequential
finding of this phase.

---

## Go / no-go, as specified

| `Δ_H` at 80k | action |
|---|---|
| `≥ +0.3 dB`, CI excludes zero | proceed to Part C |
| `0 < Δ_H < +0.3 dB` | **stop and report** — marginal |
| `≤ 0` | **stop and report** — include U1+post and X1 |

**My prediction puts us in the middle band.** I will apply the rule as written
rather than arguing a marginal result over the line.
