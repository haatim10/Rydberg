# Gated Hankel — pre-registered predictions (PROMPT 7 B3)

**Written before any G1/G2 or Part C training run.** Committed separately so
git history establishes the timestamp.

Primary presentation: **`Δ(SNR)` per bin, paired per-trial median.** A pooled
scalar is reported for continuity only and carries no decision weight.

Success criterion, as specified in B4, evaluated as two separate conditions
each with a bootstrap CI:

> **(i)** `Δ ≥ +0.85 dB` on SNR ≥ 5 dB (≥70% of stage 3's `+1.209`)
> **(ii)** `Δ ≥ −0.05 dB` on SNR < 5 dB

---

## The constraint that shapes every prediction below

A4 measured that **86% of H1's gradient norm comes from trials below 5 dB**
(U1: 79%), because `nmse_loss` averages per-sample *normalized* error and
low-SNR trials have far larger normalized error. Nothing in the gated design
changes that. So the gate's learning signal is dominated by the regime where
the correct answer is `β ≈ 0`, and the regime where the projection pays
(+2.226 dB at 15–20 dB under H1) contributes ~14% of the gradient.

This is the single fact I expect to limit both arms, and it is why my
predictions are less optimistic than the architecture would otherwise justify.

## P8 — does `β_t` learn SNR dependence in G2?

**Prediction: yes, `β` increases with SNR (decreases with `σ²`), but by less
than the optimum.** Direction: confident. Magnitude: I expect roughly

- `β ≈ 0.03 – 0.10` at the lowest SNR (−10 dB, large `σ²`)
- `β ≈ 0.25 – 0.50` at the highest SNR (+20 dB, small `σ²`)

i.e. a span of roughly 0.2–0.45, not the 0→1 span the stage-3 per-bin table
would justify. I also expect **later layers to run higher `β` than early
layers**, because a projection near the output behaves like the post-hoc arm,
which never hurt in any bin (+0.013 → +0.837), whereas an early projection is
the one whose damage the remaining nine layers must absorb.

**Falsifier:** if `β` is flat in `σ²` (total span < 0.05 across the training
`σ²` range) or *decreases* with SNR, the SNR-conditioning hypothesis is wrong
and I will say so.

## P9 — predicted `Δ(SNR)` for G2 vs U1, per bin

| SNR bin | H1 measured | **G2 predicted** |
|---|---|---|
| −10 … −5 | −0.111 | **0.00** |
| −5 … 0 | −0.506 | **−0.05** |
| 0 … +5 | −0.451 | **+0.05** |
| +5 … +10 | +0.398 | **+0.30** |
| +10 … +15 | +1.305 | **+0.70** |
| +15 … +20 | +2.226 | **+1.20** |
| **SNR ≥ 5 aggregate** | +1.209 | **≈ +0.70** |
| **SNR < 5 aggregate** | −0.333 | **≈ 0.00** |

**So I predict G2 PASSES condition (ii) and MISSES condition (i)** — the gate
repairs the low-SNR damage, which is the easy half, but recovers only ~58% of
the high-SNR gain against the 70% required. Confidence: moderate (~55%).

**Falsifier:** if G2 meets both conditions, my gradient-starvation reasoning is
wrong — the gate found the high-SNR branch despite receiving ~14% of its
signal there — and the honest conclusion is that gating solves the problem
outright.

## P10 — G1 or G2?

**Prediction: G2 beats G1 on condition (i), and G1 is close to inert.**

G1 has one scalar per layer, so it must pick a single `β` that trades low-SNR
damage against high-SNR gain under a loss that is 86% low-SNR. The minimizer of
that trade-off is small. **I predict G1's learned `β_t` ends up *below* its
0.119 initialization in most layers, and G1's `Δ(SNR)` is near zero in every
bin** — passing (ii) trivially and failing (i) badly.

That makes G1 a real control rather than filler: it isolates whether the
benefit needs *conditioning* or merely *partial strength*.

**Falsifier:** if G1 matches or beats G2 on condition (i), then a fixed partial
projection strength is what matters and the SNR conditioning is unnecessary —
the simpler and stronger result, and I will report it as such.

---

## Part C — running it, and why

B4's rule: run Part C only if A3 leaves the STE and loss-weighting explanations
unseparated. **It does, so Part C runs.**

A3 measured, against the exact float64 gradient through the SVD:

| | low SNR (−10…0) | high SNR (+10…+20) |
|---|---|---|
| median cosine (gate excluded) | 0.630 | 0.821 |
| layer 9 transformer | 1.000 | 1.000 |
| layer 0 transformer | −0.011 | 0.128 |

Fidelity is **not** high in both regimes, so the "skip Part C" branch does not
apply. It is worse at low SNR (0.630 vs 0.821), which implicates the STE — but
it is poor at *both*, and the dominant axis is depth rather than SNR, so the
STE cannot be assigned the low-SNR-*specific* damage either. Meanwhile A4
independently confirms the loss-weighting mechanism is real and large.

Both explanations are live; neither is separated from the other. Part C
(training U1 and H1 on SNR ∈ [5, 20] only) removes the loss-weighting
confound: if H1 then wins clearly, low-SNR loss dominance was the cause.

**Registered reading, before the run:** I expect H1 to beat U1 clearly on
`[5, 20]`-only training, by **≥ +1.5 dB** median paired on that range —
i.e. more than the +1.209 it manages when trained on the full range. If it
does, loss weighting was the dominant cause. If H1's advantage on `[5,20]`
training is no larger than +1.209, then loss weighting was not the binding
constraint and the STE or the constraint itself carries the blame.

**This is a mechanism probe, not a performance claim.** Per standing rule 2,
`[5, 20]` results are never reported as HS-URformer's headline; they exist only
to separate two causes. The registered operating range remains `[−10, 20]`.
