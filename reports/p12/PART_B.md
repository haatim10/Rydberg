# PROMPT 12 — Part B: classical sweeps

No training. All five items complete. Every number below is in
`results/p12/*.json`; the scoring in `reports/p12/partB_scores.json` was
computed by `scratch/p12_partB_analyze.py`, which was **committed before any
result existed** (`7f6ea9e`) so the thresholds could not drift.

Pre-registrations are in `reports/p12/PREREG_P16_P18.md`, committed at
`3741219`, before the first cell ran.

## Scorecard

| item | verdict |
|---|---|
| **P16** bridging run | **HELD** |
| **P17** pencil sweep | **FAILED** (quantisation account nonetheless refuted) |
| **P18** post-hoc vs interleaved | **FAILED** on 2 of 3 clauses |
| **B2** bracketing | **SPLIT** — met for the prior, not for the estimator |
| **B4** coarse-to-fine gates | **PASSED**, with a gate-wording ambiguity |

Two of three pre-registrations failed. Both failures cost the manuscripts
something, and both are reported here in the terms the pre-registration set.

---

## B1 — the bridging run. **P16 HELD.**

TD operator (`hs_gs_auto`, `max_iter=100`, `select_iter=25`, `cadzow_iter=4`)
at the B3 principal operating point: N=32, K=3, P=30, RSR 12 dB, six-point B3
SNR grid, 400 paired trials per point, Track B generator.

| SNR | Δ_HS median | CI95 | ratio-of-sums | mean `L̂` |
|---|---|---|---|---|
| −5 | +3.053 | [+2.902, +3.190] | +2.962 | 2.52 |
| 0 | +2.461 | [+2.262, +2.569] | +2.325 | 4.09 |
| +5 | +2.593 | [+2.410, +2.791] | +2.456 | 5.04 |
| +10 | +2.487 | [+2.345, +2.678] | +2.464 | 5.50 |
| +15 | +2.750 | [+2.589, +2.885] | +2.587 | 5.73 |
| +20 | +2.761 | [+2.646, +2.893] | +2.621 | 5.92 |

**Six-point mean: +2.684 dB.**

Scored against P16 as written:

- predicted mean band [+2.2, +2.9] → **2.684, inside** ✓
- falsifier band [+2.0, +3.1] → inside ✓
- every bin in [+2.0, +3.2] → true ✓
- no bin CI wholly outside [+1.5, +3.5] → true ✓

**P16 HELD.** My point estimate of +2.45 was 0.23 dB low; the band held.

### The operator-identity check

A1 predicted that because both families run the *same* operator, a TD-operator
run at the B3 operating point must reproduce B3's numbers. Ratio-of-sums,
bridging minus B3, at matched cells on **independent worlds**:

| SNR | −5 | 0 | +5 | +10 | +15 | +20 |
|---|---|---|---|---|---|---|
| B3 | +3.039 | +2.419 | +2.268 | +2.351 | +2.508 | +2.603 |
| bridging | +2.962 | +2.325 | +2.456 | +2.464 | +2.587 | +2.621 |
| difference | −0.077 | −0.094 | +0.188 | +0.113 | +0.079 | +0.018 |

Largest discrepancy 0.188 dB, mean absolute 0.095 dB. **The two families were
never measuring different estimators.**

### Decision rule — triggered as pre-registered

Per the brief and per P16, fixed before the run and independent of which number
came out larger: **the TD operator is now *the* proposed estimator for Paper 1,
and the B3 `n_cz = 4` numbers are demoted.**

The bridging number is **+2.684 dB mean, +2.46 to +3.05 dB per bin** — which is
*not* materially worse than the +2.27 to +3.04 dB headline it replaces. The
headline therefore survives essentially unchanged, now measured on one
consistently-specified estimator.

**But the "sensitivity check on the sweep count" the brief anticipates does not
exist.** A1 established `n_cz = 4` in both families. If Paper 1 wants a
sweep-count sensitivity check it must be run separately, varying `cadzow_iter`
explicitly. That was out of scope this turn.

---

## B2 — bracketing the extrapolated crossings. **SPLIT.**

Four cells, all above their array's committed crossing. `ALL_THREE_BRACKETED
= false`.

| cell | ρ | Δ_HS | CI95 | negative-significant? |
|---|---|---|---|---|
| N16_L9 | 0.608 | +0.078 | [+0.000, +0.251] | no |
| N16_L14 | 0.735 | +0.000 | [+0.000, +0.000] | no |
| N64_L38 | 0.596 | +0.044 | [−0.001, +0.114] | no |
| N64_L48 | 0.669 | +0.018 | [−0.014, +0.059] | no |

**ρ ≈ 0.75 is unreachable at N=64.** The ULA path drawer raises `RuntimeError`
(`rydberg_sim/channel.py:245`, `PSI_SEP_MIN`) above L=48, so ρ tops out near
0.669 there. The brief's target could not be met and the L values were chosen
from a measured `r_eff(L)` probe instead of guessed.

### The forced-order diagnostic — and a reversal

The B2 cells alone suggested Δ_HS asymptotes to zero rather than crossing it,
which would have made the array-invariant crossing an artifact of extrapolating
through near-zero values. **That reading was wrong**, and the diagnostic
(`results/p12/B2_forced.json`) shows why. Same operator, order forced to
`cap − 1` so the projection always acts, on identical worlds:

| ρ | abstention | Δ with abstention | **Δ forced active** |
|---|---|---|---|
| 0.542 | 12.0% | +0.276 [+0.200, +0.430] | +0.099 [+0.063, +0.128] |
| 0.608 | 19.2% | +0.048 [+0.000, +0.164] | +0.043 [−0.013, +0.092] |
| **0.735** | **38.8%** | **+0.000 [0, 0]** | **−0.075 [−0.186, −0.026]** |

At ρ = 0.735 the forced arm is **significantly negative**. The negative region
is real; the abstaining estimator refuses to enter it.

**[FACT] The crossing is a property of the prior, not of the estimator.** Past
the crossing the Hankel prior genuinely hurts; abstention detects that and
declines, converting harm into exactly zero. Δ_HS for the proposed estimator is
bounded below by ≈ 0, while the prior's own value crosses zero.

This *rescues and sharpens* Paper 1's claim rather than undermining it: ρ
predicts where the prior stops paying, and abstention is what stops the
estimator following it down.

**Correction to an earlier inference.** I recorded that N16_L14 abstains "in
essentially every trial". It does not — abstention is **38.8%**. The exactly
+0.000 median with a zero-width CI arises because those zeros straddle the
median while the remainder splits roughly evenly either side. So when the
projection *does* engage at high ρ, it helps about as often as it hurts.

**Verdict.** B2's success condition — a measured cell whose CI excludes zero on
the negative side — is **met for the prior** (forced-order arm) and **not met
for the proposed estimator**, which cannot produce a negative by construction.
Both halves belong in Paper 1.

⚠ The forced column is **DIAGNOSTIC ONLY** and must never be quoted as HS-GS.

---

## B3 — pencil sweep. **P17 FAILED. Quantisation account REFUTED.**

N=32, K=3, P=20, RSR 10 dB, ρ ≈ 0.233 held fixed, 250 trials per point.

| p | shape | admissible ranks | Δ_HS | CI95 | mean `L̂` |
|---|---|---|---|---|---|
| 4 | 28×5 | 5 | +2.740 | [+2.576, +2.934] | 3.11 |
| 6 | 26×7 | 7 | +3.364 | [+3.182, +3.591] | 3.73 |
| **24** | **8×25** | **8** | **+3.121** | [+2.940, +3.367] | **4.16** |
| **8** | **24×9** | **9** | **+3.199** | [+3.027, +3.393] | **4.15** |
| 12 | 20×13 | 13 | +2.911 | [+2.719, +3.155] | 4.89 |
| 16 | 16×17 | 16 | +2.809 | [+2.490, +3.115] | 4.70 |

### Shape versus count — the comparison that escapes the confound

The pre-registration stated up front that at fixed N the pencil sets
admissible-rank *count* and maximum representable *rank* together, so the sweep
cannot separate them. One pair does separate shape from count: **p=8 (24×9)**
and **p=24 (8×25)** have opposite shapes, nearly identical counts (9 vs 8), and
give **+3.199 vs +3.121** — a 0.078 dB difference with heavily overlapping CIs
and identical mean `L̂` (4.15 vs 4.16).

**[FACT] Transposing the Hankel matrix changes nothing. The effect tracks the
admissible-rank count, not the shape.**

### Verdicts

**Quantisation account: REFUTED.** Its pre-registered falsifier required
monotone dependence on the rank count across p ∈ {8,12,16,24} with span
> 0.5 dB. The dependence is **not monotone** (3.121 → 3.199 → 2.911 → 2.809)
and the span is **0.390 dB**. Both clauses fail. If anything the dependence
runs *backwards* over the ≥8 group — a coarser grid is mildly better.

**Consequence: the 0.493 dB N=16-vs-N=64 magnitude residual keeps its only
named explanation removed, and stays unexplained in Paper 1.**

**P17: FAILED**, narrowly, on both quantitative clauses:

| clause | predicted | actual |
|---|---|---|
| span over adm ≥ 8 | < 0.30 dB | **0.390 dB** |
| drop at adm 5 vs the ≥8 mean | 0.3–1.5 dB | **0.270 dB** |
| not monotone over adm ≥ 8 | true | true ✓ |

The direction of the reasoning was right; the thresholds were too tight. Note
this failed in the **opposite** direction from the standing bias correction: I
was told I over-attribute limits to structural machinery, corrected for it, and
over-corrected — predicting a *smaller* structural effect than exists. A
correction applied as a fixed offset rather than a re-examination will do that.

### Flagged, not asserted

Mean `L̂` rises with the candidate count (3.11 → 3.73 → 4.15 → 4.89). A larger
candidate set lets the held-out criterion pick a higher order, weakening the
constraint. That is a claim about the **selector**, not the Hankel geometry,
and this sweep was not designed to isolate it. It would need its own experiment.

---

## B4 — coarse-to-fine order selection. **BOTH GATES PASS.**

N=32, K=3, P=20, RSR 10 dB, SNR ~ U[−10,20], 300 trials, stride 3.

| | |
|---|---|
| agreement with exhaustive `L̂` | **93.3%** (gate 1: ≥ 90%) ✓ |
| candidates evaluated | 16.0 → **9.61** (−39.9%) |
| selection wall time | 877 s → **521 s** (**1.68×**) |
| Δ_HS exhaustive | +2.950 dB |
| Δ_HS coarse-to-fine | +2.867 dB |
| degradation, paired per trial | **0.000 dB, CI [0.000, 0.000]** (gate 2: ≤ 0.05) ✓ |

### ⚠ Gate 2 ambiguity — reported, not resolved in our favour

The paired degradation is exactly zero with a zero-width CI because in the
93.3% of trials where both searches return the same `L̂` the estimates are
**bit-identical**, so those differences are exactly 0 and dominate the median.
The 6.7% disagreements are what move the **aggregate** Δ_HS: 2.950 → 2.867, a
**0.083 dB** drop, which **exceeds** the gate's 0.05 dB threshold.

Gate 2 as written ("Δ_HS degrades by no more than 0.05 dB, with a CI supporting
that") **passes on the paired statistic and fails on the aggregate**. The
gate's wording did not anticipate a statistic whose CI collapses.

**Recommendation for PROMPT 13:** state it as *"identical on 93% of trials;
aggregate cost 0.083 dB; selection stage 1.68× faster"* rather than quoting the
zero.

This closes the manuscript's standing admission that it *describes* the fix
rather than doing it. Order selection was 57.7–85.5% of runtime.

---

## B5 — classical post-hoc versus interleaved. **P18 FAILED.**

N=32, K=3, P=30, RSR 12 dB, six-point grid, 300 paired trials per point.
Both arms use the **same** `L̂` from the same held-out selection, so the
contrast isolates placement. Positive = interleaved better.

| SNR | Δ | CI95 | |
|---|---|---|---|
| −5 | −0.081 | [−0.159, +0.047] | |
| **0** | **−0.164** | [−0.317, −0.066] | **significant negative** |
| +5 | +0.034 | [−0.050, +0.133] | |
| +10 | +0.010 | [−0.071, +0.077] | |
| +15 | +0.054 | [−0.014, +0.107] | |
| **+20** | **+0.168** | [+0.102, +0.246] | **significant positive** |

**Six-point mean: +0.003 dB.**

| clause | predicted | actual | |
|---|---|---|---|
| one-signed positive | true | **false** — 0 dB CI excludes zero on the negative side | FAILED |
| pooled magnitude | [+0.3, +1.5] dB | **+0.003 dB** | FAILED |
| growth with SNR | [+0.2, +1.5] dB | **+0.250 dB** | HELD |

**[FACT] The classical contrast changes sign with SNR, exactly as the learned
one does (Part A, A2).** I predicted one-signed positive, reasoning that EM-GS
has no trained component to compete with the projection so nothing could make
interleaving worse. Wrong twice: the sign flips, *and* the whole effect is
negligible — |Δ| ≤ 0.17 dB everywhere, against **+1.309 dB** on the learned
side at 15–20 dB.

### Cost to Paper 1

Its prose argues the projection **must** sit inside the iteration, and never
measured this classically. Measured: **worse at 0 dB, indistinguishable from +5
to +15, better only at +20 dB by 0.17 dB** — while doing ≈ 400× the projection
work (4 Cadzow sweeps after each of 100 iterations, versus one sweep at the
end).

**The placement argument does not survive as stated.** It must become a
bounded, SNR-qualified claim. The honest framing — and it is the reconciliation
document's thesis appearing in a third place — is that **placement matters
inside a trained network and barely at all in a fixed classical iteration.**

---

## What Part B changes for PROMPT 13

1. **One estimator throughout, per the B1 decision rule.** The TD operator is
   the proposed estimator; headline becomes **+2.684 dB mean, +2.46 to +3.05
   per bin**. The B3/TD "different operators" framing is wrong and must go.
2. **The sweep-count sensitivity check does not exist** as a B3/TD difference.
   Either run it separately or drop the claim.
3. **The placement claim must be SNR-qualified** or removed.
4. **The 0.493 dB residual stays unexplained**; the quantisation candidate is
   refuted in sign.
5. **The crossing is a property of the prior, not the estimator** — a sharper
   claim than the current text, and it needs the abstention mechanism stated
   alongside it.
6. **Coarse-to-fine selection can be reported** as 93% agreement / 1.68×
   faster / 0.083 dB aggregate cost.
7. **ρ ≈ 0.75 is unreachable at N=64**; any text implying the curve was probed
   that far must be corrected.
