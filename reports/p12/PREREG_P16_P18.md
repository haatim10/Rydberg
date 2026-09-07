# PROMPT 12 Part B — pre-registrations P16, P17, P18

**Committed before any Part B cell is run.** Written after Part A, so it uses
A1's finding that B3 and TD run the *same operator* (`hs_gs_auto`, adaptive
held-out order, `n_cz = 4`) and differ only in operating point and statistic.

**Bias correction applied.** Across stages 1–5 every prediction miss ran the
same way: underestimating what the system works out for itself, and
over-attributing limits to structural machinery. Both P17 and P18 are set
*lower* than my uncorrected instinct because of this.

---

## P16 — the bridging run (B1)

**Cell.** TD operator (`hs_gs_auto`, `exact_step="em_gs"`, `max_iter=100`,
`select_iter=25`, `cadzow_iter=4`) at the B3 principal operating point:
N = 32, K = 3, P = 30, RSR = 12 dB, the six-point B3 SNR grid
{−5, 0, 5, 10, 15, 20}, 400 paired trials per point.

**What is actually changing.** A1 established the operator is identical to
B3's. Relative to the committed B3 cell, only three things move:
`max_iter` 50 → 100, `select_iter` 20 → 25, and the reported statistic
(ratio-of-sums per point → paired per-trial median per bin).

### Evidence in hand

| source | Δ_HS | configuration |
|---|---|---|
| B3, N=32, P=30 | 2.268–3.039 dB (mean 2.531) | RSR 12, T=50, ratio-of-sums |
| TD B7, P=20 | 2.680 dB pooled / 2.732 high-SNR | RSR 10, T=100, paired median |
| TD B7, P=35 | 2.300 dB pooled / 2.501 high-SNR | RSR 10, T=100, paired median |

Interpolating the TD pilot trend to P = 30 gives ≈ 2.4 dB at RSR 10 dB.
Raising RSR to 12 dB makes phase retrieval easier and should *reduce* the
structural headroom slightly.

### Prediction

**Per-bin Δ_HS lands in [+2.0, +3.2] dB at every one of the six SNR points,
and the six-point mean lands in [+2.2, +2.9] dB.**

Point estimate for the mean: **+2.45 dB**.

I do **not** predict the bridging number beats +2.27 dB. I expect it to land
slightly *below* B3's mean of 2.531, because the pilot trend at P = 30 and the
switch from ratio-of-sums to paired median both push down.

### Falsifier

P16 fails if **either**:
- the six-point mean falls outside [+2.0, +3.1] dB, **or**
- any single bin's Δ_HS bootstrap CI lies wholly outside [+1.5, +3.5] dB.

### Decision rule — fixed here, before the run

Per the brief, and **regardless of which number is larger**: the TD operator
becomes *the* proposed estimator in Paper 1 and the B3 `n_cz = 4` numbers are
demoted to a sensitivity check on the sweep count. If the bridging number is
materially worse than +2.27 dB, **Paper 1 reports the smaller number as its
headline and says why.**

*Note, recorded now so it cannot be constructed afterwards:* A1 showed
`n_cz = 4` in **both** families, so the "sensitivity check on the sweep count"
the brief anticipates does not exist as a difference between B3 and TD. If a
sweep-count sensitivity check is wanted it has to be run separately, varying
`cadzow_iter` explicitly. That is out of scope this turn.

---

## P17 — pencil sweep (B3)

**Cell.** N = 32 fixed, K = 3, P = 20, RSR 10 dB, ρ held fixed by using the
default channel (L_k ~ U{3..7}, ρ ≈ 0.233). Vary the pencil parameter `p`,
which sets the Hankel shape (N−p) × (p+1) and hence the admissible-rank count
`min(N−p, p+1)`:

| p | Hankel shape | admissible ranks |
|---|---|---|
| 4 | 28 × 5 | 5 |
| 6 | 26 × 7 | 7 |
| 8 | 24 × 9 | 9 |
| 12 | 20 × 13 | 13 |
| 16 | 16 × 17 | **16** (the default, ⌊N/2⌋) |
| 24 | 8 × 25 | 8 |

**Confound, stated before the run.** At fixed N, `p` changes the number of
admissible ranks *and* the maximum representable rank — they are the same
quantity, `min(N−p, p+1)`. This experiment therefore **cannot separate**
"coarser rank grid" from "lower rank capacity". Whatever it returns, it cannot
by itself establish the quantisation account. I record this now rather than
discovering it in the analysis.

### Prediction

The quantisation account predicts Δ_HS degrades monotonically as the
admissible-rank count falls. **I predict it does not, except where capacity
binds.** Specifically:

- For admissible-rank count ≥ 8 (p ∈ {8, 12, 16, 24}), **Δ_HS varies by less
  than 0.30 dB** and shows **no monotone trend** in the rank count.
- For admissible-rank count 5 (p = 4), Δ_HS **falls**, by **0.3–1.5 dB**,
  because 5 is close to the channel's own effective rank (mean `L̂` ≈ 4.6 at
  this cell) and the constraint stops being able to represent the channel.
  That is a **capacity** effect, not a resolution effect.

**Consequence if the prediction holds:** the 0.493 dB N=16-vs-N=64 residual is
**not** explained by selector grid coarseness, and the residual stays
unexplained — which is what Paper 1 must then continue to report.

### Falsifier

P17's prediction fails if Δ_HS depends **monotonically** on the admissible-rank
count across p ∈ {8, 12, 16, 24} with a total span **exceeding 0.5 dB**. That
outcome would support the quantisation account and I would report it as such.

---

## P18 — classical post-hoc versus interleaved (B5)

**Cell.** TD configuration at the principal cell (N = 32, K = 3, P = 30,
RSR 12 dB, six-point grid, paired worlds), comparing:

- **interleaved**: HS-GS, projection after every EM-GS update (`project_every=1`);
- **post-hoc**: EM-GS run to T, then **one** Cadzow projection at the same
  selected order `L̂`.

Both arms use the same `L̂` from the same held-out selection, so the contrast
isolates *placement*, not order choice.

### The question P18 asks

A2 measured this contrast on the **learned** side and found it **changes sign**:
post-hoc wins below +5 dB (−0.636 dB at 0–5 dB), interleaved wins above +10 dB
(+1.309 dB at 15–20 dB). Stage 3 reports the classical `H0 − U0` contrast as
one-signed (+1.574 dB). **Does the classical post-hoc/interleaved contrast also
change sign with SNR, or is it one-signed?**

### Prediction

**One-signed positive: interleaved beats one-shot post-hoc in every SNR bin,
with no sign change.** The learned sign flip has a mechanism the classical case
lacks — at low SNR the trained network has already absorbed what structure is
available, so an extra in-loop constraint fights it. EM-GS has no trained
component, so nothing competes with the projection.

Magnitudes, set low per the bias correction:

- pooled over the six points: **+0.3 to +1.5 dB**, point estimate **+0.8 dB**;
- the advantage **grows with SNR**, with the top bin exceeding the bottom bin
  by **0.2–1.5 dB**;
- **no bin** shows post-hoc significantly better.

### Falsifier

P18 fails if **any** SNR bin shows post-hoc better than interleaved with a
bootstrap CI excluding zero on the negative side. That would make the classical
contrast bin-dependent like the learned one, and Paper 1's prose claim that the
projection "must sit inside the iteration" would need qualifying by SNR.

---

## Recording rule

Each of P16–P18 is scored **held / failed** in the Part B report, stated
plainly either way, including the ones that cost us something.
