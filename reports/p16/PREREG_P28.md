# PROMPT 17 Part D — pre-registration P28

**Committed standing alone, before Tier 1 batch C1 starts.**

## The runs

Batch **C1**: two additional seeds on each arm of the focused-training Δ_H
contrast, four runs.

| arm | prior | training SNR | seeds to add |
|---|---|---|---|
| `C_U1_snr5_20` | none | `[5, 20]` | 2, 3 |
| `C_H1_snr5_20` | Hankel, rank 7 | `[5, 20]` | 2, 3 |

80k / 13 epochs, `N = 32`, `K = 3`, `L_k ~ U{3..7}`, `P = 20`, RSR 10 dB,
`init="spectral"`, one driver for both arms, matched on data order and
schedule. Seed 1 is the existing stage-4 pair, both arms
`init='spectral'`, both selected at epoch 9, verified in
`reports/p16/validity.csv`.

**This batch is the decisive one.** The whole retired collapse claim rested on
`+0.078` being a small number; whether it is a *stable* small number has never
been measured with valid arms. Note what C1 does and does not do: `+0.078` is
in the valid list, confirmed spectral from the checkpoint's own config, and
re-scored to `+0.0778` under the fixed harness. C1 supplies replication. It
does not re-establish the number.

## Predictions

### P28a — across-seed SD of Δ_H under focused training

**We still have no valid seed-variance estimate for Δ_H.** The 0.244 dB figure
came from `random`-init arms and is withdrawn. Tier 0.5 measures the seed
spread of a *different* contrast — loss design, an effect of order 1.9 dB —
and there is no reason a 0.078 dB structural contrast should share its scale.

Hedged per the amended standing rule, which exists because P19 was lost by
predicting `SD <= 0.12` and measuring `0.244`:

**Predicted: the across-seed SD of Δ_H at SNR >= 5 dB is at most 0.35 dB.**
Point estimate **0.18 dB**.

The band is deliberately loose and is set from the only anchor available: the
within-run bootstrap CI on seed 1 is `[+0.050, +0.112]`, a width of 0.062 dB,
and seed spread is widened from there by roughly a factor of six rather than
the small multiple that lost P19.

**Falsifier:** P28a fails if the across-seed SD exceeds **0.35 dB**.

### P28b — is `+0.078` representative of its own distribution?

A literal reading of "does `+0.078` sit inside the three-seed range" is
vacuous: seed 1 **is** `+0.078`, so it lies in `[min, max]` by construction.
Recorded here so the test is not later credited with passing something it
cannot fail.

The non-vacuous version:

**Predicted: `+0.078` is the middle of the three seed values, not the minimum
or the maximum.** Under the null that seeds are exchangeable, this holds with
probability 1/3, so it is a real test that we expect to fail two times in
three — and that asymmetry is the point. A single-seed number quoted as a
headline is a claim that it is typical.

**Falsifier:** P28b fails if `+0.078` is the smallest or the largest of the
three.

### P28c — sign stability

**Predicted: all three seeds give Δ_H > 0 at SNR >= 5 dB.** Corrected upward
per the amended rule's first clause: this is the magnitude of a learned effect,
and the historical pattern is that the trained arms do better than instinct
allows.

**Falsifier:** any seed gives Δ_H <= 0.

## Decision rule — fixed before the run

Let `m` be the three-seed mean and `s` the across-seed SD.

- **If `m - 2s > 0`**, Δ_H under focused training is separated from zero on
  three seeds. Paper 2 may state that the structural prior retains a small but
  non-zero advantage under focused training, quoting `m` with the seed spread
  alongside and every interval labelled with the quantity it covers.
- **If `m - 2s <= 0`**, the effect is not separable from seed noise on three
  seeds. Paper 2 must then say that the focused-training advantage is **within
  seed variation and cannot be distinguished from zero at this sample size** —
  not that it is zero, which three seeds cannot establish either.
- **If P28a fails** — SD above 0.35 dB — then seed variance dominates the
  contrast, and every single-seed Δ_H in the repository, `+1.209` included, is
  reported as a single draw. This is the outcome that would most change
  Paper 2, and it is registered here as a possibility rather than discovered
  afterwards.
- **If P28b fails**, `+0.078` is an extreme of its own small sample and must
  not be quoted as a point value without the range beside it.

## Gate

**If Δ_H moves materially across seeds, stop after C1 and report before
launching C2.** Operationally: if P28a fails, or if the three-seed range
exceeds 0.5 dB, batches C2–C4 do not launch this turn. Nothing downstream is
worth running against an unstable anchor.

## Recording rule

P28a, P28b and P28c are scored **held / failed** in the Part C report, stated
plainly either way. Every interval names the quantity it is over: test
realisations or seeds. Three seeds give two degrees of freedom, so the SD
itself carries large sampling uncertainty — any statement Paper 2 makes from it
must say it was evaluated on three seeds.
