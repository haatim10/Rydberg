# PROMPT 12 Part C — pre-registrations P19, P20, P21

**Committed standing alone, before any Part C training run starts.**

All runs: 80k samples / 13 epochs, N = 32, K = 3, L_k ~ U{3..7}, P = 20,
RSR 10 dB, spectral init, matched on data order and schedule to the existing
arms. Only the seed and the stated variable change.

**Bias correction applied.** Across stages 1–5 every prediction miss ran the
same way: I underestimated what the learned components work out for themselves
and over-attributed limits to structural or gradient machinery. All three
magnitudes below are set to reflect that — each predicts the *learned* side
doing better, and the *machinery* mattering less, than my uncorrected instinct.

**Compute reality, recorded now.** A single 80k/13-epoch run costs
7,900–11,100 s on this machine (`reports/trackD_stage5_results.json`,
`train_seconds`). The brief's priority order is followed exactly: C1's focused
pair first, then C2, then C3, then the mixed-SNR seeds if budget remains. If
budget runs out, the shortfall is reported as a limitation, not hidden.

---

## P19 — across-seed spread of Δ_H under focused [5,20] training

**Runs.** `C_U1_snr5_20` and `C_H1_snr5_20` (stage-4 specs) on two additional
seeds, giving three seeds per arm. Δ_H = U1 − H1 over [5,20] dB, paired
per-trial median, on the identical test set.

**The claim at risk.** Paper 2 says the structural advantage collapses
fifteenfold, 1.209 → 0.078 dB. That survives only if seed noise is small
against the 1.131 dB contrast.

### Prediction

**The across-seed standard deviation of Δ_H under focused training is ≤ 0.12 dB,
and the full range across the three seeds is ≤ 0.25 dB.**

Point estimate for the SD: **0.07 dB**.

Justification: the arms are matched on data order, schedule and
initialisation, so seed enters only through the parameter draw and the
stochastic optimiser path. Stage-4's own within-run bootstrap CI on Δ_H was
[+0.050, +0.112], a width of 0.062 dB; seed spread is typically the same order
as, or a small multiple of, that.

**Under the bias correction** I explicitly do *not* predict the spread is large
enough to rescue the structural claim. My uncorrected instinct was to hedge
toward "seed noise might be ~0.3 dB and could swallow 0.078" — that instinct is
the exact pattern that has been wrong five times, so it is discarded.

### Falsifier

P19 fails if the across-seed SD of Δ_H exceeds **0.12 dB**, or the range
exceeds **0.25 dB**.

### Decision rule — fixed before the run

If the across-seed spread is **comparable to the 1.209 → 0.078 contrast**
(operationally: if the spread is ≥ 0.5 dB, or if any seed's Δ_H under focused
training exceeds +0.5 dB), **the fifteenfold-collapse claim is not supported and
Paper 2 must be rewritten around what remains.** That outcome will be reported
plainly if it occurs.

Separately, and regardless: once seed results exist, every CI currently in
Paper 2 must be **relabelled as over test realisations**, with seed spread
reported as its own quantity.

---

## P20 — Δ_H under the SNR-balanced loss, per bin

**Run.** `H1` trained with the SNR-balanced loss over the full [−10, 20] range,
matched to the existing balanced `U1` run (`C1_snr_balanced_P20`) on seed, data
order and schedule. Report Δ_H per bin with CIs.

**Why this is the closing experiment.** Paper 2 recommends the balanced loss but
has never measured Δ_H under it. We have Δ_H under mixed-SNR training (+1.209
dB) and under focused [5,20] training (+0.078 dB), and neither is the regime we
tell people to adopt.

### Prediction

**Δ_H under the balanced loss is small and close to the focused-training value,
not the mixed-training value.**

- Over SNR ≥ 5 dB: **Δ_H in [−0.10, +0.35] dB**, point estimate **+0.10 dB**.
- Per bin: no bin exceeds **+0.50 dB**; the top bin (15–20 dB) is the largest.
- Pooled over all SNR: **|Δ_H| ≤ 0.30 dB**.

Direction: **positive but negligible.** The balanced loss removes the same
high-SNR underfitting that focused training removed, so it should remove the
same apparent structural advantage. This is the outcome under which Paper 2's
recommendation and its negative result **agree**, and the paper is complete.

**Bias correction:** my uncorrected instinct was that balancing is a weaker
intervention than focused training (it reweights rather than restricts, and it
slightly over-corrects to a 0.427 share against an ideal 0.5), so the prior
might retain more value — perhaps +0.4 to +0.6 dB. That is the
over-attribute-to-machinery pattern again. Corrected down to +0.10 dB.

### Falsifier

P20 fails if Δ_H over SNR ≥ 5 dB **exceeds +0.35 dB** or falls **below
−0.10 dB**.

**If Δ_H recovers** (> +0.35 dB), the recommendation and the negative result are
in **tension** and Paper 2 must say so explicitly: the fix we advocate would
restore the value of the prior we argued is illusory.

---

## P21 — does a per-sample loss taken in dB reproduce the balanced result?

**Run.** `U1` trained with the per-sample loss taken in decibels,
`10 log10(‖Ĝ−G‖²/‖G‖²)`, rather than as a normalised ratio. Report the realised
per-bin gradient share, and the per-bin gain against **both** the unbalanced
baseline and the balanced-weight run.

**Why.** "Why not just use a log loss?" is the single most likely objection to
the proposed fix. It must be answered with a measurement or conceded.

### Prediction — **committed to YES, with a margin**

**Yes: the log-domain loss substantially reproduces the balanced-weighting
result.** Specifically:

- The realised per-bin gradient share under the log loss is **far flatter than
  the ratio loss** — max/min ratio **below 4**, against 30 for the ratio loss
  (`reports/trackD_stage5_results.json`, `unweighted_shares.grad_share`, span
  0.4653/0.0155).
- Against the unbalanced baseline, the log loss gains **at least +1.2 dB at
  SNR ≥ 5 dB**.
- Against the balanced-weight run, the log loss lands **within ±0.5 dB at
  SNR ≥ 5 dB** — i.e. the two are close, and the margin by which I claim
  "reproduces" is **0.5 dB**.

Mechanism: taking the log makes the per-sample loss scale-free in the same way
the per-bin reweighting does. `d/dx log x = 1/x` divides out precisely the
factor that lets low-SNR samples dominate.

**Bias correction:** the instinct to protect our own proposed fix by predicting
the log loss underperforms is exactly the over-attribute-to-our-machinery
pattern. Committed to **yes**.

### Falsifier

P21 fails if the log loss falls **more than 0.5 dB short** of the balanced-weight
run at SNR ≥ 5 dB, **or** if its per-bin gradient-share span exceeds **4**.

**If P21 holds**, Paper 2 must say plainly that a log-domain loss is a simpler
route to most of the same benefit, and position the per-bin reweighting as one
of at least two adequate fixes rather than as *the* fix. That is a real cost to
the paper's framing and it is registered here before the measurement.

---

## Recording rule

Each of P19–P21 is scored **held / failed** in the Part C report, stated plainly
either way, including the ones that cost us something.
