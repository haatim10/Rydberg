# PROMPT 12 Part C — training runs, scored against P19–P21

Source of every number below: `reports/p12/partC_scores.json`, produced by
`scratch/p12_partC_eval.py` from the runs in `results/p12/partC/`. The
acceptance thresholds in that script were committed at `28323b7`, and the
pre-registrations at `04a38a2`, both **before any Part C run finished**.

All runs: 80k samples / 13 epochs, `N = 32`, `K = 3`, `L_k ~ U{3..7}`,
`P = 20`, RSR 10 dB, spectral init, matched on data order and schedule.

## Scorecard

| | Claim at risk | Result | Status |
|---|---|---|---|
| **P19** | seed noise is small against the 1.131 dB collapse contrast | SD **0.244 dB**, range **0.448 dB** | **FAILED** — and the decision rule fired |
| **P20** | the balanced loss removes the prior's apparent value | Δ_H **+0.395 dB** over SNR ≥ 5, **+1.308 dB** in the top bin | **FAILED** — the TENSION branch |
| **P21** | a log-domain loss reproduces the balanced result to ±0.5 dB | log loss **beats** balanced by **+0.584 dB** | **FAILED** — by overshooting |

Three for three against us. Each is written out below with what it costs.

---

## P19 — across-seed spread of Δ_H under focused [5,20] training

**Predicted:** across-seed SD ≤ 0.12 dB, range ≤ 0.25 dB, point estimate 0.07.

**Measured**, three seeds, Δ_H = U1 − H1 over [5,20] dB, paired per-trial median
on the identical test set:

| seed | Δ_H (dB) | CI95 over test realisations |
|---|---|---|
| 1 | **+0.690** | [+0.639, +0.747] |
| 2 | **+0.242** | [+0.210, +0.274] |
| 3 | **+0.634** | [+0.593, +0.671] |

- across-seed **SD 0.244 dB** — twice the 0.12 dB falsifier
- across-seed **range 0.448 dB** — nearly twice the 0.25 dB falsifier

**The decision rule fired.** It was fixed beforehand as: *if the spread is
≥ 0.5 dB, or if any seed's Δ_H under focused training exceeds +0.5 dB, the
fifteenfold-collapse claim is not supported.* Two of the three seeds exceed
+0.5 dB. The claim that the structural advantage collapses from 1.209 dB to
0.078 dB — a factor of 15.5 — **is not supported by these runs**, because
0.078 dB is one draw from a distribution whose other draws are 0.69 and 0.63.

The bias correction in the pre-registration was applied in the wrong direction
here. It talked me out of the hedge that seed noise might be ~0.3 dB, on the
grounds that hedging had been the losing pattern five times. The hedge was
right this time: the measured SD is 0.244 dB.

**Consequences, both already mandated by the pre-registration:**

1. Every CI in Paper 2 is over **test realisations**, not over seeds, and must
   be relabelled as such. Seed spread is a separate quantity and must be
   reported separately.
2. The collapse must be restated as what three seeds support. Measured
   like-for-like in one environment the collapse is **1.90×** (+1.310 →
   +0.690), not 15.5× — see `reports/p12/PART_A.md`, where `stage4`'s own
   `evaluate()` also failed to reproduce its stored rows (+0.682 against a
   stored +0.100) from unchanged code and unchanged checkpoints.

---

## P20 — Δ_H under the SNR-balanced loss, per bin

**Predicted:** Δ_H over SNR ≥ 5 dB in [−0.10, +0.35] dB, point estimate +0.10;
no bin above +0.50; pooled |Δ_H| ≤ 0.30. Direction: positive but negligible.

**Measured:**

| SNR bin | n | median Δ_H (dB) | CI95 | excludes 0 |
|---|---|---|---|---|
| [−10, −5) | 346 | −0.125 | [−0.164, −0.091] | yes |
| [−5, 0) | 354 | −0.386 | [−0.429, −0.326] | yes |
| [0, 5) | 317 | −0.582 | [−0.673, −0.419] | yes |
| [5, 10) | 331 | −0.183 | [−0.280, −0.090] | yes |
| [10, 15) | 336 | **+0.375** | [+0.272, +0.457] | yes |
| [15, 20) | 316 | **+1.308** | [+1.211, +1.434] | yes |

- over SNR ≥ 5 dB: **+0.395 dB**, CI [+0.315, +0.468] — above the +0.35 band
- pooled over all SNR: **−0.081 dB** (this one is inside the predicted ±0.30)
- top bin **+1.308 dB**, far above the +0.50 per-bin threshold

**This is the TENSION branch, and it was named in advance.** The
pre-registration states: *if Δ_H recovers above +0.35 dB, the recommendation
and the negative result are in tension and Paper 2 must say so explicitly: the
fix we advocate would restore the value of the prior we argued is illusory.*

That is what happened. Under the SNR-balanced loss — the loss Paper 2
recommends — the structural prior is worth **+1.308 dB at 15–20 dB**. The
negative result and the recommendation cannot both be stated flatly. Paper 2
must carry both, and must say which regime each belongs to: the prior's
apparent value is illusory *under a loss that lets low-SNR samples dominate the
gradient*, and is real *once that is fixed at high SNR*.

The per-bin structure is the informative part and was not predicted at all: Δ_H
is **negative and significantly so** in four of six bins, crosses zero between
5–10 and 10–15 dB, and only then climbs. A scalar summary over SNR ≥ 5 dB
(+0.395) hides a sign change inside its own range.

---

## P21 — does a per-sample loss taken in dB reproduce the balanced result?

**Predicted, committed to YES with a margin:** gradient-share span below 4;
at least +1.2 dB against the unbalanced baseline at SNR ≥ 5; **within ±0.5 dB**
of the balanced-weight run.

**Measured:**

| quantity | threshold | measured | verdict |
|---|---|---|---|
| gradient-share span (max/min) | < 4 | **3.32** | held |
| share below 5 dB | — | 0.359 | (ideal 0.5) |
| log vs unbalanced, SNR ≥ 5 | ≥ +1.2 dB | **+2.418** [+2.351, +2.506] | held |
| log vs balanced, SNR ≥ 5 | within ±0.5 dB | **+0.584** [+0.519, +0.646] | **failed** |

Per-bin gradient share under the log loss: 0.093, 0.121, 0.146, 0.152, 0.181,
0.308 — against a span of 30 for the ratio loss.

**P21 failed by overshooting.** Two of its three criteria held comfortably. The
third failed because the log loss did not merely match the balanced-weighting
run, it **beat it by +0.584 dB**, with a CI excluding both zero and the 0.5 dB
margin. The registered conclusion follows a fortiori and then some:

> *If P21 holds, Paper 2 must say plainly that a log-domain loss is a simpler
> route to most of the same benefit, and position the per-bin reweighting as
> one of at least two adequate fixes rather than as **the** fix.*

The measurement is stronger than that. On this evidence the log-domain loss is
not "one of two adequate fixes" — it is the better one, at less machinery. Per-
bin reweighting requires binning the SNR, choosing bin edges, and estimating
per-bin weights; `10 log10(·)` requires none of that and recovers 0.584 dB
more. Paper 2's framing has to change accordingly, and this cost was registered
before the run.

The mechanism given in the pre-registration is unaffected and is what the
gradient shares confirm: `d/dx log x = 1/x` divides out exactly the factor that
lets low-SNR samples dominate. Reweighting approximates that division with a
step function over bins; the log does it exactly, per sample.

---

## What Part C did and did not settle

**Settled.**
- Seed variance under focused training is 0.244 dB SD, large enough that any
  single-seed contrast at that scale is uninterpretable. [FACT]
- Under the balanced loss the prior is worth +1.308 dB at 15–20 dB, with a
  sign change inside the SNR range. [FACT]
- A log-domain per-sample loss flattens the gradient share to a span of 3.32
  and outperforms per-bin reweighting by +0.584 dB at SNR ≥ 5 dB. [FACT]

**Not settled.**
- Why Δ_H is *negative* and significant in the 0–5 and 5–10 dB bins under the
  balanced loss. Nothing in Part C explains it. [HYP: none offered]
- Whether the +0.584 dB log-over-balanced margin survives across seeds. It is
  one seed per arm, and P19 is precisely the finding that one seed is not
  enough. This is the clearest follow-up and it was not run — PROMPT 14
  authorises no training.
- Whether any of this transfers outside `N = 32, P = 20, RSR 10 dB`.

**Not run, and why.** The mixed-SNR seed replicates were last in the brief's
priority order and the budget went to C1–C3. The shortfall is reported here
rather than hidden, per the pre-registration's compute-reality note.

## Bearing on Paper 1

None of Part C changes any Paper 1 number. Paper 1's claims are classical and
come from configurations A–D of its Table I; Part C is entirely about the
trained loop. The one place they meet is the placement question — interleaved
versus post-hoc projection is a wash classically (six-point mean +0.003 dB,
`results/p12/B5.json`) while internal beats post-hoc by +1.309 dB at 15–20 dB
inside the trained loop (`reports/trackD_stage3_report.md:117`). That contrast
is recorded in `docs/structure-vs-training.md` and is deliberately **not** cited
in the Paper 1 letter, because the companion work is neither on arXiv nor under
review; see `docs/open-todos.md` item 3.
