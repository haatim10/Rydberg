# PROMPT 21 Part A — pre-registration P31 (a **RE-REGISTRATION**)

**Committed standing alone, before Tier 1 batch C4 starts.**

## This is a re-registration and is correspondingly weaker

The original P21 was scored on an arm trained with `init='random'` while its
comparison arms were trained with `init='spectral'`. That arm is invalid
(`reports/p16/validity.csv`) — **but its numbers are not unseen.** For exactly
the quantities C4 measures, it reported:

| quantity | invalid `random`-init value |
|---|---|
| log − balanced, SNR ≥ 5 | **+0.343** dB, CI [+0.306, +0.392] |
| log − unbalanced, SNR ≥ 5 | **+2.228** dB, CI [+2.170, +2.310] |
| realized gradient-share span under the log loss | **3.32** |
| realized share below 5 dB under the log loss | **0.359** |

Every prediction below is made **knowing those four numbers**. The original P21
used a ±0.5 dB equivalence margin and its committed answer was "yes, the log
loss reproduces the balanced result"; on re-scoring, the verdict flipped from
FAILED to HELD on the re-evaluation alone. This test is therefore weaker than
the original was, and no verdict from it may be quoted without that label.

---

## Two discrepancies, read off disk before writing this

### D1. P31c as the brief specifies it **cannot be scored this turn**

The brief asks P31c to predict *"the per-bin sign sequence of Δ_H under
log-loss training."* Δ_H is the structural-prior contrast: an arm carrying the
Hankel projection against an otherwise identical arm without it.

**Batch C4 contains no prior-carrying arm.** From `scratch/p17_tier1.py`:

```
C4_U1log_seed1 {'hankel': False, 'snr': (-10.0, 20.0), 'loss': 'logdb', 'seed': 1}
C4_U1log_seed2 {'hankel': False, 'snr': (-10.0, 20.0), 'loss': 'logdb', 'seed': 2}
C4_U1log_seed3 {'hankel': False, 'snr': (-10.0, 20.0), 'loss': 'logdb', 'seed': 3}
```

All three are `hankel: False`. Measuring Δ_H under log-loss training needs a
matching `H1-log` arm on each seed — three further runs — and PROMPT 21's scope
section forbids *"any training run beyond C4's three."*

**P31c is therefore registered as UNSCOREABLE this turn**, with the reason
recorded rather than the prediction quietly dropped or the batch quietly
widened. What would close it: three `H1-log` runs at seeds 1–3, about three
hours, giving a fourth design for the sign sequence. That is a decision for a
later turn, and `docs/open-todos.md` records it.

A substitute that **is** measurable from C4 is registered below as **P31c′**.
It is not the same test and is not scored as though it were.

### D2. `reports/p15/rescored.csv` line 16 carries a wrong validity reason

That line marks the P21 gradient-share span `3.3229` **VALID**, with the reason
*"a property of the loss function, not of any trained model — unaffected."*

The reason does not hold. `scratch/p12_partC.py:134` computes the span by
back-propagating a loss **through a specific trained model** and taking
per-bin gradient norms; it takes `model` as its first argument. The span
therefore depends on the weights, and those weights came from an arm trained
with the wrong initialiser.

This does not change any verdict — P21's span clause was scored against a
threshold of 4.0 and 3.32 passed — but the value is **not** validated by the
reason given, and C4 re-measures it on correctly-initialised arms anyway. The
row should read INVALID, or VALID for a different reason that someone can
defend. Recorded here rather than edited, because `rescored.csv` is a record of
what was believed at the time.

---

## The runs

Batch **C4**: `U1` under the log-domain per-sample loss, three seeds.

| arm | prior | loss | training SNR | seeds |
|---|---|---|---|---|
| `C4_U1log_seed{1,2,3}` | none | log-domain per sample | `[-10, 20]` | 1, 2, 3 |

80k / 13 epochs, `N = 32`, `K = 3`, `P = 20`, RSR 10 dB, `init="spectral"`,
one driver, matched on data order and schedule. The loss is
`mean(10 log10(‖Ĝ−G‖²/‖G‖²))` taken per sample before averaging, against the
bin-reweighted arm's `mean(w(b)·‖Ĝ−G‖²/‖G‖²)`.

Contrasts scored, all against arms already in hand:

- **log − balanced**: `C4_U1log_seedN` against `results/p16/tier05/C1_seedN`;
- **log − unbalanced**: `C4_U1log_seedN` against `results/p16/tier05/U1_seedN`.

## Why this batch carries more weight than its predecessors

Wiesmayr *et al.* deweight **by condition**, which needs bin edges, a
pre-training measurement pass to estimate the weights, and a choice of
binning. A log-domain per-sample loss divides out the scale factor **exactly,
per sample**, with none of those. If it beats bin reweighting, Paper 2 has a
delta on published work rather than a replication of it. If it does not, the
paper says so plainly: bin reweighting is adequate and the log loss is a
simpler alternative that is no better.

---

## Predictions

### P31a — the log-over-balanced margin, and its seed spread

Three across-seed SDs have now been measured, on three different contrasts:
`0.072`, `0.083` and `0.150`. Per the amended rule — hedge on anything
concerning noise, spread or reproducibility — there is no basis for assuming
this arm is tighter than the loosest of those.

**Predicted (i): the across-seed SD of the log − balanced margin at SNR ≥ 5 is
at most 0.35 dB.** Point estimate **0.15 dB**.

**Predicted (ii): the three-seed mean margin lies within ±0.50 dB of zero** —
that is, the two fixes are equivalent within the same margin the original P21
used. Point estimate **+0.15 dB**, corrected downward from the invalid arm's
`+0.343` because that arm's comparison partner was also mis-trained, so its
margin is a difference of two errors rather than a measurement of one.

**Falsifiers:** P31a(i) fails if the across-seed SD exceeds **0.35 dB**.
P31a(ii) fails if the three-seed mean margin lies outside **±0.50 dB**.

### P31b — the realized gradient share under the log loss

Anchors, all on correctly-initialised arms except the last:

| loss | share below 5 dB | span (max/min) |
|---|---|---|
| conventional per-sample | 0.897 | **30.0** |
| bin-reweighted | 0.427 | **1.9** |
| log-domain (invalid arm) | 0.359 | 3.32 |

The brief asks specifically whether the log loss's span **falls below the
bin-reweighted arm's**.

**Predicted: it does not.** The span under the log loss is **at least 1.9**,
point estimate **2.8**.

The reasoning, stated so it can be judged wrong: taking the error in decibels
divides out the scale factor exactly, but it does not equalise the *gradient*,
because the derivative of a logarithm is itself inversely proportional to the
error, and the per-bin error is not constant across the range. Bin reweighting
targets the share directly and can therefore flatten it further, at the cost of
needing the bin edges and the measurement pass. The invalid arm's 3.32 points
the same way, and is weak evidence given what invalid arms have done to other
numbers here.

**Falsifier:** P31b fails if the measured span is **below 1.9**. That outcome
would say the log loss flattens the gradient better than explicit reweighting
while needing none of its machinery, which is the strongest possible version of
the C4 result and would deserve the title.

### P31c — UNSCOREABLE THIS TURN

See D1. Registered so its absence is visible. `- - - + + +` has held on three
training designs and nine seeds; whether it holds on a fourth is not answered
by this batch and must not be implied to be.

### P31c′ — the per-bin shape of the log-over-balanced margin

The substitute, and a genuine test. The bin-reweighted arm **over-corrects**:
its realized share below 5 dB is `0.427` against an ideal of `0.5`, so it
slightly over-serves the high-SNR end. A per-sample log loss divides the scale
out exactly and has no such tilt.

**Predicted: the log-over-balanced margin is larger at low SNR than at high
SNR on all three seeds** — specifically, the `[-10,-5)` bin value exceeds the
`[15,20)` bin value on each seed independently.

**Falsifier:** P31c′ fails if the `[15,20)` bin exceeds the `[-10,-5)` bin on
two or more seeds.

Under exchangeability of the six bins this is not a near-certainty: the
prediction picks a direction for a difference that could as easily run the
other way, and the invalid arm's per-bin values were never recorded, so this
one is closer to blind than the rest of P31.

---

## Decision rule — fixed before the run

Let `m` be the three-seed mean log − balanced margin and `s` its across-seed SD.

- **If `m − 2s > 0`**, the log loss beats bin reweighting on three seeds.
  Paper 2 reports a fix that improves on the published remedy, and the title
  may claim both an attribution finding and an improved fix.
- **If `|m| ≤ 0.50` and `m − 2s ≤ 0`**, the two are equivalent within the
  registered margin. Paper 2 reports bin reweighting as adequate and the log
  loss as a **simpler alternative that is no better** — in those words — and
  the title claims an attribution finding with a replication attached.
- **If `m + 2s < 0`**, the log loss is worse. Paper 2 says so and recommends
  bin reweighting.
- **If P31a(ii) fails** — the margin exceeds ±0.50 dB in either direction —
  the equivalence framing is abandoned and the measured direction is reported
  as the result.

**The title is written from this rule, not from a reading of the numbers.**

## Recording rule

P31a, P31b and P31c′ are scored **held / failed** in the Part B report, stated
plainly either way, and every one is labelled a re-registration wherever it is
quoted. P31c is reported as unscoreable with its reason. Per-seed values are
reported **per bin and under both pooling rules**, against both comparison
arms. Every interval names the quantity it is over: test realisations or seeds.
Three seeds give two degrees of freedom.
