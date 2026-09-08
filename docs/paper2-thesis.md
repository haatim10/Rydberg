# Paper 2 — what it can claim, and what it cannot

Written under PROMPT 16 Part C and committed **before** any decision about
Tier 1 retraining. Committing the framing before the budget is set is what
stops the budget being chosen to produce a headline rather than to answer a
question.

This supersedes the version specified in PROMPT 15, which must not be copied.
That version was built around four quantified nuisance sources; two of them
have since been withdrawn — the seed SD came from invalidly-initialised arms,
and the "evaluation path" discrepancy was a harness bug of my own making, not a
property of the system. A table of four nuisance sources would now overstate
what is known by a factor of two, so it is replaced below by an explicit
known / unknown split.

---

## 1. The thesis

> **A structural prior's reported value is not a property of the prior alone.
> It is inseparable from the training design, and the training design has
> almost never been controlled.**

That is the thesis Paper 2 can *motivate*. It is not the thesis Paper 2 can
currently *demonstrate*, and the difference matters enough to state at the top:
the demonstration needs Δ_H measured under two or more training designs with
correctly-initialised arms, and no such measurement exists. What Paper 2 can
demonstrate today is narrower and still worth publishing:

> **The conventional per-sample normalised loss, averaged over a wide SNR
> range, silently allocates almost all of its gradient to the low-SNR tail.
> This is measurable, it is large, and correcting it is worth more than most
> architectural additions being proposed for this problem.**

A diagnostic, a fix, and a distinction. Nothing about structural priors.

---

## 2. Known, and load-bearing

Each item below is a measurement on a correctly-initialised arm or on a loss
function. Every one is confirmed in `reports/p16/validity.csv`.

### 2.1 The gradient-share diagnostic [FACT]

Under the conventional loss, the fraction of the training gradient contributed
by each SNR bin, measured directly rather than argued:

| bin (dB) | gradient share |
|---|---|
| [−10, −5) | 0.4653 |
| [−5, 0) | 0.2929 |
| [0, 5) | 0.1384 |
| [5, 10) | 0.0606 |
| [10, 15) | 0.0272 |
| [15, 20) | 0.0155 |

<!-- src: reports/trackD_stage5_results.json, runs.C1_snr_balanced_P20.unweighted_shares.grad_share -->

**89.7% of the gradient comes from below 5 dB** (`sum(grad_share[:3]) =
0.8966`), and the share spans a factor of **30.0** (`0.4653/0.0155`) across the
range the model is asked to serve.

**This is the one result no training defect can reach.** It measures a loss
function evaluated on a training distribution — not a trained model, not a
checkpoint, not an initialiser. It would be identical if every network in this
repository were retrained from scratch.

> Note for the manuscript: the span is **30.0**. The current draft says 31.
> That is wrong and must be corrected wherever it appears.

### 2.2 Correcting it is worth +1.902 dB [FACT, single seed]

Balanced versus conventional loss, both `P = 20`, both `init='spectral'`,
paired per-trial median within each SNR bin:

| bin (dB) | gain (dB) | CI over test realisations | n |
|---|---|---|---|
| [−10, −5) | +0.039 | [+0.001, +0.080] | 346 |
| [−5, 0) | +0.044 | [−0.011, +0.105] | 354 |
| [0, 5) | +0.511 | [+0.432, +0.588] | 317 |
| [5, 10) | +1.353 | [+1.240, +1.439] | 331 |
| [10, 15) | +1.974 | [+1.849, +2.064] | 336 |
| [15, 20) | **+2.628** | [+2.515, +2.772] | 316 |
| **SNR ≥ 5** | **+1.902** | **[+1.823, +1.986]** | 983 |

<!-- src: reports/trackD_stage5_eval.json, P13_balanced_vs_uniform_P20.contrast -->

Three properties worth stating separately, because they are separately
falsifiable: the rise is **monotone** across all six bins; the top bin is
**+2.628 dB**; and there is **no low-SNR cost** — the two lowest bins are
non-negative, though the second one's interval includes zero.

**Every interval in that table is over test realisations, not over seeds.**
The seed spread is a different quantity and is not yet measured; see §3.1.
Conflating the two is exactly the error that produced the withdrawn P19, and
the manuscript must label every interval with the quantity it covers.

### 2.3 Pilot efficiency is not pilot-count generalisation [FACT]

The two differ by **+2.231 dB**, and the `P = 20` model is *worse* at `P = 25`
than at `P = 20`.

<!-- src: reports/trackD_prompt9_report.md:401; reports/trackD_stage5_eval.json, P14 -->

This is a distinction, not a gain, and it is the part of the paper least at
risk from anything in §3: it says that a number reported under one pilot count
does not transfer to another, which is a claim about evaluation practice rather
than about any model.

---

## 3. Unknown, and stated as such

### 3.1 The seed variance of everything in §2 [UNKNOWN]

We have **no valid seed-variance estimate for any quantity in this
repository.** The only figure we ever had, 0.244 dB, came from arms trained
with the wrong initialiser and is withdrawn.

PROMPT 16 Part B is measuring exactly one of these — the seed spread of the
+1.902 dB gain, over three seeds — and P27 is pre-registered in
`reports/p16/PREREG_P27.md`. Until it reports, **+1.902 dB is a single-seed
result** and the manuscript must not imply otherwise.

The decision rule is fixed in advance and repeated here so it cannot be
softened afterwards: if the seed spread is large enough that the three-seed
mean is within two standard deviations of zero, the headline is single-seed and
the paper says so **in those words**, in the abstract and at first quotation.

### 3.2 The value of the structural prior under any training design [UNKNOWN]

This is the largest gap and the one the paper must be honest about, because it
is what the earlier draft was built on.

Three claims are retired and **nothing replaces them this turn**:

| retired claim | why |
|---|---|
| the fifteenfold collapse (1.209 → 0.078) | the experiment meant to establish that 0.078 is not one draw from a wide distribution was invalid |
| the tension branch — that the recommended loss restores the prior's value | re-scored, Δ_H over SNR ≥ 5 is +0.174 dB, *inside* the pre-registered band; and the arm behind it was mis-trained anyway |
| the log-over-balanced margin of +0.584 dB | re-scored it is +0.343 dB, inside the ±0.5 dB equivalence margin; and the arm was mis-trained |

The two *numbers* 1.209 and 0.078 are sound — both from correctly-initialised
stage-3 and stage-4 arms. What has no support is the **interpretation** that
their ratio is a collapse rather than a single-seed difference of unknown
significance.

**Consequence for the manuscript.** Paper 2 may not claim that the structural
prior's advantage is illusory, nor that it is real. It may report that the
question is open and that answering it requires the seed distribution the
retired experiment failed to measure.

### 3.3 Which fix is better [UNKNOWN]

Per-bin reweighting is the **diagnostic**: the weights are explicit, and that
is what makes the gradient share measurable and the mechanism demonstrable at
all. Whether a **log-domain per-sample loss** is a better *fix* is currently
unsupported. The mechanism is appealing — `d/dx log x = 1/x` divides out
exactly the factor that lets low-SNR samples dominate, where reweighting
approximates that division with a step function over bins — but the only arm
that tested it was trained with the wrong initialiser.

**It must not be asserted until arms trained under the correct initialiser say
so.** The mechanism may be stated as motivation; the comparison may not.

---

## 4. What Tier 1 would buy, and why it was deferred

**Tier 1** is twelve runs: Δ_H — the structural-prior contrast — measured under
two training designs (mixed-SNR and balanced), three seeds per arm, both arms
per design, all with `init="spectral"`. Roughly 31 CPU-hours, about 8 hours
elapsed at four-way parallelism.

**What it would buy.** It is the only thing that converts §1's motivating
thesis into a demonstrated one. With it, Paper 2 could state that the same
prior, the same architecture and the same data budget yield materially
different reported advantages under two training designs, with seed spread
quantified so the difference is separable from noise. Without it, the paper
argues that the training design *ought* to be controlled and demonstrates the
point only on the loss function, not on the prior.

**Why it was deferred.** PROMPT 16 scoped this turn to Tier 0.5 — four runs
seeding the headline the paper already has — on the reasoning that a headline
with no seed spread is a liability regardless of what Tier 1 shows, and that
Tier 1's result is uninterpretable without a seed-variance estimate to compare
it against. Sequencing the cheap measurement first is defensible. It is also a
choice, and this paragraph exists so that it is a choice on the record rather
than a gap someone notices later.

**Standing recommendation.** Tier 1 belongs in the thesis chapter, where the
page budget can carry an open question and the compute is not competing with a
letter's deadline. If Paper 2 is submitted as a letter first, §3.2 must appear
in it as a stated limitation, not omitted.

---

## 5. Mechanical requirements for the manuscript

1. **Every interval is labelled with the quantity it is over** — test
   realisations or seeds — at every occurrence, not once in a methods note.
2. **The span is 30.0, not 31.** Correct it wherever it appears.
3. **No claim about the structural prior's value** survives from the earlier
   draft. The quarantined file `wip/spl2/` asserts the retired collapse in
   three places and carries a banner saying so.
4. **The title changes.** It is no longer about a prior that fails under
   control. On the current inventory it is about what a reported gain is a
   property of — and, more narrowly, about a loss convention that misallocates
   its gradient by a factor of thirty.
5. **Venue.** The measurement-of-practice framing fits IEEE TMLCN better than
   SPL. But that judgement was made when there were four quantified nuisance
   sources; with two withdrawn and the prior's value open, a letter reporting
   the diagnostic and the fix is the more honest scope until Tier 1 runs.
