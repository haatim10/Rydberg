# Paper 2 — what it can claim, and what it cannot

Written under PROMPT 16 Part C and committed **before** any decision about
Tier 1 retraining. Committing the framing before the budget is set is what
stops the budget being chosen to produce a headline rather than to answer a
question.

> ## REVISION 2 — 2026-09-11, under PROMPT 20 Part C
>
> **Reason: a prior-art hit, plus the Tier 1 measurements landing.** Two
> different kinds of new information, and they are kept apart below because
> they carry different risks.
>
> **Sections 1 through 5 are left exactly as first written.** That is
> deliberate. This document's value is that it predates the numbers, and
> editing it in place to match them would destroy the only property that makes
> it worth having. Everything new is in **§6**, which says which of the
> original sections it supersedes.
>
> **This is not a retrofit, and here is the test of that.** The framings were
> written down in advance as the three outcomes of `reports/p17/PREREG_P29.md`,
> committed standing alone before batch C2 ran, with numeric triggers evaluated
> in a fixed order. Outcome (a) was selected *by that rule*, not chosen after
> the fact. The prior-art hit is a different matter: it is genuinely new
> information that no pre-registration could have anticipated, and it changes
> the contribution split rather than any measurement.

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

**The absence of a low-SNR cost is stronger than it looks, and is now the
paper's most interesting single result.** Across all three seeds the balanced
arm is better in **18 of 18 bin-by-seed cells**, 17 of them with a CI excluding
zero. The two lowest bins carry weights of 0.056 and 0.111 — down-weighted by
18× and 9× relative to unit mean — and the balanced arm still wins there
(three-seed means +0.052 and +0.202 dB, every seed positive). The reweighting
therefore does **not** trade low-SNR accuracy for high-SNR accuracy, which is
what a reweighting is usually suspected of. It improves the quantity it was
weighted away from. [FACT, `reports/p16/PART_A_P27.md`]

A reading, offered as hypothesis and not established: the conventional loss was
so dominated by the low-SNR tail that the network underfit everywhere, and
rebalancing stops wasting capacity rather than relocating it. [HYP]

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

**Measured for one quantity as of PROMPT 17.** The seed spread of the
+1.902 dB gain, over three seeds, pre-registered as P27 in
`reports/p16/PREREG_P27.md` and scored in `reports/p16/PART_A_P27.md`:

| seed | gain at SNR ≥ 5 (dB) | CI over test realisations |
|---|---|---|
| 1 | +1.9024 | [+1.823, +1.986] |
| 2 | +2.2427 | [+2.158, +2.322] |
| 3 | +2.3691 | [+2.257, +2.452] |

Three-seed mean **+2.171**, SD **0.241**, range **0.467**, and
`mean − 2·SD = +1.689 > 0`, so the effect is separated from no-effect on three
seeds. P27a and P27b both held.

**Two caveats the manuscript must carry.** Three seeds give two degrees of
freedom, so the SD has large sampling uncertainty and any statement made from
it must say it was evaluated on three seeds. And seed 1 is the outlier of the
three, with its two arms trained by different drivers — the spread above is not
yet quotable as a *seed* spread until the unified-driver re-run of seed 1
lands. See `reports/p16/SEED1_CONDITIONAL.md`.

**Every other number in §2 remains unreplicated.** +2.231 and the pilot-count
results are still single-seed.

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

## 4b. Training budget, selection rule, and selected epoch — three things, not one

Table I of the quarantined draft reports "Training samples / epochs
`80,000 / 13`" as though that were the whole story. Four different epoch
numbers are in play and they are all consistent, but only if they are named
separately. The rewrite must split the single Table I row into three.

| quantity | value | where it comes from |
|---|---|---|
| **config default** | `epochs = 50` | `TrackDConfig().train.epochs` — **never used.** Both `stage2.py` and `stage5.py` override it with a module constant `EPOCHS = 13`. Dead config; do not report it. |
| **training budget** | **13 epochs**, 80,000 samples | the number actually trained. This is the `80,000 / 13` in Table I. |
| **selection rule** | **one standard error** | `select_epoch(..., "one_se")`, `stage2.py:81`. Picks the earliest epoch whose validation NMSE is within one SE of the best, not the best itself. |
| **best-validation epoch** | 9 (`B3_80k_13ep`), 12 (`C1_snr_balanced_P20`) | `selection.best_epoch` |
| **selected epoch** | **6** (`B3_80k_13ep`), **8** (`C1_snr_balanced_P20`) | `chosen_epoch`, and what `best.pt` actually holds |

Two consequences the draft does not currently state.

**The two arms of the `+1.902` contrast were selected at different epochs** —
6 and 8. That is what the one-SE rule is for and it is not a defect, but a
reader comparing two checkpoints is entitled to know they were not stopped at
the same point. It must be in the table.

**"Epoch 9 in both compared checkpoints" describes a different pair.** That is
the stage-4 focused-training pair (`C_U1_snr5_20`, `C_H1_snr5_20`), both
selected at epoch 9. It is not the pair behind `+1.902`. The two are easy to
confuse and the manuscript should not.

For the PROMPT 16 Tier 0.5 runs the same rule gave selected epochs 6, 6, 8, 8
against 13 trained — the unbalanced arms consistently earlier than the balanced
ones, which is itself a small piece of evidence that the balanced objective
keeps improving for longer.

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

---

# 6. REVISION 2 — what supersedes what

Added 2026-09-11 under PROMPT 20 Part C. Sections 1–5 above are the record as
it stood before the Tier 1 measurements and before the prior-art hit. This
section says where they are now wrong.

## 6.1 The contribution split, after the prior-art hit

Wiesmayr, Marti, Dick, Song and Studer, *"Bit Error and Block Error Rate
Training for ML-Assisted Communication"* (arXiv:2210.14103 v3, 2023),
**Section 4, "SNR Deweighted Training"**, states the loss-imbalance mechanism
this paper was built on and proposes per-condition deweighting as the remedy.
Full classification in `reports/p20/claim_audit.md`.

### Established elsewhere — cite, do not establish

- The aggregate loss over a wide SNR range is dominated by low-SNR samples,
  because a small relative improvement at low SNR moves the cost more than a
  large one at high SNR.
- The consequence: the high-SNR regime is underfitted.
- The remedy: reweight the loss by SNR condition.

**§2.1 above is superseded in framing, not in fact.** The gradient-share
numbers stand — 89.7% below 5 dB, a span of 30.0 — and they are still the one
result no training defect can reach. What changes is that they measure the
*magnitude in this system* of a published phenomenon, in an NMSE loss rather
than the BER/BLER losses Wiesmayr treat. They are not a discovery.

### This paper's contribution

1. **That the pathology corrupts attribution.** Wiesmayr identify and fix it,
   for detection and decoding. They do not ask what it does to the *evidence*
   for structural priors: that a known training defect manufactures apparent
   support for a prior, and that crediting a prior without controlling for it
   is unsound. This is now the paper's claim.
2. **The matched-adequacy control**, and the fact that it was run and
   overturned our own result.
3. **The sign-pattern finding** — `- - - + + +`, the prior costing accuracy
   below +5 dB and buying it above, replicating on every seed of two designs
   with every CI excluding zero.
4. **The averaging-window finding** — the sign of the reported gain depends on
   how the average is taken, and the reason is measured rather than asserted.
5. **The pilot-curve ambiguity.**

### Secondary, and stated as secondary

The remedy **transfers** to unrolled, magnitude-only channel estimation:
`+2.171` dB at SNR ≥ 5 across three seeds, winning **all 18 bin×seed cells**
including the two it down-weights by about 18× and 9×.

**That last detail may or may not still be novel.** The supplied description of
Wiesmayr §4 does not say whether they report the effect on the *deweighted*
regime. It is `[UNVERIFIED]` and is item 5.1 of `docs/open-todos.md`. It is the
single claim whose status most changes what remains here.

**§2.2 above is superseded in status.** `+2.171` was written there as the
paper's headline. It is no longer the headline; it is confirmation.

## 6.2 What is no longer unknown

**§3.1 is superseded.** Seed variance is now measured for three quantities, all
on three seeds with `init='spectral'`, all pre-registered before the runs:

| quantity | three-seed mean | SD over seeds | source |
|---|---|---|---|
| balanced-vs-conventional loss gain | +2.171 dB | 0.241 | `reports/p16/PART_A_P27.md` |
| Δ_H, mixed-SNR training | +1.286 dB | 0.072 | `reports/p19/PART_B.md` |
| Δ_H, focused training | +0.020 dB | 0.083 | `reports/p17/PART_C_C1.md` |

The seed-1 driver confound §3.1 warns about was resolved at exactly **0.0000 dB**
on bitwise identical weights, so those spreads are quotable as seed spreads.

**§3.2 is superseded, and this is the largest change.** It said the structural
prior's value under any training design was unknown and that Paper 2 could
claim neither that the advantage is illusory nor that it is real. Both designs
are now measured on three seeds each:

- **Mixed-SNR training:** +1.286 dB, SD 0.072, `mean − 2·SD = +1.142`, positive
  on every seed and under both pooling rules. Real, stable, substantial.
- **Matched focused training:** +0.020 dB, SD 0.083, `mean − 2·SD = −0.147`,
  one of three seeds significantly negative, ratio-of-sums negative on all
  three. No stable sign.

So the paper may now say what §3.2 forbade — but not in the shape the retired
draft said it. The claim is **not** that a number shrank by a factor. It is
that the same contrast **changes category** when both arms are trained
adequately in the regime of interest.

**The retired claims stay retired.** The fifteenfold collapse is not
rehabilitated by outcome (a); it was a bad description of what had been
measured, and it remains one.

One correction to §3.2's own text while we are here: it says "the two *numbers*
1.209 and 0.078 are sound". That was true when written and is now misleading.
`+1.209` is sound and, it turns out, **conservative** — it is the *smallest* of
its three seeds (P29b failed in that direction). `+0.078` is sound as a
measurement of seed 1 and misleading as a quantity — it is the *largest* of its
three, and its sign survives neither a change of seed nor of pooling rule.

**§4 is superseded.** Tier 1 was not deferred to the thesis chapter; it ran.
C1 and C2 are complete, C3 is running, C4 follows. The standing recommendation
in §4 to carry §3.2 as a stated limitation no longer applies in that form.

## 6.3 What is still unknown

**§3.3 stands unchanged.** Whether a log-domain per-sample loss is a better fix
than per-bin reweighting is still unsupported, and batch C4 is the test.

**C4 matters more than it did.** Wiesmayr deweight by condition. A log-domain
per-sample loss divides out the scale factor exactly rather than by a step
function over bins, and needs neither bin edges nor weight estimation. If the
log loss beats bin reweighting, that is a **delta on published work** rather
than a replication of it. If it does not, the paper says so.

Two items added to the unknown column, neither of them measurable here:

- **Whether anyone has applied SNR deweighting to unrolled estimators
  specifically.** Not searched. It bears on how much of `+2.171` is a transfer
  result and how much is a re-run.
- **Whether the pilot-curve train-versus-evaluate ambiguity has been identified
  before.** **Not searched at all** — so its novelty is unsupported rather than
  supported. Highest-priority item in `docs/open-todos.md`.

**The standing caveat on all of §6.1:** the prior-art check was three web
searches, not a systematic sweep.

## 6.4 Mechanical requirements — amendments to §5

- **§5 item 2 is already discharged.** The span reads 30.0 in both places in
  the draft; it was corrected under PROMPT 16 Part D. PROMPT 20's brief
  restates it as outstanding, and it is not.
- **§5 item 4, the title, still stands and is still blocked.** But the reason
  has changed: it is no longer blocked on the structural prior's value, which
  is now measured. It is blocked on C3, and on the fact that the paper's
  subject is now attribution rather than the loss convention.
- **New requirement:** every anticipated claim carries its citation at first
  mention, and the manuscript nowhere implies the loss-imbalance mechanism was
  identified here.
