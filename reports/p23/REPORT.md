# PROMPT 23 — report

Branch `paper2-style`, branched from `739dba4` (the end of PROMPT 22).

**Paper 1 untouched this turn:** `git diff 739dba4 -- paper/thesisproj/` is
empty. (Against `f76ed63` it is *not* empty, but that is PROMPT 22's work, not
this turn's — `f76ed63` is the Paper 2 number baseline, not the Paper 1 freeze
point. An earlier draft of this report cited the wrong one.)

**Scope note, stated because the brief left it open.** Part C says Paper 2 must
read like Paper 1, and Paper 1's artefact is `paper/thesisproj/`. So the full
style rewrite was applied to its counterpart, `paper/thesisproj2/`. Part B's
three items and *all* of Part A's content went into **both** files, since those
are facts rather than style. `wip/spl2/` keeps its compressed submission voice
and its 23 `% src:` comments.

---

## 0. The thing that matters most: a wrong digit, found and fixed

**Table II's balanced column carried $-0.704$ for the $[-5,0)$ dB bin. The
correct value is $-0.703$.**

- Exact three-seed mean from `reports/p20/p30_score.json`: $-0.70346$.
- `reports/p20/PART_B_C3.md` table B3.4 already carried $-0.703$.
- `reports/p22/review_response.md`, which I wrote last turn, mis-transcribed it
  as $-0.704$, and I copied it from there into the manuscript.

Fixed in the manuscript and in `review_response.md`, which now carries a dated
correction note. The other five values in that column verify exactly. The
column now carries a `% src:` comment naming all three sources so it cannot
drift again.

**On the standing rule.** "No number may change" exists to stop a style rewrite
altering results silently. It cannot require preserving a transcription error
that contradicts both the raw scores and the committed scoring report. The
correction is declared in the checker (`CORRECTED`) with its evidence, not left
to a commit message.

---

## 1. `docs/open-todos.md` — 5.1 closed

Marked **RESOLVED 16 Sep 2026**, answer **no**, evidence their Section 4 with
Figures 3 and 5. The 18-of-18 result stands as a finding, not a confirmation.
The entry records the three caveats, the fixed-weight variant, the unexamined-
convention observation, and the Section 7.3 replication. Original text kept in a
`<details>` block.

### A2 — the contrast, as written (thesis version)

> **This differs from what the prior work reports.** Wiesmayr *et al.* do not
> report improvement in the regime their deweighting penalises; where they
> examine it, they report the trade. In their narrow-range experiment,
> deweighted training holds a small advantage at high SNR while naive training
> keeps an even smaller advantage at the low end of the waterfall, and their
> replication on a simpler system repeats the pattern: training naively across
> the range is worst at high SNR and best at low SNR. Their headline wide-range
> result is a gain at a high-SNR operating point, and no per-bin low-SNR
> improvement is claimed. Our measurement shows improvement everywhere,
> including the two most heavily penalised bins.

### A3 — the three caveats, as written

> **Three differences must be kept in view before that contrast is read as a
> disagreement.** They are not measuring the same thing we are, not using the
> same mechanism, and not reporting the same size of effect.
>
> *The metric differs.* They measure bit and block error rate on a coded
> MIMO-OFDM system. We measure normalized channel-estimation error.
> "Improvement in the down-weighted bins" is not the same quantity in the two
> papers, and a trade in one need not imply a trade in the other.
>
> *The mechanism differs.* Their weights are recomputed each epoch from inverse
> accumulated losses, with a stability constant and a normalization at the
> centre of their grid. Ours are computed once before training and held fixed.
> Two schemes that share a motivation need not share a behaviour.
>
> *The size differs.* Their deweighting gain is $0.54$ dB at $1\%$ block error
> rate, which they describe as modest but essentially free, since it comes from
> the loss function rather than from a larger receiver or more data. The gain we
> measure is larger. Because the metrics differ, the two numbers are not
> directly comparable and we do not compare them.

### A4 — the fixed-weight acknowledgement, as written

> This is an adaptation of the remedy in [5], and the differences are worth
> stating exactly. Their weights are recomputed after every epoch from the
> inverse of the accumulated losses, with a stability constant and a
> normalization at the centre of their SNR grid. Ours are computed once, before
> training starts, and never change. They also note at the end of their Section 4
> that one could instead deweight using the loss of a fixed baseline, such as a
> classical system, rather than the adaptive scheme they implement — so a
> fixed-weight variant is raised there, though not implemented. Ours is fixed in
> that sense, with one further difference: the weights come from the network's
> own error measured before training, not from a separate classical baseline.

### A5a — the unexamined convention, as written

> They also observe, in their introduction, that systems are usually trained
> either at a single SNR or on samples drawn uniformly across the target range,
> apparently without the effect on performance at different SNRs being
> questioned. That is a published statement that the convention is unexamined,
> which is the gap this paper works in.

### A5b — the second-system replication, as written

> They also cover one architecture. Everything here is measured on a single
> unrolled network, so we cannot separate what is a property of training design
> in general from what is a property of this network in particular. There is a
> clear way to fix that, and it is not ours: Wiesmayr *et al.* repeat their whole
> experiment on a second, simpler system precisely to show that the effect is
> not an artifact of one architecture. Repeating our control on a second
> estimator is what the corresponding check would look like here, and we have
> not done it.

---

## 2. Part B — the three ported items

| item | thesis | submission |
|---|---|---|
| Table II balanced per-bin column | ✅ | ✅ **ported** |
| *Data and Pre-Registration Records* | ✅ | ✅ **ported** |
| log-loss prediction stated as **failed** | ✅ | ✅ **ported** |

Part A's content is in both as well. Verified mechanically — all seven Part A/B
items (metric, mechanism, size, fixed baseline, unexamined convention, second
system, the contrast itself) present in both files.

The six ported values were checked against the *other* file at `f76ed63` rather
than against a list I typed: the checker recomputes the baseline's value set for
the counterpart file and confirms each ported value was already there.

---

## 3. The new abstract, in full

Four blocks, no numerical list, no "vanishes" or "reverses".

> A common way to improve a trained channel estimator is to add a structural
> constraint inside the network, and the improvement that follows is usually
> credited to that constraint. This paper asks whether that credit is earned,
> for magnitude-only channel estimation at a Rydberg atomic receiver.
>
> The concern is about how such networks are trained. The training loss is an
> average of the error over samples covering a wide range of signal-to-noise
> ratios. Errors at low SNR are much larger numbers than errors at high SNR, so
> when they are averaged together the large numbers decide the average, and most
> of the training effort goes to the noisy samples. A structural constraint then
> helps most in the region the network was trained least, and the improvement it
> appears to give may be repairing a training problem rather than supplying
> information the network could not have learned.
>
> We propose a control for this. Measure how the training effort is divided
> across the SNR range, retrain both the structured and the unstructured network
> with that imbalance removed, and compare them again. We apply it to our own
> Hankel-structured estimator, using three training designs and three random
> seeds for each. The structural advantage is clearly positive under the usual
> training, close to zero when both networks are retrained on the region where
> the comparison is made, and negative when the loss is corrected instead.
> Compared seed by seed, the change from the usual training is large and clearly
> separated from zero; the individual endpoint values are not, and we do not
> claim them.
>
> Two further things follow. A consistent pattern within each SNR range survives
> where a single averaged number does not, and the sign of that averaged number
> depends on how the average is taken. Correcting the loss is an adaptation of a
> published remedy and is worth a substantial gain here, improving every one of
> the eighteen bin-by-seed cells — including the two the correction penalises
> most heavily, where the published work reports a trade rather than an
> improvement. A simpler loss taken in decibels needs no bins and gives no
> difference we can resolve. All results are simulation only.

Every number removed from it still appears in the body. The checker confirms no
value was lost — only multiplicities fell: $-1.46$ and $-1.27$ from 2 to 1,
$1.286$ and $2.171$ and $89.7$ from 3 to 2, and $5$ from 17 to 15.

---

## 4. Forward-reference sweep

Audited mechanically: every `\ref` compared against the line its `\label` is
defined on, plus a grep for the banned phrases.

| # | where | was | now |
|---|---|---|---|
| 1 | Introduction | "The control described in Section~\ref{sec:protocol} was run to describe that advantage better." | "We then ran a control to describe that advantage better." — the sentence now stands alone; the control is described two sections later on its own terms. |
| 2 | Where the Gradient Goes | "Section~\ref{sec:logloss} gives a case where a network that performs well still shows the same skew." | Sentence deleted and the surrounding claim made self-contained: "A small gradient can also mean a region that is already fitted well. The share is a reason to check, not a verdict, and we treat it that way throughout." |
| 3 | Three Training Designs | "We also did not measure $\Delta_H$ under the log-domain loss of Section~\ref{sec:logloss}…" | "One further design is absent by construction. We did not measure $\Delta_H$ under a loss that takes the per-sample error in decibels before averaging, because doing so would need a structured network trained with that loss and we did not train one." — names the thing instead of pointing at it. |
| 4 | A Simpler Loss | "…the first step of the control in Section~\ref{sec:protocol} should be read as a prompt to check…" | Now a *backward* reference in words: "This is the case promised earlier: the first step of our own control measures the share, and this experiment shows that the measurement is a prompt to check rather than a test that returns a verdict." |
| 5 | Where the Gradient Goes | "…and one of the experiments reported **later in this paper** turns out to be exactly that case." | Clause removed; the claim stands without it. |
| 6 | Three Training Designs | "A simpler loss … is examined **later in this paper**, but…" | Rewritten as #3 above. |

**Remaining `\ref` that resolve forward: four** — `fig:attr`, `fig:allcells`,
`tab:designs`, `tab:signs`. Each is referenced from inside its own subsection
and floats to the top of the page it is discussed on. That is float placement,
not a forward reference for the reader, and Paper 1 has the same four.

**C3's permitted exception: declined.** No paper-organization sentence was added
at the end of the introduction. Paper 1 does not have one, the brief says to
default to omitting it, and nothing in the rewrite needed it. Flagged here as
the brief asks.

---

## 5. Explanations added under C2

| requirement | where | what was added |
|---|---|---|
| what unrolling is | Introduction, opening | New first paragraph: iterative estimators repeat a step; unrolling lays out $T$ copies as $T$ layers with learned parameters; the shape of the computation is unchanged, the numbers inside it are now trained. Said **before** the word is used. |
| what the training loss is | The Training Problem, new eq. (5) | The per-sample normalized error written out, with why it is divided by $\lVert\bG\rVert_F^2$ (comparability across channel strengths), and what training does with it. |
| **why low-SNR samples dominate** | Introduction | New worked example. Two samples, errors $2$ and $0.02$, average $1.01$. Halving the clean one moves the average to $1.005$ — almost nothing. Improving the noisy one by the same fraction moves it to $0.51$. "Training follows whatever moves the average… The clean sample is barely trained, not because it does not matter, but because its numbers are small." |
| what gradient share means | The Training Problem | Explained before being defined: training moves along the gradient, so the size of the gradient from a group of samples says how much those samples steer the network. Then the formula, then "If every bin steered the network equally, each $q_b$ would be the same." The non-additivity caution is now explained rather than asserted — gradients pointing in different directions partly cancel, so $\sum_b\lVert g_b\rVert \neq \lVert\sum_b g_b\rVert$. And the equal-share reference is stated as our choice: "Nothing says an optimal solution must divide its effort equally." |
| what the structural step is | System Model | Already present and left as it was: the Hankel map, the four properties, and the not-a-projection argument for the same reason Paper 1 gives. |
| the two pooling rules | How the Average is Taken | Already at the right level per the brief; unchanged. |
| seed vs test-set variation | Experimental Setup | Already present and kept: a test-set interval varies over 2,000 trials with the networks fixed; a seed interval varies over three training runs; "Bootstrapping the test set does not create more training runs, so the two cannot substitute for each other." |

Also added under C5, not C2: the statement that this structural step is
**deliberately not the same operator** as the classical estimator's — fixed rank,
applied unconditionally, no selection rule, no ability to decline — "and nothing
measured here applies to it."

---

## 6. Numeric diff against `f76ed63`

```
paper/thesisproj2/haatim_thesisproj2.tex
  values: IDENTICAL set — no value altered, none lost
  new values, DECLARED (not our measurements):
    0.01, 1.005, 1.01, 0.51   [illustrative]  — the C2 worked example
    0.54                      [cited prior work] — Wiesmayr et al., their Section 4
  CORRECTED: -0.704 -> -0.703  (evidence above)

wip/spl2/haatim_structural_priors_evaluation.tex
  values: IDENTICAL set — no value altered, none lost
  PORTED from the other Paper 2 file (verified at baseline):
    -0.233, -0.703, -0.692, -0.241, -0.944, 0.567
  % src: comments: 23 at baseline, 24 now, all survive

RESULT: no number changed, no src comment lost
```

Not literally empty, and it should not be: Part B mandates ports, C2 mandates a
worked example, A3 mandates citing their $0.54$ dB, and one baseline value was
wrong. Every departure is categorised and justified by the checker rather than
waved through. **No measurement of ours changed.**

---

## 7. Build and repository state

```
git status --porcelain     clean
pages                      thesis 7, submission 7  (both 7 at baseline too)
pdflatex                   zero errors, zero undefined references,
                           zero undefined citations, both files
tests                      450 passed in 213.77s
Paper 1                    untouched this turn — git diff 739dba4 -- paper/thesisproj/ empty
```

Commits on `paper2-style`:

```
d25a648  refine the invariance checker to value-level; close open-todo 5.1
66453e2  Part A + C: close 5.1, rewrite Paper 2 in Paper 1's style
816db90  Part B: reconcile the two Paper 2 versions
```

---

## For the human, not this turn

Unchanged and still true:

- **The pilot-curve ambiguity (open-todos 5.2) was never searched.** Its novelty
  is unsupported rather than supported. Needs a literature check nobody in this
  container can do.
- **The convergence control remains unrun.** Still the single item most likely to
  change Paper 2's conclusion: if longer training on the conventional objective
  removes the contrast, the undertraining reading is supported; if it does not,
  the result is about objective dependence. The paper says plainly that the two
  readings are not separated.
