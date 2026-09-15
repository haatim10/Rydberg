# PROMPT 22 — report

Branch `paper1-review`, off `claude/new-paper-implementation-fs4ako`.
Paper 2 untouched, as required.

---

## 1. The four "already done" items — all confirmed, none regressed

| Review demand | Where it is | Status |
|---|---|---|
| "it again calls the operation a projection" | §III-C, two paragraphs plus the 5-element non-idempotence example with singular values $[9.6194,0.1295,0.0616]$ | present |
| "does not specify an all-pilot refit" | §III-D Stage two: "computed again using \emph{all} $P$ pilot columns" | present |
| "leaves power normalization unspecified" | §II, $\alpha_{\ell,k}\sim\mathcal{CN}(0,\beta_k/L_k)$ with the reason | present |
| "the second distribution remains unnamed" | §V-F: $L=10$, $\alpha\sim\mathcal{CN}(0,1)$, uniform angles, cites [14] | present |

No time spent beyond the check.

**One correction to the brief, and one to myself.** The brief's §III-B / §V-G
letters are now §III-C / §V-H, because this turn added an EM-GS subsection and
split the placement result out of "Additional Results". Separately: I claimed
mid-turn that the brief's "Fig. 2(b)" was wrong and the marker was in Fig. 1(b).
**The brief was right.** I had counted figure files and forgotten the system
diagram is Fig. 1. The manuscript uses `\ref`, so nothing was broken by the
mistake.

---

## 2. Part A — the bounds decision

### Decision: **A1**. Committed standing alone in `DECISION_A_bounds.md` (`9008526`) before any work.

**The brief's cost premise was wrong, and a file on disk settles it.** A1 was
costed at "roughly a week" on the assumption that the derivation did not exist.
It did. `scripts/constrained_crlb_fast.py` — the script that produced the
plotted curves — already contained the parameter vector, the Jacobian, the
Fisher construction, the tangent-space form and the NMSE conversion, and
`constrained_crlb.py` already contained a validation harness. A1 was
transcription plus verification: hours, not a week.

Against hours, A2 would have discarded the 7.05–7.11 dB gap, the 0.143 dB
single-baseline defence and the 4.39–4.93 dB headroom framing while a correct,
validated implementation sat unreported. That is not a trade.

### The derivation (§IV-B, six subsections, each recomputable)

Parameters $\{\psi_{\ell,k},\mathrm{Re}\,\alpha_{\ell,k},\mathrm{Im}\,\alpha_{\ell,k}\}$
with $m=3\sum_k L_k$, stating that the angular unknown is the **spatial
frequency** and not the arrival angle, and that $L_k$ is treated as **known**;
the three Jacobian derivatives with their row ordering; the Rician density in
full, $I(a)=4\beta$, and the rank-one outer-product form of each Fisher term;
why an orthonormal basis of $\mathrm{range}(D)$ is used instead of $D^\top JD$;
the NMSE conversion as a **ratio of summed traces, explicitly not a mean of
per-trial ratios**, over the same 400 trial indices the estimator curves use;
and the verification.

### Verification — the falsifier did not fire

- Rank-one bound exceeds the two-quadrature convention by **+0.73 to +2.88 dB**
  over four cells, paired on trials. Passes.
- High-SNR gap over genie ZF: **3.55 dB** ($N=8$) and **3.52 dB** ($N=16$)
  against the $10\log_{10}2=3.0103$ limit. Passes within tolerance — and
  reported as measured, about half a decibel high, with the finite-$P$ reason,
  rather than as a clean pass.
- Stored gap reproduces the manuscript's 7.05–7.11 dB exactly (range
  7.045–7.106).

### Reported against myself

The 400-trial path reads $\beta$ from an interpolation table. Measured
worst-case relative error over all 52 computed points is $4.98\times10^{-2}$,
at two RSR $=0$ dB points **the paper does not plot**. Over every plotted point
it is $\le7.4\times10^{-4}$. All three numbers are in the manuscript.

### A wrong number in the code, corrected

`constrained_crlb_fast.py` claimed over-completeness in "~42% of trials at
$N=8$". Exact value is **28.00%**, matching the 28.1% the letter's own `% src:`
comment already recorded from 6400 stored trials. Over-completeness occurs
**only** at $N=8$.

---

## 3. Part B — P32

Pre-registered standing alone (`ddb19c1`); scorers committed before running
(`5350012`, and the secondary scorer likewise).

### Outcome: **the falsifier fired. ρ is demoted and the comparative claim is deleted.**

| predictor | MAE primary | MAE secondary | sign correct |
|---|---|---|---|
| $L$ | 1.912 | 1.867 | 5/6 |
| **$L/r_{\max}$** | **0.342** | **0.559** | **6/6** |
| ρ | 0.460 | 0.609 | 5/6 |

Paired $|e_\rho|-|e_{L/r_{\max}}|$: $+0.118$ dB $[-0.163,+0.399]$ primary,
$+0.049$ $[-0.342,+0.441]$ secondary. Six cells do not separate them, and
neither favours ρ.

The prediction rule was **confirmed, not assumed** — interpolating the table at
$\rho=0.331$ and $0.400$ reproduces the manuscript's published 1.30 and 0.648
dB exactly, so this ran the paper's own predictor against alternatives on the
paper's own table.

### An error in my own pre-registration

PREREG_P32 asserted "840 trials per cell". True for the three $N=16$ cells,
**wrong for the three $N=64$ cells**, which carry 84, 85 and 83 — leaving
$n=14,15,14$ in the registered primary bin, below this project's own 20-trial
minimum. I asserted a fact about repository contents without checking all six
files; that is the standing rule and I broke it. The remedy was to score the
**also-registered** secondary under the identical rule rather than pick a new
statistic after seeing the answer. The two agree, which is the only reason the
error did not sink the result.

### Three further defects in the crossing locations

1. **Two of the three are extrapolated, not bracketed.** The repository already
   recorded this (`zero_crossing_is_bracketed_not_extrapolated = {"16": false,
   "64": false, "32": true}`); the manuscript quoted all three alike.
2. **They are not the same statistic.** $0.518$ is a fixed-5-dB number;
   $0.588$ and $0.544$ come from the field the analysis code itself names
   `pooled_SAMPLING_DESIGN_DEPENDENT`.
3. **The collapse is weaker than "roughly the same".** The like-for-like
   $N=16$ vs $N=64$ comparison shows a **one-signed** gap, mean $0.493$ dB, max
   $0.901$ dB, against a P12 prediction of $\pm0.3$ dB. P12 held by seven
   thousandths of a decibel.

All three are now stated in the manuscript.

### What survives

Normalising path count by the array's structural capacity is what predicts when
the constraint helps: raw $L$ gives 1.91 dB mean error, both normalised
descriptors give 0.34–0.61. ρ is presented as one way to normalise, kept for
convenience on unequal-strength paths, and explicitly not claimed to be better.

---

## 4. Part C — the ten defects

| # | before | after |
|---|---|---|
| 1 | Table I had no sweep count | $\ncz=4$ column added; caption states the order rule, $T$, and the search budget |
| 2 | EM-GS used throughout, never defined | New §III-B: expectation step with Bessel weight, LS step with ridge, spectral initializer, fixed budget, cited to [1, Alg. 2] with [3] as antecedent |
| 3 | ambiguous which variant is HS-GS | §III-C states the step is applied after **every** one of the $T$ updates, and that neither ordering is a special case of the other |
| 4 | dangling "also" in §IV-A | antecedent restored: $J$ block diagonal across elements, so added elements bring measurements and unknowns in fixed proportion |
| 5 | unexplained "predicted failure" marker | explained where the aperture result is discussed, with its registered band $[-0.35,+0.05]$ — **and** that the prediction was only half right |
| 6 | [5] cited before [4] | bibliography reordered by first appearance |
| 7 | no pre-registration disclosure | all four registered predictions listed, **two of four failed**, each discussed where its experiment appears |
| 8 | abstract implied all Rydberg architectures | opens with the magnitude-only array, excludes superheterodyne |
| 9 | guaranteed-sounding fallback | precise in all three places: selecting full rank disables the step; not a guarantee about the selector |
| 10 | "three important paths" ⇒ rank 3 | exact vs approximate low rank separated: the bound counts paths, not strong paths |

**Item 2 caught two errors in my own drafting.** I first wrote that the
iteration starts from zero and that the expectation step keeps the measured
magnitude. Both are false against `rydberg_sim/gs.py`: it uses a per-row
spectral initializer, and the Bessel ratio $R(\kappa)$ shrinks the imputed
field. Corrected before they went in.

**No forward referencing.** Audited mechanically across the whole file. Two
genuine violations found and removed (§II → the path-count experiment; §III-B →
Table I). Four remain, all floats referenced from within their own subsection,
which is placement rather than a forward reference for the reader. A stale
hardcoded "Section III-B" in the conclusion — created by adding a subsection —
was caught by **rendering the pages**, not by the build: a hardcoded letter is
valid LaTeX and silently wrong.

---

## 5. Part D

### D1 — the placement comparison

The re-run reproduces the committed B5 cell to the digit, so the new quantity
is measured on the same measurement.

| SNR | inter $-$ post | 95% CI | median $q$ |
|---|---|---|---|
| $-5$ | $-0.081$ | $[-0.159,+0.047]$ | **0.546** |
| $0$ | $-0.164$ | $[-0.317,-0.066]$ | 0.266 |
| $5$ | $+0.034$ | $[-0.050,+0.133]$ | 0.134 |
| $10$ | $+0.010$ | $[-0.071,+0.077]$ | 0.071 |
| $15$ | $+0.054$ | $[-0.014,+0.107]$ | 0.040 |
| $20$ | $+0.168$ | $[+0.102,+0.246]$ | 0.022 |

**At $-5$ dB the two estimates differ by 55% of the channel norm while their
errors differ by 0.08 dB.** The review's objection is confirmed as measured
fact: a difference of means cannot establish that two estimators agree.

Paired intervals the headline number lacked: six-point mean $+0.0035$ dB,
$[-0.117,+0.124]$; pooled per-trial median $+0.018$ dB, $[-0.008,+0.049]$.
Both contain zero — "0.003 dB" was never a precise zero.

The mean also hides a sign change: significantly worse at 0 dB, significantly
better at 20 dB, similar magnitudes.

§V-H now concludes narrowly: the two placements are equally accurate on average
while being **visibly different estimators**; post-hoc costs one Cadzow step
instead of $T$, so **it is the better practical choice here, and the paper says
so even though it is not the variant it proposes.**

### D2 — intervals on the negatives

Path-count sweep reported across its whole range. At $L=14$, $+0.046$ dB with
$[-0.050,+0.136]$ **containing** zero — the constraint has stopped helping. At
$L=16$, $-0.117$ dB with $[-0.206,-0.038]$ **excluding** zero — it is doing
measurable harm. Only the second says the constraint should be switched off.

Forced-active at $\rho=0.735$: $-0.075$ dB, $[-0.186,-0.026]$, excluding zero,
at a $38.8\%$ abstention rate.

### D3 — setup gaps

Normalized error and per-trial dB gain as equations; the three summaries and
why they are not interchangeable; how the twelve operating points aggregate;
$T$, order search, $\ncz$, TB and TD, and the six SNR values.

**The 276/266 counts are stated as measured with the cause explicitly not
asserted.** I did not find it. It is in `docs/open-todos.md` as
must-resolve-before-submission rather than written into the paper as a guess.

---

## 6. Part E — `docs/open-todos.md` §7

Deferred as instructed: the full three-method campaign, absolute runtime
reporting, $N=8$ trial-level forensics, the tutorial-material reduction.

**Five new items**, two of them hazards:

- **7.3.1** `constrained_crlb.py` and `constrained_crlb_fast.py` write to the
  **same results path**. Running validation to completion silently overwrites
  the 400-trial file the published curves depend on with 10-trial data, whose
  ±0.31 dB jitter would move the published gap. This turn's run was cut off
  before its sweep wrote and the file was confirmed intact — that was luck.
- **7.3.2** The thesis-style rewrite **dropped content the letter still has**:
  all 21 `% src:` comments, the Fisher construction, the tangent-space
  justification, the per-SNR placement breakdown, and several limitations
  including the 0.493 dB residual and the unbracketed crossings. Parts of the
  external review were right about the thesis version while the letter was
  already correct. **Diff the two before submission.**
- 7.3.3 a wrong asserted percentage in code (fixed); 7.3.4 the crossings mix
  statistics; 7.3.5 the 276/266 counts.

---

## 7. Build and repository state

```
git status --porcelain   (clean at time of report)
pages                    13   (was 10 pre-revision)
pdflatex                 zero errors, zero undefined references,
                         zero undefined citations, zero overfull boxes
tests                    450 passed in 64.77s
manuscript diff          684 insertions, 139 deletions
```

Commits on `paper1-review`, in order:

```
9008526  Part A: the bounds decision, standing alone
9802fe6  Part A: write the derivation, verify it, keep the curves
ddb19c1  Part B: pre-register P32, standing alone
5350012  Part B: commit the P32 scorer before it runs
ee506f0  Part B: score P32 — the falsifier fired, rho is demoted
ee0e406  Part D1: commit the placement measurement before it runs
bc574af  Part C: nine manuscript defects + no-forward-referencing
cc864fc  Part D2 + E: negative-result intervals, open-todos
e77d5b6  fix a stale hardcoded section letter
ce76200  Part D1: two placements are different estimators, equal error
```

### Numeric changes to published values

**None.** No headline number was recomputed. Everything added is either a new
quantity ($q_i$, the three-predictor errors), an interval on a number that was
already published bare, or a correction to prose. The one numeric correction is
in a code comment (42% → 28.00%), not in the paper.

### Branch note

PROMPT 22 says "Branch as `paper1-review`" while the session's standing
instruction designates `claude/new-paper-implementation-fs4ako`. Work is on
`paper1-review`. **Both will be pushed** so neither instruction is violated and
nothing is stranded.
