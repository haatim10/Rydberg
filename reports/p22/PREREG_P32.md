# PREREG P32 — does ρ predict better than L, or than L/r_max?

**Committed standing alone, before the comparison is scored.** Branch
`paper1-review`. Date 2026-09-15.

---

## The claim under test

The abstract of Paper 1 says effective rank

> "describes this behaviour more consistently than the number of propagation
> paths alone."

The paper shows ρ works. It has never shown ρ works **better** than anything.
That is an unsupported comparative, and P32 either supports it or removes it.

**Two different bars, and they are not the same bar.** The abstract's literal
claim is against $L$. The brief sets the bar at $L/r_{\max}$, which is a
strictly harder test, because $L/r_{\max}$ already does most of what ρ is
supposed to do — it normalises for array size. Both are scored. Registering the
distinction now so neither outcome can be reframed into the other afterwards.

## The three predictors

$$L,\qquad L/r_{\max},\qquad \rho = r_{\mathrm{eff}}/r_{\max}$$

with $r_{\max}(N)=\lceil N/2\rceil$.

## The prediction rule, identical for all three

Stated because the manuscript never stated it (brief item B2).

- **Interpolation:** piecewise-linear on a **fixed eight-point table**, with no
  regression and no spline. Outside the table's range, linear extrapolation
  from the two nearest points.
- **The table** is Track B Experiment C
  (`trackB_hankel_emgs/results/experiment_C_path_count.csv`): $N=32$,
  $r_{\max}=16$, $P=20$, fixed SNR $5$ dB, $L\in\{2,4,6,8,10,12,14,16\}$, 300
  trials per row. The same eight rows index all three predictors — only the
  $x$-coordinate changes. This is what makes the comparison fair.
- **How ρ is formed:** $r_{\mathrm{eff}}$ is the Roy–Vetterli effective rank of
  the Hankel lifting of **one channel column**; ρ is the **median over all
  columns and all draws** at a configuration, divided by $r_{\max}$. So ρ is a
  per-configuration scalar, not a per-trial quantity. Stating this plainly
  because the manuscript's wording could be read as per-trial, and it is not.
- **Verification that this is the rule the paper already used:** interpolating
  the table at $\rho=0.331$ gives $1.303$ dB and at $\rho=0.400$ gives $0.648$
  dB. The manuscript quotes $1.30$ and $0.648$. The rule is confirmed, not
  guessed.

## Held-out set

The six `B1_*` cells of `results/track_d/partB9/`, none of which is in the
table:

| cell | $N$ | $L$ | $r_{\max}$ | $L/r_{\max}$ | $r_{\mathrm{eff}}$ | ρ |
|---|---|---|---|---|---|---|
| B1_N16_L2 | 16 | 2 | 8 | 0.250 | 1.88 | 0.235 |
| B1_N16_L4 | 16 | 4 | 8 | 0.500 | 3.10 | 0.388 |
| B1_N16_L7 | 16 | 7 | 8 | 0.875 | 4.35 | 0.544 |
| B1_N64_L8 | 64 | 8 | 32 | 0.250 | 6.33 | 0.198 |
| B1_N64_L14 | 64 | 14 | 32 | 0.438 | 9.89 | 0.309 |
| B1_N64_L29 | 64 | 29 | 32 | 0.906 | 16.24 | 0.508 |

840 trials per cell, SNR drawn $\mathcal{U}[-10,20]$ and binned post hoc.

**These cells are at $N=16$ and $N=64$; the table is at $N=32$.** That is the
point of the test. $L$ cannot transfer across $N$ even in principle: $L=7$ sits
near the ceiling at $N=16$ ($r_{\max}=8$) and far below it at $N=64$
($r_{\max}=32$), so the same $L$ describes opposite regimes.

## Held-out statistic

Paired per-trial median in the $[5,10)$ SNR bin — primary by standing rule, and
the bin closest to the table's fixed 5 dB. The `high_snr_ge5` aggregate is
reported as secondary.

**A confound, declared in advance.** The table is at fixed SNR 5 dB; the
held-out cells draw SNR and bin post hoc. `scratch/trackD_partB9_analysis.py`
already warns these are not a paired comparison. This inflates every absolute
error. It does **not** bias the ranking, because all three predictors are
scored against the same held-out numbers and any common offset shifts them
equally. P32 is a ranking question, so the confound is survivable — but it caps
what the absolute errors mean, and they will be reported as ranking evidence
only.

## Metrics

1. **Held-out error:** MAE and RMSE in dB over the six cells.
2. **Classification:** does the predictor get the sign of the measured gain
   right — i.e. whether the structural step helps? With six cells this is a
   weak instrument and will be reported as such, not as an accuracy figure that
   sounds firmer than six points can support.

## Predicted outcome, recorded before scoring

**[HYP]** $L$ is worst by a wide margin — MAE at least $1.0$ dB worse than
either normalised predictor — because it cannot transfer across $N$.

**[HYP]** ρ and $L/r_{\max}$ come out close, **with no reliable separation on
six cells.** Reasoning: both the table and the held-out cells draw equal-power
paths at fixed $L$, so $r_{\mathrm{eff}}/L$ follows a similar saturation curve
in both, and the two predictors are close to reparameterisations of each other
on this data. At the cap-saturating end the measured ratios are $0.62$
($N{=}16$, $L{=}7$), $0.55$ ($N{=}32$, $L{=}16$) and $0.56$ ($N{=}64$,
$L{=}29$) — similar enough that I do not expect ρ to separate.

I expect the abstract's literal claim (against $L$) to survive and the harder
claim (against $L/r_{\max}$) to fail to establish.

## Decision rule, fixed now

**ρ counts as showing a predictive advantage over $L/r_{\max}$** if and only if
both hold:

1. $\mathrm{MAE}(\rho) < \mathrm{MAE}(L/r_{\max})$; **and**
2. the six paired per-cell differences of absolute error,
   $|e_\rho| - |e_{L/r_{\max}}|$, have a two-sided 95% Student-$t$ interval on
   five degrees of freedom that excludes zero.

Criterion 2 is what stops a win by a tenth of a decibel on six points being
reported as a result.

**Consequences, fixed now:**

- **If ρ beats $L/r_{\max}$ by the rule above:** the comparative claim stays and
  is strengthened to name $L/r_{\max}$ as the thing it beats.
- **If ρ beats $L$ but not $L/r_{\max}$:** the abstract's literal claim is
  true and is kept, **but** it must be stated against $L$ explicitly, and the
  paper must say that ρ shows no measured advantage over the simpler
  $L/r_{\max}$ on this evidence. No reframing around "consistency" or
  "interpretability" to avoid reporting that.
- **If ρ does not beat $L$ either:** ρ is demoted to an exploratory descriptor
  in both abstract and conclusion, and the comparative claim is deleted
  outright.

## Falsifier

If ρ's held-out MAE is **worse** than $L/r_{\max}$'s, the comparative claim as
currently worded is not merely unsupported — it is pointing the wrong way, and
the demotion in the third bullet applies regardless of how ρ scores against
$L$.

## Also to be produced (brief item B2)

Uncertainty on the three near-zero-gain locations $0.588$, $0.518$, $0.544$,
which the manuscript currently quotes bare. These come from linear
interpolation in the bracketing pair of measured points, and the existing code
already flags whether a crossing is bracketed or extrapolated. Interval by
bootstrap over trials. **No threshold is registered for this** — it is a
missing-uncertainty repair, not a hypothesis test, and inventing a pass mark
for it would be theatre.
