# PROMPT 22 Part B — P32 scored

Pre-registration: `reports/p22/PREREG_P32.md`, commit `ddb19c1`, standing alone.
Scorers: `scratch/p22_score_p32.py` (commit `5350012`) and
`scratch/p22_score_p32b.py`, each committed before it ran.

---

## Outcome: the falsifier fired. ρ is demoted and the comparative claim is deleted.

**ρ does not beat $L/r_{\max}$ under either registered statistic. It is
slightly worse under both.**

---

## 1. The prediction rule reproduces the manuscript

Interpolating the eight-point Experiment C table at $\rho=0.331$ gives $1.303$
dB and at $\rho=0.400$ gives $0.648$ dB. The manuscript quotes $1.30$ and
$0.648$. **The rule was confirmed, not guessed** — so the comparison below runs
the paper's own predictor against two alternatives, on the paper's own table.

## 2. Registered primary — paired median in the $[5,10)$ bin

| cell | measured | pred $L$ | pred $L/r_{\max}$ | pred ρ | $|e_L|$ | $|e_{L/r}|$ | $|e_\rho|$ |
|---|---|---|---|---|---|---|---|
| B1_N16_L2 | 4.613 | 7.043 | 3.556 | 3.000 | 2.431 | 1.056 | 1.612 |
| B1_N16_L4 | 1.407 | 3.556 | 1.038 | 0.758 | 2.150 | 0.369 | 0.648 |
| B1_N16_L7 | 0.135 | 1.415 | 0.046 | $-0.108$ | 1.280 | 0.088 | 0.242 |
| B1_N64_L8 | 3.884 | 1.038 | 3.556 | 4.088 | 2.846 | 0.327 | 0.205 |
| B1_N64_L14 | 1.553 | 0.046 | 1.415 | 1.536 | 1.507 | 0.138 | 0.017 |
| B1_N64_L29 | 0.080 | $-1.179$ | 0.005 | 0.044 | 1.259 | 0.074 | 0.036 |

| predictor | MAE | RMSE | max | sign correct |
|---|---|---|---|---|
| $L$ | 1.912 | 2.005 | 2.846 | 5/6 |
| **$L/r_{\max}$** | **0.342** | **0.482** | 1.056 | **6/6** |
| ρ | 0.460 | 0.721 | 1.612 | 5/6 |

Paired $|e_\rho|-|e_{L/r_{\max}}|$: mean $+0.118$, 95% CI $[-0.163,+0.399]$,
does not exclude zero.

## 3. An error in my own pre-registration, and what I did about it

PREREG_P32 asserted "840 trials per cell". **That is wrong.** It holds for the
three $N=16$ cells (853, 840, 874) and not for the three $N=64$ cells, which
carry 84, 85 and 83. In the registered primary bin that leaves $n=14,15,14$ —
below the 20-trial minimum this project's own `by_bin` applies before it will
report a bin at all.

I asserted a fact about repository contents without checking all six files. The
standing rule exists precisely for this and I broke it.

The remedy is not to choose a new statistic after seeing the answer. It is to
score the statistic that was **also registered**, under the identical rule and
the identical decision criterion.

## 4. Registered secondary — paired median over SNR $\ge 5$ dB

$n = 411, 415, 418, 41, 46, 37$ — every cell clears the threshold.

| predictor | MAE | RMSE | max | sign correct |
|---|---|---|---|---|
| $L$ | 1.867 | 1.990 | 3.276 | 5/6 |
| **$L/r_{\max}$** | **0.559** | **0.785** | 1.683 | **6/6** |
| ρ | 0.609 | 0.978 | 2.239 | 5/6 |

Paired $|e_\rho|-|e_{L/r_{\max}}|$: mean $+0.049$, 95% CI $[-0.342,+0.441]$,
does not exclude zero.

**The two statistics agree.** The conclusion is robust to my pre-registration
error, which is the only reason that error did not sink the result.

## 5. Scoring against the registered rule

| registered criterion | outcome |
|---|---|
| $L$ worst by $\ge1.0$ dB MAE | **HELD** — worse by 1.570 (primary), 1.308 (secondary) |
| ρ and $L/r_{\max}$ close, no reliable separation | **HELD** — both paired CIs include zero |
| ρ beats $L/r_{\max}$ (lower MAE **and** paired CI excluding zero) | **FAILED** — ρ has the *higher* MAE in both |
| **Falsifier:** ρ's MAE worse than $L/r_{\max}$'s | **FIRED**, under both statistics |

The falsifier's registered consequence is the third bullet of the decision
rule, and it applies regardless of how ρ scores against $L$:

> ρ is demoted to an exploratory descriptor in both abstract and conclusion,
> and the comparative claim is deleted outright.

**This is done.** Not softened into "ρ remains interpretable", not reframed
around consistency. The prediction I registered was that this would happen, and
it happened.

**What is still true and still worth saying:** both *normalised* predictors beat
raw $L$ decisively — 1.9 dB MAE against 0.34–0.61. Array-size normalisation is
what carries the effect. That is a fact about normalisation, not a claim that ρ
is special, and the manuscript now says the former.

## 6. Three further defects in the crossing locations

The manuscript reads: *"the HS-GS gain approaches zero at $\rho=0.588$, $0.518$
and $0.544$ … these values stay roughly the same."* Three separate problems sit
inside that sentence.

**(a) Two of the three are extrapolated, not bracketed.** The repository has
recorded this all along — `reports/trackD_partB9_analysis.json` carries
`zero_crossing_is_bracketed_not_extrapolated = {"16": false, "64": false,
"32": true}`. All three $N=16$ measurements are positive ($+4.130$, $+1.428$,
$+0.313$) and all three $N=64$ measurements are positive, so neither curve has a
measured pair straddling zero. The crossing is obtained by running a line past
the last measured point. The analysis code's own docstring says such a value
"must be labelled as such wherever it is quoted." The manuscript does not label
it. My independent bootstrap agrees: 0% of draws bracket the $N=16$ crossing and
5% the $N=64$ one.

**(b) The three numbers are not the same statistic.** $0.518$ comes from
Experiment C at a **fixed** 5 dB. $0.588$ and $0.544$ come from the B1 cells'
**pooled** value over a drawn SNR range — the field the code names
`pooled_SAMPLING_DESIGN_DEPENDENT`, and which the project's standing rule
demotes below the per-bin median. `trackD_partB9_analysis.py` warns in terms:
"A pooled B1 number and an Experiment C number are therefore NOT a paired
comparison." Setting the three side by side is the comparison the code says not
to make.

**(c) The collapse is weaker than "roughly the same" implies.** The script's own
like-for-like internal comparison — $N=16$ against $N=64$ on their own curves,
same design, same draw, same estimator — gives a **one-signed** gap with mean
$0.493$ dB and max $0.901$ dB. P12 predicted "curves within $\pm0.3$ dB at
matched $r_{\mathrm{eff}}/r_{\max}$"; the measured max gap is three times that.
The P12 falsifier was a systematic ordering above $0.5$ dB, and the measured
mean ordering is $0.493$ dB. **P12 held by seven thousandths of a decibel, with
a one-signed ordering.** That is not a collapse worth the word "invariant".

### Crossings with intervals, as B2 asked

From the $N=32$ table, bootstrapping each row's published CI:

| predictor | crossing | 95% CI | bracketed |
|---|---|---|---|
| $L$ | 14.566 | [13.679, 15.326] | yes, 99.8% of draws |
| $L/r_{\max}$ | 0.910 | [0.855, 0.958] | yes, 99.8% |
| ρ | 0.518 | [0.499, 0.533] | yes, 99.8% |

From the held-out cells, bootstrapping each cell's own interval:

| | crossing | 95% CI | bracketed |
|---|---|---|---|
| $N=16$, ρ | 0.565 | [0.557, 0.574] | **no**, 0% of draws |
| $N=64$, ρ | 0.526 | [0.505, 0.555] | **no**, 5% of draws |

These differ from the published $0.588$ and $0.544$ because they use the
per-bin-consistent SNR $\ge5$ median rather than the pooled statistic. The
intervals are narrow and the extrapolation is the dominant uncertainty, not the
sampling — which is why labelling the extrapolation matters more than quoting a
CI.

## 7. What this costs the paper, stated plainly

The effective-rank contribution is weakened, not destroyed.

- ρ **does** organise the behaviour: it beats raw $L$ by a wide margin and the
  ordering it produces is correct on 5 of 6 held-out cells.
- ρ has **no measured advantage** over the much simpler $L/r_{\max}$, and the
  simpler quantity needs no singular value decomposition and no knowledge of
  the noiseless channel.
- The "boundary is invariant across array size" reading rests on two
  extrapolated crossings, a statistic mismatch, and a one-signed 0.49 dB
  ordering.

The honest version of the contribution is: **normalising path count by the
array's structural capacity is what predicts when the constraint helps**, and ρ
is one way to do that normalisation which this evidence does not distinguish
from the simpler one.
