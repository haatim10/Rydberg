# PROMPT 22 Part A — the derivation, and its verification

Branch `paper1-review`. Decision committed standing alone in
`reports/p22/DECISION_A_bounds.md` (commit `9008526`) **before** this work.

**Outcome: A1 completed. The falsifier did not fire.** Both validations pass,
and the derivation as written reproduces the stored curves.

---

## 1. What was actually missing, and what was not

The brief asked for a derivation. The honest finding is that most of it already
existed in two places and had simply never been brought into the manuscript:

| Item | Already existed in | In the manuscript before this turn |
|---|---|---|
| parameter vector, Jacobian | `scripts/constrained_crlb_fast.py` | no |
| Rician likelihood, $\beta$ | `rydberg_sim/crlb.py` docstring | partly (§IV-A, by example only) |
| rank-one Fisher, block structure | `paper/paper1/haatim_hsgs_letter.tex` §IV | no |
| tangent-space form and why | letter §IV, with `% src:` | equation only, unexplained |
| NMSE conversion | `constrained_crlb_fast.py` | no |
| validation harness | `constrained_crlb.py` `validate()` | no |

**This is a regression, not a gap.** `haatim_hsgs_letter.tex` carries 21
`% src:` provenance comments; `haatim_thesisproj.tex` carried **zero**. When the
letter was rewritten in the author's plainer style, the derivation content and
every provenance comment were dropped. §IV was left holding an equation with no
construction behind it. That is how the review came to be right about a paper
whose implementation was correct all along.

`% src:` comments are restored for every number added this turn.

---

## 2. The derivation as now written (§IV-B)

Six subsections, each recomputable:

1. **Parameters.** $\boldsymbol\phi=\{\psi_{\ell,k},\mathrm{Re}\,\alpha_{\ell,k},\mathrm{Im}\,\alpha_{\ell,k}\}$, $m=3\sum_k L_k$. Stated explicitly: the angular unknown is the **spatial frequency** $\psi=\pi\sin\theta$, not the arrival angle $\theta$ — differentiating with respect to $\theta$ evaluates the tangent space at the wrong point (`scripts/audit_verify.py` PART 4b). Also stated: $L_k$ is treated as **known**, so the bound is for a receiver that knows more than our estimator does.
2. **Jacobian.** The three derivative expressions, with the real/imaginary row ordering that matches the Fisher blocks.
3. **Information.** The Rician density in full, $I(a)=4\beta$, $\beta=(\mathbb{E}[z^2R^2(\kappa)]-a^2)/\sigma^4$, $R=I_1/I_0$, and $J_n=\sum_p 4\beta\,\mathbf{u}\mathbf{u}^T$ — every term rank one.
4. **Why $U$ and not $D$.** Over-completeness, with the exact frequency (below).
5. **NMSE conversion.** Eq. (30): ratio of summed traces to summed $\lVert G\rVert_F^2$, **not** a mean of per-trial ratios, over the same 400 trial indices the estimator curves use.
6. **Verification**, reported with numbers.

---

## 3. A number that was wrong in the code, now corrected

`constrained_crlb_fast.py` said the path parametrisation is over-complete in
"~42% of trials at N=8". **That figure is wrong.**

With $L_k\sim\mathcal{U}\{3,\ldots,7\}$ i.i.d. and $K=3$, over-completeness
means $3\sum_k L_k > 2NK$. Computed exactly over all $5^3=125$ equally likely
draws:

| $N$ | ambient $2NK$ | $P(3\sum L_k > 2NK)$ |
|---|---|---|
| 8 | 48 | **0.2800** |
| 16 | 96 | 0.0000 |
| 32 | 192 | 0.0000 |

The letter's `% src:` comment already said 28.1% from 6400 stored $N=8$ trials,
which matches the exact 28.00% to Monte-Carlo precision. So the letter was
right and the code comment was wrong; the two had been sitting inconsistent.
The code comment is now corrected and says so.

The cleaner statement, now in the manuscript: over-completeness happens **only**
at $N=8$. At $N=16$ and $N=32$ it cannot happen at all, because $2NK$ is 96 and
192 against at most $3\times21=63$ parameters. Also confirmed:
$3\times\mathbb{E}[\sum_k L_k]=45.0$, matching the measured mean tangent rank of
44.6 at $N=32$ (slightly below 45 because occasional near-collinear paths lose a
direction to the $10^{-9}$ cut).

---

## 4. Verification, rerun 2026-09-15

`python3 scripts/constrained_crlb.py` — `validate()` output, verbatim:

### Validation 1 — the real rank-one bound must exceed the two-quadrature one

A magnitude measurement constrains one real direction, so its real Fisher term
is rank one. The convention $F=\sum\beta\,mm^H$ credits both quadratures, can
only add information, and must therefore give a strictly *lower* bound. The
test is the inequality, not equality. Paired on the same trials.

| $N$ | $P$ | SNR | rank-1 | 2-quad | excess dB | |
|---|---|---|---|---|---|---|
| 8 | 30 | 5 | $-10.122$ | $-10.915$ | $+0.793$ | ok |
| 8 | 10 | 15 | $-11.367$ | $-14.244$ | $+2.877$ | ok |
| 16 | 30 | 0 | $-6.215$ | $-6.943$ | $+0.728$ | ok |
| 32 | 30 | 10 | $-15.700$ | $-16.473$ | $+0.774$ | ok |

**Passes**, range $+0.73$ to $+2.88$ dB.

### Validation 2 — high-SNR gap over genie ZF must be $10\log_{10}2$

| | measured gap | required | |
|---|---|---|---|
| $N=8$, $P=30$ | $3.5454$ dB | $3.0103$ | ok |
| $N=16$, $P=30$ | $3.5244$ dB | $3.0103$ | ok |

**Passes** within the $\pm0.7$ dB tolerance.

**Reported honestly rather than as a clean pass:** both sit about half a decibel
*above* the limit. That is a finite-$P$ effect — the rank-one terms in $J_n$
have not averaged to $SS^H$ at $P=30$ — and the manuscript now states the
measured numbers and the reason, rather than quoting only the 3.0103 limit.

---

## 5. Reproduction of the plotted numbers

`results/track_b/constrained_crlb.json`, 400 trials, configuration A
($N=32$, $P=30$, RSR 12 dB):

| SNR (dB) | unconstrained | constrained | gap |
|---|---|---|---|
| $-5$ | $+0.233$ | $-6.873$ | 7.106 |
| $0$ | $-5.490$ | $-12.554$ | 7.063 |
| $+5$ | $-10.741$ | $-17.809$ | 7.068 |
| $+10$ | $-15.787$ | $-22.859$ | 7.071 |
| $+15$ | $-20.771$ | $-27.831$ | 7.060 |
| $+20$ | $-25.928$ | $-32.973$ | 7.045 |

Range **7.045–7.106 dB** = the manuscript's "7.05–7.11 dB". Reproduced.

Conditioning at these points: tangent rank 44.6 of ambient 192,
$\mathrm{cond}(U^TJU)\approx3.1$. Worst condition number over all 52 computed
points is $52.3$ (at $P=6$). The inverse is not delicate anywhere.

---

## 6. The measured weakness, stated with its number

The 400-trial path reads $\beta$ from an interpolation table rather than
integrating per measurement. The table's error is **measured at run time**,
probed at amplitudes drawn from each point's own $|\lambda|$ population, and
stored per point.

- Worst over **all 52 computed points**: $4.976\times10^{-2}$, at
  `N8_P30_snr+5_rsr+0`.
- Second worst: $1.913\times10^{-2}$, at `N32_P30_snr+5_rsr+0`.
- Worst over the points **this paper actually plots**: $7.4\times10^{-4}$.

Both bad points are RSR $=0$ dB cells of the `b6` reference-power sweep, which
the manuscript does not plot. The manuscript states all three numbers.

---

## 7. A repository hazard found while doing this

`scripts/constrained_crlb.py` (validation, 10 trials/point) and
`scripts/constrained_crlb_fast.py` (production, 400 trials/point) **write to the
same output path**, `results/track_b/constrained_crlb.json`.

Running the validation script to completion therefore silently overwrites the
400-trial file the paper's curves depend on with 10-trial data — and the 10-trial
jitter is $\pm0.31$ dB, which is large enough to move the published gap. During
this turn the validation run was cut off before its sweep wrote, and the
400-trial file was confirmed intact (`git status` clean under `results/`), but
that was luck.

Recorded in `docs/open-todos.md`. Not fixed this turn: changing an output path
is outside Part A's authorisation and would need the plotting scripts checked
for the same path.

---

## 8. Falsifier: did not fire

The decision fixed it in advance: *if `validate()` fails either check, or the
written derivation cannot reproduce the stored JSON within Monte-Carlo jitter,
A1 has failed and A2 is forced.*

Both checks passed. The stored gap reproduces the manuscript's quoted range
exactly. **A1 stands, the curves stay, and the "indicative references only and
not as verified bounds" sentence is deleted** — under A1 it would be false.
