# PROMPT 22 Part A — the bounds decision

**Committed standing alone, before any of the work it authorises.**

Date: 2026-09-15. Branch: `paper1-review`.

---

## Decision: A1. Complete the derivation. Keep the curves.

---

## Why, and a correction to the brief's premise

The brief costs A1 at "roughly a week" and A2 at "a day", and warns against
choosing A2 merely because it is cheaper. **Both cost estimates are wrong,
because they assume the derivation does not exist. It does.**

Standing rule: *a gate must not be built on a premise that a file on disk can
settle.* Three files settle this one.

`scripts/constrained_crlb_fast.py` is what produced the plotted curves. Its
module docstring and code already fix every item the brief demands:

| Brief's requirement | Where it already is |
|---|---|
| real parameter vector | `jacobian()` / the `Dc` assembly: $\theta=\{\psi_{\ell k},\ \mathrm{Re}\,\alpha_{\ell k},\ \mathrm{Im}\,\alpha_{\ell k}\}$, $m=3\sum_k L_k$ |
| Rician likelihood and noise convention | `rydberg_sim/crlb.py` docstring: $p(z\mid\lambda)=(2z/\sigma^2)e^{-(z^2+\lvert\lambda\rvert^2)/\sigma^2}I_0(2z\lvert\lambda\rvert/\sigma^2)$, $w\sim\mathcal{CN}(0,\sigma^2 I)$ |
| Fisher-information construction | $I(a)=4\beta$, $\beta=(\mathbb{E}[z^2R^2(\kappa)]-a^2)/\sigma^4$, $R=I_1/I_0$; block diagonal over receive elements; gradient $\partial a/\partial\mathrm{Re}\,G_{nk}=\mathrm{Re}(c_k)$, $\partial a/\partial\mathrm{Im}\,G_{nk}=-\mathrm{Im}(c_k)$, $c_k=e^{-\jmath\angle\lambda}S_{kp}$ |
| geometric tangent basis $U$, which path parameters are unknown | orthonormal basis of $\mathrm{range}(D)$ by SVD, cut at `TANGENT_RTOL = 1e-9`; all three parameters per path are unknown |
| invertibility conditions for $(U^\top J U)$ | the reason $U$ is used at all: $D^\top J D$ is genuinely singular when $3\sum_k L_k > 2NK$. Conditioning is *recorded per point* in `jacobian_cond` |
| conversion to NMSE, and averaging over realizations | ratio of summed traces to summed $\lVert G\rVert_F^2$ over the same 400 trial indices the estimator curves use, then $10\log_{10}$ |

There is also a validation harness already written (`constrained_crlb.py`
`validate()`): the real rank-1 bound must exceed the two-quadrature convention
paired on trials, and the high-SNR gap over genie ZF must be
$10\log_{10}2 = 3.0103$ dB.

**So A1 is not a week of new mathematics. It is transcription of a derivation
that exists in code, plus running the verification and reporting it.** The
honest estimate is hours.

That changes the decision completely. A2 was only ever attractive against a
week of work. Against hours, cutting §IV would discard:

- the 7.05–7.11 dB CCRB gap — the paper's only statement of what the structure
  is worth in principle;
- the 0.143 dB figure — the paper's only defence for a single baseline;
- the 4.39–4.93 dB headroom framing.

and would do so while a correct, validated implementation sat in the repository
unreported. That is the wrong trade at any price, and at this price it is not a
trade at all.

## Confirmation that the stored numbers are the paper's numbers

`results/track_b/constrained_crlb.json`, 400 trials, config-A points
($N=32$, $P=30$, RSR 12 dB):

| SNR (dB) | unconstrained | constrained | gap |
|---|---|---|---|
| $-5$ | $+0.233$ | $-6.873$ | 7.106 |
| $0$ | $-5.490$ | $-12.554$ | 7.063 |
| $+5$ | $-10.741$ | $-17.809$ | 7.068 |
| $+10$ | $-15.787$ | $-22.859$ | 7.071 |
| $+15$ | $-20.771$ | $-27.831$ | 7.060 |
| $+20$ | $-25.928$ | $-32.973$ | 7.045 |

Range 7.045–7.106, which is the manuscript's "7.05–7.11 dB". The number in the
paper is the number in the file.

Tangent-space rank 44.6 of an ambient $2NK=192$ at these points, and
$\mathrm{cond}(U^\top J U)\approx 3.1$ — well conditioned, so the inverse is not
delicate here.

## What this decision does not cover, recorded now rather than discovered later

**One measured weakness I will report rather than bury.** The $\beta$
interpolation table's measured worst-case relative error over all 52 computed
points is $4.98\times10^{-2}$. That worst case is at `N8_P30_snr+5_rsr+0`, and
the second worst ($1.91\times10^{-2}$) is `N32_P30_snr+5_rsr+0`. **Both are
RSR = 0 dB points from the `b6` sweep, which the manuscript does not plot.**
Across every point the manuscript does use, the measured error is
$\le 7.4\times10^{-4}$.

This is a real limitation of the fast path and it will be stated with its
number, not asserted away. It does not change the decision, because it does not
touch the plotted curves.

## What A1 commits me to

1. Write §IV so a reader can reproduce the curves: parameter vector,
   likelihood, Fisher construction, tangent basis, invertibility, NMSE
   conversion and averaging.
2. Re-run the verification and report the result, pass or fail. **If
   verification fails, this decision is void and A2 is forced.** Recording that
   now so the outcome cannot be reframed afterwards.
3. Delete the "indicative references only and not as verified bounds"
   sentence — under A1 it becomes false, which is the point.
4. Keep the unbiasedness caveat. It is about applicability, and it stays true.

## Falsifier

If `validate()` fails either check, or if the derivation as written cannot be
made to reproduce `results/track_b/constrained_crlb.json` to within Monte-Carlo
jitter, then A1 has failed and the curves come out under A2. No third option,
and no softening of §IV to cover a verification that did not pass.
