# Rydberg Atomic MIMO — formulae, methods and results

**Track A** — reproduction of Cui et al., *"Towards Atomic MIMO Receivers"* (IEEE JSAC 43(3), March 2025).
**Track B** — a proposed structure-aware channel estimator, HS-GS, tested on the exact nonlinear model.

| | |
|---|---|
| Track A branch | `track-a-cui-reproduction` @ `ce118f0` (frozen) |
| Track B branch | `track-b-ula-channel-estimation` @ `786df67` |
| Config fingerprint | `46dbd9f1cf57d1cc` |
| Track B trials | 24 100 (B3 21 700 + B4 2 400) |
| Audit checks | 141 (deep) + 21 (Step-0), 0 failures |
| Linearizations used | **0** |

---

## 0. What was done

Two tracks, one physical model.

**Track A — reproduction, frozen.** Figures 5, 6, 7(a), 7(b) and 8 were reproduced from the paper's
specification, not from curve extraction. Figures 7(a), 7(b) and 8 match Cui closely: SNR/RSR shifts of
0.05–1.4 dB and BER ratios of 0.78–0.97. Figures 5 and 6 match qualitatively with a documented ≈2 dB
offset traced to channel conditioning, not to an algorithmic error. Two prose claims were also reproduced
without curve extraction: the EM-GS-to-ZF gap (measured 3.79–3.97 dB; Cui states "3 ~ 4 dB") and the Fig. 8
improvement from RSR 0 to 20 dB (measured 39.1×; Cui states "more than one order of magnitude").

**Track B — a proposed estimator, tested.** HS-GS (Hankel-Structured Gerchberg–Saxton) was designed and
evaluated against Cui's two baselines on the exact nonlinear model. The hypothesis under test: geometric
structure becomes more useful as array dimension $N$ grows relative to channel path complexity. The final
experiment was built to *test* that hypothesis, not to find a configuration where the new method looks good.

**Result: the hypothesis is supported.** The advantage over EM-GS is −0.19 dB at $N=8$, +0.78 dB at
$N=16$ and +2.85 dB at $N=32$, and the sign flips precisely where the algebra says the structural
constraint stops being vacuous.

---

## 1. The observation model

A Rydberg atomic receiver measures the **magnitude** of the total electric field at each sensor. The phase
is destroyed by the measurement. That single fact makes this a phase-retrieval problem rather than a linear
inverse problem, and it governs every algorithm below.

### 1.1 Detection (Track A)

$$\mathbf{z} = \left|\, \mathbf{A}^{H}\mathbf{s} + \mathbf{b} + \mathbf{w} \,\right| \tag{1}$$

- $\mathbf{A}\in\mathbb{C}^{K\times N}$ — known channel (Cui eq. 16)
- $\mathbf{s}\in\mathbb{C}^{K}$ — unknown QAM symbols, $\mathbb{E}|s_k|^2=1$
- $\mathbf{b}\in\mathbb{C}^{N}$ — known local-oscillator reference
- $\mathbf{w}\sim\mathcal{CN}(0,\sigma^2\mathbf{I})$ — noise, **inside** the magnitude
- $|\cdot|$ is the elementwise amplitude — $|\mathcal{E}|$, never $|\mathcal{E}|^2$

### 1.2 Channel estimation (Track B)

$$\mathbf{Z} = \left|\, \mathbf{G}\mathbf{S} + \mathbf{B} + \mathbf{W} \,\right| \tag{2}$$

with $\mathbf{G}\in\mathbb{C}^{N\times K}$ unknown, $\mathbf{S}\in\mathbb{C}^{K\times P}$ known pilots,
$\mathrm{vec}(\mathbf{W})\sim\mathcal{CN}(0,\sigma^2\mathbf{I})$.

> **Hard constraint held throughout.** This is the *exact* nonlinear model. No strong-reference
> linearization is used in any Track-B estimator reported here. A runtime tripwire replaces the linearized
> solver with a function that raises, then runs the full estimator: it was called **0 times**.

### 1.3 Canonical form

Both reduce to one canonical problem, which is why a single validated solver serves both. Per receive
element $n$:

$$\mathbf{z} = \left|\, \mathbf{M}^{H}\mathbf{u} + \mathbf{b} + \mathbf{w} \,\right| \tag{3}$$

| | Detection | Channel estimation |
|---|---|---|
| $\mathbf{M}$ | $\mathbf{A}$ | $\overline{\mathbf{S}}$ (conjugate pilots) |
| $\mathbf{u}$ | $\mathbf{s}$ | $\overline{\mathbf{g}_n}$ (conjugate row) |
| $\mathbf{z}$ | $\mathbf{z}$ | $\mathbf{Z}_{n,:}$ |

The conjugation was verified by noiseless recovery: the estimator returns $\mathbf{G}$ to a relative error
of $2.4\times10^{-15}$, and not $\overline{\mathbf{G}}$.

---

## 2. Channel models

### 2.1 Track A — 3GPP TR 38.901 clustered channel

Cui generates coefficients with the 3GPP TR 38.901 model but publishes only Table I:

$$a_{n,k} = \sum_{c=1}^{N_c}\sum_{r=1}^{M_r} \alpha_{c,r}\, e^{-\jmath 2\pi f_c \tau_c}\,
\left(\boldsymbol{\mu}_{eg}^{T}\boldsymbol{\epsilon}_{n,c,r}\right)\,
e^{-\jmath (n-1)\pi\sin\theta_{c,r}} \tag{4}$$

| Table I parameter | Value |
|---|---|
| clusters $N_c$ / rays per cluster $M_r$ | 23 / 20 |
| path gains $\alpha_{c,r}$ | $\mathcal{CN}(0,1)$ |
| incident angles $\theta$ | $\mathcal{U}(-90°, 90°)$ |
| max angle spread per cluster | $\mathcal{U}(-5°, 5°)$ |
| max delay spread | $\mathcal{U}(0, 30\ \mathrm{ns})$ |
| carrier $f_c$ | 5 GHz |

Rows are normalized so $\mathrm{mean}_n|a_{n,k}|^2=1$, which is what makes the SNR/RSR definitions exact.
Cui §VI-A samples the polarization vector $\boldsymbol{\epsilon}$ **per antenna element**; that choice
whitens the array response and is the origin of the documented ≈2 dB offset in Figs. 5–6.

### 2.2 Track B — geometric ULA channel

$$\psi_{\ell,k} = \pi\sin\theta_{\ell,k}, \qquad
g_{n,k} = \sum_{\ell=1}^{L_k}\alpha_{\ell,k}\, e^{-\jmath (n-1)\psi_{\ell,k}} \tag{5}$$

| | |
|---|---|
| angles | $\theta_{\ell,k}\sim\mathcal{U}[-\pi/2,\pi/2]$ |
| gains | $\alpha_{\ell,k}\sim\mathcal{CN}(0,\beta_k/L_k)$, so $\mathbb{E}\lvert g_{n,k}\rvert^2=\beta_k$ |
| paths | $L_k\sim\mathcal{U}\{3,\dots,7\}$, iid per user per realization |
| manifold | $\mathbf{a}(\theta)=[1,e^{-\jmath\psi},\dots,e^{-\jmath(N-1)\psi}]^{T}$, $\lVert\mathbf{a}\rVert^2=N$ |

---

## 3. SNR and RSR calibration

These two definitions set every operating point in every figure. Both have a factor-$K$ trap.

### 3.1 SNR — Cui eq. (36)

$$\mathrm{SNR} = \frac{\mathbb{E}\left(|\mathbf{a}_n^{H}\mathbf{s}|^2\right)}{\mathbb{E}\left(|w_n|^2\right)} \tag{6}$$

With row-normalized $\mathbf{A}$, unit-energy QAM and independent users the numerator is
$\sum_k \mathbb{E}|a_{n,k}|^2\mathbb{E}|s_k|^2 = K$. Hence:

$$\sigma^2 = \frac{K}{\mathrm{SNR}_{\mathrm{lin}}}, \qquad \mathrm{SNR}_{\mathrm{lin}} = 10^{\mathrm{SNR_{dB}}/10} \tag{7}$$

This is *total* signal power over all $K$ users, so fixing SNR does not fix per-user SNR: doubling $K$ at
fixed SNR doubles $\sigma^2$.

### 3.2 RSR — Cui eq. (37)

$$\mathrm{RSR} = \frac{\mathbb{E}\left(|b_n|^2\right)}{\mathbb{E}\left(|a_{n,k}s_k|^2\right)} \tag{8}$$

The denominator is a **single user's** contribution, not the sum over $K$ — the easiest place to introduce
a factor-$K$ error. With $\beta_{\mathrm{ref}}=1$:

$$|\alpha_b| = \sqrt{\mathrm{RSR}_{\mathrm{lin}}}\quad\text{(not }\sqrt{K\,\mathrm{RSR}_{\mathrm{lin}}}\text{, not }\sqrt{\mathrm{RSR}_{\mathrm{lin}}/K}\text{)} \tag{9}$$

The audit tests that the implemented value differs from both incorrect alternatives, and measures achieved
SNR and RSR empirically: **2.82 dB and 12.15 dB** against targets of 3 and 12 dB.

---

## 4. Algorithms

### 4.1 Spectral initialization

With $\bar{\mathbf{m}}_q=[\mathbf{m}_q;\,b_q]$:

$$\mathbf{M}_{\mathrm{spec}} = \sum_q z_q\, \bar{\mathbf{m}}_q \bar{\mathbf{m}}_q^{H}
\in\mathbb{C}^{(D+1)\times(D+1)},\qquad \bar{\mathbf{u}}_0 = \text{principal eigenvector} \tag{10}$$

### 4.2 Biased Gerchberg–Saxton (Cui Algorithm 1)

$$\boldsymbol{\lambda}^{t-1} = \mathbf{M}^{H}\mathbf{u}^{t-1} + \mathbf{b}, \qquad
\boldsymbol{\theta}^{t} = \angle\boldsymbol{\lambda}^{t-1} \tag{11}$$

$$\mathbf{y}^{t} = \mathbf{z}\odot e^{\jmath\boldsymbol{\theta}^{t}},\qquad
\mathbf{r}^{t} = \mathbf{y}^{t}-\mathbf{b} \tag{12}$$

$$\left(\mathbf{M}\mathbf{M}^{H}\right)\mathbf{u}^{t} = \mathbf{M}\mathbf{r}^{t} \tag{13}$$

The measured amplitude $\mathbf{z}$ is kept exactly; only the phase comes from the current iterate. Solved
via the normal equations, never an explicit inverse. $t_0=50$ iterations throughout both tracks.

### 4.3 EM-GS (Cui Algorithm 2)

Same phase and least-squares steps, but the restored observation is weighted by the Bessel ratio — the
conditional mean of the Rician amplitude, i.e. a soft, SNR-aware version of the hard magnitude substitution:

$$R(\kappa) = \frac{I_1(\kappa)}{I_0(\kappa)}, \qquad
\boldsymbol{\kappa} = \frac{2}{\sigma^2}\,\mathbf{z}\odot|\boldsymbol{\lambda}| \tag{14}$$

$R$ is monotone increasing with $R(0)=0$ and $R(\kappa)\to1$. That limit explains an observed behaviour:
at high SNR or high RSR, $R\to1$ and EM-GS degenerates into plain GS, so the two become nearly
indistinguishable — visible in every high-SNR figure. Evaluated with exponentially scaled Bessel functions
to avoid overflow, never as a raw ratio; checked against SciPy and against the quadrature definition
$I_n(\kappa)=\frac{1}{\pi}\int_0^\pi e^{\kappa\cos t}\cos(nt)\,dt$.

### 4.4 Genie ZF with known phase

$$\mathbf{r} = \mathbf{z}\odot e^{\jmath\boldsymbol{\theta}}-\mathbf{b},\qquad
\left(\mathbf{M}\mathbf{M}^{H}\right)\hat{\mathbf{u}} = \mathbf{M}\mathbf{r} \tag{15}$$

A benchmark, not a valid estimator. It is allowed to sit *below* the CRLB of §4.6, because that bound
applies to estimators seeing only $\mathbf{z}$.

### 4.5 Exhaustive search — LS and ML

$$J_{\mathrm{LS}}(\mathbf{u}) = \left\|\, \mathbf{z} - |\mathbf{M}^{H}\mathbf{u}+\mathbf{b}|\,\right\|_2^2 \tag{16}$$

$$J_{\mathrm{ML}}(\mathbf{u}) = \sum_q \left[-\frac{|\lambda_q|^2}{\sigma^2}
+ \log I_0\!\left(\frac{2 z_q|\lambda_q|}{\sigma^2}\right)\right] \tag{17}$$

LS and ML are not assumed identical: (17) is the $\mathbf{u}$-dependent part of the exact Rician
log-likelihood, evaluated in the log domain. Feasible for Figs. 7(a)/8 ($4^3=64$ candidates) but not for
Fig. 7(b) ($16^6\approx1.7\times10^7$) — which is exactly why Cui omits it there and so do we.

### 4.6 Cramér–Rao lower bound

$$p(z\mid\lambda) = \frac{2z}{\sigma^2}\exp\!\left(-\frac{z^2+|\lambda|^2}{\sigma^2}\right)
I_0\!\left(\frac{2z|\lambda|}{\sigma^2}\right) \tag{18}$$

$$\mathbf{F} = \sum_q \beta_q\, \mathbf{m}_q\mathbf{m}_q^{H},\qquad
\beta_q = \frac{\mathbb{E}\left[z_q^2R^2(\kappa_q)\right]-|\lambda_q|^2}{\sigma^4} \tag{19}$$

$\beta_q$ is evaluated by numerical quadrature over a two-sided window centred on $|\lambda|$, never
clipped to its high-SNR limit. As $\mathrm{SNR}\to\infty$, $\beta_q\to 1/(2\sigma^2)$, so
$\mathbf{F}\to\frac{1}{2\sigma^2}\mathbf{M}\mathbf{M}^{H}$ and the bound approaches
$2\sigma^2(\mathbf{M}\mathbf{M}^{H})^{-1}$ — exactly $10\log_{10}2 = 3.0103$ dB above the genie-ZF
covariance. That is the price of losing phase; the audit measures it at 3.0103 dB.

---

## 5. Metrics and aggregation

### 5.1 NMSE — ratio of sums, never mean of ratios

$$\mathrm{NMSE} = \frac{\sum_{\mathrm{trials}}\|\mathbf{s}-\tilde{\mathbf{s}}\|_2^2}
{\sum_{\mathrm{trials}}\mathbb{E}\|\mathbf{s}\|_2^2}, \qquad
\mathrm{NMSE_{dB}} = 10\log_{10}(\mathrm{NMSE}) \tag{20}$$

For channel estimation, Frobenius energies:
$\mathrm{NMSE}_G=\sum\|\hat{\mathbf{G}}-\mathbf{G}\|_F^2/\sum\|\mathbf{G}\|_F^2$.

- It is $10\log_{10}$, **not** $20\log_{10}$ — NMSE is already a power ratio, so a factor-2-in-dB error is the trap.
- The denominator is the *expected* symbol energy $K$ for unit-energy QAM, not a per-trial $\|\mathbf{s}\|^2$.
- $\tilde{\mathbf{s}}$ is the *continuous* solver output, before any constellation demapping.

Averaging per-trial dB values gives a different and smaller number — a geometric rather than arithmetic
mean of the energies. The audit constructs a case where the two differ by 7 dB.

### 5.2 BER — global bit errors over global bits

$$\mathrm{BER} = \frac{\sum_{\mathrm{trials}}(\text{bit errors})}{\sum_{\mathrm{trials}}(\text{bit count})} \tag{21}$$

Not the mean of per-trial BERs. Gray mapping gives $\mathrm{SER}/\log_2 M \le \mathrm{BER} \le \mathrm{SER}$;
the audit verifies the Gray property by walking every adjacent constellation pair. Zero-error points are
*omitted* from log axes rather than plotted at zero; their one-sided 95% Wilson upper bounds ($z=1.645$)
are retained in the aggregate files.

---

## 6. Track A — reproduction results

All five figures were generated from the paper's stated configuration. Where a curve disagrees with Cui,
the disagreement is reported rather than tuned away.

| Figure | Configuration | Trials | Agreement with Cui |
|---|---|---|---|
| Fig. 5 — NMSE vs SNR | 36×3, 16-QAM, RSR 12 dB | 2 000/pt | qualitative, ≈2 dB offset |
| Fig. 6 — NMSE vs RSR | 36×3, 16-QAM, SNR 3 dB | 500/pt | qualitative, ≈2 dB offset |
| Fig. 7(a) — BER vs SNR | 36×3, 4-QAM | 333 000 | close — ratios 0.78–0.97 |
| Fig. 7(b) — BER vs SNR | 100×6, 16-QAM | 122 000 | close — gap 3.79–3.97 dB |
| Fig. 8 — BER vs RSR | 36×3, 4-QAM, SNR 3 dB | 239 000 | close — 39.1× over 0→20 dB |

![Fig. 5](https://github.com/haatim10/Rydberg/raw/track-a-cui-reproduction/results/final_figures/fig5_clean.png)

**Fig. 5 — detection NMSE vs SNR.** GS and EM-GS converge as SNR rises, exactly as $R(\kappa)\to1$ in
eq. (14) predicts. Genie ZF sits below the CRLB because it is given the phase.

![Fig. 6](https://github.com/haatim10/Rydberg/raw/track-a-cui-reproduction/results/final_figures/fig6_clean.png)

**Fig. 6 — detection NMSE vs RSR.** ZF is flat in RSR (fitted slope −0.0026 dB/dB, not significant), as it
must be since the genie ZF error does not depend on $\mathbf{b}$. That flatness is a correctness check.

![Fig. 7a](https://github.com/haatim10/Rydberg/raw/track-a-cui-reproduction/results/final_figures/fig7a_clean.png)

**Fig. 7(a) — BER vs SNR, small scale.** 333 000 trials, 1 998 000 bits per algorithm. At 240 000 bits the
one-sided 95% Wilson bound is $1.1\times10^{-5}$, below Cui's plotted floor of $\approx5\times10^{-5}$.

![Fig. 7b](https://github.com/haatim10/Rydberg/raw/track-a-cui-reproduction/results/final_figures/fig7b_clean.png)

**Fig. 7(b) — BER vs SNR, large scale.** Exhaustive search excluded exactly as Cui excludes it.
Measured EM-GS-to-ZF gap 3.79–3.97 dB; Cui states "between 3 ~ 4 dB".

![Fig. 8](https://github.com/haatim10/Rydberg/raw/track-a-cui-reproduction/results/final_figures/fig8_clean.png)

**Fig. 8 — BER vs RSR.** Improvement from RSR 0 to 20 dB measured at 39.1×; Cui states "more than one
order of magnitude".

> **A defect in the paper, resolved empirically.** Cui's Fig. 8 *caption* says 16-QAM; the *body text* says
> 4-QAM. Rather than guess, both were run. Median BER ratio to Cui's published curve: **0.82 at 4-QAM**
> versus **24.12 at 16-QAM**. The body text is correct and the caption is in error. The 16-QAM variant is
> retained as a diagnostic, clearly labelled as not a reproduction.

---

## 7. HS-GS — the proposed estimator

Cui's row adapter is separable across receive elements: element $n$ is estimated from $z_n$ alone. But a
ULA channel is not an arbitrary vector — eq. (5) is a coupling **along** $n$, in exactly the direction an
unstructured sweep cannot see. HS-GS enforces that coupling without ever estimating an angle.

### 7.1 The constraint is exact, not a surrogate

By **Kronecker's theorem**, a length-$N$ sequence is a sum of $L$ complex exponentials *if and only if* its
Hankel matrix has rank $L$. So the feasible set is precisely the set of channels the ULA model can generate:

$$\min_{\mathbf{G}}\; J(\mathbf{G}) = \left\|\mathbf{Z} - |\mathbf{G}\mathbf{S}+\mathbf{B}|\right\|_F^2
\quad\text{s.t.}\quad \mathrm{rank}\,\mathcal{H}(\mathbf{g}_k) \le L_k \tag{22}$$

No angle grid is introduced and no angle is ever estimated — $\theta$ and $\alpha$ do not appear.

### 7.2 Why Cadzow rather than OMP or ESPRIT

Three candidates were built and measured before choosing:

- **Angular OMP** discretizes $\psi$ onto a grid, so off-grid paths leave an irreducible bias — measured at
  ≈6.9% residual even on noiseless, exactly-structured data. Not grid-free.
- **ESPRIT** is grid-free and exact in the noiseless case (angles recovered to $7.6\times10^{-15}$), but it
  is a *parameter estimator*, not a projection: not idempotent, no variational characterization, and it
  commits hard to $\hat L$ angles. With the pencil bound exceeded it collapsed to −3.45 dB.
- **Cadzow** is grid-free *and* a genuine alternating projection between two closed sets, each with an exact
  projector: rank-$\le L$ matrices via SVD truncation (Eckart–Young–Mirsky), and Hankel matrices via
  anti-diagonal averaging. That gives the structural step a variational meaning the other two lack.

### 7.3 The projection must live inside the iteration

Projecting once then running GS to convergence does nothing, because GS is a contraction toward its own
unstructured fixed point — the fixed points of $T_{\mathrm{GS}}^{\infty}\circ P_S$ are exactly those of
$T_{\mathrm{GS}}$. Measured, not assumed: the gain decays monotonically with the number of unconstrained
iterations after projection — **+1.30 dB at 1, +0.06 at 10, exactly 0.00 at 50**. Interleaving instead gives

$$T = P_S \circ T_{\mathrm{GS}}, \qquad \text{fixed points satisfy } \mathbf{g} = P_S(T_{\mathrm{GS}}(\mathbf{g})) \tag{23}$$

simultaneously structured and consistent with the measurement update. Every measurement step is a genuine
`max_iter=1` call into the audited EM-GS on the exact model; chaining such calls reproduces a single
`max_iter=t` call bit-for-bit.

### 7.4 Model order from held-out pilots

The in-sample residual cannot select $\hat L$: a projection constrains $\mathbf{G}$ to a smaller set, so it
can only fit the fitted pilots worse, and the in-sample criterion therefore always prefers the largest $L$.
This was a real design flaw found in testing — $\Delta J>0$ while $\Delta\mathrm{NMSE}<0$, with sign
agreement of only 1–8 out of 10. The fix splits the pilot columns and scores on the held-out half, using no
ground truth:

$$\hat L = \arg\min_{L}\ \left\|\mathbf{Z}_{\mathrm{val}} -
\left|\hat{\mathbf{G}}(L)\mathbf{S}_{\mathrm{val}}+\mathbf{B}_{\mathrm{val}}\right|\right\|_F^2 \tag{24}$$

### 7.5 The identifiability caveat that drives everything

A length-$N$ Hankel matrix has rank at most

$$\mathrm{cap}(N) = \max_{p}\min(N-p,\,p+1) = \lceil N/2 \rceil \tag{25}$$

So when $L_k \ge \lceil N/2\rceil$ the true channel already saturates the achievable rank and **the
constraint is vacuous**. At $N=8$ the cap is 4, so for $L_k\ge5$ — 60% of the $\mathcal{U}\{3..7\}$ prior —
the structure carries no information at all. This is a property of the configuration, not of the algorithm,
and it is why the array-size hypothesis is the right thing to test.

> **Verified reduction.** When $\hat L$ reaches the rank cap the projection is a no-op and HS-GS reduces to
> EM-GS **bit-for-bit** — checked at $(N=8, P=10)$ and $(N=16, P=30)$ with
> $\lVert\hat{\mathbf{G}}_{\mathrm{HS}}-\hat{\mathbf{G}}_{\mathrm{EM}}\rVert_\infty = 0.00\times10^{0}$.
> Those trials are **exact ties**, not small differences — which matters for reading test H below.

---

## 8. Track B — results

Three estimators — biased GS, EM-GS and HS-GS — on **identical** common-random-number worlds, all on the
exact model of eq. (2). Fixed throughout: $K=3$, $L_k\sim\mathcal{U}\{3..7\}$, RSR = 12 dB, $t_0=50$,
$\beta=1$, $c=1$, $d=\lambda/2$, master seed 20250820.

### 8.1 Experiments and trial budget

Per trial the store keeps the error numerator $\|\hat{\mathbf{G}}-\mathbf{G}\|_F^2$ for each estimator
**and** the denominator $\|\mathbf{G}\|_F^2$ separately, so the pooled ratio-of-sums of eq. (20) — and any
bootstrap of it — is exactly reconstructible. Bootstrap CIs resample *trials*, paired across estimators so
CRN pairing is preserved, 2 000 resamples.

| Experiment | Sweep | Points | Trials |
|---|---|---|---|
| B1 / B2 (frozen baseline) | SNR, then $P$, at $N=8$; GS and EM-GS only | 18 | 400/pt |
| B3 | NMSE vs SNR, $N\in\{8,16,32\}\times P\in\{10,30\}$ | 36 | 21 700 |
| B4 | NMSE vs pilot length $P$, at $N=16$ | 6 | 2 400 |
| B5 | scaling summary, derived from B3 | — | — |

B4's $P=10$ and $P=30$ points are the same CRN worlds B3 already evaluates, so they were **copied rather
than recomputed**. $N=16$ was fixed *a priori* as the smallest tested array whose rank cap (8) exceeds
max $L_k$ (7) — not chosen from the curves.

### 8.2 The adaptive trial rule

Points started at 400 trials. The rule for extending was fixed before looking at the numbers and applied
mechanically: extend only if the bootstrap 95% CI on the gain **(a)** contains 0, so the *sign* is
undetermined, or **(b)** is wider than 1.5 dB, so the *magnitude* is undetermined. **5 of 36 points**
qualified and went to 1 200 trials; the other 31 stayed at 400. Nothing already computed was recomputed.

### 8.3 B3 — channel NMSE vs SNR

![B3 gain](results/track_b/final/b3_gain_vs_snr.png)

**B3 — HS-GS gain over EM-GS.** The array-size ordering is clean and never crosses at either pilot length.
At $P=30$ the curves are flat in SNR; at $P=10$ all three slope downward and $N=8$ and $16$ cross into
negative territory. No error bars are drawn — the CIs are in the table below.

![B3 P=30](results/track_b/final/b3_nmse_vs_snr_P30.png)

**B3 — raw NMSE, $P=30$.** At $N=8$ the three curves are visually indistinguishable, at $N=16$ a thin
separation opens, and only at $N=32$ does HS-GS pull clearly away. The absolute axis hides an effect the
gain axis makes legible.

![B3 P=10](results/track_b/final/b3_nmse_vs_snr_P10.png)

**B3 — raw NMSE, $P=10$.** HS-GS visibly crosses *above* EM-GS past ≈5 dB at $N=8$ and ≈13 dB at $N=16$ —
the negative gain, directly visible.

| N | P | SNR | n | GS | EM-GS | HS-GS | Gain | Gain 95% CI | Win | Active | L̂ |
|--:|--:|----:|--:|---:|------:|------:|-----:|:------------|----:|-------:|---:|
| 8 | 10 | -5 | 400 | 6.92 | 6.59 | 5.13 | **+1.47** | [+1.30, +1.64] | 76% | 86% | 1.86 |
| 8 | 10 | +0 | 400 | 2.29 | 2.20 | 1.35 | **+0.85** | [+0.61, +1.07] | 61% | 83% | 2.06 |
| 8 | 10 | +5 | 1200 | -2.31 | -2.42 | -2.52 | **+0.10** | [-0.01, +0.20] | 40% | 70% | 2.53 |
| 8 | 10 | +10 | 400 | -7.18 | -7.26 | -6.48 | -0.78 | [-1.12, -0.47] | 22% | 59% | 3.00 |
| 8 | 10 | +15 | 400 | -11.62 | -11.66 | -10.34 | -1.32 | [-1.77, -0.88] | 17% | 47% | 3.35 |
| 8 | 10 | +20 | 1200 | -15.48 | -15.50 | -13.27 | -2.23 | [-2.70, -1.77] | 11% | 40% | 3.47 |
| 8 | 30 | -5 | 400 | 1.57 | 0.41 | -0.37 | **+0.78** | [+0.66, +0.91] | 68% | 87% | 1.93 |
| 8 | 30 | +0 | 400 | -4.88 | -5.22 | -4.96 | -0.26 | [-0.39, -0.12] | 35% | 75% | 2.55 |
| 8 | 30 | +5 | 400 | -10.63 | -10.74 | -10.34 | -0.41 | [-0.55, -0.26] | 27% | 59% | 3.20 |
| 8 | 30 | +10 | 400 | -15.76 | -15.80 | -15.57 | -0.23 | [-0.34, -0.12] | 19% | 40% | 3.54 |
| 8 | 30 | +15 | 400 | -20.73 | -20.74 | -20.58 | -0.17 | [-0.27, -0.08] | 12% | 26% | 3.72 |
| 8 | 30 | +20 | 400 | -26.00 | -26.00 | -25.89 | -0.11 | [-0.22, -0.02] | 11% | 19% | 3.80 |
| 16 | 10 | -5 | 400 | 6.91 | 6.62 | 4.21 | **+2.41** | [+2.22, +2.62] | 92% | 97% | 2.34 |
| 16 | 10 | +0 | 400 | 2.21 | 2.10 | 0.07 | **+2.03** | [+1.84, +2.23] | 87% | 96% | 2.40 |
| 16 | 10 | +5 | 400 | -2.40 | -2.50 | -3.36 | **+0.85** | [+0.67, +1.03] | 68% | 93% | 3.43 |
| 16 | 10 | +10 | 400 | -7.17 | -7.22 | -7.54 | **+0.32** | [+0.07, +0.55] | 61% | 90% | 4.61 |
| 16 | 10 | +15 | 1200 | -11.61 | -11.64 | -11.49 | -0.15 | [-0.42, +0.10] | 57% | 88% | 4.97 |
| 16 | 10 | +20 | 1200 | -15.52 | -15.54 | -14.86 | -0.68 | [-1.09, -0.26] | 56% | 88% | 5.09 |
| 16 | 30 | -5 | 400 | 1.63 | 0.54 | -1.06 | **+1.60** | [+1.45, +1.75] | 90% | 97% | 2.56 |
| 16 | 30 | +0 | 400 | -4.95 | -5.25 | -5.87 | **+0.61** | [+0.48, +0.75] | 72% | 96% | 3.64 |
| 16 | 30 | +5 | 400 | -10.59 | -10.70 | -11.17 | **+0.47** | [+0.32, +0.60] | 70% | 93% | 4.55 |
| 16 | 30 | +10 | 400 | -15.80 | -15.84 | -16.46 | **+0.62** | [+0.48, +0.75] | 79% | 92% | 4.92 |
| 16 | 30 | +15 | 400 | -20.78 | -20.79 | -21.44 | **+0.65** | [+0.56, +0.75] | 79% | 92% | 5.25 |
| 16 | 30 | +20 | 400 | -25.94 | -25.95 | -26.58 | **+0.63** | [+0.54, +0.73] | 77% | 90% | 5.46 |
| 32 | 10 | -5 | 400 | 6.93 | 6.62 | 2.30 | **+4.33** | [+4.10, +4.55] | 99% | 100% | 2.07 |
| 32 | 10 | +0 | 400 | 2.26 | 2.17 | -1.66 | **+3.83** | [+3.62, +4.01] | 98% | 100% | 2.28 |
| 32 | 10 | +5 | 400 | -2.42 | -2.53 | -5.23 | **+2.70** | [+2.48, +2.91] | 92% | 99% | 3.95 |
| 32 | 10 | +10 | 400 | -7.16 | -7.21 | -10.08 | **+2.87** | [+2.55, +3.19] | 94% | 100% | 5.08 |
| 32 | 10 | +15 | 400 | -11.71 | -11.75 | -14.34 | **+2.59** | [+2.21, +2.96] | 89% | 99% | 5.71 |
| 32 | 10 | +20 | 1200 | -15.39 | -15.41 | -18.12 | **+2.71** | [+2.25, +3.18] | 86% | 99% | 5.78 |
| 32 | 30 | -5 | 400 | 1.64 | 0.53 | -2.51 | **+3.04** | [+2.90, +3.17] | 99% | 100% | 2.45 |
| 32 | 30 | +0 | 400 | -4.91 | -5.25 | -7.67 | **+2.42** | [+2.27, +2.57] | 97% | 100% | 4.05 |
| 32 | 30 | +5 | 400 | -10.58 | -10.69 | -12.96 | **+2.27** | [+2.12, +2.40] | 96% | 100% | 5.02 |
| 32 | 30 | +10 | 400 | -15.78 | -15.81 | -18.16 | **+2.35** | [+2.21, +2.50] | 97% | 100% | 5.51 |
| 32 | 30 | +15 | 400 | -20.75 | -20.76 | -23.27 | **+2.51** | [+2.38, +2.63] | 98% | 100% | 5.88 |
| 32 | 30 | +20 | 400 | -25.93 | -25.94 | -28.54 | **+2.60** | [+2.47, +2.74] | 98% | 100% | 5.95 |
Pooled NMSE$_G$ in dB (ratio of sums, eq. 20). "Gain" is
$10\log_{10}(\sum \mathrm{num}_{\mathrm{EM}} / \sum \mathrm{num}_{\mathrm{HS}})$ — positive means HS-GS is better. "Win" is the per-trial fraction where
HS-GS beats EM-GS. "Active" is the fraction of trials where the structural constraint was not vacuous.

### 8.4 B4 — channel NMSE vs pilot length

![B4](results/track_b/final/b4_nmse_vs_pilots_N16.png)

**B4 — $N=16$, SNR = 5 dB.** HS-GS is below both baselines at every pilot length from 6 to 40, and the
advantage does not wash out as pilots grow.

| P | GS | EM-GS | HS-GS | Gain | Gain 95% CI | Win | Active |
|--:|---:|------:|------:|-----:|:------------|----:|-------:|
| 6 | 1.65 | 1.65 | 1.07 | **+0.58** | [+0.42, +0.75] | 48% | 68% |
| 10 | -2.40 | -2.50 | -3.36 | **+0.85** | [+0.67, +1.03] | 68% | 93% |
| 14 | -5.30 | -5.41 | -6.23 | **+0.82** | [+0.65, +1.00] | 73% | 94% |
| 20 | -8.11 | -8.22 | -8.66 | **+0.44** | [+0.30, +0.59] | 70% | 95% |
| 30 | -10.59 | -10.70 | -11.17 | **+0.47** | [+0.32, +0.60] | 70% | 93% |
| 40 | -12.13 | -12.23 | -12.70 | **+0.48** | [+0.36, +0.59] | 71% | 90% |
All six CIs lie strictly above zero. Note $P=6$ ($=2K$, the minimum): the win rate is **48%**, below half,
while the pooled gain is **+0.58 dB**. Not a contradiction — the pooled metric is a ratio of sums, moved by
*how much* HS-GS wins when it wins, not how often. At the shortest pilot the constraint is active in only
68% of trials and the order selector has just 2 held-out columns, so it engages less often but pays off
substantially when it does.

### 8.5 B5 — scaling with array size

![B5](results/track_b/final/b5_gain_scaling_vs_N.png)

**B5 — the hypothesis test.** Both panels monotone in $N$, both crossing their neutral line between $N=8$
and $N=16$.

| N | cap $\lceil N/2\rceil$ | $P(L_k<\text{cap})$ | $2NK$ | $3\mathbb{E}[\sum L_k]$ | $\rho(N)$ | Mean gain | P=10 | P=30 | Win rate | Active |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 4 | 20% | 48 | 44.6 | 1.08× | −0.19 | −0.32 | −0.07 | 33.5% | 58% |
| 16 | 8 | 100% | 96 | 44.6 | 2.15× | **+0.78** | **+0.80** | **+0.76** | 74.0% | 93% |
| 32 | 16 | 100% | 192 | 44.6 | 4.30× | **+2.85** | **+3.17** | **+2.53** | 95.3% | 100% |

Structural redundancy $\rho(N) = 2NK / 3\sum L_k$: unstructured $\mathbf{G}$ has $2NK$ real parameters; the
geometric model has 3 per path (one angle, one complex gain). Only $N$ moves with the array — the path
budget does not — so $\rho$ grows linearly in $N$. Increment per doubling: −0.19 → +0.78 dB (+0.97), then
+0.78 → +2.85 dB (+2.07).

> **Terminology — read this before quoting $\rho(N)$.** It is a **parameter count**, and the rank cap is an
> algebraic fact about Hankel matrices. Together they are a *structural redundancy* / *representational
> informativeness* argument. They are **not an identifiability theorem**. Nothing here proves the
> constrained problem has a unique solution, that the alternating projection converges to it, or that the
> estimator attains any bound. Three values of $N$ also cannot identify a functional form — no growth law
> is fitted and none should be quoted.

### 8.6 Interpretation tests A–H

Eight adversarial checks, all evaluated numerically from the stored per-trial data rather than from
expectation.

| | Question | Answer |
|---|---|---|
| A | Is $N=8$ still mixed at scale? | Mixed, and systematically **negative** at high SNR: 3 points positive, 8 negative, 1 straddling; worst −2.23 dB |
| B | Credible positive gain at $N=16$? | Yes — 10/12 points with CI entirely above 0 |
| C | Is $N=32$ larger than $N=16$? | Yes — CI strictly above at **12/12** shared points |
| D | Does win rate increase with $N$? | Yes, monotone — 33.5% → 74.0% → 95.3% |
| E | Consistent with the redundancy argument? | Yes — both monotone; sign flip lands where cap first exceeds max $L_k$ |
| F | Driven by a few catastrophic EM-GS trials? | No — dropping the worst 5% of EM-GS trials moves the gain by a median of −0.05 dB |
| G | Does HS-GS floor out at high SNR? | No at $P=30$ (slopes −1.03/−1.01/−1.04 dB/dB vs EM-GS −1.02/−1.01/−1.01). At $P=10$ both flatten together |
| H | Stable pooled vs median? | 25/36 raw, **33/36 once exact ties are excluded** |

> **Two readings that need care.**
>
> **Test H's raw 25/36 is an artifact, not a disagreement.** When the order selector reaches the rank cap,
> HS-GS *is* EM-GS bit-for-bit, so the per-trial ratio is exactly 1.000 and contributes a median gain of
> +0.00 dB. At $N=8$ those ties are the majority — the tie fraction matches the constraint-inactive
> fraction to the trial (57.8% tie vs 42.2% active at $P=10$, SNR 20). Excluding ties, the medians agree in
> sign with pooled and are *more* negative at $N=8$.
>
> **The $P=10$ downslope is not an array-size effect.** It appears at all three $N$. With only 3 held-out
> pilot columns the order selector increasingly picks $\hat L$ at the rank cap, switching the projection
> off — constraint-active falls 86% → 40% at $N=8$ as SNR rises. That is pilot-starved order selection, not
> the structural prior failing at high SNR. The $P=30$ panel, where the constraint stays active 90–100%, is
> the cleaner read on the structure itself.

---

## 9. Audit and reproducibility

Two audits: a 141-check deep audit of the whole repository (69 + 72 checks, zero BLOCKER/HIGH/MEDIUM
findings), and a 21-check Step-0 gate run immediately before the final Monte Carlo, whose job was to make
the comparison falsifiable rather than flattering.

| # | Step-0 check | Result |
|---|---|---|
| 1 | Track A untouched | HEAD `ce118f0` = pushed remote, worktree clean |
| 2 | Generator implements the frozen ULA model | eq. (5) closed form vs generator: max\|diff\| 0.00e+00; $\theta\in[-\pi/2,\pi/2]$; $L\cdot\mathbb{E}\|\alpha\|^2 = 1.0002$ |
| 3 | Observation is exact | $\mathbf{Z}=\|\mathbf{GS}+\mathbf{B}+\mathbf{W}\|$ bit-exact over 50 worlds, max dev 0.00e+00 |
| 4 | Linearized-estimator tripwire | solver monkeypatched to raise → **0 calls** |
| 5 | Identical CRN worlds across estimators | world is a deterministic function of (trial, $P$, SNR) |
| 6 | $L_k$ per frozen spec | 6 000 draws: support {3..7}, mean 5.0111, uniformity cv 0.0190 |
| 7 | HS-GS is the audited version | sha256 `59ea0d0a…`, byte-identical to HEAD and to the smoke-run commit |
| 8 | Inactive constraint reduces to baseline | $\lVert\hat{\mathbf{G}}_{\mathrm{HS}}-\hat{\mathbf{G}}_{\mathrm{EM}}\rVert_\infty=0.00\mathrm{e}{+}00$ at two configurations |

### 9.1 Checkpointing, and why it mattered

Each Monte Carlo point writes an `.npz` every 25 trials; a rerun loads what is there and computes only the
missing trial indices, asserting no duplicate index on every write. Verified before launch by showing that
a *resumed* 2+2-trial run is identical to a fresh 4-trial run across 36 points and 324 arrays.

That verification earned its keep. The final run was interrupted four times — once by a container reclaim
and three times by my own error (§10.3). Every resume picked up at the exact trial where it stopped.
**No trial was ever computed twice, and no data was lost.**

---

## 10. Corrections and failures

Errors found and fixed during the work, recorded because a reproduction study that reports only its
successes is not a reproduction study. Several of these were mine.

### 10.1 Findings I retracted after checking

- **A false 38.901 match.** I measured a near-match to Cui's Figs. 5–6 under a particular angle handling,
  then caught that it was entirely an angle-*clipping* artifact: clip gave +1.47 dB, wrap gave +0.01 dB.
  Retracted before it reached any conclusion.
- **A "polarization bug" that wasn't.** I called per-element polarization sampling a bug; Cui §VI-A
  specifies it explicitly. Corrected.
- **A wrong Monte Carlo claim.** I asserted that zero observed bit errors required 10× more trials.
  Recomputing the Wilson bound gave $1.1\times10^{-5}$ against Cui's plotted floor of $\approx5\times10^{-5}$
  — the claim was wrong and was withdrawn.

### 10.2 Design flaws in my own estimator, found by testing it

- **The in-sample objective is invalid as a selector** — $\Delta J>0$ while $\Delta\mathrm{NMSE}<0$, sign
  agreement 1–8/10. Led to the held-out rule of eq. (24).
- **An unconstrained re-solve erases the projection** — gain decaying +1.30 → +0.06 → 0.00 dB. Led to the
  interleaved map of eq. (23).
- **MDL order estimation was poor on clean data** (12–18 correct out of 40), so it was replaced by the
  held-out pilot residual.

### 10.3 Process failures during the final run

- **I wrote into the frozen Track-A tree.** The Track-B plotting script had inherited an output path
  pointing at Track A's `final_figures/` and deposited five untracked figures there. No tracked file was
  modified and nothing was committed to that branch, but writing there at all violated the freeze. Files
  deleted, script redirected, Track A verified back at `ce118f0`, clean and identical to its remote.
- **I killed my own Monte Carlo three times.** I launched the run in the background and then blocked on it
  with a foreground wait; each wait hit its timeout and was terminated with SIGTERM, which the harness sends
  to the whole process group — taking the `nohup`'d job with it, since `nohup` blocks SIGHUP, not SIGTERM.
  I also initially misattributed two of those stalls to container restarts. Fixed by detaching the run into
  its own process group with `setsid`, after which it completed in the estimated 11 minutes.

---

## 11. What is and is not established

### 11.1 Supported by the data

- The HS-GS advantage over EM-GS **grows monotonically with $N$**, and the sign of the effect flips between
  $N=8$ and $N=16$ — precisely where $\lceil N/2\rceil$ first exceeds max $L_k$.
- Credible positive gain at $N=16$ (10/12 points) and $N=32$ (12/12), by bootstrap CI on the pooled
  ratio-of-sums.
- A real **deficit** at $N=8$, systematic at high SNR — reported rather than hidden.
- The gain is not outlier-driven (test F) and shows no high-SNR floor at adequate pilot length (test G).
- Track A: Figs. 7(a), 7(b) and 8 reproduce Cui closely; Figs. 5–6 qualitatively with a documented, traced
  ≈2 dB offset.

### 11.2 Not established, and should not be claimed

- **No identifiability theorem.** The redundancy argument is a parameter count.
- **No convergence guarantee** for the alternating projection, and no claim that it attains any bound.
- **No growth law.** Three values of $N$ cannot identify a functional form; nothing is fitted.
- **Nothing about $N=64$** or any array size not tested.
- **Two points remain sign-undetermined** after 1 200 trials: $(N{=}8, P{=}10, \mathrm{SNR}\,5)$ at
  +0.10 [−0.01, +0.20] and $(N{=}16, P{=}10, \mathrm{SNR}\,15)$ at −0.15 [−0.42, +0.10]. Both CIs bound the
  effect tightly near zero, so the magnitude is determined and only the sign of a ≈0 effect is open;
  extension was stopped rather than burn hours refining a conclusive null.

### 11.3 Status

Track A is frozen. Track B is ready to freeze, with the two null points flagged above. No Track C,
machine-learning, or unrolling work has been started.

---

*Every equation transcribed from the implementation and verified numerically. Uncertainty is retained in
the tables, never drawn on the plots.*
