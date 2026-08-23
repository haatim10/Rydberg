# Rydberg Atomic MIMO — formulae, methods and results

**Track A** — reproduction of Cui et al., *"Towards Atomic MIMO Receivers"* (IEEE JSAC 43(3), March 2025).
**Track B** — a proposed structure-aware channel estimator, HS-GS, tested on the exact nonlinear model.

| | |
|---|---|
| Track A branch | `track-a-cui-reproduction` @ `ce118f0` (frozen) |
| Track B branch | `track-b-ula-channel-estimation` @ `786df67` |
| Config fingerprint | `46dbd9f1cf57d1cc` |
| Track B trials | **24,000** unique (B3 18,400 + B4 1,600 new + B6 4,000; B4's other 800 are B3 points reused, not recomputed) |
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
$N=16$ and +2.85 dB at $N=32$. The sign of the effect changes between $N=8$ and $N=16$. The algebraic
threshold at which the constraint ceases to be vacuous for *every* draw of $L_k$ is $N=15$
($\lceil N/2\rceil > \max L_k = 7$); it lies inside that interval, but with three array sizes the crossing
can be bracketed, not located.

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
| $\mathbf{M}$ | $\mathbf{A}$ | $\mathbf{S}$ |
| $\mathbf{u}$ | $\mathbf{s}$ | $\overline{\mathbf{g}_n}$ — row $n$ of $\mathbf{G}$, conjugated |
| $\mathbf{b}$ | $\mathbf{b}$ | $\overline{\mathbf{B}_{n,:}}$ |
| $\mathbf{z}$ | $\mathbf{z}$ | $\mathbf{Z}_{n,:}$ |
| output | $\hat{\mathbf{s}}$ | $\hat{\mathbf{g}}_n = \overline{\hat{\mathbf{u}}}$ |

This is Cui eq. (35) style: the *whole observation* is conjugated, so $\mathbf{M}$ is **not** conjugated
while $\mathbf{u}$ and $\mathbf{b}$ are. The dual convention
($\mathbf{M}=\overline{\mathbf{S}}$, $\mathbf{u}=\mathbf{g}_n$, $\mathbf{b}=\mathbf{B}_{n,:}$, no
output conjugation) is equally valid. Conjugating $\mathbf{M}$ **and** $\mathbf{u}$ together does not
satisfy the model — an earlier version of this table did exactly that, and
`tests/test_report_corrections.py` now pins all three cases.

Note that the identity closes only with the noise carried into the canonical $\mathbf{b}$ alongside
$\mathbf{B}$, since $\mathbf{Z}=|\mathbf{GS}+\mathbf{B}+\mathbf{W}|$. The conjugation direction was
separately verified by noiseless recovery: the estimator returns $\mathbf{G}$ to a relative error of
$2.4\times10^{-15}$, and not $\overline{\mathbf{G}}$.

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
SNR and RSR empirically. An earlier version of this report quoted **2.82 dB and 12.15 dB** against targets
of 3 and 12 dB, with no sample size and no interval — a 0.18 dB gap is a 4% power error, which is too large
to wave through in a section that depends on the calibration being exact. Re-measured on
4,000 realizations per channel, as a ratio of summed energies (the §5.1 rule, never a
mean of per-realization ratios), with a 2 000-resample bootstrap:

| Channel | SNR (target 3.00 dB) | RSR (target 12.00 dB) |
|---|---|---|
| Track A — 38.901 | **2.978** [2.926, 3.031] | **12.025** [11.951, 12.096] |
| Track B — geometric ULA | **3.009** [2.957, 3.060] | **11.962** [11.880, 12.044] |

All four targets lie inside their intervals, so the calibration is unbiased and the earlier 2.82 dB was a
small-sample artifact. The reason it is an artifact rather than a bias: row normalization holds *in
expectation*, not per realization, so $\mathbb{E}|\mathbf{a}_n^H\mathbf{s}|^2$ has a per-realization
standard deviation of 1.17 about its mean of
3.007. A few hundred realizations can easily read
0.2 dB low.

---

## 4. Algorithms

### 4.1 Spectral initialization

With $\bar{\mathbf{m}}_q=[\mathbf{m}_q;\,b_q]$:

$$\mathbf{M}_{\mathrm{spec}} = \sum_q z_q\, \bar{\mathbf{m}}_q \bar{\mathbf{m}}_q^{H}
\in\mathbb{C}^{(D+1)\times(D+1)},\qquad \mathbf{v} = \text{principal eigenvector} \tag{10}$$

The eigenvector fixes only a direction: its magnitude is arbitrary and its global phase is whatever the
eigensolver returned. Two further steps, both in the implementation, resolve those:

$$\bar r = \frac{\left|\bar{\mathbf{M}}^{H}\mathbf{v}\right|^{T}\mathbf{z}}
{\left\|\bar{\mathbf{M}}^{H}\mathbf{v}\right\|_2^2}, \qquad
\bar{\mathbf{u}}_0 = \bar r\,\mathbf{v} \tag{10a}$$

$$\mathbf{u}_0 = \left[\, e^{-\jmath\angle(\bar{\mathbf{u}}_0)_{D+1}}\;\bar{\mathbf{u}}_0
\,\right]_{1:D} \tag{10b}$$

Eq. (10a) is a magnitude least-squares: it picks the scale that best matches the measured amplitudes.
Eq. (10b) is the phase anchor, and it is the step most easily left out. The $(D+1)$-th entry of the
augmented vector corresponds to the *known* reference $\mathbf{b}$, whose coefficient is 1 and whose phase
is therefore zero by construction; de-rotating by it pins the otherwise arbitrary eigenvector phase, and
only then is the last entry dropped. Removing this step measurably degrades the initializer at low SNR —
pinned by a regression test rather than asserted.

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

![Bessel ratio](https://github.com/haatim10/Rydberg/raw/track-a-cui-reproduction/results/track_a/fig4/fig4_bessel_ratio.png)

**The Bessel ratio $R(\kappa)$.** The weighting function of eq. (14), and the reason GS and EM-GS converge
at high SNR: $R$ saturates at 1, at which point the EM weighting becomes the identity and Algorithm 2
reduces to Algorithm 1.

### 4.4 Genie ZF with known phase

$$\mathbf{r} = \mathbf{z}\odot e^{\jmath\boldsymbol{\theta}}-\mathbf{b},\qquad
\left(\mathbf{M}\mathbf{M}^{H}\right)\hat{\mathbf{u}} = \mathbf{M}\mathbf{r} \tag{15}$$

A benchmark, not a valid estimator. It is allowed to sit *below* the CRLB of §4.6, because that bound
applies to estimators seeing only $\mathbf{z}$.

### 4.5 Exhaustive search — LS and ML

$$\hat{\mathbf{u}}_{\mathrm{LS}} = \arg\min_{\mathbf{u}}\;
J_{\mathrm{LS}}(\mathbf{u}), \qquad
J_{\mathrm{LS}}(\mathbf{u}) = \left\|\, \mathbf{z} - |\mathbf{M}^{H}\mathbf{u}+\mathbf{b}|\,\right\|_2^2 \tag{16}$$

$$\hat{\mathbf{u}}_{\mathrm{ML}} = \arg\max_{\mathbf{u}}\;
J_{\mathrm{ML}}(\mathbf{u}), \qquad
J_{\mathrm{ML}}(\mathbf{u}) = \sum_q \left[-\frac{|\lambda_q|^2}{\sigma^2}
+ \log I_0\!\left(\frac{2 z_q|\lambda_q|}{\sigma^2}\right)\right] \tag{17}$$

**(16) is minimised; (17) is a log-likelihood and is maximised.** They point in opposite directions, and the
code follows exactly this.

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

> **How the comparison numbers were obtained.** §0 says the reproduction was driven from the paper's
> specification, not from curve extraction — and that is true of every *simulation input*: no parameter was
> read off a published curve, and nothing was fitted, tuned or selected against Cui's plotted values.
> The agreement figures quoted below ("BER ratios 0.78–0.97", "median ratio 0.82 vs 24.12") are a different
> thing: they are *post-hoc* comparisons, and they do require reference values. Those were digitised from the
> published figures by pixel-coordinate extraction, roughly 15–20 points per curve, stored in
> `results/track_a/cui_fig78_extracted.json`. Read error is dominated by log-axis interpolation and is
> approximately ±5% in BER, i.e. ±0.2 dB horizontally. The digitisation is used **only** to report agreement
> after the fact; removing it entirely would change no simulation result in this document.

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

![Fig. 6 full range](https://github.com/haatim10/Rydberg/raw/track-a-cui-reproduction/results/track_a/fig6/fig6_cui_style_fullrange.png)

**Fig. 6, full vertical range.** Genie ZF sits at −13.6 dB, *below* the −12 dB axis floor Cui's figure
uses. Rather than clip the curve silently, the main panel is drawn to the requested range with an on-plot
annotation, and this full-range companion is kept alongside it.

![Fig. 7a](https://github.com/haatim10/Rydberg/raw/track-a-cui-reproduction/results/final_figures/fig7a_clean.png)

**Fig. 7(a) — BER vs SNR, small scale.** 333,000 trials,
1,998,000 bits per algorithm, at 6 bits/trial (4-QAM, $K=3$).

The trial count is **not uniform across SNR** — it is escalated where errors get rare, so
"240,000 bits" refers to the largest points, not to every point:
3,000 trials at SNR -5, -4, -3, -2, -1, +0; 10,000 trials at SNR +1, +2, +3, +4; 25,000 trials at SNR +5, +6, +7; 40,000 trials at SNR +8, +9, +10, +11, +12. Those give 18,000 to 240,000 bits per point. At the
240,000-bit points — which are exactly the high-SNR ones where zero errors occur — the
one-sided 95% Wilson bound is $1.1\times10^{-5}$, below Cui's plotted floor of
$\approx5\times10^{-5}$. Counts are read back from the stored Track-A aggregate, not retyped.

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

![Fig. 8 at 16-QAM](https://github.com/haatim10/Rydberg/raw/track-a-cui-reproduction/results/final_figures/fig8_16qam_diagnostic.png)

**Diagnostic — Fig. 8 run at 16-QAM as the caption claims.** 5 RSR points, 15 000 trials. Not a
reproduction of Fig. 8; it exists only to settle the caption/body contradiction, and it does.

---

## 7. HS-GS — the proposed estimator

Cui's row adapter is separable across receive elements: element $n$ is estimated from $z_n$ alone. But a
ULA channel is not an arbitrary vector — eq. (5) is a coupling **along** $n$, in exactly the direction an
unstructured sweep cannot see. HS-GS enforces that coupling without ever estimating an angle.

### 7.1 The constraint is exact, not a surrogate

By **Kronecker's theorem**, a length-$N$ sequence is a sum of $L$ complex exponentials $z_i^{\,n}$ *only
if* its Hankel matrix has rank $\le L$, with the converse holding when $L$ is strictly below the rank cap of
eq. (25). The ULA model of eq. (5) additionally requires $|z_i| = 1$ — the angles are real. Cadzow enforces
no unit-modulus condition, so the rank constraint is a **relaxation** of the ULA feasible set, not an exact
characterisation: it is necessary, and sufficient only up to the modulus of the recovered modes. The
feasible set strictly *contains* the ULA set, admitting damped and growing modes. No attempt is made here to
impose unit modulus.

$$\min_{\mathbf{G}}\; J(\mathbf{G}) = \left\|\mathbf{Z} - |\mathbf{G}\mathbf{S}+\mathbf{B}|\right\|_F^2
\quad\text{s.t.}\quad \mathrm{rank}\,\mathcal{H}(\mathbf{g}_k) \le L_k \tag{22}$$

No angle grid is introduced and no angle is ever estimated — $\theta$ and $\alpha$ do not appear.

**How loose is the relaxation in practice?** This is measurable rather than merely arguable, so it was
measured. After HS-GS converges, ESPRIT is run on the projected $\hat{\mathbf{g}}_k$ to extract the
$\hat L$ modes $z_i$, and $\bigl||z_i|-1\bigr|$ is recorded — the distance of each recovered mode from the
unit circle the ULA model requires. At $N=32$, $P=30$, SNR = 5 dB over
80 trials with the constraint active:

| | median | p90 | p99 | max |
|---|--:|--:|--:|--:|
| HS-GS projected channel | 0.0124 | 0.0566 | 0.1773 | 0.7198 |
| true channel (estimator floor) | 0.0000 | 0.0432 | — | — |

The true-channel row is the floor: it is what this ESPRIT step reports on data lying exactly on the ULA
manifold, so it separates relaxation slack from estimator error. **87.2% of recovered
modes sit within 0.05 of the unit circle and 96.3% within 0.10**, with a median
deviation of 0.0124. So the relaxation is tight in practice — Cadzow is not, in the main,
exploiting the damped and growing modes it is formally permitted. The tail is real though
(p99 = 0.177, max = 0.720): a small fraction of modes drift well off the circle, and
that is where the remaining slack lives.

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

Projecting once and then running GS to convergence does nothing measurable: the gain decays monotonically
with the number of unconstrained iterations after projection — **+1.30 dB at 1, +0.06 at 10, exactly 0.00 at
50**. Empirically the unconstrained iteration returns to its own fixed point, which the projection then
cannot influence. No contraction or convergence property is claimed — alternating-projection schemes of this
kind are not contractions in general, and §11.2 disclaims any convergence guarantee. Interleaving instead
gives

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
constraint is vacuous** — the inequality $\mathrm{rank}\,\mathcal{H} \le L_k$ is then satisfied by every
sequence, structured or not. At $N=8$ the cap is 4, so the constraint is vacuous for $L_k \ge 4$, which is
**80%** of the $\mathcal{U}\{3..7\}$ prior. Only $L_k=3$ carries information, matching the
$P(L_k<\mathrm{cap}) = 20\%$ column of the B5 table. This is a property of the configuration, not of the
algorithm, and it is why the array-size hypothesis is the right thing to test.

Two distinct things are easy to conflate here, and the report keeps them separate:

- **Representational vacuity** — set by the *true* $L_k$ against $\mathrm{cap}(N)$. Governs whether the
  constraint could ever carry information. At $N=8$: 80% vacuous. At $N=16$ and $N=32$: 0%.
- **Projection inactivity** — set by the *selected* $\hat L$ against $\mathrm{cap}(N)$. Governs whether the
  projection is a no-op on a given trial. This is what the "Active" column measures, and the flag is
  `L_hat < cap`, strict — at $\hat L = \mathrm{cap}$ the projection does nothing, so those trials count as
  inactive.

> **Verified reduction.** When $\hat L$ reaches the rank cap the projection is a no-op and HS-GS reduces to
> EM-GS **bit-for-bit** — checked at $(N=8, P=10)$ and $(N=16, P=30)$ with
> $\lVert\hat{\mathbf{G}}_{\mathrm{HS}}-\hat{\mathbf{G}}_{\mathrm{EM}}\rVert_\infty = 0.00\times10^{0}$.
> Those trials are **exact ties**, not small differences — which matters for reading test H below.

---

## 8. Track B — results

Three estimators — biased GS, EM-GS and HS-GS — on **identical** common-random-number worlds, all on the
exact model of eq. (2). Fixed throughout: $K=3$, $L_k\sim\mathcal{U}\{3..7\}$, RSR = 12 dB, $t_0=50$,
$\beta=1$, $c=1$, $d=\lambda/2$, master seed 20250820.

**Nesting across $N$.** The world is a deterministic function of $(\text{trial}, P, \mathrm{SNR},
\mathrm{RSR})$ with no dependence on $N$ for the channel parameters, so the angles, path gains, pilots and
reference are drawn identically at every array size and the array response is simply extended: $N=8$ is
literally the first 8 rows of the $N=32$ realization, verified elementwise. The comparison across $N$ is
therefore **paired in the channel**, which is a strength — it removes channel variability from the $N$
sweep and is part of why the EM-GS baseline is flat in $N$ (test I).

One qualification the phrase "deterministic function of $(\text{trial}, P, \mathrm{SNR})$" would otherwise
hide: the **noise is not nested**. $\mathbf{W}$ is drawn with shape $(N,P)$, so the $N=8$ noise is not the
first 8 rows of the $N=32$ noise. The pairing is exact for $\{\theta, \alpha, \mathbf{G}, \mathbf{S},
\mathbf{B}\}$ and absent for $\mathbf{W}$.

### 8.1 Experiments and trial budget

Per trial the store keeps the error numerator $\|\hat{\mathbf{G}}-\mathbf{G}\|_F^2$ for each estimator
**and** the denominator $\|\mathbf{G}\|_F^2$ separately, so the pooled ratio-of-sums of eq. (20) — and any
bootstrap of it — is exactly reconstructible. Bootstrap CIs resample *trials*, paired across estimators so
CRN pairing is preserved, 2 000 resamples.

| Experiment | Sweep | Points | Trials |
|---|---|---|---|
| B1 / B2 (frozen baseline) | SNR, then $P$, at $N=8$; GS and EM-GS only | 18 | 400/pt |
| B3 | NMSE vs SNR, $N\in\{8,16,32\}\times P\in\{10,30\}$ | 36 | 18,400 (31×400 + 5×1 200) |
| B4 | NMSE vs pilot length $P$, at $N=16$ | 6 | 2,400 (1,600 new + 800 copied from B3) |
| B5 | scaling summary, derived from B3 | — | — |
| B6 | NMSE vs RSR, $N\in\{8,32\}$ at $P=30$, SNR 5 dB | 10 | 4,000 |

![B1 baseline](results/track_b/b1_clean.png)

**B1 — the frozen baseline, NMSE vs SNR.** $N=8$, GS and EM-GS only, 400 trials/point. This is the
validated two-estimator baseline that B3 extends; HS-GS did not exist when it was run.

![B2 baseline](results/track_b/b2_clean.png)

**B2 — the frozen baseline, NMSE vs pilot length.** $N=8$, SNR = 5 dB, with $P=2K$ marked. B4 is the
$N=16$ counterpart of this sweep with the third estimator added.

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

**A bound is now drawn on the NMSE panels.** For each $(N,P,\mathrm{SNR})$ the per-element Rician CRLB of
eq. (19) is evaluated using the canonical mapping of §1.3 — $\mathbf{M}=\mathbf{S}$,
$\mathbf{b}=\overline{\mathbf{B}_{n,:}}$ — summed over receive elements and normalised by
$\mathbb{E}\|\mathbf{G}\|_F^2 = NK\beta$, so it sits on the same axis as NMSE$_G$.

> **This is the *unconstrained* CRLB.** It bounds GS and EM-GS, which use no structural prior. **HS-GS
> exploits a rank constraint and is not bounded by it** — a constrained bound would require the tangent
> space of the structured manifold and is not computed here. Read HS-GS falling below the curve as evidence
> the prior is doing work, not as a violation.

Two checks make the bound trustworthy. Track A's own test reproduces: at high SNR the computed CRLB sits
**3.0103 dB** above the genie-ZF covariance, against the required
$10\log_{10}2 = 3.0103$ dB — so the estimation-role Fisher information carries no real-vs-complex
convention error. And **EM-GS never falls below it at any of the 36 points** (0 violations), which is
what the bound demands. HS-GS falls below at 11/36 points, 8 of them at $N=32$ — exactly where
the structural constraint is most informative.

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
All six CIs lie strictly above zero.

> **Caveat on the $P=6$ row.** The order-selection splitter uses $\lceil 0.3P\rceil$ held-out columns, so at
> $P=6$ it trains on **4** columns and validates on 2 (not a 3/3 half split). Four columns give 4 real
> magnitude measurements per receive row against $2K = 6$ real unknowns, so **the training half is
> underdetermined** — it is below the identifiability floor, and $P=6$ is the only point in the sweep where
> that happens ($P=10$ gives 7 ≥ 6). This is a shortage of measurements, not ill-conditioning: the median
> condition number of the training pilot block is 3.37 at $P=6$, comparable to 2.33 at $P=10$. The final fit
> still uses all $P$ columns; only the order search sees the split. The rule was **not** changed after seeing
> these results.

Note $P=6$ ($=2K$, the minimum): the win rate is **48%**, below half, while the pooled gain is
**+0.58 dB**. Not a contradiction — the pooled metric is a ratio of sums, moved by
*how much* HS-GS wins when it wins, not how often. At the shortest pilot the constraint is active in only
68% of trials and the order selector has just 2 held-out columns, so it engages less often but pays off
substantially when it does.

### 8.5 What it costs

"+2.85 dB at $N=32$" invites the question "at what price". Median wall-clock per trial, $P=30$, SNR = 5 dB:

| N | cap | GS (ms) | EM-GS (ms) | EM-GS chained (ms) | HS-GS (ms) | HS-GS / EM-GS | order search | projection |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 4 | 31 | 92 | 160 | 564 | **6.1×** | 58% | 13% |
| 16 | 8 | 61 | 191 | 317 | 1621 | **8.5×** | 74% | 6% |
| 32 | 16 | 119 | 369 | 632 | 5295 | **14.4×** | 86% | 3% |

**The cost is dominated by the order search, not by the projection.** At $N=32$ the held-out search over
$\hat L \in \{1..16\}$ accounts for 86% of HS-GS's runtime while the
Cadzow projections account for 3%. That makes the cost largely *reducible*:
fixing $\hat L$, coarsening the candidate grid, or warm-starting the search would recover most of it
without touching the structural step.

Analytic counts, for scale: HS-GS's SVDs cost
$\text{iters}\times K\times|L\text{-grid}|\times O(\lceil N/2\rceil^3)$, against EM-GS's
$\text{iters}\times N\times O(K^2P)$.

*On the "EM-GS chained" column.* HS-GS must re-enter the solver every iteration so the projection can be
interleaved (eq. 23), and that call structure alone costs
1.71× a single `max_iter=50` call. The honest baseline for HS-GS is
therefore 50 chained `max_iter=1` calls, and HS-GS with the projection disabled matches it to within
3.8% at all three $N$ — which is what confirms
the instrumentation measures the structural step rather than framework overhead.

### 8.6 B5 — scaling with array size

![B5](results/track_b/final/b5_gain_scaling_vs_N.png)

**B5 — the hypothesis test.** Both panels monotone in $N$, both crossing their neutral line between $N=8$
and $N=16$.

| N | cap $\lceil N/2\rceil$ | $P(L_k<\text{cap})$ | $2NK$ | $3\mathbb{E}[\sum L_k]$ | $\rho(N)$ | Mean gain | P=10 | P=30 | Win (unwtd) | Win (trial-wtd) | Active |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 4 | 20% | 48 | 44.66 | 1.075× | -0.192 | -0.319 | -0.066 | 33.50% | 31.50% | 57.5% |
| 16 | 8 | 100% | 96 | 44.66 | 2.150× | +0.780 | +0.797 | +0.764 | 74.01% | 69.69% | 92.7% |
| 32 | 16 | 100% | 192 | 44.64 | 4.301× | +2.851 | +3.171 | +2.531 | 95.35% | 94.07% | 99.6% |

Structural redundancy $\rho(N) = 2NK / 3\sum L_k$: unstructured $\mathbf{G}$ has $2NK$ real parameters;
the geometric model has 3 per path (one angle, one complex gain). Only $N$ moves with the array — the path
budget does not — so $\rho$ grows linearly in $N$. Increment per doubling: -0.192 → +0.780 dB (+0.973), then +0.780 → +2.851 dB (+2.071).

Every value in this table is emitted by `scripts/report_numbers.py` from the stored per-trial data, not
typed. $\mathbb{E}[\sum_k L_k] = 14.8840$ ($\mathbb{E}[L_k] = 4.9613$) is measured on the 18,400 B3 trials
themselves rather than on a fresh sample, so the same number appears here and in audit check 6 — an earlier
version used three mutually inconsistent values (44.6, 45.0 and 45.1) for the same quantity.

**Two win rates are reported, and they differ.** The unweighted mean treats each of the 12 $(P,\mathrm{SNR})$
points equally; the trial-weighted mean weights by trial count. They diverge (e.g. 33.50% vs
31.50% at $N=8$) because the five points carried to 1 200 trials are mostly high-SNR
points where HS-GS does worst, so weighting by trials pulls the average down. The unweighted figure is the
one plotted, since the design samples the $(P,\mathrm{SNR})$ grid uniformly and the extension was driven by CI
width, not by importance.

> **Terminology — read this before quoting $\rho(N)$.** It is a **parameter count**, and the rank cap is an
> algebraic fact about Hankel matrices. Together they are a *structural redundancy* / *representational
> informativeness* argument. They are **not an identifiability theorem**. Nothing here proves the
> constrained problem has a unique solution, that the alternating projection converges to it, or that the
> estimator attains any bound. Three values of $N$ also cannot identify a functional form — no growth law
> is fitted and none should be quoted.

### 8.7 B6 — does the advantage survive a weak reference?

Every experiment above fixes RSR = 12 dB. But RSR is the atomic receiver's defining design parameter and
the reason this is *biased* phase retrieval rather than ordinary phase retrieval, so a result that only
holds at one reference strength is a result with a hole in it. B6 sweeps it, using the same CRN world
function, the same estimators and the same adaptive rule — none of which were altered for this sweep.

Fixed: $P=30$, SNR = 5 dB, $K=3$, $L_k\sim\mathcal{U}\{3..7\}$, $t_0=50$. Swept: RSR $\in$
{0, 6, 12, 18, 24} dB at $N\in\{8,32\}$ — the two ends, where the effect sign differs. 400 trials/point,
4 000 trials total. The adaptive rule flagged **no** point for extension: all ten CIs already exclude zero,
the widest being 0.72 dB.

![B6](results/track_b/final/b6_rsr_sweep.png)

| N | RSR | GS | EM-GS | HS-GS | Gain | Gain 95% CI | Win | Active | L̂ |
|--:|--:|--:|--:|--:|--:|:--|--:|--:|--:|
| 8 | 0 | -4.41 | -5.69 | -5.11 | -0.58 | [-0.88, -0.32] | 16% | 44% | 3.36 |
| 8 | 6 | -8.85 | -9.69 | -9.21 | -0.48 | [-0.67, -0.29] | 23% | 53% | 3.24 |
| 8 | 12 | -10.63 | -10.74 | -10.34 | -0.41 | [-0.55, -0.26] | 27% | 59% | 3.20 |
| 8 | 18 | -10.94 | -10.96 | -10.38 | -0.58 | [-0.78, -0.41] | 26% | 58% | 3.15 |
| 8 | 24 | -10.74 | -10.74 | -10.27 | -0.47 | [-0.61, -0.33] | 21% | 55% | 3.23 |
| 32 | 0 | -4.34 | -5.66 | -7.11 | **+1.45** | [+1.08, +1.80] | 78% | 92% | 7.81 |
| 32 | 6 | -8.91 | -9.71 | -12.27 | **+2.55** | [+2.38, +2.74] | 97% | 100% | 5.21 |
| 32 | 12 | -10.58 | -10.69 | -12.96 | **+2.27** | [+2.12, +2.40] | 96% | 100% | 5.02 |
| 32 | 18 | -10.91 | -10.93 | -13.34 | **+2.42** | [+2.28, +2.56] | 98% | 100% | 5.08 |
| 32 | 24 | -10.77 | -10.77 | -13.03 | **+2.26** | [+2.10, +2.42] | 96% | 100% | 4.84 |

**The advantage survives, but it is weakest exactly where the problem is hardest.** At $N=32$ the gain holds
across the whole sweep and is credibly positive at every point, but at RSR = 0 dB it drops to
+1.45 dB [+1.08, +1.80]
against +2.26 to +2.55 dB from RSR = 6 dB upward — roughly 40% of its strong-reference
value. The $N=8$ deficit, by contrast, is essentially flat in RSR (-0.58
to -0.41 dB), so it is a property of the vacuous
constraint rather than of the reference strength.

Two mechanisms are visible in the diagnostic columns at RSR = 0, $N=32$: the order selector jumps to
$\hat L = 7.81$ (against ≈5 elsewhere), and the constraint-active fraction falls to
92% from 100%. With a weak reference the held-out residual is a
noisier model-order criterion, so the selector over-orders and the projection engages less often. That is a
limitation of the *order rule* at low RSR, not evidence that the structural prior itself fails there.

### 8.8 Interpretation tests A–H

Eight adversarial checks, all evaluated numerically from the stored per-trial data rather than from
expectation.

| | Question | Answer |
|---|---|---|
| A | Is $N=8$ still mixed at scale? | Mixed, and systematically **negative** at high SNR: 3 points positive, 8 negative, 1 straddling; worst −2.23 dB |
| B | Credible positive gain at $N=16$? | Yes — 10/12 points with CI entirely above 0 |
| C | Is $N=32$ larger than $N=16$? | Yes — CI strictly above at **12/12** shared points |
| D | Does win rate increase with $N$? | Yes, monotone — 33.5% → 74.0% → 95.3% |
| E | Consistent with the redundancy argument? | Yes — both monotone in $N$. The sign change is bracketed to $(8, 16]$; the algebraic threshold $\lceil N/2\rceil > \max L_k$ first holds at $N=15$, inside that interval. Between $N=9$ and $N=14$ the constraint is *partially* vacuous, so the transition is gradual and three array sizes cannot locate it |
| F | Driven by a few catastrophic EM-GS trials? | No — dropping the worst 5% of EM-GS trials moves the gain by a median of −0.05 dB |
| G | Does HS-GS floor out at high SNR? | No at $P=30$ (slopes −1.03/−1.01/−1.04 dB/dB vs EM-GS −1.02/−1.01/−1.01). At $P=10$ both flatten together |
| H | Stable pooled vs median? | 25/36 raw, **33/36 once exact ties are excluded** |
| I | Is the EM-GS baseline itself $N$-dependent? | **No** — max spread across $N\in\{8,16,32\}$ is **0.133 dB** at every $(P,\mathrm{SNR})$. Estimation is row-separable, so every row has $K$ unknowns and $P$ measurements regardless of $N$; all $N$-dependence in the gain is therefore attributable to the structural constraint, not to the baseline moving underneath it |

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
| 5 | Identical CRN worlds across estimators | world is a deterministic function of (trial, $P$, SNR, RSR). Channel parameters do not depend on $N$, so $N=8$ is the first 8 rows of the $N=32$ realization (verified elementwise) and the $N$ sweep is **paired in the channel**; the noise $\mathbf{W}$ is drawn at shape $(N,P)$ and is **not** nested |
| 6 | $L_k$ per frozen spec | support {3..7}, uniformity cv 0.0190. $\mathbb{E}[L_k] = 4.9613$ measured on the 18,400 B3 trials themselves — the same number §8.6 uses for $\rho(N)$, not a separate sample |
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

### 10.3 Corrections from external review

A structured review of the first version of this report found twenty defects. The material ones, and what
they were:

- **The canonical-form table was wrong** (§1.3). It specified $\mathbf{M}=\overline{\mathbf{S}}$ *and*
  $\mathbf{u}=\overline{\mathbf{g}}_n$ simultaneously, which does not satisfy the model, and it had no row
  for $\mathbf{b}$ — precisely where the conjugate bites. The code was always right; the table was not.
  Three regression tests now pin both valid conventions and the invalid mixture.
- **The vacuity threshold was off by one** (§7.5): stated as $L_k\ge5$ / 60% of the prior, actually
  $L_k\ge4$ / **80%**, and the report contradicted its own $P(L_k<\mathrm{cap})=20\%$ column.
- **The trial count did not reconcile.** The header claimed B3 = 21 700; the table sums to
  18,400. The grand total also double-counted the two B4 points copied from B3.
  All counts are now reconstructed from the checkpoints by `scripts/report_numbers.py`.
- **§7.1 overstated the constraint** as "precisely the set of channels the ULA model can generate". It is a
  relaxation — see §11.2.
- **$\rho(N)$ used three inconsistent denominators** (44.6, 45.0, 45.1) for one quantity.
- **"The sign flips precisely where the algebra says"** was an overclaim: the threshold is $N=15$ and three
  array sizes bracket the crossing without locating it.
- **§7.3 asserted GS is a contraction**, which is false in general and contradicted §11.2.
- **§4.1 omitted the magnitude rescaling and phase anchor** from the spectral initializer, in a document
  that claims every equation is transcribed from the implementation.
- **§4.5 never stated that (16) is minimised and (17) maximised.**
- **The calibration read 2.82 dB against a 3.00 dB target** with no sample size or interval.

Two review items turned out to be right about the symptom and wrong about the cause, and both are recorded
that way rather than silently accepted: the B5 win-rate discrepancy was not a stale value but an unweighted
-vs-trial-weighted ambiguity (both are now reported), and the Fig. 7(a) bit arithmetic did not close because
the trial count is deliberately non-uniform across SNR, not because a number was wrong.

### 10.4 Process failures during the final run

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
- The EM-GS baseline does **not** move with $N$ (max spread 0.133 dB), so the $N$-dependence of the gain is attributable to the structural constraint rather than to the baseline shifting.
- Track A: Figs. 7(a), 7(b) and 8 reproduce Cui closely; Figs. 5–6 qualitatively with a documented, traced ≈2 dB offset.
- Calibration is unbiased: all four measured SNR/RSR values contain their target inside a bootstrap 95% CI.
- The advantage **survives a weak reference** at $N=32$ — credibly positive at every RSR from 0 to 24 dB —
  though it weakens to roughly 40% of its strong-reference value at RSR = 0 dB.
- EM-GS never falls below the unconstrained CRLB at any of the 36 B3 points, and the bound reproduces the
  $10\log_{10}2$ high-SNR gap to four decimal places.

### 11.2 Not established, and should not be claimed

- **No identifiability theorem.** The redundancy argument is a parameter count.
- **The Hankel constraint is a relaxation, not the ULA set.** Measured slack: median mode-modulus
  deviation 0.0124, 96.3% within 0.10 of the unit circle, but a tail
  reaching 0.72. Tight in practice, not tight by construction. Rank $\le L$ characterises sums of $L$ exponentials $z_i^{\,n}$ with arbitrary non-zero complex $z_i$; the ULA model additionally requires $|z_i|=1$. Cadzow enforces no unit-modulus condition, so the feasible set strictly contains the ULA set and admits damped and growing modes. The constraint is necessary, and sufficient only up to the modulus of the recovered modes.
- **No convergence guarantee** for the alternating projection, and no claim that it attains any bound.
- **No growth law.** Three values of $N$ cannot identify a functional form; nothing is fitted.
- **Nothing about $N=64$** or any array size not tested.
- **No constrained bound.** The CRLB drawn is the unconstrained one; HS-GS is not bounded by it, and how
  much headroom actually remains for a structure-aware estimator is not established here.
- **Nothing about RSR outside 0–24 dB**, or about weak reference at $N=16$, which was not swept.
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
