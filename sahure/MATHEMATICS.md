# Exact-Rician adaptive Hankel estimation

## Observation and likelihood

Use the unchanged Paper 1 generator: \(Z=|GS+B+W|\), with independent
\(W_{np}\sim\mathcal{CN}(0,v)\). Here \(v=\sigma^2\) is total complex
variance, so each real component has variance \(v/2\). Aggregate SNR and
single-user RSR retain the repository calibration. For equal average user
powers, reference-to-total-average-signal ratio is RSR minus \(10\log_{10}K\).
Realized reference-to-signal ratios are recorded separately.

For \(a=|\mu|\), the amplitude density is

\[
p(z|a,v)=\frac{2z}{v}\exp[-(z^2+a^2)/v]I_0(2za/v).
\]

The implemented per-measurement objective is

\[
\ell(z,\mu,v)=\frac{(z-|\mu|)^2}{v}-\log I_{0e}(2z|\mu|/v).
\]

It includes the observation-only \(z^2/v\) term and omits
\(-\log(2z/v)\). Adding that latter term gives the full negative log density.
The omitted term is identical for models evaluated on the same measurements.
Scaled Bessel functions prevent exponential overflow.

## Gradient convention and zero fields

With \(M=GS+B\), \(\kappa=2Z|M|/v\), and
\(Q=Z[I_1(\kappa)/I_0(\kappa)]M/|M|\),

\[
D=\frac{\partial\mathcal L}{\partial G^*}=v^{-1}(M-Q)S^H.
\]

For a real loss, PyTorch's complex gradient is
\(\partial\mathcal L/\partial\Re G+j\partial\mathcal L/\partial\Im G=2D\).
Directional finite differences check \(\Re\langle 2D,\delta G\rangle\).

At \(M=0\), evaluate the continuous limit, rather than adding an amplitude
offset. Writing \(u=|M|^2\), \(x^2=4z^2u/v^2\), the small-field formulas are

\[
\log I_0(x)=x^2/4-x^4/64+x^6/576+O(x^8),
\quad Q=M\frac{z^2}{v}(1-x^2/8+x^4/48+O(x^6)).
\]

These series are used when \(x^2<10^{-4}\), solely to evaluate the exact
Rician functions stably. They are not a strong-reference approximation of
the observation model. The inactive large-field series branch is bounded
to avoid overflowing intermediate powers. A \(10^{-30}\) squared-amplitude
guard is used only inside the inactive regular branch at zero.

## Two data updates

For \(\gamma_t=2\operatorname{sigmoid}(a_t)\), initialized to one:

\[
G_{dc}=G-\frac{\gamma_t}{\|S\|_2^2}(M-Q)S^H
\]

is the gradient update with
\(\eta_t=\gamma_t v/(2\|S\|_2^2)\) and update \(G-2\eta_tD\).

The other core is

\[
G_{EM}=(Q-B)S^H(SS^H)^{-1},\qquad
G_{dc}=G+\gamma_t(G_{EM}-G).
\]

Right multiplication by the inverse is evaluated by a linear solve, without
forming the inverse. When \(SS^H\) is nonsingular,

\[
G_{EM}=G-vD(SS^H)^{-1}.
\]

The two cores change update geometry, not the likelihood. Relaxed updates
with \(\gamma_t\ne1\) and structural corrections are not claimed to be
likelihood-monotone. A context set must have enough independent pilots;
the split function rejects fewer than \(2K\) context columns.

## Spatial operator and features

Each user's \(N\)-element vector is lifted to a balanced Hankel matrix with
\(p=\lfloor(N-1)/2\rfloor\), dimensions \((N-p)\times(p+1)\), and capacity
\(q=\lceil N/2\rceil\). Antidiagonal averaging is the inverse on exact
Hankel matrices. Let \(s_H=\|H\|_F/\sqrt q\) and
\(\tau=s_H[0.001+\operatorname{softplus}(u)]\). The initial relative threshold
is 0.1; the 0.001 floor makes the structural solve nonsingular.

\[
\widehat\sigma_i=\frac{\sigma_i^3}{\sigma_i^2+\tau^2},\qquad
\widehat H=H-\tau^2H(H^HH+\tau^2I)^{-1}.
\]

The right-hand form is equivalent to
\(H-\tau^2(HH^H+\tau^2I)^{-1}H\) and uses the smaller Gram matrix for this
balanced lift. The code normalizes by \(s_H\) before solving. The operation
does not impose an exact rank, and it is not presented as the proximal map
of an established objective. No straight-through or identity-backward
approximation is used. Singular values, but not singular vectors, supply
the adaptive features.

The eight features, for each user and stage, are:

1. Entropy effective rank of the current Hankel singular values divided by \(q\).
2. Largest singular value divided by their sum.
3. Fraction of that sum in the last \(\lceil q/2\rceil\) singular values.
4. \(\log[1+\operatorname{mean}(Z_A-|G_{dc}S_A+B_A|)^2/v]\), pooled over the observed array and pilots.
5. \(\log[1+v\operatorname{tr}((S_AS_A^H)^{-1})/(K\widehat p_k)]\), with \(\widehat p_k=N^{-1}\|g_{dc,k}\|^2\).
6. \(P_A/(2K)\).
7. \(\log q\).
8. \((t+1)/T\).

Small denominator guards prevent division by zero in features. No true
channel, path count, or true effective rank is an input. An 8–16–2 SiLU
network shared across users and stages produces relative threshold and
gate \(\beta=\operatorname{sigmoid}(w)\). It has 178 parameters; the ten
update scalars bring the adaptive model to 188. The global version uses
two learned values per stage (30 total parameters); the unstructured
version has only the ten update scalars.

\[
g^{t+1}_k=g_{dc,k}+\beta_{t,k}(\mathcal H^\dagger\widehat H_k-g_{dc,k}).
\]

The initial gate is 0.1. Threshold and gate can compensate for one another.
Neither output should be called an estimated rank. Save the actual relative
correction \(\|g^{t+1}_k-g_{dc,k}\|/(\|g_{dc,k}\|+\epsilon)\). A zero gate
is exactly the unstructured update; a sigmoid gate does not guarantee that
every correction helps.

## Self-supervision and scope

Each training realization is split by pilot column, 70% context and 30%
held out. The spectral initializer is recomputed from context only. Loss is
mean held-out Rician NLL, and splits are resampled each epoch. Validation
uses an independent realization set and a fixed split. The lowest
validation-NLL checkpoint is frozen before generating the test set.
Ground-truth channels are stored separately and are never loaded by
training or checkpoint selection. Deployment uses all available pilots.

This objective predicts held-out observations; it does not guarantee minimum
channel NMSE. All reported channel errors require independent empirical
evaluation. The pilot is one SNR, one RSR, and one training seed, and cannot
establish robustness, complexity transfer or equivalence to supervision.
