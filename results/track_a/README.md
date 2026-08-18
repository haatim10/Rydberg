# Track A — Cui validation notes

Source used: Cui, Zeng, Huang, *Towards Atomic MIMO Receivers*, arXiv:2404.04864
(IEEE JSAC 2025). **No PDF is stored in this repository.** Equations and Table I
were read from the public arXiv HTML/PDF.

Track A **does not** use `rydberg_sim.channel.generate_ula_channel`.
`channel_model` is `cui_38901` (Track B is `ula_geometric`).

## Implemented from the paper

| Item | Reference | Implementation |
|---|---|---|
| Observation | eq. (22) | `z = \|A^H s + b + w\|`, `A ∈ C^{K×N}` |
| Path / Rabi coupling | eq. (10), (15), (16) | clustered sum with per-element `μ·ε` |
| Polarization | §VI-A | `ε_{nkl}`, `ε_{b,n}` on the unit circle ⊥ incident direction |
| Table I clusters / rays | Table I | 23 clusters × 20 rays |
| Table I path gains | Table I | i.i.d. `CN(0,1)` |
| Table I AoA | Table I | cluster mean `U(-90°, 90°)` |
| Table I cluster AS | Table I | ray offset `U(-5°, 5°)` |
| Table I delay spread | Table I | per-user max `U(0, 30 ns)`; cluster delay `U(0, max)` |
| Carrier / dipole direction | §VI-A | 5 GHz; `μ_eg ∝ ŷ` |
| SNR | eq. (37) | `σ² = K / SNR_lin` after row-normalizing `A` |
| RSR | eq. (38) | scale `b` so `mean\|b_n\|² = RSR_lin` |
| Iterations | §VI-A | `t0 = 50` |
| Fig. 5 defaults | §VI-B | `N×K=36×3`, 16-QAM, RSR = 12 dB |
| Fig. 4 | §IV-D | existing `bessel_ratio` (not reimplemented) |
| Solvers / CRLB | Alg. 1–2, §VI | existing `biased_gs`, `em_gs`, `cui_crlb`, `zf_known_phase` |

CM-ZF is still **not implemented** (as in Step 7).

## Documented deviations (source did not uniquely specify)

These are **not** silent guesses. They are the closest model supported by Table I
plus the array/LO facts implied by an angle-dependent 38.901-style generator.

1. **Full 3GPP TR 38.901 CDL is not reconstructed.** Cui cites 38.901 but then
   replaces the 38.901 cluster-power / Laplacian-AoA / exponential-delay
   profiles with Table I (`CN(0,1)` gains, uniform angles). We implement
   Table I, not an unofficial CDL-A/B/C drop-in.

2. **Array geometry is not in Table I.** Assumption A1: half-wavelength ULA
   along `x`, `n = 0…N-1`, phase `exp(-j n π sin θ)`. Implemented locally in
   `channel_cui.py`; **not** by calling Track-B `steering_vector`.

3. **Coordinate frame for “incident angle” is not given.** Assumption: azimuth
   in the x–z plane, θ = 0 broadside.

4. **Intra-cluster delays are not given.** Rays in a cluster share the cluster
   delay. Delay mainly randomizes carrier phase at 5 GHz (30 ns ≫ 1/f).

5. **Ray AoAs that fall outside ±90° are clipped** to the Table I support.

6. **LO azimuth is not in Table I.** Assumption A2: default 0° broadside.
   Polarization of `b` is still random per element as specified.

7. **Physical `|μ_eg|/ℏ · ρ √P` is not used as an absolute SI scale.**
   Table I already sets path gains to `CN(0,1)`. `μ_eg` contributes only
   its **direction** through `μ·ε`. Each user-row of `A` is then scaled so
   `mean_n |a_{nk}|² = 1`, which makes eq. (37)–(38) well-defined.

8. **Per-element polarization** is drawn independently across antennas for the
   same path (paper samples `ε_{nkl}`). Path gain `α_{kℓ}` is shared.

This smoke study does **not** claim numerical overlay on the published Fig. 5
curves. It checks orientation, CRN, calibration, and qualitative SNR trends.
