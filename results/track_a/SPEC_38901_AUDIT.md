# Track-A channel audit against the actual TR 38.901 specification

Source supplied by the user: Zhu, Wang, Hua, Mao, Jiang, Yao, *"3GPP TR 38.901
Channel Model"* (book chapter, V6, 2020-08-20). This is the first audit round
with the **actual generation procedure** in hand rather than only Cui's Table I.

Question under test: does faithfully implementing TR 38.901 explain the
~2 dB constant offset between our Fig. 5 / Fig. 6 curves and Cui's published
curves (established earlier by pixel extraction, ≈1.57× in channel
conditioning)?

**Answer: no.** Verdict C from the previous audit stands, now confirmed
against the specification rather than inferred from Table I.

---

## 1. Two genuine deviations found in our implementation

### D1 — polarization drawn per antenna element instead of per ray (CONFIRMED BUG)

Spec eq. (10), the NLOS channel coefficient:

```
H_{u,s,n,m}(t) = sqrt(P_n/M) [F_rx,u,θ ; F_rx,u,φ]^T
                 [ e^{jΦ^θθ_{n,m}}            sqrt(κ^-1_{n,m}) e^{jΦ^θφ_{n,m}} ]
                 [ sqrt(κ^-1_{n,m}) e^{jΦ^φθ_{n,m}}   e^{jΦ^φφ_{n,m}}          ]
                 [F_tx,s,θ ; F_tx,s,φ]
                 e^{j2π (r̂_rx,n,m · d̄_rx,u)/λ0} · ...
```

The random polarization phases `Φ^{θθ,θφ,φθ,φφ}` carry subscript **`(n,m)`
— cluster and ray only**. The receive-element index `u` enters *only* through
the element field pattern `F_rx,u` (identical for identical elements) and the
array steering phase `e^{j2π r̂·d̄_u/λ0}`.

`channel_cui.py::_polarization_couplings` draws
`psi = rng.uniform(0, 2π, size=N)` — an **independent polarization per antenna
element**, inside the per-ray loop. This multiplies the steering vector
elementwise by an i.i.d. random vector, whitening the array response. It is why
the previous audit found "essentially i.i.d.-Rayleigh" statistics.

Physically: for a plane wave, polarization is a property of the arriving path,
constant across the aperture. Per-element draws are unphysical.

**Measured effect of fixing it: +0.12 dB** (Tr((AA^H)^-1) 0.08850 → 0.09103).
Real correctness fix; immaterial to results.

### D2 — no cluster power structure (CONFIRMED OMISSION)

Spec eqs. (12), (13), (15), (16) build an exponential power-delay profile with
per-cluster shadowing:

```
τ'_n = -r_τ DS ln(X_n),  X_n ~ U(0,1)            (12)
τ_n  = sort(τ'_n - min τ'_n)                      (13)
P'_n = exp(-τ_n (r_τ-1)/(r_τ DS)) · 10^(-Z_n/10),  Z_n ~ N(0, ζ²)   (15)
P_n  = P'_n / Σ P'_n                              (16)
```

and eq. (18) then **derives cluster angles from those powers**:

```
φ'_{n,AOA} = 2 (ASA/1.4) sqrt(-ln(P_n / max P_n)) / C_φ     (18)
φ_{n,m,AOA} = φ_{n,AOA} + c_ASA α_m                          (19)
```

Our generator uses equal-power CN(0,1) rays and cluster angles drawn
independently `U(-90°, 90°)`.

Spec parameters used (UMa NLOS, no fitting): `r_τ = 2.3`, `ζ = 3 dB`
(Table VI); `C_φ = 1.289`, `c_ASA = 15°`, `μ_lgASA = 2.08 − 0.27 log10(fc)`,
`σ_lgASA = 0.11` (Table VIII); ray offsets `α_m` from Table VII. At Cui's
5 GHz carrier this gives **ASA ≈ 77.9°**.

**Measured effect: +0.02 dB** (PDP alone). See §2 for the combined case.

---

## 2. The apparent +1.57 dB match was an artifact of angle clipping

Implementing the full chain initially appeared to close most of the gap:

```
current model                          Tr = 0.09104
full 38.901 chain (ASA ~ lognormal)    Tr = 0.13078   → +1.57 dB
                     (target from Cui pixel extraction: +1.96 dB)
```

That result **does not survive scrutiny**. With ASA ≈ 78°, eq. (18) places
weak clusters far outside ±90° (a cluster 20 dB below the peak lands at ≈185°).
Clipping those to the ULA field of view piles them onto `sin θ = ±1`,
manufacturing artificial coherence at two directions.

TR 38.901 azimuth is a full 360° circle, so wrapping — not clipping — is
correct. Under wrapping the effect vanishes entirely:

| angle handling | Tr((AA^H)^-1) | shift vs current |
|---|---|---|
| clip at ±90° | 0.12728 ± 0.00211 | **+1.47 dB** |
| wrap (correct) | 0.09094 ± 0.00030 | **+0.01 dB** |
| reject outside FoV | degenerate (all rays lost for some users) | — |

Mechanism isolation (400 trials each, wrapping throughout):

| variant | Tr | shift |
|---|---|---|
| baseline (current model) | 0.09066 ± 0.00025 | — |
| PDP only, uniform angles | 0.09110 ± 0.00031 | +0.02 dB |
| derived angles only, flat PDP | 0.09666 ± 0.00152 | +0.28 dB |
| both (full 38.901) | 0.12728 ± 0.00211 | +1.47 dB **(clip artifact)** |

Seed stability of the clipped variant (+1.47/+1.50/+1.44 dB across three seeds)
confirms it is a reproducible property of clipping, not noise.

i.i.d. complex-Gaussian reference: `K/(N-K) = 3/33 = 0.09091`. Both the current
model (0.09066) and the correctly-wrapped spec model (0.09094) sit on it.

---

## 3. End-to-end confirmation with the real solvers

300 trials, common random numbers, production `biased_gs` / `em_gs` /
`zf_known_phase`, ratio-of-sums NMSE. Negative delta = spec version *better*.

| SNR | RSR | algorithm | current | spec-38901 | delta |
|---|---|---|---|---|---|
| −2 | 12 | biased_gs | −3.272 | −3.311 | −0.039 |
| −2 | 12 | em_gs | −3.718 | −3.848 | −0.130 |
| −2 | 12 | genie_zf | −8.310 | −8.520 | −0.211 |
| 3 | 12 | biased_gs | −9.008 | −9.446 | −0.439 |
| 3 | 12 | em_gs | −9.162 | −9.599 | −0.437 |
| 3 | 12 | genie_zf | −13.506 | −13.587 | −0.081 |
| 8 | 12 | biased_gs | −14.702 | −14.853 | −0.151 |
| 8 | 12 | em_gs | −14.745 | −14.892 | −0.147 |
| 8 | 12 | genie_zf | −18.284 | −18.598 | −0.314 |
| 3 | 0 | biased_gs | −4.178 | −3.108 | +1.069 |
| 3 | 0 | em_gs | −5.962 | −4.835 | +1.127 |
| 3 | 0 | genie_zf | −13.555 | −13.639 | −0.085 |
| 3 | 25 | biased_gs | −9.863 | −10.006 | −0.143 |
| 3 | 25 | em_gs | −9.863 | −10.009 | −0.146 |
| 3 | 25 | genie_zf | −13.685 | −13.661 | +0.024 |

Every delta except the RSR = 0 corner is inside the 0.5 dB stop-gate, and the
**sign is predominantly negative** — the spec-faithful channel is marginally
*better*, i.e. it moves *away* from Cui's curves, not toward them. The
RSR = 0 exception (+1.1 dB, GS/EM-GS only) is a low-RSR regime where GS is
weakest; it does not produce the roughly constant offset seen in Cui.

---

## 4. Verdict

**Verdict C stands, now spec-confirmed.** Faithfully implementing TR 38.901 —
per-ray polarization, exponential PDP with cluster shadowing, power-derived
cluster angles, Table VII ray offsets, all parameters taken from the spec's own
tables at Cui's carrier frequency with nothing fitted — still yields
essentially i.i.d.-Rayleigh array statistics for this array/parameter
combination, and does not reproduce Cui's ~2 dB offset.

Per the standing stop-gate ("if it produces essentially the same i.i.d.-Rayleigh
channel and <0.5 dB change, STOP and report before wasting hours"), Fig. 5 and
Fig. 6 were **not** regenerated: the curves would be visually indistinguishable
from the delivered ones.

### Recommendation

D1 (per-ray polarization) is a genuine fidelity fix worth applying on its own
merits, as a new channel variant leaving `cui_38901` and all existing stores
intact. It is not worth a full figure regeneration by itself (+0.12 dB).

### Reproduce

```
PYTHONPATH=. python3 scripts/audit_38901/diag_pol.py        # D1 isolation
PYTHONPATH=. python3 scripts/audit_38901/diag_full38901.py  # full chain + ASA sweep
PYTHONPATH=. python3 scripts/audit_38901/diag_robust.py     # clip-vs-wrap, mechanisms
PYTHONPATH=. python3 scripts/audit_38901/diag_e2e.py        # end-to-end solver NMSE
```
