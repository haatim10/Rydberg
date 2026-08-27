# Track D (URformer) — Repository Audit

**Status:** audit only. No package created, no estimator written, no repository file modified,
no dependency declared. Three new files only (`reports/trackD_audit.md`,
`reports/trackD_audit.json`, `scratch/trackD_audit_probe.py`).

**Target (next phase, not this one):** implement the URformer estimator of Xiao, Wang, Zeng, Xu,
Li and Nallanathan, *"Channel Estimation for Rydberg Atomic Quantum Receivers: Unrolled Phase
Retrieval From Holographic Snapshots," IEEE Signal Processing Letters*, vol. 33, pp. 1696–1700,
2026 — on **our existing geometric ULA model**, as an isolated Track D package, so that the only
major change versus the current pipeline is the estimator.

Audited tree: branch `claude/new-paper-implementation-fs4ako` @ `8e3792d` (identical to `main`).

---

## 0. Two findings that change the plan

Read these before anything else.

### 0.1 The Hankel / Track-B code is **not in this checkout**

`origin/main` and this branch contain **Track A only**. Every Track-B artifact — the geometric ULA
drivers, HS-GS, the Hankel projection, the 24,000-trial stores, the manuscript — lives on
`origin/claude/adversarial-audit-gs6iid` @ `54c88d5`, which is 22 files and ~3,750 lines ahead.

This matters because the audit was asked to inventory Track-B Hankel code so Track D does not touch
it, and because Track D is eventually to be merged with the Hankel plan.

**What makes the audit still valid:** the modules that define every convention Track D depends on are
**byte-identical** between the two branches. `git diff HEAD origin/claude/adversarial-audit-gs6iid --
rydberg_sim/` touches only `__init__.py`, `channel_cui.py`, `monte_carlo.py`, `spectral.py`, and the
Track-A/Track-B drivers. It does **not** touch `channel.py`, `gs.py`, `forward.py`, `pilots.py`,
`reference.py`, `metrics.py`, `calibration.py`, `baselines.py`, `config.py` or `rng.py`. So every
probe result below holds on both branches.

Track-B modules were inventoried **read-only** via `git show`, without checking the branch out.

**Decision needed (D1): which branch is Track D's base?** See §7.

### 0.2 The prompt's Task-4 item 2 hypothesis is wrong

The prompt asserts: *"Paper uses a single LO vector broadcast across pilots (`B = b1ᵀ`, rank 1).
Ours is time-varying across `p`. Keep ours."*

**Our `B` is rank 1.** `reference.py:192` builds it as `B = np.outer(c * alpha_b * a_b, s_b)`, and
the baseline `s_b[p] = 1` (`reference.py:123`) makes it exactly `b 1ᵀ`. Measured
`numpy.linalg.matrix_rank(B) = 1`; the column ratio `B[:,1]/B[:,0]` has spread `0.000e+00`.

So on this point we already **match** the paper; there is nothing to keep or diverge from. The
repository supports a caller-supplied `s_b` for a later ablation, but that is not the baseline.
This is recorded as a corrected discrepancy, not a defect.

---

## 1. Inventory

Every row states what Track D **reuses** versus what it must **add**. Nothing listed as reuse may be
reimplemented in the next phase.

| # | Item | Path · symbol | Signature / shapes | Status |
|---|---|---|---|---|
| 1 | Trial generator | `monte_carlo.py:506` `generate_channel_estimation_trial(spec, trial_index, snr_db, rsr_db)` | → `ChannelEstimationTrial` carrying `G,H,S,B,W,Z,Y,Psi,E,sigma2,alpha_b,theta,psi,alpha,A_k,L_k,beta_k` | **reuse with wrapper** — requires `spec.track == "B"`; Track D needs its own spec or a thin adapter |
| 2 | Geometric ULA channel | `channel.py:251` `generate_ula_channel(cfg, trial_index, *, rng)` | → `ChannelRealization`; `G,H` `(N,K)` c128 | **reuse as-is** |
| 2a | ↳ `L_k` | `config.py:92` fixed `tuple[int,...]`, **not** redrawn per realization | — | see §7 D3 |
| 2b | ↳ `θ_{ℓ,k}` | `channel.py:233` `rng.uniform(-π/2, π/2, size=L)`, rejection-sampled against `PSI_SEP_MIN=1e-10` and full-column-rank `A_k` (≤64 draws) | `(L_k,)` f64 | **reuse as-is** |
| 2c | ↳ `α_{ℓ,k}` | `channel.py:239` `CN(0, β_k/L_k)`, Re/Im iid `N(0, var/2)` | `(L_k,)` c128 | **reuse as-is** |
| 2d | ↳ normalization | none applied post hoc; `E|h_{n,k}|² = β_k` holds by construction of `α` | — | **reuse as-is** |
| 3 | Steering | `channel.py:149` `steering_matrix(theta, N)` → `(N,L)`; `:159` `steering_vector(theta,N)` → `(N,)`; `:144` `spatial_frequency(θ)=π sin θ` | `‖a‖²=N` | **reuse as-is** |
| 4 | Pilots | `pilots.py:117` `generate_gaussian_pilots(*, K, P, master_seed=None, trial_index=None, rng=None)` → `PilotMatrix.S` `(K,P)` c128 | `CN(0,1)`, **complex**, rejection-sampled to full row rank (`≤64` draws, `rel_tol=1e-8`) | **reuse as-is** |
| 5 | Reference `B` | `reference.py:140` `generate_reference_field(*, N, P, alpha_b, vartheta, c=1.0, s_b=None)` → `ReferenceField.B` `(N,P)` c128 | `B = outer(c·α_b·a_b, s_b)`, **rank 1**; deterministic, consumes no RNG | **reuse as-is** |
| 6 | Noise | `forward.py:157` `exact_forward(G,S,B,sigma2,rng_noise=...)` → `ExactObservation` | `W ~ CN(0,σ²)` `(N,P)`, injected **inside** the magnitude; `σ²=0` ⇒ `W=0` and no RNG draw | **reuse as-is** |
| 7 | **SNR** | `calibration.py:108` `snr_db_to_sigma2(snr_db, beta_k, c=1.0)` | **as implemented:** `sigma2 = c² · sum(beta_k) / 10**(snr_db/10)` — *total* over all K users | **reuse as-is** |
| 8 | **RSR** | `calibration.py:127` `rsr_db_to_alpha_magnitude(rsr_db, beta_ref, *, e_s_b_sq=1.0)` | **as implemented:** `|α_b| = sqrt(RSR_lin · beta_ref / E[\|s_b\|²])` — **single-user** denominator; not `sqrt(K·RSR)`, not `sqrt(RSR/K)` | **reuse as-is** — but see §6.1, the paper's RSR differs by a factor K |
| 9 | GS | `gs.py:263` `biased_gs(M,z,b,*,max_iter,u0=None,ridge=0.0,store_iterates=False)`; adapter `:348` `biased_gs_channel_rows(S,Z,B,*,max_iter,ridge,G0=None)` | canonical `z=\|Mᴴu+b+w\|`, `M (D,Q)`, `z (Q,)`, `b (Q,)` | **reuse as-is** |
| 10 | EM-GS | `gs.py:521` `em_gs(M,z,b,sigma2,*,max_iter,u0,ridge,...)`; adapter `:618` `em_gs_channel_rows(S,Z,B,sigma2,*,max_iter,ridge,G0=None)` | `σ² > 0` enforced (`gs.py:468`) | **reuse as-is** |
| 10a | ↳ **`R(κ)`** | `gs.py:426` `bessel_ratio(x)` | `R = I₁/I₀` via **`scipy.special.ive`** (exponentially **scaled**, shared `e^{-\|x\|}` cancels); asymptotic `1 − 1/(2x) − 1/(8x²)` for `x > 1e4`; `R(0)=0` exact; `np.clip(·,0,1)` | **reuse as-is** — see §4 for one cosmetic caveat |
| 10b | ↳ `κ` | `gs.py:476` `em_kappa(z,lam,sigma2)` = `(2/σ²)·z⊙\|λ\|` | factor 2, `sigma2` not `sigma` | **reuse as-is** |
| 11 | LS / M-step | `gs.py:326-331` | `lam = Mᴴu + b`; `y = z·exp(j∠lam)`; `r = y − b`; `rhs = M @ r`; `u = solve(M @ Mᴴ + ridge·I, rhs)`. **No explicit pinv.** Channel adapter (`gs.py:397-407`): `M = S`, `b = conj(B[n])`, `z = Z[n]`, output `G_hat[n] = conj(u_hat)` | **reuse as-is** |
| 12 | Spectral init | `spectral.py:278` `spectral_initialize(M,z,b)` → `SpectralInitResult.u0`; `:330` `spectral_initialize_channel_rows` | augmented dictionary `m̄=[m;b]`, principal eigenvector, magnitude-LS rescale `r̄`, phase anchor on entry `D+1` | **reuse as-is — present and validated**, 14 tests in `tests/test_spectral.py` incl. `test_rbar_formula`, `test_phase_anchor_and_eigensolver_invariance`, `test_high_snr_high_rsr_relative_error` |
| 13 | Linearised closed-form LS | `baselines.py:416` `linearised_closed_form_ls(Y,S,Psi,*,observation_source,ridge)` | per-element real `Φ_n ∈ R^{P×2K}`, `ǧ_n ∈ R^{2K}`; `observation_source ∈ {"exact_magnitude","ideal_linear"}` **required** | **present and validated** — 6 tests in `tests/test_baselines.py` incl. `test_linearised_ls_noiseless_ideal_model`, `test_linearised_complex_sign_check`, `test_linearised_ls_high_rsr_exact_magnitude`. **Do not invent one.** Track D likely does not use it (see §6.2) |
| 14 | **NMSE** | `metrics.py:216` `channel_nmse(G_hat,G,*,expected_channel_energy=None)`; accumulator `metrics.py:350` `NmseAccumulator` | **Both forms exist.** Per trial, `monte_carlo.py:958` stores `instantaneous_nmse = ‖Ĝ−G‖²_F/‖G‖²_F` **and** `error_energy`/`true_energy` separately. The **aggregate** is `NmseAccumulator.nmse_linear` (`metrics.py:386`) `= Σ error_energy / Σ true_energy` — **ratio-of-sums**. Also returns an unaligned primary plus a diagnostic phase-aligned variant and `likely_phase_anchor_problem` | **reuse as-is** — Track D **must** aggregate ratio-of-sums to be comparable |
| 15 | **Track-B Hankel — DO NOT TOUCH** | on `origin/claude/adversarial-audit-gs6iid` only: `rydberg_sim/track_b_structure.py` (`hankel_matrix`, `hankel_to_vector`, `hankel_rank`, `hankel_project`, `esprit_project`, `angular_project`, `project_matrix`), `track_b_proposed.py` (`hs_gs`, `hs_gs_auto`, `cadzow_project`, `select_order_heldout`), `track_b_drivers.py`, `track_b_prototype.py`, `tests/test_track_b_adapter.py`, `scripts/*track_b*.py`, and the whole isolated package `trackB_hankel_emgs/` (15 modules) | — | **do not modify** |
| 16 | Monte Carlo | `rng.py:132` `get_trial_rngs(master_seed, trial_index)` → 6 independent named streams (`channel, pilots, reference, noise, data, solver`) from `SeedSequence(entropy=seed, spawn_key=(trial,))`; `rng.py:104` `operating_point_spawn_key(trial, snr_db, rsr_db)` quantises dB to integer keys so `(trial,SNR,RSR)` addresses a stable world. Storage: flat CSV, `RESULT_COLUMNS` (`monte_carlo.py:130`), one row per (trial, algorithm, metric). `run_experiment(spec, out, n_workers)` supports multiprocessing | **reuse as-is** — trial 137 in isolation == trial 137 after 0..136 |
| 17 | Tests | **pytest**, `pytest.ini`: `pythonpath = .`, `testpaths = tests`, `addopts = -q`. Invoke `python3 -m pytest`. **Baseline: 274 passed, 1 skipped, 90 s** | | **reuse convention** |
| 18 | Config / CLI | Frozen `@dataclass` + `SimulationConfig.create(...)` classmethod validating invariants. **No argparse, no hydra, no YAML anywhere.** CLI is a hand-rolled `_cli(argv)` dispatching on `sys.argv[1]` (`track_a.py:225`). Isolated packages instead use a frozen module-level constant `config.py` with provenance comments plus `run_all.sh` (`trackB_hankel_emgs/`) | **follow the isolated-package pattern** |
| 19 | Plotting | matplotlib `Agg`; `plt.rcParams` set once per plotting module (`figure.dpi: 200`, `savefig.bbox: "tight"`, `savefig.facecolor: "white"`); `figsize` `(3.6,2.7)` single / `(7.0,2.7)` two-panel; **both `.png` and `.pdf`** emitted per figure; outputs under `results/<track>/<figure>/` with a co-located `README.md` recording settings, trial counts and acceptance checks | **reuse convention** |

### What Track D must **add** (nothing above duplicated)

1. A torch forward/EM-GS layer that is **numerically identical** to `gs.py` at `α=1`, `FilterNet ≡ bessel_ratio`.
2. `FilterNet` (2-layer MLP, scalar `κ` → `(0,1)`), the scalar gate `α = σ(g)`, the channel Transformer (`K` tokens × `2M`, `d_model`, `L_enc` pre-LN encoder blocks), and the `T_UR`-layer unrolled stack.
3. A training loop (Adam + cosine annealing), dataset generation/caching, and a train/val/test split keyed off the existing seeding scheme.
4. Closed-box CNN and Transformer baselines.
5. A Track D spec/driver + `run_all.sh`, and the two figures (NMSE vs SNR; NMSE vs P).

---

## 2. Probe: shapes and orientation — (a)

`N=8, K=2, P=8`, all `complex128`/`float64`.

```
  G   shape=(8, 2)      dtype=complex128
  H   shape=(8, 2)      dtype=complex128
  S   shape=(2, 8)      dtype=complex128
  B   shape=(8, 8)      dtype=complex128
  W   shape=(8, 8)      dtype=complex128
  Z   shape=(8, 8)      dtype=float64
  E   shape=(8, 8)      dtype=complex128

  G @ S + B + W  reproduces E exactly: max|diff| = 0.000e+00
```

Repository code forming `Y` — `rydberg_sim/forward.py:206, 226-227`:

```python
signal = np.matmul(G_arr, S_arr, dtype=np.complex128)
E = np.asarray(signal + B_arr + W, dtype=np.complex128)
Z = np.abs(E).astype(np.float64, copy=False)
```

**The forward model is `Y = G @ S + B`, with `G ∈ C^{N×K}`, `S ∈ C^{K×P}`, `B ∈ C^{N×P}`.**
This matches the prompt's hypothesis exactly. No transpose, no variant.

## 3. Probe: bias structure — (b)

```
  numpy.linalg.matrix_rank(B) = 1
  s_b (reference symbols)     = [1.+0.j 1.+0.j 1.+0.j 1.+0.j] ...
  B[:,1]/B[:,0] constant across n?  spread = 0.000e+00
```

`B` is the outer product `b s_bᵀ`, rank 1. With the default `s_b[p] = 1` it is exactly `b 1ᵀ`.
See §0.2 — this contradicts the prompt and matches the paper.

(In the probe, `vartheta = 0` additionally makes `a_b = 1`, so `B` is flat across `n` as well.
That is a configuration choice, not a structural property: a nonzero `vartheta` varies `B` across
`n` while keeping rank 1.)

## 4. Probe: steering sign — (c)

`θ = 30°`, `ψ = π sin 30° = 1.5707963268`.

```
  fitted per-element phase increment = -1.5707963268
  -psi = -1.5707963268   +psi = +1.5707963268
```

**Convention is `e^{−j n ψ}`, `n = 0..N−1` (0-based) — i.e. `e^{−j(n−1)ψ}` 1-based. Negative
exponent.** Repository code, `channel.py:156`: `np.exp(-1j * n * psi)`.

The paper uses `e^{+j…}`. Ours is the conjugate. **Keep ours** (Task 4 item 5 confirmed).

## 5. Probe: gates — (d), (e)

### (d) Oracle-phase gate — **PASS**

`Y_oracle = G @ S + B` formed exactly, reconstructed from its own magnitude and phase, fed through
the repository M-step (`M = S`, `b = conj(B[n])`, output `conj(u)`), bias subtracted:

```
  max absolute error  |G_hat - G|_max            = 7.190047e-16
  max relative error  ||G_hat-G||_F / ||G||_F    = 2.865117e-16

  GATE (rel < 1e-10): PASS
  cross-check vs conj(G): rel err = 1.548785e+00   (large, as required)
```

The conjugation direction is confirmed in both directions: the M-step recovers `G`, **not** `Ḡ`.

### (e) EM-GS degenerate limits — **both PASS**

**(e1)** `W = 0`, `Ĝ⁽⁰⁾ = G`: one EM-GS iteration returns `G`.

| `σ²` | rel err after 1 iteration |
|---|---|
| `1e-06` | `4.696228e-08` |
| `1e-09` | `4.696224e-11` |

The residual is not solver error — it is the exact `1 − R(κ)` bias, which vanishes linearly in `σ²`
(two decades of `σ²` buy three decades of error, consistent with `1 − R(κ) ≈ σ²/(4z|λ|)`). The truth
is a fixed point only in the limit, which is the correct EM-GS behaviour.

**(e2)** `σ² → 0` ⇒ `R(κ) → 1` ⇒ EM-GS → GS. Same `G0`, noiseless `Z`, 5 iterations:

| `σ²` | `‖G_em − G_gs‖_F/‖G_gs‖_F` | `min R(κ)` |
|---|---|---|
| `1e-1` | `8.817274e-03` | `0.8590358372` |
| `1e-3` | `7.896639e-05` | `0.9987128569` |
| `1e-5` | `7.891268e-07` | `0.9999871368` |
| `1e-7` | `7.891214e-09` | `0.9999998714` |
| `1e-9` | `7.891212e-11` | `0.9999999987` |

Clean first-order convergence, two decades per two decades.

## 6. Probe: Bessel stability — (f)

`N=8, K=2, P=8`, RSR = 10 dB, SNR = 5 dB, `σ² = 0.632456`:

```
  kappa   min    =   0.646614
  kappa   median =  33.213745
  kappa   max    = 116.291203

  R(kappa)  min = 0.3075083278   max = 0.9956911250
  any NaN? False    any Inf? False    R within [0,1]? True
```

Overflow probe:

| `x` | `R(x)` |
|---|---|
| `0` | `0.000000000000` |
| `1e0` | `0.446389965897` |
| `1e2` | `0.994987373005` |
| `1e4` | `0.999949998750` |
| `1e4+1` | `0.999950003750` |
| `1e6` | `0.999999500000` |
| `1e12` | `0.999999999999` |
| `1e300` | `1.000000000000` |

**No overflow, no NaN, at any `κ` reachable in this model.** The branch switch at `x = 1e4` is
continuous to 8 significant figures (`0.999949998750` → `0.999950003750`).

**One cosmetic caveat, reported for completeness and not a defect:** at `x = 1e300` the asymptotic
branch (`gs.py:423`) emits `RuntimeWarning: overflow encountered in multiply` when forming `8.0*x*x`.
The term correctly underflows to `0.0` and the returned value is right. Observed `κ_max ≈ 116` is
~298 orders of magnitude away, so this is unreachable in practice. **No change proposed;** noted only
so a future Track D run that raises warnings to errors is not surprised by it.

## 7. Probe: SNR / RSR realization — (g)

Definitions **as implemented** (`calibration.py`):

```
SNR = E|(GS)_np|² / E|W_np|²    ->  sigma2   = c² · sum_k beta_k / SNR_lin
RSR = E|B_np|² / E|g_nk s_kp|²  ->  |alpha_b| = sqrt(RSR_lin · beta_ref / E|s_b|²)
```

RSR's denominator is a **single user**, not the sum over `K`.

Measured as ratio of summed energies over 400 realizations (`N=8, K=2, P=8`):

| requested SNR | measured SNR | requested RSR | measured RSR |
|---|---|---|---|
| 5.0 dB | **5.093 dB** | 10.0 dB | **9.893 dB** |
| 0.0 dB | **0.093 dB** | 10.0 dB | **9.893 dB** |
| 12.0 dB | **12.093 dB** | 12.0 dB | **11.893 dB** |

Both definitions are implemented as documented. The residual ≈ 0.09 dB is a finite-sample artifact
of the same kind the technical report documents at 4,000 realizations (row normalization holds in
expectation, not per realization); it is uniform across operating points, which is what a correctly
calibrated generator looks like. No CI was computed here — this probe is a convention check, not the
calibration measurement.

Single-realization helpers `measure_snr` / `measure_rsr` returned 4.071 dB and 7.824 dB on one draw,
which is expected scatter at `N·P = 64` samples and is why the aggregate above uses 400.

---

## 8. Paper-vs-repository discrepancies

Confirmed, intentional, **not to be "fixed"**.

| # | Item | Paper | Repository | Action |
|---|---|---|---|---|
| 1 | Pilot orientation | `S ∈ C^{P×K}`, writes `H Sᵀ`, M-step `(Sᵀ)†` | `S ∈ C^{K×P}`, `G S`, M-step `solve(S Sᴴ, S r)` | **Keep ours.** Never mechanically transcribe `(S.T).pinv()` |
| 2 | Bias structure | `B = b1ᵀ`, rank 1 | **also rank 1** (`outer(c·α_b·a_b, s_b)`, `s_b ≡ 1`) | **Prompt hypothesis was wrong — we already match.** Nothing to reconcile |
| 3 | Channel model | clustered Saleh–Valenzuela, `√(M/N_ray)` scaling | geometric specular ULA, `α ~ CN(0, β_k/L_k)` | **Keep ours.** Not reproducing S-V |
| 4 | Transduction gain | `G` scalar **outside** the magnitude: `z = G\|Hs+b+w\|` | `c` folded in, `G = cH`, `Z = \|GS+B+W\|`, `c = 1.0` baseline | **Keep ours.** Estimate `G`; `Ĥ = Ĝ/c`. Note the repo does **not** currently report `Ĥ` separately — NMSE is computed on `G` (`monte_carlo.py:945`), and since `c` is a known real scalar the NMSE is identical for both |
| 5 | Steering sign | `e^{+j…}` | `e^{−j n ψ}` (measured, §4) | **Keep ours** |
| 6 | **`R(κ)` definition** | *never defined* — the paper says only "the ratio of modified Bessel functions" | `I₁(κ)/I₀(κ)` via `scipy.special.ive`, from Cui et al. | **Repository is the authority.** Record that the paper is silent |

### 8.1 A sixth discrepancy the prompt did not list: RSR differs by a factor of K

The paper (§IV) defines `RSR = E(|b|²)/E(|Hs_p|²)` — a **multi-user** denominator, the full `Hs_p`.
Our `calibration.py` uses Cui's **single-user** denominator `E|g_{n,k}s_{k,p}|²`.

These differ by exactly `K`. At the paper's `RSR = 10 dB` with `K` users, "10 dB" in the paper's
convention is `10 + 10log₁₀K` dB in ours (`+4.77 dB` at `K=3`). This is the same factor-`K` trap the
technical report flags for SNR/RSR, in a new place.

**Nothing to fix** — but Track D must state which convention its figures use, and if any number is
ever compared against the paper's Fig. 3, the offset must be applied explicitly. Flagged as
decision **D4**.

### 8.2 `L_k` is fixed in `SimulationConfig`, not redrawn

`config.py:92` stores `L_k` as a frozen tuple. The Track-B study draws `L_k ~ U{3..7}` i.i.d. per
user per realization (`trackB_hankel_emgs/config.py`: `L_MIN, L_MAX = 3, 7`), which is implemented in
the Track-B drivers on the audit branch, **not** in `generate_ula_channel`. Track D must decide
whether to match Track B's random `L_k` (needed for a like-for-like HS-GS comparison later) or fix
it. Flagged as decision **D3**.

---

## 9. Environment

| | |
|---|---|
| Python | 3.11 (`/usr/local/bin/python3`) |
| NumPy | **2.4.6** |
| SciPy | **1.17.1** |
| **PyTorch** | **ABSENT** |
| CUDA | **No.** No `nvidia-smi`, no GPU. **CPU-only.** |
| CPUs | **4** |
| RAM | **15.7 GiB** (`MemTotal: 16461084 kB`) |
| Test framework | pytest; `python3 -m pytest`; `pytest.ini` sets `pythonpath=.`, `testpaths=tests`, `addopts=-q` |
| Baseline | **274 passed, 1 skipped**, 90 s |

**Dependency declaration: there is none.** No `requirements.txt`, no `pyproject.toml`, no `setup.py`,
no `setup.cfg`. `pytest.ini` is the only config file. Dependencies are implicit.

**The container arrived with numpy, scipy and pytest missing** — nothing in the repository could run.
I installed `numpy`, `scipy`, `pytest` (and `pypdf` to read the paper) into the container to run the
probe and the baseline suite. These are **container-level installs that touch no tracked file**; no
dependency was declared anywhere, per the stop condition.

**`torch.special.i0e` / `i1e` could not be checked** — torch is not installed. Both have existed
since torch 1.7, so availability is near-certain, but this is unverified and is listed as an open
question rather than asserted.

**PyTorch would be a new dependency**, and on CPU-only hardware with 4 cores. Flagged as decision
**D2**; see §10.

---

## 10. Decisions needed before implementation

Flagging each condition the prompt asked me to raise immediately:

| | Condition | Status |
|---|---|---|
| ✅ | spectral initializer missing or unvalidated | **No** — present (`spectral.py:278`) and validated by 14 tests |
| ✅ | linearised/closed-form LS initializer missing | **No** — present (`baselines.py:416`), validated by 6 tests. Nothing to invent |
| ✅ | oracle-phase gate failed | **No** — passed at `2.87e-16`, 6 orders inside the `1e-10` gate |
| ⚠️ | **PyTorch is a new dependency** | **Yes** — see D2 |
| ✅ | `Y` orientation differs from the prompt's hypothesis | **No** — `Y = G @ S + B` exactly as hypothesised |

### D1 — Which branch is Track D's base? *(blocking)*

Track A only (`main`, this branch) or the full Track-A+B tree
(`origin/claude/adversarial-audit-gs6iid`)?

**Recommendation: base Track D on the audit branch.** The stated end goal is to merge URformer with
the Hankel plan, and the HS-GS/EM-GS comparison, the `L_k ~ U{3..7}` convention and the frozen
`trackB_hankel_emgs/config.py` provenance all live there. Building on `main` means re-deriving them
later and risks Track D silently diverging from the Track-B operating point it must eventually be
compared against. Core conventions are identical either way, so this costs nothing now and saves a
reconciliation later.

### D2 — PyTorch on CPU-only hardware *(blocking)*

Adding torch is unavoidable for an unrolled network. Two consequences:

- **It is a new, undeclared dependency.** The repository declares none at all. Do you want Track D to
  introduce the first `requirements.txt` (pinning numpy/scipy/pytest/torch), or stay implicit?
- **4 CPU cores, no GPU.** Training `T_UR` unrolled layers each containing `L_enc` Transformer
  encoder blocks, over 50 epochs, is materially slower here than the paper's setup. This is a real
  budget question, not a blocker: I would size the study to what CPU allows and report trial counts
  and training time honestly, the way Track B tiered its budget. Worth deciding the target scale
  before I build rather than after.

### D3 — Random or fixed `L_k`? *(blocking for comparability)*

Match Track B (`L_k ~ U{3..7}` i.i.d. per user per realization) or fix `L_k`? **Recommendation:
match Track B**, so the eventual HS-URformer ablation is like-for-like.

### D4 — Which RSR convention do Track D figures use?

Ours (single-user, Cui) or the paper's (multi-user)? **Recommendation: ours**, with the `10log₁₀K`
offset stated explicitly wherever a paper number is quoted.

### D5 — Table I is unavailable

The paper's Table I is a **scanned image**; text extraction did not recover it. Missing: `M`, `K`,
`L`, `C_l`, `d_model`, `L_enc`, `T_UR`, learning rate, dataset size, batch size. I can choose
defensible defaults and document them as Track A documents its eight deviations — but if you can read
those values off the PDF, that removes a whole class of guesswork.

### D6 — Baseline initialization *(scientific, raised earlier)*

The paper initializes `Ĥ⁽⁰⁾` as random complex Gaussian for everything, including the GS/EM-GS
baselines. We have a validated spectral initializer that is a much stronger start. **Recommendation:
report both** — random-init EM-GS to reproduce the paper's claim, spectral-init EM-GS as the honest
control. The gap may narrow substantially, and that is a finding worth having rather than avoiding.

---

## 11. `git status --porcelain`

```
?? reports/
?? scratch/
```

Zero modified files. The two untracked directories contain exactly the three permitted files:
`reports/trackD_audit.md`, `reports/trackD_audit.json`, `scratch/trackD_audit_probe.py`.
(`git status --porcelain` collapses wholly-untracked directories; the per-file listing from
`git status --porcelain -uall` is given in the turn summary.)

**No implementation has begun.**
