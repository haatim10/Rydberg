# Track D (URformer) — implementation and verification report

Rendered from `reports/trackD_verify.json`, `reports/trackD_timing.json` and
`reports/trackD_audit.json`. Not reconstructed from memory.

**Status: built and verified. Nothing trained, no D-experiment run.**
15/15 gates pass. Awaiting approval before training.

| | |
|---|---|
| Base commit | `54c88d5d5888923d973b6ff6a429e51da75b61df` |
| Audit commit | `74984c7` → cherry-picked as `7b0955d` |
| torch | 2.13.0+cu130, **CPU-only** (`cuda_available: false`) |
| Test suite | **378 passed, 1 skipped** (baseline 358+1, plus 20 new Track D tests) |

---

## 1. Files created

**Package** (`trackD_urformer/`): `__init__.py`, `config.py`, `dataset.py`,
`torch_forward.py`, `filter_net.py`, `transformer.py`, `urformer.py`,
`baselines.py`, `train.py`, `evaluate.py`, `verify.py`, `runner.py`,
`plot_results.py`, `run_all.sh`, `README.md`.

**Tests**: `tests/test_trackD_urformer.py` (20 tests).

**Reports**: `reports/trackD_verify.json`, `reports/trackD_verify.md`,
`reports/trackD_timing.json`, `reports/trackD_phase2.md`.

**Scratch**: `scratch/trackD_timing_probe.py`.

**Sanctioned modification outside the package**: `requirements.txt` — new file,
see item 18.

The layout follows the prompt's proposal exactly; no deviation was needed.

## 2. Existing code reused — reimplemented nothing

| Purpose | Reused symbol |
|---|---|
| Trial world | `monte_carlo.generate_channel_estimation_trial` |
| Channel | `channel.generate_ula_channel`, `channel.steering_matrix` |
| `L_k ~ U{3..7}` | `track_b_drivers.draw_L_k`, `track_b_drivers.track_b_spec` |
| Pilots | `pilots.generate_gaussian_pilots` |
| Reference field | `reference.generate_reference_field` |
| Forward / noise | `forward.exact_forward`, `forward.reference_phase_matrix` |
| SNR / RSR | `calibration.snr_db_to_sigma2`, `rsr_db_to_alpha_magnitude`, `make_alpha_b`, `reference_user_beta` |
| GS / EM-GS | `gs.biased_gs_channel_rows`, `gs.em_gs_channel_rows` |
| `R(κ)` | `gs.bessel_ratio` (as the reference for gate E) |
| Spectral init | `spectral.spectral_initialize_channel_rows` |
| Linearised LS | `baselines.linearised_closed_form_ls` |
| NMSE | `metrics.channel_nmse` |
| Seeding | `rng.get_operating_point_rngs` |

Track A and Track B code paths are untouched.

## 3. Confirmed matrix shapes

`G (b,N,K)` complex · `S (b,K,P)` complex · `B (b,N,P)` complex ·
`Z (b,N,P)` real · `W (b,N,P)` complex.

Gate A validated `(N,K,P,batch)` = `(8,2,8,1)`, `(16,3,20,4)`, `(32,3,30,2)`,
`(8,4,12,3)` plus **5 negative cases** that must raise (K mismatch, unbatched,
real `G`, complex `Z`, `N` mismatch). All raised. **PASS**

## 4. Confirmed system-model equations

Repository code, `rydberg_sim/forward.py:206,226-227`:

```python
signal = np.matmul(G_arr, S_arr, dtype=np.complex128)
E = np.asarray(signal + B_arr + W, dtype=np.complex128)
Z = np.abs(E).astype(np.float64, copy=False)
```

So `Y = G @ S + B`, `Z = |Y + W|`. The prompt's hypothesised orientation is
confirmed; no transpose, and `(S.T).pinv()` was never ported.

M-step, `rydberg_sim/gs.py:326-331`, per row with `M=S`, `u=conj(g_n)`,
`b=conj(B[n])`. Unwound, this is ordinary least squares
`G = R S^H (S S^H)^{-1}`, which is what `torch_forward.least_squares_G`
computes in batched form. **The equivalence is asserted, not assumed** —
gate C.

## 5. Oracle-phase recovery error

From the audit (`reports/trackD_audit.json`), re-confirmed in gate G and in
`test_least_squares_recovers_G_not_conjugate`:

- relative error **2.87e-16** (threshold 1e-10)
- cross-check against `conj(G)`: **1.55** — large, so the M-step recovers `G`
  and not its conjugate.

## 6. NumPy ↔ Torch LS parity — gate C

| dtype | max relative error | tolerance | result |
|---|---|---|---|
| float64 | **5.597e-13** | 1e-12 | PASS |
| float32 | **1.479e-07** | 1e-05 | PASS |

Cases: `(N,P)` = `(8,8)`, `(16,20)`, `(32,30)`.

Forward-model parity (gate B): float64 **9.799e-17**, float32 **5.678e-08**.

The float32 figures are the **training-precision floor**: ~1e-7 relative is
what single precision can represent, so no learned result should ever be
interpreted below that.

## 7. GS-degeneration error — gate D

`α = 0`, residual disabled ⇒ one URformer layer ≡ one classical biased-GS
update.

| dtype | relative error | tolerance | result |
|---|---|---|---|
| float64 | **5.871e-13** | 1e-12 | PASS |
| float32 | **1.910e-07** | 1e-05 | PASS |

## 8. EM-GS-degeneration error — gate E

FilterNet replaced by exact `i1e/i0e`, `α = 1`, residual disabled ⇒ one layer
≡ one classical EM-GS iteration.

| dtype | relative error | tolerance | result |
|---|---|---|---|
| float64 | **5.700e-13** | 1e-10 | PASS |
| float32 | **2.429e-07** | 1e-05 | PASS |

Torch `i1e/i0e` vs `scipy.special.ive`: **2.488e-16** over this gate's observed
κ range, **κ ∈ [1.51, 55.90]** (the audit's separate probe configuration
reached κ_max ≈ 116; both are far below any overflow regime). Raw `I₁/I₀` is
never formed anywhere.

**Gate J** additionally confirms the classical limits: noiseless fixed point
`3.28e-14`, and EM-GS → GS as `σ²→0` at `3.06e-14`.

## 9. Conjugation test — gate G

Deterministic one-path channel, `θ = 30°`, `α = 1`, `ψ = 1.5707963268`.

- fitted phase slope **−1.5707963268** ⇒ convention is `e^{−jnψ}`, matching
  `channel.py:156` and the audit
- NumPy → Torch: **exactly 0.0**
- through LS: **9.748e-17** vs `G`; **1.4142** vs `conj(G)` (large, as required)
- tokenize → detokenize: **exactly 0.0** vs `G`; **1.4142** vs `conj(G)`
- slope after the full round trip: **−1.570796326794896**, bit-identical to the
  original

**No conjugation flip survives anywhere in the chain.** PASS

## 10. Gradient test — gate H

float32, batch 4, `T_UR=3`, loss 0.2243. All gradients finite and strictly
non-zero across every unrolled layer and every module:

| layer | FilterNet | gate | Transformer |
|---|---|---|---|
| 0 | 7.897e-03 | 1.354e-03 | 2.791e-01 |
| 1 | 3.809e-02 | 5.317e-03 | 3.220e-01 |
| 2 | 7.188e-02 | 1.559e-02 | 4.272e-01 |

Values verbatim from `reports/trackD_verify.json`. Every norm grows toward the
output layer — the expected pattern for an unrolled network trained on the
final layer only, and evidence that gradient signal reaches layer 0 rather
than vanishing. The gate gradients are the smallest by ~2 orders of magnitude,
which is consistent with `α` being a single scalar per layer against the
Transformer's 158,592 parameters.

## 11. Tiny-dataset overfit — gate I (hard stop)

32 deterministic samples, `N=16, K=3, P=20`, `T_UR=4`, `d_model=32`,
`L_enc=2`, 1500 steps, float32.

**Best NMSE −136.12 dB** (target `< −25 dB`). **PASS**

That number is deep enough to warrant an explicit caveat. On the *same* 32
samples the classical estimators reach:

| estimator (100 iters, random init) | NMSE |
|---|---|
| EM-GS | −7.24 dB |
| GS | −6.88 dB |
| linearised LS | −5.91 dB |

So the ~129 dB gap is **memorization of a 32-sample training set**, which is
exactly what gate I is designed to detect (sufficient capacity, and gradients
that actually flow). −136 dB sits at the float32 numerical floor
(ε≈1e-7 ⇒ NMSE≈1e-14). **It is not a generalization result and must not be
quoted as one.** Held-out performance is unknown until a real training run.

## 12. Parameter count

Reference config: `N=32, K=3, T_UR=10, d_model=64, L_enc=3, n_heads=4`.

| Module | Per layer | × 10 layers |
|---|---|---|
| FilterNet | 97 | 970 |
| Gate | 1 | 10 |
| Transformer | 158,592 | 1,585,920 |
| **Total** | **158,690** | **1,586,900** |

Untied weights, so per-layer sums equal the total. The Transformer is
**99.94%** of all parameters — the learned filter and gate together are 980
parameters, under 0.07%.

Initial `α_t` = **0.1192** for all 10 layers (`gate_init="near_gs"`, `g=−2.0`),
so the network starts close to plain GS.

> **CORRECTION (PROMPT 3 item 2).** This section originally continued: "the
> untrained URformer *is* the classical algorithm." **That was wrong.** The
> Transformer residual is exactly zero at init, and the architecture *can be
> forced* to exact GS or exact EM-GS — but the **default untrained** network is
> neither. With `α = 0.1192` and the default random FilterNet, the effective
> multiplier `α·R_learned + (1−α)` measures **0.934–0.941**, not 1, so the
> untrained model sits **0.1230** relative from one GS step and **0.0947** from
> one EM-GS step. Full algebra, measurements and the four separated claims are
> in `reports/trackD_stage1_plan.md` §2.

Gate F's exact-`0.0` requirement concerns the **Transformer residual only**,
which is genuinely zero at initialization; it never implied the whole network
equalled a classical estimator.

## 13. GPU/CPU availability

**CPU-only.** No GPU, no `nvidia-smi`, `torch.cuda.is_available() == False`.
4 cores, 15.7 GiB RAM, 30 GB free disk.

## 14. Estimated training memory

- parameters: **6.05 MB** (float32)
- Adam moments (2×): **12.11 MB**
- activations, batch 32, `T_UR=10`: order 10–50 MB
- **total well under 100 MB.** Memory is not a constraint; the 15.7 GiB machine
  is nowhere near limiting.

## 15. Measured training speed

Measured, not assumed — `scratch/trackD_timing_probe.py`, 200 samples × 3
epochs, median of 3, after one untimed warm-up epoch.

| threads | s/epoch | samples/s |
|---|---|---|
| 1 | 0.870 | 229.8 |
| 2 | 0.698 | 286.4 |
| 4 | 0.782 | 255.9 |

**Thread count barely matters and the ordering is not stable.** An earlier
identical run ranked 4 threads fastest (0.683 s/epoch) and 1 thread slowest
(0.959); this run ranks 2 fastest. The spread across all settings is within
run-to-run noise on a shared 4-core box, so the prompt's hypothesis — that
dispatch overhead would make *fewer* threads clearly faster — is **not
supported**; nothing here justifies a strong choice. `TrainConfig.num_threads`
is set to 1 for determinism and reproducibility rather than speed, and the
difference costs at most ~20%.

Per-`N` throughput (at 2 threads):

| N | samples/s | hours per training |
|---|---|---|
| 8 | 350.9 | 0.79 |
| 16 | 316.7 | 0.88 |
| 32 | 251.3 | 1.11 |

Dataset generation: **1.27 ms/sample** ⇒ 25.4 s for 20,000. Negligible, and
cached in memory after the first epoch.

## 16. Estimated runtime

**One training** (20,000 samples × 50 epochs, `N=32`): **0.97 h** compute
\+ 0.007 h generation ≈ **1.0 hour**.

**Full training matrix** (§6 of the prompt; 3 initializers are 3 *separately
trained models*, not a runtime flag):

| Experiment | Trainings | Detail | Hours |
|---|---|---|---|
| D1 — NMSE vs SNR | 3 | `N=32, P=20`, 3 initializers | 3.33 |
| D2 — NMSE vs pilots | 3 | `N=32`, `P~U{6,10,15,20,30}`, 3 initializers | 3.33 |
| D3 — NMSE vs array size | 6 | `N∈{8,16}` × 3 initializers; the `N=32` column **reuses D1** | 5.01 |
| **Total** | **12** | | **11.67 h** |

GS, EM-GS and linearised LS require no training. D2 needs no per-`P` model
because `P` is architecture-free — but its training distribution must cover the
sweep. D3 needs one model per `N` because of shape-locking.

~11.7 h of CPU is entirely feasible here. **CPU is sufficient; no GPU is
needed.**

## 17. Recommended final dataset size

**Keep the paper's 20,000**, and treat it as a floor rather than a tuned value.

Reasoning: at 1.27 ms/sample, data is essentially free — 20,000 samples cost
25 seconds to generate and 6 MB to hold. The binding cost is the 50 training
epochs, not the data. Since generation is cheap and the model has 1.59 M
parameters against 20,000 × 96 = 1.92 M real supervised targets, the
sample-to-parameter ratio is only ~1.2:1, which is thin. Gate I confirmed the
network *can* memorize, so overfitting is a live risk, not a hypothetical.

I would therefore also run a **40,000-sample control** for the D1 reference
configuration (cost: +1 hour) to check the val curve does not diverge from
train. If it does, raise the headline size rather than adding regularization —
data is the cheapest fix available here.

## 18. Discrepancies between Xiao et al. and our implementation

Recorded in `config.PAPER_DIVERGENCES` and rendered into every report.

| # | Item | Paper | Ours | Why |
|---|---|---|---|---|
| 1 | `K` | 4 | **3** | Track B's frozen `K=3`, so the later HS ablation is like-for-like |
| 2 | Channel | Saleh–Valenzuela, 4 clusters × 10 subrays | geometric ULA, `L_k~U{3..7}` | We keep our model; clusters/subrays do not apply |
| 3 | **RSR denominator** | multi-user `E‖Hs_p‖²` | single-user (Cui eq. 37) | Differs by exactly `K`. **Both stored on every row**: `rsr_ours_dB` and `rsr_paper_equiv_dB = rsr_ours_dB + 10log₁₀K` (+4.77 dB at K=3) |
| 4 | Steering sign | `e^{+j…}` | `e^{−jnψ}` | `channel.py:156`, confirmed by gate G |
| 5 | Pilot orientation | `S∈C^{P×K}`, `HSᵀ`, `(Sᵀ)†` | `S∈C^{K×P}`, `GS`, `RS^H(SS^H)^{-1}` | `gs.py:326-331` |
| 6 | Transduction gain | outside the magnitude | `c` folded in, `G=cH`, `c=1` | Known real scalar ⇒ NMSE identical |
| 7 | `R(κ)` | **never defined** — "the ratio of modified Bessel functions" | `I₁/I₀` via scaled Bessels | Repository is the authority |
| 8 | torch wheel | n/a | PyPI CUDA-linked build, run CPU-only | **The prompt asked for `--index-url https://download.pytorch.org/whl/cpu`, but that host is blocked by this environment's proxy policy (403 on CONNECT).** PyPI is the only reachable index. `cuda_available` is False either way; this is a packaging-size cost (~2.5 GB unused nvidia wheels), not a behavioural difference. `requirements.txt` documents the preferred command for unblocked machines. |

Two further points the prompt asserted that the repository contradicts, both
already settled by the audit and re-confirmed here: **`B` is rank 1**, not
time-varying (no time-varying-`B` code path was added), and **κ is small**
(max ≈ 116), so the FilterNet warm-start grid is calibrated to a measured range
rather than an assumed `logspace(-3,5)`.

## 19. Assumptions I had to make

1. **`n_heads = 4`, `ffn_mult = 4`, `dropout = 0`** — the paper gives
   `d_model=64` and `L_enc=3` but not the head count or FFN ratio. These are
   standard Transformer defaults; dropout is off because the dataset is
   synthetic and effectively unlimited.
2. **`filter_hidden = 32`** — the paper says only "compact two-layer MLP".
3. **`gate_init = "near_gs"` (`g=−2`)** as the default, per the prompt's
   recommendation; the paper does not specify gate initialization.
4. **Zero-initialized Transformer output projection** — not stated in the
   paper. Chosen so the untrained network equals the classical algorithm, which
   makes gate F exact and gives training a sane starting point.
5. **Validation/test sizes 2,000 each** — the paper states only the 20,000
   training figure.
6. **`N=32`** taken as the reference array size, matching Table I's `M=32` and
   the top of Track B's `N_GRID`.
7. **RSR = 10 dB in *our* convention** — the paper's Fig. 3 value is 10 dB in
   *its* convention. I used 10 dB in ours and store both fields. If you want a
   literal match to the paper's operating point, set `rsr_db = 14.77`; this is
   a one-line config change and is flagged as a decision, not silently chosen.
8. **Outlier criterion `NMSE_linear > 1.0`** (worse than returning zero),
   declared in `evaluate.py` *before* any run so it cannot be chosen after
   seeing results.
9. **SNR quantized to the millidB grid** — forced by
   `rydberg_sim.rng.db_to_key`, which rejects off-grid values. Repository wins.

## 20. `git status --porcelain`

See the turn summary for the verbatim listing. Additive only: new files under
`trackD_urformer/`, `reports/`, `tests/`, `scratch/`, plus the sanctioned
`requirements.txt`. **Zero existing files modified.**

---

## Not done, by instruction

No Hankel/Cadzow/ESPRIT/rank-truncation anywhere. No antenna-token variant. No
closed-box Transformer baseline. Deep supervision implemented but off. No
training launched, no D-experiment run, no figure produced.
