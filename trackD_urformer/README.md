# Track D — URformer

Unrolled phase retrieval for Rydberg atomic channel estimation, implemented on
**our existing geometric ULA model** so the only major change versus the current
pipeline is the estimator.

**Reference.** J. Xiao, J. Wang, M. Zeng, H. Xu, X. Li and A. Nallanathan,
"Channel Estimation for Rydberg Atomic Quantum Receivers: Unrolled Phase
Retrieval From Holographic Snapshots," *IEEE Signal Processing Letters*,
vol. 33, pp. 1696–1700, 2026.

| | |
|---|---|
| Base commit | `54c88d5d5888923d973b6ff6a429e51da75b61df` (`claude/adversarial-audit-gs6iid`) |
| Audit commit | `74984c7`, cherry-picked onto this branch |
| Audit report | `reports/trackD_audit.json` — authority on every repository convention |
| Verification | `reports/trackD_verify.json` / `.md` — **15/15 gates pass** |
| Status | **built and verified, NOT trained.** No D-experiment has been run. |

## Status: nothing has been trained

`train.py`, `runner.py`, `plot_results.py` and `run_all.sh` are built and left
deliberately unrun. `runner.py` refuses without `--i-have-approval`;
`run_all.sh` refuses without `TRACKD_APPROVED=1`. To inspect the plan without
running anything:

```bash
PYTHONPATH=. python3 -m trackD_urformer.runner plan
```

To re-run the verification gates (~2 minutes, CPU):

```bash
PYTHONPATH=. python3 -m trackD_urformer.verify
```

## What this is

The paper replaces three of EM-GS's rigid blocks with learnable ones and keeps
the fourth. Per unrolled layer `t`:

```
Y        = G^(t-1) @ S + B
Y_direct = Z * unit_phase(Y)                   eps-guarded; no torch.angle
kappa    = 2 Z |Y| / sigma2                    eps-guarded
R_learn  = FilterNet_t(kappa)          [1]     learned, replaces I1/I0
alpha_t  = sigmoid(g_t)                [2]     learned stabilizing gate
Y_rec    = alpha_t (Y_direct R_learn) + (1 - alpha_t) Y_direct
G_lin    = LS(Y_rec - B, S)            [3]     NOT learned - repository M-step
G^(t)    = G_lin + Former_t(G_lin)     [4]     learned user-token Transformer
```

`T_UR = 10` layers, untied weights.

## Design decisions worth knowing

**The network starts as the classical algorithm.** `gate_init="near_gs"` puts
every `alpha_t` at 0.119, and the Transformer's output projection is
zero-initialized, so at step 0 the URformer *is* biased GS with a small EM
correction. Training moves it away from there. This is why gate F can require
the residual to be *exactly* 0.0 rather than merely small.

**The LS step is never learned.** It reuses the repository's M-step, written in
batched form as `G = R S^H (S S^H)^{-1}`. That this equals the repository's
per-row canonical solve is asserted, not assumed — gate C, `< 1e-12` in float64.

**Shape-locking.** The user-token scheme locks the Transformer to `(N, K)`: the
projections are `2N`-dimensional and the positional embedding is `K × d_model`.
So **D3 needs one trained model per `N`** — evaluating an `N=32` model at
`N=16` is a shape error, not a performance question, and the code raises rather
than silently reshaping. `P` and SNR are architecture-free.

**Both RSR conventions are stored on every row.** The paper's RSR uses a
multi-user denominator; ours uses Cui's single-user one. They differ by exactly
`K`. Every result row carries `rsr_ours_dB` and
`rsr_paper_equiv_dB = rsr_ours_dB + 10 log10 K`, so no figure caption has to
carry the correction by hand.

**Initializers are a scientific control, not a convenience.** The paper
random-inits everything, *including* its own GS/EM-GS baselines, which is what
makes its baselines look weak. We report all three — `random` (reproduces their
claim), `spectral` and `linearized_ls` (the honest controls) — and the
initializer is named in every label, legend and stored row. Note that
`URformer-random` and `URformer-spectral` are two **separately trained models**,
not one model with a runtime flag; this multiplies the training matrix.

## What is deliberately absent

Per the phase's stop conditions:

- **No Hankel anything** — no projection, no Cadzow, no rank truncation, no
  ESPRIT, no low-rank loss, not as a layer and not as an option. Track B stays
  separate; HS-URformer is a later phase.
- **No antenna-token Transformer variant** — user tokens only.
- **No closed-box Transformer baseline** — deferred.
- **Deep supervision is off** — implemented behind `ModelConfig.deep_supervision`
  but not enabled in the reference run.

## Files

| File | Role |
|---|---|
| `config.py` | Frozen dataclasses; serialized into every checkpoint and report. Records Table I and every divergence from it. |
| `dataset.py` | Thin wrapper over `rydberg_sim`'s generator. Disjoint train/val/test seed ranges, `fixed_S`/`random_S`, per-sample SNR. |
| `torch_forward.py` | Differentiable forward model, batched LS, fixed-weight GS/EM-GS layers. No new mathematical convention. |
| `filter_net.py` | Learnable Bessel replacement + measured-range warm start. |
| `transformer.py` | User-token residual correction. Shape-locked to `(N,K)`. |
| `urformer.py` | The unrolled network and its parameter accounting. |
| `baselines.py` | Thin adapters over existing GS / EM-GS / initializers. Reimplements nothing. |
| `train.py` | Adam + cosine annealing, resumable checkpoints. **Not run.** |
| `evaluate.py` | Paired Monte Carlo, per-trial rows, ratio-of-sums aggregation. **Not run.** |
| `verify.py` | Gates A–K. Writes `reports/trackD_verify.{json,md}`. |
| `runner.py`, `plot_results.py`, `run_all.sh` | Scaffolding. Refuse to run without approval. |

## Reused from the repository — reimplemented nothing

`generate_channel_estimation_trial`, `generate_ula_channel`, `draw_L_k`,
`track_b_spec`, `generate_gaussian_pilots`, `generate_reference_field`,
`exact_forward`, `snr_db_to_sigma2`, `rsr_db_to_alpha_magnitude`,
`biased_gs_channel_rows`, `em_gs_channel_rows`, `bessel_ratio`,
`spectral_initialize_channel_rows`, `linearised_closed_form_ls`,
`channel_nmse`, `get_operating_point_rngs`, `steering_matrix`,
`reference_phase_matrix`.

Track A and Track B code paths are untouched.

## Two constraints discovered while building

1. **SNR must lie on a millidB grid.** `rydberg_sim.rng.db_to_key` encodes the
   operating point as an integer number of millidB and *rejects* anything off
   that grid, so per-sample SNR sampling is quantized to 0.001 dB. The
   repository's convention wins; the resolution is far finer than any effect we
   measure.
2. **`trackB_hankel_emgs` is not importable as a package** — it does
   `import config`, expecting its own working directory. Track D therefore
   reuses the underlying `rydberg_sim.track_b_drivers` helpers directly rather
   than importing Track B. Track B is not modified.
