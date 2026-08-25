# Audit of the existing Track-B implementation

Performed before writing any new code. Every finding below was checked by
reading the source and, where it is a behavioural claim, by executing it.
`verify_results.py` re-runs the executable ones on every invocation.

## 1. Code map of the existing implementation

| Component | File | Notes |
|---|---|---|
| System model / config | `rydberg_sim/config.py` | `SimulationConfig`, frozen |
| Geometric ULA channel | `rydberg_sim/channel.py` | `generate_ula_channel` |
| AoA + path gains | `rydberg_sim/channel.py` | inside the same generator |
| Pilot generator | `rydberg_sim/pilots.py` | `generate_gaussian_pilots` |
| Reference field `B` | `rydberg_sim/reference.py` | `generate_reference_field` |
| Noise + observation | `rydberg_sim/forward.py` | `exact_forward` |
| Biased GS / EM-GS | `rydberg_sim/gs.py` | `biased_gs_channel_rows`, `em_gs_channel_rows` |
| Hankel lifting | `rydberg_sim/track_b_structure.py` | `hankel_matrix`, `hankel_to_vector` |
| Cadzow projection | `rydberg_sim/track_b_proposed.py` | `cadzow_project` |
| Rank selection | `rydberg_sim/track_b_proposed.py` | `select_order_heldout` |
| HS-GS estimator | `rydberg_sim/track_b_proposed.py` | `hs_gs`, `hs_gs_auto` |
| NMSE | `rydberg_sim/metrics.py` | `channel_nmse` |
| Trial generation | `rydberg_sim/monte_carlo.py` | `generate_channel_estimation_trial` |
| Track-B defaults / drivers | `rydberg_sim/track_b_drivers.py` | `track_b_spec`, `draw_L_k` |
| Monte Carlo scripts | `scripts/run_b3.py`, `run_b7_pathcount.py` | checkpointed sweeps |
| Result store | `results/track_b/**/*.npz` | per-trial numerator + shared denominator |

## 2. Is Cadzow post-hoc or interleaved?

**Interleaved.** `hs_gs` runs

```python
for t in range(1, max_iter + 1):
    G = _exact_iteration(exact_step, S, Z, B, sigma2, G, ridge)
    if active and t % project_every == 0:
        for k in range(G.shape[1]):
            G[:, k] = cadzow_project(G[:, k], L_hat, ...)
```

with `project_every = 1` by default, so the projection is applied after **every
one** of the 50 EM-GS updates, not once after convergence. This has NOT been
changed. Post-hoc projection (`project_every = max_iter`) is available as an
ablation in `ablations.py` and is reported separately.

## 3. How is the target rank selected?

`select_order_heldout` — a **held-out pilot residual**, not an oracle:

* pilot columns are split, the last `val_frac = 0.3` held out;
* for each candidate `L = 1 .. r_max`, HS-GS is run on the fitting columns for
  `select_iter` iterations;
* the candidate minimising the magnitude residual on the held-out columns wins.

Only `S`, `Z`, `B` enter — all observable at the receiver. The true `L_k` never
appears (`verify_results.py` check 5 greps the function body to confirm).

**Known limitation, preserved deliberately:** the selector returns a *single
scalar* `L_hat` applied to all `K` user columns, even though the users have
different true orders.

## 4. Verified model facts

| Fact | Value | How checked |
|---|---|---|
| Observation | `Z = |GS + B + W|` exactly | identity holds to 0.0 |
| Noise | inside the modulus, `CN(0, sigma2)` i.i.d. | implied by the identity |
| `sigma2` | `K / SNR_lin` (Cui eq. 36) | 0.94868 at SNR 5 dB, K=3 |
| RSR | single-user denominator (Cui eq. 37) | `|b| = sqrt(RSR_lin)` |
| Reference `B` | constant over (n,p), `angle(b) = 0` | one unique `|B|` |
| AoA | uniform in **theta**, not `sin theta` | KS p=0.150 vs 0.0000 |
| Path gain | `CN(0, beta_k / L_k)`, `beta_k = 1` | equal average per-path power |
| `c` | 1, so `G == H` exactly | direct comparison |
| `L_k` | `U{3..7}` i.i.d. per user per trial | dedicated RNG substream |
| Pilots | `CN(0,1)`, redrawn, full row rank, `P >= 2K` | NOT orthogonal, max off-diag 0.233 |
| Rank ceiling | `r_max = ceil(N/2)` = 4 / 8 / 16 | Hankel shapes (5,4)/(9,8)/(17,16) |
| Cadzow sweeps | 4 | `cadzow_project(n_iter=4)` |
| NMSE | ratio of sums, `10 log10` | never a mean of dB, never 20 log10 |

## 5. Fairness properties inherited from the existing code

Two facts make the comparison structurally fair rather than fair by convention,
and both were executed, not assumed:

1. **`em_gs_channel_rows(..., max_iter=T)` equals `T` chained `max_iter=1`
   calls, bit for bit** (`max|diff| = 0.0`). So writing the baseline as a chain
   of single steps costs nothing and lets the Hankel variant reuse the identical
   update.
2. **With the projection inactive, HS-GS returns EM-GS bit for bit**
   (`max|diff| = 0.0`). The projection is therefore the *only* difference.

`generate_channel_estimation_trial` returns one **frozen, read-only** object
holding `G, S, B, W, Z, sigma2`. Both estimators are handed that same object,
so channel, pilots, reference and noise cannot diverge between them.

## 6. What this package reuses vs. adds

Reused unchanged: channel, pilots, reference, forward model, EM-GS, Hankel
lifting, Cadzow, rank selection, RNG policy. **No algorithm is reimplemented.**

Added: a thin wrapper layer (`system_model.py`, `em_gs.py`,
`hankel_projection.py`, `hankel_em_gs.py`) that exposes the same code under
names that make the baseline/Hankel delta obvious; a checkpointed paired runner;
three experiment scripts; an independent verifier; plots.

`hankel_em_gs.hankel_em_gs` was checked to reproduce
`rydberg_sim.track_b_proposed.hs_gs_auto` **bit for bit** on a shared world.
