# Does Hankel low-rank structure improve EM-GS?

A self-contained, reproducible experiment package answering exactly one
question:

> Under the current Track-B sparse geometric ULA model, does adding a low-rank
> Hankel projection to EM-GS improve channel-estimation NMSE?

Nothing here concerns clustered channels, CRLB interpretation, or broader RAQR
claims. The scope is the question above.

---

## A. Files to inspect, in order

| # | File | What it is |
|---|---|---|
| 1 | `AUDIT.md` | Audit of the existing implementation, done **before** any new code. Answers where every component lives, whether Cadzow is interleaved or post-hoc, and how the rank is selected. |
| 2 | `config.py` | Every parameter, each **inherited** from the audited implementation with its provenance. Nothing here was tuned for this study. |
| 3 | `system_model.py` | Channel, AoAs, path gains, pilots, noise, magnitude-only observation. Thin wrapper over `rydberg_sim/`; no model is reimplemented. |
| 4 | `em_gs.py` | **Baseline only.** No Hankel processing appears in this file. |
| 5 | `hankel_projection.py` | Lifting, reconstruction, truncated SVD, anti-diagonal averaging, Cadzow, rank selection. The derivation of `rank(H) <= L` is in the module docstring. |
| 6 | `hankel_em_gs.py` | The Hankel estimator. Its docstring shows the baseline and the variant side by side; **the only difference is one projection line.** |
| 7 | `runner.py` | Paired Monte Carlo driver, checkpointed and resumable. |
| 8 | `experiment_snr.py` / `experiment_array_size.py` / `experiment_path_count.py` | Experiments A / B / C. |
| 9 | `verify_results.py` | Independently reloads the raw stores, recomputes every headline number, and runs 13 sanity checks. |
| 10 | `plot_results.py` | The four figures. Reads only `results/summary.json`; no number is hard-coded. |
| 11 | `ablations.py` | Schedule, oracle-rank and Cadzow-sweep variants, reported **separately** from the baseline. |

## B. Run commands

```bash
python experiment_snr.py          # A: NMSE vs SNR      (also B's N=8 column)
python experiment_path_count.py   # C: gain vs true L   (the mechanism test)
python experiment_array_size.py   # B: gain vs N        (adds N=16, N=32)
python verify_results.py          # tables + sanity checks -> results/*.csv, summary.json
python plot_results.py            # figures/ (reads the verified summary only)

./run_all.sh                      # all three sweeps, in the order above
python diagnostic_spectrum.py     # the singular-value diagnostic
python ablations.py               # ablations, reported separately
```

Every sweep checkpoints every 50 trials and resumes where it stopped, so all of
these are safe to interrupt and re-run. Environment overrides: `N_TRIALS`,
`N_TRIALS_LARGE`, `PROCS`.

## C. What makes the comparison fair

Fairness here is structural, not a convention that could silently break. Two
properties, both **executed** rather than asserted (checks 1 and 2):

1. **One frozen world per trial.** `generate_channel_estimation_trial` returns a
   single read-only object holding `G, S, B, W, Z, sigma2`. Both estimators are
   handed that same object, so channel, AoAs, path gains, pilots, reference and
   noise cannot differ between them. The arrays are non-writeable, so neither
   estimator can mutate the other's input.
2. **With the projection disabled, the Hankel estimator *is* the baseline,
   bit for bit** (`max|diff| = 0.0`). Both are built from the same
   `em_gs_step` function object, run for the same 50 iterations, from the same
   initialisation, with no early stopping in either. The projection is the only
   difference.

Additionally, each trial records a `paired_ok` flag re-deriving
`Z == |GS + B + W|` after both estimators have run, so a mutation would be
caught in the stores themselves.

## D. Design choices worth knowing before reading results

* **Cadzow is interleaved, not post-hoc.** `PROJECT_EVERY = 1`: the projection
  runs after every one of the 50 EM-GS updates. This is the audited behaviour
  and was not changed. Post-hoc is an ablation.
* **The rank is *not* an oracle.** `L_hat` comes from a held-out pilot residual
  using only `S, Z, B`. The true `L_k` never enters (check 5).
* **`L_hat` is one scalar shared by all K users**, even though their true orders
  differ. A known limitation of the audited implementation, preserved.
* **The projection has a hard ceiling.** `r_max = ceil(N/2)` = 4 / 8 / 16 for
  N = 8 / 16 / 32. At `L_hat >= r_max` the constraint is satisfied by every
  vector, the projection is a no-op, and the estimator degenerates to EM-GS.
  `active_frac` in the tables reports how often it was actually binding.
* **SNR grid.** The repository's established sweep is −5…20 dB. −10 dB is added
  because the brief asked for it; it is an extension, not a replacement.
* **Trial budget is tiered** (600 at N=8, 400 elsewhere) because cost is
  dominated by rank selection and scales ~8.5x from N=8 to N=32. Per-point
  trial counts and 95% paired-bootstrap CIs are reported for every point.

## E. Statistics

* NMSE is pooled as a **ratio of sums**, `10 log10( sum||Ghat-G||_F^2 /
  sum||G||_F^2 )` — never a mean of per-trial decibels (Jensen), and `10 log10`
  not `20 log10` because both arguments are already energies.
* Per-trial numerator and the shared denominator are stored **separately**, so
  any pooling, subset or bootstrap can be recomputed later without rerunning.
* The gain is `Delta_H = NMSE_EMGS,dB − NMSE_Hankel,dB`. **Positive means Hankel
  helps.** Its 95% CI is a *paired* bootstrap (2000 resamples): the resample
  index is drawn once and applied to both estimators, so the shared channel
  realisation cancels.
* `mean gain` (unweighted mean of the per-operating-point gains) and `max gain`
  (best single point) are reported as separate columns and never conflated.

## G. Results (10,800 paired trials, 13/13 checks passing)

Regenerate with `python verify_results.py`; the numbers below are copied from
its output and also live in `results/*.csv`.

**A -- NMSE vs SNR, N = 8** (600 trials/point). The projection **helps below
0 dB and hurts at and above it**, both with CIs excluding zero:

| SNR | EM-GS | Hankel | gain | 95% CI |
|---|---|---|---|---|
| -10 | 5.576 | 4.832 | **+0.744** | [+0.662, +0.829] |
| -5 | 0.463 | -0.255 | **+0.717** | [+0.624, +0.817] |
| 0 | -5.198 | -4.920 | **-0.278** | [-0.390, -0.174] |
| +5 | -10.625 | -10.222 | **-0.403** | [-0.521, -0.289] |
| +10 | -15.832 | -15.641 | -0.190 | [-0.277, -0.106] |
| +15 | -20.734 | -20.535 | -0.198 | [-0.298, -0.106] |
| +20 | -25.896 | -25.706 | -0.190 | [-0.389, -0.051] |

**B -- gain vs array size.** Mean and max are different quantities and are
reported separately:

| N | r_max | trials | mean gain | max gain (at SNR) | win rate | constraint active |
|---|---|---|---|---|---|---|
| 8 | 4 | 4200 | +0.03 dB | +0.74 dB (-10) | 33.4% | 54.1% |
| 16 | 8 | 2800 | +0.81 dB | +1.60 dB (-5) | 78.0% | 92.7% |
| 32 | 16 | 1400 | +2.45 dB | +3.09 dB (-5) | 96.3% | 99.4% |

**C -- gain vs true path count** (N = 32, r_max = 16, 300 trials/point). The
gain decays **strictly monotonically** to zero as L reaches the rank ceiling:

| L | gain | 95% CI | win rate | E[L_hat] | L_hat - L |
|---|---|---|---|---|---|
| 2 | +7.043 | [+6.735, +7.335] | 100.0% | 2.13 | +0.13 |
| 4 | +3.556 | [+3.379, +3.729] | 99.3% | 4.02 | +0.02 |
| 6 | +1.792 | [+1.636, +1.956] | 94.0% | 5.67 | -0.33 |
| 8 | +1.038 | [+0.927, +1.155] | 90.0% | 7.30 | -0.70 |
| 10 | +0.577 | [+0.455, +0.695] | 79.0% | 8.44 | -1.56 |
| 12 | +0.266 | [+0.191, +0.339] | 73.0% | 9.81 | -2.19 |
| 14 | +0.046 | [-0.050, +0.136] | 60.3% | 10.55 | -3.45 |
| 16 | -0.117 | [-0.206, -0.038] | 45.0% | 11.63 | -4.37 |

**Ablations** (N=32, L=4 fixed, 120 trials -- diagnostics, NOT replacements for
the baseline config):

| variant | NMSE dB | gain dB |
|---|---|---|
| EM-GS baseline | -10.646 | 0.000 |
| **Hankel, interleaved (BASELINE CONFIG)** | **-14.355** | **+3.709** |
| Hankel, post-hoc (project once at end) | -14.067 | +3.421 |
| Hankel, ORACLE rank L_hat = L (not deployable) | -14.723 | +4.077 |
| Hankel, 1 Cadzow sweep | -14.647 | +4.001 |
| Hankel, 8 Cadzow sweeps | -14.319 | +3.673 |

Two of these say the inherited defaults are **not** optimal at this operating
point: a single Cadzow sweep beats the audited 4 by 0.29 dB, and oracle rank
would buy 0.37 dB over the held-out selector. Neither was adopted -- the
baseline is what the audit found, and changing it after seeing these numbers is
exactly the tuning this study forbids. They are recorded as leads.

## H. Conclusion

Under the current sparse geometric ULA Track-B model, adding a low-rank Hankel
projection to EM-GS improves NMSE **conditionally, not generally**:

* it **helps when the channel is sparse relative to the array**, growing with
  array size (+0.03 / +0.81 / +2.45 dB mean at N = 8 / 16 / 32) and shrinking
  with path count (+7.04 dB at L = 2 down to zero at L = 14);
* it **hurts at N = 8 for SNR >= 0 dB** (-0.19 to -0.40 dB, CIs excluding
  zero), where r_max = 4 is comparable to L_k in {3..7} so the projection is
  either inactive or truncating genuine channel components;
* it **turns slightly negative at L = r_max** (-0.117 dB, CI [-0.206, -0.038]),
  where the constraint is vacuous and the selector under-selects by 4.4.

The path-count sweep is the load-bearing evidence: the gain vanishes precisely
where the low-rank structure does, which is what rules out generic denoising as
the explanation. Both baselines are flat in L over the same sweep, confirming
the sweep is not confounded.

Scope: this is a statement about the tested i.i.d.-uniform-AoA, equal-per-path-
power geometric model only. Nothing here speaks to clustered propagation.

## F. What was NOT done

No threshold, sweep count, initialisation, operating point, path distribution
or trial filter was changed after seeing a result. Where the Hankel projection
hurts, the tables and figures show it.
