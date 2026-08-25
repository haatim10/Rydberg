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

## F. What was NOT done

No threshold, sweep count, initialisation, operating point, path distribution
or trial filter was changed after seeing a result. Where the Hankel projection
hurts, the tables and figures show it.
