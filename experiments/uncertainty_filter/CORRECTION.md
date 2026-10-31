# Correction: relaxed-output uncertainty calibration

**Result: the filter is still not adopted.** This correction fixes the target of
the Phase-1 calibration with the already saved raw records. It does not train a
filter, select a setting, or create new test draws.

## What was wrong

At receiver stage `t`, the posterior calculation first gives the full EM update
`G_EM`. The receiver does not filter that value directly. It filters

    G_dc = G + gamma_t (G_EM - G).

With the current iterate held fixed, the conditional covariance of `G_dc` is
`gamma_t^2` times the conditional covariance of `G_EM`. The original Phase-1
records stored the latter but scored error of the former. It also put a final
diagnostic row in the same correlation: that row is the already-produced input to
the final filter, while a fresh posterior covariance predicts a *next* EM step.
Those are not the same random variable.

## What was changed

- `stages.py` now records each frozen `gamma_t`.
- A new calibration run will multiply both the posterior proxy and the nominal
  EM proxy by `gamma_t^2` for stages 1–10.
- `relaxed_audit.py` applies the same exact frozen factors to the saved original
  arrays, then writes a separate result folder. It excludes stage 11 rather than
  inventing a variance estimate for its existing final-filter input.
- The original `results/phase1/` arrays and summary are preserved unchanged.

This is a post-hoc correctness audit, not a new selection experiment. Its purpose
is to check whether the original negative result survives the correct target.

## Corrected result

The corrected gate uses stages 1–10, 120 cells, 128 channels per cell, three
receiver seeds, three users, and both starts: 2,764,800 analysed records, or
1,382,400 per start. The original stored data contain 3,041,280 records because
they also include the stage-11 diagnostic row.

| Group | Posterior proxy rho | Nominal proxy rho | Gain [95% interval] |
| --- | ---: | ---: | ---: |
| Main gate: receiver start, RSR 5/12/20 | 0.6072 | 0.5902 | +0.0169 [+0.0165, +0.0174] |
| Poor start, RSR 5/12/20 | 0.5969 | 0.5750 | +0.0219 [+0.0214, +0.0225] |
| Weak reference, receiver start | 0.3064 | 0.2761 | +0.0303 [+0.0290, +0.0316] |
| Weak reference, poor start | 0.1294 | 0.0831 | +0.0463 [+0.0446, +0.0480] |
| Stage 1 | 0.2961 | 0.2665 | +0.0296 [+0.0286, +0.0306] |
| Stage 5 | 0.7388 | 0.7315 | +0.0073 [+0.0067, +0.0079] |
| Stage 10 | 0.9447 | 0.9414 | +0.0033 [+0.0030, +0.0036] |

The pre-registered gate requires a main gain of at least +0.05 and no group more
than 0.02 worse. The correction has no worse group, but its main gain is only
0.0169. **Gate 1 still fails.**

The full machine-readable record, exact gamma values, bootstrap settings, and
source provenance are in `results/phase1_relaxed_audit/summary.json`.

## Decision

The correction makes the audit technically sound, but does not make the proposed
uncertainty filter promising enough to justify a larger experiment. The sensible
outcome remains: keep the frozen receiver, do not add this filter to the paper's
main method, and retain this as a documented negative result if needed.
