# Implementation handoff and remaining gates

## Implemented now

- Isolated NumPy estimator with exact Rician EM updates, soft Hankel matrix solves and real-coordinate information filtering.
- Scalar, isotropic-information, hard-rank, fixed/noise-selected classical and native Paper 1 comparisons.
- Fresh paired simulation splits, atomic saved predictions, frozen selection and source hashes.
- Numerical, gradient, split, restart, budget and dependency-lock tests.
- Read-only 40-run completion audit; the worker waits for the old study first.
- A two-hour cumulative local compute cap, partial results and explicit failure states.
- Evidence ledger, initial novelty matrix, one-correction theory note and frozen feasibility protocol.

This is the first gated stage of the approved plan. It is not completion of a six-week research project.

## After the worker finishes

1. Review `legacy_audit.json` and all completed-study bins. Correct the interpretation, never alter frozen outputs. Compare selected and final checkpoints and classical costs separately. Keep any newly found problem visible.
2. Review the new screen's primary criteria, worst cells, error tails, timings, isotropic comparator, Paper 1 results and oracle/poor-initialization diagnostics. A pass based only on the oracle is impossible under the implemented rule. A small scalar advantage is not automatically a novelty result.
3. If the screen fails, record the failure and stop expanding A. Do not train a larger network or change the threshold. The backup decomposition needs a separate declared feasibility study.
4. If the screen passes and the geometry mechanism remains credible, extend the read-only benchmark to the best saved independently implemented learned estimator on these fresh development worlds; add stronger regularized-likelihood optimization if current classical controls remain weak. Independently tune post-processing controls before claiming an interleaving advantage.
5. Complete the closest-prior-art matrix and compare the actual final Paper 1 wording. Only then decide whether a small learned controller addresses a remaining problem.

## Week 3 gate: small learning, not a new large queue

Confirm GPU access, memory, limits and cost first. On CPU, benchmark a tiny shared controller before quoting a runtime. It may observe only current estimates, residuals, pilots/reference, known noise and iteration. Keep exact abstention. No true-channel/rank/family inputs. Check gradients for the entire differentiable path, not only the soft Hankel helper. Current software does not claim that full learned path is implemented.

Use matched training examples and data orders, matching search budgets, and a predeclared equal-cell dB checkpoint objective. Do not impose this new objective on historical results. Require improvement over nonlearned tuning before a training expansion.

## Weeks 4–5 gate: independent confirmation

Freeze all methods, stopping rules, selection, costs and success criteria before new final data. Keep the approved three families, seven SNR values, P=20, five independent dataset/pilot replicates, two initializations per replicate where learning exists, and at least 200 new test cases per cell/replicate. Count shared test cases correctly. Report dataset/pilot uncertainty separately from training variation.

Keep the approved target: at least 0.5 dB macro gain over preselected strong classical AND learned controls with positive multiplicity-adjusted intervals; simultaneous upper degradation bounds below 0.25 dB in every primary cell. State the precise adjusted interval construction in a separate frozen confirmation protocol. Do not reinterpret failure as speed success; a speed claim needs a declared accuracy–cost experiment.

Stress one factor at a time: P=12/32, unseen pilots, angular spreads/mixed scattering, noise input ±3 dB, reference perturbations and nearly noiseless observations. All reference perturbations are robustness stress tests, not a revived recalibration algorithm. Do not run the full Cartesian product.

## Writing and honest limits

System model and related-work notes can be drafted now. Write the method only after feasibility review, results after confirmation, and the qualitative abstract and conclusion last. Use simple language, no forward references, one background paragraph, one motivation paragraph, then contributions. No manuscript edits are made by this package.

Neither a local correction bound nor a completed training schedule proves convergence. Simulations with known receiver parameters are not hardware validation. A research contribution can fail; the stopping gates are part of the implementation rather than obstacles to work around.
