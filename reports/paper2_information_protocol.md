# Paper 2 — frozen first feasibility protocol

Date: 21 September 2026. Stage: implementation and development screening, not confirmation.

## Question and limits

Does direction-dependent measurement information improve a smooth structural correction beyond making the same correction smaller with a scalar?

This is a candidate mechanism, not verified novelty. Nothing here revives recalibration, changes the manuscript, modifies the old experiments, or authorizes a large training queue. The first worker waits for the 40-run confirmation study to finish, then audits it. A failed audit, stopped old process or exhausted old budget blocks the new study. The old study is not automatically restarted.

The local compute allowance is at most two hours after the dependency finishes. This conservatively includes the old-study audit, tuning, screening, and diagnostics. Each individual estimate can finish before a budget check; ordinary estimates take seconds or less. Waiting for the old study does not consume this compute allowance. The allowance is cumulative across ordinary restarts. Case/method records are atomic, deterministic and resumable. A budget stop is not a scientific failure. The worker does not extend itself or launch learning after a pass.

## System model

Z = |H S + B + w|, with independent circular complex Gaussian noise and E|w|² = v. N=32 antennas, K=4 users, P=20 pilot slots. S uses the existing full-row-rank complex Gaussian pilot generator (unit expected entry power), with new seeds. The known reference has real positive amplitude corresponding to ten times expected received signal power. Expected channel coefficient power is 32; normalization uses its declared square root, not an observed true-channel norm. Reference and noise variance are known. No hardware realism is asserted by this simulation.

Families are sparse geometric channels with 3–7 paths per user, four clusters of ten rays within ±5 degrees of cluster centers, and 40 independent rays per user. Steering uses half-wavelength spacing. All families have the same expected coefficient power. The sparse case has four users here and is NOT the old three-user experiment.

For each family and SNR in {-5, 5, 15, 25} dB, generate 16 tuning cases and 64 screening cases. These are **192 tuning and 768 screening cases**, not training examples. Tuning and screening use disjoint random streams and three separate pilot matrices each. Paired methods see exactly the same channel, noise, pilots and initialization within each case. Multiple pilot matrices are a development check, not five independent confirmation replicates.

## Estimators

All operation ablations start from receiver-only linearized least squares and use the exact Rician EM-GS update. Inputs to the public estimator are only observations, pilots, reference, known noise variance and a configuration. No family, true rank, true channel or true case error is available. Ground truth is used by the evaluator for tuning/reporting and explicitly labelled oracle diagnostics only.

The initial ablation uses ten interleaved classical/structural steps. Compare:

- EM-GS with 1, 3, 5, 10, 20, 50 and 100 fixed iterations.
- Ridge-regularized EM-GS with 10 or 50 iterations and ridge 0.01, 0.1 or 1. Ridge is the normalized latent least-squares penalty; its equivalent channel-prior precision is ridge/v. It is not claimed to be a globally optimal MAP solver.
- One rank-seven Hankel projection per iteration, at 10 and 50 iterations.
- Smooth Hankel correction with lambda in {0.1, 0.3, 1, 3} and scalar multiplier in {0.05, 0.2, 0.5, 1}.
- The same smooth correction filtered by (I + tau J)^(-1), tau in {0.03, 0.1, 1, 10}, using the current estimate's real-coordinate Rician Fisher information.
- An isotropic-information control: replace each antenna's information block by its mean eigenvalue times identity. This controls for attenuation strength informed by measurements but without directional geometry.

The scalar, information, and isotropic searches each contain 16 candidates plus the exact no-correction control. Parameter selection is independent per receiver-available noise bin, across all three families with equal weight. The estimator uses expected signal power and known variance to select the nearest noise bin; it never receives the true channel power or family label. This gives the scalar control an SNR-dependent setting, not only a weak global multiplier.

Selection minimizes the equal-weight mean of per-family dB mean-linear NMSE within each tuning noise bin. Freeze the selection file before generating screening results. No screening-driven parameter changes. All candidate errors are retained; numerical failures stop the worker rather than being dropped.

Screening includes selected classical, scalar, directional and isotropic rules; EM-GS 10/50/100; hard rank-seven 10/50; post-processing-only versions of the selected scalar and information rules; and **native Paper 1 adaptive-rank/abstention**, with its own spectral initializer, 25 selection iterations, 50 final iterations and four Cadzow sweeps. Label the native initializer difference. Its internal held-out pilots are part of the same P=20 budget. Post-processing controls use the preselected interleaved parameters and are diagnostic controls, not independently optimized methods.

The strongest learned estimator is **not yet evaluated on these fresh worlds**. It belongs to the next benchmark extension before any publication claim; the old-study audit reports it under its original protocol. This first screen tests the mechanism, not a new learned state of the art.

## Numerical definitions and checks

Hankel embedding is 16x17 per user. Soft shrinkage is X - lambda²(XX*+lambda²I)^(-1)X, followed by anti-diagonal averaging. This is exactly the singular rule s³/(s²+lambda²), not an identity-gradient approximation. Lambda=0 is an exact identity branch.

Coordinates for each antenna are [Re(h), Im(h)] in normalized units. Each magnitude observation supplies a rank-one real matrix: I_a g gᵀ. With field u and c=exp(-i arg(u))S, g=[Re(c), -Im(c)]. I_a is expected squared Rician amplitude score. A precomputed dimensionless quadrature table approximates this scalar, tested against finer quadrature and simulated score covariance. It is **not** the repository's older complex-information shortcut. The table has 1,025 log-spaced points and 160 integration nodes; the preflight checks interpolation tolerance.

Tests cover independent likelihood values and derivatives; score covariance; one-observation rank; gradient signs; exact no-correction; EM-GS agreement with the native implementation; SVD equivalence of smooth solves; zero/small/repeated singular values; finite autograd derivatives; normalized unit scaling; high-SNR finite outputs; deterministic split generation; restart behavior; selection isolation; locks and budget guards. CUDA agreement is explicitly skipped unless CUDA is available. No GPU training is requested or launched.

Separate diagnostics compare estimated and oracle information, zero initialization, nearly noiseless (80 dB) observations, and changed/repeated pilots at fixed channel and nominal SNR. Only these controlled geometry diagnostics rescale different pilots to match total energy; the primary Gaussian pilot draws are unmodified. The repeated-pilot diagnostic uses an explicitly labelled tiny ridge for singular solves. Oracle diagnostics cannot be used to pass the screen. P=12/32 and broad mismatch tests are not part of this first batch.

## Decision and metrics

Positive gain means the comparator has larger error. Per cell report 10log10(mean per-case NMSE), severe-error frequency NMSE>1, and full estimator time including initialization. The primary macro-average equally weights all 12 cells. Also report every classical/structural comparator, every cell, family-average degradation and worst-cell degradation.

Advance to scientific review only if the implementable directional rule beats the selected scalar rule by at least 0.3 dB on the screening macro-average and worsens no family average by over 0.2 dB. Do not select between methods using screening truth. A pass is not enough to launch neural learning: inspect the isotropic control, Paper 1 comparison, costs, diagnostics and novelty overlap first. A failure stops expansion of A; any backup idea needs a separate small protocol.

These are development thresholds, not confirmation confidence intervals or publication standards. The 4–6 week confirmation criteria in the user's plan remain unchanged: fresh independent datasets/pilots, strong classical and learned comparators, predefined multiplicity-adjusted uncertainty and per-cell protection.

## Commands and outputs

Run from the repository with the existing `.venv-paper2` Python:

```
../../.venv-paper2/bin/python -m paper2_information.experiment preflight
../../.venv-paper2/bin/python -m paper2_information.experiment launch
../../.venv-paper2/bin/python -m paper2_information.experiment status
```

The new output folder is `results/paper2_information_feasibility`. It contains source/configuration manifest, preflight evidence, the independent legacy audit, per-case predictions/errors/diagnostics, frozen selection, budget, status, summary and plain-language report. An interrupted partial stage can resume with the same launch command, provided source hashes still match. A scientific or numerical blocker is not automatically retried. No dashboard or manuscript changes are part of this package.
