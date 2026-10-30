# Preflight checks

23 September 2026. All checks listed here passed before study launch.

- 31 existing tests across the numerical, pipeline and matched-control suites passed. These cover Rician density/derivatives, complex noise variance, EM updates, small and repeated singular values, float32/float64 comparison, observation-only training, held-out isolation, finite gradients and checkpoint recovery.
- Five new feature-check tests passed: full-model forward/backward equivalence; masked-input invisibility; deterministic optimizer restart; disjoint pilot subsets and finite gradients; and a tiny isolated training → selection → fresh evaluation → saved-output audit → report test. Temporary smoke-test results are not research outcomes.
- Training is identical across feature arms except access to the declared controller inputs. The full-input wrapper reproduces the old architecture's outputs and parameter gradients exactly in the unit test, including nonconstant controller weights.
- Protected historical code/data/checkpoint hashes were checked during initialization. No old model source or result was edited.
- Measured training-batch time is about 0.18 seconds on one CPU thread for 32 examples. Measured validation-batch time is about 0.069 seconds. Estimated training plus validation is about 65 minutes; an allowance for evaluation, startup and variability brings the estimate to about 87 minutes. These are estimates, not a promised completion time.

The runner is capped at two hours and saves complete epochs. The Mac must remain on and awake for local computation to advance. This launch does not move training to a remote machine or update the old dashboard.

Live progress is recorded in `status.json`, `run.log` and each job's `history.json`. After completion, `audit.json`, `summary.json` and `REPORT.md` must be inspected together before interpreting results.
