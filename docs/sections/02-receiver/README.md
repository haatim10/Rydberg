# Complete receiver, improvements and baselines

The current complete receiver is implemented in `tcom_reliability_20261021/`.
It builds on earlier frozen modules rather than copying their code into a new
directory. Those dependencies are included in this branch and in `codex`.

## Receiver operation

1. Use the adaptive learned estimator through nominal 30 dB and the fixed
   trained schedule above that point.
2. Above 20 dB, apply the stored policy for extra EM updates and final filtering.
3. In the specified weak-reference/high-SNR region, check each antenna's
   magnitude residual. A triggered check can try an algebraic starting estimate.
4. Refine a valid alternative with EM and the final filter. Accept it only when
   its fitting Rician loss improves.

All online inputs are measured pilots, the known reference, supplied noise
variance and stored model/policy settings. The fitting-loss acceptance rule
does not guarantee lower true-channel error or lower BER.

## Sequence of experiments

| Campaign | Question addressed |
| --- | --- |
| `tcom_followups_20261017` | Sparse comparison, common detector and high-SNR range |
| `tcom_runtime_bypass_20261017` | Runtime-matched sparse baseline and filtering bypass |
| `tcom_pilot_em_20261018` | Pilot-matched retraining and short EM continuation |
| `tcom_retrained_bypass_20261018` | Bypass using the retrained weights |
| `tcom_transition_repair_20261019` | Weak-reference transition failures |
| `tcom_dual_guard_20261019`, `tcom_incumbent_guard_20261019` | Preserve earlier gains while correcting transition cases |
| `tcom_filter_continuation_20261019` | Remove the strong-reference 40 dB filter switch-off jump |
| `tcom_runtime_repair_20261020` | Reduce runtime after the repairs |
| `tcom_start_ablation_20261020`, `tcom_start_budget_20261020` | Learned, fixed and classical starts with matched processing |
| `tcom_hybrid_20261020`, `tcom_hybrid_sensitivity_20261020` | Combine adaptive and fixed schedules and check sensitivity |
| `tcom_reliability_20261021` | Residual-triggered recovery and final full-grid comparison |

Each campaign has code at the repository root and evidence under `results/`.
The [catalog](../../EXPERIMENT_CATALOG.md) supplies direct report links.

## Read the final evidence

- [Final report](../../../results/tcom_reliability_20261021/RESULTS.txt)
- [Protocol](../../../results/tcom_reliability_20261021/protocol.json)
- [Stored policy](../../../results/tcom_reliability_20261021/policy.json)
- [Completion record](../../../results/tcom_reliability_20261021/completion.json)
- [Inference code](../../../tcom_reliability_20261021/inference.py)

The final study covers 153 settings: 3 reference strengths, 3 path regimes and
17 SNR values. Its main configuration is 32 antennas, 3 users and 30 fitting
pilots. NMSE and BER use paired observations; timing includes the complete
estimation call, including any triggered recovery.

The reported full-grid p90 is 7.254 ms in the slower timing window, below the
10.74 ms target. This is a **p90 target**, not a guarantee for every call; the
reported maximum is higher. Model loading and data detection are not channel
estimator runtime. See the frozen timing records for exact boundaries.

The proposed receiver improves mean-setting NMSE in dB, but the compact sparse/EM
baseline has better pooled linear NMSE and aggregate BER in the final table.
Different aggregations answer different questions; do not describe this as a
win over every baseline on every metric.

Known reference phase and noise variance are assumed. The algebraic recovery
requires its design-rank check; constant-magnitude QPSK fitting pilots do not
satisfy the particular 15-column design used here. This is a simulation study,
not a hardware or coded-link validation.

The uncertainty-based filter discussed after these experiments is **not part of
this implementation**. No new estimator behaviour was added during archiving.
