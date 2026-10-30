# Paper 2 evidence ledger

Updated 21 September 2026. This is a synthesis of saved reports and code, not a fresh replay of every historical experiment. Historical files remain unchanged. The new 40-run audit is deliberately separate and can run only after the existing study completes.

## Findings to retain, with their limits

| Evidence | Status | Defensible statement | Do not infer |
|---|---|---|---|
| `reports/trackD_stage2_report.md` | Historical development result | More distinct training examples improved the ordinary estimator under approximately matched sample-presentation budgets. | A fully optimized baseline, proof of an architectural ceiling, or causal identification of every earlier failure. Check metric sign conventions before reusing that report's confidence intervals: its table presents gain and intervals with opposite signs. |
| `reports/trackD_stage3_report.md` | Historical development result | In-loop and post-processing Hankel comparisons showed metric-dependent outcomes. Earlier-layer straight-through gradients poorly matched projection derivatives. | The approximate gradient necessarily caused observed error, or a positive median gain implies lower mean-linear NMSE. |
| `reports/trackD_stage4_report.md` | Failed protection screen; useful training diagnostic | Scalar and SNR gates reduced damage but missed their protection criterion. Focused high-SNR training reduced the paired-median Hankel gap from about 1.21 to 0.08 dB in that experiment. | All structural gains are explained by undertraining. Assertions such as “the gain was never the prior's” and “gating is the right architecture” overstate the evidence and must not be copied into a paper. |
| `results/paper2_long_training/RESULTS_REPORT.md` | Completed, audited historical comparison | On the sparse K=3 setup, selected-checkpoint high-SNR paired-median gain was 1.136 dB at 13 epochs and 1.256 dB at 52; final-checkpoint gain fell from 1.104 to 0.717 dB. | Convergence, a duration-only effect, or independent channel-population replication. Cosine-schedule duration changed too; three training seeds share data and pilots. |
| Same longer-training report | Metric caveat | At 52 epochs, selected high-SNR mean-normalized-error gain was about 0.850 dB, not 1.256 dB; full-range normalized-error gain was negative. | A full-range error improvement from a positive paired median. |
| `results/paper2_snr_switch/RESULTS_REPORT.md` | Diagnostic, limited value | Switching between two models preserved conditional benefits but gave only about 0.05 dB full-range mean-NMSE improvement with extra storage/processing. | A compelling standalone novel contribution or robustness with unknown noise level. |
| `results/paper2_original_channel/RESULTS_REPORT.md` | Completed historical comparison | In K=4/40-ray simulations, selected high-SNR paired-median gain was -4.388 dB for independent rays and -0.184 dB for clustered rays; final clustered gain was +0.821 dB. | A universal SNR threshold for Hankel usefulness, or equivalence of these channels to the old K=3 sparse setup. |
| Same original-channel report | Plausible failure mechanism | Noiseless rank-seven projection itself distorted independent-ray channels more than clustered channels. | A lower bound on final learned-model error or complete causal proof; the learned correction can restore components. |
| `results/paper2_baseline_learning/RESULTS_REPORT.md` | One-seed failed normalization screen | Fixed normalization did not resolve the discrepancy; near-noiseless classical recovery substantially exceeded small-data neural recovery. | The published paper is wrong, or every optimization setting has been ruled out. |
| Classical-start and `results/paper2_loss_weighting/RESULTS_REPORT.md` | Partial improvement, failed development targets | Exact classical initialization and high-SNR weighting improved some bins, but weighted final accuracy still fell below initial EM-GS10 at high SNR and failed both screens. | Reproduction of published URformer. The modified classical-start learned estimator is independently implemented and differently parameterized. |
| `results/paper2_hankel_confirmation` | Training finished; test classical evaluation still active when this ledger was written | Forty training records exist. Reported validation gains are provisional development evidence, not final test outcomes. | Replacing unfinished test outcomes with validation scores or presenting five training seeds as five datasets. |
| `paper2_information` | New implementation, no scientific outcome yet | A real-coordinate Rician-information filter, smooth Hankel solves, matched controls, numerical checks and a budgeted screen are implemented. | Novelty, superiority, convergence, hardware performance, or any positive scientific result before the screen runs. |

## Open problems and how this implementation handles them

1. **Published-baseline discrepancy:** remains unresolved. Keep the reconstructed and classical-start methods named separately. No claim against the authors' correctness.
2. **Mixed protocols:** keep K=3 and K=4, sparse and rich, selected and final checkpoints, paired medians and mean-linear errors in separate columns. Do not pool them into one headline.
3. **Rigid rank and approximate backward pass:** the new soft operation does not impose rank seven and uses matrix-solve derivatives. Hard rank seven remains a diagnostic control; this does not retrospectively attribute old failures to gradients.
4. **Weak classical controls:** new screen adds noise-bin selected stopping and ridge penalties, keeps fixed iteration counts, and includes native Paper 1. All receiver rules are tuned without screening truth.
5. **Strong learned comparator:** still required on the fresh protocol before confirmation. Existing learned outputs only support their own protocol. First test the nonlearned mechanism rather than train another matrix.
6. **Measurement geometry versus scalar strength:** compare directional filtering with both an SNR-dependent scalar rule and an isotropic-information rule. A directional result that merely matches these controls is not the proposed contribution.
7. **Dataset/pilot uncertainty:** the small screen uses fresh splits and several pilots, but the future confirmation must use independent dataset/pilot replicates and separately report training randomness.
8. **Reality gap:** reference and variance remain known. Later ±3 dB noise-input errors, reference perturbations and unseen pilots are simulation stress tests, not measured receiver drift. Recalibration remains cancelled.
9. **Novelty:** the accompanying novelty matrix records substantial overlap. The resolvent formula and one-step bound are ordinary mathematical tools, not a new theorem by themselves.

## Audit outputs and publication gate

The new worker writes `results/paper2_information_feasibility/legacy_audit.json` only after the 40-run study reports completion and releases its run lock. It checks all 40 records, 720 epochs, 94 test outputs, data/source hashes, trial alignment, regenerated example inputs, training permutations, validation selection, headline error statistics and representative checkpoint replay. Audit failure blocks the feasibility batch. Representative replay is not a full forward replay of all checkpoints.

All already inspected data are development evidence for the new method. A feasibility pass permits review and further testing; it does not justify a publication claim. Write a qualitative results-based abstract only after independent confirmation.
