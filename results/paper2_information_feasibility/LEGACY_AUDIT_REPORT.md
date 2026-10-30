# Completed 40-run study: independent numerical audit

Audit passed: True. Verified counts: {'training_records': 40, 'epoch_histories': 720, 'test_files': 94, 'data_hashes': 64, 'checkpoint_replays': 16, 'data_regeneration_cases': 24}.

Positive gain means Hankel reduced error. Gains below average paired per-training-seed dB mean-NMSE differences, not per-case medians.

| Channel | Checkpoint / objective | Full-range gain (dB) | SNR ≥5 dB gain (dB) |
|---|---|---:|---:|
| literal_independent | selected/standard | -1.194 | -5.129 |
| literal_independent | selected/weighted | -1.261 | -5.384 |
| literal_independent | final/standard | -1.100 | -5.033 |
| literal_independent | final/weighted | -1.220 | -5.407 |
| clustered_5deg | selected/standard | 0.347 | 1.383 |
| clustered_5deg | selected/weighted | 0.330 | 1.614 |
| clustered_5deg | final/standard | 0.334 | 1.176 |
| clustered_5deg | final/weighted | 0.332 | 1.232 |

## Classical controls and the declared screen

- literal_independent: declared screen passed = False; selected learned Hankel versus the validation-selected classical comparator at high SNR = 2.372 dB.
- clustered_5deg: declared screen passed = False; selected learned Hankel versus the validation-selected classical comparator at high SNR = -0.237 dB.

Read legacy_audit.json for every bin, severe-error frequency, timing, selected/final results and seed intervals. Read the old study’s summary.json for its conditional case bootstrap outputs.

Checkpoint replay is representative (16 x 32 cases), not every checkpoint on every test case. Seed intervals do not include dataset/pilot uncertainty. Numerical audit is not a novelty or publication verdict.

Five training seeds share the same dataset and pilot matrix. Their intervals do not establish replication across channel populations. The comparison is an independently implemented classical-start learned estimator, not certified reproduction of published URformer. Rank seven and the identity-backward approximation remain features of this old protocol.

This audit checks numerical consistency; it does not establish a universal Hankel advantage, explain every failure causally, or certify novelty. The new feasibility work uses fresh development worlds and separate success rules. No historical output or manuscript has been edited.
