# Primary method and evidence decision

Date: 22 September 2026. This records how the completed evidence will be presented. It is not a retroactive preregistration.

Retain the original mixed-condition study's validation-selected EM core, ten stages, 188 parameters, learned relative threshold and learned gate. Retain all three original validation-selected checkpoints. Do not select the best test seed or replace EM with the GD core because its test average is higher.

| Seed | Selected epoch | Checkpoint relative to repository | SHA-256 |
|---:|---:|---|---|
| 9101 | 18 | `results/sahure_expansion/checkpoints/seed_9101/em_adaptive_best.pt` | `b664708d4b04dd68be7d503199163f42d027aa7a2afc5b110a70e2474dee9b12` |
| 9102 | 19 | `results/sahure_expansion/checkpoints/seed_9102/em_adaptive_best.pt` | `afe564f3714b536574f53fb1de56446f16a753dc57b0caf99547f0ad78305648` |
| 9103 | 20 | `results/sahure_expansion/checkpoints/seed_9103/em_adaptive_best.pt` | `2f78fd5c2f7c7eca82a9a6e7d017b6722e4ebb55074bedeac670ff670c3f8525` |

Authoritative source/configuration hashes remain in `results/sahure_checks/protocol.json`; model selections remain in `results/sahure_checks/frozen_selection.json` and the original expansion selection. The review verified these records without changing them.

Gate-one and final-only models remain ablations. Gate saturation is reported openly and is not called learned structural abstention. Independent-dataset models are replication evidence, not substitutes for the original primary selection. Metrics average per-seed scores; predictions are not ensembled.

The separate draft does not replace the original manuscript. Abstract and conclusion are deferred until the contribution and outstanding checks are settled. No new training was launched as part of this review.
