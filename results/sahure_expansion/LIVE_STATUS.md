# Broader Hankel experiment — progress

Updated: 2026-09-21 18:19 UTC.
State: **complete**. Phase: **evaluate**.

18 models; 20 epochs each; 16,384 training and 2,048 validation worlds. Full test results are generated only after model selection is frozen.

| Model | Seed 9101 | Seed 9102 | Seed 9103 |
|---|---:|---:|---:|
| em_none | 20/20 | 20/20 | 20/20 |
| em_global | 20/20 | 20/20 | 20/20 |
| em_adaptive | 20/20 | 20/20 | 20/20 |
| gd_none | 20/20 | 20/20 | 20/20 |
| gd_global | 20/20 | 20/20 | 20/20 |
| gd_adaptive | 20/20 | 20/20 | 20/20 |

Completed training epochs: **360/360**.

Three independent training runs share the same training/validation worlds. Checkpoint selection uses equally weighted held-out measurement likelihood across 126 conditions. Current validation scores are not channel-estimation test results.

After training: 32 operating conditions, 400 paired trials each, followed by serial timing and a provenance audit. Duplicate reference conditions and the two noise-variance offsets reuse the appropriate worlds.

The full pipeline and experiment settings are in `sahure_expansion/README.md`. The completed pilot is preserved in `results/sahure_pilot/`.

[Read the results report](/Users/haatim/.codex/.chatgpt-projects/g-p-6aa7dfe2b65c8191b548f107c7ddcd75/tmp/Rydberg/results/sahure_expansion/REPORT.md)
