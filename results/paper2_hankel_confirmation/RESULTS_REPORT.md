# Large Hankel study: automated results report

All results are from the frozen independent simulation protocol, not an author-code reproduction.
Completed 40 runs and 720 epochs. Matched parameter/order checks passed.
Selected and final validation checkpoints replayed exactly before test access.

## Primary results

| Population | Selected loss | High-SNR gain (dB) | Adjusted seed interval | Screen |
|---|---|---:|---|---|
| literal_independent | standard | -5.129 | [-5.20874181344241, -5.049716018242057] | False |
| clustered_5deg | standard | 1.383 | [0.8139786826213611, 1.9524593691497711] | False |

A passing screen is not a publication or novelty guarantee. A failed screen does not prove equivalence.
Inspect summary.json for all bins, both losses, final checkpoints, severe errors and every classical control.
Five seeds vary training randomness on shared data, not five independently sampled datasets. Case bootstrap intervals condition on the trained models.
Training/validation curves must still be reviewed for continued trends and selection sensitivity; completing 18 epochs does not establish convergence.
Inference timings include native Paper1 rank selection; architectures and numerical precision differ. No universal speed or hardware claim is justified.
This automated report and the saved artifacts require a final human-facing scientific review before any manuscript change.
