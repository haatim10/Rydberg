# PROMPT 15 Part A — why evaluation did not reproduce itself

**Gate status: PASSED.** Evaluation on a given checkpoint is bitwise
reproducible, a regression test pins it, and the published stage-3 and stage-4
rows reproduce from the committed checkpoints.

**But the answer is not the one the brief expected, and it is worse.** The
cause is not nondeterminism. It is a wrong initialiser in my own PROMPT 12
Part C scripts — first in evaluation, and then, on further checking, in
training. PROMPT 12 Part C is invalid and cannot be repaired by re-scoring.

---

## A1. Diagnosis

### The six candidate causes are all ruled out, by measurement

`scratch/p15_a1_probe.py`, 100 trials, two stage-4 arms:

| Test | Result |
|---|---|
| `stage4.evaluate()` twice in one process | **bitwise identical**, 0/100 differ |
| `p12_partC_eval.eval_models()` twice | **bitwise identical**, 0/100 |
| The two code paths against each other | **bitwise identical**, per trial |
| Stored test-set SNR draws vs today's | **bitwise identical**, all 2000 |

So test worlds are not redrawn, SNR is not resampled, pairing is intact, and
there is no GPU — the container is CPU-only and reproduces regardless.

Also ruled out by direct check rather than assertion:

- **Code drift.** `git diff 910ec42..HEAD` over `trackD_urformer/` and
  `rydberg_sim/` is **empty**. The model, dataset, config and forward are
  byte-identical to when the checkpoints were committed.
- **Environment drift.** `requirements.txt` pins numpy 2.4.6, scipy 1.17.1,
  torch 2.13.0, and the container has exactly those.
- **Partial checkpoint loading.** All 480 `state_dict` keys are present and
  overwritten on load; no non-persistent buffers, no unregistered parameters.
  The file fully determines the model.
- **A stale checkpoint or wrong epoch.** `scratch/p15_a1_which_epoch.py` loads
  each surviving `ep000.pt … ep012.pt` and compares against the stored rows:

  ```
  C_U1_snr5_20: best.pt records epoch 9
     ep008  max|rel|=1.114e-01
     ep009  max|rel|=0.000e+00   <== MATCH
     ep010  max|rel|=8.119e-02
  C_H1_snr5_20: best.pt records epoch 9
     ep009  max|rel|=0.000e+00   <== MATCH
  ```

  Both arms reproduce **bitwise** from the committed `best.pt`. The results
  file and the checkpoints agree perfectly.

### The bisect that localised it

Stage 3 stored per-trial NMSE for arms that use no trained weights at all.
`scratch/p15_a1_bisect.py`, 40 trials:

```
U0_em_gs         bitwise=YES  n_diff=0/40  median stored=-5.7311  now=-5.7311
H0_hs_em_gs      bitwise=YES  n_diff=0/40  median stored=-7.7049  now=-7.7049
oracle_phase     bitwise=YES  n_diff=0/40  median stored=-10.8972 now=-10.8972
```

The worlds, the spectral initialiser and the whole NumPy path are exact. The
divergence was confined to the learned arms.

### The cause

`TrackDConfig().train.init` defaults to **`"random"`** (`config.py:324`).
`stage4.main()` overrides it to `"spectral"` before training and before
evaluating (`stage4.py:341`). My PROMPT 12 scripts did not:

| script | sets `train.init` | consequence |
|---|---|---|
| `trackD_urformer/stage{1,2,3,4}.py` | yes | correct |
| nine other `scratch/trackD_*` evaluators | yes | correct |
| **`scratch/p12_partC_eval.py`** | **no** | evaluated with a random `G0` |
| **`scratch/p12_partC.py`** | **no** | **trained** with a random `G0` |

**Why this survived every check.** `make_initial_G("random", ..., seed=trial)`
is seeded per trial. The wrong evaluation was therefore *perfectly
deterministic* — bitwise reproducible across processes and across two
independent code paths. Determinism and correctness are different properties,
and every test applied so far, including the ones the brief specified, tested
only the first.

### The second defect, which re-scoring cannot fix

The checkpoints record the config they were trained under:

```
results/track_d/stage4/C_U1_snr5_20/best.pt    train.init='spectral'
results/track_d/stage4/C_H1_snr5_20/best.pt    train.init='spectral'
results/p12/partC/C1_U1_seed2/best.pt          train.init='random'
results/p12/partC/C1_H1_seed3/best.pt          train.init='random'
results/p12/partC/C2_H1_balanced/best.pt       train.init='random'
results/p12/partC/C3_U1_logdb/best.pt          train.init='random'
```

**Every arm trained under PROMPT 12 Part C learned from a random initial
estimate, while the stage-4 arms they were compared against learned from the
spectral one.** So:

- **P19 is not a seed comparison.** It varies the seed *and* the initialiser.
  Seed 1 is the spectral stage-4 pair; seeds 2 and 3 are random-trained. No
  evaluation can separate the two factors.
- **P20 and P21** compare a random-trained arm (`C2_H1_balanced`,
  `C3_U1_logdb`) against a spectrally-trained baseline, and re-evaluating them
  under the spectral initialiser now mismatches their own training.

The defect is in the weights, not in the scoring. **PROMPT 12 Part C must be
retrained, not re-scored.**

## A2. Fix and re-score

- `scratch/p12_partC_eval.py:eval_models` now sets `init="spectral"`.
- `scratch/p12_partC.py:run_cfg` now sets `init="spectral"`, and its docstring
  — which claimed the config was "identical to the stage-4/5 config" — is
  corrected, because that claim was the error.
- `tests/test_trackd_eval_reproducibility.py` pins six properties: the config
  default is still `"random"` (so the override stays necessary); `eval_models`
  actually hands the model a spectral `G0`, checked by spying on the dataset
  rather than by reading source; `run_cfg` sets it for training; `evaluate()`
  is deterministic in-process; the published stage-4 rows reproduce from
  `best.pt` bitwise; and the stage-4 checkpoints record `train.init='spectral'`.
  **All pass.**
- `reports/p15/rescored.csv` gives stored, re-scored and difference for every
  quantity, with a validity column. `reports/p15/partC_scores.json` is the
  re-scored output; `reports/p12/partC_scores.json` is kept as the record of
  what was wrongly reported.

## A3. Blast radius

**Unaffected — the published stage-3 and stage-4 rows are sound.** They were
produced by the stage scripts, which set the initialiser correctly, and stage
4's reproduce bitwise today.

| quoted number | status |
|---|---|
| `+1.209` dB (H1−U1, SNR ≥ 5, stage 3) | **stands** |
| `+0.078` dB (C_U1−C_H1, stage 4) | **stands** — and the re-score reproduces it exactly |
| `+1.902`, `+2.231` dB | **stands** — stage-5 arms, and `stage5.py:277` sets `init="spectral"`; every stage-5 checkpoint records `train.init='spectral'` |
| `+1.309` dB (internal vs post-hoc, 15–20 dB) | **stands** |
| `89.7%` | **stands** — recomputed as `sum(grad_share[:3]) = 0.8966` from `reports/trackD_stage5_results.json`, `unweighted_shares` |
| gradient-share span | **resolved to 30.0**, and the manuscript's `31` is wrong. From the same field, `0.4653/0.0155 = 30.03`; `PART_C.md`'s `30` is correct. Unaffected by any of this, being a property of the loss function on the training distribution rather than of a trained model. |

**Invalidated — everything in `reports/p12/PART_C.md`.** For the record, the
re-scored values and how far they moved:

| quantity | reported | re-scored | Δ |
|---|---|---|---|
| P19 Δ_H seed 1 | +0.690 | **+0.078** | −0.612 |
| P19 Δ_H seed 2 | +0.242 | +0.245 | +0.003 |
| P19 Δ_H seed 3 | +0.634 | +0.632 | −0.002 |
| P19 across-seed SD | 0.244 | 0.284 | +0.040 |
| P20 Δ_H over SNR ≥ 5 | +0.395 | **+0.174** | −0.221 |
| P20 top bin [15,20) | +1.308 | **+0.919** | −0.389 |
| P21 log − balanced | +0.584 | **+0.343** | −0.241 |

Three consequences, stated plainly because each cost us a conclusion:

1. **Seed 1 corrects to exactly the published stage-4 value.** `+0.0778` dB, the
   same figure `reports/trackD_stage4_results.json` records. The
   "stage 4's own `evaluate()` no longer reproduces its own stored rows"
   finding in `PART_A.md` was my bug, not the repository's.
2. **The P20 TENSION branch no longer fires.** Re-scored, Δ_H over SNR ≥ 5 is
   `+0.174` dB, *inside* the pre-registered band `[−0.10, +0.35]`. P20 still
   fails on the per-bin threshold (top bin `+0.919` > `+0.50`), but the claim
   that "the recommended loss restores the value of the prior we argued is
   illusory" is not supported by these numbers.
3. **P21 flips from FAILED to HELD** on re-evaluation alone: the log loss lands
   `+0.343` dB from the balanced run, inside the `±0.5` dB margin. The
   pre-registration's committed "yes" was right, and `PART_C.md`'s claim that
   the log loss *beats* reweighting by `+0.584` dB is withdrawn.

**None of these three re-scored values may be used either**, because the arms
behind them were trained with the wrong initialiser. They are recorded to show
the size and direction of the error, not as results.

## A4. Paper 1 exposure — confirmed clear

No number in the frozen letter originates in the trained loop:

- §IV-F's placement result is the **classical** B5 wash, `+0.003` dB six-point
  mean, from `results/p12/B5.json` — an EM-GS/HS-GS comparison with no network.
- The internal-versus-post-hoc contrast (`+1.309` dB) is **deliberately
  uncited**; see `docs/open-todos.md` item 3.
- Every `% src:` comment in `paper/paper1/haatim_hsgs_letter.tex` resolves to
  `results/track_b/`, `results/p12/B{1..5}.json`, `results/p14/aperture/`,
  `trackB_hankel_emgs/`, or `reports/trackD_partB9_analysis.json` — the last
  being classical Track D sweeps, not URformer runs.

Paper 1 stays frozen and is unaffected.

## What Part B and Part C now need

Part B's B1 (projection disabled at inference on `H1-balanced`) and B2 (the
three-design per-bin table) both read from arms trained with the wrong
initialiser, so neither is worth running on the current checkpoints. Part C's
priority list is unchanged in shape but larger in cost: the mixed-SNR,
balanced and log arms all need retraining with `init="spectral"`, and only then
do P24–P26 mean anything.

That is a scope change against the brief and it is the honest consequence of
the diagnosis, so it is reported here rather than absorbed silently.
