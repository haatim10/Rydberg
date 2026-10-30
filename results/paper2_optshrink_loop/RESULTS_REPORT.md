# Paper 2 results with OptShrink inside the estimation loop

Completed 8 October 2026. No existing model, manuscript, experiment source or
historical result was changed. No new neural model was trained.

## Main finding

The stronger classical comparison removes our learned model's average accuracy
advantage on the primary measure. Noise-dependent, development-tuned OptShrink
inside Rician EM is **0.143 dB better** on the equal-weight seven-SNR average.
The paired 95% interval for this classical advantage is **0.086 to 0.198 dB**,
conditional on the three existing models and the frozen development selection.

There is still a useful difference to investigate: our model makes fewer severe
errors in the main test range. Its severe-error rate is **0.89%**, compared with
**5.02%** for the selected in-loop baseline. All these severe errors occur at
−10 dB SNR. At that SNR, the corresponding rates are **6.25% and 35.16%**. Thus
this result does not establish a broad reliability advantage across all noise
levels or channel conditions.

The two methods have similar processing times in the measured 5 dB setting.
The learned model does not retain a large speed advantage against this new
baseline. The older advantage against costly HS-GS rank search remains valid
within its own comparison.

The primary accuracy comparison did not favor the learned model. Its lower
severe-error rate was also a predeclared outcome, but must not be used to
retroactively rename the primary accuracy result a success.

## What was tested

The unchanged simulator uses 32 antennas, three users and 30 magnitude
observations, with known reference and noise variance. Main channels have
3–7 paths per user and reference power 12 dB above one user's average power.
The seven SNRs are −10, −5, 0, 5, 10, 15 and 20 dB. Separate stress tests use
a weak reference or two paths per user at 5 dB SNR.

There were 224 development cases, 32 per main SNR. Classical settings were
selected using withheld-pilot Rician likelihood: 21 observations for estimation
and nine for scoring. No channel truth entered selection. The grid contained
ranks 1–15, mixing strengths 0.25, 0.5 and 1, a no-filter option, and iteration
budgets 10, 20, 50 and 100.

After selection was frozen, 1,152 fresh test cases were generated: 128 in each
of the seven main and two stress cells. All methods saw the same observations,
pilots and reference within each case, and used all 30 observations at inference.
The learned results average the three unchanged model seeds; those predictions
are not three independent channel datasets.

The main OptShrink baseline chooses settings from the supplied noise variance,
not channel truth or a channel-family label. Its frozen choices were:

| SNR, dB | Rank | Mixing strength | EM iterations |
|---:|---:|---:|---:|
| −10 | 1 | 0.5 | 10 |
| −5 | 2 | 0.5 | 10 |
| 0 | 4 | 0.5 | 10 |
| 5 | 5 | 1.0 | 10 |
| 10 | 7 | 1.0 | 10 |
| 15 | 7 | 1.0 | 20 |
| 20 | 7 | 1.0 | 20 |

Both stress cells use the 5 dB settings without retuning. The single global
in-loop control instead uses rank six, strength one and 20 iterations.

This is an adaptation of [OptShrink's original matrix-filtering rule](https://arxiv.org/html/1306.6042v4)
to a Rician EM loop. It is not a reproduction of the full
[LRHD method](https://arxiv.org/abs/2012.07433), and does not transfer an
optimal-denoising theorem to this nonlinear, finite-size receiver problem.

## Main results

For each SNR, NMSE is 10 log10 of total squared channel error divided by total
channel energy. The primary score averages the seven cell scores equally in
dB. Lower is better. A severe error means that a case's squared error exceeds
its true channel energy. Learned rates are averaged over the three model seeds.

| Method | Seven-SNR average, dB | Mean per-case normalized error, linear | Severe errors |
|---|---:|---:|---:|
| Learned model, ten stages | −14.319 | 0.1968 | 0.89% |
| In-loop OptShrink, noise-dependent settings | **−14.462** | 0.2153 | 5.02% |
| Final-only OptShrink, noise-dependent settings | −14.374 | 0.2089 | 3.79% |
| In-loop OptShrink, one global setting | −13.501 | 0.3950 | 14.06% |
| Rician-selected HS-GS | −12.890 | 0.3588 | 13.06% |
| Ordinary EM, 100 iterations | −10.242 | 0.8027 | 23.77% |
| In-loop OptShrink, fixed ten iterations | −14.120 | 0.2155 | 5.02% |
| In-loop OptShrink, fixed 20 iterations | −14.378 | 0.2303 | 6.47% |
| In-loop OptShrink, fixed 50 iterations | −14.342 | 0.2378 | 7.59% |
| In-loop OptShrink, fixed 100 iterations | −14.335 | 0.2392 | 8.04% |

Each fixed-iteration control still has its rank and strength selected separately
at each noise level using development observations. They are not test-selected
winners. The learned model's three individual primary scores are −14.430,
−14.179 and −14.348 dB.

The ranking depends on what is measured. The learned model has lower mean
per-case linear error, which is more influenced by the difficult low-SNR cases.
The predeclared equal-SNR dB score favors the selected in-loop baseline. Neither
quantity should be presented as though it were the other.

### Where the difference occurs

Positive gain below favors the learned model over the selected in-loop baseline.
Per-SNR intervals are descriptive paired-case 95% intervals, without adjustment
for multiple comparisons.

| SNR, dB | Learned NMSE, dB | In-loop OptShrink NMSE, dB | Learned gain, dB | Interval for gain, dB |
|---:|---:|---:|---:|---:|
| −10 | −1.184 | −0.461 | +0.723 | +0.610 to +0.841 |
| −5 | −4.138 | −4.445 | −0.307 | −0.414 to −0.202 |
| 0 | −9.088 | −9.384 | −0.296 | −0.410 to −0.180 |
| 5 | −14.114 | −14.320 | −0.205 | −0.399 to −0.007 |
| 10 | −19.394 | −19.417 | −0.023 | −0.113 to +0.066 |
| 15 | −23.991 | −24.088 | −0.097 | −0.223 to +0.027 |
| 20 | −28.322 | −29.118 | −0.796 | −1.030 to −0.560 |

Against Rician-selected HS-GS, the learned model still improves the primary
average by **1.429 dB**, with a conditional interval of **1.334 to 1.522 dB**.
Against one global in-loop OptShrink setting, its gain is 0.818 dB. Allowing the
classical baseline to adjust settings to known noise level changes the conclusion.

The tuned final-only control is also strong. Its primary score is only 0.088 dB
worse than the tuned in-loop method, and 0.055 dB better than the learned model.
Do not attribute the change from the earlier final-filter diagnostic solely to
placing OptShrink inside the loop: noise-dependent tuning, early stopping and
fresh cases also changed. The previous study remains a separate protocol.

### The stress tests do not remove the earlier limitations

| Stress case | Learned NMSE, dB | Selected in-loop NMSE, dB | Rician HS-GS NMSE, dB |
|---|---:|---:|---:|
| Weak reference, RSR 0 dB | −7.069 | −5.026 | **−8.307** |
| Two paths per user | −16.814 | −16.406 | **−17.833** |

The learned model beats the selected in-loop configuration in both stress
cases, but HS-GS still beats the learned model by 1.238 and 1.019 dB,
respectively. The predeclared 50- and 100-iteration OptShrink controls also do
better at weak reference: −8.790 and −8.981 dB. Those diagnostic results cannot
be used to replace the primary comparator after testing. They show that a
stopping rule chosen at a stronger reference need not transfer well.

## Processing time

These are warmed, serial, batch-one CPU times at SNR 5 dB, measured on 24
separate worlds with three repetitions. They include initialization and online
selection, but not loading models or reading files. All numerical libraries use
one thread and double precision. The environment was macOS 26.6.2, arm64,
Python 3.12.5, NumPy 2.4.6, SciPy 1.17.1 and PyTorch 2.13.0.

| Method | Mean time per estimate |
|---|---:|
| Learned ten-stage model | **5.67 ms** |
| Selected in-loop OptShrink | **6.20 ms** |
| Selected final-only OptShrink | 3.99 ms |
| Rician-selected HS-GS | 356.65 ms |
| Ordinary EM100 | 11.14 ms |

The learned number is the average cost of one model across seeds, not the cost
of an ensemble. Individual model means range from 5.22 to 6.02 ms. The mean of
their individual medians is 4.98 ms; the in-loop median is 5.60 ms. Small timing
differences should not be treated as reliable algorithmic speed guarantees.
The large speedup applies to the rank-search baseline, not to every classical
method. These timings cover one SNR and one implementation/device only.

Development search used about 245 seconds of numerical computation. It was an
offline cost, not a free online search. Earlier neural training costs were not
rerun or included in the latency table. Training and search budgets are not
equal between the learned and classical approaches.

The runner's first timing pass precomputed its fixed-SNR configuration lookup.
A separate end-to-end timing audit included that lookup inside the measured
boundary. The table uses this second pass, saved in
`timing_including_dispatch.json`; the original `timing.json` is preserved.

## Verification and limits

Seven numerical unit tests passed, covering the OptShrink weights, rectangular
matrices, repeated/zero singular values, scaling, mixing, deterministic prefixes,
noise-level mapping and equivalence to ordinary EM without filtering.

All 224 development records and 1,152 test result files were checked. Selection
was recalculated from saved development scores. Test identities, shared input
pairing and post-freeze generation were checked. Every channel-error statistic
was recalculated from the saved estimates and truth. All source and checkpoint
hashes remained unchanged.

Eighteen selected/final OptShrink replays using a separate scalar-weight
implementation agreed within 4.6 × 10⁻¹⁵ in channel entries. Twenty-seven
representative learned-checkpoint replays agreed exactly. The main run,
including its first timing pass and audit, took 13.8 minutes; the additional
timing audit and report preparation followed it.

The 4,000 paired bootstrap resamples retain all model predictions for each
sampled case. They quantify test-case uncertainty conditional on development
selection and the frozen models. They do not quantify changes from independent
training datasets. Reference calibration and noise variance are known here.
This is still a simulation of the effective magnitude model, not atomic
dynamics or a hardware demonstration.

## What this means for the manuscript

1. Add this comparator to the main evidence. Do not retain a general claim that
   the learned model is both more accurate and much faster than strong
   classical alternatives.
2. Keep the supported HS-GS comparison, but state which baseline it concerns.
   Present the new result as an accuracy/reliability trade-off.
3. Treat the low-SNR severe-error result as a candidate strength to confirm,
   not as established novelty or a substitute for the failed primary comparison.
4. Keep the weak-reference and few-path limitations. Longer classical iteration
   controls can outperform both the selected baseline and the learned model
   under weak references.
5. Before a large retraining study, decide whether low-SNR reliability is the
   intended contribution. A new confirmation would need a predeclared failure
   metric and classical controls tuned to that same objective, plus independent
   training datasets. Do not select that claim only because it looks favorable
   on these already examined cases.

No larger training matrix or manuscript revision was launched automatically.
The outstanding independent-dataset, trained-depth/controller-width and tight
pilot-budget checks are not completed by this experiment.

## Evidence files

In this results directory: `summary.json`, `verification.json`,
`frozen_selection.json`, `manifest.json`, `artifact_hashes.json`,
`timing_including_dispatch.json`, `comparison.png`, and the separate development,
test-observation, test-truth and prediction files.

The isolated implementation and fixed protocol are in `paper2_optshrink_loop/`
at the repository root. The README gives run, resume and independent-analysis
commands. The two additional scripts under `reports/` generate the scientific
figure and audit the full runtime boundary; they do not change any estimates.
