# Baseline discrepancy: forensic findings and remaining blocker

## Status

**The published URformer baseline is not reproduced. No validated correction that closes the gap has been identified.**

The investigation now has a more specific result: under our explicitly chosen stochastic channel, pilot, noise and loss conventions, a numerical Bayes lower-bound diagnostic is incompatible with the target curve at the examined SNRs. A training-only change cannot be assumed to solve that incompatibility.

This is a finding about the compatibility of OUR simulator assumptions with the paper's reported expected performance. It is not a finding that the paper is wrong. It does not identify which undocumented setting differs, and does not exclude additional implementation issues.

No new training was launched. All completed experiments, their drivers, data and source manifests are unchanged. No author was contacted. The cancelled calibration project was not resumed.

## 1. Checks completed

1. Visually inspected page 4 of the downloaded arXiv v3 PDF, including Figures 3 and 4 and Table I. Figure 3's URformer curve and the previously extracted values agree visually. The target is not a mislabeled classical curve. Source: results/paper2_reproduction/source/paper_v3.pdf.
2. Read the relevant TeX equations and compared the independent reconstruction to them. Known assumptions remain explicit: gain distribution, joint ray-angle statistics, pilot generation, SNR-training distribution, several activation/initialization settings, and data reuse.
3. Re-ran seven existing reproduction/diagnostic tests: all passed. These include an independent complex-equation check of one layer, encoder sequence, SNR calibration, correct mean-linear NMSE metric, matched diagnostic weights/noise/order, and a tiny training/resume pipeline.
4. Replayed all six saved reproduction models (three seeds × selected/final) on the entire 2,000-case validation set. Each reproduces its recorded validation score to 1e-7 dB.
5. Applied five frozen-weight interventions per saved model, giving 30 validation evaluations. These are diagnostics, not retrained replacement estimators or independent generalization tests.
6. Checked least squares against a complex pseudoinverse (maximum absolute difference 4.30e-6 in float32), coherent rescaling of channel/measurements/reference/noise in EM-GS (4.19e-6), and the noiseless true-channel fixed point of GS (3.02e-6).
7. Checked a new normalized-loss Bayes calculation against six numerical/unit tests, all passing. Details below.
8. Rechecked frozen source hashes for the three completed experiments. No source changes.

These checks rule out several specific suspected mistakes in the checked paths. They are not exhaustive proof that the reconstruction matches the authors' code.

A repeated targeted public search did not locate the authors' implementation. That is not proof no implementation exists. The local arXiv source archive contains paper sources/figures, not training code; direct access to the publisher page failed in this tool on this pass.

## 2. What the saved-model interventions show

Scores below are NMSE in dB, pooled in linear units over three training seeds. Lower is better. All evaluations use the existing validation set, not the final test set. The last column is a 15–20 dB bin, not evaluation exactly at 20 dB.

| Checkpoint | In-memory intervention | Full-range validation | 15–20 dB validation |
|---|---|---:|---:|
| selected | original | -5.436 | -14.186 |
| selected | GS_filter | -1.577 | -6.186 |
| selected | exact_Bessel_learned_gate | -1.855 | -6.188 |
| selected | no_Transformer | -3.240 | -9.404 |
| selected | exact_EMGS_no_Transformer | -1.166 | -16.833 |
| final | original | -4.317 | -13.974 |
| final | GS_filter | -0.688 | -5.254 |
| final | exact_Bessel_learned_gate | -0.971 | -5.259 |
| final | no_Transformer | -3.611 | -12.220 |
| final | exact_EMGS_no_Transformer | -1.166 | -16.833 |

- original: unchanged trained reconstruction.
- GS_filter: remove learned filtering by setting its gate to zero, but retain trained Transformers.
- exact_Bessel_learned_gate: replace the learned filter with the exact Bessel ratio, retaining the learned gate and Transformers.
- no_Transformer: bypass trained residual corrections, retaining learned filters/gates.
- exact_EMGS_no_Transformer: exact Bessel, gate one, no Transformer. This must reduce to ten classical EM-GS steps.

Replacing the learned filter in an already trained network substantially worsens performance. Therefore this is not a simple plug-in fix. The filters and corrections co-adapted. This does not prove that a model trained from the start with an exact filter would be bad.

The exact-EM-GS intervention matches the independent ten-step classical implementation numerically in all bins. It scores −16.833 dB in the highest validation bin, versus −19.268 dB for 100-step EM-GS. The neural reconstruction can improve full-range performance while being worse at high SNR.

The learned model therefore has room for improvement relative to some classical operating points. That is separate from whether the published target is attainable under our chosen channel distribution.

## 3. More informative diagnostic: an ideal estimator with extra information

The ordinary receiver observes magnitudes, Z=|HS+B+W|. To make estimation easier, the diagnostic gives an imaginary receiver:

- the full complex noisy observation Y=HS+W, including phase;
- every true path angle for the current channel;
- the exact prior distribution of the path gains;
- the true noise variance and pilots.

It does NOT give the channel itself or the realized gains. Its remaining uncertainty can be computed because gains in OUR generator are independent circular complex Gaussian.

An estimator with this extra information can always discard it and imitate a magnitude-only estimator: form |Y+B| and run that estimator. Consequently, the optimal expected error with the extra information cannot be worse than the best expected error achievable without it.

This is an expected-risk statement under the stated prior, not a bound on every finite test sample.

### Exact loss convention

A usual coherent LMMSE score is not sufficient here: it minimizes unnormalized squared error, whereas the paper reports the mean of per-case normalized squared errors. The diagnostic explicitly optimizes the latter.

Let x=vec(H), and condition on the known angles and coherent observation. Its Gaussian posterior is x ~ CN(mu,C). Define

A = E[1 / ||x||² | observation, angles],

b = E[x / ||x||² | observation, angles].

The normalized-loss optimal estimate is b/A. Its minimum conditional expected loss is

risk = 1 − ||b||²/A.

This follows by expanding E[||x−estimate||²/||x||²]. There is no unbiased-estimator requirement and this is not a classical CRLB.

For a proper complex Gaussian, let

M(t) = det(I+tC)^(-1) exp(−t mu^H (I+tC)^(-1) mu).

The integrals are A=integral M(t) dt and b=integral (I+tC)^(-1)mu M(t) dt, over t≥0. These follow from the Laplace representation of 1/||x||². The implementation evaluates these integrals in the covariance eigenbasis.

The prior is block diagonal across users, with each user covariance (32/40) times the steering matrix times its conjugate transpose. With column-stacked channels, the coherent observation matrix is S^T tensor I32, and its precision is (conj(S) S^T) tensor I32 divided by noise variance. Given prior R, the posterior covariance is computed stably using a PSD square root, without inverting a possibly near-singular R.

For the outer expectation, posterior means are sampled from CN(0,R−C) and ray angles from our declared generator. This avoids confusing E[error/energy] with E[error]/E[energy].

### Numerical results

A first 32-angle-set, 32-posterior-mean-per-set calculation was followed by a separate confirmation with 128 fresh angle sets and 64 posterior means per set: 8,192 means per mode/SNR. No neural test scores are used by this calculation. Confirmation angles start at trial ID 12,100,000 and use independent random seed 2026092320.

| Our assumed channel | SNR | Numerical genie Bayes NMSE | Nominal 95% Monte Carlo interval | Published Figure 3 |
|---|---:|---:|---|---:|
| literal_independent | 5 | -13.334 | [-13.363, -13.304] | −21.734 |
| literal_independent | 20 | -27.559 | [-27.588, -27.530] | −32.024 |
| clustered_5deg | 5 | -16.521 | [-16.590, -16.453] | −21.734 |
| clustered_5deg | 20 | -30.487 | [-30.558, -30.418] | −32.024 |

The published points sit below (better than) even the ideal coherent/known-angle limits calculated for both of our proposed channel interpretations. At 5 dB the gap is about 8.40 dB for independent rays and 5.21 dB for ±5° clusters. At 20 dB the corresponding gaps are 4.47 and 1.54 dB.

The small Monte Carlo intervals are conditional on this exact generator, pilot matrix and mathematical model. They do not quantify uncertainty about what the authors implemented. They are approximate pointwise t intervals over independent angle-set averages; they are not rigorous numerical-certification bounds.

### Validation of the calculation

Six tests passed:

- zero posterior uncertainty yields zero risk and returns the known mean;
- a zero posterior mean yields zero action and risk one;
- scaling units and complex unitary rotation preserve risk and transform the estimate correctly;
- the action and risk agree with direct posterior Monte Carlo (150,000 Gaussian draws in a separate 16-dimensional case);
- posterior covariance agrees with an independent Gaussian-conditioning formula;
- vectorization, pilot precision and conditional channel covariance match the generator, including a reconstruction from the generator's actual angle/gain random streams.

The first itemized test combines two limit checks; covariance/vectorization and generator-reconstruction are separate test methods. Doubling quadrature resolution and the integration interval on the first angle set of each mode/SNR changes risk by less than 3.7e-12. The independent larger outer simulation confirms the exploratory results.

This derivation and implementation are useful diagnostic evidence, not a publication-ready theorem package. They should receive independent mathematical review before being cited as a formal result in a manuscript.

## 4. What is established, and what remains unresolved

Established within the checked setup:

- The large discrepancy is not a graph-reading error.
- Saved-checkpoint evaluation is reproducible.
- The checked complex update/least-squares equations, SNR bookkeeping and metric implementation pass their tests.
- Replacing one learned component after training does not close the gap.
- Merely training a more capable network on our current declared distribution cannot be assumed to reproduce a curve that is numerically below its ideal expected-risk limit.
- Correctly changing physical units leaves the measurement information unchanged, although numerical training behavior can change.

Still unresolved:

- The actual joint distribution of cluster/ray angles and any correlations between users.
- The gain distribution and normalization, including any per-realization channel normalization between the paper's tilde-h and h notation.
- The exact pilot matrix, pilot power convention, and whether it changes between examples.
- How signal power, noise variance, reference amplitude and SNR were computed for the neural and classical curves.
- Whether one network or a separate network per SNR produced Figure 3.
- Whether channel/noise/initial guesses are fixed or regenerated across epochs and evaluation; the data split procedure.
- Exact code for model initialization, normalization, tokenization/reshape, activations and loss reduction.
- Whether the plotted evaluation uses the same channel population and metric as our interpretation of the equations.

A different, more correlated channel distribution or a different noise/evaluation convention can change the ideal limit. Therefore the result does NOT establish a flaw in the published paper. It establishes that our present reconstruction assumptions are not sufficient to reproduce that target as the same statistical task.

There may also be a training/architecture issue contributing to our remaining gap relative to suitable classical methods. The genie bound is not an achievable benchmark for our actual receiver, so the neural-to-genie gap cannot be assigned entirely to deficient training.

## 5. Required next step

Obtain the authors' channel-generation, training and Figure-3 evaluation code, or clarification sufficient to reconstruct it.

The smallest useful request is:

1. The channel generator, including gains, joint ray/cluster-angle generation, and normalization.
2. The actual pilot/noise/reference construction and exact SNR/RSR calculation.
3. Mixed-SNR versus separate-per-SNR training, example regeneration, and train/validation/test splitting.
4. Model definition and initialization, including the exact reshape and any undocumented input/output normalization.
5. The raw Figure-3 errors or plotting/evaluation code and checkpoint rule.

Once obtained, verify one deterministic generated example and one forward update, then train the ordinary model under the clarified protocol on disjoint data. Only after the baseline is credible should a matched Hankel comparison be commissioned.

Changing the noise level, channel prior, or reported metric just to hit the published numbers would change the experiment, not resolve the reproduction. These were not changed.

## Files

- scripts/paper2_baseline_forensics.py: read-only validation replay and interventions.
- scripts/paper2_baseline_oracle.py: coherent known-angle normalized-loss calculation.
- tests/test_paper2_baseline_oracle.py: six checks.
- frozen_weight_probe_seed1.json and frozen_weight_probe_seeds2_3.json: all 30 probe scores.
- oracle_pilot.json and oracle_confirmation.json: exploratory and fresh-case confirmation results.

Source target: Xiao et al., arXiv:2509.12586v3, Figure 3, https://arxiv.org/html/2509.12586v3 . The plotted values were already extracted from the supplied source archive and were visually rechecked against its PDF.

