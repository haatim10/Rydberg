# Channel Estimation for Rydberg Receivers Using Learned Hankel Filtering

Research draft — 23 September 2026

## Abstract

Rydberg receivers use excited atoms to sense radio signals. In a magnitude-only receiver, the measured signal strength does not directly reveal the wireless channel. Recovering the channel is difficult because the measurements contain noise and do not directly provide phase. Channel structure can help, but a fixed structural rule may also remove useful information. This paper studies a small learning-based estimator that adjusts the amount of Hankel filtering using the current measurements. The estimator alternates between fitting the receiver observations and filtering the channel estimate. It is trained by predicting a withheld part of the observations, without using true channels as training targets. In simulations, the method gives lower average channel error and shorter processing time than the tested classical Hankel estimator. Further tests show that using both channel-structure information and noise-related information improves average accuracy. The benefit is not uniform: the classical method remains better with a weak reference signal and in some channels with few paths. The results support learned Hankel filtering as a useful option for the tested receiver model, while showing the need to state its operating limits clearly.

## 1. Introduction

Rydberg atoms are atoms in highly excited states that can respond to radio-frequency electric fields. This makes them useful for studying new ways to receive wireless signals. In the receiver model used in this paper, signals from several users are combined with a known reference signal. The receiver measures the strength of that combined signal, rather than its full complex value. The wireless channel describes how each user's signal changes in strength and phase before reaching the receiver. Estimating this channel is an important step in recovering the transmitted information. Learning-based methods have already been developed for Rydberg channel estimation, including URformer. [1]

Signals arriving through a small number of paths often form a repeated pattern across a regular receiver array. A Hankel matrix arranges the channel samples so that this pattern can be used for estimation. However, a channel does not always follow a simple structural model closely. Filtering too strongly can remove real channel details, while filtering too weakly can leave noise in the estimate. Trying many possible structural settings also takes time. This motivates a small model that learns how much filtering to apply from the available measurements, without requiring the true channel during training.

The contributions of this paper are:

1. A small channel estimator that combines measurement-based updates with learned Hankel filtering. The amount of filtering changes with the observations.
2. A training method that uses only measured signal magnitudes, known pilots, the reference signal and the assumed noise level. True channels are not used as training targets or to choose the saved model.
3. Simulation results that compare accuracy, processing time and simpler versions of the method. The comparisons include conditions where the proposed method is worse, not only those where it improves the result.

## 2. Related work

Learning without clean target signals is already used in signal reconstruction. One approach uses part of the measurements to form an estimate and another part to train the model. [2] Self-supervised learning has also been used to remove noise from Rydberg sensor signals. [3] Learning-based Hankel reconstruction and self-supervised low-rank reconstruction have been studied in other applications. [4, 5] Therefore, neither self-supervision nor learned Hankel processing is new on its own.

The focus here is a particular channel-estimation problem: recovering a complex multiuser channel from noisy magnitude measurements with a known reference. We combine the receiver's noise model with a small network that controls smooth Hankel filtering. Our earlier companion work studies classical Hankel estimation and rank selection. Here, the emphasis is on learning the filtering strength without channel labels and measuring the resulting accuracy and processing cost. We do not claim to introduce self-supervision to Rydberg sensing or to reproduce the published URformer results.

## 3. Receiver and channel model

We consider a receiver array with \(N\) elements and \(K\) users. Each user sends \(P\) known training symbols, called pilots. These pilots allow the receiver to estimate the channel before detecting unknown data.

The observations are modelled as

\[
Z=|GS+B+W|.
\]

Here, \(G\) is the unknown channel matrix, \(S\) contains the known pilots, \(B\) is the known reference signal, and \(W\) is noise. Their sizes are \(N\times K\), \(K\times P\), \(N\times P\) and \(N\times P\), respectively. The magnitude operation acts on each entry. Thus, each observation is a nonnegative signal magnitude, not a direct measurement of phase.

The reference is important because its combination with the user signals makes the measured magnitudes depend on their relative phases. However, a known reference does not guarantee that every possible channel can be recovered accurately.

The noise is independent circular complex Gaussian noise with variance \(v\). Its real and imaginary parts each have variance \(v/2\). We assume that the reference and noise variance are known, except in tests that deliberately supply an incorrect noise variance. Receiver conversion gain is absorbed into the model's scaling. These are modelling assumptions, not claims about all physical Rydberg receivers.

The main simulations use an equally spaced linear array with half-wavelength spacing. A channel from user \(k\) is the sum of several arriving paths:

\[
g_{nk}=\sum_{\ell=1}^{L_k}\alpha_{\ell k}
e^{-j\pi n\sin\theta_{\ell k}},\qquad n=0,\ldots,N-1.
\]

The number of paths is \(L_k\). Each path has a complex gain \(\alpha_{\ell k}\) and an arrival angle \(\theta_{\ell k}\). Angles are drawn uniformly between \(-\pi/2\) and \(\pi/2\). Gains are independent complex Gaussian values with variance \(1/L_k\), giving equal average power for each user. Pilot entries are independent complex Gaussian values with unit variance. The reference has a known spatial response and a constant waveform across the pilot observations.

Signal-to-noise ratio, or SNR, measures the total average user-signal power relative to noise. Reference-to-signal ratio, or RSR, measures reference power relative to one user's average power. These are different definitions. With three equal-power users, an RSR of 12 dB is about 7.23 dB relative to their combined average signal power.

## 4. Proposed estimator

### 4.1. Using the measurements

The estimator begins with an initial channel estimate formed from the observations, pilots and reference. It then repeats two steps ten times: update the estimate using the measurements, and apply Hankel filtering.

The measurement update uses the Rician distribution. This is the probability model for the magnitude of a complex signal corrupted by the Gaussian noise assumed here. It accounts for the fact that noise acts before the receiver takes the magnitude. Treating the final magnitudes as an ordinary signal plus Gaussian noise would not give the same model.

We use an expectation-maximization update, abbreviated EM. It updates the channel using the current estimate of the unobserved complex signal. Each stage has a learned step size. This makes it possible to change how far the estimate moves at that stage. The update uses the known pilots and reference; it does not use the true channel.

### 4.2. Filtering the channel

For each user, the channel samples across the array are placed into a Hankel matrix. Its entries are constant along each antidiagonal. This arrangement captures the repeated spatial pattern produced by the path model. With 32 array elements, the matrix has 17 rows and 16 columns.

The singular values of this matrix describe the strength of its different matrix components. Together, these values are called its singular-value spectrum. A common structural method keeps only a chosen number of components and removes the rest. Our method instead reduces them smoothly:

\[
\widehat\sigma_i=\frac{\sigma_i^3}{\sigma_i^2+\tau^2}.
\]

Here, \(\sigma_i\) is a singular value and \(\tau\) controls filtering strength. A larger \(\tau\) gives stronger filtering. Small singular values are reduced by a larger fraction than large ones. This does not mean that every small component is noise; some may contain real channel information.

The filtered matrix is converted back to a channel vector by averaging entries on each antidiagonal. A learned mixing factor controls how much of this correction is added to the measurement-based estimate. Neither this factor nor the filtering strength is treated as an estimate of the true number of paths.

### 4.3. Choosing the filtering strength

A small neural network chooses the filtering strength and mixing factor. It uses eight quantities calculated from the current estimate and the known receiver settings. These describe:

- how the singular values are distributed;
- how well the estimate fits the measured magnitudes;
- the noise level relative to the estimated channel strength;
- the number of pilots, the matrix size and the current stage.

The difference between a measured magnitude and its predicted value is called a measurement residual. The same network is used for every user and every stage. It has eight inputs, 16 hidden units and two outputs. Together with ten learned update step sizes, the estimator has 188 trainable parameters.

The model does not receive the true channel, the true path count or the channel-generation label. Its settings change with its input measurements, but its trained weights remain fixed during use. The complete estimator is not guaranteed to improve the channel error at every stage.

### 4.4. Training without true channels

Each training example contains 30 pilot observations. We randomly give 21 to the estimator and withhold nine. The estimator forms a channel estimate from the first group. That estimate is then used to predict the measurements in the withheld group. The Rician probability model measures how well those predictions explain the observed magnitudes.

The model is trained to improve this prediction. This is called self-supervised learning because the observations supply the training information; no true-channel target is needed. The split changes each training epoch, where an epoch is one pass through the training examples.

Separate validation examples are used to choose the saved model. Once training is complete, the receiver uses all available pilots. No extra pilots or further weight training are required during estimation. Using all 30 observations rather than 21 changes the model's input conditions, so its performance under this change is tested rather than assumed.

## 5. Simulation study

### 5.1. Setup and comparisons

The main setting has 32 array elements, three users and 30 pilots. Training uses 16,384 examples, with 2,048 separate validation examples. It covers SNRs from −10 to 20 dB, RSRs of 5, 12 and 20 dB, and several path counts. Half the examples use random counts from three to seven paths per user. The other half use fixed counts of 2, 4, 8, 12 or 16.

Models are trained for 20 epochs with batches of 32 and an Adam learning rate of 0.001. Validation gives equal weight to 126 combinations of SNR, RSR and path setting. The main study trains 18 models using two measurement-update choices, three filtering choices and three training seeds. A seed changes the initial weights, training order and pilot splits. These three seeds share the same training examples. EM is kept as the main update because it was chosen using validation performance before testing.

The main evaluation includes 32 conditions with 400 simulated cases each. Two noise-input tests reuse cases from another condition, giving 12,000 distinct cases. Methods are compared on the same channel, pilots and noise within each case.

The main classical comparison is HS-GS, a Hankel-structured iterative estimator that searches over candidate ranks. Other comparisons include EM without learned filtering, gradient-based methods, hard Hankel filtering and simpler learned versions. A global-filtering control learns one relative filtering strength and mixing factor for each stage, rather than choosing them from the controller inputs. Its absolute strength still scales with the current Hankel matrix.

Accuracy is measured by normalized mean squared error, or NMSE:

\[
\mathrm{NMSE}_{\rm dB}=10\log_{10}
\frac{\sum_i\|\widehat G_i-G_i\|_F^2}
{\sum_i\|G_i\|_F^2}.
\]

This compares total squared channel error with total channel energy. A lower value is better. The main average gives equal weight to the seven SNR settings. We average scores from the training seeds; we do not combine their predicted channels.

Uncertainty intervals are calculated by repeatedly resampling whole test cases, keeping competing methods paired. These intervals describe variation in test cases for the saved models. They do not measure all the uncertainty caused by choosing a different training dataset.

### 5.2. Main results

| SNR (dB) | Proposed NMSE (dB) | Improvement over HS-GS (dB) |
|---:|---:|---:|
| −10 | −1.254 | 4.696 |
| −5 | −4.122 | 1.687 |
| 0 | −9.191 | 1.620 |
| 5 | −14.348 | 1.422 |
| 10 | −19.171 | 1.101 |
| 15 | −24.260 | 1.062 |
| 20 | −28.563 | 0.139 |

The average improvement over HS-GS is 1.675 dB, with a paired-test interval of 1.618 to 1.730 dB. Results are not equally strong at every SNR. At 20 dB, the interval includes zero, so that setting does not establish a clear improvement.

At the reference timing setting, the proposed estimator takes a median of 8.24 ms per case, compared with 525.62 ms for HS-GS. This is about 64 times faster in the tested implementation. Timing includes initialization and rank search where required. It uses one CPU thread on an Apple M2, processes one case at a time, and uses double precision. It covers 100 cases with three timing repeats. These measurements do not establish the same speed difference on other hardware or against differently optimized software. Training time is not included in inference time.

### 5.3. Further checks

A separate study uses two new training datasets. The average gains over HS-GS are 1.939 and 1.837 dB. Gains over the global-filtering control are 2.252 and 2.185 dB. These results show that the benefit is not limited to the original training dataset. Each new dataset uses one training seed, so this is still a limited replication.

We also compare against a supervised version that uses true channels during training. The label-free version has about a 1 dB advantage under the main averaging rule. However, this advantage becomes smaller when the saved-model selection rule or error averaging rule is changed. The supervised version is better in some settings, including the two-path test. It also behaves differently when moving from 21 input pilots during training to 30 during testing. We therefore do not conclude that self-supervision is generally better than supervision.

Applying filtering throughout the estimator gives about a 0.70 dB average improvement over applying it only at the end. It also takes roughly twice as long in that separate timing test. Repeated filtering therefore offers an accuracy–time choice, rather than a benefit without cost.

The learned mixing factor is usually close to one. Fixing it at one gives only a small average improvement and worsens some training seeds on difficult cases. We keep the original model and do not present this factor as a reliable decision to turn filtering on or off.

### 5.4. Which inputs help?

A smaller study tests the information given to the filtering network. It compares all inputs with three restricted versions: channel-spectrum inputs only, noise/residual inputs only, and basic size/stage inputs only. Basic size and stage information is retained in every version. The underlying channel update and Hankel filter are unchanged.

All four versions are trained from scratch using the same 4,096 development examples, three seeds and 12 epochs. They are tested on two fresh sets containing 2,304 cases in total. Each set contains the seven main SNR settings and two separate difficult conditions. This is a smaller-budget check, not a replacement for the main study.

| Restricted controller inputs | Full model's average improvement, test set 1 (dB) | Test set 2 (dB) |
|---|---:|---:|
| Channel spectrum only | 1.245 | 1.216 |
| Noise and measurement residual only | 0.568 | 0.577 |
| Basic size and stage only | 2.226 | 2.213 |

The average gains are positive for all three training seeds in both test sets. Intervals adjusted for the three comparisons also remain above zero. Using final models instead of validation-selected models gives the same scores in this study.

However, the full model is not better at every setting. At 5 dB SNR, it is about 0.39–0.40 dB worse than the noise/residual-only version. The full model therefore fails the study's predeclared requirement that no main setting should worsen by more than 0.25 dB against that version. It also loses around 0.82–0.93 dB against that version in the separate two-path test.

The result supports using both types of information to improve the average under this training setup. It does not show that both are necessary in every channel or that the same gains will hold after a larger training study. Removing an input from the small network also does not remove all related information from the estimator: the measurement update still uses noise variance, and the filter still acts on the channel matrix.

### 5.5. Where the method does not help

In the main study, HS-GS is better when the reference is weak and in some channels with few paths. The proposed method loses 1.617 dB at RSR 0 dB, 1.104 dB in the two-path test and 0.203 dB in the three-path test.

These are important limits. A model chosen for good average performance need not be the best choice for a very simple channel or a weak reference. Matching the assumed noise distribution does not guarantee that the estimation problem is easy to solve.

Other tests show gains for the tested clustered and unequal-power channels and for supplied noise variances that are wrong by ±3 dB. Results also change with array size and user count. Some path counts overlap the training examples, so the path-count sweep is not evidence that the method works on wholly unfamiliar channel types.

## 6. Discussion

The useful result is not that Hankel filtering always helps. It is that a small model can learn how strongly to use this structure, without true-channel training targets, and provide a useful average accuracy and processing-time result in the tested setting.

The input-removal study gives more evidence about this result. The full set of inputs performs better on average than either input group alone. At the same time, the losses in individual settings show why the average must not be the only reported result.

All results are simulations. Reference calibration is assumed known, and no physical receiver measurements are used. We have not proved that the channel is uniquely recoverable in every setting or that the complete learned estimator converges. The study also does not establish superiority over published URformer performance. These limits keep the claim focused on what was tested.

## 7. Conclusion

This paper studied a small, label-free channel estimator for magnitude-only Rydberg receiver measurements. The estimator combines measurement-based updates with Hankel filtering whose strength is learned from the current estimate and receiver information. Simulations show lower average error and shorter processing time than the tested classical Hankel estimator. Separate tests support the use of both channel-structure and noise-related inputs. The gains are not uniform, and weak-reference and few-path cases remain important limitations. The method is therefore a useful option for the tested model, rather than a replacement that is better under all conditions.

## References

1. [URformer: author manuscript](https://arxiv.org/html/2509.12586v3).
2. [Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28378).
3. [Self-Supervised Learning with Noisy Dataset for Rydberg Microwave Sensors Denoising](https://arxiv.org/html/2601.01924v1).
4. [Exponential Signal Reconstruction with Deep Hankel Matrix Factorization: author project](https://csrc.xmu.edu.cn/index_cn/project_new/DHMF_Dataset_public/DHMF_EN.html).
5. [Matrix Completion-Informed Deep Unfolded Equilibrium Models for Self-Supervised k-Space Interpolation in MRI](https://arxiv.org/html/2309.13571v1).
