# Self-Supervised Channel Estimation with Adaptive Hankel Shrinkage for Rydberg Receivers

Working research draft, 22 September 2026. Not submission-ready. The abstract and conclusion are intentionally deferred until the contribution and remaining checks are settled. The original manuscript and completed experiments remain unchanged. This draft includes a numerical provenance record.

## 1. Introduction

Rydberg receivers use the response of excited atoms to sense radio-frequency fields. They are being studied as an alternative receiver technology for wireless communication. In the magnitude-only receiver model considered here, the receiver measures the strength of the combined user signals and a known reference signal. It does not directly observe the complex wireless channel. Channel estimation must therefore recover information that is hidden by the magnitude operation. Model-based learning has already been applied to this problem, including Transformer-assisted unfolded estimation. [URformer](https://arxiv.org/html/2509.12586v3).

A channel measured across a regular antenna array often has useful spatial structure. A Hankel matrix formed from a channel vector can have low rank when only a few propagation paths contribute. However, a fixed rank can remove real channel details, and repeated rank searches can be expensive. A practical estimator should use this structure without requiring the true path count or labelled channels for training. Our aim is to learn how strongly to shrink the channel's Hankel singular values using information available at the receiver. The relevant question is whether this gives a useful balance between estimation error and processing time, including the conditions where it does not help.

This work makes three contributions:

1. We develop a compact estimator that combines Rician measurement updates with measurement-dependent soft Hankel shrinkage. It has ten stages and 188 trainable parameters, with no deployment-time rank search.
2. We train the estimator using held-out pilot magnitudes rather than true channels. Matched controls test global shrinkage, no structural correction, final-only correction, a fixed-open gate and supervised training.
3. We evaluate accuracy and processing cost on simulated channels, including separate training-data replicates and changes in receiver conditions. We report weak-reference and simple-channel cases where the classical structured estimator remains better.

## 2. Relation to earlier work

The individual ingredients are not new. Separating measured data into reconstruction and loss subsets is an established approach to self-supervised inverse problems. [SSDU](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28378). Self-supervised Rydberg signal denoising also exists, although that task concerns noisy sensor traces rather than the multiuser complex channel considered here. [Rydberg sensor denoising](https://arxiv.org/html/2601.01924v1).

Learned Hankel reconstruction has been used for exponential signals, and self-supervision has been combined with structured-low-rank unfolded reconstruction. [Deep Hankel Matrix Factorization](https://csrc.xmu.edu.cn/index_cn/project_new/DHMF_Dataset_public/DHMF_EN.html), [self-supervised structured-low-rank MRI](https://arxiv.org/html/2309.13571v1). Our method addresses a nonlinear biased-magnitude observation model using a Rician loss and a small controller for an explicit smooth spectral operation. This is a specific algorithmic design, not a claim to introduce self-supervision or learned low-rank reconstruction.

Our companion Paper 1 studies classical Hankel-structured estimation, rank selection and structural abstention. The distinction here is learning soft shrinkage from unlabelled observations and studying its accuracy–cost trade-off. The nearly saturated gate in our experiments does not support a new claim about learning when to abstain. We also do not claim to reproduce or outperform the published URformer results. A full comparison with the recent XL-MIMO DeepTensor conference paper remains an open literature check, so priority claims are excluded from this draft.

## 3. System model

Consider an array with \(N\) elements, \(K\) single-antenna users and \(P\) pilot observations. The channel is \(G\in\mathbb C^{N\times K}\), the known pilot matrix is \(S\in\mathbb C^{K\times P}\), and the known reference is \(B\in\mathbb C^{N\times P}\). After normalizing the receiver conversion gain, the observations are

\[
Z=|GS+B+W|,\qquad W_{np}\sim\mathcal{CN}(0,v).
\]

The magnitude is taken element by element. Noise samples are independent, and \(v\) is the total complex noise variance. Each real noise component has variance \(v/2\). Reference calibration and noise variance are assumed known, except in explicitly labelled mismatch tests. This is a simulation model, not a validated description of every physical Rydberg receiver.

The main channel model uses a half-wavelength uniform linear array:

\[
g_{nk}=\sum_{\ell=1}^{L_k}\alpha_{\ell k}
e^{-j\pi n\sin\theta_{\ell k}},\qquad n=0,\ldots,N-1.
\]

Angles are uniform on \([-\pi/2,\pi/2]\), and independent path gains have distribution \(\mathcal{CN}(0,\beta_k/L_k)\). The primary setting uses \(\beta_k=1\). Pilot entries are circular complex Gaussian with unit variance. The reference has a known array response and constant commanded waveform across pilot columns.

SNR is defined using total average user power, whereas the reference-to-signal ratio, RSR, uses one user's average power. For equal-power users,

\[
\mathrm{SNR}=10\log_{10}(K/v),\qquad
\mathrm{RSR}=10\log_{10}(|\alpha_b|^2/\beta_k).
\]

Thus, an RSR of 12 dB with three users corresponds to about 7.23 dB relative to aggregate average signal power. These definitions matter when comparing experiments from different sources.

The reference carries phase information through its interference with the user signals. It does not by itself guarantee that every channel can be uniquely recovered. We require a full-row-rank context pilot matrix for the channel update. The implementation also requires at least \(2K\) context columns; this check is not a global identifiability theorem.

## 4. Estimator

### 4.1. Measurement update

If \(a=|\mu|\), the magnitude observation has density

\[
p(z\mid a,v)=\frac{2z}{v}
\exp\!\left[-\frac{z^2+a^2}{v}\right]I_0(2za/v).
\]

We evaluate the negative log likelihood using a scaled Bessel function:

\[
\ell(z,\mu,v)=\frac{(z-|\mu|)^2}{v}
-\log I_{0e}(2z|\mu|/v).
\]

An observation-only term is omitted because it is identical for models evaluated on the same measurements. Stable small-field expressions are used near zero. The likelihood does not rely on a strong-reference approximation.

At a channel estimate \(G\), define \(M=GS+B\), \(\kappa=2Z|M|/v\), and

\[
Q=Z\frac{I_1(\kappa)}{I_0(\kappa)}\frac{M}{|M|}.
\]

The continuous limit is used when \(M=0\). The primary model takes a relaxed expectation-maximization update:

\[
G_{\rm EM}=(Q-B)S^H(SS^H)^{-1},\qquad
G_{\rm dc}=G+\gamma_t(G_{\rm EM}-G),
\]

where \(\gamma_t=2\operatorname{sigmoid}(a_t)\) is learned for each stage. The inverse is implemented by a linear solve. A spectral initializer uses only the input observations and their known pilots and reference. The model does not receive a true channel or a true path count.

### 4.2. Soft spatial correction

Each column of \(G_{\rm dc}\) is lifted into a balanced Hankel matrix \(H\). For \(N=32\), this matrix has size \(17\times16\). Let \(q\) be its smaller dimension and \(s_H=\|H\|_F/\sqrt q\). The controller supplies a positive relative threshold, giving \(\tau=s_H[0.001+\operatorname{softplus}(u)]\).

The spectral operation is

\[
\widehat\sigma_i=\frac{\sigma_i^3}{\sigma_i^2+\tau^2},\qquad
\widehat H=H-\tau^2H(H^HH+\tau^2I)^{-1}.
\]

Large singular values are changed less than small ones. Unlike hard rank truncation, this rule does not declare a fixed number of surviving components. Normalized linear solves implement the operation, and gradients pass through those solves rather than through an identity-backward approximation. Antidiagonal averaging maps the result back to a channel vector:

\[
g_k^{t+1}=g_{{\rm dc},k}
+\beta_{t,k}\bigl(\mathcal H^\dagger\widehat H_k-g_{{\rm dc},k}\bigr).
\]

The gate \(\beta_{t,k}\) lies between zero and one. A zero correction recovers the unstructured update, but a nonzero gate does not guarantee lower estimation error. Neither the gate nor the threshold is an estimated physical path count. We do not claim that the spectral rule is the proximal operator of a particular established objective, or that the full relaxed network has a convergence guarantee.

### 4.3. Small shared controller

A shared 8–16–2 network with a SiLU activation produces the threshold and gate. Its inputs describe the current Hankel spectrum, measurement residual, estimated noise-to-channel scale, pilot count, Hankel size and stage number. The three spectral inputs are normalized entropy effective rank, the largest singular value's share, and the lower half's share of the singular-value sum. No generator labels or channel errors are inputs.

The controller has 178 parameters. Ten update scalars give 188 parameters in total. The global control instead learns one relative threshold and gate per stage, giving 30 parameters including the updates. Its absolute threshold still scales with \(s_H\). The unstructured control has ten parameters. These controls test whether conditioning shrinkage on the observations adds value; they do not isolate the necessity of every input feature.

### 4.4. Training without channel labels

Each training example has 30 pilot columns. A random split gives 21 columns to the estimator and reserves nine columns for the loss. Initialization and channel updates use only the 21 input columns. The predicted channel is scored by the Rician likelihood of the nine held-out magnitudes. Splits change each epoch.

Validation uses independent examples and a fixed split. Checkpoint selection gives equal weight to 126 SNR, RSR and path-regime strata. Training uses Adam with learning rate 0.001, batches of 32 and 20 epochs. True channels are not used in the self-supervised loss or selection rule.

Deployment uses all available pilots with frozen weights. No extra observations are collected for adaptation, and no online weight optimization is performed. This changes the pilot-count input from its training value; successful transfer must be measured rather than assumed. Held-out likelihood is a prediction objective, not a guarantee of minimum channel NMSE.

## 5. Evaluation

### 5.1. Protocols and metrics

The main study trains 18 models: two measurement-update cores, three structural variants and three training seeds. They share 16,384 training worlds and 2,048 independent validation worlds. The seeds change initialization, data order and masks, not the underlying channel population. EM is the primary core because it was selected by validation likelihood before test generation.

Training covers SNRs from −10 to 20 dB in 5 dB steps, RSRs of 5, 12 and 20 dB, and several path regimes. Half the examples use random three-to-seven-path channels; the other half use fixed counts of 2, 4, 8, 12 or 16. The primary system has \(N=32\), \(K=3\) and \(P=30\).

The main evaluation has 32 conditions with 400 paired worlds each. Two noise-input variants reuse existing worlds, leaving 12,000 distinct worlds. Comparators include HS-GS, fixed-iteration EM, Rician and amplitude-gradient solvers, stopping-controlled versions, a hard-Hankel control and linearized least squares. HS-GS includes its rank search in processing cost. A stopping-controlled run reaching its 1,000-update cap is not labelled converged.

For each condition and training seed, channel NMSE is

\[
10\log_{10}\frac{\sum_i\|\widehat G_i-G_i\|_F^2}
{\sum_i\|G_i\|_F^2}.
\]

We average these dB scores across seeds, not their predicted channels. The primary score averages seven SNR-cell scores equally. Positive gain means the comparator's score minus the proposed method's score. Severe errors have per-world NMSE above one.

The main intervals use paired resampling of whole worlds and are pointwise, conditional on trained weights. They do not measure uncertainty across new training datasets. A separate follow-up contains 15 newly trained models and two fresh evaluation banks, each with seven primary cells and two stress cells of 200 worlds. It includes matched supervised, final-only and fixed-gate controls, plus two independently generated training-data replicates. Its six primary contrasts use multiplicity-adjusted intervals within each bank. The follow-up is not the earlier proposed five-dataset confirmation.

### 5.2. Main accuracy and processing cost

| SNR (dB) | Proposed NMSE (dB) | Gain over HS-GS (dB) | Gain over global shrinkage (dB) |
|---:|---:|---:|---:|
| −10 | −1.254 | 4.696 | 5.693 |
| −5 | −4.122 | 1.687 | 3.289 |
| 0 | −9.191 | 1.620 | 2.316 |
| 5 | −14.348 | 1.422 | 1.342 |
| 10 | −19.171 | 1.101 | 0.316 |
| 15 | −24.260 | 1.062 | 0.103 |
| 20 | −28.563 | 0.139 | 0.848 |

The mean gain over HS-GS is 1.675 dB, with a conditional paired-world interval of [1.618, 1.730] dB. The seed-specific mean gains range from 1.533 to 1.787 dB. At 20 dB SNR, the gain interval is [−0.017, 0.294] dB, so that cell does not establish a positive improvement.

At the declared timing condition of SNR 5 dB and RSR 12 dB, median processing times are 8.24 ms for the proposed method and 525.62 ms for HS-GS. This is a 63.8-fold measured speedup. Timing includes initialization and estimator processing, uses serial batch-one execution on an Apple M2 CPU with one thread and double precision, and covers 100 worlds with three repeats. It is an implementation-specific result, not a hardware or GPU speed guarantee. Offline training is a separate cost.

### 5.3. Fresh-data checks and ablations

| Contrast | Bank 1 mean gain (dB) | Bank 2 mean gain (dB) |
|---|---:|---:|
| Independent-data adaptive vs global shrinkage | 2.252 | 2.185 |
| Independent-data adaptive vs no structure | 4.250 | 4.191 |
| Independent-data adaptive vs HS-GS | 1.939 | 1.837 |
| Original adaptive vs matched supervised recipe | 1.015 | 1.012 |
| Original adaptive vs final-only correction | 0.697 | 0.694 |
| Fixed-open gate vs original adaptive | 0.064 | 0.066 |

The independent-data results support repeatability beyond the original training bank. Each new bank uses one fixed training seed, so dataset and initialization variation are not fully crossed.

The supervised comparison uses the same architecture, context inputs, example budget and training schedule, but minimizes linear channel NMSE. It selects checkpoints by validation channel error. Selecting its checkpoint by held-out likelihood instead reduces the proposed method's advantage to about 0.73 dB. Pooling linear error across SNRs instead of giving equal weight to cell dB scores reduces the advantage to about 0.36 dB. The supervised model is also better on the two-path stress cases. These findings do not support a general claim that self-supervision is better than supervision.

A post-hoc validation replay provides further context. The supervised models perform better on their own channel objective when both methods receive 21 input pilots. Their transfer to all 30 pilots is less consistent across seeds. Approximately 93% of their validation channel-loss value comes from SNR at or below 0 dB. Objective weighting and input-length transfer are therefore relevant limitations of this comparison, not established causal explanations.

Repeated structural correction improves accuracy over final-only processing, but costs more. In the follow-up timing sample, the original model takes about 4.65 ms and the final-only model about 2.05 ms. These timings form a separate within-study comparison and are not substituted into the main speedup calculation.

The learned gate is nearly one in the main study. Fixing it at one slightly improves the average, but worsens two of three seeds on the two-path stress cases. The fixed-gate implementation has not been pruned and retains 188 stored parameters. The evidence supports learned shrinkage strength more clearly than a learned on/off decision.

### 5.4. Operating limits

HS-GS remains better in important cases. In the main study, the proposed model loses 1.617 dB at RSR 0 dB, 1.104 dB with two paths, and 0.203 dB with three paths. Exact treatment of magnitude noise does not remove weak-reference ambiguity or optimization difficulty.

Positive transfer results include the tested clustered, ten-path and unequal-power channels, and supplied-noise errors of ±3 dB. Results also vary with array size and user count; not every transfer cell has a clear advantage. The path-count sweep partly overlaps training support and must not be described as wholly unseen channel complexity. These tests do not establish that the method resolves the independent 40-ray failures found in earlier, different protocols.

## 6. Discussion and limitations

The main finding is a useful simulated accuracy–cost trade-off from learning a small structural controller rather than a large channel-to-channel mapping. Adaptive shrinkage outperforms the matched global control in the primary average and in two additional training-data replicates. The available experiments do not yet separate the value of spectral features from that of noise and residual features.

The model assumes known reference calibration and usually known noise variance. It has no hardware validation, no global identifiability proof and no convergence proof for the complete learned iteration. Three seeds sharing a dataset are not three independent channel populations. Processing times depend on implementations, and training cost must accompany any eventual efficiency claim.

The method should therefore be presented as a compact label-free estimator with measured benefits and explicit failure regions. It should not be presented as universally better than classical estimation, as an improvement to reproduced published URformer performance, or as the first use of self-supervised low-rank learning.

## Draft evidence and completion notes

These notes are for manuscript preparation and are not intended as a journal section.

- Main protocol, accuracy, transfer and timing: `results/sahure_expansion/REPORT.md`, `summary.json`, `frozen_selection.json`, `audit.json` and `timings.jsonl`.
- Follow-up controls and independent-data checks: `results/sahure_checks/REPORT.md`, `summary.json`, `protocol.json`, `timing.json` and `audit.json`.
- New metric, checkpoint, input-length and gate review: `results/sahure_submission_review/REVIEW.md` and `review.json`.
- Primary method decision: `MODEL_DECISION.md` in this draft directory. Closest-work comparison: `results/sahure_submission_review/NOVELTY.md`.
- Required additions: feature ablation, complete closest-work review, final statistical and figure review, training-cost table, full bibliography, qualitative abstract and conclusion. A stronger supervised claim requires deployment-matched and objective-matched controls on fresh evaluation data.
