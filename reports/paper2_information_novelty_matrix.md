# Novelty boundary: measurement-informed structural correction

Primary-source pages checked 21 September 2026. This is an initial overlap audit, not a certified exhaustive novelty search. “Not established here” means the current source inspection is insufficient, not that the paper lacks the property. Abstract-level sources do not settle detailed theorem assumptions.

| Work | Observations / assumptions | Learned or structural component | Guarantee / evaluation evidenced by source | Overlap and distinction still to establish |
|---|---|---|---|---|
| [URformer, v3 May 2026](https://arxiv.org/abs/2509.12586) | Biased phase retrieval from Rydberg holographic snapshots | Learned Bessel filter, gate and Transformer correction in unrolled EM-GS | Numerical channel-estimation study; theorem claims not assessed here | Unrolling and Transformer correction are established. New method must add measurement-geometry protection, not relabel this architecture. Our reconstruction is not verified author-code reproduction. |
| [Atomic hybrid sparse/diffuse estimation, May 2026](https://arxiv.org/abs/2605.01825) | Frequency-domain channel with sparse and diffuse parts | Atomic norm plus quadratic regularization; off-grid delay estimation | Dual analysis, channel-parameter CRBs; synthetic and real data | Structured-plus-diffuse modelling is not new. “Atomic” means atomic norm, not a Rydberg receiver. Detailed nonlinear magnitude-model distinctions and identifiability must be checked before using backup B. |
| [Structure-informed pilot-limited MIMO, September 2026](https://jcis.sbrt.org.br/jcis/article/view/963) | Sparse pilot observations of wideband MIMO channel tensors | CP/Tucker low-rank completion and a 3D U-Net residual | Synthetic and DeepMIMO evaluation; abstract describes diffuse/hardware residual recovery | Low-rank reconstruction plus learned residual already exists. A residual network alone cannot be our novelty. Exact assumptions and baseline fairness need full-text review. |
| [Data-consistent networks for nonlinear inverse problems](https://arxiv.org/abs/2003.11253) | Known nonlinear forward map and data-driven solution manifold | Classical regularization plus data-consistent neural network | Source describes convergence with noise and stability to structural changes | Protecting observations has substantial prior art. Our local Fisher penalty is softer and does not inherit their full convergence result. Need a concrete problem-specific advantage over data-consistent baselines. |
| [Deep-SLR](https://arxiv.org/abs/1912.03433) | Structured low-rank inverse problems, developed for imaging | Learned generalization of structured low-rank algorithms | Application-specific reconstruction study; formal properties require full-text audit | Learning a structural operation is insufficient novelty. The current prototype is nonlearned and specifically tests directional measurement information. |
| [Rician likelihood self-supervised learning](https://arxiv.org/abs/2307.07072) | Magnitude data with Rician noise in quantitative MRI | Statistically appropriate self-supervised loss | Application-specific bias/estimation analysis | A Rician loss alone is not new. Here it supplies local expected score information; useful distinction still requires direct literature and empirical comparison. |
| [Invertible ResNets for inverse imaging](https://epubs.siam.org/doi/10.1137/24M1696524) | Inverse imaging and invertible residual networks | Learned reconstruction with regularization properties | Listed in the approved plan; detailed assumptions not independently rechecked in this implementation turn | Do not borrow a global stability/convergence claim for a one-step local filter. Full theorem comparison remains a Week 1 task. |
| Paper 1, local `rydberg_sim/track_b_proposed.py` | Same magnitude receiver model; native spectral start and pilot holdout | Adaptive Hankel rank with abstention at the rank cap | Existing local experimental evidence, native implementation retained | Rank choice, skipping projection and showing sparse-channel gains are already Paper 1. Compare directly; our candidate difference is direction-specific attenuation at fixed structural operation, not a new rank selector. |

## Candidate distinction to test, not to announce

Use measurement information to protect well-observed channel directions while allowing a smooth structural correction in less-observed directions. The intended evidence must show that **measurement geometry**, not simply SNR, correction strength, model rank or training quality, explains an advantage across matched channels and pilots.

The formula (I+tau J)^(-1)d is a standard quadratic-penalty solution. Its appearance in a Rydberg estimator is not by itself sufficient for publication. A defensible contribution requires a new useful mechanism, careful distinctions from existing inverse-problem regularization, and independent tests against strong controls.

## Remaining focused searches before a novelty claim

- Fisher-metric / Gauss–Newton metric priors and anisotropic regularization in phase retrieval.
- Null-space networks and nonlinear data-consistency corrections, including approximate Jacobian null-space methods.
- Plug-and-play and posterior/covariance-aware denoisers with measurement-preserving components.
- Adaptive singular-value shrinkage, low-rank bias control and information-directed regularization in channel estimation.
- Closest references of the six directly checked papers, and an exact claim-by-claim comparison with the final Paper 1 text.

These are unresolved work items, not evidence of an empty research area. No “first,” “never done before,” or verified novelty statement is authorized by this matrix.
