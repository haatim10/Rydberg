# Focused novelty check, 21 September 2026

This is a live research assessment, not certification of novelty. Existing
SAHURE literature notes remain unchanged. Primary sources inspected during
the submission-check setup:

| Work | Verified overlap | Difference that still needs defending |
|---|---|---|
| Liu et al., [Self-Supervised Learning with Noisy Dataset for Rydberg Microwave Sensors Denoising](https://arxiv.org/html/2601.01924v1), 2026 | Self-supervised Rydberg sensing already exists. Their methods section describes noisy-to-noisy experimental signal pairs and neural denoising of time/frequency traces, rather than requiring clean targets. | Our proposed task is multiuser complex-channel inversion of pilot-indexed biased magnitudes with a Rician observation model. It is not a claim to introduce self-supervision to Rydberg sensors. Their experimental validation is stronger than our simulation-only evidence in that respect. |
| Luo et al., [Matrix Completion-Informed Deep Unfolded Equilibrium Models for Self-Supervised k-Space Interpolation in MRI](https://arxiv.org/abs/2309.13571), 2023 | Self-supervised structured-low-rank reconstruction and unfolding already coexist. Their abstract describes nonexpansive network constraints and fixed-point reconstruction. | Our small finite-depth measurement-conditioned shrinkage operates on spatial channel Hankel matrices under nonlinear magnitude observations. We do not inherit their convergence guarantees. Full equation-by-equation comparison remains open. |
| [Exponential Signal Reconstruction with Deep Hankel Matrix Factorization](https://arxiv.org/abs/2007.06246), 2020 | Learned unrolling with low-rank Hankel factorization predates this project. | Smooth shrinkage, conditioning features and the held-out magnitude likelihood must be compared specifically; adding Hankel to a network is not novelty. Abstract-level inspection, not a complete architecture comparison. |
| [DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors](https://arxiv.org/abs/2204.03145), 2022 | Self-supervised low-rank factorization with deep-network priors exists. | Our amortized shared controller is different from fitting network-generated factors, but a new application alone does not establish a strong contribution. Abstract-level inspection. |
| Xiao et al., [URformer author manuscript](https://arxiv.org/abs/2509.12586) | Learned unfolded Rydberg channel estimation already exists. | Preserve exact Rician data processing and learn a small spatial structural controller without channel labels. No claim of reproducing or beating published URformer is permitted by the current experiments. |

The [official CCAI 2026 program](https://ccai.net/CCAI2026-Program.pdf) also lists
*Self-Supervised DeepTensor for Low-Rank Channel Estimation in XL-MIMO Systems*.
This verifies the existence/title, not detailed method equivalence. The full
publisher/author text was not obtained in this pass, so its exact overlap
remains unresolved; do not rely on a search-aggregator description as proof.

## Safe provisional claim

Test whether a small, measurement-conditioned soft Hankel controller, learned
from held-out Rician observations, offers a useful channel-estimation
accuracy/cost tradeoff versus rank-search HS-GS and simpler matched controls.
No first-ever self-supervised Rydberg, learned Hankel, or low-rank-unrolling
claim. No theoretical convergence claim from the finite-depth network.

## Still required before submission

Full-text equation/assumption comparisons to the closest structured-low-rank
self-supervised methods; forward/backward citation tracing; full XL-MIMO
conference paper; Paper 1 distinction; matched supervised/final-only/fixed-gate
results and fresh-dataset replication. This note records the search honestly
rather than treating missing search hits as evidence of absence.
