# Focused prior-work comparison

Reviewed 22 September 2026. This is a scoped literature audit, not a proof of absence of earlier work. The differences listed are methodological distinctions, not guarantees of publishable novelty.

| Prior work and source | Verified overlap | Relevant distinction and claim restriction |
|---|---|---|
| [URformer, author manuscript v3](https://arxiv.org/html/2509.12586v3) | Unfolded Rydberg channel estimation with learned processing and a Transformer. | Our finite-depth model retains Rician processing and learns a small spatial shrinkage controller without channel labels. Published URformer performance has not been reproduced; no superiority claim against that published method. |
| [Liu et al., Rydberg sensor denoising](https://arxiv.org/html/2601.01924v1) | Label-free training from noisy signal pairs for experimental Rydberg signal denoising. | Our task is multiuser complex-channel inversion from pilot-indexed magnitudes. We have simulation evidence only. “First self-supervised Rydberg method” is ruled out. |
| [Yaman et al., SSDU](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28378) | Separating measured data into reconstruction and loss subsets already enables label-free physics-guided learning. | We use a nonlinear magnitude observation model and a Rician loss. Pilot splitting itself is not a contribution. |
| [Luo et al., self-supervised structured-low-rank MRI](https://arxiv.org/html/2309.13571v1) | Structured-low-rank reconstruction, self-supervision and unfolding already coexist. Equations 3–6 use a linear sampling model, learned null-space filtering and projection onto measured complex samples. Equation 14 compares predictions with the available measurement set. | Our observation is a biased magnitude, not a known complex sample. We use explicit smooth singular-value attenuation and ten finite updates, not their fixed-point construction. Their stability/convergence results do not transfer to this model. The methods comparison was inspected beyond the abstract. |
| [Huang et al., Deep Hankel Matrix Factorization: author project](https://csrc.xmu.edu.cn/index_cn/project_new/DHMF_Dataset_public/DHMF_EN.html) | Learned Hankel reconstruction of exponentials with factor updates and a data-consistency module predates this project. The project describes synthetic training and NMR evaluation. | Our explicit spectral rule is controlled by measurement features and trained on held-out magnitudes. Neither “learned Hankel” nor “fast Hankel reconstruction” alone is new. This pass inspected the author project, not every equation in the final journal version. |
| [DeepTensor](https://arxiv.org/abs/2204.03145) | Self-supervised low-rank decomposition using deep-network priors is established. | Our shared controller amortizes estimation across observations rather than fitting generated factors for each object. Abstract-level comparison only in this pass. |
| Companion Paper 1, local manuscript | Hankel structure, adaptive rank selection and structural abstention are already part of this project's existing work. | Paper 2 must emphasize label-free learning of soft shrinkage and measured processing cost. It must not recycle rank selection as the new contribution. Publication status of Paper 1 is not assumed. |

## Unresolved closest work

The [official CCAI 2026 program](https://ccai.net/CCAI2026-Program.pdf) lists *Self-Supervised DeepTensor for Low-Rank Channel Estimation in XL-MIMO Systems*. The associated [publisher DOI](https://doi.org/10.1109/CCAI69603.2026.11642005) did not yield accessible full text during this review. Its exact forward model, training method, controls and claims remain unverified. A program title or aggregator summary is not enough to establish the distinction.

## Provisional contribution boundary

Investigate a compact, measurement-conditioned soft Hankel controller trained from held-out Rician observations for magnitude-only multiuser channel estimation. Support the claim through matched controls and explicit operating limits. Do not claim a new Hankel rank law, a new self-supervision principle, guaranteed rank recovery, global convergence, superiority to all supervised estimators, or hardware validation.

The current comparisons support an application-specific algorithm paper more directly than a fundamental new learning principle. The distinctive benefit still needs to survive a feature ablation and a complete closest-work check.
