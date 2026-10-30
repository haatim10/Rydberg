# Focused related-work check

Checked 23 September 2026. This records inspected sources, not a guarantee of exhaustive novelty.

| Work | Evidence inspected | Relevant comparison |
|---|---|---|
| [URformer](https://arxiv.org/abs/2509.12586) | Author v3 abstract and the earlier saved method review | Same receiver problem class; learned unfolded processing already exists. The present method retains the Rician field update and uses a small shrinkage controller. Published URformer performance is not a reproduced comparator. |
| [SSDU](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28378) | Earlier primary-paper review retained | Training using separate reconstruction and loss measurement subsets is established. Pilot splitting is not the novelty claim. |
| [Rydberg sensor denoising](https://arxiv.org/abs/2601.01924) | Author manuscript metadata and earlier saved review | Self-supervised Rydberg sensing already exists. Here the output is a multiuser complex channel, rather than a denoised sensor trace. No hardware comparison is made. |
| [Deep Hankel matrix factorization](https://arxiv.org/pdf/2007.06246) | Author full text, model and loss in Sections III-IV, especially Eq. (14) | Learns factor updates using fully sampled signal targets. Our channel estimator instead trains on withheld noisy magnitudes. Learned Hankel recovery and speed improvements are not new by themselves. |
| [Self-supervised structured-low-rank MRI](https://arxiv.org/html/2309.13571v1) | Forward model and reconstruction, Eqs. (3)-(6), and self-supervised formulation | Uses sampled complex measurements and learned null-space filtering. Its reconstruction guarantees cannot be transferred to the present nonlinear magnitude model. |
| [Original DeepTensor](https://arxiv.org/pdf/2204.03145) | Full-text Section 3, Eqs. (6)-(7) | Networks generate low-rank factors fitted to an observed object. Here a shared controller is trained across cases and its weights remain fixed at inference. This does not establish an advantage over DeepTensor, which was not benchmarked. |
| [Rician self-supervised MRI](https://arxiv.org/pdf/2307.07072) | Full-text theory and noise model | Statistically appropriate magnitude likelihoods in label-free learning predate this work. The receiver-specific combination and empirical behavior, not the likelihood alone, are the possible contribution. |
| Paper 1 | Existing companion manuscript and decision record | Classical Hankel filtering, rank selection and structural abstention are prior project work. Paper 2 needs to stand on label-free soft-filter control and its evidence, not recycled rank-selection claims. |

## Unresolved conference paper

The [official CCAI program](https://ccai.net/CCAI2026-Program.pdf), p. 22, lists *Self-Supervised DeepTensor for Low-Rank Channel Estimation in XL-MIMO Systems*, poster P14, presented by Zhenkun Tan. The [publisher landing page](https://ieeexplore.ieee.org/document/11642005), [DOI](https://doi.org/10.1109/CCAI69603.2026.11642005) and publisher PDF endpoint did not provide readable full text in this pass. Exact-title and author searches did not locate an accessible primary manuscript. A third-party abstract was not used to certify detailed differences. No payment, author contact or access-control bypass was attempted.

The paper cites the verified program for the existence of this work, not as evidence about its equations. The general manuscript retains an explicit limitation. Replacing that with a full-paper comparison remains a scientific release condition, not an editorial formality.

## Publishing disclosure

[IEEE's AI-content guidance](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/) calls for disclosure of the system, affected sections and extent of generated content. The acknowledgment identifies Codex and its involvement in text, mathematical exposition, code and figure preparation. The author must check its accuracy and the eventual venue's current rules. No human-review attestation has been made on the author's behalf.
