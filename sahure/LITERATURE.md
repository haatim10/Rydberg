# IEEE-focused overlap matrix

Checked 21 September 2026. Entries below are IEEE publications. Author
manuscripts, author repositories and university records are used where
publisher full text was inaccessible. This is an overlap audit, not proof
that no related method exists.

| Work | Established component | Difference being tested here | Review depth / source |
|---|---|---|---|
| Xiao et al., *Channel Estimation for Rydberg Atomic Quantum Receivers: Unrolled Phase Retrieval From Holographic Snapshots*, IEEE SPL 33, 1696–1700, 2026 | EM-GS unfolding, learned Bessel filtering, gating, channel Transformer | Keep the exact Bessel/Rician data core; learn a small shared spatial Hankel correction from held-out pilot likelihood | [IEEE publisher record](https://doi.org/10.1109/LSP.2026.3685170), [author manuscript v3](https://arxiv.org/abs/2509.12586v3). Architecture verified; no new head-to-head reproduction in this pilot. |
| Xu et al., *Channel Estimation for Rydberg Atomic Receivers*, IEEE WCL 14(9), 2957–2961, 2025 | Strong-reference linearization and PGD; 2-D slice rank constraints | All methods receive exact magnitude observations. Compare the solved 1-D linearized objective; the new spatial prior is a 1-D Hankel lift | [IEEE DOI](https://doi.org/10.1109/LWC.2025.3584040), [author manuscript](https://arxiv.org/abs/2503.08985), supplied local PDF. Equations 10–13 concern linearization/1-D objective; the 2-D slice constraint is distinct. |
| Yaman et al., *Ground-truth Free Multi-mask Self-supervised Physics-guided Deep Learning in Highly Accelerated MRI*, IEEE ISBI 2021, 1850–1854 | Disjoint measurement subsets for data consistency and self-supervised loss | Transfer the principle to biased magnitude observations with held-out Rician NLL | [IEEE DOI](https://doi.org/10.1109/ISBI48211.2021.9433924), [author institutional record](https://experts.umn.edu/en/publications/ground-truth-free-multi-mask-self-supervised-physics-guided-deep-/), [author manuscript](https://par.nsf.gov/servlets/purl/10235064). Splitting principle verified; no claim that measurement splitting is new. |
| Ke et al., *Learned Low-Rank Priors in Dynamic MR Imaging*, IEEE TMI 40(12), 3698–3710, 2021 | Learned singular-value processing in model-based reconstruction | A measurement-conditioned threshold/gate for per-user spatial Hankel matrices, using a smooth rational shrinkage and exact Rician data processing | [IEEE DOI](https://doi.org/10.1109/TMI.2021.3096218), [authors' implementation](https://github.com/Keziwen/SLR-Net), [coauthor publication page](https://wenqihuang.me/). Abstract/code-level overlap verified; not an exhaustive full-paper comparison. |

## Claim boundary

The observation model, Rician likelihood, unrolling, learned singular-value
processing, and measurement-split self-supervision each have precedent.
The candidate contribution is their narrowly specified combination: a small
shared measurement-conditioned spatial correction, trained without channel
labels, that can replace exhaustive per-realization HS-GS rank search.

Whether that is useful depends on accuracy and measured computation against
HS-GS and simpler globally learned shrinkage. The generated pilot report
supplies the current empirical assessment. If the adaptive model fails the
controls, remove the adaptation claim. A fixed-SNR pilot cannot establish a
publication-level robustness claim, and no "first self-supervised Rydberg
receiver" claim is justified by this matrix.

Search terms included IEEE/Rydberg/channel estimation/unfolding,
IEEE/Hankel/self-supervised/channel estimation, and the four exact titles.
Unverified search hits and non-IEEE papers were not treated as IEEE evidence.
Before a manuscript claims novelty, extend citation tracing around these
works and compare the finalized algorithm against their full texts.
