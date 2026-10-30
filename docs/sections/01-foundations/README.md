# Foundations and earlier studies

This section contains the work leading to the merged receiver: classical
structured estimation, small learned filters, validation of the training method,
and alternatives that were tested but not adopted. These are separate campaigns,
not additional samples of one final experiment.

## Main components

| Path from repository root | Purpose |
| --- | --- |
| `rydberg_sim/` | Shared simulator and original estimation methods |
| `trackB_hankel_emgs/` | Classical Hankel-structured EM/GS work |
| `trackD_urformer/` | Earlier URformer/HS-URformer adaptation; distinct from the October v3 reproduction |
| `sahure/` | Exact Rician mathematics, adaptive controller, training and inference |
| `sahure_expansion/`, `sahure_checks/`, `sahure_features/` | Expanded evaluations, checks and feature ablations |
| `p1_revision/`, `p1_followup/` | Paper 1 revisions and supporting experiments |
| `paper2_review_checks/`, `paper2_review_round2/` | Paper 2 review responses and additional checks |
| `paper2_information/`, `scripts/paper2_*.py` | Earlier information, learning, reproduction and confirmation studies |
| `compact_baseline/`, `*optshrink*/`, `lrhd_screen/`, `rician_*/` | Alternative models, shrinkage methods and likelihood-based comparisons |

Use the [experiment catalog](../../EXPERIMENT_CATALOG.md) for report links and the
matching `results/` directories. Failed and negative comparisons are retained.

## How the learned core works

The core alternates an exact Rician EM channel update with a spatial Hankel
filter. An 8-input, 16-hidden-unit, 2-output network chooses a threshold scale
and mixing weight from observable features. Ten learned step sizes complete
the 188-parameter adaptive model. A fixed schedule is a separate control.

The singular-value map is `s -> s^3 / (s^2 + tau^2)`. The controller chooses
`tau` and `beta`; it does not directly output every complex channel entry.
The structure assumes a linear antenna array and a channel that can usefully
be described by a small number of spatial components.

The final main-paper training uses 30 input pilots and 9 withheld pilot targets.
The estimated channel predicts the withheld magnitudes; their Rician likelihood
provides the training loss. True channels generate simulations and score NMSE,
but are not the self-supervised neural training targets. Earlier pilots and
the matched URformer study have their own recorded training sizes and settings.

## Read and verify

- [Core README](../../../sahure/README.md): interface, original pilot and numerical checks.
- [Mathematics](../../../sahure/MATHEMATICS.md): statistical model and operators.
- [Original literature discussion](../../../sahure/LITERATURE.md): historical claim boundaries.
- [Reproduction guide](../../REPRODUCING.md): environment and archive verification.

Older READMEs describe the stage at which they were written. For example, the
original pilot's deferred larger campaign was later carried out elsewhere;
the pilot README itself is preserved as a dated record.

Selected epoch weights are included even when an older study did not call its
file `best.pt`. The archive inventory distinguishes validation-selected epochs
from the last saved epoch. A last saved epoch does not by itself prove that a
run reached its originally planned budget; inspect its completion record.
