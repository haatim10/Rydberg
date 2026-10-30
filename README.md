# Rydberg array channel-estimation research

**The complete, current research archive is on the [codex branch](https://github.com/haatim10/Rydberg/tree/codex).** This default branch retains the original Track-A code; use the links below for the later receiver, URformer studies and merged manuscript.

The archive was organized on **30 October 2026**. Each section has a README explaining its purpose, code, evidence and limitations. The five topic branches are cumulative: later ones include earlier dependencies.

## Research sections

| Section | Guide | Topic branch |
| --- | --- | --- |
| Foundations and self-supervised learning | [README](https://github.com/haatim10/Rydberg/blob/codex/docs/sections/01-foundations/README.md) | [Foundations](https://github.com/haatim10/Rydberg/tree/research/self-supervised-20261030) |
| Complete receiver and baselines | [README](https://github.com/haatim10/Rydberg/blob/codex/docs/sections/02-receiver/README.md) | [Receiver](https://github.com/haatim10/Rydberg/tree/research/tcom-receiver-20261030) |
| URformer reproduction and matched training | [README](https://github.com/haatim10/Rydberg/blob/codex/docs/sections/03-urformer/README.md) | [Comparison](https://github.com/haatim10/Rydberg/tree/research/urformer-comparison-20261030) |
| SNR and clustered-channel audits | [README](https://github.com/haatim10/Rydberg/blob/codex/docs/sections/04-audits/README.md) | [Audits](https://github.com/haatim10/Rydberg/tree/research/urformer-audits-20261030) |
| Merged TCOM paper | [README](https://github.com/haatim10/Rydberg/blob/codex/docs/sections/05-paper/README.md) | [Paper](https://github.com/haatim10/Rydberg/tree/paper/merged-tcom-20261030) |

## Current paper

The current manuscript is the **21 October 2026, 10-page merged TCOM draft**, *Channel Estimation for Rydberg Arrays With Learned Filtering and Recovery Checks*.

- [Main PDF](https://github.com/haatim10/Rydberg/blob/codex/output/pdf/rydberg_merged_tcom_10page.pdf)
- [Supplement PDF](https://github.com/haatim10/Rydberg/blob/codex/output/pdf/rydberg_merged_tcom_10page_supplement.pdf)
- [Overleaf ZIP](https://github.com/haatim10/Rydberg/blob/codex/output/pdf/rydberg_merged_tcom_10page_overleaf.zip)
- [Source and version index](https://github.com/haatim10/Rydberg/blob/codex/paper/README.md)

This is a draft, not an acceptance or submission record. Later URformer comparisons and the two audits are separate evidence and have not been silently added to the manuscript.

## What is backed up

The archive includes code, papers, reports, figures, protocols, selected/final model weights and compact audit evidence. It is not a complete backup of the approximately 46 GB local workspace. Large generated banks, most raw arrays, intermediate weights and duplicate bundles remain local, as requested; nothing was deleted.

- [Exact included/local-only inventories and verification](https://github.com/haatim10/Rydberg/blob/codex/docs/archive_20261030/README.md)
- [Experiment catalog](https://github.com/haatim10/Rydberg/blob/codex/docs/EXPERIMENT_CATALOG.md)
- [Reproduction guide](https://github.com/haatim10/Rydberg/blob/codex/docs/REPRODUCING.md)

Published URformer accuracy remains unreproduced. The comparison concerns the documented implementation; similar runtime is not a speed advantage. Both later audits retain their negative findings. The proposed uncertainty-based filter has not been implemented.
