# Rydberg array channel-estimation research

Research code, experiment records, trained weights and manuscripts for estimating a
wireless channel from magnitude-only Rydberg-array measurements,
`Z = |G S + B + W|`, with known pilots `S` and a known reference `B`
(see [SystemModel.pdf](SystemModel.pdf)).

**Start with [`archive/complete`](https://github.com/haatim10/Rydberg/tree/archive/complete).**
It holds the full organized archive. This default branch keeps only the original
Track-A simulation code (`rydberg_sim/`, `tests/`, `results/track_a/`).

## Current paper

The current manuscript is the **21 October 2026, 10-page merged TCOM draft**,
*Channel Estimation for Rydberg Arrays With Learned Filtering and Recovery Checks*.
It is a draft, not a submission or acceptance record.

- [Main PDF](https://github.com/haatim10/Rydberg/blob/archive/complete/output/pdf/rydberg_merged_tcom_10page.pdf)
- [Supplement PDF](https://github.com/haatim10/Rydberg/blob/archive/complete/output/pdf/rydberg_merged_tcom_10page_supplement.pdf)
- [Overleaf ZIP](https://github.com/haatim10/Rydberg/blob/archive/complete/output/pdf/rydberg_merged_tcom_10page_overleaf.zip)
- [Source and version index](https://github.com/haatim10/Rydberg/blob/archive/complete/paper/README.md)

## Branches

Every branch has a `README.md` at its root that says what it holds and its former name.

### Main line

| Branch | Contents |
| --- | --- |
| `main` | Default branch: this index and the original Track-A code |
| [`archive/complete`](https://github.com/haatim10/Rydberg/tree/archive/complete) | **Complete archive.** Every study, receiver, URformer comparison, audits, paper, selected weights and the file inventory |

### Topic checkpoints

These are cumulative: each one adds a topic to the one before, and all of them are
contained in `archive/complete`. Use one when you want a smaller checkout that stops
at a given topic.

| Branch | Adds | Guide |
| --- | --- | --- |
| [`research/01-foundations`](https://github.com/haatim10/Rydberg/tree/research/01-foundations) | Classical structured estimation, the small learned controller, self-supervised training, earlier studies | [README](https://github.com/haatim10/Rydberg/blob/archive/complete/docs/sections/01-foundations/README.md) |
| [`research/02-receiver`](https://github.com/haatim10/Rydberg/tree/research/02-receiver) | Complete receiver: EM continuation, high-SNR filtering, recovery checks, baselines, NMSE/BER/timing | [README](https://github.com/haatim10/Rydberg/blob/archive/complete/docs/sections/02-receiver/README.md) |
| [`research/03-urformer-comparison`](https://github.com/haatim10/Rydberg/tree/research/03-urformer-comparison) | URformer reproduction attempt and matched-supervision comparison | [README](https://github.com/haatim10/Rydberg/blob/archive/complete/docs/sections/03-urformer/README.md) |
| [`research/04-urformer-audits`](https://github.com/haatim10/Rydberg/tree/research/04-urformer-audits) | SNR-convention and clustered-channel audits (negative findings kept) | [README](https://github.com/haatim10/Rydberg/blob/archive/complete/docs/sections/04-audits/README.md) |
| [`paper/tcom-merged`](https://github.com/haatim10/Rydberg/tree/paper/tcom-merged) | Merged TCOM manuscript, supplement and Overleaf package | [README](https://github.com/haatim10/Rydberg/blob/archive/complete/docs/sections/05-paper/README.md) |

### Separate work not in the archive

These branches hold files that exist nowhere else. Keep them.

| Branch | Contents |
| --- | --- |
| [`track-a/cui-reproduction`](https://github.com/haatim10/Rydberg/tree/track-a/cui-reproduction) | Cui et al. figure reproduction (Figs. 4-8), reproduction audit, clean final figures and the technical report PDF |
| [`track-a/early-modules`](https://github.com/haatim10/Rydberg/tree/track-a/early-modules) | Earliest Track-A code with root-level `system.py`, `algorithms.py`, `final_baseline.py` |
| [`paper/paper1-restructure`](https://github.com/haatim10/Rydberg/tree/paper/paper1-restructure) | Paper 1 advisor restructure and the SPL2 structural-priors draft |
| [`paper/paper2-style-rewrite`](https://github.com/haatim10/Rydberg/tree/paper/paper2-style-rewrite) | Paper 2 rewritten in Paper 1's style, with a number-invariance checker |

### History snapshots

Read-only. Everything in these is also in `archive/complete`; they keep older
commits easy to find.

| Branch | Snapshot |
| --- | --- |
| [`history/fig5-norm-diag`](https://github.com/haatim10/Rydberg/tree/history/fig5-norm-diag) | 18 Aug 2026: Track-A Fig. 5 wrap-up and row-normalization diagnostic |
| [`history/track-b-exact`](https://github.com/haatim10/Rydberg/tree/history/track-b-exact) | 21 Aug 2026: Track-B clean B1/B2 figures |
| [`history/track-b-ula`](https://github.com/haatim10/Rydberg/tree/history/track-b-ula) | 22 Aug 2026: Track-B ULA channel estimation and technical report |
| [`history/report-corrections`](https://github.com/haatim10/Rydberg/tree/history/report-corrections) | 26 Aug 2026: technical-report figure corrections |
| [`history/adversarial-audit`](https://github.com/haatim10/Rydberg/tree/history/adversarial-audit) | 26 Aug 2026: adversarial audit and CCRB regeneration |
| [`history/paper1-review`](https://github.com/haatim10/Rydberg/tree/history/paper1-review) | 15 Sep 2026: Paper 1 review prompts up to P22 Part E |

### Former names

The branches were renamed on 30 October 2026. Each renamed branch is the old branch
plus one commit that adds its README and updates links. Dated records inside the
archive keep the names used when they were written.

| Former name | New name |
| --- | --- |
| `codex` | `archive/complete` |
| `research/self-supervised-20261030` | `research/01-foundations` |
| `research/tcom-receiver-20261030` | `research/02-receiver` |
| `research/urformer-comparison-20261030` | `research/03-urformer-comparison` |
| `research/urformer-audits-20261030` | `research/04-urformer-audits` |
| `paper/merged-tcom-20261030` | `paper/tcom-merged` |
| `track-a-cui-reproduction` | `track-a/cui-reproduction` |
| `cursor/ula-config-channel-8c77` | `track-a/early-modules` |
| `paper1-restructure` | `paper/paper1-restructure` |
| `paper2-style` | `paper/paper2-style-rewrite` |
| `claude/track-b-exact` | `history/track-b-exact` |
| `track-b-ula-channel-estimation` | `history/track-b-ula` |
| `report-corrections` | `history/report-corrections` |
| `claude/adversarial-audit-gs6iid` | `history/adversarial-audit` |
| `paper1-review`, `claude/new-paper-implementation-fs4ako` | `history/paper1-review` |
| `cursor/fig5-norm-diag-8c77` | `history/fig5-norm-diag` |

[`tools/delete_old_branches.sh`](tools/delete_old_branches.sh) removes the former
names from GitHub after checking that each new branch is built on the old commit.

## What is backed up

The archive includes code, papers, reports, figures, protocols, selected/final model
weights and compact audit evidence. It is not a complete backup of the approximately
46 GB local workspace; large generated banks and most raw arrays remain local.

- [Included/local-only inventories and verification](https://github.com/haatim10/Rydberg/blob/archive/complete/docs/archive_20261030/README.md)
- [Experiment catalog](https://github.com/haatim10/Rydberg/blob/archive/complete/docs/EXPERIMENT_CATALOG.md)
- [Reproduction guide](https://github.com/haatim10/Rydberg/blob/archive/complete/docs/REPRODUCING.md)

Published URformer accuracy remains unreproduced. Both later audits retain their
negative findings. The proposed uncertainty-based filter has not been implemented.

## This branch: Track-A code

```bash
pip install numpy scipy matplotlib pytest
```

```bash
pytest
```

[results/track_a/README.md](results/track_a/README.md) describes the Track-A results.
