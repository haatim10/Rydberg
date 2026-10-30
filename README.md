# Rydberg array channel estimation

> **Branch `archive/complete`** (formerly `codex`, base commit `832b9558adbf`).  
> The complete research archive (organized 30 October 2026): every study, the complete receiver, URformer comparison and audits, the merged TCOM paper, selected weights and the archive inventory. **Start here.**  
> All branches: [branch index on `main`](https://github.com/haatim10/Rydberg#branches).

Research code, experiment records, trained weights and manuscripts for estimating
a wireless channel from magnitude measurements and a known reference signal.

**Start with the [`archive/complete` branch](https://github.com/haatim10/Rydberg/tree/archive/complete).**
The research archive was organized on **30 October 2026**. The newest merged
manuscript is the **21 October 2026, 10-page TCOM draft**. Later URformer studies
are separate evidence; they have not been silently added to that manuscript.

## Find the right section

| Section | What it explains | Guide |
| --- | --- | --- |
| Foundations and earlier studies | Classical structured estimation, the small learned controller, training and earlier negative results | [README](docs/sections/01-foundations/README.md) |
| Complete receiver and baselines | Pilot-matched retraining, EM continuation, high-SNR filtering, recovery, NMSE, BER and timing | [README](docs/sections/02-receiver/README.md) |
| URformer comparison | Published-setting reproduction attempt and comparison with matched training labels and pilots | [README](docs/sections/03-urformer/README.md) |
| Reproduction audits | Noise-convention and clustered-ray investigations, including negative findings | [README](docs/sections/04-audits/README.md) |
| Merged paper | Current PDF, supplement, Overleaf ZIP and older versions | [README](paper/README.md) |

The [experiment catalog](docs/EXPERIMENT_CATALOG.md) maps each result directory to
its code and saved reports. [Results](results/README.md) explains the file types.

## What the receiver does

The measurement model is `Z = abs(G @ S + B + W)`: `G` is the unknown complex
channel, `S` contains known pilots, `B` is the known coherent reference, and `W`
is complex noise. The sensor provides magnitudes, so phase has to be inferred.

1. Start with a channel estimate obtained from the measured pilots.
2. Use the exact Rician EM update to estimate the missing complex field and
   improve the channel estimate.
3. Arrange each user's estimated antenna channel in a Hankel matrix. A small
   network controls how strongly its singular values are reduced and how much
   of the filtered estimate is used.
4. The complete receiver chooses an adaptive or fixed schedule, may add EM
   refinement, and can try an algebraic recovery estimate when a residual check
   indicates a problem. The stored policy defines these choices.

The learned core has **188 parameters**. The full hybrid stores that core and a
**30-parameter fixed schedule**, for **218 neural parameters per paired fit**.
The Hankel filter acts on the estimated channel, after an EM update. It is not
a physical filter placed before the magnitude sensor.

Training of the self-supervised neural components uses withheld noisy pilot
measurements, not true-channel targets. Inference uses the measured pilots,
known reference and supplied noise variance. The separate supervised comparison
deliberately uses true-channel labels during training and is clearly identified.

## Current evidence and limits

- The complete receiver study uses 32 antennas, **3 users and 30 fitting pilots**.
  It reports NMSE, a common magnitude-domain QPSK detector and complete-call timing.
- The matched URformer study uses 32 antennas, **4 users and 20 input pilots**.
  It compares the adaptive core, not the complete high-SNR hybrid.
- These two studies have different channel-power/reference conventions. Their
  error and runtime values must not be pooled as if they were one experiment.
- The sparse/EM baseline still wins some aggregate measures, including BER and
  pooled linear NMSE in the final receiver table. No universal superiority is claimed.
- Our URformer implementation has not reproduced the published accuracy.
  The matched study supports claims about these implementations only. Their
  measured inference times are similar; parameter count is not a speed claim.
- Both later audits report negative results. The clustered audit did not pass
  its training gate, so its proposed retraining and overfit test were not run.
- The suggested uncertainty-based filter is a future idea, **not implemented**.

See the section guides for exact reports and limitations. The manuscript remains
a draft; this archive does not mean it was submitted or accepted.

## Branches

Five cumulative topic branches preserve the work in stages; later branches
include earlier dependencies. `archive/complete` contains the complete organized archive.
Older snapshots are kept as `history/*` branches; the full branch index is on [`main`](https://github.com/haatim10/Rydberg#branches).

| Branch | Added topic |
| --- | --- |
| [`research/01-foundations`](https://github.com/haatim10/Rydberg/tree/research/01-foundations) | Foundations, earlier studies and selected weights |
| [`research/02-receiver`](https://github.com/haatim10/Rydberg/tree/research/02-receiver) | Complete receiver and baseline campaigns |
| [`research/03-urformer-comparison`](https://github.com/haatim10/Rydberg/tree/research/03-urformer-comparison) | Reproduction, matched training and comparisons |
| [`research/04-urformer-audits`](https://github.com/haatim10/Rydberg/tree/research/04-urformer-audits) | SNR and clustered-channel audits |
| [`paper/tcom-merged`](https://github.com/haatim10/Rydberg/tree/paper/tcom-merged) | Manuscripts, PDFs and Overleaf packages |

## Using the archive

Read [REPRODUCING.md](docs/REPRODUCING.md) before running a study. Many historical
drivers write to fixed result paths; use a separate checkout for regeneration.
Code and published records retain their original paths and bytes.

The user chose a research archive rather than a complete machine backup. Code,
reports, figures, protocols, compact numerical evidence and selected/final weights
are included. Large simulation banks, prediction arrays, intermediate weights,
duplicate bundles and temporary files remain local. The exact file inventory,
hashes and exclusions are in [the archive guide](docs/archive_20261030/README.md).
Some original audits require omitted raw files; their saved pass records are
preserved, but a lightweight clone cannot rerun every original audit unchanged.
