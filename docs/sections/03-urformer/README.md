# URformer reproduction and matched comparison

These studies answer different questions. A completed training run is not proof
that a published result has been reproduced.

## 1. Earlier adaptation and review checks

`trackD_urformer/` contains the older adaptation. `tcom_review_checks_20261022/`
contains later receiver-review checks and an adapted URformer comparison. Their
settings differ from the separate published-setting reproduction below.

## 2. Published-setting v3 reproduction attempt

- [Implementation and original README](../../../urformer_reproduction_20261022/README.txt)
- [Result report](../../../results/urformer_reproduction_20261022/RESULTS.txt)
- [Frozen protocol](../../../results/urformer_reproduction_20261022/protocol.json)
- [Summary](../../../results/urformer_reproduction_20261022/summary.json)

This uses 32 antennas, 4 users, 20 pilots, 40 rays, three training seeds, 20,000
training examples and 50 epochs. Its v3 implementation follows the printed
encoder without the extra FFN input normalization used in the older version.
Unspecified choices are recorded in the protocol.

The published accuracy remains unreproduced. At 5 dB, the approximate published
target is -21.6 dB NMSE, while the validation-selected reproduction is about
-7.66 dB. At 20 dB, the corresponding numbers are -31.9 and about -16.40 dB.
Published curve readings are approximate. This gap is retained explicitly.

## 3. Matched architecture and label comparison

- [Study README](../../../matched_supervision_20261023/README.txt)
- [Primary final-epoch results](../../../results/matched_supervision_20261023/RESULTS.txt)
- [Full summary, including validation-selected results](../../../results/matched_supervision_20261023/summary.json)
- [Protocol](../../../results/matched_supervision_20261023/protocol.json)

Four cases are trained from scratch: our adaptive core and our URformer
implementation, each with and without true-channel labels. They share 20,000
training examples, 20 input pilots, 9 withheld targets, 50 epochs, three seeds,
the same spectral start and common evaluation data. Test inference never sees
the true channel. The primary rule uses epoch 50; the separately reported
validation-selected rule uses the same held-out Rician likelihood for all arms.

At 20 input pilots the reported p90 times are approximately 4.5-4.6 ms for both
architectures. The 188-parameter controller uses much less weight storage, but
this study does not show a meaningful runtime advantage over URformer.

The proposed core performs better than this documented URformer implementation
in the reported matched comparison. This does **not** establish superiority to
the unreproduced published URformer accuracy or to every possible tuning of it.
Self-supervised training is evaluated for both architectures, not only ours.

This comparison excludes our fixed high-SNR schedule, extra EM, final filter and
recovery branch. Do not compare its 4-user/20-pilot timings directly with the
complete receiver's 3-user/30-pilot results.

## Weights and reproduction

Both `best.pt` and `final.pt` are archived for the three-seed reproduction and
all four arms of the matched study. Resume snapshots and large generated banks
remain local. Read [REPRODUCING.md](../../REPRODUCING.md) before using the original
drivers; they write to their recorded output paths.

The [next section](../04-audits/README.md) explains the two completed attempts to
diagnose the reproduction gap. Neither established the cause of that gap.
