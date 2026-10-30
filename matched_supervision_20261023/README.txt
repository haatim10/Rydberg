Matched architecture and supervision study, 23 October 2026

Four cases: proposed estimator and URformer implementation, each with and
without true-channel training labels. All test inference hides the channel.

Each model starts from scratch, using the same 20,000 training channels,
2,000 validation channels, 20 context pilots, nine separate noisy pilot
targets, 50 epochs, batch32, Adam .001 with cosine scheduling, norm clipping1,
and three seeds. Supervised/self-supervised initial network weights are
identical within each architecture. Both architectures receive the same
spectral channel initialization and exact unregularized pilot LS map.

Data availability is matched: all four cases have the same29raw training
pilot measurements. Only20enter the model. Supervised training minimizes
true-channel NMSE. Without labels, training minimizes exact Rician loss on
the other9measurements. The self-supervised worker cannot load label files.
Primary checkpoints use exactly50epochs. Additional best checkpoints use
the same heldout validation likelihood for every case, never test error.

Physical setup: four users,32antennas,40rays,reference ratio10dB using total
multiuser signal power. All training SNRs span nominal-10..20dB. Details and
unreported-paper assumptions are frozen in protocol.json. This is a fair
comparison of our implementations; published URformer accuracy remains
unreproduced and is not a measured competitor in the main result plot.

Ordinary-SNR adaptive model only: no extra EM, final filter, fixed high-SNR
schedule or recovery branches. Our common-front-end implementation is
checked against the unchanged original adaptive estimator before fitting.
No old experiment or manuscript files are edited.

Fresh primary test:2,000independent channels on the seven SNR values with
20inputpilots. The same models are additionally tested at10..35pilots at
SNR5dB; that panel is a transfer diagnostic for all four arms equally.
BER: same first512fresh channels and32common data vectors, full256hypothesis
Rician detector, same known reference/noise. Timing: two fixed isolated
windows after training and all accuracy jobs; one thread, PyTorch float64,
complete calls including shared initialization. No timing retries to pass.

Matched examples and updates do not imply equal training FLOPs or establish
the best achievable tuned performance. Learning curves and actual time are
retained. Three fits share one training population; channel-bootstrap
intervals are descriptive conditional on those fits, not universal guarantees.

Run from the repository root with the scientific Python environment:
python -c 'from matched_supervision_20261023.common import freeze; freeze()'
python -m matched_supervision_20261023.data
python -m matched_supervision_20261023.checks preflight
python -m matched_supervision_20261023.execute

The runner trains six processes, each one CPU thread. A worker handles one
label mode and one seed, fitting both architectures with the same batch
order. Checkpoints resume at complete epochs. After all12fits finish, it
audits label separation/checkpoints, evaluates NMSE and BER, measures two
timing windows, and creates plots/reports. Model initialization, data,
training, selection and evaluation sources are frozen before training.
