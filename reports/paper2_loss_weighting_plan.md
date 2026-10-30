# Fixed high-SNR loss weighting diagnostic

User authorized “do it”. Fixed before observing new training results.

Keep the classical-start architecture and physical units unchanged. Two arms:
standard NMSE and high-SNR-weighted NMSE. For each training example use weight
10 if SNR >=15dB, otherwise1, divided by the mean raw weight on the entire
training set. Optimize mean(weight * per-example NMSE). Do not normalize by
each batch's weight sum. Factor10 is a prechosen diagnostic setting, not tuned
or optimal. Training SNR is simulator metadata; inference receives no new input.
No model, data-distribution, optimizer, clipping, or schedule changes.

Reuse the classical-start study's 4096 train IDs24m+,512 validation IDs25m+,
seed2026092501,batch32,12epochs,Adam.001,cosine decay. Thus the standard control
must replay the prior classical-start result. Same initial weights and batch
orders across arms. Validation is reused development data, NOT an independent
confirmation or final test. No Hankel, manuscript or dashboard changes.

Classical references remain EM-GS10,100 and stopping chosen on separate256
development cases IDs26m+ from{1,3,5,10,20,50,100}. Compare also to the previous
ordinary reconstruction on the same512 validation cases, verifying provenance
and sample IDs. No new ordinary training is necessary.

Primary reporting: final-epoch UNWEIGHTED mean-linear NMSE overall and six SNR
bins. Secondary: same full-range unweighted one-standard-error checkpoint rule.
Report the weighted arm vs standard classical-start, ordinary reconstruction,
and initial EM-GS10. Success screen: >=0.5dB overall improvement vs ordinary AND
<=0.5dB deterioration vs EM-GS10 in[15,20). Report each condition separately.
One seed; no significance/novelty/publication claim. Do not search more weights
or launch multi-seed runs automatically. Negative results remain results.

Isolated output results/paper2_loss_weighting; oneCPUthread; run lock, source/data
hashes, epoch checkpoints/resume and finite-gradient checks. Tests must verify
the weighted objective and gradients, factor-one equivalence, no architecture
change, and exact weighted-run resume. Recorded train_online_nmse remains
unweighted for comparability; gradients alone use the weighted objective.
On completion replay selected/final checkpoints, verify control replay and
matched orders, recompute gains, save a plain-language results report.
