# Submission checks, declared before new training and test generation

Preserve the completed SAHURE pilot/expansion, historical studies and manuscript.
This is a follow-up using an already inspected research direction, not a claim
that the entire project was preregistered or that acceptance is guaranteed.

## Matched controls (nine new models)

Use the expansion's unchanged 16,384 training / 2,048 validation worlds and
seeds 9101/9102/9103. EM was selected by the previous validation rule; do not
switch to GD because its test score was better. All models have ten updates,
Adam 0.001, batch 32 and twenty epochs. No hyperparameter search.

- Supervised: the same adaptive architecture and same 21-column context input,
  masks, orders and example budget, but mean per-world channel NMSE loss.
  Channel labels are loaded only on this explicit path. Select by equally
  weighted validation-stratum channel NMSE; additionally retain and report the
  held-out-NLL-selected checkpoint to expose selection effects. This is a
  matched supervision comparison, not a fully optimized supervised upper bound.
- Final-only: independently train the same controller but use its structural
  correction only after the last of the ten data updates.
- Gate-one: independently train with the structural gate fixed to one.
  Threshold adaptation and update-step learning remain active. This tests gate
  necessity, not whether all adaptation is necessary.

The latter controls select using exactly the original equal-stratum held-out
Rician likelihood. They reuse no true channel labels. All deployment uses all
30 pilots. The original three adaptive/global/no-structure checkpoints remain
frozen comparison models. Record selected and final checkpoints for new models.

## Independent dataset/pilot replication (six new models)

Two additional independent train/validation datasets use the unchanged
generator and mixture, with new master seeds (2026101001/2026101002 and
2026102001/2026102002). Each retrains EM adaptive/global/no-structure for the
same budget. Initialization/order/mask seed is fixed to 9101 across datasets
to isolate dataset variation. Three old training seeds describe initialization
variability separately. Two fresh datasets are a limited replication, not the
previously proposed five-dataset confirmation or a precise population interval.

## Fresh paired evaluation

Freeze all weights, code and selection before generating any new test data.
Two independent test banks: seven SNRs -10,-5,0,5,10,15,20 dB at N32/K3/P30,
RSR12 and random 3–7 paths; additionally RSR0 and two-path stress cases at SNR5.
200 paired worlds per cell = 3,600 distinct test worlds. All models share each
bank's observations/noise/truth. Bank r pairs with retraining replicate r;
report original fixed models on both banks too. Classical controls: native
HS-GS with its rank-selection cost, EM10 and EM100. No new hard-rank tuning.

Primary metric: equal-weight average of per-SNR ratio-of-sums NMSE dB. Also
report every cell, NMSE>1 frequency, worst-cell deterioration, seed and dataset
variation. Do not count seed predictions as independent worlds. Paired 4,000-
resample intervals condition on fitted models and each test bank; only two
dataset replicates do not justify a reliable dataset-population CI. Secondary
stress tests remain separate. No aggregate-only success or equivalence claim.
Use Bonferroni-adjusted family-95% world intervals for the six declared primary
contrasts within each bank (quantiles 0.05/12 and 1-0.05/12). This is not a
joint interval across banks; per-cell 95% intervals remain descriptive.

Timing follows accuracy workers, one CPU thread, batch one, twenty SNR5 worlds
per bank, three repeats. Include initialization and HS-GS selection. Differences
in training objectives and implementation remain explicit.

## Safety and reproducibility

Run existing and added tests and benchmark four training batches per arm first.
At most two workers; twelve-hour wall-time cap. No GPU or paid resources.
Save optimizer and complete-epoch state, use deterministic epoch masks/orders,
process locks and immutable source/input hashes. Stop on numerical failure.
Interrupted epochs repeat from the previous complete checkpoint; inference
results resume per model/cell and classical per world. Do not silently repair
an SVD failure or alter a frozen algorithm. The cancelled calibration project
remains cancelled. No manuscript or dashboard changes in this campaign.

## Literature check

Maintain a separate primary-source overlap note. Neither self-supervision,
Rydberg sensing, unrolling nor low-rank learning alone is claimed as new.
Close comparisons and the additional full-text review remain necessary even
if these numerical checks pass.
