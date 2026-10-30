# Large Hankel confirmation study

User authorized a large study and explicitly chose “Does Hankel add value over
strong baselines?” Written before new training/test results. This is a locally
frozen prospective protocol, not external registration or guaranteed publication.

## Question and scope

Test whether rank7 Hankel inside an unrolled estimator adds useful accuracy over
matched strong learned and classical controls, and whether the answer depends on
spatial structure. This does not reproduce the authors' URformer curve. All
learned arms use our explicitly specified classical-start model. The ordinary
reconstruction was already investigated; we do not label this replacement as
published URformer. No new calibration project or manuscript edits.

Two channel populations: literal_independent and clustered_5deg, exactly the
prior documented generators (N32,K4,P20,40rays,expected per-element power32,
fixed Gaussian pilots seed777000001,ensemble multiuser RSR10dB,known reference
and noise variance). Clustered offsets +/-5degrees and boundary clipping are
our simulation assumptions, not established hardware statistics. This is not
the original K3 sparse manuscript channel; existing results for that channel
remain separate. Conclusions apply to these two declared populations only.

## Frozen training matrix:40 runs

For each population, five paired initialization/order seeds2026092601..605.
User subsequently limited runtime to around15hours, BEFORE launch or study
results. A full preflight estimated13.3–15.7hours for20epochs, so the final
pre-launch budget is18epochs (same seeds/arms/data) to leave runtime headroom.
Revised fixed budget:16,000 train,1,200 validation cases;18epochs,
batch32,Adam.001,cosine decay tozero,
no weight decay or gradient clipping. One CPU thread,float32. Four arms:

- standard_U: ten exact EM-GS layers with zero-output-initialized PaperFormer
  corrections, ordinary mean per-example NMSE loss.
- standard_H: same, rank7 one-sweep Hankel step between each EM-GS and correction.
- weighted_U / weighted_H: same pair, loss weights10 at SNR>=15dB,1otherwise,
  normalized by mean weight across the training set.

Hankel uses the existing16x17 lift and identity-backward approximation; no SVD
gradient. All four arms have identical learned initial parameters and sample
order within each seed/population. H changes initial outputs, not parameters;
only U initially equals ordinary EM-GS10. Fixed rank7 is NOT a claim that the
40-ray channels truly have rank7. Quantify noiseless projection distortion.
Both losses are included because the small weighting diagnostic failed its
screen, not because it established a superior baseline. No new weight/rank sweep
for the neural arms.18epochs does not establish convergence. Report curves and
late trends. Larger duration changes schedule as well as training amount.

Total720epochs,11.52million training-example presentations,360,000updates.
Save every epoch model,per-example validation errors,optimizer/scheduler/RNG and
batch-order hashes. Resume last complete epoch only. Source/data hash checks,
exclusive run lock; stop non-finite loss/gradients, no silent fixes.

## Fresh splits and test protection

Independent-mode trial starts: train30m,val31m,test32m,tune33m.
Clustered-mode: train40m,val41m,test42m,tune43m. Counts train16,000,val1,200,
test1,800,tune600 per mode. No overlap with earlier studies. Training uniform
[-10,20]; val/test/tune balanced over six5dB bins, uniform inside each bin.
These are distinct channel/noise realizations, not the same channel replicated
at several SNRs. Seeds vary weights/order, not the fixed training dataset.

Only train/val/tune caches are generated before training. Do not generate or
evaluate test cases until ALL40training runs finish and selection.json is
frozen. During the queue report progress/validation only; do not alter protocol.
For each run select earliest epoch within1SE of best full-range unweighted
validation NMSE, as before. Retain final18epoch scores separately.

For each population, choose the primary loss (standard or weighted) using pooled
mean-linear validation error of U's selected checkpoints across5seeds; choose
standard on a tie. Then compare H to U with THAT SAME loss. Do not choose the
loss by H's test advantage. Report both loss contrasts as secondary results.
For comparison with external controls also select best U and best H loss
independently on pooled validation error, before test access; label this
validation-selected pipeline comparison, not a one-factor ablation.

## Strong classical references

1. EM-GS10 and100, same random G0 as neural arms.
2. One global EM-GS stopping count selected from{1,3,5,10,20,50,100} on tune set.
3. Fixed-rank7 structured EM-GS10 and50,one Cadzow sweep, same G0.
4. Paper1 hs_gs_auto: ALL ranks1..16,final30%pilot-column holdout,25selection
   iterations,50final iterations,4Cadzow sweeps,native initialization and
   float64. This is the existing algorithm on the new channel populations,
   NOT a reproduction of Paper1's original numerical setup.
5. Paper1-paired unstructured EM-GS50 with its same native initialization and
   float64; rank16 must bypass structure and match this reference.

Classical global stopping uses tune truth as a development-only choice; Paper1
per-case rank selection uses observed held-out pilots, never true channels.
All reported methods receive the same total20observed pilots; Paper1 refits all
20after selection. Report actual inference times including selection overhead,
no parameter-count-only efficiency claim. Select strongest classical pipeline
on full-range validation error before test. Report all controls, including
per-SNR disadvantages that a full-range average can hide.

## Outcomes and interpretation fixed before results

Primary: selected-checkpoint mean-linear NMSE gain10log10(mean(err_U)/mean(err_H))
in SNR>=5dB, calculated per paired seed then averaged over5seeds, separately
for the two populations. Report seedSD and paired t intervals. For two primary
population claims use97.5%two-sided intervals (Bonferroni familywise95%). Five
seeds still give uncertain inference conditional on one dataset/pilot matrix.
Secondary: final checkpoint, full range, all6bins, paired median gain, severe
errors(NMSE>1), and absolute error. Report case-resampling uncertainty separately
from seed variation; do not treat5x1800predictions as independent experiments.

Publication-candidate screen (not acceptance/significance guarantee): primary
gain>=0.5dB with adjusted lower interval>0; no >0.2dB full-range degradation;
validation-selected H must also improve high-SNR mean NMSE by>=0.5dB over the
strongest validation-selected classical control. If only structured mode passes,
claim only structured-mode evidence. Final/selected disagreement or persistent
learning trends qualify conclusions. Negative results are retained. No test
driven extra seeds,rank/loss changes or automatic follow-up study.

Interpret these as stronger evidence about this specified model, not definitive
proof of novelty, convergence, hardware utility or general superiority. A
literature/novelty assessment and manuscript claim audit remain necessary.

## Execution

Driver scripts/paper2_hankel_confirmation.py; separate results directory.
Preflight: degenerations, matching, source preservation, numerical gradients,
new data IDs, tiny end-to-end pipeline, checkpoint resumption and timing.
Revised local target about12–15hours, roughly7–10GB new checkpoints/data. A15hour
worker-time guard stops at a safe check if the protocol cannot finish; incomplete
results are not confirmation. Saved worker time counts on resumption; no
automatic extension is authorized. Mac must remain awake/powered. A background process or
tmux does not execute while the computer is off. No external compute purchase.
