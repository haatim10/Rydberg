URformer v3: paper-setting reproduction attempt, 22 October 2026

Purpose
Implement the equations and published simulation/training settings in
https://arxiv.org/html/2509.12586v3 before calling URformer a reproduced baseline.
This experiment is separate from the earlier 3-user, 20-epoch adaptation.

What follows the paper
32 antennas, 4 users, 4x10 rays, independent uniform ray directions as printed
in Table I; the unnormalized steering vector and sqrt(M/40) scaling printed
in Eqs1-2; 10 unrolled layers, 3 encoders per layer, width64; raw kappa
FilterNet; learned gate; user tokens; linear least-squares update;
20,000 training examples, batch32, Adam .001, cosine50epochs; supervised NMSE;
random complex Gaussian initial channels. Reference ratio10dB uses TOTAL
multiuser expected signal power. GS and EM-GS each use100iterations.

What cannot be established from the paper
Pilot distribution, training SNR distribution, gain distribution, array spacing,
reference spatial phase, heads, FFN details, FilterNet hidden width, optimizer
defaults/clipping, network initialization, data-split sizes, seeds, and whether
separate models are trained for each pilot count are not fully specified.
All choices are explicitly listed in the frozen protocol.json. We found no
author-code/weight link in the arXiv source or targeted web search.

Implementation difference from earlier adaptation
Separate fresh model.py follows the printed encoder equation FFN(A)+A;
no extra feedforward-input or output LayerNorm; raw kappa rather than log1p;
random initial channel rather than spectral; neutral initial gates; standard
random output projection; no Hankel, EM continuation, filtering, recovery.
These are not changes to our proposed receiver.

Scope
Three independent full50epoch fits. Primary: final50epoch weights. Additional:
minimum-validation-NMSE checkpoint (including epoch0). Same test channels,
noise and random initial channels for every method and fit. Test2000channels
at the seven Fig3 SNR values. Fig4 is a diagnostic transfer of the same
P20-trained models to P10..35, because the authors' per-P training policy is
not stated. Fig3 P20/SNR5 and Fig4 P20/SNR5 are the same test cell.
We do not silently retrain or tune after observing the final test curves.

Reproduction standard
Training completion and passing numerical checks do not establish that the
published performance is reproduced. Compare the resulting curves with the
published curve, report any gap, and retain all missing-setting limitations.
Published values digitized from the plotted markers are approximate, not
authors' raw data. No superiority claim on our separate 3-user model follows
from this 4-user experiment. No current manuscript changes are made.

Run from repository root using the numerical Python environment:
python -m urformer_reproduction_20261022.study freeze   (once only)
python -m urformer_reproduction_20261022.study generate
python -m urformer_reproduction_20261022.study preflight
python -m urformer_reproduction_20261022.execute
python -m urformer_reproduction_20261022.checks
python -m urformer_reproduction_20261022.report

execute resumes per-epoch checkpoints if interrupted and only starts evaluation
after all three workers finish. It uses a process lock to prevent duplicates.
Scientific model/study files are frozen by SHA256 before training. Later
orchestration/check/report files do not affect training or model selection.
The final completion record hashes all code. Original receiver protocols and
previous completion records are checked before and after numerical phases.
