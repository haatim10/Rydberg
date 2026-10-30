"""Generate tables and a scientific figure directly from the saved audit."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/merged_review_checks')
PAPER=Path('paper/merged_tcom')
d=json.loads((ROOT/'summary.json').read_text())
t=json.loads((ROOT/'timing.json').read_text())['summary']
m=d['macro'];cells=d['cells']
ci=lambda x: f'[{x[0]:.3f},{x[1]:.3f}]'
gain=m['one_sweep']['nmse_db']-m['learned']['nmse_db']
hardgain=m['hs_rician']['nmse_db']-m['one_sweep']['nmse_db']

main=rf"""The added one-sweep HS-Rician control reaches ${m['one_sweep']['nmse_db']:.3f}$~dB,
improving on four sweeps by {hardgain:.3f}~dB. Learning retains a
{gain:.3f}~dB average gain, with paired interval
${ci(m['one_sweep']['learned_gain_db_ci95'])}$~dB. This closes the
one-versus-four-sweep gap on the same bank, not the comparison to every
possible hard-rank method. In the separate weak-reference and two-path
cells, one sweep instead beats learning by
{abs(cells['weak_reference']['one_sweep']['learned_gain_db']['value']):.3f} and
{abs(cells['two_paths']['one_sweep']['learned_gain_db']['value']):.3f}~dB.
"""
(PAPER/'followup_main.tex').write_text(main)

tail=rf"""At the original threshold $e_i/d_i>1$, the seven-SNR reduction in
failure rate versus in-loop OptShrink is
{m['loop_selected']['tail']['1.0']['percent']-m['learned']['tail']['1.0']['percent']:.2f}
percentage points, with paired interval
${ci(m['loop_selected']['tail']['1.0']['minus_learned_pp_ci95'])}$.
At $-10$~dB alone the interval is
${ci(cells['snr_-10']['loop_selected']['tail']['1.0']['minus_learned_pp_ci95'])}$
percentage points. Learned seed-specific rates there are
5.47\%, 5.47\% and 7.81\%, versus 35.16\% for OptShrink.

\begin{{table}}[!t]
\caption{{Post-hoc error-threshold sensitivity on the stronger bank.
Rates average seven SNR cells; positive differences favor learning.}}
\label{{tab:tails}}\centering\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}rrrr@{{}}}}\toprule
NMSE & Learned & In-loop & OptShrink minus learned\\
threshold & (\%) & OptShrink (\%) & 95\% interval (points)\\\midrule
"""
for k,label in [('0.5','0.5'),('1.0','1'),('2.0','2')]:
    learned=m['learned']['tail'][k]['percent'];opt=m['loop_selected']['tail'][k]
    interval='---' if k=='2.0' else '$'+ci(opt['minus_learned_pp_ci95'])+'$'
    tail+=f"${label}$ & {learned:.2f} & {opt['percent']:.2f} & {interval}\\\\\n"
tail+=r"""\bottomrule\end{tabular}
\end{table}
The conclusion depends on the error threshold (Table~\ref{tab:tails}).
At threshold 0.5, OptShrink has fewer exceedances; at threshold 2,
neither method has any observed exceedance. Zero observed events do not
establish zero population risk. These exploratory, pointwise intervals
retain paired worlds and fixed models and are not adjusted for multiple
thresholds. Learning therefore does not dominate the whole error
distribution. The original severe-error definition and primary NMSE
metric are unchanged.
"""
(PAPER/'followup_tails.tex').write_text(tail)

timing=rf"""The post-review timing run measures all five methods together on the
same independent 24-world SNR 5~dB timing bank, with three repetitions.
Mean times are {t['learned']['mean_ms']:.2f}~ms for one adaptive model,
averaged across three trained models, {t['loop_selected']['mean_ms']:.2f}~ms
for in-loop OptShrink, {t['final_selected']['mean_ms']:.2f}~ms for final-only
OptShrink, and {t['one_sweep']['mean_ms']:.2f}/{t['hs_rician']['mean_ms']:.2f}~ms
for one-/four-sweep HS-Rician. Initialization and setting selection or
lookup are included; model loading and file I/O are excluded. These are
new paired timing measurements, not averages with the earlier timing run.
Backend differences remain. Learning avoids the hard method's per-case
search, but has no large speed advantage over offline-tuned OptShrink.
"""
(PAPER/'followup_timing.tex').write_text(timing)

supp=r"""\section{Matched One-Sweep Baseline and Tail-Error Audit}
This post-review follow-up uses the already inspected 1,152-world
OptShrink bank: 128 worlds in each of seven primary SNR cells and two
stress cells. It generates no new channels and trains no new model.
The protocol is frozen before running the added comparator, not before
the historical bank was inspected. It must not be described as a new
blinded confirmation. The seven primary cells contain 896 distinct worlds.

The one-sweep HS-Rician estimator fits every rank 1--16 on the first
21 pilots for 25 EM updates, scores Rician likelihood on the other nine,
then refits the selected rank with 100 updates and all 30 pilots.
Each candidate and refit uses one Cadzow sweep. The sweep count is the
only changed setting relative to the saved HS-Rician baseline; the new
one-sweep candidates select their own ranks. Rank 16 bypasses filtering.
No true channel is an estimator input. Learned checkpoints, OptShrink
settings, initialization and observation budgets remain unchanged.

\begin{table*}[!t]
\caption{Matched saved-bank results, lower dB NMSE is better. Stress cells
are excluded from the seven-SNR macro-average. Learned values average
three fixed-model dB scores, not ensemble estimates.}
\centering\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrrr@{}}\toprule
Cell & HS, one sweep & HS, four sweeps & In-loop OptShrink & Final OptShrink & Learned\\\midrule
"""
for name,cell in cells.items():
    label=name.replace('snr_','SNR ').replace('weak_reference','RSR 0 dB').replace('two_paths','Two paths')
    supp+=label+' & '+' & '.join(f"{cell[k]['nmse_db']:.3f}" for k in ['one_sweep','hs_rician','loop_selected','final_selected','learned'])+r'\\'+'\n'
supp+=r'\midrule Seven-SNR mean & '+' & '.join(f"{m[k]['nmse_db']:.3f}" for k in ['one_sweep','hs_rician','loop_selected','final_selected','learned'])+r'\\\bottomrule\end{tabular}\end{table*}'+'\n'
supp+=main+'\n'+tail.replace('tab:tails','tab:supp_tails')+'\n'+timing
supp+=r"""
\subsection{What the Uncertainty Includes}
The new audit uses 4,000 bootstrap draws, resampling 128 whole worlds
within each cell. A draw keeps all paired methods and learned seed
outputs together. NMSE first sums errors and channel energies within
each cell, converts their ratio to dB, averages fixed-model scores,
and then equally averages the seven cells. Failure rates instead count
per-world error/energy ratios above the stated threshold and average
fixed-model rates. Training seeds are not independent test observations.
The intervals are pointwise, conditional on the saved models and
exploratory. Thresholds 0.5 and 2 were added for sensitivity analysis;
the original threshold 1 was not selected to maximize a new contrast.
No risk dominance or safety-critical guarantee is inferred. A degenerate
empirical bootstrap when there are no events is not evidence of known
zero event probability; such intervals are omitted from the table.

\subsection{Verification and Numerical Scope}
The run checks original source and checkpoint hashes, saved observations
and truth hashes, and all 1,152 trial identities. It recomputes the old
prediction errors from channel arrays, rather than copying headline
statistics. Four-sweep replay agrees exactly with the existing solver;
full-rank bypass is exactly equal for one and four sweeps. Each added
prediction retains its channel, selected rank, all candidate scores,
trial ID and processing time. A run lock prevents duplicate writers,
and completed cases are reused on resume. Earlier result files are not
overwritten. The accompanying evidence archive contains this follow-up's
protocol, code, source hashes, predictions and aggregate outputs. The
large original observation/checkpoint bank is retained in the separate
Paper 2 reproduction archive. These checks validate the recorded
computation, not external replication or hardware performance.

The rest of this supplement retains the earlier learned-study evidence.
Statements about its original timing samples, baselines and data counts
refer to those separately labelled protocols, not the new follow-up.
"""
(PAPER/'support/merged_followup.tex').write_text(supp)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'legend.fontsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(6.5,2.7),layout='constrained')
snrs=list(range(-10,21,5));keys=[f'snr_{s:+d}' for s in snrs]
styles=[('hs_rician','HS-Rician, four sweeps','#999999','--'),
        ('one_sweep','HS-Rician, one sweep','#be7430','-.'),
        ('final_selected','Final-only OptShrink','#297060',':'),
        ('loop_selected','In-loop OptShrink','#3e67aa','-'),
        ('learned','Learned EM','#bd3542','-')]
for k,label,color,ls in styles:
    axs[0].plot(snrs,[cells[c][k]['nmse_db'] for c in keys],label=label,color=color,ls=ls,lw=1.25,marker='o',ms=2.5)
for k,label,color in [('one_sweep','vs. one-sweep HS','#be7430'),('loop_selected','vs. in-loop OptShrink','#3e67aa')]:
    y=np.array([cells[c][k]['learned_gain_db']['value'] for c in keys])
    bounds=np.array([cells[c][k]['learned_gain_db']['ci95'] for c in keys])
    axs[1].errorbar(snrs,y,yerr=np.maximum(0,np.stack([y-bounds[:,0],bounds[:,1]-y])),label=label,color=color,marker='o',ms=3,lw=1,capsize=2)
axs[0].set(ylabel='NMSE (dB)',xlabel='SNR (dB)');axs[0].legend(frameon=False)
axs[1].set(ylabel='Learned gain (dB)',xlabel='SNR (dB)');axs[1].axhline(0,color='gray',lw=.7);axs[1].legend(frameon=False)
for ax in axs: ax.grid(alpha=.2);ax.set_xticks(snrs)
fig.savefig(PAPER/'figures/optshrink.pdf')
fig.savefig(ROOT/'comparison.png',dpi=160)
print('Generated merged comparison tables, supplement section and figure.')
