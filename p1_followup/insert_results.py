from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/p1_followup';paper=ROOT/'paper/paper1/full_revision';p=paper/'haatim_hsgs_revised.tex'
s=p.read_text();data=json.loads((out/'analysis/summary.json').read_text())
assert 'app:followup' not in s
appendix=r'''
\section{Sweep Count and Rank-Selection Stability}
\label{app:followup}

\subsection{Number of structural sweeps}

We use 100 fresh paired trials at each combination of $L\in\{3,7,14\}$
and SNR $\in\{0,10\}$~dB, giving 600 independent channels. We keep
$N=32$, $K=3$, $P=30$, RSR $=12$~dB and the 100/25 update budgets.
Both hard truncation and smooth filtering are tested with one, two,
four and eight sweeps per application. Each setting selects its own rank
or threshold using the same trailing 30\% validation set, then refits
all pilots. Thus, we test the whole selection-and-estimation procedure;
we do not hold the selected rank fixed across sweep counts.

Fig.~\ref{fig:sweep-followup} shows that more sweeps need not improve
channel estimation. One hard-rank sweep improves on four by
$0.275$--$0.652$~dB across the six settings, with all six pointwise 95\%
intervals above zero. For example, the gain at $L=7$ and SNR $=10$~dB
is $0.652$~dB $[0.488,0.824]$. The original four-sweep choice is
therefore not optimal in these tests. Fitting the structural constraint
more closely does not necessarily estimate the noisy channel better.
Because selection is repeated, this comparison includes changes in both
the selected rank and the structural processing.

Smooth filtering also depends on the sweep count, but no tested count
has the highest point-estimate gain in every setting. Its threshold is
rescaled on each sweep, so increasing the count changes the accumulated
suppression. The original comparison between hard and smooth processing
must therefore be read with its stated four-sweep setting. These checks
do not establish a universal best count or a stopping rule.

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{fig/sweep_sensitivity.pdf}
\caption{Sweep sensitivity on 100 fresh paired trials per panel. Each
method and sweep count selects its own rank or threshold. Gain is the
difference between ratio-of-sums NMSE values; positive values favor the
structured method over EM-GS. Bars are pointwise 95\% intervals from
4,000 paired world resamples.}
\label{fig:sweep-followup}
\end{figure*}

TIMING_PARAGRAPH

\subsection{Changing the held-out pilots}

On the same 600 worlds, we repeat four-sweep HS-GS using three random
70/30 pilot partitions, in addition to the original trailing split.
Each partition is a separate single-split run. We average the three
random-partition squared errors within each world before forming the
ratio-of-sums comparison and its 4,000-resample interval. The independent
sample size remains 100 per setting. This is not a multi-fold estimator
or an average of the resulting channel estimates.

Table~\ref{tab:split-stability} shows no consistent performance benefit
from random splitting. One pointwise interval is positive, at $L=7$
and SNR $=10$~dB; the other five include zero. This does not establish
equivalence. Random versus fixed partitions choose different ranks in
$15.7$--$86.7$\% of the paired comparisons. The small average performance
differences can therefore hide substantial variation in the chosen rank.
With exchangeable pilot columns, there is no preferred column position
in the population model; this test mainly examines sensitivity within
a particular observed world. Repeated selection adds computation and
has not been shown here to reduce deployment cost.

\begin{table}[!t]
\caption{Random versus trailing pilot split. Each row uses 100 worlds;
three random partitions are averaged within a world for the gain.}
\label{tab:split-stability}
\centering\footnotesize
\begin{tabular}{rrrc}
\toprule
$L$ & SNR & Rank differs (\%) & Gain [95\% interval] (dB)\\
\midrule
SPLIT_ROWS
\bottomrule
\end{tabular}
\end{table}

\subsection{Selected rank and estimation harm}

We also reanalyse the 2,400 existing trials used for the six-method
comparison. They are not additional independent trials. These cells have
the same path count for every user; under the stated distinct-path model,
the noiseless Hankel rank is $L$. Table~\ref{tab:rank-harm} reports how
often the shared selected rank is below $L$ and how often HS-GS has
larger squared channel error than its paired EM-GS estimate.

Selecting below the true rank is not automatically a failure. At $L=7$
and SNR $=0$~dB, 315 of 400 trials select below seven, but only 31 of
those 315 trials are worse than EM-GS. Suppressing weak components can
help in noise. Conversely, at $L=14$ and SNR $=0$~dB, HS-GS is worse
in 191 of 400 trials. The full-rank option does not prevent the selector
from choosing a harmful smaller rank. These are descriptive associations,
not proof that matching the true rank minimizes channel error.

\begin{table}[!t]
\caption{Rank and harm diagnostics from the existing 400 trials per row.
Harm means strictly larger squared error than EM-GS.}
\label{tab:rank-harm}
\centering\footnotesize
\begin{tabular}{rrrrr}
\toprule
$L$ & SNR & Median $\widehat r$ & $\widehat r<L$ (\%) & Harm (\%)\\
\midrule
RANK_ROWS
\bottomrule
\end{tabular}
\end{table}

'''
rows=[]
for r in data['splits']:
 a=r['random_mean_vs_fixed'];rows.append(f"{r['L']} & {r['snr']} & {100*r['random_vs_fixed_rank_disagreement']:.1f} & ${a['gain']:+.3f}$ $[{a['lo']:.3f},{a['hi']:.3f}]$\\\\")
appendix=appendix.replace('SPLIT_ROWS','\n'.join(rows))
rows=[]
for r in data['historical']:
 rows.append(f"{r['L']} & {r['snr']:g} & {r['median_selected']:g} & {100*r['groups']['below']['n']/r['n']:.2f} & {100*r['harm_fraction']:.2f}\\\\")
appendix=appendix.replace('RANK_ROWS','\n'.join(rows))
times=json.loads((out/'timing/summary.json').read_text())
def t(kind,q):return next(r['median_seconds'] for r in times if r['kind']==kind and r['sweeps']==q)
timing=(r'''A separate serial check uses 12 fresh worlds at $L=7$ and SNR $=5$~dB,
with the same array, pilots, reference and update budgets. On the same
CPU in double precision with one numerical-library thread, the median
hard-rank times for one, two, four and eight sweeps are
'''+', '.join(f'${t("hard",q):.3f}$' for q in (1,2,4,8))+r'''~s, including initialization, selection and final fitting.
The corresponding smooth-filter times are
'''+', '.join(f'${t("smooth",q):.3f}$' for q in (1,2,4,8))+r'''~s. Method order is randomly permuted within each timing world.
These small timing samples describe one implementation and machine;
ordinary host activity is not controlled. Parallel campaign times are
not used as inference benchmarks.''')
appendix=appendix.replace('TIMING_PARAGRAPH',timing)
s=s.replace('\\clearpage\n\\IEEEtriggeratref',appendix+'\\clearpage\n\\IEEEtriggeratref',1)
s=s.replace('Reference strength & 32',r'Sweep/split stability & 32 & 30 & 12 & $\{0,10\}$ & 100/25 & 100\\'+'\nReference strength & 32',1)
needle='and candidate grids are fixed before inspecting the corresponding results.'
s=s.replace(needle,needle+''' The sweep-count and random-split checks in
Appendix~\\ref{app:followup} add 600 independent worlds, reused across
all their variants, and a separate 12-world serial timing check.''',1)
needle='and its parameter selection together for the intended channel and pilot\nconditions, including cases where the rule can cause harm.'
s=s.replace(needle,needle+r'''
The focused sweep check reinforces this point: one hard-rank sweep
outperforms the original four-sweep setting in all six tested conditions.
Changing the held-out pilots often changes the selected rank, without a
consistent average gain from random splitting. Appendix~\ref{app:followup}
reports these checks; the original curves retain their declared settings.
''',1)
# Make setting-specific claim explicit where smooth comparison is summarized.
s=s.replace('Smooth singular-value filtering improves on hard-rank HS-GS in five of\nsix new comparison settings','With four sweeps per application, smooth singular-value filtering improves\non hard-rank HS-GS in five of six comparison settings',1)
p.write_text(s)
shutil.copy2(out/'analysis/sweep_sensitivity.pdf',paper/'fig/sweep_sensitivity.pdf')
