"""Redraw the retained effective-rank results; no simulation or refitting."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
rows=json.loads((root/'effective_rank_plot_data.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'legend.fontsize':7,'lines.linewidth':1.1,'lines.markersize':3,'pdf.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(3.167,2.12),layout='constrained')
for n,col in [(16,'#2864B4'),(32,'#454545'),(64,'#C5443D')]:
 rr=rows[str(n)]; y=np.array([r['median_gain'] for r in rr]);ci=np.array([r['median_ci'] for r in rr]).T
 ax.errorbar([r['rho'] for r in rr],y,yerr=np.maximum(0,np.vstack([y-ci[0],ci[1]-y])),fmt='o-',color=col,capsize=2,label=f'$N={n}$')
ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.2)
ax.set(xlabel=r'Normalized effective rank $\rho$',ylabel='Paired-median gain (dB)')
ax.legend(frameon=False,loc='upper right')
fig.savefig(root/'figures/effective_rank.pdf')
