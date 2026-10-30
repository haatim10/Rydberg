"""Plot only the completed, frozen comparison; no fitting or selection."""
import os
from pathlib import Path
ROOT=Path('results/paper2_optshrink_loop')
os.environ['MPLCONFIGDIR']=str(ROOT/'matplotlib_config')
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    status=json.loads((ROOT/'status.json').read_text())
    if status['state']!='complete': raise RuntimeError('comparison is not complete')
    data=json.loads((ROOT/'summary.json').read_text())
    snrs=np.arange(-10,21,5);cells=[data['cells'][f'snr_{s:+d}'] for s in snrs]
    colors=dict(adaptive10='#155e75',loop_selected='#c2410c',hs_rician='#64748b',final_selected='#7c3aed')
    labels=dict(adaptive10='Learned, 10 stages',loop_selected='In-loop OptShrink, tuned',
                hs_rician='Rician-selected HS-GS',final_selected='Final-only OptShrink, tuned')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.3),layout='constrained')
    ax=axes[0]
    for name in labels:
        values=[c['methods'][name]['nmse_db'] for c in cells]
        ax.plot(snrs,values,'o-',lw=2,ms=4,color=colors[name],label=labels[name])
    ax.set(xlabel='SNR (dB)',ylabel='Channel NMSE (dB; lower is better)',
           title='128 fresh paired cases at each SNR',xticks=snrs)
    ax.grid(alpha=.18);ax.legend(fontsize=8,frameon=False)
    ax=axes[1]
    means=np.array([c['contrasts']['loop_selected']['learned_gain_db'] for c in cells])
    intervals=np.array([c['contrasts']['loop_selected']['interval'] for c in cells])
    ax.errorbar(snrs,means,yerr=np.stack([means-intervals[:,0],intervals[:,1]-means]),
                fmt='o',capsize=4,color=colors['adaptive10'])
    ax.axhline(0,color='#475569',lw=1)
    ax.set(xlabel='SNR (dB)',ylabel='Learned gain over in-loop OptShrink (dB)',
           title='Above zero favors the learned model',xticks=snrs)
    ax.grid(alpha=.18)
    ax.text(.03,.03,'95% paired-case intervals; conditional on frozen models.\nPer-SNR intervals are descriptive, not multiplicity adjusted.',
            transform=ax.transAxes,fontsize=7.5,va='bottom')
    path=ROOT/'comparison.png';fig.savefig(path,dpi=180);plt.close(fig)
    sidecar=dict(source_sha256=hashlib.sha256((ROOT/'summary.json').read_bytes()).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (ROOT/'comparison_provenance.json').write_text(json.dumps(sidecar,indent=2)+'\n')
    print(path)

if __name__=='__main__': main()
