"""Figures and numeric tables from saved results only; never tune or fit."""
import os
import json
import hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/pdfs/mpl-round2'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    paths=[ROOT/'results/paper2_optshrink_loop/summary.json',
           ROOT/'results/paper2_review_round2/calibration/summary.json',
           ROOT/'results/paper2_review_round2/geometry/summary.json',
           ROOT/'results/paper2_review_round2/FINAL_NUMERICAL_AUDIT.json']
    opt,cal,geo,audit=[json.loads(p.read_text()) for p in paths]
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    snrs=np.arange(-10,21,5);cells=[opt['cells'][f'snr_{s:+d}'] for s in snrs]
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.6),layout='constrained')
    for name,label,style in [('adaptive10','Learned, ten stages','o-'),('loop_selected','In-loop OptShrink','s--'),('hs_rician','Rician-ranked HS-GS','^:')]:
        axes[0].plot(snrs,[c['methods'][name]['nmse_db'] for c in cells],style,label=label,ms=3,lw=1.2)
    axes[0].set(xlabel='SNR (dB)',ylabel='NMSE (dB)',title='(a) Fresh paired comparison',xticks=snrs)
    axes[0].legend(frameon=False,fontsize=7)
    values=np.array([c['contrasts']['loop_selected']['learned_gain_db'] for c in cells])
    ci=np.array([c['contrasts']['loop_selected']['interval'] for c in cells])
    axes[1].errorbar(snrs,values,yerr=np.stack((values-ci[:,0],ci[:,1]-values)),fmt='o',capsize=3,ms=3)
    axes[1].axhline(0,color='.5',lw=.8);axes[1].set(xlabel='SNR (dB)',ylabel='Learned gain (dB)',title='(b) Positive favors learned',xticks=snrs)
    for ax in axes:ax.grid(alpha=.2);ax.tick_params(labelsize=7)
    for ext in ('pdf','png'):fig.savefig(HERE/f'figures/round2_optshrink.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    text=[r'''\begin{table}[!t]
\caption{Static calibration: gain of corrected over supplied inputs}
\centering\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrr@{}}\toprule
Input condition & Mean gain & Worst cell\\
 & (dB) & gain (dB)\\\midrule
''']
    labels=[('matched','Correct inputs'),('noise_minus6',r'Variance $-6$ dB'),('noise_plus6',r'Variance $+6$ dB'),
            ('amplitude_minus10pct',r'Amplitude $-10\%$'),('amplitude_plus10pct',r'Amplitude $+10\%$'),
            ('joint_plus',r'Joint $+10\%/+6$ dB'),('phase_plus10deg',r'Phase $+10^\circ$')]
    for key,label in labels:
        values=[c['corrected_gain_db'] for c in cal['cells'].values() if c['condition']['id']==key]
        text.append(f'{label} & {np.mean(values):.3f} & {np.min(values):.3f}'+r'\\'+'\n')
    text.append(r'''\bottomrule\end{tabular}
\parbox{\columnwidth}{\footnotesize Equal average over six SNR/RSR cells.
Positive gain means lower error. These are descriptive point estimates,
not a simultaneous robustness guarantee.}
\end{table}
''')
    (HERE/'round2_tables.tex').write_text(''.join(text))
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.6),layout='constrained')
    pilots=[6,8,10,12,20,30]
    for snr in (-5,5,15):
        rows=[geo['cells'][f'pilots_P{p}_snr{snr:+d}'] for p in pilots]
        axes[0].plot(pilots,[r['methods']['learned']['nmse_db'] for r in rows],'o-',ms=3,label=f'SNR {snr} dB')
    axes[0].set(xlabel='Total pilots P',ylabel='Learned NMSE (dB)',title='(a) Tight pilot budget',xticks=pilots)
    axes[0].legend(fontsize=7,frameon=False)
    refs=[-10,0,5,12]
    for method,label,style in [('learned','Learned','o-'),('em_spectral','EM, spectral start','s--'),('em_truth_initial_ORACLE','EM, true start (oracle)','^:')]:
        axes[1].plot(refs,[geo['cells'][f'reference_{r}_snr+20']['methods'][method]['nmse_db'] for r in refs],style,ms=3,label=label)
    axes[1].set(xlabel='Reference strength (dB)',ylabel='NMSE (dB)',title='(b) SNR 20 dB',xticks=refs)
    axes[1].legend(fontsize=6.5,frameon=False)
    for ax in axes:ax.grid(alpha=.2);ax.tick_params(labelsize=7)
    for ext in ('pdf','png'):fig.savefig(HERE/f'figures/round2_geometry.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    rows=audit['rows']
    labels={'baseline':'Ten stages / width 16','depth5':'Five stages','depth20':'Twenty stages',
            'width8':'Width 8','width32':'Width 32','experts':'Two experts','curriculum':'Curriculum'}
    table=[r'''\begin{table*}[!t]
\caption{Separately trained controls: all 56 conditions, equal-weight dB average}
\centering\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrrrrr@{}}\toprule
Variant & Parameters & Selected & Final & Gain & Worst-cell & Severe & Time\\
 & & NMSE (dB) & NMSE (dB) & (dB) & gain (dB) & (\%) & (ms)\\\midrule
''']
    for key,r in rows.items():
        table.append(f"{labels[key]} & {r['parameters']} & {r['selected_nmse_db']:.3f} & {r['final_nmse_db']:.3f} & {r['gain_db']:.3f} & {r['worst_cell_gain_db']:.3f} & {r['severe_percent']:.2f} & {r['mean_ms']:.2f}"+r'\\'+'\n')
    table.append(r'''\bottomrule\end{tabular}
\parbox{\textwidth}{\footnotesize All rows use dataset zero, three initialization seeds, 4,096 training cases and 12 epochs.
Gain is relative to ten stages/width 16; positive favors the variant.
Severe rates and gains use selected checkpoints. Time is individual-model mean inference time, not an ensemble time.
This protocol differs from the original study.}
\end{table*}
\begin{table*}[!t]
\caption{Initialization variability and conditional gains in the reduced-budget study}
\centering\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrrr@{}}\toprule
Variant & Selected seed scores (dB) & Selected epochs & Gain interval (dB) & RSR 0 gain & Two-path gain\\\midrule
''')
    for key,r in rows.items():
        seeds=', '.join(f'{v:.3f}' for v in r['selected_seed_macros'])
        epochs=', '.join(str(v) for v in r['selected_epochs'])
        interval='---' if key=='baseline' else f"$[{r['interval'][0]:.3f},{r['interval'][1]:.3f}]$"
        table.append(f"{labels[key]} & {seeds} & {epochs} & {interval} & {r['low_reference_gain_db']:.3f} & {r['two_path_gain_db']:.3f}"+r'\\'+'\n')
    table.append(r'''\bottomrule\end{tabular}
\parbox{\textwidth}{\footnotesize Initialization seeds are 9201, 9202 and 9203 in order.
Intervals are unadjusted conditional 95\% paired-case intervals, not intervals over training datasets or initialization seeds.
Subgroup gains average all matching conditions; they do not imply improvement in every matching cell.}
\end{table*}
''')
    (HERE/'round2_training_tables.tex').write_text(''.join(table))
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.6),layout='constrained')
    for arm,label,style in [('depth5','Five stages','o-'),('baseline','Ten stages','s--'),('depth20','Twenty stages','^:')]:
        axes[0].plot(snrs,[audit['per_snr_scores'][arm][str(s)] for s in snrs],style,label=label,ms=3,lw=1.2)
    axes[0].set(xlabel='SNR (dB)',ylabel='NMSE (dB)',title='(a) Separately trained depths',xticks=snrs)
    axes[0].legend(frameon=False,fontsize=7)
    names=list(labels)[1:];values=np.array([rows[k]['gain_db'] for k in names]);ci=np.array([rows[k]['interval'] for k in names])
    axes[1].errorbar(values,np.arange(len(names)),xerr=np.stack((values-ci[:,0],ci[:,1]-values)),fmt='o',capsize=3,ms=3)
    axes[1].set(yticks=np.arange(len(names)),yticklabels=[labels[k] for k in names],xlabel='Gain over baseline (dB)',title='(b) Positive favors variant')
    axes[1].axvline(0,color='.5',lw=.8);axes[1].invert_yaxis()
    for ax in axes:ax.grid(alpha=.2);ax.tick_params(labelsize=7)
    for ext in ('pdf','png'):fig.savefig(HERE/f'figures/round2_training.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]}
    (HERE/'round2_material_sources.json').write_text(json.dumps(hashes,indent=2)+'\n')

if __name__=='__main__':main()
