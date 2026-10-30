"""Redraw audited results for IEEE column widths; no experiment changes."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(REPO/'tmp/pdfs/mpl-paper2'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=HERE/'figures';OUT.mkdir(parents=True,exist_ok=True)
paths=[REPO/'results/sahure_expansion/summary.json',REPO/'results/sahure_expansion/frozen_selection.json',
       REPO/'results/sahure_features/summary.json']
main,frozen,features=[json.loads(p.read_text()) for p in paths]
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8,
    'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
    'lines.linewidth':1.3,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':240})
BLUE='#15639b';ORANGE='#b34d18';DARK='#222222';GREEN='#2c7961';GRAY='#6b6b6b'

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.025)
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight',pad_inches=.025,dpi=240)
    plt.close(fig)

def grid(ax):ax.grid(True,alpha=.18,linewidth=.5);ax.set_axisbelow(True)

# Compact method diagram, not a hardware schematic.
fig,ax=plt.subplots(figsize=(7.16,1.82));ax.set_xlim(0,10);ax.set_ylim(0,2.7);ax.axis('off')
def box(x,y,w,h,text,fill='#f4f5f6',color=DARK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03,rounding_size=.04',facecolor=fill,edgecolor=color,lw=.7))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.75,color=DARK))
box(.05,1.42,1.4,.62,'Measurements\n+ known inputs')
box(1.8,1.42,1.4,.62,'Initial channel\nestimate')
box(3.65,1.42,1.7,.62,'Rician update\nfrom measurements')
box(5.85,1.42,1.7,.62,'Smooth\nHankel filtering',fill='#eaf3f9',color=BLUE)
box(8.12,1.42,1.75,.62,'Estimated\nchannel')
for x1,x2 in [(1.45,1.8),(3.2,3.65),(5.35,5.85),(7.55,8.12)]:arrow((x1,1.73),(x2,1.73))
ax.add_patch(FancyBboxPatch((3.45,1.22),4.3,1.08,boxstyle='round,pad=.03',facecolor='none',edgecolor=GRAY,lw=.7,linestyle='--'))
ax.text(5.6,2.47,'Repeat for 10 stages',ha='center',fontsize=8)
box(4.55,.40,2.15,.53,'Small shared controller',fill='#fff4e9',color=ORANGE)
arrow((4.50,1.42),(4.85,.93));arrow((6.7,.67),(7.0,1.40))
ax.text(3.12,.65,'Estimate features\nand known noise',ha='center',va='center',fontsize=7.5)
arrow((4.06,.67),(4.55,.67))
ax.text(7.44,.69,'Filtering\nstrength + mix',ha='left',va='center',fontsize=7.5)
ax.text(5,.06,'Training: estimate from 21 pilots; score on 9 withheld pilots.  Use: all pilots, fixed learned weights.',ha='center',fontsize=7.3)
fig.subplots_adjust(left=0,right=1,top=1,bottom=0);save(fig,'fig1_method')

fig,ax=plt.subplots(figsize=(3.48,2.5))
styles={'em_adaptive':('Proposed',BLUE,'o','-'),'hs_gs':('HS-GS',DARK,'s','--'),
        'em_global':('Global filtering',ORANGE,'D','-.'),'em_none':('No filtering',GRAY,'^',':')}
points=frozen['axes']['snr'];x=[p['value'] for p in points]
for key,(label,color,mark,ls) in styles.items():
    values=[main['cells'][p['cell']]['methods'][key] for p in points]
    y=np.array([v['nmse_db'] for v in values]);ci=np.array([v['nmse_ci'] for v in values])
    ax.plot(x,y,label=label,color=color,marker=mark,linestyle=ls,markersize=3.3)
    ax.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.10,lw=0)
ax.set(xlabel='SNR (dB)',ylabel='Channel NMSE (dB)',xticks=x,ylim=(-31,7));grid(ax)
ax.legend(loc='lower left',frameon=False);fig.tight_layout(pad=.5);save(fig,'fig2_snr')

fig,axs=plt.subplots(1,2,figsize=(7.16,2.42))
for ax,axis,xlabel in zip(axs,['rsr','paths'],['Reference-to-signal ratio (dB)','Paths per user']):
    p=frozen['axes'][axis];x=np.array([r['value'] for r in p]);v=[main['cells'][r['cell']]['methods']['em_adaptive'] for r in p]
    y=np.array([r['gain_vs_hs_db'] for r in v]);ci=np.array([r['gain_vs_hs_ci'] for r in v])
    ax.axhspan(-3,0,color=ORANGE,alpha=.045);ax.axhline(0,color=DARK,lw=.8)
    ax.errorbar(x,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),color=BLUE,fmt='o-',markersize=3.5,capsize=2)
    ax.set(xlabel=xlabel,ylabel='Gain over HS-GS (dB)',ylim=(-2.6,2.65));grid(ax)
    ax.text(.98,.07,'Below zero: HS-GS is better',ha='right',transform=ax.transAxes,fontsize=7,color=ORANGE)
    ax.set_xticks(x)
axs[0].set_title('(a) Reference strength',fontsize=8);axs[1].set_title('(b) Channel path count',fontsize=8)
fig.tight_layout(pad=.6,w_pad=1.2);save(fig,'fig3_limits')

fig,ax=plt.subplots(figsize=(3.48,2.15))
keys=['em_none','em_global','em_adaptive','rician_converged','hs_gs']
labels=['No filtering','Global filtering','Proposed','Rician gradient*','HS-GS']
y=np.arange(len(keys));times=np.array([main['timing'][k]['median_seconds']*1000 for k in keys])
ax.barh(y,times,height=.55,color=[GRAY,ORANGE,BLUE,GREEN,DARK],left=0)
for i,t in enumerate(times):ax.text(t*1.12,i,f'{t:.2f}',va='center',fontsize=7)
ax.set_xscale('log');ax.set(xlabel='Median time per case (ms, log scale)',yticks=y,yticklabels=labels,xlim=(1,1350))
ax.invert_yaxis();grid(ax);fig.tight_layout(pad=.5);save(fig,'fig4_runtime')

fig,axs=plt.subplots(1,2,figsize=(7.16,2.65),gridspec_kw={'width_ratios':[1,1.1]})
labels=['Spectrum only','Noise / residual only','Size / stage only']
others=['spectral','measurement','context'];y=np.arange(3)
for bank,offset,color,mark,label in [('bank1',-.10,BLUE,'o','Test set 1'),('bank2',.10,ORANGE,'s','Test set 2')]:
    rows=[features['banks'][bank]['contrasts'][f'full_vs_{a}'] for a in others]
    point=np.array([r['gain_db'] for r in rows]);ci=np.array([r['adjusted_ci'] for r in rows])
    axs[0].errorbar(point,y+offset,xerr=[point-ci[:,0],ci[:,1]-point],fmt=mark,color=color,capsize=2,label=label,ms=4)
    b=features['banks'][bank]['cells'];snrs=[-10,-5,0,5,10,15,20]
    gains=[b[f'snr_{s}']['selected']['measurement']['nmse_db']-b[f'snr_{s}']['selected']['full']['nmse_db'] for s in snrs]
    axs[1].plot(snrs,gains,color=color,marker=mark,ms=3.4,label=label)
axs[0].set(yticks=y,yticklabels=labels,xlabel='Average full-controller gain (dB)',xlim=(-.05,2.6))
axs[0].invert_yaxis();axs[0].axvline(0,lw=.7,color=DARK);axs[0].legend(loc='upper left',frameon=False,fontsize=7)
axs[0].set_title('(a) Gain over restricted inputs',fontsize=8)
axs[1].axhline(0,lw=.7,color=DARK)
axs[1].axhline(-.25,color=GRAY,ls='--',lw=.7,label='Protection limit')
axs[1].set(xlabel='SNR (dB)',ylabel='Full vs. noise / residual only (dB)',xticks=snrs)
axs[1].set_title('(b) Average gains hide local losses',fontsize=8)
axs[1].legend(loc='upper left',frameon=False,fontsize=6.5)
for ax in axs:grid(ax)
fig.tight_layout(pad=.6,w_pad=1.15);save(fig,'fig5_features')

record={'inputs':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'figures':sorted(p.name for p in OUT.glob('*.pdf')),
        'note':'Saved audit summaries only. No training, tuning or new evaluation. Original results unchanged.'}
(HERE/'figure_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print('Created',len(record['figures']),'vector figures in',OUT)
