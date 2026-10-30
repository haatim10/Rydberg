"""Reporting-only redraw of frozen adjacent-step data with a shared y label."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/private/tmp/tcom_filter_continuation_report_matplotlib'
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tcom_followups_20261017.common import REPO,sha,write_json


def render():
    root=REPO/'results/tcom_filter_continuation_20261019/boundary'
    source=root/'summary.json';record=json.loads(source.read_text());source_digest=sha(source)
    styles={'current_repair_mean':('#d15a8a','Latest refinement','--'),
            'repair_mean':('#0969da','Filter continuation','-'),'em100':('#26965c','Direct EM100',':')}
    fig,axes=plt.subplots(3,2,figsize=(11.5,10.2),sharex=True)
    for i,regime in enumerate(('L2','L4','random3to7')):
        for j,ref in enumerate((12.,20.)):
            ax=axes[i,j]
            rows=sorted([r for r in record['adjacent_changes'] if r['regime']==regime and r['rsr']==ref],key=lambda r:r['lower_snr'])
            assert len(rows)==11
            for name,(color,label,linestyle) in styles.items():
                ax.plot([(r['lower_snr']+r['upper_snr'])/2 for r in rows],
                        [r['methods'][name]['change_db'] for r in rows],
                        color=color,label=label,ls=linestyle,marker='.',lw=1.5)
            ax.axhline(0,color='black',lw=.8);ax.axvline(38.75,color='#555555',lw=.8,ls=':')
            ax.set_title(f'{regime}, reference {ref:g} dB');ax.grid(alpha=.2)
            if i==2:ax.set_xlabel('Midpoint of adjacent SNR samples (dB)')
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,.015))
    fig.supylabel('Adjacent NMSE change (dB; positive means worse)',x=.015)
    fig.suptitle('Every adjacent boundary change, using coupled channels and noise\nRaw changes, not slopes; sample spacing is 0.25 or 0.5 dB',y=.99)
    fig.tight_layout(rect=(.035,.055,1,.94))
    outputs=[root/'nmse_adjacent_changes_readable.png',root/'nmse_adjacent_changes_readable.svg']
    fig.savefig(outputs[0],dpi=180);fig.savefig(outputs[1]);plt.close(fig)
    assert sha(source)==source_digest
    write_json(root/'plot_rendering.json',dict(passed=True,renderer_sha256=sha(__file__),
        source_summary_sha256=source_digest,original_plot_sha256=sha(root/'nmse_adjacent_changes.png'),
        output_hashes={str(p.relative_to(REPO)):sha(p) for p in outputs},
        scope='Reporting-only label/layout redraw from the same frozen JSON values. No inference, selection, data, statistical analysis or original plot changed.'))
    print('Reporting-only adjacent-change plots rendered.',flush=True)


if __name__=='__main__':render()
