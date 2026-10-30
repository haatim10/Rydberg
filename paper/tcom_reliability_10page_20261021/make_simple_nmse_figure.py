"""Change only NMSE legend and y-axis wording; preserve frozen values, axes, and geometry."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tcom-clean-ber-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVIDENCE = HERE / 'evidence/reliability'
SOURCE = REPO / 'results/tcom_reliability_20261021'
if not (SOURCE/'nmse_summary.json').exists():
    SOURCE = EVIDENCE/'source'
FIGURES = HERE/'figures'
NAMES = ['receiver_mean','prior_mean','latest_mean','compact_sparse','global40_mean','algebraic20']
LABELS = {'receiver_mean':'Proposed receiver', 'prior_mean':'Previous hybrid',
          'latest_mean':'Full repair', 'compact_sparse':'Sparse + EM',
          'global40_mean':'Fixed schedule + EM40', 'algebraic20':'Algebraic + EM20'}
COLORS = {'receiver_mean':'#0072B2', 'prior_mean':'#D55E00', 'latest_mean':'#CC79A7',
          'compact_sparse':'#009E73', 'global40_mean':'#E69F00', 'algebraic20':'#444444'}
STYLES = {'receiver_mean':'-', 'prior_mean':'--', 'latest_mean':':',
          'compact_sparse':'-.', 'global40_mean':'--', 'algebraic20':':'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def dump(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    bound = ['protocol.json','policy.json','nmse_summary.json','completion.json']
    before = {f:sha(SOURCE/f) for f in bound}
    n, protocol, completion = [load(SOURCE/f) for f in ('nmse_summary.json','protocol.json','completion.json')]
    assert completion['status']=='complete' and completion['all_required_audits_passed']
    assert n['selection_sha256']==before['policy.json']
    assert n['protocol_sha256']==before['protocol.json']
    ledger = load(EVIDENCE/'figure_values.json')
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
                         'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
                         'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,
                         'lines.linewidth':1.15,'axes.linewidth':.65})
    snrs, refs = protocol['snrs'], protocol['rsrs']
    fig, axes = plt.subplots(1,3,figsize=(6.8,2.65),sharex=True,sharey=True)
    verified = 0
    for ax, ref in zip(axes,refs):
        for name in NAMES:
            vals = [float(np.mean([c['methods'][name]['nmse_db'] for c in n['cells']
                                  if c['rsr']==ref and c['snr']==snr])) for snr in snrs]
            assert vals==ledger['values']['nmse'][f'{name}/rsr{ref:g}']
            line,=ax.plot(snrs,vals,color=COLORS[name],ls=STYLES[name],
                          marker='o' if name=='receiver_mean' else None,ms=2.1,label=LABELS[name])
            np.testing.assert_array_equal(line.get_ydata(),vals)
            verified += len(vals)
        ax.set_title(f'RSR {ref:g} dB'); ax.set_xlabel('SNR (dB)')
        ax.grid(True,color='#dddddd',lw=.45); ax.set_xticks([0,20,40,60]); ax.set_xlim(-1,61)
    axes[0].set_ylabel('Mean NMSE (dB)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,
               frameon=False,bbox_to_anchor=(.5,1.01),columnspacing=1.5,handlelength=2.6)
    fig.subplots_adjust(left=.088,right=.993,bottom=.18,top=.73,wspace=.14)
    for ext in ('pdf','png'): fig.savefig(FIGURES/f'reliability_nmse.{ext}',dpi=240)
    plt.close(fig)
    assert before=={f:sha(SOURCE/f) for f in bound}
    output={f'figures/reliability_nmse.{ext}':sha(FIGURES/f'reliability_nmse.{ext}') for ext in ('pdf','png')}
    result=dict(passed=True,change='Legend and y-axis wording only',labels=LABELS,exact_values_verified=verified,
                original_axis_limits_ticks_styles_and_geometry_preserved=True,
                source_hashes=before,frozen_sources_unchanged=True,generated_hashes=output,generator_sha256=sha(__file__))
    dump(EVIDENCE/'simple_nmse_figure_validation.json',result)
    v=load(EVIDENCE/'validation.json')
    v['generated_hashes'].update(output)
    v['nmse_figure_revision']=dict(generator='make_simple_nmse_figure.py',validation='simple_nmse_figure_validation.json')
    dump(EVIDENCE/'validation.json',v)
    manifest={str(p.relative_to(EVIDENCE)):sha(p) for p in sorted(EVIDENCE.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    dump(EVIDENCE/'SHA256SUMS.json',manifest)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
