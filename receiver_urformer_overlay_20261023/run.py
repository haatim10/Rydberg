"""Add unchanged proposed-receiver weights to the existing URformer test.

This is a transfer experiment, not matched-condition retraining.
All original reproduction and manuscript artifacts remain untouched.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/rydberg-overlay-matplotlib')
import json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from sahure.data import Observations
from urformer_reproduction_20261022 import study as ur
from tcom_reliability_20261021 import study as proposed,inference as receiver
from tcom_hybrid_20261020 import inference as hybrid

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'results/receiver_urformer_overlay_20261023'
CODE=Path(__file__).resolve()
SEEDS=(9101,9102,9103)
GRID=[(20,s) for s in ur.SNR]+[(p,5.) for p in ur.PILOTS if p!=20]


def frozen():
    ur.verify_protocol();proposed.verify_protocol()
    spec=json.loads((OUT/'protocol.json').read_text())
    assert ur.sha(CODE)==spec['source_sha256']
    for name,h in spec['input_hashes'].items():assert ur.sha(REPO/name)==h,name


def freeze():
    assert not (OUT/'protocol.json').exists()
    ur.verify_protocol();proposed.verify_protocol()
    paths=[ur.OUT/'data/test.npz',ur.OUT/'data/manifest.json',ur.OUT/'summary.json',
           ur.OUT/'published_curve_approx.json',ur.OUT/'completion.json',
           proposed.OUT/'protocol.json',proposed.OUT/'policy.json',proposed.OUT/'completion.json',
           *sorted((ur.OUT/'test').glob('*.npz'))]
    ur.write(OUT/'protocol.json',dict(
        purpose='Add frozen proposed receiver to the same four-user URformer reproduction plots.',
        scope='Transfer test: proposed weights trained on three users,30inputpilots,geometric3-7/2/4/8/12/16path channels; tested unchanged on four-user40ray channel and10-35pilots.',
        fairness='Computed methods use identical test measurements,pilots,reference,noise variance,truth and2000physicalchannels. Training conditions differ; not a matched-training architecture comparison.',
        inputs='Original observations reconstructed without rescaling,normalization,extra pilots,or new noise.',
        receiver='Full unchanged receiver entry point,original selected policy and188parameter adaptive checkpoints,threefits9101/2/3. No retuning or additional refinement.',
        branch='All tested cases must follow the existing ordinary-SNR adaptive branch. No high-SNR algebraic recovery is exercised; that initializer is declared forK3.',
        initialization='Each method retains its published/frozen initializer: proposed spectral,URformer/classical reproduction random. No common-initializer ablation claim.',
        scoring='Per-example squared error divided by true channel energy; mean linearly over2000channels and3fits,then10log10. Same as existing URformer plot.',
        seeds=SEEDS,grid=GRID,source_sha256=ur.sha(CODE),
        input_hashes={str(p.relative_to(REPO)):ur.sha(p) for p in paths},
        weights=json.loads((proposed.OUT/'protocol.json').read_text())['checkpoint_hashes'],
        selection='No new training,validation selection,gate or schedule tuning. All12conditions are reported.',
        created_unix=time.time()))


def observations(a,p,snr):
    z,s,b,v,_=ur.observation(a,p,snr,double=True)
    n=len(z)
    return Observations(z,s[None].expand(n,-1,-1).clone(),torch.ones_like(z,dtype=torch.complex128)*b,v)


@torch.no_grad()
def checks(est,rule):
    a={k:v[:4] for k,v in ur.data('val').items()};maximum=0.;scalar_error=0.
    for p,snr in ((10,-10.),(20,5.),(35,20.)):
        obs=observations(a,p,snr)
        assert all(plan['ordinary'] and plan['family']=='adaptive' and plan['steps']==0
                   for plan in hybrid.plans(obs,receiver.BASE_POLICY))
        for seed in SEEDS:
            g,diag=receiver.gated(obs,seed,rule,est,diagnostics=True)
            expected=est['starts'].adaptive_models[seed](*obs.args())
            difference=float((g-expected).abs().max());maximum=max(maximum,difference)
            assert difference<1e-8,difference
            assert g.shape==(4,32,4) and torch.isfinite(g).all()
            assert not diag['eligible'].any() and not diag['triggered'].any()
            scalar=torch.cat([receiver.gated(obs.batch(slice(i,i+1)),seed,rule,est) for i in range(4)])
            difference=float((scalar-g).abs().max());scalar_error=max(scalar_error,difference)
            assert difference<1e-8,difference
    ur.write(OUT/'checks.json',dict(passed=True,canonical_torch_max_error=maximum,
        batch_one_max_error=scalar_error,full_pipeline_equals_original_adaptive_branch=True,
        no_high_snr_recovery=True,cases='4validationchannels x3conditions x3fits'))


@torch.no_grad()
def evaluate(est,rule):
    a=ur.data('test');rows=[]
    for p,snr in GRID:
        path=OUT/'cells'/f'P{p}_snr{snr:g}.npz'
        if not path.exists():
            obs=observations(a,p,snr);ratios=[]
            assert all(x['ordinary'] for x in hybrid.plans(obs,receiver.BASE_POLICY))
            for seed in SEEDS:
                parts=[]
                for start in range(0,len(obs),64):
                    sl=slice(start,start+64)
                    g,diag=receiver.gated(obs.batch(sl),seed,rule,est,diagnostics=True)
                    assert not diag['eligible'].any() and not diag['triggered'].any()
                    assert torch.isfinite(g).all()
                    parts.append(ur.ratios(g,a['h'][sl].to(torch.complex128)).numpy())
                ratios.append(np.concatenate(parts))
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,ratios=np.stack(ratios),P=p,snr=snr)
        with np.load(path) as f:r=f['ratios'].copy()
        assert r.shape==(3,2000) and np.isfinite(r).all()
        rows.append(dict(P=p,snr=snr,nmse_db=float(10*np.log10(r.mean())),
                         nmse_linear=float(r.mean()),per_fit_nmse_db=(10*np.log10(r.mean(1))).tolist()))
        print('Proposed transfer',p,snr,rows[-1]['nmse_db'],'dB',flush=True)
    frozen();ur.write(OUT/'scores.json',dict(rows=rows,passed=True))
    return rows


def plot_and_report(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=json.loads((ur.OUT/'summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12.,4.7))
    plt.rcParams.update({'font.size':10})
    for ax,which in zip(axes,('fig3','fig4')):
        points=source['comparison'][which]['points']
        xx=[v['snr'] if which=='fig3' else v['P'] for v in points]
        lookup=lambda q:next(r for r in rows if r['P']==q['P'] and r['snr']==q['snr'])
        values=[lookup(q) for q in points]
        ax.plot(xx,[r['nmse_db'] for r in values],'-*',color='#178252',lw=2.4,ms=9,
                label='Proposed (original weights, transfer test)',zorder=6)
        ax.fill_between(xx,[min(r['per_fit_nmse_db']) for r in values],
                        [max(r['per_fit_nmse_db']) for r in values],color='#178252',alpha=.12)
        ax.plot(xx,[r['ours_nmse_db'] for r in points],'-o',color='#1967a3',lw=1.5,label='Our URformer: final epoch 50')
        ax.plot(xx,[r['published_approx_db'] for r in points],'--o',color='#8b60a8',mfc='none',label='Published URformer (approx.)')
        ax.plot(xx,[r['em100_nmse_db'] for r in points],'-s',color='#d77b28',label='Our EM-GS: 100 steps')
        ax.plot(xx,[r['published_em_approx_db'] for r in points],'--s',color='#d77b28',mfc='none',label='Published EM-GS (approx.)')
        ax.plot(xx,[r['gs100_nmse_db'] for r in points],':',color='#777777',label='Our GS: 100 steps')
        ax.set(xlabel='SNR (dB)' if which=='fig3' else 'Number of pilots',ylabel='NMSE (dB)',xticks=xx,
               title='20 pilots; four users' if which=='fig3' else 'SNR 5 dB; four users')
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=7.8,loc='upper right')
    fig.text(.5,.015,'Proposed receiver: original three-user / 30-pilot weights, tested without retraining. Same test inputs for computed curves. Lower is better.',
             ha='center',fontsize=8.3)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(OUT/'comparison_with_proposed.png',dpi=190)
    fig.savefig(OUT/'comparison_with_proposed.svg');plt.close(fig)
    lines=['Proposed receiver added to the URformer comparison','',
        'The green curve is our unchanged proposed receiver, averaged over its three trained fits.',
        'This is a transfer test. Its weights were trained with three users and30pilots on a different channel model.',
        'Here it sees the same four-user measurements and pilots as the computed URformer and classical methods.',
        'No retraining, extra EM steps, normalization or test-based tuning was performed.',
        'All these conditions use the original ordinary-SNR adaptive branch. This does not test high-SNR recovery for four users.',
        'The published URformer curve remains unreproduced; this plot cannot establish superiority over that published method.',
        '', 'SNR sweep with20pilots. Lower NMSE is better.',
        'SNR | Proposed transfer | Our URformer | Our EM-GS | Published URformer approx.']
    for q in source['comparison']['fig3']['points']:
        r=next(r for r in rows if r['P']==20 and r['snr']==q['snr'])
        lines.append(f"{q['snr']:4g} | {r['nmse_db']:17.3f} | {q['ours_nmse_db']:12.3f} | {q['em100_nmse_db']:9.3f} | {q['published_approx_db']:25.1f}")
    lines+=['','Pilot sweep at SNR5dB:']
    for q in source['comparison']['fig4']['points']:
        r=next(r for r in rows if r['P']==q['P'] and r['snr']==5.)
        lines.append(f"{q['P']:2d} pilots: proposed transfer {r['nmse_db']:.3f} dB; our URformer {q['ours_nmse_db']:.3f} dB; EM-GS {q['em100_nmse_db']:.3f} dB.")
    text='\n'.join(lines)+'\n';(OUT/'RESULTS.txt').write_text(text)
    import shutil
    project=REPO.parents[1]
    shutil.copy2(OUT/'comparison_with_proposed.png',project/'Receiver_Comparison_With_URformer.png')
    (project/'Receiver_Comparison_With_URformer_Results.txt').write_text(text)
    paths=[CODE,*[p for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json']]
    dest=project/'Receiver_Comparison_With_URformer_Evidence.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(REPO)))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    frozen();ur.write(OUT/'completion.json',dict(passed=True,conditions=12,cases_per_condition=2000,
        fits=3,bundle_sha256=ur.sha(dest),code_sha256=ur.sha(CODE),finished_unix=time.time()))
    print(text,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not (OUT/'protocol.json').exists():freeze()
    frozen();est=proposed.engine();rule=proposed.check_selection()[0]['candidate_rule']
    checks(est,rule);rows=evaluate(est,rule);plot_and_report(rows)
