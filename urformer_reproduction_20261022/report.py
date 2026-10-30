"""Presentation and checks; never used for model/checkpoint selection."""
import json, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from .study import OUT, CODE, REPO, SEEDS, SNR, PILOTS, sha, write, verify_protocol


def digitize():
    # Standalone evidence bundles retain our readings and source hashes/URL,
    # rather than redistributing the externally published article/figures.
    if not (OUT/'reference/fig3.png').exists():
        return json.loads((OUT/'published_curve_approx.json').read_text())['curves']
    # Axis calibration from the published raster's labeled tick centers.
    # Purple marker centers are measured locally; round to .1dB, not raw data.
    specs = [dict(name='fig3',x0=105.5,x1=1035.,xmin=-10,xmax=20,
                  y0=179.,units_per_pixel=-10/(315.5-179.),
                  x=SNR,targets=[195,325,411,474,538,590,615],em_targets=[63,123,189,256,330,398,462]),
             dict(name='fig4',x0=105.5,x1=1034.,xmin=10,xmax=35,
                  y0=159.,units_per_pixel=-5/(252.-159.),
                  x=PILOTS,targets=[483,535,569,589,602,616],em_targets=[150,218,269,302,326,344])]
    result={}
    for spec in specs:
        a=np.array(Image.open(OUT/'reference'/f"{spec['name']}.png").convert('RGB'))
        color=np.array([139,96,168]);mask=(np.abs(a.astype(int)-color).max(-1)<8)
        rows=[]
        for x,target,em_target in zip(spec['x'],spec['targets'],spec['em_targets']):
            xp=spec['x0']+(x-spec['xmin'])/(spec['xmax']-spec['xmin'])*(spec['x1']-spec['x0'])
            yy,xx=np.where(mask & (np.abs(np.arange(a.shape[1])[None,:]-xp)<3)
                          & (np.abs(np.arange(a.shape[0])[:,None]-target)<18))
            assert len(yy)>0
            center=(float(yy.min())+float(yy.max()))/2
            db=(center-spec['y0'])*spec['units_per_pixel']
            orange=(np.abs(a.astype(int)-np.array([244,122,32])).max(-1)<8)
            ey,ex=np.where(orange & (np.abs(np.arange(a.shape[1])[None,:]-xp)<3)
                          & (np.abs(np.arange(a.shape[0])[:,None]-em_target)<18))
            assert len(ey)>0
            ec=(float(ey.min())+float(ey.max()))/2
            rows.append(dict(x=float(x),nmse_db_approx=round(db,1),pixel_center_y=center,
                             em_nmse_db_approx=round((ec-spec['y0'])*spec['units_per_pixel'],1)))
        result[spec['name']]=rows
    write(OUT/'published_curve_approx.json',dict(
        source='arXivv3 Fig3/Fig4 purple URformer and orange EM-GS markers, automated raster center measurement',
        caveat='Approximate visual reference, .1dB rounding and roughly .2dB reading uncertainty; not authors raw numerical data.',
        settings=specs,curves=result))
    return result


def summary():
    verify_protocol();assert json.loads((OUT/'test/completed.json').read_text())['passed']
    pub=digitize();rows=[]
    for p in sorted((OUT/'test').glob('*.npz')):
        with np.load(p) as a:
            row=dict(P=int(a['P']),snr=float(a['snr']),methods={})
            for key in a.files:
                if key.startswith(('URformer','GS','EM')):
                    v=a[key];assert len(v)==2000 and np.isfinite(v).all() and (v>=0).all()
                    row['methods'][key]=dict(nmse_linear=float(v.mean()),nmse_db=float(10*np.log10(v.mean())))
            energy=a['truth_energy']
            for label in ('final','best'):
                v=np.stack([a[f'URformer_{s}_{label}'] for s in SEEDS])
                point=float(v.mean());random=np.random.default_rng(2026102250)
                boots=[]
                # Resample physical channels, keeping the three fits paired.
                averaged=v.mean(0)
                for _ in range(2000):boots.append(averaged[random.integers(0,len(averaged),len(averaged))].mean())
                row[f'URformer_{label}']=dict(nmse_linear=point,nmse_db=float(10*np.log10(point)),
                    seed_db_range=[float(10*np.log10(v.mean(1).min())),float(10*np.log10(v.mean(1).max()))],
                    descriptive_channel_bootstrap95_db=(10*np.log10(np.quantile(boots,[.025,.975]))).tolist(),
                    ratio_of_error_sums_db=float(10*np.log10((v*energy[None]).sum()/(3*energy.sum()))))
            rows.append(row)
    histories={}
    for s in SEEDS:
        f=OUT/'training'/str(s)
        import torch
        best=torch.load(f/'best.pt',weights_only=False,map_location='cpu')
        final=torch.load(f/'final.pt',weights_only=False,map_location='cpu')
        assert final['epoch']==50 and len(final['history'])==50
        assert best['best_score']==min([entry['validation_nmse'] for entry in final['history']]+[final['history'][0]['validation_nmse'],best['best_score']])
        histories[str(s)]=dict(final_epoch=50,best_epoch=best['epoch'],
            final_validation_nmse_db=final['history'][-1]['validation_db'],best_validation_nmse_db=10*np.log10(best['best_score']),
            training_seconds=sum(x['seconds'] for x in final['history']),
            final_weights_sha256=sha(f/'final.pt'),best_weights_sha256=sha(f/'best.pt'))
    comparison={}
    for fig,points in pub.items():
        selected=[]
        for point in points:
            p,snr=(20,point['x']) if fig=='fig3' else (int(point['x']),5.)
            row=next(r for r in rows if r['P']==p and r['snr']==snr)
            db=row['URformer_final']['nmse_db'];ref=point['nmse_db_approx']
            selected.append(dict(P=p,snr=snr,ours_nmse_db=db,published_approx_db=ref,difference_db=db-ref,
                                 em100_nmse_db=row['methods']['EM100']['nmse_db'],gs100_nmse_db=row['methods']['GS100']['nmse_db'],
                                 published_em_approx_db=point['em_nmse_db_approx'],
                                 em_difference_db=row['methods']['EM100']['nmse_db']-point['em_nmse_db_approx']))
        comparison[fig]=dict(points=selected,max_absolute_difference_db=max(abs(x['difference_db']) for x in selected),
                             all_within_one_db_visual_diagnostic=all(abs(x['difference_db'])<=1 for x in selected),
                             em_max_absolute_difference_db=max(abs(x['em_difference_db']) for x in selected))
    result=dict(status='complete',protocol_sha256=sha(OUT/'protocol.json'),histories=histories,rows=rows,
                comparison=comparison,interpretation='Independent paper-setting implementation with disclosed missing-setting assumptions. Training completion is not published-performance reproduction. Fig4 is transfer of P20-trained weights.',
                prior_records_unchanged=True)
    write(OUT/'summary.json',result)
    return result


def plots(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10.4,4.1))
    for ax,which,label in zip(axs,('fig3','fig4'),('SNR (dB)','Number of pilots')):
        points=result['comparison'][which]['points'];x=[p['snr'] if which=='fig3' else p['P'] for p in points]
        ours=[p['ours_nmse_db'] for p in points];published=[p['published_approx_db'] for p in points]
        ax.plot(x,ours,'o-',color='#1967a3',label='Our URformer: final epoch 50')
        ax.plot(x,published,'o--',color='#8b60a8',markerfacecolor='none',label='Published URformer (approx.)')
        ax.plot(x,[p['em100_nmse_db'] for p in points],'s-',color='#d77b28',label='Our EM-GS: 100 steps')
        ax.plot(x,[p['published_em_approx_db'] for p in points],'s--',color='#d77b28',markerfacecolor='none',label='Published EM-GS (approx.)')
        ax.plot(x,[p['gs100_nmse_db'] for p in points],':',color='#777777',label='Our GS: 100 steps')
        ax.set_xlabel(label);ax.set_ylabel('NMSE (dB)');ax.grid(alpha=.2);ax.set_xticks(x)
        ax.set_title('20 pilots, reference ratio 10 dB' if which=='fig3' else 'SNR 5 dB; trained with 20 pilots')
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=190);fig.savefig(OUT/'comparison.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,3.9))
    for seed in SEEDS:
        hist=json.loads((OUT/'training'/str(seed)/'history.json').read_text())['history']
        ax.plot([x['epoch'] for x in hist],[x['validation_db'] for x in hist],label=f'Seed {seed}')
    ax.set(xlabel='Training epoch',ylabel='Validation NMSE (dB)',title='Independent URformer fits');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=190);plt.close(fig)


def report(result):
    point5=next(r for r in result['rows'] if r['P']==20 and r['snr']==5.)
    pub5=next(r for r in result['comparison']['fig3']['points'] if r['snr']==5.)
    verdict=('The computed URformer curve is within 1 dB of the digitized Fig. 3 curve at every point.'
             if result['comparison']['fig3']['all_within_one_db_visual_diagnostic'] else
             'The published URformer accuracy was NOT reproduced by this implementation under the disclosed assumptions.')
    lines=['URformer paper-setting reproduction attempt','',verdict,
           f"At SNR 5 dB with 20 pilots: our final 50-epoch mean NMSE = {point5['URformer_final']['nmse_db']:.3f} dB; our best-validation mean NMSE = {point5['URformer_best']['nmse_db']:.3f} dB; published plotted value approximately {pub5['published_approx_db']:.1f} dB.",
           'The numerical checks passed. This checks our calculations; it does not establish that our code matches the authors code.',
           '',
           'All three independent fits completed 50 epochs on 20,000 training examples.',
           '32 antennas, 4 users, 40 rays, 20 pilots, and a 10 dB reference ratio using the paper total-signal definition.',
           'This is our implementation of the published equations. Author code/weights were not found.',
           'The paper omits important settings; every assumption is in protocol.json.',
           '', 'Primary test results (mean normalized channel error, converted to dB; 2,000 channels per point):',
           'SNR dB | our URformer dB | published URformer approx. dB | our EM-GS100 dB | published EM approx. dB']
    for p in result['comparison']['fig3']['points']:
        lines.append(f"{p['snr']:6g} | {p['ours_nmse_db']:14.3f} | {p['published_approx_db']:25.1f} | {p['em100_nmse_db']:14.3f} | {p['published_em_approx_db']:21.1f}")
    lines+=['','Additional best-validation checkpoint results (selected without looking at the test set):',
            'SNR dB | mean NMSE dB']
    for snr in SNR:
        row=next(r for r in result['rows'] if r['P']==20 and r['snr']==snr)
        lines.append(f"{snr:6g} | {row['URformer_best']['nmse_db']:12.3f}")
    lines+=['','Pilot sweep (same models trained with 20 pilots; the paper does not state how models were trained for each pilot count):',
            'Pilots | our URformer dB | published URformer approx. dB | our EM-GS100 dB | published EM approx. dB']
    for p in result['comparison']['fig4']['points']:
        lines.append(f"{p['P']:6g} | {p['ours_nmse_db']:14.3f} | {p['published_approx_db']:25.1f} | {p['em100_nmse_db']:14.3f} | {p['published_em_approx_db']:21.1f}")
    lines+=['','Final 50-epoch models are primary. Best models were selected using separate validation data.',
            'No changes to our receiver, old results, or manuscript. No test-guided tuning in this attempt.',
            'This four-user experiment does not establish which estimator is better on our three-user problem.',
            'CNN and standalone Transformer baselines were omitted because their architectures were not specified.',
            '', 'Source: https://arxiv.org/html/2509.12586v3','']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines))
    project=REPO.parents[1]
    (project/'URformer_Reproduction_Results.txt').write_text('\n'.join(lines))
    import shutil
    shutil.copy2(OUT/'comparison.png',project/'URformer_Reproduction_Comparison.png')
    return project


def bundle(project):
    paths=list(CODE.glob('*.py'))+[CODE/'README.txt']
    paths += [REPO/p for p in verify_protocol()['protected_hashes']]
    # The paper remains a linked external reference, not redistributed source.
    paths += [p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.pt','.npz') and not {'data','reference'}.intersection(p.relative_to(OUT).parts) and p.name not in ('execution.lock','completion.json')]
    paths += list((OUT/'test').glob('*.npz'))
    paths += [OUT/'data/manifest.json']
    for seed in SEEDS:
        paths += [OUT/'training'/str(seed)/f'{label}.pt' for label in ('final','best')]
    dest=project/'URformer_Reproduction_Evidence.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in paths:z.write(p,str(p.relative_to(REPO)))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    verify_protocol()
    write(OUT/'completion.json',dict(status='complete',finished_unix=time.time(),protocol_sha256=sha(OUT/'protocol.json'),
        summary_sha256=sha(OUT/'summary.json'),bundle_sha256=sha(dest),bundle_bytes=dest.stat().st_size,
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in CODE.glob('*.py')},
        scope='Includes six model/checkpoint files, per-example test errors, protocols, source, figures and logs. Input banks regenerable from source and seeds; hashes included. Latest optimizer checkpoints omitted.'))


if __name__=='__main__':
    r=summary();plots(r);project=report(r);bundle(project);print((OUT/'RESULTS.txt').read_text())
