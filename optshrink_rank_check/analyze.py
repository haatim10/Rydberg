import json
import numpy as np
import torch
from sahure_expansion.data import world
from sahure.classical import likelihood
from paper2_optshrink_loop.estimator import trajectory as np_trajectory, shrink, estimate as np_estimate
from .estimator import parameters
from .run import ROOT,PARENT,CONFIG,SPECS,sha,write,save,load,initialize,models,tensors,frozen_parent,fixed_configs
from fair_optshrink.estimator import lookup_key


def reference(row,parent):
    obs=tensors(row);entry=parent['choices'][lookup_key(obs)[0]];z,s,b,v=[row[k][0] for k in ('z','s','b','sigma2')]
    settings=[];scores=[];outputs=[]
    for p in ('loop','final'):
        base=entry[p];t=base['iterations'];params=parameters(base['strength'])
        gg=np_trajectory(z[:,:21],s[:,:21],b[:,:21],v,params if p=='loop' else [params[0]],(t,))[t]
        if p=='final': gg=shrink(np.repeat(gg,16,axis=0),params)
        score=likelihood(z[:,21:],gg@s[:,21:]+b[:,21:],v).mean((-2,-1));r=int(score.argmin())
        cfg=dict(**params[r],iterations=t,placement=p);settings.append(cfg);scores.append(score)
        outputs.append(np_estimate(z,s,b,v,cfg))
    which=int(scores[1].min()<scores[0].min())
    return outputs+[outputs[which]],np.stack(scores),settings+[settings[which]]


@torch.no_grad()
def audit(check):
    initialize(); frozen=json.loads((ROOT/'frozen_selection.json').read_text());digest=sha(ROOT/'frozen_selection.json')
    parent=frozen_parent();learned=models();ids=set();hashes={};replays=0;max_error=0.
    for bank in range(2):
        for j,sp in enumerate(SPECS):
            check('audit',bank=bank,cell=j)
            folder=ROOT/'data'/f'bank{bank}'/sp['id'];meta=json.loads((folder/'manifest.json').read_text())
            if meta['selection_sha256']!=digest or meta['generated_unix']<frozen['frozen_unix']: raise RuntimeError('freeze chronology')
            for p,h in meta['hashes'].items():
                if sha(p)!=h: raise RuntimeError('input hash mismatch')
                hashes[p]=h
            d,g=load(folder/'observations.npz'),load(folder/'truth.npz');p=ROOT/'predictions'/f'bank{bank}_{sp["id"]}.npz'
            out=load(p);hashes[str(p)]=sha(p);names=out['names'].tolist()
            if str(out['selection_sha256'])!=digest: raise RuntimeError('prediction freeze')
            for other in (g['trial_ids'],out['trial_ids']): np.testing.assert_array_equal(d['trial_ids'],other)
            if ids.intersection(d['trial_ids']): raise RuntimeError('duplicate worlds')
            ids.update(d['trial_ids'])
            for i in range(meta['count']):
                w,_=world(i,meta['seed'],**sp)
                for k,v in zip(('z','s','b','sigma2'),(w.Z,w.S,w.B,w.sigma2)): np.testing.assert_array_equal(d[k][i],v)
                np.testing.assert_array_equal(g['g'][i],w.G)
            row={k:d[k][:1] for k in ('z','s','b','sigma2')};obs=tensors(row)
            expected,scores,cfgs=reference(row,parent)
            np.testing.assert_allclose(out['rank_scores'][0],scores,rtol=3e-8,atol=3e-9)
            for name,ee,cfg,config in zip(('rank_loop','rank_final','rank_joint'),expected,cfgs,out['rank_configs'][0]):
                np.testing.assert_array_equal(config,[cfg['rank'],cfg['strength'],cfg['iterations'],int(cfg['placement']=='final')])
                actual=out['estimates'][names.index(name),0]
                np.testing.assert_allclose(actual,ee,rtol=3e-8,atol=3e-9);max_error=max(max_error,float(abs(actual-ee).max()));replays+=1
            for name,m in learned.items():
                np.testing.assert_allclose(out['estimates'][names.index(name),0],m(*obs.args()).numpy()[0],rtol=3e-8,atol=3e-9);replays+=1
            for name,cfg in fixed_configs(obs,parent).items():
                ee=np_estimate(d['z'][0],d['s'][0],d['b'][0],float(d['sigma2'][0]),cfg)
                np.testing.assert_allclose(out['estimates'][names.index(name),0],ee,rtol=3e-8,atol=3e-9);replays+=1
    result=dict(status='passed',regenerated_test=len(ids),representative_replays=replays,
        independent_rank_max_abs_error=max_error,hashes=hashes,scope='numerical audit conditional on frozen models/settings')
    write(ROOT/'audit.json',result);return result


def analyze(check=lambda *a,**k:None):
    verification=audit(check);ee=[];pp=[]
    for bank in range(2):
        eb=[];pb=[]
        for sp in SPECS:
            out=load(ROOT/'predictions'/f'bank{bank}_{sp["id"]}.npz'); names=out['names'].tolist()
            truth=load(ROOT/'data'/f'bank{bank}'/sp['id']/'truth.npz')['g']
            eb.append((abs(out['estimates']-truth[None])**2).sum((-2,-1)));pb.append((abs(truth)**2).sum((-2,-1)))
        ee.append(eb);pp.append(pb)
    errors=np.asarray(ee);power=np.asarray(pp);ratio=errors/power[:,:,None,:]
    save(ROOT/'all_errors.npz',squared_error=errors,channel_power=power,names=np.array(names),cell_ids=np.array([s['id'] for s in SPECS]))
    groups={n:[names.index(n)] for n in names if not n.startswith('learned_')};groups['learned']=[names.index('learned_'+str(s)) for s in CONFIG['seeds']]
    db=10*np.log10(errors.sum((0,3))/power.sum((0,2))[:,None]);bankdb=10*np.log10(errors.sum(-1)/power.sum(-1)[...,None])
    primary={g:float(db[:126,ii].mean()) for g,ii in groups.items()};cells={}
    for j,sp in enumerate(SPECS):
        cells[sp['id']]={g:dict(nmse_db=float(db[j,ii].mean()),mean_case_nmse=float(ratio[:,j,ii,:].mean()),
            exceedance={str(t):float((ratio[:,j,ii,:]>t).mean()) for t in (.5,1.,2.)}) for g,ii in groups.items()}
    save(ROOT/'error_distributions.npz',**{g:np.sort(ratio[:,:126,ii,:].ravel()) for g,ii in groups.items()})
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);comparators=('rank_joint','rank_loop','rank_final');samples=[]
    for start in range(0,CONFIG['bootstrap_samples'],100):
        check('bootstrap',done=start);n=CONFIG['cases'];count=min(100,CONFIG['bootstrap_samples']-start)
        w=rng.multinomial(n,np.ones(n)/n,size=(count,2,126))
        e=np.einsum('rbcn,bcmn->rcm',w,errors[:,:126],optimize=True);p=np.einsum('rbcn,bcn->rc',w,power[:,:126],optimize=True)
        d=10*np.log10(e/p[...,None]);learn=d[...,groups['learned']].mean(-1)
        samples.append(np.stack([(d[...,names.index(c)]-learn).mean(1) for c in comparators],-1))
    samples=np.concatenate(samples);contrasts={};alpha=.05/6
    for i,c in enumerate(comparators):
        contrasts[c]=dict(gain_db=primary[c]-primary['learned'],simultaneous95_macro_ci=np.quantile(samples[:,i],[alpha,1-alpha]).tolist(),
            worst_cell_deterioration_db=float(np.max(db[:126,groups['learned']].mean(-1)-db[:126,names.index(c)])))
    timingrows=[json.loads((ROOT/'timing'/f'{sp["id"]}.json').read_text()) for sp in SPECS]
    timing={g:dict(mean_ms=float(1000*np.mean([v for row in timingrows[:126] for i in ii for v in row['seconds'][names[i]]])),
        median_ms=float(1000*np.median([v for row in timingrows[:126] for i in ii for v in row['seconds'][names[i]]]))) for g,ii in groups.items()}
    summary=dict(primary=primary,cells=cells,contrasts=contrasts,timing=timing,
        bank_primary={g:bankdb[:,:126,ii].mean((1,2)).tolist() for g,ii in groups.items()},
        learned_seed_primary={str(s):float(db[:126,names.index('learned_'+str(s))].mean()) for s in CONFIG['seeds']},
        primary_threshold_rates={g:{str(t):float((ratio[:,:126,ii,:]>t).mean()) for t in (.5,1.,2.)} for g,ii in groups.items()},
        ratio_of_sums_pooled={g:float(np.mean(10*np.log10(errors[:,:126,ii].sum((0,1,3))/power[:,:126].sum()))) for g,ii in groups.items()},
        mean_case_linear={g:float(ratio[:,:126,ii].mean()) for g,ii in groups.items()},audit=verification,
        caveats=['Shared rank across users; offline-frozen blending/iteration budgets, not unrestricted per-case optimization.',
        'Same3 original checkpoints, one shared training dataset; intervals conditional on models and validation settings.',
        'Fresh follow-up test bank; do not pool with older studies. No publication or universal superiority guarantee.'])
    # Independent scalar-loop recalculation of the headline from raw arrays.
    for group,ii in groups.items():
        scalar=np.mean([10*np.log10(sum(float(errors[b,c,m,n]) for b in range(2) for n in range(CONFIG['cases']))/
            sum(float(power[b,c,n]) for b in range(2) for n in range(CONFIG['cases']))) for c in range(126) for m in ii])
        np.testing.assert_allclose(scalar,primary[group],rtol=1e-12,atol=1e-12)
    summary['independent_headline_recalculation']='passed';write(ROOT/'summary.json',summary)
    lines=['FINAL ONLINE-RANK OPTSHRINK COMPARISON','',
        '4096 new paired worlds;126 primary conditions plus2 separate stress conditions.',
        'No new neural training. Rank chosen from21/9 observed pilot split; final refit uses all30.',
        'All rank-search costs counted; original and preceding studies unchanged.','',
        'PRIMARY MACRO NMSE (lower better) / MEAN END-TO-END TIME']
    lines += [f'{g}: {primary[g]:.3f} dB / {timing[g]["mean_ms"]:.3f} ms' for g in groups]
    lines += ['', 'LEARNED GAINS (positive favours learning), adjusted95% macro intervals']
    lines += [f'{c}: {v["gain_db"]:.3f} dB, {v["simultaneous95_macro_ci"]}; worst-cell harm {v["worst_cell_deterioration_db"]:.3f} dB' for c,v in contrasts.items()]
    lines += ['', 'THRESHOLD EXCEEDANCE .5 /1 /2']
    lines += [g+': '+' / '.join(f'{100*summary["primary_threshold_rates"][g][str(t)]:.3f}%' for t in (.5,1.,2.)) for g in groups]
    lines += ['', 'STRESS CELLS (not in primary average)']
    lines += [sp['id']+': '+', '.join(f'{g} {cells[sp["id"]][g]["nmse_db"]:.3f}' for g in ('learned','rank_joint','opt_selected','em100')) for sp in SPECS[126:]]
    lines += ['', 'Audit and independent statistic recalculation passed. Full details:summary.json.',*summary['caveats']]
    (ROOT/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
