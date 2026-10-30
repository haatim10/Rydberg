"""Independent, read-only evidence checks for paired training and EM tails.

The audit writes only its own reports. It never chooses a checkpoint, tail,
sparse configuration, or test cell. Training and evaluation sources stay frozen.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import ast
import json
from pathlib import Path
import time
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import posterior_field,right_solve
from .training_protocol import CONFIG,sha,write_json
from .training_data import split_indices,paired_split,mask_evidence,tensor_digest,allocation,load_set
from .training import initialize_model,model_state_digest,validate
from . import continuation

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'results/tcom_pilot_em_20261018'
OUT=ROOT/'audit'


def read(path):
    return json.loads(Path(path).read_text())


def source_check(record):
    for name,digest in record['source_hashes'].items():
        assert sha(REPO/name)==digest, f'Frozen source differs: {name}'


def design():
    protocol=read(ROOT/'training_protocol.json');source_check(protocol)
    assert protocol['config']==CONFIG
    # Inspect the explicit simulation boundary independently of comments.
    data_source=REPO/'tcom_pilot_em_20261018/training_data.py'
    tree=ast.parse(data_source.read_text())
    simulate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='simulate_observation')
    returns=[n.value for n in ast.walk(simulate) if isinstance(n,ast.Return)]
    assert len(returns)==1 and isinstance(returns[0],ast.Tuple)
    assert [ast.unparse(x) for x in returns[0].elts]==['w.Z','w.S','w.B','w.sigma2']
    load=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_set')
    assert "('z', 's', 'b', 'sigma2')" in ast.unparse(load)
    rng=np.random.default_rng(2026101800)
    obs=Observations(torch.tensor(rng.uniform(.1,3,(4,32,39))),
        torch.tensor((rng.normal(size=(4,3,39))+1j*rng.normal(size=(4,3,39)))/np.sqrt(2)),
        torch.full((4,32,39),4+0j,dtype=torch.complex128),torch.full((4,),.3,dtype=torch.float64))
    columns=split_indices(len(obs),2026101810)
    a21,a30,held=paired_split(obs,2026101810)
    changed=obs.z.clone();changed.scatter_(-1,columns[2][:,None].expand(-1,32,-1),100.)
    changed21,changed30,changedheld=paired_split(Observations(changed,obs.s,obs.b,obs.sigma2),2026101810)
    assert all(torch.equal(a,b) for a,b in zip(a21.args(),changed21.args()))
    assert all(torch.equal(a,b) for a,b in zip(a30.args(),changed30.args()))
    assert not torch.equal(held.z,changedheld.z)
    model=initialize_model(9101).double().eval()
    with torch.no_grad():
        assert torch.equal(model(*a21.args()),model(*changed21.args()))
        assert torch.equal(model(*a30.args()),model(*changed30.args()))
    gathered=[]
    ids=torch.tensor([3,0,2])
    for col,full in zip(columns,(a21,a30,held)):
        subset=obs.batch(ids).columns(col[ids])
        assert all(torch.equal(a,b) for a,b in zip(subset.args(),full.batch(ids).args()))
        gathered.append(full.z.shape[-1])
    initial={}
    for seed in CONFIG['seeds']:
        a=initialize_model(seed);b=initialize_model(seed)
        assert model_state_digest(a)==model_state_digest(b)
        initial[str(seed)]=model_state_digest(a)
    # Arbitrary supplied estimates prove the continuation starts at the caller's G.
    fitting=a30
    g=torch.tensor(rng.normal(size=(4,32,3))+1j*rng.normal(size=(4,32,3)))
    before=g.clone();saved,info=continuation.trajectory(fitting,g,audit=True)
    assert torch.equal(g,before) and torch.equal(saved[0],before)
    reference=before.clone();op=right_solve(fitting.s.mH,fitting.s@fitting.s.mH)
    parity=0.
    for step in range(1,6):
        reference=(posterior_field(fitting.z,reference@fitting.s+fitting.b,fitting.sigma2)-fitting.b)@op
        if step in saved:
            parity=max(parity,float((reference-saved[step]).abs().max()))
            assert torch.allclose(reference,saved[step],rtol=1e-11,atol=1e-11)
    scalar=torch.cat([continuation.estimate(fitting.batch(slice(i,i+1)),before[i:i+1],5) for i in range(4)])
    assert torch.allclose(scalar,saved[5],rtol=1e-11,atol=1e-11)
    result=dict(passed=True,completed_unix=time.time(),training_protocol_sha256=sha(ROOT/'training_protocol.json'),
        source_hashes_match=True,simulation_returns_only=['Z','S','B','sigma2'],
        heldout_magnitude_perturbation_leaves_context_and_prediction_bitwise_unchanged=True,
        minibatch_gather_bitwise_equivalent=True,pilot_counts=gathered,
        paired_initial_state_sha256=initial,mask=mask_evidence(4,2026101810),
        continuation=dict(arbitrary_initial_channel=True,input_not_mutated=True,tail0_bitwise=True,
            maximum_canonical_error=parity,maximum_scalar_error=float((scalar-saved[5]).abs().max()),
            maximum_nll_increase=info['maximum_nll_increase']),
        interpretation='The fresh context21 arm controls the fitting-pilot count with common30-context validation selection. It is not an exact rerun of the earlier21-context checkpoint-selection protocol.',
        audit_source_sha256=sha(__file__))
    write_json(OUT/'design.json',result)
    return result


def training():
    statuses={seed:read(ROOT/f'training_workers/seed_{seed}/status.json') for seed in CONFIG['seeds']}
    assert all(s['state']=='complete' for s in statuses.values()),'All six training arms must finish before the training audit.'
    protocol=read(ROOT/'training_protocol.json');source_check(protocol)
    manifests={};banks={}
    for split in ('train','val'):
        path=ROOT/'training_data'/split;manifest=read(path/'manifest.json')
        expected={'z.npy','s.npy','b.npy','sigma2.npy','conditions.npy'}
        assert set(manifest['file_hashes'])==expected
        assert {p.name for p in path.glob('*.npy')}==expected
        for name,digest in manifest['file_hashes'].items():assert sha(path/name)==digest
        count=CONFIG[f'{split}_cases'];seed=CONFIG[f'{split}_data_seed']
        assert manifest['description']==dict(split=split,count=count,seed=seed,P=39)
        ids=np.load(path/'conditions.npy',allow_pickle=False)
        np.testing.assert_array_equal(ids,allocation(count,seed))
        shapes={'z':(count,32,39),'s':(count,3,39),'b':(count,32,39),'sigma2':(count,)}
        for name,shape in shapes.items():assert np.load(path/(name+'.npy'),mmap_mode='r').shape==shape
        assert len(set(ids))==126
        manifests[split]=sha(path/'manifest.json');banks[split]=dict(cases=count,seed=seed,condition_counts=manifest['condition_counts'])
    val,conditions=load_set('val')
    _,val30,held=paired_split(val,CONFIG['validation_mask_seed'])
    workers={};selected={}
    validation_atol=1e-6;validation_rtol=1e-6
    for seed in CONFIG['seeds']:
        status=statuses[seed]
        paired=[];initial=initialize_model(seed);initial_digest=model_state_digest(initial)
        initial_score,_=validate(initial,val30,held,conditions)
        for arm in ('context21','context30'):
            folder=ROOT/f'checkpoints/{arm}/seed_{seed}'
            best=torch.load(folder/'em_adaptive_best.pt',map_location='cpu',weights_only=False)
            latest=torch.load(folder/'em_adaptive_latest.pt',map_location='cpu',weights_only=False)
            for state in (best,latest):
                assert state['seed']==seed and state['arm']==arm and state['selection_context']==30
                assert state['initial_state_sha256']==initial_digest
                assert state['protocol_sha256']==sha(ROOT/'training_protocol.json')
                assert state['data_manifest_hashes']==manifests
                assert state['validation_mask_evidence']==mask_evidence(CONFIG['val_cases'],CONFIG['validation_mask_seed'])
            assert latest['epoch']==CONFIG['epochs']==20
            assert latest['optimizer_updates']==20*CONFIG['train_cases']//CONFIG['batch_size']
            history=latest['history'];assert len(history)==20
            for epoch,h in enumerate(history):
                assert h['epoch']==epoch+1
                order=torch.randperm(CONFIG['train_cases'],generator=torch.Generator().manual_seed(seed+20000+epoch))
                evidence=dict(mask=mask_evidence(CONFIG['train_cases'],seed+10000+epoch),order_sha256=tensor_digest(order),epoch=epoch+1)
                assert h['evidence']==evidence
            # Epoch0 is an allowed checkpoint. If it won, compare its full state to initialization.
            score=min([initial_score]+[h['val30_nll'] for h in history])
            if best['epoch']==0:
                fresh=initialize_model(seed)
                assert all(torch.equal(best['model'][k],v) for k,v in fresh.state_dict().items())
                assert best['best_score']==score==initial_score
            else:
                assert best['best_score']==score==best['val30']
                assert best['epoch']==next(h['epoch'] for h in history if h['val30_nll']==score)
            assert latest['best_score']==best['best_score']
            # Independently reload the selected weights, then recompute the
            # actual30-context heldout loss using training precision/batching.
            reloaded=initialize_model(seed)
            reloaded.load_state_dict(best['model'],strict=True)
            recomputed,strata=validate(reloaded,val30,held,conditions)
            absolute_error=abs(recomputed-best['best_score'])
            tolerance=validation_atol+validation_rtol*abs(best['best_score'])
            assert absolute_error<=tolerance,(arm,seed,recomputed,best['best_score'])
            np.testing.assert_allclose(strata,best['validation_strata30'],
                atol=validation_atol,rtol=validation_rtol)
            for kind in ('best','latest'):
                assert sha(folder/f'em_adaptive_{kind}.pt')==status['outputs'][arm][kind+'_sha256']
            paired.append([h['evidence'] for h in history])
            selected[f'{arm}_{seed}']=dict(epoch=best['epoch'],val30_nll=best['best_score'],
                recomputed_val30_nll=recomputed,recomputed_absolute_error=absolute_error,
                validation_comparison_tolerance=tolerance,
                checkpoint_sha256=sha(folder/'em_adaptive_best.pt'),updates=latest['optimizer_updates'],
                training_seconds=latest['train_seconds'])
        assert paired[0]==paired[1]
        workers[str(seed)]=dict(initial_state_sha256=initial_digest,initial_val30_nll=initial_score,
            epoch_masks_orders_equal=True,status_sha256=sha(ROOT/f'training_workers/seed_{seed}/status.json'))
    result=dict(passed=True,completed_unix=time.time(),training_protocol_sha256=sha(ROOT/'training_protocol.json'),
        banks=banks,manifest_hashes=manifests,workers=workers,selected=selected,
        source_and_data_hashes_match=True,no_channel_truth_persisted=True,paired_updates=10240,
        checkpoint_minimum30_context_nll_verified=True,
        selected_weights_validation_nll_recomputed=True,
        validation_comparison=dict(atol=validation_atol,rtol=validation_rtol,
            precision='float32/complex64, same32-case validation batches, equal126-stratum means'),
        audit_source_sha256=sha(__file__))
    write_json(OUT/'training.json',result);return result


def study():
    """Verify all frozen choices and every scored test/guard array without refitting."""
    from . import study as runner
    out=ROOT/'evaluation';p=read(out/'protocol.json');source_check(p)
    for name,digest in p['original_inputs']['hashes'].items():assert sha(REPO/name)==digest
    assert sha(ROOT/'training_protocol.json')==p['training_protocol_sha256']
    models=read(out/'checkpoints.json');tails=read(out/'tails.json');cal=read(out/'calibration.json');selection=read(out/'selection.json')
    for name,item in models['checkpoints'].items():assert sha(REPO/item['path'])==item['sha256']
    assert tails['checkpoint_sha256']==sha(out/'checkpoints.json')
    assert cal['tails_sha256']==sha(out/'tails.json')
    for probe in cal['probes'].values():
        assert probe['p90_ms']==float(np.quantile([x['ms'] for x in probe['samples']],.9))
    for family in runner.FAMILIES:
        times=[x['ms'] for seed in runner.SEEDS for x in cal['probes'][f'{family}_{seed}']['samples']]
        assert cal['caps_ms'][family]==float(np.quantile(times,.9))
        expected=[k for k,item in cal['families'].items() if item['p90_ms']<=cal['caps_ms'][family]]
        assert cal['eligibility'][family]==expected
    for key,item in cal['families'].items():
        factors=(0.,) if item['config']['inner_steps']==0 else (.03,3.)
        assert item['p90_ms']==max(cal['probes'][key+f'_f{factor:g}']['p90_ms'] for factor in factors)
    for file,key in [('checkpoints.json','checkpoint_sha256'),('tails.json','tails_sha256'),('calibration.json','calibration_sha256')]:assert selection[key]==sha(out/file)
    choices={}
    for snr in runner.SNRS:
        key=f'{snr:g}';path=out/'development'/f'tails_snr{key}.npz'
        assert sha(path)==tails['loss_hashes'][path.name]
        a=runner.load(path);names=list(map(str,a['names']))
        assert a['losses'].shape==(36,72)
        for family in runner.FAMILIES:
            table=[(float(a['losses'][[names.index(f'{family}_{seed}_tail{t}') for seed in runner.SEEDS]].mean()),t) for t in runner.TAILS]
            loss,chosen=min(table)
            assert tails['choices'][family][key]['selected_tail']==chosen
            choices[f'{family}_snr{key}']=chosen
        sparsefile=out/'development'/f'sparse_snr{key}.json'
        assert sha(sparsefile)==selection['development_hashes'][sparsefile.name]
        sparse=read(sparsefile)
        for family in runner.FAMILIES:
            eligible=[x for x in sparse['scores'] if x['family_key'] in cal['eligibility'][family]]
            for x in eligible:
                assert np.isclose(x['mean_nll'],np.mean(x['per_world_nll']),rtol=0,atol=1e-12)
                assert cal['families'][x['family_key']]['p90_ms']<=cal['caps_ms'][family]
            best=min(eligible,key=lambda x:(x['mean_nll'],x['p90_ms'],x['config']['iterations'],x['config']['inner_steps'],x['config']['dictionary_size'],x['config']['factor']))
            assert best==sparse['selected'][family]==selection['sparse_choices'][key][family]
    assert p['frozen_unix']<=models['frozen_unix']<=tails['frozen_unix']<=cal['completed_unix']<=selection['frozen_unix']
    files=list(sorted((out/'test').glob('*.npz')))+list(sorted((out/'guards').glob('*.npz')))
    assert len(files)==17
    maxerr=0.;test_hashes={};tails_exact=True
    for path in files:
        assert path.stat().st_mtime>=selection['frozen_unix']
        a=runner.load(path);n=64 if bool(a['guard']) else 128
        assert int(a['seed'])==runner.TEST_SEED and str(a['selection_sha256'])==sha(out/'selection.json')
        assert a['z'].shape==(n,32,30) and a['truth_g'].shape==(n,32,3)
        names=list(map(str,a['method_names']));estimate=a['estimates'];g=a['truth_g']
        error=np.sum(np.abs(estimate-g[None])**2,axis=(-2,-1));energy=np.sum(np.abs(g)**2,axis=(-2,-1))
        np.testing.assert_array_equal(error,a['errors']);np.testing.assert_array_equal(energy,a['truth_energy'])
        np.testing.assert_array_equal(error/energy[None],a['nmse'])
        snrkey=f'{min(runner.SNRS,key=lambda s:abs(s-float(a["snr"]))):g}'
        for family in runner.FAMILIES:
            t=tails['choices'][family][snrkey]['selected_tail']
            for seed in runner.SEEDS:
                assert np.array_equal(estimate[names.index(f'{family}_{seed}_selected')],estimate[names.index(f'{family}_{seed}_tail{t}')])
        # Recreate one independent anchor from the predeclared generator per cell.
        local=1000 if bool(a['guard']) else 0
        L={'L2':2,'L4':4,'random3to7':None}[str(a['regime'])]
        observation,truth,paths=runner.generated(runner.TEST_SEED,[local],float(a['snr']),float(a['rsr']),L,truth=True)
        observation=runner.first30(observation)
        np.testing.assert_array_equal(truth[0],g[0]);np.testing.assert_array_equal(paths[0],a['paths'][0])
        for key,value in zip(('z','s','b','sigma2'),observation.args()):np.testing.assert_array_equal(value.numpy()[0],a[key][0])
        maxerr=max(maxerr,float(np.max(np.abs(error-a['errors']))));test_hashes[str(path.relative_to(ROOT))]=sha(path)
    result=dict(passed=True,completed_unix=time.time(),source_hashes_match=True,
        checkpoint_hashes_match=True,tail_minima_verified=choices,sparse_eligible_minima_verified=15,
        latency_caps_and_eligibility_recomputed=True,
        freeze_order_verified=True,test_cells_after_final_freeze=17,
        earliest_test_seconds_after_freeze=min(p.stat().st_mtime for p in files)-selection['frozen_unix'],
        recomputed_errors_exact=True,selected_tail_arrays_exact=True,regenerated_anchor_cells=17,
        test_hashes=test_hashes,selection_sha256=sha(out/'selection.json'),audit_source_sha256=sha(__file__))
    write_json(OUT/'study.json',result);return result


def ber():
    """Recreate data and spot-check decoding independently of aggregate reports."""
    from . import ber as runner
    from tcom_followups_20261017.ber_detector import draw_data,detect,verify_detector
    out=ROOT/'ber';protocol=read(out/'protocol.json');source_check(protocol)
    summary=read(out/'summary.json');selection_hash=sha(ROOT/'evaluation/selection.json')
    assert summary['protocol_sha256']==sha(out/'protocol.json')
    assert summary['selection_sha256']==selection_hash
    total={n:0 for n in runner.NAMES};decoded=0;duplicates=0;data_hashes={}
    for cell_index,(regime,snr) in enumerate(runner.cells()):
        label=f'{regime}_snr{snr}';a=runner.load(out/'cells'/(label+'.npz'))
        source=runner.load(ROOT/'evaluation/test'/(label+'.npz'))
        assert str(a['source_sha256'])==sha(ROOT/'evaluation/test'/(label+'.npz'))
        assert str(a['selection_sha256'])==selection_hash
        assert str(a['data_sha256'])==sha(out/'data'/(label+'.npz'))
        assert str(a['truth_sha256'])==sha(out/'truth'/(label+'.npz'))
        data=runner.load(out/'data'/(label+'.npz'));truth=runner.load(out/'truth'/(label+'.npz'))
        names=list(map(str,a['method_names']));source_names=list(map(str,source['method_names']))
        np.testing.assert_array_equal(a['bits_per_user'],np.full((128,3),512))
        assert np.all(a['errors_per_user']>=0) and np.all(a['errors_per_user']<=a['bits_per_user'])
        for case in (0,127):
            rng=np.random.default_rng(np.random.SeedSequence([2026101805,cell_index,case]))
            z,bits,indices=draw_data(source['truth_g'][case],source['b'][case,:,:1],float(source['sigma2'][case]),rng,256)
            np.testing.assert_array_equal(z,data['z'][case]);np.testing.assert_array_equal(bits,truth['bits'][case])
            np.testing.assert_array_equal(indices,truth['symbol_indices'][case])
            for name in ('perfect_csi','old_9101_tail0','context30_9102_selected','sparse_fresh_context30'):
                g=source['truth_g'][case] if name=='perfect_csi' else source['estimates'][source_names.index(name),case]
                prediction=detect(data['z'][case],g,data['b'][case],float(data['sigma2'][case]))
                errors=np.count_nonzero(prediction!=truth['bits'][case],axis=(0,2))
                np.testing.assert_array_equal(errors,a['errors_per_user'][names.index(name),case]);decoded+=1
        for mi,name in enumerate(names):
            total[name]+=int(a['errors_per_user'][mi].sum())
            other=str(a['duplicate_of'][mi])
            if other:
                assert name!='perfect_csi'
                g=source['estimates'][source_names.index(name)]
                other_g=source['truth_g'] if other=='perfect_csi' else source['estimates'][source_names.index(other)]
                np.testing.assert_array_equal(g,other_g)
                np.testing.assert_array_equal(a['errors_per_user'][mi],a['errors_per_user'][names.index(other)])
                duplicates+=1
        data_hashes[label]=str(a['data_sha256'])
    bits=15*128*256*3*2
    for name,count in total.items():
        row=summary['overall']['methods'][name]
        assert row['bit_errors']==count and row['bits_per_fit']==bits
        assert np.isclose(row['ber'],count/bits,rtol=0,atol=1e-16)
    for group,members in runner.GROUPS.items():
        assert np.isclose(summary['overall']['methods'][group]['ber'],np.mean([total[m]/bits for m in members]),rtol=0,atol=1e-16)
    result=dict(passed=True,completed_unix=time.time(),detector_checks=verify_detector(),
        source_protocol_hashes_match=True,regenerated_data_cases=30,independently_redecoded_cases_methods=decoded,
        verified_exact_prediction_duplicates=duplicates,total_errors=total,overall_aggregation_exact=True,
        data_hashes=data_hashes,protocol_sha256=sha(out/'protocol.json'),selection_sha256=selection_hash,
        audit_source_sha256=sha(__file__))
    write_json(OUT/'ber.json',result);return result


def fixed_tails():
    from . import ber_fixed_tails as runner
    from . import ber as core
    from tcom_followups_20261017.ber_detector import detect
    out=runner.OUT;protocol=read(out/'protocol.json');source_check(protocol)
    assert protocol['core_protocol_sha256']==sha(core.OUT/'protocol.json')
    summary=read(out/'summary.json');assert summary['protocol_sha256']==sha(out/'protocol.json')
    assert summary['selection_sha256']==sha(ROOT/'evaluation/selection.json')
    total={n:0 for n in runner.NAMES};decodes=0;duplicates=0
    for regime,snr in core.cells():
        label=f'{regime}_snr{snr}';a=core.load(out/'cells'/(label+'.npz'))
        sourcepath=core.SOURCE/'test'/(label+'.npz');source=core.load(sourcepath)
        assert protocol['frozen_unix']<=sourcepath.stat().st_mtime
        base=core.load(core.OUT/'cells'/(label+'.npz'))
        assert str(a['core_cell_sha256'])==sha(core.OUT/'cells'/(label+'.npz'))
        assert str(a['source_sha256'])==sha(sourcepath)
        for kind in ('data','truth'):
            assert str(a[kind+'_sha256'])==str(base[kind+'_sha256'])==sha(core.OUT/kind/(label+'.npz'))
        data=core.load(core.OUT/'data'/(label+'.npz'));truth=core.load(core.OUT/'truth'/(label+'.npz'))
        source_names=list(map(str,source['method_names']));names=list(map(str,a['method_names']));base_names=list(map(str,base['method_names']))
        assert tuple(names)==runner.NAMES
        np.testing.assert_array_equal(a['bits_per_user'],base['bits_per_user'])
        for mi,name in enumerate(names):
            total[name]+=int(a['errors_per_user'][mi].sum())
            other=str(a['duplicate_of'][mi])
            if other:
                origin,other_name=other.split('/',1)
                other_g=source['truth_g'] if other_name=='perfect_csi' else source['estimates'][source_names.index(other_name)]
                np.testing.assert_array_equal(source['estimates'][source_names.index(name)],other_g)
                other_errors=base['errors_per_user'][base_names.index(other_name)] if origin=='core' else a['errors_per_user'][names.index(other_name)]
                np.testing.assert_array_equal(a['errors_per_user'][mi],other_errors);duplicates+=1
        for case in (0,127):
            for name in ('old_9101_tail1','old_9103_tail5','context30_9102_tail2','context30_9103_tail5'):
                g=source['estimates'][source_names.index(name),case]
                prediction=detect(data['z'][case],g,data['b'][case],float(data['sigma2'][case]))
                error=np.count_nonzero(prediction!=truth['bits'][case],axis=(0,2))
                np.testing.assert_array_equal(error,a['errors_per_user'][names.index(name),case]);decodes+=1
    bits=15*128*256*3*2
    for name,count in total.items():
        row=summary['overall']['methods'][name]
        assert row['bit_errors']==count and row['bits_per_fit']==bits
        assert np.isclose(row['ber'],count/bits,rtol=0,atol=1e-16)
    for name,members in runner.GROUPS.items():
        assert np.isclose(summary['overall']['methods'][name]['ber'],np.mean([total[m]/bits for m in members]),rtol=0,atol=1e-16)
    result=dict(passed=True,completed_unix=time.time(),protocol_frozen_before_all_test_cells=True,
        core_data_and_truth_hashes_identical=True,source_hashes_match=True,
        independently_redecoded_cases_methods=decodes,exact_prediction_duplicates_verified=duplicates,
        total_errors=total,aggregation_verified=True,protocol_sha256=sha(out/'protocol.json'),
        audit_source_sha256=sha(__file__))
    write_json(OUT/'fixed_tails.json',result);return result


def timing():
    """Check final latency arithmetic only; never invoke an estimator."""
    from itertools import product
    out=ROOT/'evaluation';record=read(out/'final_timing.json')
    protocol=read(out/'protocol.json');selection=read(out/'selection.json')
    calibration=read(out/'calibration.json');summary=read(out/'nmse_summary.json')
    selection_hash=sha(out/'selection.json');timing_hash=sha(out/'final_timing.json')
    assert record['selection_sha256']==selection_hash==summary['selection_sha256']
    assert summary['timing_sha256']==timing_hash
    assert summary['timing']==record['methods']
    assert summary['cap_verification']==record['cap_verification']
    assert record['completed_unix']>=selection['frozen_unix']
    families=('old','context21','context30');seeds=(9101,9102,9103)
    kinds=('tail0','tail1','tail2','tail5','selected')
    raw_names={f'{f}_{s}_{kind}' for f,s,kind in product(families,seeds,kinds)}
    raw_names|={'prior_sparse_matched','prior_sparse_10ms','prior_sparse_20ms',
        'prior_sparse_50ms','prior_sparse_full','em100','optshrink'}
    raw_names|={f'sparse_fresh_{f}' for f in families}
    groups={f'{f}_{kind}_mean':[f'{f}_{s}_{kind}' for s in seeds] for f,kind in product(families,kinds)}
    assert set(record['raw_samples'])==raw_names
    assert set(record['methods'])==raw_names|set(groups)
    snrs=protocol['final_timing']['snrs'];indices=protocol['final_timing']['local_indices']
    regimes=[r[0] for r in protocol['regimes']];repeats=protocol['final_timing']['repeats']
    assert record['worlds']==len(snrs)*len(regimes)*len(indices)==30
    assert record['repeats']==repeats==3
    expected=set(product(snrs,regimes,indices,range(repeats)))
    checked=0;maximum_discrepancy=0.;outliers=[]
    def statistics(values):
        return dict(p90_ms=float(np.quantile(values,.9)),median_ms=float(np.median(values)),mean_ms=float(np.mean(values)))
    def check_statistics(saved,values):
        nonlocal checked,maximum_discrepancy
        calculated=statistics(values)
        for key,value in calculated.items():
            discrepancy=abs(saved[key]-value);maximum_discrepancy=max(maximum_discrepancy,discrepancy)
            assert discrepancy<=1e-12,(key,saved[key],value)
            checked+=1
    rows=dict(record['raw_samples'])
    for name,samples in rows.items():
        keys=[(s['snr'],s['regime'],s['index'],s['repeat']) for s in samples]
        assert len(keys)==len(expected)==90 and set(keys)==expected
        times=np.asarray([s['ms'] for s in samples])
        assert np.isfinite(times).all() and np.all(times>0)
        median=np.median(times)
        for s in samples:
            if s['ms']>10*median:outliers.append(dict(method=name,median_ms=float(median),**s))
    # Pool single-fit calls. There is no sum of three models for an ensemble.
    for group,parts in groups.items():
        rows[group]=[sample for part in parts for sample in record['raw_samples'][part]]
        assert len(rows[group])==270
    for name,samples in rows.items():
        saved=record['methods'][name]
        assert set(saved['per_snr'])=={f'{s:g}' for s in snrs}
        check_statistics(saved,[s['ms'] for s in samples])
        for snr in snrs:
            chosen=[s['ms'] for s in samples if s['snr']==snr]
            assert len(chosen)==(54 if name in groups else 18)
            check_statistics(saved['per_snr'][f'{snr:g}'],chosen)
    caps={}
    for family in families:
        pipeline=record['methods'][family+'_selected_mean']['p90_ms']
        sparse=record['methods']['sparse_fresh_'+family]['p90_ms']
        cap=calibration['caps_ms'][family]
        expected_cap=dict(cap_ms=cap,selected_pipeline_p90_ms=pipeline,sparse_p90_ms=sparse,
            sparse_exceeds_calibrated_cap=sparse>cap,sparse_exceeds_fresh_pipeline=sparse>pipeline)
        assert record['cap_verification'][family]==expected_cap
        caps[family]=dict(**expected_cap,sparse_to_pipeline_ratio=sparse/pipeline)
    result=dict(passed=True,completed_unix=time.time(),timing_sha256=timing_hash,
        selection_sha256=selection_hash,raw_methods=55,raw_samples=4950,pooled_groups=15,
        summaries_checked=70,scalar_statistics_checked=checked,maximum_absolute_discrepancy_ms=maximum_discrepancy,
        arithmetic_absolute_tolerance_ms=1e-12,all_samples_finite_and_positive=True,
        every_method_has_complete_balanced90_sample_grid=True,
        group_definition='270 separate single-fit calls pooled across3fixedfits; no ensemble sum',
        nmse_summary_timing_copy_and_hash_match=True,cap_verification=caps,
        raw_samples_above10times_method_median=outliers,
        interpretation='Calibrated caps and fresh observed latency are reported separately. Runtime samples, including outliers, were retained without modification.',
        benchmark_rerun=False,previous_audits_rerun=False,audit_source_sha256=sha(__file__))
    write_json(OUT/'timing.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('design','training','study','ber','fixed_tails','timing','all'));args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    phases=('design','training','study','ber','fixed_tails','timing') if args.phase=='all' else (args.phase,)
    for phase in phases:print(json.dumps(globals()[phase]()),flush=True)


if __name__=='__main__':main()
