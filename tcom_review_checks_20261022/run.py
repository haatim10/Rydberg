"""Frozen-receiver review diagnostics. Never modifies earlier scientific files."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time,platform
from pathlib import Path
import numpy as np
import torch
from scipy.special import i0e
from sahure.data import Observations
from tcom_reliability_20261021 import study as old,inference as rec
from tcom_reliability_20261021 import ber as oldber
from tcom_hybrid_20261020 import inference as hyb
from tcom_incumbent_guard_20261019 import risk_data
from tcom_pilot_em_20261018 import continuation
from tcom_followups_20261017.ber_detector import detect,hypotheses
REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'results/tcom_review_checks_20261022'
SEEDS=tuple(old.SEEDS)
VARIANTS=('full','no_recovery','no_final_filter','neither')
ROBUST_ARMS=[('correct',0.,0.,0.)]+[(f'noise_{x:+g}db',x,0.,0.) for x in (-2.,-1.,1.,2.)]+[(f'reference_level_{x:+g}db',0.,x,0.) for x in (-1.,1.)]+[(f'reference_phase_{x:+g}deg',0.,0.,x) for x in (-10.,-5.,5.,10.)]
MARGINS=(1e-12,1e-10,1e-8,1e-6,1e-4,1e-2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp.json');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def save(p,**x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp.npz');np.savez_compressed(t,**x);t.replace(p)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def ob(a):return Observations(*(torch.from_numpy(a[k]) for k in ('z','s','b','sigma2')))
def frozen():
 spec=json.loads((OUT/'protocol.json').read_text())
 for name,h in spec['source_hashes'].items():assert sha(REPO/name)==h,name
 for name,h in spec['checkpoint_hashes'].items():assert sha(REPO/name)==h,name
 return spec
def freeze():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
 p=json.loads((old.OUT/'protocol.json').read_text());policy=json.loads((old.OUT/'policy.json').read_text())
 files={**p['source_hashes'],str(Path(__file__).relative_to(REPO)):sha(__file__)}
 for n in ('protocol.json','policy.json','completion.json','nmse_summary.json','ber/summary.json'):files[str((old.OUT/n).relative_to(REPO))]=sha(old.OUT/n)
 spec=dict(name='Review checks with unchanged final receiver',version=1,seeds=SEEDS,source_hashes=files,checkpoint_hashes=p['checkpoint_hashes'],rule=policy['candidate_rule'],
  ablations=dict(variants=VARIANTS,grid='Same153 final cells,128cases each,3fits; full/no_recovery predictions and counts reused after hash verification',
   definition='no_final_filter disables only final post-start filter for BOTH base and alternative; stage filters/weights/table/EM/gate/acceptance unchanged; gate recomputed; neither also disables recovery',
   evidence='Diagnostic intervention on frozen weights; no retraining, feature deletion, schedule tuning or final-test selection'),
  margins=dict(values=MARGINS,scope='All actual triggered/accepted final cases; compute mean fitting losses independently from stored predictions; float32 score arithmetic sensitivity also shown; not full float32 inference'),
  robustness=dict(seed=2026102201,data_seed=2026102202,refs=[5.,20.],snrs=[10.,32.5,45.],regimes=old.REGIMES,count_per_reference_path=32,
   arms=ROBUST_ARMS,methods=[f'{family}_{s}' for family in ('receiver','prior') for s in SEEDS]+['compact_sparse','em100'],
   data='Fresh independent reference/path blocks; each physical anchor shared across SNR/method/offset. Correct physical observations never altered by supplied calibration error.',
   detector='Primary: all methods use same wrong supplied reference/variance,64jointQPSKvectors independent pilot noise. Extra reference-phase-adjusted channel NMSE is scoring only, never inference.',
   scope='18settings,192physicalanchors,576conditioncases; diagnostic small-grid study, not full153grid validation. Offsets not tuned based on errors.'),
  timing=dict(windows=2,calls='27newphysicalanchors reused over original17SNRs;12fit/variant arms plus compactSparse andEM100; isolated after numerical jobs',target_ms=10.74,interpretation='Diagnostic batch-one CPU p90; no retries to pass'),
  statistics=dict(bootstrap=10000,seed=2026102203,paired='Resample within independent reference/path blocks; share indices acrossSNR/method/fits/offsets',scope='All review-study intervals descriptive,conditional on storedfits; no tuning/retraining'),
  environment=dict(python=platform.python_version(),torch=torch.__version__,one_thread=True),created_unix=time.time())
 write(OUT/'protocol.json',spec);frozen();print('Review protocol frozen',sha(OUT/'protocol.json'),flush=True)

@torch.no_grad()
def base(obs,seed,engine,final_filter=True):
 plans=hyb.plans(obs,rec.BASE_POLICY);groups={}
 for i,p in enumerate(plans):groups.setdefault((p['family'],p['steps'],p['ordinary'],p['reference_key'],p['snr_key']),[]).append(i)
 out=torch.empty((len(obs),32,3),dtype=torch.complex128)
 for (family,steps,ordinary,_,_),ii in groups.items():
  ids=torch.tensor(ii);part=obs if len(ii)==len(obs) else obs.batch(ids)
  g=engine['starts'].start(family,part,seed=seed)
  if not ordinary:
   if steps:g=continuation.estimate(part,g,steps)
   if final_filter:g=rec.finish(part,g)
  out[ids]=g
 return out
@torch.no_grad()
def without_filter(obs,seed,engine,rule):
 g0=base(obs,seed,engine,False);plan=hyb.plans(obs,rec.BASE_POLICY)
 eligible=np.array([p['reference_key']=='5' and p['snr_db']>30+1e-10 for p in plan]);score=np.full(len(obs),np.nan)
 ids=np.flatnonzero(eligible);trigger=np.zeros(len(obs),bool);accept=trigger.copy();g=g0.clone()
 if len(ids):
  part=obs.batch(torch.from_numpy(ids));res=part.z.numpy()-np.abs(g0[ids].numpy()@part.s.numpy()+part.b.numpy())
  score[ids]=(2*np.mean(res**2,axis=-1)/part.sigma2.numpy()[:,None]).max(-1);trigger=eligible&(score>rule['threshold']);ids=np.flatnonzero(trigger)
  if len(ids):
   part=obs.batch(torch.from_numpy(ids));initial,valid,_=rec.algebraic_start(part)
   if valid.any():
    local=np.flatnonzero(valid);sub=part.batch(torch.from_numpy(local));alt=continuation.estimate(sub,initial[local],rule['tail'])
    oldloss=rec.fitting_nll(sub,g0[ids[local]]);newloss=rec.fitting_nll(sub,alt);ok=np.isfinite(newloss)&(newloss<oldloss-1e-12)
    chosen=ids[local[ok]];g[chosen]=alt[ok];accept[chosen]=True
 return g,g0,dict(trigger=trigger,accepted=accept,score=score)
def err(g,truth):return np.sum(np.abs(np.asarray(g)-truth)**2,axis=(-2,-1))
def ber_counts(channels,data,reference,variance):
 bits=hypotheses(3)[1];result=np.zeros(channels.shape[:2],np.int64);cache={}
 for mi,gg in enumerate(channels):
  key=hashlib.sha256(gg.tobytes()).hexdigest()
  if key in cache:result[mi]=result[cache[key]];continue
  cache[key]=mi
  for j in range(len(gg)):result[mi,j]=np.count_nonzero(detect(data[j],gg[j],reference[j],float(variance[j]))!=bits)
 return result

def margin():
 frozen();rows=[];triggered=0;changed=np.zeros(len(MARGINS),int);scaled_changes=0;float32_changes=0;nonfinite32=0;max_arithmetic=0.
 for p in sorted((old.OUT/'test/cells').glob('*.npz')):
  a=load(p);obs=ob(a);names=list(a['method_names']);z,s,b,v=[x.numpy() for x in obs.args()]
  for si,seed in enumerate(SEEDS):
   mask=a['triggers'][si];accepted=a['accepted'][si]
   if not mask.any():continue
   assert np.array_equal(mask,accepted),'Rejected triggered alternative needs a dedicated reconstruction'
   g0=a['estimates'][names.index(f'prior_{seed}')];g1=a['estimates'][names.index(f'receiver_{seed}')]
   l0=rec.fitting_nll(obs,torch.from_numpy(g0));l1=rec.fitting_nll(obs,torch.from_numpy(g1));gain=l0-l1
   decisions=np.stack([l1<l0-m for m in MARGINS]);changed+=np.count_nonzero(decisions[:,mask]!=accepted[mask],axis=1)
   scale=32*np.finfo(float).eps*np.maximum(1,np.maximum(abs(l0),abs(l1)));scaled_changes+=int(np.count_nonzero((l1[mask]<l0[mask]-scale[mask])!=accepted[mask]))
   def f32(gg):
    zz=z.astype(np.float32);vv=v.astype(np.float32);amp=np.abs(gg.astype(np.complex64)@s.astype(np.complex64)+b.astype(np.complex64))
    return (((zz-amp)**2/vv[:,None,None])-np.log(i0e(2*zz*amp/vv[:,None,None]))).mean((-2,-1))
   with np.errstate(all='ignore'):f0,f1=f32(g0),f32(g1)
   finite=np.isfinite(f0)&np.isfinite(f1);nonfinite32+=int(np.count_nonzero(mask&~finite));float32_changes+=int(np.count_nonzero((f1[mask&finite]<f0[mask&finite]-1e-12)!=accepted[mask&finite]))
   if np.any(mask&finite):max_arithmetic=max(max_arithmetic,float(np.max(abs((f0-f1)[mask&finite]-gain[mask&finite]))))
   triggered+=int(mask.sum())
   for j in np.flatnonzero(mask):rows.append(dict(cell=p.stem,seed=seed,index=int(j),old=float(l0[j]),alternative=float(l1[j]),gain=float(gain[j]),float32_old=float(f0[j]) if finite[j] else None,float32_alternative=float(f1[j]) if finite[j] else None))
 gains=np.array([r['gain'] for r in rows]);result=dict(passed=True,cases=triggered,minimum_gain=float(gains.min()),gain_quantiles=dict(zip(('min','p1','p5','median','p95','max'),np.quantile(gains,[0,.01,.05,.5,.95,1]).tolist())),changed_decisions_by_absolute_margin={str(m):int(n) for m,n in zip(MARGINS,changed)},scale_aware_32eps_changed=scaled_changes,float32_score_changed=float32_changes,float32_nonfinite=nonfinite32,max_float32_loss_difference_error=max_arithmetic,rows=rows,scope='Actual triggered cases; float32 evaluates losses only, not end-to-end inference')
 write(OUT/'acceptance_margin.json',result);frozen();print('Acceptance audit', {k:v for k,v in result.items() if k!='rows'},flush=True)

def ablate():
 spec=frozen();engine=old.engine();count=0
 for ref in old.RSRS:
  for regime,_ in old.REGIMES:
   for snr in old.SNRS:
    name=old.cellname(regime,ref,snr);p=OUT/'ablation/cells'/f'{name}.npz'
    if p.exists():continue
    src=old.OUT/'test/cells'/f'{name}.npz';a=load(src);obs=ob(a);names=list(a['method_names']);pred=[];checks=[];triggers=[]
    for si,seed in enumerate(SEEDS):
     full=a['estimates'][names.index(f'receiver_{seed}')];prior=a['estimates'][names.index(f'prior_{seed}')]
     # Independent adapter reproduces original base with filtering enabled.
     checked=base(obs,seed,engine,True).numpy();diff=float(abs(checked-prior).max());assert diff<1e-10,(name,seed,diff);checks.append(diff)
     nofilter,neither,diag=without_filter(obs,seed,engine,spec['rule']);pred.append(np.stack((full,prior,nofilter.numpy(),neither.numpy())));triggers.append(diag['accepted'])
    pred=np.stack(pred);energy=a['truth_energy'];errors=err(pred,a['truth_g'][None,None]);data=load(old.OUT/'ber/data'/f'{name}.npz')['z']
    counts=ber_counts(pred.reshape(12,len(obs),32,3),data,a['b'][...,:1],a['sigma2']).reshape(3,4,len(obs))
    saved=load(old.OUT/'ber/cells'/f'{name}.npz');bnames=list(saved['method_names'])
    for si,seed in enumerate(SEEDS):
     for vi,label in [(0,'receiver'),(1,'prior')]:np.testing.assert_array_equal(counts[si,vi],saved['errors_per_user'][bnames.index(f'{label}_{seed}')].sum(-1))
    save(p,errors=errors,bit_errors=counts,truth_energy=energy,base_parity=np.array(checks),nofilter_accepted=np.stack(triggers),snr=np.array(snr),rsr=np.array(ref),regime=np.array(regime),source_sha256=np.array(sha(src)))
    count+=1;print('Ablation',count,'/153',name,flush=True)
 frozen();write(OUT/'ablation/completed.json',dict(passed=True,cells=len(list((OUT/'ablation/cells').glob('*.npz'))),variants=VARIANTS))

def robust():
 spec=frozen();cfg=spec['robustness'];engine=old.engine();selection=json.loads((old.OUT/'policy.json').read_text());count=0
 for ref in cfg['refs']:
  for ri,(regime,L) in enumerate(old.REGIMES):
   ap=OUT/'robustness/anchors'/f'{regime}_rsr{ref:g}.npz'
   if ap.exists():anchor=load(ap)
   else:
    anchor=old.make_anchor(cfg['seed'],range(cfg['count_per_reference_path']),L,ref,truth=True,snrs=cfg['snrs']);save(ap,**anchor)
   for snr in cfg['snrs']:
    truth=anchor['g'];correct=old.observed(anchor,snr);variance=correct.sigma2.numpy();reference=correct.b.numpy()[...,:1];xs,bits=hypotheses(3)
    noise=[]
    for index in range(len(correct)):
     rng=np.random.default_rng(np.random.SeedSequence([cfg['data_seed'],ri,int(ref),index]));raw=rng.standard_normal((32,64,2));noise.append((raw[...,0]+1j*raw[...,1])/np.sqrt(2))
    data=np.abs(truth@xs.T+reference+np.sqrt(variance)[:,None,None]*np.stack(noise))
    for arm,offset,level,phase in ROBUST_ARMS:
     name=f'{regime}_rsr{ref:g}_snr{snr:g}_{arm}';p=OUT/'robustness/cells'/f'{name}.npz'
     if p.exists():continue
     obs=Observations(correct.z,correct.s,correct.b*(10**(level/20)*np.exp(1j*np.deg2rad(phase))),correct.sigma2*10**(offset/10))
     outputs=[];diagnostic=[]
     for seed in SEEDS:
      prior=hyb.hybrid(obs,seed,rec.BASE_POLICY,engine);full,d=rec.from_base(obs,prior,spec['rule'],diagnostics=True);outputs.extend((full.numpy(),prior.numpy()));diagnostic.append(d['accepted'])
     for method in ('compact_sparse','em100'):outputs.append(old.estimate_method(obs,method,selection,engine).numpy())
     order=[n for seed in SEEDS for n in (f'receiver_{seed}',f'prior_{seed}')]+['compact_sparse','em100'];pred=np.stack(outputs)
     counts=ber_counts(pred,data,obs.b.numpy()[...,:1],obs.sigma2.numpy())
     errors=err(pred,truth[None]);adjusted=err(pred*np.exp(-1j*np.deg2rad(phase)),truth[None])
     save(p,errors=errors,phase_adjusted_errors=adjusted,truth_energy=np.sum(abs(truth)**2,axis=(-2,-1)),bit_errors=counts,method_names=np.array(order),accepted=np.stack(diagnostic),arm=np.array(arm),snr=np.array(snr),rsr=np.array(ref),regime=np.array(regime),reference_phase_deg=np.array(phase),noise_offset_db=np.array(offset),reference_level_offset_db=np.array(level))
     count+=1;print('Robustness',count,'/198',name,flush=True)
 frozen();write(OUT/'robustness/completed.json',dict(passed=True,cells=len(list((OUT/'robustness/cells').glob('*.npz'))),settings=18,arms=11,physical_anchors=192,condition_cases_per_arm=576))

def ur_preflight():
 frozen()
 from trackD_urformer.config import ModelConfig,NumericConfig
 from trackD_urformer.urformer import URformer,count_parameters
 from trackD_urformer.train import nmse_loss
 from trackD_urformer.torch_forward import least_squares_G,forward_field
 from tcom_runtime_bypass_20261017.sparse_fast import spectral_initial
 a=load(old.OUT/'test/cells/random3to7_rsr12_snr10.npz');obs=ob(a).batch(slice(0,32));z,s,b,v=[x.to(torch.complex64 if x.is_complex() else torch.float32) for x in obs.args()];truth=torch.from_numpy(a['truth_g'][:32]).to(torch.complex64)
 torch.manual_seed(2026102204);cfg=ModelConfig(use_hankel=False,use_transformer=True);numeric=NumericConfig(dtype='float32');model=URformer(32,3,cfg,numeric);g0=torch.from_numpy(spectral_initial(obs.z.numpy(),obs.s.numpy(),obs.b.numpy())).to(torch.complex64)
 g=torch.from_numpy(a['truth_g'][:32]);rhs=g@obs.s
 np.testing.assert_allclose(least_squares_G(rhs,obs.s).numpy(),g.numpy(),atol=1e-11,rtol=1e-11)
 model._set_test_mode(filter_override='exact_bessel',alpha=1.,disable_residual=True)
 from tcom_runtime_bypass_20261017.sparse_fast import posterior_field
 gtest=model.layers[0](g,obs.z,obs.s,obs.b,obs.sigma2) if False else None
 # Float32 degeneration check, including the pilot orientation and complex signs.
 actual=model.layers[0](g0,z,s,b,v).detach().numpy();mu=g0.numpy()@s.numpy()+b.numpy();q=posterior_field(z.numpy(),mu,v.numpy());expected=(q-b.numpy())@s.numpy().conj().swapaxes(-1,-2)@np.linalg.inv(s.numpy()@s.numpy().conj().swapaxes(-1,-2))
 np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-5);model._clear_test_mode()
 opt=torch.optim.Adam(model.parameters(),lr=1e-3);durations=[];losses=[]
 for step in range(8):
  tick=time.perf_counter();opt.zero_grad();pred=model(g0,z,s,b,v);loss=nmse_loss(pred,truth);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();durations.append(time.perf_counter()-tick);losses.append(float(loss.detach()))
 result=dict(passed=True,parameters=count_parameters(model),batch_size=32,batch_step_seconds=durations,smoke_loss=losses,median_steady_step_seconds=float(np.median(durations[2:])),estimated_20epoch_16384sample_onefit_minutes=float(np.median(durations[2:])*512*20/60),estimated_50epoch_20000sample_onefit_minutes=float(np.median(durations[2:])*625*50/60),cpu_threads=torch.get_num_threads(),mps=torch.backends.mps.is_available(),trained_comparison=False,notes='Eight readiness updates on held-out test subset are a discarded smoke model; never a trained comparison or selected checkpoint. Real training must start fresh on disjoint data, with complete validation. No Hankel modules enabled.')
 write(OUT/'urformer_preflight.json',result);print('URformer preflight',{k:v for k,v in result.items() if k!='parameters'},flush=True)

if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','margin','ablate','robust','ur_preflight'));args=parser.parse_args();globals()[args.phase]()
