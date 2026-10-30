"""Two unrepeated, isolated, complete batch-one timing windows."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json,time,argparse,platform
import numpy as np,torch
from .run import OUT,REPO,SEEDS,old,rec,hyb,base,without_filter,frozen,load,write,sha
from .urformer import U,ModelConfig,NumericConfig,URformer
from tcom_runtime_bypass_20261017.sparse_fast import spectral_initial

def plan():
 p=OUT/'timing/protocol.json'
 if p.exists():return json.loads(p.read_text())
 value=dict(source_sha256=sha(__file__),seed=2026102205,indices=[[500,501,502],[600,601,602]],grid=list(old.SNRS),refs=list(old.RSRS),regimes=list(old.REGIMES),cases_per_window=459,methods=17,includes='Complete native frozen full/prior or ablated pipeline;complete spectral initialization plus10layerURformer; conversions,routing,solves included;loads/constants preloaded. Detection excluded.',precision='float64/complex128',threads=1,retry='No repeat until passing; preserve all windows',interpretation='Host software p90, not hard percall deadline. Requires all numeric/trainingjobs completed before measuring.')
 write(p,value);return value
@torch.no_grad()
def run():
 frozen();p=plan();assert p['source_sha256']==sha(__file__)
 for path in ('ablation/completed.json','robustness/completed.json','urformer/test/completed.json'):assert json.loads((OUT/path).read_text())['passed']
 assert all(json.loads((U/'checkpoints'/f'seed_{s}'/'status.json').read_text())['state']=='complete' for s in SEEDS)
 assert all(r['returncode']==0 for r in json.loads((U/'workers.json').read_text()))
 models={}
 for seed in SEEDS:
  state=torch.load(U/'checkpoints'/f'seed_{seed}'/'best.pt',map_location='cpu',weights_only=False);model=URformer(32,3,ModelConfig(use_hankel=False,use_transformer=True),NumericConfig(dtype='float64')).double().eval();model.load_state_dict(state['model']);models[seed]=model
 engine=old.engine();selection=json.loads((old.OUT/'policy.json').read_text());rule=frozen()['rule'];menu=[(v,s) for v in ('full','no_recovery','no_final_filter','neither','urformer') for s in SEEDS]+[('compact_sparse',0),('em100',0)]
 def estimate(obs,family,seed):
  if family=='full':return rec.gated(obs,seed,rule,engine,rec.BASE_POLICY)
  if family=='no_recovery':return hyb.hybrid(obs,seed,rec.BASE_POLICY,engine)
  if family=='no_final_filter':return without_filter(obs,seed,engine,rule)[0]
  if family=='neither':return base(obs,seed,engine,False)
  if family=='urformer':
   g0=torch.from_numpy(spectral_initial(obs.z.numpy(),obs.s.numpy(),obs.b.numpy()));return models[seed](g0,*obs.args())
  return old.estimate_method(obs,family,selection,engine)
 for window,indices in enumerate(p['indices'],1):
  dest=OUT/'timing'/f'window{window}.json'
  if dest.exists():continue
  rows=[]
  for ref in old.RSRS:
   for regime,L in old.REGIMES:
    anchor=old.make_anchor(p['seed'],indices,L,ref,truth=False)
    for j,index in enumerate(indices):
     for snr in old.SNRS:rows.append((dict(reference=ref,regime=regime,snr=snr,index=index),old.observed(anchor,snr).batch(slice(j,j+1))))
  assert len(rows)==459
  warm=[rows[0][1],next(o for m,o in rows if m['reference']==5 and m['snr']==60)]
  for obs in warm:
   for family,seed in menu:estimate(obs,family,seed)
  records=[];rng=np.random.default_rng(np.random.SeedSequence([p['seed'],window]));tick=time.time()
  for ci,wi in enumerate(rng.permutation(len(rows)),1):
   meta,obs=rows[wi]
   for mi in rng.permutation(len(menu)):
    family,seed=menu[mi];start=time.perf_counter_ns();g=estimate(obs,family,seed);ms=(time.perf_counter_ns()-start)/1e6;assert torch.isfinite(g).all()
    records.append(dict(**meta,family=family,seed=seed,ms=ms))
   if ci%100==0:print('Timing window',window,ci,'/459',flush=True)
  scores={}
  for family,seed in menu:
   selected=[r for r in records if r['family']==family and r['seed']==seed];x=np.array([r['ms'] for r in selected]);weak=np.array([r['ms'] for r in selected if r['reference']==5 and r['snr']>30]);scores[f'{family}_{seed}']=dict(full_p90_ms=float(np.quantile(x,.9)),target_p90_ms=float(np.quantile(weak,.9)),max_ms=float(x.max()),over_budget=int(np.count_nonzero(x>10.74)),calls=len(x))
  write(dest,dict(passed=True,window=window,protocol_sha256=sha(OUT/'timing/protocol.json'),records=records,scores=scores,seconds=time.time()-tick));print('Timing window',window,'complete',flush=True)
 frozen()
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);run()
