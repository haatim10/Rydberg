"""One fixed URformer architecture, trained on matched pilot banks; no search."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,json,time,subprocess,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np,torch
from sahure.data import Observations
from trackD_urformer.config import ModelConfig,NumericConfig
from trackD_urformer.urformer import URformer,count_parameters
from trackD_urformer.train import nmse_loss
from tcom_pilot_em_20261018.training_data import load_set,split_indices,paired_split,condition_grid
from tcom_pilot_em_20261018.training_protocol import CONFIG,ROOT as DATA
from tcom_runtime_bypass_20261017.sparse_fast import spectral_initial
from rydberg_sim.track_b_drivers import draw_L_k,track_b_spec
from rydberg_sim.monte_carlo import generate_channel_estimation_trial
from sahure.math import nll
from .run import OUT,REPO,sha,write,load,save,old,oldber,ob,ber_counts,err,SEEDS,frozen
U=OUT/'urformer'
def spec():
 p=json.loads((U/'protocol.json').read_text())
 for n,h in p['source_hashes'].items():assert sha(REPO/n)==h,n
 for n,h in p['input_hashes'].items():assert sha(REPO/n)==h,n
 return p
def freeze():
 assert not (U/'protocol.json').exists();sources=[Path(__file__),*sorted((REPO/'trackD_urformer').glob('*.py')),REPO/'tcom_pilot_em_20261018/training_data.py']
 inputs=[DATA/'training_protocol.json']
 for split in ('train','val'):
  folder=DATA/'training_data'/split;mf=json.loads((folder/'manifest.json').read_text());inputs.append(folder/'manifest.json')
  for n,h in mf['file_hashes'].items():assert sha(folder/n)==h;inputs.append(folder/n)
 p=dict(model='URformer adaptation on our geometric channel; supervised NMSE training',model_config=asdict(ModelConfig(use_hankel=False,use_transformer=True)),numeric_config=asdict(NumericConfig(dtype='float32')),N=32,K=3,P=30,
  seeds=SEEDS,epochs=20,train_cases=16384,validation_cases=2048,batch_size=32,learning_rate=.001,optimizer='Adam',scheduler='CosineAnnealingLR,Tmax20',gradient_clip=1.,
  inputs='Exact matched39-observation banks;30context masks fitseed+10000+epoch, minibatchorder fitseed+20000+epoch,9columns excluded frommodelinput',
  truth='Regenerate G from original simulator keys and verify all original39observations bitwise at float32; labels used only for this supervised baseline',
  initialization='Same spectral initializer used by our estimators, recomputed from context30 only',
  selection='Minimum equal126-stratum mean supervised normalized squared error on separate fixed30context validation split; epoch0 eligible; strict improvement. HeldoutNLL additionally reported, never selected on test.',
  compared_training_budget='Same examples,30fittingpilots,epochs,batchsize and initialLR as our model; cosine schedule and supervisedloss follow URformer; architecture same10layers/3encoders/64hidden. Not the published50epoch20kclustered-channel experiment.',
  baseline_scope='Standalone URformer, no extra Hankel filter/EM/recovery; ordinary<=20 comparison isolates learnedstarts; wholegrid includes OODhighSNR. No claim of matched postprocessing architecture ablation.',
  final_evaluation='Same immutable153testcells and64jointQPSKdata as final receiver,3trainedfits separate; batch1oneCPUthread doubleprecision timing after jobsidle',
  source_hashes={str(p.relative_to(REPO)):sha(p) for p in sources},input_hashes={str(p):sha(p) for p in inputs},created_unix=time.time())
 write(U/'protocol.json',p);print('URformer matched protocol frozen',sha(U/'protocol.json'),flush=True)
def truth():
 spec()
 for split in ('train','val'):
  out=U/'data'/f'{split}_truth.npy'
  if out.exists():continue
  a,ids=load_set(split);grid=condition_grid();seed=CONFIG[f'{split}_data_seed'];values=[];tick=time.time()
  for i,cid in enumerate(ids):
   c=grid[int(cid)];L=c['L'];paths=draw_L_k(i,3,master_seed=seed) if L is None else (L,)*3
   cfg=track_b_spec(P=39,n_trials=i+1,N=32,K=3,L=paths,master_seed=seed,experiment='tcom_pilot_em_20261018');w=generate_channel_estimation_trial(cfg,i,c['snr'],c['rsr'])
   for actual,expected in zip((w.Z,w.S,w.B,w.sigma2),a.args()):np.testing.assert_array_equal(np.asarray(actual,dtype=expected.numpy().dtype),expected[i].numpy())
   values.append(w.G)
   if (i+1)%2048==0:print('Truth replay',split,i+1,'/',len(a),flush=True)
  out.parent.mkdir(parents=True,exist_ok=True);np.save(out,np.asarray(values,dtype=np.complex64));write(U/'data'/f'{split}_replay.json',dict(passed=True,cases=len(a),observations_verified_all=True,truth_sha256=sha(out),seconds=time.time()-tick))
 print('All original training/validation observations exactly reproduced; target labels stored only in new baseline directory.',flush=True)
def initial(a):
 return torch.from_numpy(spectral_initial(a.z.numpy().astype(np.float64),a.s.numpy().astype(np.complex128),a.b.numpy().astype(np.complex128))).to(torch.complex64)
@torch.no_grad()
def validate(model,a,h,g,conditions):
 model.eval();ratios=[];losses=[]
 for start in range(0,len(a),32):
  sl=slice(start,start+32);obs=a.batch(sl);pred=model(initial(obs),*obs.args());ratios.append((abs(pred-g[sl])**2).sum((-2,-1))/(abs(g[sl])**2).sum((-2,-1)));hold=h.batch(sl);losses.append(nll(hold.z,pred@hold.s+hold.b,hold.sigma2).mean((-2,-1)))
 ratios=torch.cat(ratios).double();losses=torch.cat(losses).double();ids=torch.from_numpy(conditions);counts=torch.bincount(ids,minlength=126);assert (counts>0).all()
 score=(torch.zeros(126,dtype=torch.float64).scatter_add_(0,ids,ratios)/counts).mean();hl=(torch.zeros(126,dtype=torch.float64).scatter_add_(0,ids,losses)/counts).mean();assert torch.isfinite(score) and torch.isfinite(hl)
 return float(score),float(hl)
def checkpoint(path,blob):
 tmp=path.with_suffix('.tmp.pt');torch.save(blob,tmp);tmp.replace(path)
def train(seed):
 spec();frozen();torch.manual_seed(seed);np.random.seed(seed);folder=U/'checkpoints'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
 train,_=load_set('train');val,conditions=load_set('val');_,va,vh=paired_split(val,CONFIG['validation_mask_seed']);gt=torch.from_numpy(np.load(U/'data/train_truth.npy'));gv=torch.from_numpy(np.load(U/'data/val_truth.npy'))
 model=URformer(32,3,ModelConfig(use_hankel=False,use_transformer=True),NumericConfig(dtype='float32'));opt=torch.optim.Adam(model.parameters(),lr=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
 latest=folder/'latest.pt'
 if latest.exists():
  state=torch.load(latest,map_location='cpu',weights_only=False);assert state['protocol_sha256']==sha(U/'protocol.json');model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
 else:
  score,hl=validate(model,va,vh,gv,conditions);state=dict(epoch=0,best_score=score,history=[],seed=seed,protocol_sha256=sha(U/'protocol.json'),parameters=count_parameters(model),val_score=score,heldout_nll=hl)
  state.update(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict());checkpoint(folder/'best.pt',state);checkpoint(latest,state)
 for epoch in range(state['epoch'],20):
  ids=split_indices(len(train),seed+10000+epoch)[1];order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+20000+epoch));tick=time.time();total=0.;model.train();gradmax=0.
  for start in range(0,len(train),32):
   j=order[start:start+32];obs=train.batch(j).columns(ids[j]);opt.zero_grad();pred=model(initial(obs),*obs.args());loss=nmse_loss(pred,gt[j]);assert torch.isfinite(loss),('nonfinite',seed,epoch,start)
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);gradmax=max(gradmax,float(norm));opt.step();total+=float(loss.detach())*len(j)
   if start%4096==0:write(folder/'status.json',dict(state='training',epoch=epoch+1,completed=start,total=len(train),seed=seed,seconds=time.time()-tick))
  sched.step();score,hl=validate(model,va,vh,gv,conditions);entry=dict(epoch=epoch+1,train_loss=total/len(train),validation_nmse=score,validation_db=10*np.log10(score),heldout_nll=hl,seconds=time.time()-tick,max_preclip_gradient=gradmax)
  best=score<state['best_score'];state['history'].append(entry);state.update(epoch=epoch+1,best_score=min(score,state['best_score']),val_score=score,heldout_nll=hl,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict());checkpoint(latest,state)
  if best:checkpoint(folder/'best.pt',state)
  write(folder/'history.json',dict(seed=seed,selected_minimum=state['best_score'],history=state['history']));print('URformer',seed,'epoch',epoch+1,'val',entry['validation_db'],'dB',entry['seconds'],'seconds',flush=True)
 write(folder/'status.json',dict(state='complete',epoch=20,seed=seed,best_score=state['best_score']));spec();frozen()
def evaluate():
 spec();frozen();models={}
 for seed in SEEDS:
  state=torch.load(U/'checkpoints'/f'seed_{seed}'/'best.pt',map_location='cpu',weights_only=False);m=URformer(32,3,ModelConfig(use_hankel=False,use_transformer=True),NumericConfig(dtype='float64')).double().eval();m.load_state_dict(state['model']);models[seed]=m
 from tcom_runtime_bypass_20261017.sparse_fast import spectral_initial
 for ci,p in enumerate(sorted((old.OUT/'test/cells').glob('*.npz')),1):
  dest=U/'test/cells'/p.name
  if dest.exists():continue
  a=load(p);obs=ob(a);g0=torch.from_numpy(spectral_initial(obs.z.numpy(),obs.s.numpy(),obs.b.numpy()));pred=[]
  with torch.no_grad():
   for seed,m in models.items():pred.append(m(g0,*obs.args()).numpy())
  pred=np.stack(pred);data=load(old.OUT/'ber/data'/p.name)['z'];counts=ber_counts(pred,data,a['b'][...,:1],a['sigma2']);save(dest,estimates=pred,errors=err(pred,a['truth_g'][None]),bit_errors=counts,truth_energy=a['truth_energy'],source_sha256=np.array(sha(p)),snr=a['snr'],rsr=a['rsr'],regime=a['regime']);print('URformer evaluation',ci,'/153',p.stem,flush=True)
 write(U/'test/completed.json',dict(passed=True,cells=len(list((U/'test/cells').glob('*.npz')))));spec();frozen()
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','truth','train','evaluate'));parser.add_argument('--seed',type=int);a=parser.parse_args();train(a.seed) if a.phase=='train' else globals()[a.phase]()
