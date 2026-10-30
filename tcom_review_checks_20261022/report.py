"""Plain-English results; experimental decisions remain frozen."""
import json,hashlib
from pathlib import Path
import numpy as np
from .run import OUT,REPO,SEEDS,VARIANTS,old,load,write,frozen,sha
from .urformer import U,spec

def timing():
 windows=[json.loads((OUT/'timing'/f'window{i}.json').read_text()) for i in (1,2)];result={}
 for family in (*VARIANTS,'urformer','compact_sparse','em100'):
  per=[]
  for w in windows:
   rows=[v for k,v in w['scores'].items() if k.startswith(family+'_')];per.append(dict(worst_fit_full_p90_ms=max(v['full_p90_ms'] for v in rows),worst_fit_target_p90_ms=max(v['target_p90_ms'] for v in rows),maximum_ms=max(v['max_ms'] for v in rows),over_budget=sum(v['over_budget'] for v in rows),calls=sum(v['calls'] for v in rows)))
  result[family]=dict(windows=per,full_p90_ms=max(v['worst_fit_full_p90_ms'] for v in per),target_p90_ms=max(v['worst_fit_target_p90_ms'] for v in per),meets_both_scopes_in_both_windows=all(v['worst_fit_full_p90_ms']<=10.74 and v['worst_fit_target_p90_ms']<=10.74 for v in per),maximum_ms=max(v['maximum_ms'] for v in per),over_budget=sum(v['over_budget'] for v in per))
 return result

def audit():
 frozen();spec();n=0;maxerr=0.
 for p in sorted((OUT/'ablation/cells').glob('*.npz')):
  a=load(p);o=load(old.OUT/'test/cells'/p.name);b=load(old.OUT/'ber/cells'/p.name);assert str(a['source_sha256'])==sha(old.OUT/'test/cells'/p.name)
  for fit,s in enumerate(SEEDS):
   for j,family in enumerate(('receiver','prior')):
    np.testing.assert_array_equal(a['errors'][fit,j],o['errors'][list(o['method_names']).index(f'{family}_{s}')]);np.testing.assert_array_equal(a['bit_errors'][fit,j],b['errors_per_user'][list(b['method_names']).index(f'{family}_{s}')].sum(-1));n+=len(a['truth_energy'])
  assert np.isfinite(a['errors']).all() and (a['errors']>=0).all();maxerr=max(maxerr,float(a['base_parity'].max()))
 assert len(list((OUT/'ablation/cells').glob('*.npz')))==153
 robust=list((OUT/'robustness/cells').glob('*.npz'));assert len(robust)==198
 for p in robust:
  a=load(p);assert np.isfinite(a['errors']).all() and np.isfinite(a['phase_adjusted_errors']).all();assert (a['bit_errors']>=0).all() and (a['bit_errors']<=384).all()
 phase_cells=0;phase_bit_changes=0;phase_error_difference=0.
 for p in robust:
  if 'reference_phase' not in p.stem:continue
  a=load(p);q=load(p.parent/(p.stem.split('_reference_phase')[0]+'_correct.npz'));phase_cells+=1
  phase_bit_changes+=int(np.count_nonzero(a['bit_errors']!=q['bit_errors']));phase_error_difference=max(phase_error_difference,float(np.max(abs(a['phase_adjusted_errors']-q['errors']))))
 ur=list((U/'test/cells').glob('*.npz'));assert len(ur)==153
 for p in ur:
  a=load(p);o=load(old.OUT/'test/cells'/p.name);assert str(a['source_sha256'])==sha(old.OUT/'test/cells'/p.name);assert np.isfinite(a['estimates']).all();np.testing.assert_allclose(a['errors'],np.sum(abs(a['estimates']-o['truth_g'][None])**2,axis=(-2,-1)),rtol=1e-14,atol=1e-14)
 # A second implementation of the 64-hypothesis Rician detector on24decodes.
 from scipy.special import i0e
 from tcom_followups_20261017.ber_detector import hypotheses
 symbols,bits=hypotheses(3);decodes=0
 for p in ur[::25]:
  u=load(p);a=load(old.OUT/'test/cells'/p.name);data=load(old.OUT/'ber/data'/p.name)['z']
  for fit,j in ((0,0),(1,17),(2,63),(0,127)):
   amplitude=abs(u['estimates'][fit,j]@symbols.T+a['b'][j,:,:1]);z=data[j];v=float(a['sigma2'][j]);loss=np.empty((64,64))
   for h in range(64):
    mu=amplitude[:,h,None];loss[:,h]=(((z-mu)**2/v)-np.log(i0e(2*z*mu/v))).sum(0)
   errors=np.count_nonzero(bits[np.argmin(loss,axis=1)]!=bits);assert errors==u['bit_errors'][fit,j];decodes+=1
 uchecks=json.loads((U/'independent_checks.json').read_text());assert uchecks['passed']
 result=dict(passed=True,urformer_conventions_and_precision=uchecks,original_predictions_and_bit_counts_rechecked=n,base_dispatch_max_error=maxerr,ablation_cells=153,robustness_cells=198,urformer_cells=153,independent_rician_decodes=decodes,all_sources_and_original_weights_unchanged=True,phase_cells_checked=phase_cells,phase_changed_bit_counts=phase_bit_changes,maximum_phase_adjusted_squared_error_difference=phase_error_difference)
 write(OUT/'audit.json',result);return result

def run():
 a=json.loads((OUT/'ablation/summary.json').read_text());m=json.loads((OUT/'acceptance_margin.json').read_text());r=json.loads((OUT/'robustness/summary.json').read_text());u=json.loads((U/'summary.json').read_text());t=timing();checks=audit();lines=[]
 def add(s=''):lines.append(s)
 add('REVIEW CHECKS AND MATCHED-BUDGET URFORMER COMPARISON - 22 OCTOBER 2026')
 add('All earlier models, policies and scientific result files were preserved. No receiver retuning or new repair was performed. The current paper has not been edited by these checks.')
 add();add('1. ACCEPTANCE CUTOFF')
 add(f"Checked {m['cases']} actual recovery decisions. Minimum mean fitting-loss improvement: {m['minimum_gain']:.9f}.")
 add('Changed decisions at cutoffs 1e-12,1e-10,1e-8,1e-6,1e-4,0.01: '+str(list(m['changed_decisions_by_absolute_margin'].values())))
 add(f"Scale-aware rounding margin changes: {m['scale_aware_32eps_changed']}. Float32 LOSS-SCORE changes: {m['float32_score_changed']}, nonfinite scores: {m['float32_nonfinite']}.")
 add('This tests saved decisions and score arithmetic, not full float32 inference or all possible future channels. No new cutoff was selected.')
 add();add('2. REMOVE ONE PART AT A TIME - SAME153SETTINGS,1152INDEPENDENTCHANNELS,3TRAININGRUNS')
 add(f"{'Method':24s} {'Mean NMSE dB':>13s} {'BER %':>12s} {'Slower full p90 ms':>20s}")
 s=a['all']['scores']
 for i,name in enumerate(a['variants']):add(f"{name:24s} {s['mean_setting_nmse_db'][i]:13.6f} {100*s['ber'][i]:12.6f} {t[name]['full_p90_ms']:20.3f}")
 add('Each setting receives equal weight in the dB score. BER is unchanged across these four versions on this test. Removing the final filter leaves the trained stage filters intact. The combined-removal version is also shown.')
 add('Descriptive paired95% intervals for reference-minus-full NMSE:')
 for name,c in a['intervals_descriptive'].items():add(f"  {name}: {c['nmse_gain_reference_minus_full_ci95']}")
 add();add('3. CALIBRATION ERRORS - SMALL FIXED SCREEN')
 add('18settings: references5/20dB; SNR10/32.5/45dB; paths2/4/random3-7.192independentchannels,576conditioncases peroffset. Actual observations are identical across methods and offsets; only supplied calibration changes. All methods use the same supplied calibration in the detector.')
 add(f"{'Input change':32s} {'Proposed NMSE':>14s} {'Proposed BER %':>16s} {'Sparse BER %':>14s} {'Phase-adjusted NMSE':>21s}")
 for name,rr in r['arms'].items():
  q=rr['all']['scores'];aligned=rr['all']['phase_adjusted_scores'];add(f"{name:32s} {q['mean_setting_nmse_db'][0]:14.5f} {100*q['ber'][0]:16.6f} {100*q['ber'][2]:14.6f} {aligned['mean_setting_nmse_db'][0]:21.5f}")
 add('A constant reference-phase offset can be absorbed by a rotation of the estimated channel. Absolute channel NMSE therefore changes even when magnitude predictions and BER do not. Phase-adjusted NMSE removes the KNOWN EXPERIMENTAL offset only for scoring; that information never enters inference. This does not test time-varying or independent per-antenna phase errors.')
 worst=[]
 for f in (OUT/'robustness/cells').glob('*noise_+2db.npz'):
  x=load(f);y=load(f.parent/(f.stem.split('_noise_')[0]+'_correct.npz'))
  def score(z):return float(np.mean(10*np.log10(z['errors'][[0,2,4]].sum(-1)/z['truth_energy'].sum())))
  worst.append((score(x)-score(y),f.stem,score(x),score(y)))
 add('Largest setting-level NMSE losses for noise input +2dB (loss,setting,wrong-inputNMSE,correct-inputNMSE):')
 for row in sorted(worst,reverse=True)[:4]:add('  '+str(row))
 add('The largest losses are near the strong-reference32.5dB transition; aggregate averages alone hide this sensitivity. No new switch was tuned to remove it.')
 add('This is a representative sensitivity screen, not a new whole153setting robustness claim. Per-range and per-method scores are retained in robustness/summary.json.')
 add();add('4. URFORMER - ONE FIXED ARCHITECTURE, THREE SEPARATELY TRAINED RUNS')
 params=[];selected=[]
 for seed in SEEDS:
  import torch
  b=torch.load(U/'checkpoints'/f'seed_{seed}'/'best.pt',map_location='cpu',weights_only=False);last=torch.load(U/'checkpoints'/f'seed_{seed}'/'latest.pt',map_location='cpu',weights_only=False);assert last['epoch']==20 and b['val_score']==last['best_score'];assert b['protocol_sha256']==sha(U/'protocol.json');params.append(b['parameters']['totals']['all_parameters']);selected.append(b['epoch'])
 add(f'URformer parameters perrun: {params}; selected validation epochs: {selected}.')
 add('Same16384training examples,2048validation examples,30inputpilots,20epochs,batch32,initialLR0.001,and per-run training masks/orders as our controller. URformer uses true-channel NMSE labels and a cosine learning-rate schedule. Its spectral initialization uses only the30inputpilots. Original39pilot observations were regenerated and matched exactly before adding labels in a separate new directory.')
 add('Architecture:10unrolledlayers,3Transformerencoders/layer,width64; no Hankel additions. This is a matched-budget adaptation to ourK=3geometric channel, not the publishedK=4clustered-channel50epoch experiment. The baseline is standalone: it does not receive our additional high-SNR EM/filter/recovery policy. The ordinary-SNR comparison directly compares the learned starts. No architecture/initialization/hyperparameter search was performed.')
 for scope in ('all','ordinary','weak_target','high'):
  q=u['scopes'][scope]['scores'];add();add(scope+f" ({u['scopes'][scope]['settings']} settings)");add(f"{'Method':24s} {'Mean NMSE dB':>13s} {'Pooled NMSE':>14s} {'BER %':>12s}")
  for i,name in enumerate(u['methods']):add(f"{name:24s} {q['mean_setting_nmse_db'][i]:13.6f} {q['pooled_nmse'][i]:14.8f} {100*q['ber'][i]:12.6f}")
 add();add('5. RUNTIME - TWO ISOLATED WINDOWS AFTER ALL NUMERICAL JOBS')
 add('Complete batch-one CPU inference,one thread,doubleprecision. All3fits must pass full-grid and weak-target p90 in both windows. Model loading and BER decoding excluded. Our receiver uses its optimized NumPy/SciPy implementation; URformer uses the prepared PyTorch implementation. These are implementation measurements, not a universal algorithmic speed limit.')
 for name,q in t.items():add(f"{name:24s} fullp90={q['full_p90_ms']:.3f}ms targetp90={q['target_p90_ms']:.3f}ms maximum={q['maximum_ms']:.3f}ms calls>10.74ms={q['over_budget']} all-p90-pass={q['meets_both_scopes_in_both_windows']}")
 add();add('6. VERIFICATION AND LIMITS')
 add(json.dumps(checks,indent=2));add('Bootstrap intervals are descriptive and conditional on the trained runs and simulation populations. Zero observed errors do not prove zero population BER or equivalence. These checks do not test new user counts,pilot designs,hardware,or new training populations. A 20epoch budget comparison does not establish URformer convergence or reproduce its published50epoch performance.')
 (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n');write(OUT/'summary.json',dict(acceptance={k:v for k,v in m.items() if k!='rows'},ablations=a,robustness=r,urformer=u,timing=t,audit=checks))
 manifest={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*.json')) if p.name!='SHA256SUMS.json'};write(OUT/'SHA256SUMS.json',manifest);print('Final plain-English report written',OUT/'RESULTS.txt',flush=True)
if __name__=='__main__':run()
