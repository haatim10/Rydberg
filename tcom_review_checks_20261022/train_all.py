import os,sys,subprocess,time,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results/tcom_review_checks_20261022/urformer';processes=[];streams=[]
for seed in (9101,9102,9103):
 stream=(out/f'train_seed_{seed}.log').open('a');streams.append(stream)
 process=subprocess.Popen([sys.executable,'-u','-m','tcom_review_checks_20261022.urformer','train','--seed',str(seed)],cwd=root,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(root)))
 processes.append((seed,process));print('Started URformer seed',seed,'pid',process.pid,flush=True)
while True:
 state=[{'seed':s,'pid':p.pid,'returncode':p.poll()} for s,p in processes]
 (out/'workers.json').write_text(json.dumps(state,indent=2)+'\n')
 if all(x['returncode'] is not None for x in state):break
 time.sleep(15)
for f in streams:f.close()
print(json.dumps(state),flush=True)
if any(x['returncode'] for x in state):raise SystemExit(1)
