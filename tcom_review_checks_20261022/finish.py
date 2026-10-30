"""Sequential completion; keeps accuracy jobs separate from timing."""
import os,json,time,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'results/tcom_review_checks_20261022'
def run(module,phase,log):
 print('Starting',module,phase,flush=True)
 with (O/log).open('w') as f:
  p=subprocess.run([sys.executable,'-u','-m',module,phase] if phase else [sys.executable,'-u','-m',module],cwd=R,env=dict(os.environ,PYTHONPATH=str(R)),stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{module}:{phase} failed; see {log}')
def wait(path,training=False):
 while not path.exists():
  workers=O/'urformer/workers.json'
  if training and workers.exists():
   records=json.loads(workers.read_text())
   if any(r['returncode'] not in (None,0) for r in records):raise RuntimeError('URformer training failure; see individual log')
  time.sleep(5)
wait(O/'ablation/completed.json')
run('tcom_review_checks_20261022.analysis','ablation','ablation_summary.log')
run('tcom_review_checks_20261022.run','robust','robustness.log')
run('tcom_review_checks_20261022.analysis','robust','robustness_summary.log')
while True:
 records=json.loads((O/'urformer/workers.json').read_text())
 if any(r['returncode'] not in (None,0) for r in records):raise RuntimeError('Training failed')
 if all(r['returncode']==0 for r in records):break
 time.sleep(10)
run('tcom_review_checks_20261022.urformer','evaluate','urformer_evaluation.log')
run('tcom_review_checks_20261022.analysis','urformer','urformer_summary.log')
run('tcom_review_checks_20261022.timing',None,'timing.log')
print('All scheduled checks and URformer comparison finished.',flush=True)
