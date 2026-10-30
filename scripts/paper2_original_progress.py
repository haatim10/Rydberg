"""Read-only local progress screen. Never imports or controls the training driver."""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/paper2_original_channel'
REPRO = ROOT/'results/paper2_reproduction'
DIAGNOSTIC = ROOT/'results/paper2_reproduction_diagnostic'
CONFIRM = ROOT/'results/paper2_hankel_confirmation'
PAGE = Path(__file__).with_name('paper2_all_progress.html')
PORT = 8767
MODES = {'literal_independent':'Independent ray directions',
         'clustered_5deg':'Clustered directions (±5° assumption)'}


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError,ValueError):
        return default


def alive(pid):
    if not isinstance(pid,int) or pid<=0:
        return False
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    return True


def original_snapshot():
    root=read(DATA/'status.json',{})
    pid=read(DATA/'pid.json',{}).get('pid')
    live=alive(pid)
    mode=root.get('mode')
    active_name=f'seed{root.get("seed")}_{root.get("arm")}_50ep'
    detail=read(DATA/mode/'status.json',{}) if mode in MODES else {}
    rates=read(DATA/'smoke.json',{}).get('timing',{})
    runs=[]; active=None; completed=0; epochs=0
    for seed in (1,2,3):
        for m,title in MODES.items():
            for arm in ('U','H'):
                name=f'seed{seed}_{arm}_50ep'
                folder=DATA/m/name
                hist=read(folder/'history.json',[])
                finished=(folder/'result.json').exists()
                current=(root.get('state')=='running' and mode==m and name==active_name and not finished)
                batches=detail.get('batches',625)
                batch=detail.get('batch',0) if current and detail.get('run')==name else 0
                recorded_epoch=detail.get('epoch',0) if current and detail.get('run')==name else 0
                if recorded_epoch<=len(hist):
                    batch=0
                extra=min(1.,batch/max(1,batches)) if recorded_epoch>len(hist) else 0.
                display_epoch=min(50,max(len(hist)+1,recorded_epoch)) if current else len(hist)
                record={'name':name,'mode':m,'channel':title,'seed':seed,'arm':arm,
                    'model':'Without Hankel' if arm=='U' else 'With Hankel',
                    'saved_epochs':len(hist),'epoch':display_epoch,'batch':batch,
                    'batches':batches,'fraction':(len(hist)+extra)/50,
                    'finished':finished,'active':current,'history':hist,
                    'last_validation_db':hist[-1]['validation_db'] if hist else None,
                    'last_train_loss':hist[-1]['train_loss'] if hist else None}
                if current:
                    active=record
                runs.append(record); completed+=int(finished); epochs+=len(hist)
    try:
        status_file=DATA/mode/'status.json' if mode in MODES else DATA/'status.json'
        stamp=status_file.stat().st_mtime
        age=max(0,time.time()-stamp)
    except OSError:
        age=None; stamp=None
    state=root.get('state','unknown')
    if state=='failed':
        label='Training failed'
    elif state=='complete':
        label='Training complete · review pending'
    elif live is False:
        label='Training process not running'
    elif age is not None and age>180:
        label='No recent training update'
    elif state=='generating_data':
        label='Preparing datasets'
    else:
        label='Training running' if live else 'Process status unverified'
    # Empirical epoch time where available; otherwise scale the initial timing
    # by observed slowdown. This is an estimate, not a confidence interval.
    ratios=[]; observed={}
    for m in MODES:
        for arm in ('U','H'):
            vals=[x['seconds'] for r in runs if r['mode']==m and r['arm']==arm for x in r['history'][-5:]]
            bench=rates.get(f'{m}_{arm}',{})
            base=625*bench.get('training_seconds_per_batch',0)+63*bench.get('validation_seconds_per_batch',0)
            if vals:
                observed[(m,arm)]=statistics.median(vals)
                if base>0: ratios.append(observed[(m,arm)]/base)
    scale=statistics.median(ratios) if ratios else 1.2
    remaining=0.
    for r in runs:
        bench=rates.get(f'{r["mode"]}_{r["arm"]}',{})
        base=625*bench.get('training_seconds_per_batch',0)+63*bench.get('validation_seconds_per_batch',0)
        seconds=observed.get((r['mode'],r['arm']),base*scale)
        remaining+=50*(1-r['fraction'])*seconds
    eta=[remaining/3600*1.05,remaining/3600*1.4] if state!='complete' and remaining>0 and live and (age is None or age<180) and state!='failed' else None
    return {'state':state,'label':label,'alive':live,'pid':pid,'runs':runs,'active':active,
            'epochs_done':epochs,'epochs_total':600,'runs_done':completed,'runs_total':12,
            'percent':100*sum(r['fraction'] for r in runs)/12,
            'eta_hours':eta,'updated_at':datetime.fromtimestamp(stamp).astimezone().isoformat() if stamp else None,
            'age_seconds':age,'error':root.get('error'),
            'served_at':datetime.now().astimezone().isoformat()}


def reproduction_snapshot():
    root=read(REPRO/'status.json',{})
    pid=read(REPRO/'pid.json',{}).get('pid')
    live=alive(pid)
    state=root.get('state','unknown')
    runs=[]; active=None
    for seed in (1,2,3):
        name=f'seed{seed}_U_50ep'
        folder=REPRO/name
        hist=read(folder/'history.json',[])
        finished=(folder/'result.json').exists()
        current=state=='training' and root.get('run')==name and not finished
        epoch=root.get('epoch',0) if current else len(hist)
        batch=root.get('batch',0) if current and epoch>len(hist) else 0
        batches=root.get('batches',625)
        extra=min(1,batch/max(1,batches)) if current else 0
        record={'name':name,'seed':seed,'channel':'Paper-equation reconstruction',
            'model':'Ordinary URformer · no Hankel','saved_epochs':len(hist),
            'epoch':epoch,'batch':batch,'batches':batches,'fraction':min(1,(len(hist)+extra)/50),
            'finished':finished,'active':current,'history':hist,
            'last_validation_db':hist[-1]['validation_db'] if hist else None,
            'last_train_loss':hist[-1]['train_loss'] if hist else None}
        if current: active=record
        runs.append(record)
    try:
        stamp=(REPRO/'status.json').stat().st_mtime
        age=max(0,time.time()-stamp)
    except OSError:
        stamp=None;age=None
    label={'complete':'Training complete · reproduction review pending',
           'failed':'Reproduction attempt failed',
           'generating_data':'Preparing datasets',
           'classical_evaluation':'Evaluating GS and EM-GS reference methods'}.get(state,'Training running')
    if state not in ('complete','failed'):
        if live is False: label='Training process not running'
        elif live is None: label='Process status unverified'
        elif age is not None and age>180: label='No recent progress update · check needed'
    times=[h['seconds'] for r in runs for h in r['history'][-5:]]
    seconds=statistics.median(times) if times else 625*read(REPRO/'smoke.json',{}).get('seconds_per_training_batch',0)*1.2
    remaining=sum(50*(1-r['fraction'])*seconds for r in runs)
    remaining+=sum(not r['finished'] for r in runs)*300
    eta=[remaining/3600,remaining/3600*1.4] if remaining and live and state not in ('failed','complete') and (age is None or age<180) else None
    return {'state':state,'label':label,'alive':live,'pid':pid,'runs':runs,'active':active,
        'epochs_done':sum(r['saved_epochs'] for r in runs),'epochs_total':150,
        'runs_done':sum(r['finished'] for r in runs),'runs_total':3,
        'percent':100*sum(r['fraction'] for r in runs)/3,'eta_hours':eta,
        'age_seconds':age,'updated_at':datetime.fromtimestamp(stamp).astimezone().isoformat() if stamp else None,
        'error':root.get('error'),'comparison':read(REPRO/'summary.json',{}).get('results',[]),
        'classical':read(REPRO/'classical.json',{}),
        'targets':read(REPRO/'figure3_targets.json',{}).get('curves_nmse_db',{}),
        'snrs':[-10,-5,0,5,10,15,20]}


def diagnostic_snapshot():
    root=read(DIAGNOSTIC/'status.json',{})
    manifest=read(DIAGNOSTIC/'manifest.json',{})
    budget=manifest.get('epochs',10)
    pid=read(DIAGNOSTIC/'pid.json',{}).get('pid')
    live=alive(pid)
    state=root.get('state','unknown')
    runs=[];active=None
    for arm,title in (('mixed','Mixed SNR · −10 to 20 dB'),('fixed5','Fixed SNR · 5 dB')):
        folder=DIAGNOSTIC/arm
        hist=[{**h,'validation_db':h.get('validation_5db_nmse_db')} for h in read(folder/'history.json',[])]
        finished=(folder/'result.json').exists()
        current=state=='training' and root.get('arm')==arm and not finished
        epoch=max(len(hist)+1,root.get('epoch',0)) if current else len(hist)
        batch=root.get('batch',0) if current and root.get('epoch',0)>len(hist) else 0
        batches=root.get('batches',625)
        extra=min(1,batch/max(1,batches)) if current else 0
        record={'name':arm,'seed':1,'channel':title,'model':'Ordinary URformer · no Hankel',
            'saved_epochs':len(hist),'epoch':min(budget,epoch),'epoch_budget':budget,
            'batch':batch,'batches':batches,'fraction':min(1,(len(hist)+extra)/budget),
            'finished':finished,'active':current,'history':hist,
            'last_validation_db':hist[-1]['validation_db'] if hist else None,
            'last_train_loss':hist[-1]['train_loss'] if hist else None}
        if current:active=record
        runs.append(record)
    progress_file=DIAGNOSTIC/'status.json'
    preparation=''
    if state=='preparing' and root.get('arm') in ('mixed','fixed5'):
        cache_file=DIAGNOSTIC/root['arm']/'status.json'
        cache=read(cache_file,{})
        if cache.get('state')=='generating_data':
            progress_file=cache_file
            preparation=f"{cache.get('split','')} data · {cache.get('samples',0):,} / {cache.get('total',0):,} examples"
    try:
        stamp=progress_file.stat().st_mtime;age=max(0,time.time()-stamp)
    except OSError:
        stamp=None;age=None
    label={'complete':'Complete · validation diagnostic only','failed':'Diagnostic failed',
           'preparing':'Preparing diagnostic data','training':'Diagnostic training running'}.get(state,'Not started')
    if state not in ('complete','failed'):
        if live is False:label='Diagnostic process not running'
        elif live is None:label='Process status unverified'
        elif age is not None and age>180:label='No recent diagnostic update'
    times=[h['seconds'] for r in runs for h in r['history'][-5:]]
    seconds=statistics.median(times) if times else 0
    if not seconds and active and active['batch']:
        seconds=root.get('epoch_elapsed_seconds',0)*active['batches']/active['batch']*1.1
    remaining=sum(budget*(1-r['fraction'])*seconds for r in runs)
    eta=[remaining/3600,remaining/3600*1.4] if remaining and live and state not in ('failed','complete') and (age is None or age<180) else None
    return {'state':state,'label':label,'alive':live,'pid':pid,'runs':runs,'active':active,
        'epochs_done':sum(r['saved_epochs'] for r in runs),'epochs_total':2*budget,
        'runs_done':sum(r['finished'] for r in runs),'runs_total':2,
        'percent':100*sum(r['fraction'] for r in runs)/2,'eta_hours':eta,
        'age_seconds':age,'updated_at':datetime.fromtimestamp(stamp).astimezone().isoformat() if stamp else None,
        'error':root.get('error'),'preparation':preparation,
        'diagnostic_results':read(DIAGNOSTIC/'summary.json',{}).get('checkpoint_results',{})}


def confirmation_snapshot():
    root=read(CONFIRM/'status.json',{}); manifest=read(CONFIRM/'manifest.json',{})
    budget=read(CONFIRM/'budget.json',{}); preflight=read(CONFIRM/'preflight.json',{})
    epochs=manifest.get('epochs',18); modes=manifest.get('modes',list(MODES))
    seeds=manifest.get('seeds',list(range(2026092601,2026092606)))
    arms=manifest.get('arms',['standard_U','standard_H','weighted_U','weighted_H'])
    pid=read(CONFIRM/'pid.json',{}).get('pid'); live=alive(pid)
    state=root.get('state','not_started'); runs=[]; active=None
    status_path=CONFIRM/'status.json'
    detail={}
    if state=='training' and root.get('mode') in modes and root.get('seed') in seeds:
        status_path=CONFIRM/root['mode']/f"seed{root['seed']}"/'status.json'
        detail=read(status_path,{})
    for seed in seeds:
        for mode in modes:
            for arm in arms:
                folder=CONFIRM/mode/f'seed{seed}'/'development'/arm
                hist=[{**h,'validation_db':h.get('validation_nmse_db'),'train_loss':h.get('train_online_nmse')} for h in read(folder/'history.json',[])]
                done=(folder/'result.json').exists()
                current=state=='training' and root.get('mode')==mode and root.get('seed')==seed and root.get('arm')==arm and not done
                matching=current and detail.get('arm')==arm
                ep=max(len(hist)+1,detail.get('epoch',0) if matching else 0) if current else len(hist)
                batch=detail.get('batch',0) if matching and detail.get('epoch',0)>len(hist) else 0
                batches=detail.get('batches',manifest.get('counts',{}).get('train',16000)//32)
                fraction=min(1,(len(hist)+batch/max(1,batches))/epochs)
                record={'name':f'{mode}/{seed}/{arm}','mode':mode,'channel':MODES.get(mode,mode),
                    'seed':seeds.index(seed)+1,'arm':arm,'model':('Weighted loss' if arm.startswith('weighted') else 'Standard loss')+(' · with Hankel' if arm.endswith('_H') else ' · without Hankel'),
                    'epoch_budget':epochs,'saved_epochs':len(hist),'epoch':min(epochs,ep),'batch':batch,'batches':batches,
                    'fraction':fraction,'finished':done,'active':current,'history':hist,
                    'last_validation_db':hist[-1]['validation_db'] if hist else None,
                    'last_train_loss':hist[-1]['train_loss'] if hist else None}
                runs.append(record)
                if current: active=record
    try: stamp=status_path.stat().st_mtime; age=max(0,time.time()-stamp)
    except OSError: stamp=None;age=None
    labels={'not_started':'Preparing study','generating_data':'Preparing fresh datasets','training':'Training running',
        'classical_evaluation':'Evaluating classical controls','test_evaluation':'Evaluating frozen test set',
        'complete':'Complete · scientific review pending','failed':'Stopped · error needs attention',
        'budget_exhausted':'Stopped at 15-hour limit · study incomplete'}
    label=labels.get(state,state)
    if state not in ('complete','failed','budget_exhausted','not_started'):
        if live is False: label='Worker not running · checkpoint saved'
        elif age is not None and age>300: label='No recent saved update · check needed'
    remaining_training=0.
    for r in runs:
        times=[h['seconds'] for x in runs if x['arm']==r['arm'] for h in x['history'][-3:]]
        rate=statistics.median(times) if times else r['batches']*preflight.get('seconds_per_training_batch',{}).get(r['arm'],.1)*1.1
        remaining_training+=(epochs-r['saved_epochs']-(r['batch']/max(1,r['batches']) if r['active'] else 0))*rate
    # Count the expensive native reference's saved cases, including partial files
    # via progress where active. Shorter classical and neural scoring gets padding.
    counts=manifest.get('counts',{'val':1200,'test':1800})
    remaining_native=0
    for mode in modes:
        for split in ('val','test'):
            if not (CONFIRM/mode/'classical'/split/'paper1_auto.npz').exists():
                n=counts[split]
                if state=='classical_evaluation' and root.get('method')=='paper1_auto' and root.get('mode')==mode and root.get('split')==split:
                    n=max(0,n-root.get('done',0))
                remaining_native+=n
    estimate=remaining_training+remaining_native*preflight.get('paper1_seconds_per_case',2.1)+300
    eta=[estimate/3600,estimate*1.2/3600] if live and state not in ('complete','failed','budget_exhausted') and (age is None or age<300) else None
    spent=budget.get('worker_seconds',0); allowed=budget.get('limit_seconds',15*3600)
    phase='Training and validation; final test results remain locked.'
    if (CONFIRM/'selection.json').exists(): phase='Model choices frozen; final test evaluation unlocked.'
    if state=='classical_evaluation': phase=f"{root.get('method')} · {root.get('split')} · {root.get('done',0):,}/{root.get('total',0):,} cases"
    if state=='generating_data': phase=f"{root.get('split')} data · {root.get('done',0):,}/{root.get('total',0):,} cases"
    return {'id':'confirmation','title':'Current · Does Hankel add value over strong baselines?',
        'description':f'{len(runs)} runs · 5 paired seeds × 2 channel types × 2 losses × with/without Hankel · {epochs} epochs each',
        'note':'Independent study, not published-URformer reproduction. Includes EM-GS and Paper 1 adaptive Hankel. Training completion is not full-study completion.',
        'state':state,'label':label,'pid':pid,'alive':live,'runs':runs,'active':active,
        'epochs_done':sum(r['saved_epochs'] for r in runs),'epochs_total':len(runs)*epochs,
        'runs_done':sum(r['finished'] for r in runs),'runs_total':len(runs),
        'percent':100*sum(r['fraction'] for r in runs)/max(1,len(runs)),'eta_hours':eta,
        'updated_at':datetime.fromtimestamp(stamp).astimezone().isoformat() if stamp else None,'age_seconds':age,
        'error':root.get('error') or root.get('reason'),'preparation':phase,'phase':phase,
        'worker_hours_used':spent/3600,'worker_hours_left':max(0,allowed-spent)/3600,
        'budget_warning':bool(eta and eta[1]>(allowed-spent)/3600),
        'comparison':[],'classical':{},'targets':{},'diagnostic_results':{}}


def snapshot():
    original=original_snapshot()
    original.update(id='hankel',title='1 · Does Hankel help on the original-style channel?',
        description='12 runs · 2 channel interpretations × ordinary/Hankel models × 3 seeds',
        note='A matched comparison of our estimators. This is NOT a verified reproduction of the published URformer.',
        comparison=[],classical={},targets={})
    reproduction=reproduction_snapshot()
    reproduction.update(id='reproduction',title='2 · Can we reproduce the published ordinary URformer?',
        description='3 runs · ordinary baseline only · 3 seeds · no Hankel',
        note='Independent reconstruction from the paper, with documented assumptions. Published-curve agreement remains unverified.')
    if reproduction['state']=='complete':
        reproduction['label']='Complete · published performance not reproduced'
        reproduction['note']='All three runs finished. Large published-curve discrepancies remain; this is not a verified reproduction.'
    diagnostic=diagnostic_snapshot()
    diagnostic.update(id='diagnostic',title='3 · Does training at one SNR help the baseline?',
        description='2 runs · mixed-SNR versus fixed-5 dB training · one matched seed · 10 epochs each',
        note='Both models use identical 5 dB validation examples. First 10 epochs of a 50-epoch schedule; no Hankel and no final-test claim.')
    return {'experiments':[confirmation_snapshot(),original,reproduction,diagnostic],
        'served_at':datetime.now().astimezone().isoformat()}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?')[0]=='/api/status':
            data=json.dumps(snapshot(),allow_nan=False).encode(); content_type='application/json'
        elif self.path in ('/','/index.html'):
            data=PAGE.read_bytes(); content_type='text/html; charset=utf-8'
        else:
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type',content_type)
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(data)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):
            pass

    def log_message(self,*args):
        pass


def launch():
    with (DATA/'progress_server.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(DATA/'progress_server_pid.json',{}).get('pid')
        if alive(old):
            print(f'Existing screen server PID {old}'); return
        with (DATA/'progress_server.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'serve'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        (DATA/'progress_server_pid.json').write_text(json.dumps({'pid':child.pid,'port':PORT})+'\n')
        print(f'Screen started at http://127.0.0.1:{PORT}/, PID {child.pid}')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['serve','launch','snapshot'])
    command=p.parse_args().command
    if command=='launch': launch()
    elif command=='snapshot': print(json.dumps(snapshot(),indent=2))
    else:
        print(f'Read-only training screen: http://127.0.0.1:{PORT}/',flush=True)
        ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
