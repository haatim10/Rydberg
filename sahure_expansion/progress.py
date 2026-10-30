"""Human-readable progress for this running campaign; no external notifications."""
import argparse
from datetime import datetime,timezone
import json
import time
from .protocol import ROOT,CONFIG


def update():
    status=json.loads((ROOT/'campaign_status.json').read_text())
    lines=['# Broader Hankel experiment — progress','',
        f'Updated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}.',
        f'State: **{status["state"]}**. Phase: **{status["phase"]}**.','',
        '18 models; 20 epochs each; 16,384 training and 2,048 validation worlds. '
        'Full test results are generated only after model selection is frozen.','',
        '| Model | Seed 9101 | Seed 9102 | Seed 9103 |','|---|---:|---:|---:|']
    completed=0;times=[]
    for core in ['em','gd']:
        for structure in ['none','global','adaptive']:
            name=f'{core}_{structure}';values=[]
            for seed in CONFIG['seeds']:
                path=ROOT/'checkpoints'/f'seed_{seed}'/f'{name}_history.json'
                hist=json.loads(path.read_text()) if path.exists() else []
                completed+=len(hist);times.extend(x['seconds'] for x in hist)
                values.append(f'{len(hist)}/20')
            lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines+=['',f'Completed training epochs: **{completed}/360**.','',
        'Three independent training runs share the same training/validation worlds. '
        'Checkpoint selection uses equally weighted held-out measurement likelihood across 126 conditions. '
        'Current validation scores are not channel-estimation test results.','',
        'After training: 32 operating conditions, 400 paired trials each, followed by serial timing and a provenance audit. '
        'Duplicate reference conditions and the two noise-variance offsets reuse the appropriate worlds.','',
        'The full pipeline and experiment settings are in `sahure_expansion/README.md`. '
        'The completed pilot is preserved in `results/sahure_pilot/`.']
    if status['state']=='failed': lines+=['',f'Failure: {status.get("error", "see campaign log")}']
    if (ROOT/'REPORT.md').exists(): lines+=['',f'[Read the results report]({(ROOT/"REPORT.md").resolve()})']
    (ROOT/'LIVE_STATUS.md').write_text('\n'.join(lines)+'\n')
    return status['state']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        state=update()
        if not args.watch or state in ['complete','failed']: break
        time.sleep(30)
