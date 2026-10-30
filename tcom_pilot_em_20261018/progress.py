"""Lightweight progress snapshot; never imports Torch or changes experiment data."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results/tcom_pilot_em_20261018'


def snapshot():
    rows = []
    for seed in (9101, 9102, 9103):
        status_path = ROOT / f'training_workers/seed_{seed}/status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'state': 'pending'}
        for arm in ('context21', 'context30'):
            history_path = ROOT / f'checkpoints/{arm}/seed_{seed}/em_adaptive_history.json'
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            row = dict(seed=seed, arm=arm, completed_epochs=len(history), worker_state=status['state'])
            if history:
                row.update(last_val30_nll=history[-1]['val30_nll'],
                           best_val30_nll=min(h['val30_nll'] for h in history),
                           last_training_seconds=history[-1]['seconds'])
            if status.get('state') == 'running':
                row.update(active_arm=status.get('arm'), active_epoch=status.get('epoch'))
            if status.get('state') == 'failed':
                row['error'] = status.get('error')
            rows.append(row)
    return dict(at=datetime.now(timezone.utc).isoformat(),
                completed_model_epochs=sum(r['completed_epochs'] for r in rows),
                planned_model_epochs=120, models=rows)


if __name__ == '__main__':
    import sys
    value = snapshot()
    if '--compact' in sys.argv:
        print(value['at'], f"epochs {value['completed_model_epochs']}/{value['planned_model_epochs']}")
        for r in value['models']:
            print(r['seed'], r['arm'], r['completed_epochs'], r['worker_state'],
                  'best_val30=', round(r['best_val30_nll'], 6) if 'best_val30_nll' in r else None,
                  'last_train_s=', round(r['last_training_seconds'], 1) if 'last_training_seconds' in r else None)
    else:
        print(json.dumps(value, indent=2))
