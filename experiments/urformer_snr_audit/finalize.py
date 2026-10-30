"""Verify completed artifacts and record final hashes after all file logs close."""
from pathlib import Path
import json, ast, hashlib
ROOT=Path(__file__).resolve().parent
from run import snapshot, sha, write

def main():
    for p in ROOT.glob('*.py'):
        ast.parse(p.read_text())
    receipt=json.loads((ROOT/'protocol_receipt.json').read_text())
    completed=json.loads((ROOT/'compute_complete.json').read_text())
    for field,path in [('config_sha256','config.json'),('script_sha256','run.py')]:
        assert receipt[field]==completed[field]==sha(ROOT/path),field
    protected=json.loads((ROOT/'protected_before.json').read_text())
    assert snapshot()==protected,'Earlier protected files changed'
    summary=json.loads((ROOT/'summary.json').read_text())
    assert len(summary['task2_and_3']['rows'])==120
    assert len(summary['task1']['rows'])==5
    assert summary['selection']['validation_winner']=='reproduction_ensemble'
    final=json.loads((ROOT/'complete.json').read_text())
    assert final['report_sha256']==sha(ROOT/'REPORT.md')
    assert final['summary_sha256']==sha(ROOT/'summary.json')
    for row in summary['task2_and_3']['rows']:
        assert row['calibration']['per_entry_channel_power']['mean']>0
        assert len(row['metrics'])==2
    write('final_verification.json',dict(passed=True,protected_files=len(protected),classical_rows=120,oracle_rows=5,config_and_computation_unchanged=True,selection_used_validation_only=True,neural_training_performed=False))
    files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='manifest.json' and 'matplotlib_cache' not in p.parts and '__pycache__' not in p.parts}
    write('manifest.json',files)
    print('Final verification passed:',len(protected),'protected files unchanged;',len(files),'audit files hashed.')

if __name__=='__main__':main()
