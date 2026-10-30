"""Verify completed artifacts, then hash files after the run logs have closed."""
import ast,json
from pathlib import Path
from audit import ROOT,sha,write,verify_protected

def main():
    for p in ROOT.glob('*.py'):ast.parse(p.read_text())
    protocol=json.loads((ROOT/'protocol_receipt.json').read_text());done=json.loads((ROOT/'compute_complete.json').read_text())
    assert protocol['config_sha256']==done['config_sha256']==sha(ROOT/'config.json')
    assert protocol['audit_script_sha256']==done['audit_script_sha256']==sha(ROOT/'audit.py')
    assert protocol['checks_sha256']==sha(ROOT/'checks.py')
    count=verify_protected();summary=json.loads((ROOT/'summary.json').read_text())
    assert len(summary['validation'])==11 and len(summary['confirmation'])==2
    assert len(summary['rows'])==130 and len(summary['covariances'])==11
    assert summary['validation_selection']['selected_before_confirmation']
    selected=summary['validation_selection']['selected_generator']['name']
    assert {a['generator']['name'] for a in summary['confirmation']}=={selected,'independent'}
    assert (ROOT/'data/confirmation_common.npz').stat().st_mtime>=summary['validation_selection']['selection_unix']
    report=json.loads((ROOT/'report_complete.json').read_text())
    assert report['report_sha256']==sha(ROOT/'REPORT.md') and report['summary_sha256']==sha(ROOT/'summary.json')
    write('final_verification.json',dict(passed=True,protected_files_unchanged=count,validation_generators=11,covariance_draws_per_generator=20000,validation_draws_per_generator=2000,confirmation_draws_per_generator=2000,total_condition_rows=130,confirmation_selection_was_frozen=True,configuration_and_computation_hashes_unchanged=True,neural_training_performed=False))
    files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='manifest.json' and 'matplotlib_cache' not in p.parts and '__pycache__' not in p.parts}
    write('manifest.json',files)
    print('Verification passed:',count,'protected files unchanged;',len(files),'audit files hashed.')

if __name__=='__main__':main()
