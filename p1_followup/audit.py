"""Independent verification of released aggregates and manuscript numbers."""
import json,hashlib
import numpy as np
from p1_followup.run import OUT,CELLS,write

def main():
    summary=json.loads((OUT/'analysis/summary.json').read_text())
    checked=0
    for cell in CELLS:
        worlds=[json.loads((OUT/'raw'/f'{cell["seed"]+i}.json').read_text()) for i in range(cell['n'])]
        den=sum(w['den'] for w in worlds);em=sum(w['em_error'] for w in worlds)
        for kind in ('hard','smooth'):
            totals={q:sum(next(r['error'] for r in w['rows'] if (r['kind'],r['sweeps'],r['split'])==(kind,q,0)) for w in worlds) for q in (1,2,4,8)}
            for q,total in totals.items():
                row=next(r for r in summary['sweeps'] if (r['L'],r['snr'],r['kind'],r['sweeps'])==(cell['L'],cell['snr'],kind,q))
                assert abs(row['vs_em']['nmse']-10*np.log10(total/den))<1e-12
                assert abs(row['vs_em']['gain']-10*np.log10(em/total))<1e-12
                assert abs(row['vs_four']['gain']-10*np.log10(totals[4]/total))<1e-12
                checked+=1
        fixed=sum(next(r['error'] for r in w['rows'] if (r['kind'],r['sweeps'],r['split'])==('hard',4,0)) for w in worlds)
        random=sum(sum(r['error'] for r in w['rows'] if r['split']>0)/3 for w in worlds)
        row=next(r for r in summary['splits'] if (r['L'],r['snr'])==(cell['L'],cell['snr']))
        assert abs(row['random_mean_vs_fixed']['gain']-10*np.log10(fixed/random))<1e-12
        checked+=1
    write(OUT/'analysis/independent_audit.json',dict(passed=True,aggregates_checked=checked,script_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest()))
    print(f'{checked} independent aggregate checks passed')
if __name__=='__main__':main()
