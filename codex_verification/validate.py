"""Numerical equivalence gates, separate from the Monte Carlo campaign."""
import json,time
from pathlib import Path
import numpy as np
from codex_verification import solver
from codex_verification.worlds import make_world
from rydberg_sim.gs import em_gs_channel_rows,bessel_ratio
from rydberg_sim.track_b_proposed import hs_gs,select_order_heldout,cadzow_project


def main():
    rows=[]
    x=np.r_[0,np.logspace(-10,8,1000)]
    np.testing.assert_allclose(solver.ratio(x),bessel_ratio(x),rtol=2e-13,atol=2e-15)
    for n,p,s,it,sel in [(8,10,-5.,100,25),(16,30,20.,20,8),
                          (32,30,5.,100,25),(64,20,0.,8,5)]:
        w=make_world(9200000+n,N=n,P=p,snr=s)
        t=time.perf_counter()
        r,sc=solver.select(w.S,w.Z,w.B,w.sigma2,iterations=sel)
        r0,sc0=select_order_heldout(w.S,w.Z,w.B,w.sigma2,max_iter=sel)
        assert r==r0,(r,r0)
        np.testing.assert_allclose(sc,list(sc0.values()),rtol=1e-8,atol=1e-8)
        g=solver.fit(w.S,w.Z,w.B,w.sigma2,iterations=it,ranks=[r,(n+1)//2])
        h=hs_gs(w.S,w.Z,w.B,w.sigma2,L_hat=r,max_iter=it).G_hat
        e=em_gs_channel_rows(w.S,w.Z,w.B,w.sigma2,max_iter=it).G_hat
        np.testing.assert_allclose(g[0],h,rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(g[1],e,rtol=1e-9,atol=1e-10)
        for sweeps in (1,4):
            a=solver.project(e[None],max(1,n//4),sweeps)[0]
            b=np.stack([cadzow_project(e[:,k],max(1,n//4),n_iter=sweeps)
                        for k in range(e.shape[1])],axis=1)
            np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-11)
        rows.append(dict(N=n,P=p,snr=s,iterations=it,selection_iterations=sel,
                         rank=r,max_abs_hs=float(np.max(abs(g[0]-h))),
                         max_abs_em=float(np.max(abs(g[1]-e))),seconds=time.perf_counter()-t))
        print(rows[-1],flush=True)
    out=Path("codex_verification/results");out.mkdir(exist_ok=True)
    (out/"validation.json").write_text(json.dumps({"passed":True,"cases":rows},indent=2))

if __name__=="__main__": main()
