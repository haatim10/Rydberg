"""Separate serial timing of production and vectorized implementations.

Do not run concurrently with the Monte Carlo campaign. Times include the
spectral initialization in each component, as the actual pipelines do.
"""
import json,platform,time
from pathlib import Path
import numpy as np
from codex_verification.worlds import make_world
from codex_verification import solver
from rydberg_sim.gs import em_gs_channel_rows
from rydberg_sim.track_b_proposed import hs_gs,select_order_heldout,cadzow_project

ROOT=Path(__file__).resolve().parent


def call(fn):
    t=time.perf_counter();v=fn();return v,time.perf_counter()-t


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    rows=[]
    for n in (8,16,32,64):
        for trial in range(3):
            w=make_world(12_000_000+n*100+trial,N=n,P=20,snr=5.,rsr=10.,family="TD")
            # Alternate implementation order across trials.
            for mode in (("production","vectorized") if trial%2==0 else ("vectorized","production")):
                if mode=="production":
                    (r,_),ts=call(lambda:select_order_heldout(w.S,w.Z,w.B,w.sigma2,max_iter=25))
                    h,th=call(lambda:hs_gs(w.S,w.Z,w.B,w.sigma2,L_hat=r,max_iter=100).G_hat)
                    e,te=call(lambda:em_gs_channel_rows(w.S,w.Z,w.B,w.sigma2,max_iter=100).G_hat)
                    _,tp=call(lambda:np.stack([cadzow_project(e[:,k],r,n_iter=4) for k in range(3)],axis=1))
                else:
                    (r,_),ts=call(lambda:solver.select(w.S,w.Z,w.B,w.sigma2,iterations=25))
                    h,th=call(lambda:solver.fit(w.S,w.Z,w.B,w.sigma2,iterations=100,ranks=[r])[0])
                    e,te=call(lambda:solver.fit(w.S,w.Z,w.B,w.sigma2,iterations=100)[0])
                    _,tp=call(lambda:solver.project(e[None],r,4)[0])
                rows.append(dict(N=n,trial=trial,implementation=mode,rank=r,
                    select_seconds=ts,fit_seconds=th,em_seconds=te,post4_seconds=tp,
                    total_interleaved_seconds=ts+th,total_final_seconds=ts+te+tp,
                    selection_fraction=ts/(ts+th)))
                print(rows[-1],flush=True)
                (ROOT/"results/runtime.json").write_text(json.dumps(dict(platform=platform.platform(),
                    N_values=[8,16,32,64],P=20,K=3,RSR=10,SNR=5,T=100,selection_iterations=25,
                    sweeps=4,repetitions=3,rows=rows,complete=len(rows)==24),indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8,"legend.fontsize":7,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.8),layout="constrained")
    ns=(8,16,32,64)
    for mode,col in (("production","#2864B4"),("vectorized","#C5443D")):
        med=lambda key:[float(np.median([r[key] for r in rows if r["N"]==n and r["implementation"]==mode])) for n in ns]
        ax[0].plot(ns,med("total_interleaved_seconds"),"o-",color=col,label=mode+", interleaved")
        ax[0].plot(ns,med("total_final_seconds"),"s--",color=col,label=mode+", final-only")
        ax[1].plot(ns,np.array(med("selection_fraction"))*100,"o-",color=col,label=mode)
    ax[0].set(yscale="log",xlabel="Array size N",ylabel="Median end-to-end time (seconds)",title="(a) Rank selection included; final-only uses 4 sweeps")
    ax[1].set(xlabel="Array size N",ylabel="Rank-selection share of total time (%)",title="(b) Interleaved estimator; three trials per point")
    for a in ax:a.set_xticks(ns);a.grid(alpha=.2);a.legend()
    for ext in ("png","pdf","svg"):fig.savefig(ROOT/"publication"/f"runtime.{ext}",dpi=300,bbox_inches="tight")
    path=ROOT/"publication/runtime.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)

if __name__=="__main__":main()
