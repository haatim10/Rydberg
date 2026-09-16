"""Replay representative stored trials through the unmodified estimator."""
import json
import numpy as np
from codex_verification.run import cells,world,OUT
from codex_verification.analyze import load
from rydberg_sim.gs import em_gs_channel_rows
from rydberg_sim.track_b_proposed import hs_gs,select_order_heldout,cadzow_project

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    cs={c["tag"]:c for c in cells()};rows=[]
    for tag in ("path_L16","rho_N64_L29","rho_N64_L48",
                "distribution_clustered","users_K4","placement_S+5"):
        c=cs[tag];w=world(c,0);d=load(c)
        rank,scores=select_order_heldout(w.S,w.Z,w.B,w.sigma2,max_iter=c["select_iterations"])
        assert rank==d["rank"][0]
        np.testing.assert_allclose(list(scores.values()),d["scores"][0],rtol=1e-8,atol=1e-8)
        e=em_gs_channel_rows(w.S,w.Z,w.B,w.sigma2,max_iter=c["iterations"]).G_hat
        h=hs_gs(w.S,w.Z,w.B,w.sigma2,L_hat=rank,max_iter=c["iterations"]).G_hat
        es={"em":e,"hs":h}
        if c["placement"]:
            for count in (1,4):es[f"post{count}"]=np.stack([cadzow_project(e[:,k],rank,n_iter=count) for k in range(c["K"])],axis=1)
            es["inter1_sharedrank"]=hs_gs(w.S,w.Z,w.B,w.sigma2,L_hat=rank,max_iter=c["iterations"],cadzow_iter=1).G_hat
        errors={}
        for name,g in es.items():
            num=float(np.sum(abs(g-w.G)**2));old=float(d["num_"+name][0])
            np.testing.assert_allclose(num,old,rtol=1e-8,atol=1e-10)
            errors[name]=abs(num-old)/max(abs(old),1e-30)
        rows.append(dict(tag=tag,trial=int(d["trial"][0]),rank=rank,relative_error_differences=errors))
        print(rows[-1],flush=True)
    (OUT/"stored_trial_spotchecks.json").write_text(json.dumps(dict(passed=True,rows=rows),indent=2))

if __name__=="__main__":main()
