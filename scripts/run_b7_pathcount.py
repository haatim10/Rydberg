"""B7 -- controlled path count L. Direct test of the structural assumption.

Every other Track-B experiment draws L_k ~ U{3..7} at random, so nothing so
far isolates the effect of path count. Here L_k = L is FIXED and identical
for all users, and L is swept from very sparse up to the Hankel rank cap.

Hypothesis under test (stated before running, not after):
    L increases -> Hankel rank increases -> the strict low-rank prior carries
    less information -> the HS-GS advantage should shrink, and should vanish
    once L reaches cap(N) = ceil(N/2), where the constraint is vacuous.

N = 32 (cap 16), P = 30, SNR = 5 dB, RSR = 12 dB, 400 trials/point, same
CRN world function, same estimators, same hyperparameters as B3.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
import run_b3 as rb3
from rydberg_sim.gs import biased_gs_channel_rows, em_gs_channel_rows
from rydberg_sim.monte_carlo import generate_channel_estimation_trial
from rydberg_sim.track_b_drivers import TRACK_B_K, TRACK_B_RSR_DB, track_b_spec
from rydberg_sim.track_b_proposed import hankel_rank_cap, hs_gs_auto

B7_N, B7_P, B7_SNR = 32, 30, 5.0
B7_L = (2, 4, 6, 8, 10, 12, 14, 16)          # up to cap(32) = 16
STORE = REPO/"results"/"track_b"/"b7"

def world(t, L):
    sp = track_b_spec(P=B7_P, n_trials=t+1, N=B7_N, K=TRACK_B_K,
                      L=(L,)*TRACK_B_K, experiment="track_b_b7")
    return generate_channel_estimation_trial(sp, t, B7_SNR, TRACK_B_RSR_DB)

def run_point(args):
    L, nt = args
    path = STORE/f"L{L:02d}.npz"
    rb3.OUT = STORE
    d = rb3.load_point(path, TRACK_B_K)
    have = {int(x) for x in d["trial"]}
    todo = [t for t in range(nt) if t not in have]
    if not todo: return f"L={L}: complete ({len(have)})"
    t0=time.time()
    buf={k:[] for k in ("trial","denom","L_hat","active","L_true")}
    buf.update({f"num_{e}":[] for e in rb3.ESTIMATORS})
    def flush():
        if not buf["trial"]: return
        for k in list(d):
            new=np.asarray(buf[k],dtype=d[k].dtype).reshape((len(buf[k]),)+d[k].shape[1:])
            d[k]=np.concatenate([d[k],new],axis=0); buf[k]=[]
        rb3.save_point(path,d)
    for i,t in enumerate(todo):
        w=world(t,L)
        G={"biased_gs":biased_gs_channel_rows(w.S,w.Z,w.B,max_iter=rb3.GS_MAX_ITER).G_hat,
           "em_gs":em_gs_channel_rows(w.S,w.Z,w.B,w.sigma2,max_iter=rb3.GS_MAX_ITER).G_hat}
        r=hs_gs_auto(w.S,w.Z,w.B,w.sigma2,**rb3.HS_KW)
        assert not r.linearised_model_used
        G["hs_gs"]=r.G_hat
        buf["trial"].append(t); buf["denom"].append(float(np.linalg.norm(w.G,"fro")**2))
        buf["L_hat"].append(int(r.L_hat)); buf["active"].append(bool(r.constraint_active))
        buf["L_true"].append([int(v) for v in w.L_k])
        for e in rb3.ESTIMATORS: buf[f"num_{e}"].append(float(np.sum(np.abs(G[e]-w.G)**2)))
        if (i+1)%rb3.CHUNK==0: flush()
    flush()
    return f"L={L}: +{len(todo)} trials ({(time.time()-t0)/60:.1f} min)"

def main():
    import multiprocessing as mp
    nt=int(os.environ.get("B7_TRIALS","400"))
    STORE.mkdir(parents=True,exist_ok=True)
    (STORE/"config.json").write_text(json.dumps({
        "fingerprint":rb3.FP,"N":B7_N,"P":B7_P,"snr_db":B7_SNR,
        "rsr_db":TRACK_B_RSR_DB,"L_grid":B7_L,"n_trials":nt,
        "rank_cap":hankel_rank_cap(B7_N),"K":TRACK_B_K,
        "note":"L_k FIXED and identical for all users, unlike B3-B6 where "
               "L_k ~ U{3..7} at random. Direct test of the structural "
               "assumption. Hypothesis stated before running.",
        "observation":"EXACT Z = |G S + B + W|, no linearization"},indent=2))
    print(f"B7: N={B7_N} cap={hankel_rank_cap(B7_N)}, L in {B7_L}, {nt}/pt",flush=True)
    with mp.Pool(int(os.environ.get("B7_PROCS","4"))) as pool:
        for m in pool.imap_unordered(run_point,[(L,nt) for L in B7_L]): print("  ",m,flush=True)
    print("B7 done",flush=True)

if __name__=="__main__": main()
