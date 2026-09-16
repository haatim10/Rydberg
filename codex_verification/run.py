"""Fresh, fixed-budget Paper 1 replication, checkpointed by trial chunks.

Run from repository root: python -m codex_verification.run --workers 6
No committed Claude result is read by this simulation driver.
"""
import argparse,hashlib,json,os,platform,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from codex_verification.worlds import make_world
from codex_verification.solver import estimate,effective_ranks

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"


def cells():
    out=[]
    def add(tag,**kw):
        c=dict(tag=tag,N=32,K=3,P=30,snr=5.,rsr=12.,L=None,
               family="TB",channel="ula",iterations=100,select_iterations=25,
               placement=False,forced=False,n=400)
        c.update(kw);out.append(c)
    # Fig. 2(a) is the N32/P30 subset of Fig. 2(b), freshly simulated once.
    for n in (8,16,32):
        for p in (10,30):
            for s in (-5.,0.,5.,10.,15.,20.):
                add(f"array_N{n}_P{p}_S{s:+.0f}",N=n,P=p,snr=s)
    for l in range(2,17,2):
        add(f"path_L{l}",L=l,iterations=50,select_iterations=20,n=300)
    # Figure 3 source design: pooled random SNR for N16/N64; N32 context is
    # path-count sweep above (different config/statistic, labeled in figures).
    for n,ls in [(16,(2,4,7,9,14)),(64,(8,14,29,38,48))]:
        for l in ls:
            add(f"rho_N{n}_L{l}",N=n,L=l,P=20,snr=None,rsr=10.,family="TD",
                n=400 if n==16 else 300,forced=n==16 and l in (7,9,14))
    for channel in ("clustered","ten_path"):
        add(f"distribution_{channel}",channel=channel,P=20,snr=None,
            rsr=10.,family="TD",n=400)
    for k,p in ((2,13),(3,20),(4,27)):
        add(f"users_K{k}",K=k,P=p,L=5,snr=None,rsr=10.,family="TD")
    for s in (-5.,0.,5.,10.,15.,20.):
        add(f"placement_S{s:+.0f}",snr=s,placement=True,n=300)
    for l in range(2,17,2):
        add(f"matchedrho_N32_L{l}",L=l,P=20,snr=None,rsr=10.,family="TD",n=400)
    for i,c in enumerate(out):
        c["seed_base"]=10_000_000+i*10_000
    return out


def source_hash():
    h=hashlib.sha256()
    paths=list(ROOT.glob("*.py"))+list((ROOT.parent/"rydberg_sim").glob("*.py"))
    # Analysis and plot additions do not change simulation provenance.
    paths=[p for p in paths if p.parent!=ROOT or p.name in ("run.py","solver.py","worlds.py")]
    for p in sorted(paths): h.update(str(p.relative_to(ROOT.parent)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def trial_snr(c,t):
    if c["snr"] is not None:return c["snr"]
    rng=np.random.default_rng(np.random.SeedSequence([c["seed_base"],t,0x534e52]))
    return float(np.round(rng.uniform(-10,20),3))


def world(c,t):
    return make_world(c["seed_base"]+t,**{k:c[k] for k in
        ("N","K","P","rsr","L","family","channel")},snr=trial_snr(c,t))


def chunk(c,start,stop,fp):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    path=OUT/"raw"/c["tag"]/f"{start:05d}.npz"
    if path.exists():
        with np.load(path) as f:
            assert str(f["fingerprint"])==fp
            assert int(f["count"])==stop-start
        return c["tag"],stop-start,0.
    t0=time.perf_counter();rows=[]
    for t in range(start,stop):
        w=world(c,t)
        es,r,scores,times=estimate(w,iterations=c["iterations"],
            select_iterations=c["select_iterations"],placement=c["placement"],forced=c["forced"])
        row=dict(trial=c["seed_base"]+t,snr=trial_snr(c,t),den=float(np.sum(abs(w.G)**2)),
                 rank=r,rho_columns=effective_ranks(w.G)/((c["N"]+1)//2),
                 scores=scores,seconds_select=times[0],seconds_joint_fit=times[1])
        for name,g in es.items():row["num_"+name]=float(np.sum(abs(g-w.G)**2))
        if c["placement"]:
            row["q_post1"]=float(np.linalg.norm(es["hs"]-es["post1"])/np.linalg.norm(w.G))
            row["q_post4"]=float(np.linalg.norm(es["hs"]-es["post4"])/np.linalg.norm(w.G))
        if r==(c["N"]+1)//2:
            assert np.array_equal(es["em"],es["hs"]),"Full-rank fallback differs"
        rows.append(row)
    arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    assert all(np.all(np.isfinite(v)) for v in arrays.values()),"nonfinite result"
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp.npz")
    np.savez_compressed(temp,**arrays,fingerprint=fp,count=stop-start)
    temp.replace(path)
    return c["tag"],stop-start,time.perf_counter()-t0


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=6)
    ap.add_argument("--chunk",type=int,default=10)
    a=ap.parse_args()
    assert json.loads((OUT/"validation.json").read_text())["passed"]
    cs=cells();fp=source_hash()
    import scipy
    manifest=dict(source_fingerprint=fp,base_commit="739dba4",cells=cs,
        numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),
        platform=platform.platform(),workers=a.workers,chunk=a.chunk,
        independent_trial_indices=True,snr_definition="aggregate user power / complex noise variance",
        stopping="fixed trial counts, no outcome-dependent stopping",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    OUT.mkdir(exist_ok=True)
    if (OUT/"manifest.json").exists():
        old=json.loads((OUT/"manifest.json").read_text())
        assert old["source_fingerprint"]==fp and old["cells"]==cs
    else:(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    jobs=[(c,t,min(t+a.chunk,c["n"]),fp) for c in cs for t in range(0,c["n"],a.chunk)]
    total=sum(c["n"] for c in cs);done=0;started=time.perf_counter()
    # Interleave cell chunks, so early checkpoints cover every experiment.
    jobs.sort(key=lambda x:x[1])
    print(f"START {len(cs)} cells, {total} fresh paired trials, {a.workers} workers",flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs=[pool.submit(chunk,*j) for j in jobs]
        for f in as_completed(futs):
            tag,n,sec=f.result();done+=n
            status=dict(completed_trials=done,total_trials=total,
                        elapsed_seconds=time.perf_counter()-started,last_cell=tag)
            (OUT/"progress.json").write_text(json.dumps(status))
            print(f"{done}/{total} {tag} chunk_seconds={sec:.1f}",flush=True)
    (OUT/"COMPLETE.json").write_text(json.dumps(status,indent=2))

if __name__=="__main__":main()
