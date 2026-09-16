"""Summarize fresh trial stores and produce publication-style plots.

No old result array is used. Historical numbers are explicit comparison
targets, never inputs to an estimator, stopping rule, or new fitted curve.
"""
import csv,json
from pathlib import Path
import numpy as np
from codex_verification.run import cells,OUT,source_hash

ROOT=Path(__file__).resolve().parent
PUB=ROOT/"publication"
SNRS=(-5,0,5,10,15,20)


def ci(x,kind="median",b=None,seed=20260915):
    x=np.asarray(x);rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(x),(4000,len(x)))
    if kind=="median":v=np.median(x[ix],axis=1)
    elif kind=="ratio":v=10*np.log10(x[ix].sum(axis=1)/np.asarray(b)[ix].sum(axis=1))
    elif kind=="rho":v=np.median(x[ix].reshape(len(ix),-1),axis=1)
    else:raise ValueError(kind)
    return [float(v) for v in np.percentile(v,[2.5,97.5])]


def load(c):
    merged=OUT/"trials"/(c["tag"]+".npz")
    if merged.exists():
        with np.load(merged) as f:
            assert str(f["fingerprint"])==source_hash()
            d={k:f[k] for k in f.files if k!="fingerprint"}
        assert len(d["trial"])==c["n"]
        np.testing.assert_array_equal(d["trial"],c["seed_base"]+np.arange(c["n"]))
        return d
    fs=sorted((OUT/"raw"/c["tag"]).glob("*.npz")); parts=[]
    for f in fs:
        with np.load(f) as d:
            assert str(d["fingerprint"])==source_hash()
            parts.append({k:d[k] for k in d.files if k not in ("fingerprint","count")})
    d={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
    assert len(d["trial"])==c["n"],(c["tag"],len(d["trial"]),c["n"])
    np.testing.assert_array_equal(d["trial"],c["seed_base"]+np.arange(c["n"]))
    return d


def summary(c,d):
    delta=10*np.log10(d["num_em"]/d["num_hs"])
    r=dict(tag=c["tag"],N=c["N"],K=c["K"],P=c["P"],L=c["L"],n=len(delta),
           snr=c["snr"],em_db=float(10*np.log10(d["num_em"].sum()/d["den"].sum())),
           hs_db=float(10*np.log10(d["num_hs"].sum()/d["den"].sum())),
           median_gain=float(np.median(delta)),median_ci=ci(delta),
           ratio_gain=float(10*np.log10(d["num_em"].sum()/d["num_hs"].sum())),
           ratio_ci=ci(d["num_em"],"ratio",d["num_hs"]),
           rho=float(np.median(d["rho_columns"])),rho_ci=ci(d["rho_columns"],"rho"),
           inactive_fraction=float(np.mean(d["rank"]==(c["N"]+1)//2)))
    if c["snr"] is None:
        r["bins"]=[]
        for low,high in zip(range(-10,20,5),range(-5,25,5)):
            mask=(d["snr"]>=low)&(d["snr"]<(high if high<20 else 20.001))
            x=delta[mask];r["bins"].append(dict(low=low,high=high,n=len(x),
                median=float(np.median(x)),ci=ci(x)))
        x=delta[d["snr"]>=5];r["high_snr"]=dict(n=len(x),median=float(np.median(x)),ci=ci(x))
    if c["forced"]:
        x=10*np.log10(d["num_em"]/d["num_forced"])
        r["forced_gain"]=float(np.median(x));r["forced_ci"]=ci(x)
    if c["placement"]:
        for name in ("post1","post4","inter1_sharedrank"):
            x=10*np.log10(d["num_"+name]/d["num_hs"])
            r[name+"_contrast"]=float(np.median(x));r[name+"_ci"]=ci(x)
        x=10*np.log10(d["num_post1"]/d["num_inter1_sharedrank"])
        r["matching1_contrast"]=float(np.median(x));r["matching1_ci"]=ci(x)
        for name in ("post1","post4"):r["q_"+name]=float(np.median(d["q_"+name]))
    return r


def crossing(rows,metric):
    pts=sorted([(r["rho"],r[metric]) for r in rows])
    for (x,y),(xx,yy) in zip(pts,pts[1:]):
        if y>0 and yy<0:return dict(value=x+(xx-x)*y/(y-yy),kind="bracketed positive/negative")
        if y>0 and yy==0:return dict(value=xx,kind="first zero; no negative bracket")
    (x,y),(xx,yy)=pts[-2:]
    return dict(value=None if y==yy else x+(xx-x)*y/(y-yy),kind="extrapolated, not observed")


def save(fig,name):
    for ext in ("png","pdf","svg"):
        path=PUB/f"{name}.{ext}"
        fig.savefig(path,dpi=300,bbox_inches="tight")
        if ext=="svg":path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    import matplotlib.pyplot as plt
    plt.close(fig)


def errplot(ax,x,y,intervals,**kw):
    e=np.asarray(intervals).T
    ax.errorbar(x,y,yerr=np.maximum(0,np.vstack([np.asarray(y)-e[0],e[1]-np.asarray(y)])),
                capsize=2,**kw)


def plots(rs,ds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
        "axes.labelsize":8,"axes.titlesize":9,"legend.fontsize":7,
        "lines.linewidth":1.4,"lines.markersize":4,"axes.spines.top":False,
        "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,
        "axes.grid":True,"grid.alpha":.22,"savefig.facecolor":"white"})
    blue,red,green,purple="#2864B4","#C5443D","#26866D","#8053A3"
    head=[rs[f"array_N32_P30_S{s:+d}"] for s in SNRS]
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.85),layout="constrained")
    for key,label,color,style in [("em_db","EM-GS",blue,"s--"),("hs_db","HS-GS",red,"o-"),
        ("crlb_db","Rician CRLB","#777777",":"),("ccrb_db","Geometric CCRB","black","-")]:
        ax[0].plot(SNRS,[r[key] for r in head],style,color=color,label=label)
    ax[0].set(xlabel="Aggregate SNR (dB)",ylabel="Ratio-of-sums NMSE (dB)",title="(a) N = 32, P = 30; 400 paired trials/point")
    ax[0].legend(loc="lower left")
    for p,color in ((10,red),(30,purple)):
        for metric,style,lab in [("median_gain","o-","median"),("ratio_gain","^--","ratio of sums")]:
            y=[np.mean([rs[f"array_N{n}_P{p}_S{s:+d}"][metric] for s in SNRS]) for n in (8,16,32)]
            ax[1].plot((8,16,32),y,style,color=color,label=f"P = {p}, {lab}")
    ax[1].axhline(0,color="gray",lw=.7);ax[1].set(xticks=[8,16,32],xlabel="Array size N",ylabel="Mean gain over six SNR points (dB)",title="(b) Array-size comparison, RSR = 12 dB")
    ax[1].legend(loc="upper left");save(fig,"figure2_performance_array")
    path=[rs[f"path_L{l}"] for l in range(2,17,2)]
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.8),layout="constrained")
    for metric,interval,label,col in [("ratio_gain","ratio_ci","Ratio of sums",blue),("median_gain","median_ci","Paired median",red)]:
        errplot(ax[0],[r["L"] for r in path],[r[metric] for r in path],[r[interval] for r in path],fmt="o-",color=col,label=label)
    ax[0].axhline(0,color="gray",lw=.7);ax[0].set(xlabel="Paths per user L",ylabel="HS-GS gain (dB)",title="(a) N = 32, SNR = 5 dB; 300 trials/point");ax[0].legend()
    ax[1].plot([r["L"] for r in path],[r["em_db"] for r in path],"s--",color=blue,label="EM-GS")
    ax[1].plot([r["L"] for r in path],[r["hs_db"] for r in path],"o-",color=red,label="HS-GS")
    ax[1].set(xlabel="Paths per user L",ylabel="Ratio-of-sums NMSE (dB)",title="(b) Absolute error, P = 30, RSR = 12 dB");ax[1].legend();save(fig,"path_count")
    for matched in (False,True):
        fig,ax=plt.subplots(figsize=(4.7,3.25),layout="constrained")
        for n,ls,color in [(16,(2,4,7,9,14),blue),(32,tuple(range(2,17,2)),"#454545"),(64,(8,14,29,38,48),red)]:
            prefix="rho_N"+str(n)+"_L" if n!=32 else ("matchedrho_N32_L" if matched else "path_L")
            rr=[rs[prefix+str(l)] for l in ls]
            metric,interval=("ratio_gain","ratio_ci") if n==32 and not matched else ("median_gain","median_ci")
            label=f"N = {n}"+(" (fixed 5 dB; ratio of sums)" if n==32 and not matched else " (pooled median)")
            errplot(ax,[r["rho"] for r in rr],[r[metric] for r in rr],[r[interval] for r in rr],fmt="o--" if n==32 and not matched else "o-",color=color,label=label)
        ax.axhline(0,color="gray",lw=.7);ax.set(xlabel=r"Normalized effective rank $\rho$",ylabel="HS-GS gain (dB)",title="Matched configuration: P = 20, RSR = 10 dB" if matched else "Original Figure 3 design: configurations differ")
        ax.legend();save(fig,"figure3_matched" if matched else "figure3_original_design")
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.8),layout="constrained")
    names=["distribution_clustered","distribution_ten_path"]; rr=[rs[k] for k in names]
    errplot(ax[0],[0,1],[r["median_gain"] for r in rr],[r["median_ci"] for r in rr],fmt="o",color=blue,label="Fresh trials, 95% CI")
    ax[0].scatter([0,1],[1.30,.648],marker="x",color=red,label="Manuscript prediction",s=40)
    ax[0].set(xticks=[0,1],xticklabels=["Clustered (4 × 10)","10 independent paths"],ylabel="Pooled median gain (dB)",title="(a) 400 new trials per distribution");ax[0].legend()
    for tag,color,label in [(names[0],blue,"Clustered"),(names[1],red,"10 paths")]:
        bins=rs[tag]["bins"];errplot(ax[1],[b["low"]+2.5 for b in bins],[b["median"] for b in bins],[b["ci"] for b in bins],fmt="o-",color=color,label=label)
    ax[1].set(xlabel="Aggregate SNR bin center (dB)",ylabel="Paired median gain (dB)",title="(b) Dependence on SNR");ax[1].legend();save(fig,"distribution_transfer")
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.8),layout="constrained")
    rr=[rs[f"users_K{k}"] for k in (2,3,4)]
    errplot(ax[0],[2,3,4],[r["high_snr"]["median"] for r in rr],[r["high_snr"]["ci"] for r in rr],fmt="o-",color=blue)
    ax[0].set(xticks=[2,3,4],xlabel="Users K (P = 13, 20, 27)",ylabel="Paired median gain (dB)",title="(a) SNR ≥ 5 dB; fixed L = 5")
    rr=[rs[f"rho_N16_L{l}"] for l in (7,9,14)]
    for key,interval,label,color in [("median_gain","median_ci","Adaptive HS-GS",blue),("forced_gain","forced_ci","Forced rank 7",red)]:
        errplot(ax[1],[r["rho"] for r in rr],[r[key] for r in rr],[r[interval] for r in rr],fmt="o-",color=color,label=label)
    ax[1].axhline(0,color="gray",lw=.7);ax[1].set(xlabel=r"Normalized effective rank $\rho$",ylabel="Pooled median gain (dB)",title="(b) N = 16; forced constraint diagnostic");ax[1].legend();save(fig,"users_and_forced_constraint")
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.85),layout="constrained")
    rr=[rs[f"placement_S{s:+d}"] for s in SNRS]
    for key,interval,label,color in [("post1_contrast","post1_ci","Interleaved 4 vs final 1",purple),("post4_contrast","post4_ci","Interleaved 4 vs final 4",blue),("matching1_contrast","matching1_ci","Interleaved 1 vs final 1",red)]:
        errplot(ax[0],SNRS,[r[key] for r in rr],[r[interval] for r in rr],fmt="o-",color=color,label=label)
    ax[0].axhline(0,color="gray",lw=.7);ax[0].set(xlabel="Aggregate SNR (dB)",ylabel="Gain of interleaving over final-only (dB)",title="(a) Paired medians; shared selected rank");ax[0].legend()
    for key,label,color in [("q_post1","Final 1 sweep",purple),("q_post4","Final 4 sweeps",blue)]:
        ax[1].plot(SNRS,[r[key] for r in rr],"o-",color=color,label=label)
    ax[1].set(xlabel="Aggregate SNR (dB)",ylabel="Median estimate distance / channel norm",title="(b) Interleaved-4 versus final-only estimates");ax[1].legend();save(fig,"projection_placement")


def main():
    assert (OUT/"COMPLETE.json").exists(),"Full campaign must finish before final plots"
    PUB.mkdir(exist_ok=True)
    cs=cells();ds={c["tag"]:load(c) for c in cs};rs={c["tag"]:summary(c,ds[c["tag"]]) for c in cs}
    for s in SNRS:
        tag=f"array_N32_P30_S{s:+d}"
        with np.load(OUT/"bounds"/(tag+".npz")) as f:
            a=f["rows"];np.testing.assert_array_equal(a[:,0],ds[tag]["trial"])
            np.testing.assert_allclose(a[:,3],ds[tag]["den"],rtol=1e-13)
            rs[tag]["crlb_db"]=float(10*np.log10(a[:,1].sum()/a[:,3].sum()))
            rs[tag]["ccrb_db"]=float(10*np.log10(a[:,2].sum()/a[:,3].sum()))
    cross={}
    for n,ls in [(16,(2,4,7)),(64,(8,14,29))]:
        cross[str(n)]=crossing([rs[f"rho_N{n}_L{l}"] for l in ls],"median_gain")
    cross["32"]=crossing([rs[f"path_L{l}"] for l in range(2,17,2)],"ratio_gain")
    array=[]
    for n in (8,16,32):
        rr=[rs[f"array_N{n}_P{p}_S{s:+d}"] for p in (10,30) for s in SNRS]
        array.append(dict(N=n,**{key:float(np.mean([r[key] for r in rr])) for key in ("median_gain","ratio_gain","em_db","inactive_fraction")}))
    result=dict(cells=rs,array_summary=array,original_design_crossings=cross,
                all_trials=sum(c["n"] for c in cs),bootstrap="4000 paired resamples; pointwise 95% percentile CIs")
    (PUB/"verification_results.json").write_text(json.dumps(result,indent=2))
    scalar_keys=[k for k,v in next(iter(rs.values())).items() if not isinstance(v,(dict,list))]
    with (PUB/"cell_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=scalar_keys,extrasaction="ignore",lineterminator="\n");w.writeheader();w.writerows(rs.values())
    plots(rs,ds)
    print(json.dumps({"array":array,"crossings":cross,"total":result["all_trials"]},indent=2))

if __name__=="__main__":main()
