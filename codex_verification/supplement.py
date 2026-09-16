"""Secondary checks using only the completed fresh campaign."""
import json
import numpy as np
from codex_verification.run import cells,OUT
from codex_verification.analyze import load,PUB


def linear(x,y,q):
    """Linear interpolation, explicitly linear extrapolation at the ends."""
    x=np.asarray(x);y=np.asarray(y)
    i=int(np.clip(np.searchsorted(x,q)-1,0,len(x)-2))
    return float(y[i]+(y[i+1]-y[i])*(q-x[i])/(x[i+1]-x[i]))


def main():
    cs={c["tag"]:c for c in cells()}
    rr=json.loads((PUB/"verification_results.json").read_text())["cells"]
    rng=np.random.default_rng(20260916);b=3000
    tags=[f"matchedrho_N32_L{l}" for l in range(2,17,2)]
    targets=[f"rho_N{n}_L{l}" for n,ls in [(16,(2,4,7)),(64,(8,14,29))] for l in ls]
    bs={};data={}
    for tag in tags+targets+[f"placement_S{s:+d}" for s in (-5,0,5,10,15,20)]:
        d=load(cs[tag]);data[tag]=d
        ix=rng.integers(0,len(d["trial"]),(b,len(d["trial"])))
        delta=10*np.log10(d["num_em"]/d["num_hs"])
        bs[tag]=dict(gain=np.median(delta[ix],axis=1),
                     rho=np.median(d["rho_columns"][ix].reshape(b,-1),axis=1))
        if tag.startswith("placement"):
            for name,a,z in [("original","post1","hs"),("matching4","post4","hs"),
                             ("matching1","post1","inter1_sharedrank")]:
                bs[tag][name]=np.median((10*np.log10(d["num_"+a]/d["num_"+z]))[ix],axis=1)
    placement={}
    pts=[f"placement_S{s:+d}" for s in (-5,0,5,10,15,20)]
    for name,key in [("original","post1_contrast"),("matching4","post4_contrast"),("matching1","matching1_contrast")]:
        v=np.mean([bs[t][name] for t in pts],axis=0)
        placement[name]=dict(mean_of_six_medians=float(np.mean([rr[t][key] for t in pts])),
                            ci95=np.percentile(v,[2.5,97.5]).tolist())
    xt={"raw_L":np.array([cs[t]["L"] for t in tags]),
        "L_over_cap":np.array([cs[t]["L"]/16 for t in tags]),
        "rho":np.array([rr[t]["rho"] for t in tags])}
    y=np.array([rr[t]["median_gain"] for t in tags])
    predictions=[];errors={key:[] for key in xt};boot_errors={key:[] for key in xt}
    for t in targets:
        c=cs[t];qs=dict(raw_L=c["L"],L_over_cap=c["L"]/(c["N"]/2),rho=rr[t]["rho"])
        pred={key:linear(xt[key],y,q) for key,q in qs.items()}
        predictions.append(dict(tag=t,measured=rr[t]["median_gain"],predictions=pred,
            extrapolated={k:bool(q<xt[k].min() or q>xt[k].max()) for k,q in qs.items()}))
        for key,q in qs.items():
            errors[key].append(abs(pred[key]-rr[t]["median_gain"]))
            vals=[]
            for i in range(b):
                xx=np.array([bs[a]["rho"][i] for a in tags]) if key=="rho" else xt[key]
                yy=np.array([bs[a]["gain"][i] for a in tags])
                qq=bs[t]["rho"][i] if key=="rho" else q
                vals.append(abs(linear(xx,yy,qq)-bs[t]["gain"][i]))
            boot_errors[key].append(vals)
    mae={key:float(np.mean(v)) for key,v in errors.items()}
    diff=np.mean(boot_errors["rho"],axis=0)-np.mean(boot_errors["L_over_cap"],axis=0)
    result=dict(placement=placement,matched_prediction=dict(
        calibration="Fresh N32, P20, RSR10, T100, selection25, random SNR, pooled median",
        targets="Six original N16/N64 cells, same design and statistic",
        method="Piecewise linear interpolation; explicit linear endpoint extrapolation, no target fitting",
        MAE_db=mae,MAE_rho_minus_normalized_L_ci95=np.percentile(diff,[2.5,97.5]).tolist(),
        rows=predictions,warning="All six target gains are positive; sign accuracy cannot validate failure detection."))
    (PUB/"supplemental_statistics.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
