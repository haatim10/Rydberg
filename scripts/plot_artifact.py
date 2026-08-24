"""Figures for the Track-B research artifact. Plain style, no decoration.

Simple: default white background, straightforward lines, readable markers,
units on every axis, concise legends. Nothing is smoothed or interpolated.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO))
TB=REPO/"results/track_b"; OUT=TB/"artifact"
plt.rcParams.update({"font.size":10,"axes.labelsize":10,"legend.fontsize":9,
    "xtick.labelsize":9,"ytick.labelsize":9,"lines.linewidth":1.5,
    "lines.markersize":5,"figure.dpi":300,"savefig.dpi":300,
    "savefig.bbox":"tight","axes.grid":True,"grid.alpha":0.3})
GS=dict(c="tab:orange",marker="o",label="GS")
EM=dict(c="tab:red",marker="s",label="EM-GS")
HS=dict(c="tab:blue",marker="^",label="HS-GS")
UN=dict(c="tab:green",ls="--",label="Unconstrained CRLB")
CO=dict(c="tab:purple",ls=":",label="Constrained CRLB")

def save(fig,stem):
    OUT.mkdir(parents=True,exist_ok=True)
    for e in ("png","pdf"): fig.savefig(OUT/f"{stem}.{e}")
    plt.close(fig); print(f"  {stem}")

S3=json.loads((TB/"b3/summary.json").read_text())
S4=json.loads((TB/"b4/summary.json").read_text())
S6=json.loads((TB/"b6/summary.json").read_text())
CC=json.loads((TB/"constrained_crlb.json").read_text())

# ---- A1: NMSE vs SNR, baselines + CRLB (N=8, the frozen baseline size) ----
def A1():
    fig,ax=plt.subplots(figsize=(5.2,3.8))
    sub=sorted((r for r in S3 if r["N"]==8 and r["P"]==30),key=lambda r:r["snr_db"])
    x=[r["snr_db"] for r in sub]
    ax.plot(x,[r["pooled_db"]["biased_gs"] for r in sub],**GS)
    ax.plot(x,[r["pooled_db"]["em_gs"] for r in sub],**EM)
    k=[f"N8_P30_snr{s:+.0f}" for s in x]
    ax.plot(x,[CC["unconstrained_rank1"]["b3"][j] for j in k],**UN)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel(r"Channel NMSE$_G$ (dB)")
    ax.set_title("A1: baselines vs SNR ($N$=8, $K$=3, $P$=30, RSR=12 dB)",fontsize=10)
    ax.legend(); fig.tight_layout(); save(fig,"A1_nmse_vs_snr_baselines")

# ---- A2: NMSE vs pilots ----
def A2():
    rows=json.loads((TB/"baseline_preliminary.json").read_text())["rows"]
    sub=[r for r in rows if r["sweep"]=="B2 (SNR=5.0 dB)"]
    fig,ax=plt.subplots(figsize=(5.2,3.8))
    for alg,st in (("biased_gs",GS),("em_gs",EM)):
        p=sorted((r for r in sub if r["algorithm"]==alg),key=lambda r:r["x"])
        ax.plot([q["x"] for q in p],[q["nmse_db"] for q in p],**st)
    ax.axvline(6,c="0.6",ls=":",lw=1); ax.annotate("$P=2K$",(6,ax.get_ylim()[1]),
        xytext=(3,-12),textcoords="offset points",fontsize=8,color="0.4")
    ax.set_xlabel("Pilot length $P$"); ax.set_ylabel(r"Channel NMSE$_G$ (dB)")
    ax.set_title("A2: baselines vs pilot length ($N$=8, SNR=5 dB)",fontsize=10)
    ax.legend(); fig.tight_layout(); save(fig,"A2_nmse_vs_pilots_baselines")

# ---- A3: NMSE vs RSR ----
def A3():
    fig,ax=plt.subplots(figsize=(5.2,3.8))
    sub=sorted((r for r in S6 if r["N"]==8),key=lambda r:r["rsr_db"])
    x=[r["rsr_db"] for r in sub]
    ax.plot(x,[r["pooled_db"]["biased_gs"] for r in sub],**GS)
    ax.plot(x,[r["pooled_db"]["em_gs"] for r in sub],**EM)
    ax.set_xlabel("RSR (dB)"); ax.set_ylabel(r"Channel NMSE$_G$ (dB)")
    ax.set_title("A3: baselines vs reference strength ($N$=8, $P$=30, SNR=5 dB)",fontsize=10)
    ax.legend(); fig.tight_layout(); save(fig,"A3_nmse_vs_rsr_baselines")

# ---- B1: NMSE vs N (PRIMARY proposed-method figure) ----
def B1():
    fig,(a,b)=plt.subplots(1,2,figsize=(9.2,3.8))
    Ns=[8,16,32]
    for ax,P in ((a,10),(b,30)):
        for alg,st in (("em_gs",EM),("hs_gs",HS)):
            y=[np.mean([r["pooled_db"][alg] for r in S3 if r["N"]==N and r["P"]==P])
               for N in Ns]
            ax.plot(Ns,y,**st)
        ax.set_xscale("log",base=2); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
        ax.set_xlabel("Array size $N$"); ax.set_title(f"$P$={P}",fontsize=10)
    a.set_ylabel(r"Channel NMSE$_G$, mean over SNR (dB)"); a.legend()
    fig.suptitle("B1: proposed method vs array size ($K$=3, RSR=12 dB)",fontsize=10)
    fig.tight_layout(); save(fig,"B1_nmse_vs_array_size")
    # the gain itself
    fig,ax=plt.subplots(figsize=(5.2,3.8))
    for P,mk in ((10,"o"),(30,"s")):
        g=[np.mean([r["gain_hs_vs_em_db"] for r in S3 if r["N"]==N and r["P"]==P]) for N in Ns]
        ax.plot(Ns,g,marker=mk,label=f"$P$={P}")
    ax.axhline(0,c="0.5",ls=":",lw=1)
    ax.set_xscale("log",base=2); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
    ax.set_xlabel("Array size $N$")
    ax.set_ylabel(r"$\Delta_{\mathrm{HS}}(N)$ (dB)")
    ax.set_title(r"B1b: $\Delta_{\rm HS}=$NMSE$_{\rm EM\text{-}GS}-$NMSE$_{\rm HS\text{-}GS}$",fontsize=10)
    ax.legend(); fig.tight_layout(); save(fig,"B1b_gain_vs_array_size")

# ---- B2: NMSE vs SNR, EM-GS vs HS-GS at N=32 ----
def B2():
    fig,ax=plt.subplots(figsize=(5.2,3.8))
    sub=sorted((r for r in S3 if r["N"]==32 and r["P"]==30),key=lambda r:r["snr_db"])
    x=[r["snr_db"] for r in sub]
    ax.plot(x,[r["pooled_db"]["em_gs"] for r in sub],**EM)
    ax.plot(x,[r["pooled_db"]["hs_gs"] for r in sub],**HS)
    k=[f"N32_P30_snr{s:+.0f}" for s in x]
    ax.plot(x,[CC["unconstrained_rank1"]["b3"][j] for j in k],**UN)
    ax.plot(x,[CC["constrained"]["b3"][j] for j in k],**CO)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel(r"Channel NMSE$_G$ (dB)")
    ax.set_title("B2: proposed vs SNR ($N$=32, $K$=3, $P$=30, RSR=12 dB)",fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); save(fig,"B2_nmse_vs_snr_proposed")

# ---- B4: win rate vs N ----
def B4():
    fig,ax=plt.subplots(figsize=(5.2,3.8)); Ns=[8,16,32]
    for P,mk in ((10,"o"),(30,"s")):
        w=[100*np.mean([r["win_rate_vs_em"] for r in S3 if r["N"]==N and r["P"]==P]) for N in Ns]
        ax.plot(Ns,w,marker=mk,label=f"$P$={P}")
    ax.axhline(50,c="0.5",ls=":",lw=1)
    ax.set_xscale("log",base=2); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
    ax.set_xlabel("Array size $N$"); ax.set_ylabel("Trials where HS-GS beats EM-GS (%)")
    ax.set_title("B4: per-trial win rate vs array size",fontsize=10)
    ax.legend(); fig.tight_layout(); save(fig,"B4_win_rate_vs_N")

# ---- C1: singular-value spectrum of the Hankel matrix ----
def C1():
    from rydberg_sim.track_b_drivers import track_b_world
    from rydberg_sim.track_b_structure import hankel_matrix
    from rydberg_sim.track_b_proposed import best_pencil, cadzow_project
    from rydberg_sim.gs import em_gs_channel_rows
    N=32; w=track_b_world(3,30,5.0,N=N); p=best_pencil(N); k=0
    Lk=int(w.L_k[k])
    em=em_gs_channel_rows(w.S,w.Z,w.B,w.sigma2,max_iter=50).G_hat
    cz=cadzow_project(em[:,k],Lk)
    fig,ax=plt.subplots(figsize=(5.2,3.8))
    for g,lab,st in ((w.G[:,k],f"true channel ($L_k$={Lk})",dict(c="k",marker="o")),
                     (em[:,k],"EM-GS estimate",dict(c="tab:red",marker="s")),
                     (cz,f"after Cadzow (rank {Lk})",dict(c="tab:blue",marker="^"))):
        s=np.linalg.svd(hankel_matrix(g,p),compute_uv=False)
        ax.semilogy(np.arange(1,len(s)+1),s/s[0],**st,label=lab)
    ax.axvline(Lk+0.5,c="0.6",ls=":",lw=1)
    ax.set_xlabel("Singular value index"); ax.set_ylabel("Normalised singular value")
    ax.set_title(f"C1: Hankel spectrum, $N$=32, one user, SNR=5 dB",fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); save(fig,"C1_hankel_singular_values")

# ---- C3: gap to each bound ----
def C3():
    fig,ax=plt.subplots(figsize=(5.2,3.8))
    for N,c in ((8,"0.5"),(16,"tab:red"),(32,"tab:blue")):
        sub=sorted((r for r in S3 if r["N"]==N and r["P"]==30),key=lambda r:r["snr_db"])
        x=[r["snr_db"] for r in sub]
        ax.plot(x,[r["pooled_db"]["hs_gs"]-CC["constrained"]["b3"][f"N{N}_P30_snr{s:+.0f}"]
                   for r,s in zip(sub,x)],marker="^",c=c,label=f"$N$={N}")
    ax.axhline(0,c="0.5",ls=":",lw=1)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("HS-GS above constrained CRLB (dB)")
    ax.set_title("C3: distance from the structured bound ($P$=30)",fontsize=10)
    ax.legend(); fig.tight_layout(); save(fig,"C3_gap_to_constrained_crlb")

if __name__=="__main__":
    print("Artifact figures:")
    for f in (A1,A2,A3,B1,B2,B4,C1,C3):
        try: f()
        except Exception as e: print(f"  FAILED {f.__name__}: {e}")
