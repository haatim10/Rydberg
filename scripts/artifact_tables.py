"""Machine-readable result tables (CSV) for every artifact figure."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO))
TB=REPO/"results/track_b"; OUT=TB/"artifact"; OUT.mkdir(parents=True,exist_ok=True)
S3=json.loads((TB/"b3/summary.json").read_text())
S4=json.loads((TB/"b4/summary.json").read_text())
S6=json.loads((TB/"b6/summary.json").read_text())
CC=json.loads((TB/"constrained_crlb.json").read_text())
def w(name,hdr,rows):
    with open(OUT/name,"w",newline="") as f:
        c=csv.writer(f); c.writerow(hdr); c.writerows(rows)
    print(f"  {name}  ({len(rows)} rows)")

# B3 master table
rows=[]
for r in sorted(S3,key=lambda r:(r["N"],r["P"],r["snr_db"])):
    k=f"N{r['N']}_P{r['P']}_snr{r['snr_db']:+.0f}"; p=r["pooled_db"]; m=r["median_per_trial_db"]
    rows.append([r["N"],r["P"],r["snr_db"],r["n_trials"],
        f"{p['biased_gs']:.4f}",f"{p['em_gs']:.4f}",f"{p['hs_gs']:.4f}",
        f"{CC['unconstrained_rank1']['b3'][k]:.4f}",f"{CC['constrained']['b3'][k]:.4f}",
        f"{r['gain_hs_vs_em_db']:.4f}",f"{r['gain_ci95_db'][0]:.4f}",f"{r['gain_ci95_db'][1]:.4f}",
        f"{r['win_rate_vs_em']:.4f}",f"{r['tie_frac']:.4f}",f"{r['constraint_active_frac']:.4f}",
        f"{r['mean_L_hat']:.3f}",f"{r['mean_sum_L_true']:.3f}",
        f"{m['biased_gs']:.4f}",f"{m['em_gs']:.4f}",f"{m['hs_gs']:.4f}",
        f"{r['gain_trimmed95_db']:.4f}",f"{r['em_worst_share']:.4f}"])
w("table_B3_nmse_vs_snr.csv",["N","P","SNR_dB","trials","GS_dB","EMGS_dB","HSGS_dB",
    "uncon_CRLB_dB","con_CRLB_dB","gain_dB","gain_ci_lo","gain_ci_hi","win_rate",
    "tie_frac","active_frac","mean_Lhat","mean_sumL","med_GS_dB","med_EMGS_dB",
    "med_HSGS_dB","gain_trim95_dB","EMGS_worst_share"],rows)

# B4
rows=[]
for r in sorted(S4,key=lambda r:r["P"]):
    p=r["pooled_db"]
    rows.append([r["P"],r["n_trials"],f"{p['biased_gs']:.4f}",f"{p['em_gs']:.4f}",
        f"{p['hs_gs']:.4f}",f"{CC['unconstrained_rank1']['b4'][f'P{r[chr(34)+chr(80)+chr(34)]}' if False else 'P'+str(r['P'])]:.4f}",
        f"{CC['constrained']['b4']['P'+str(r['P'])]:.4f}",
        f"{r['gain_hs_vs_em_db']:.4f}",f"{r['gain_ci95_db'][0]:.4f}",f"{r['gain_ci95_db'][1]:.4f}",
        f"{r['win_rate_vs_em']:.4f}",f"{r['constraint_active_frac']:.4f}",f"{r['mean_L_hat']:.3f}"])
w("table_B4_nmse_vs_pilots.csv",["P","trials","GS_dB","EMGS_dB","HSGS_dB","uncon_CRLB_dB",
    "con_CRLB_dB","gain_dB","gain_ci_lo","gain_ci_hi","win_rate","active_frac","mean_Lhat"],rows)

# B6
rows=[]
for r in sorted(S6,key=lambda r:(r["N"],r["rsr_db"])):
    k=f"N{r['N']}_P30_snr+5_rsr{r['rsr_db']:+.0f}"; p=r["pooled_db"]
    rows.append([r["N"],r["rsr_db"],r["n_trials"],f"{p['biased_gs']:.4f}",f"{p['em_gs']:.4f}",
        f"{p['hs_gs']:.4f}",f"{CC['unconstrained_rank1']['b6'][k]:.4f}",
        f"{CC['constrained']['b6'][k]:.4f}",f"{r['gain_hs_vs_em_db']:.4f}",
        f"{r['gain_ci95_db'][0]:.4f}",f"{r['gain_ci95_db'][1]:.4f}",
        f"{p['biased_gs']-p['em_gs']:.4f}",f"{r['win_rate_vs_em']:.4f}",
        f"{r['constraint_active_frac']:.4f}",f"{r['mean_L_hat']:.3f}"])
w("table_B6_nmse_vs_rsr.csv",["N","RSR_dB","trials","GS_dB","EMGS_dB","HSGS_dB",
    "uncon_CRLB_dB","con_CRLB_dB","gain_dB","gain_ci_lo","gain_ci_hi",
    "EMGS_advantage_over_GS_dB","win_rate","active_frac","mean_Lhat"],rows)

# scaling summary
rows=[]
for N in (8,16,32):
    sub=[r for r in S3 if r["N"]==N]
    rows.append([N,int(np.ceil(N/2)),f"{np.mean([r['gain_hs_vs_em_db'] for r in sub]):.4f}",
        f"{np.mean([r['gain_hs_vs_em_db'] for r in sub if r['P']==10]):.4f}",
        f"{np.mean([r['gain_hs_vs_em_db'] for r in sub if r['P']==30]):.4f}",
        f"{np.mean([r['win_rate_vs_em'] for r in sub]):.4f}",
        f"{np.mean([r['constraint_active_frac'] for r in sub]):.4f}",
        f"{np.mean([r['pooled_db']['em_gs'] for r in sub]):.4f}",
        f"{np.mean([r['pooled_db']['hs_gs'] for r in sub]):.4f}"])
w("table_B5_scaling_vs_N.csv",["N","rank_cap","mean_gain_dB","gain_P10_dB","gain_P30_dB",
    "win_rate","active_frac","mean_EMGS_dB","mean_HSGS_dB"],rows)
print("tables written to", OUT)
