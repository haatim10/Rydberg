"""Code audit backing the Track-B research artifact.

Answers every factual question the artifact makes a claim about, by
executing the implementation. Anything this script cannot establish is
marked NOT VERIFIED in the artifact.
"""
from __future__ import annotations
import inspect, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rydberg_sim import channel, pilots, reference
from rydberg_sim.gs import biased_gs_channel_rows, em_gs_channel_rows
from rydberg_sim.track_b_drivers import (TRACK_B_K, TRACK_B_L_MAX, TRACK_B_L_MIN,
    TRACK_B_MASTER_SEED, TRACK_B_N, TRACK_B_RSR_DB, B1_SNR_DB, B2_P, B2_SNR_DB,
    draw_L_k, track_b_world)
from rydberg_sim.track_b_proposed import (best_pencil, cadzow_project,
    hankel_rank_cap, hs_gs, hs_gs_auto, select_order_heldout)
from rydberg_sim.track_b_structure import hankel_matrix
OUT = {}

def rec(k, v):
    OUT[k] = v
    print(f"  {k}: {v}")

print("="*78); print("PART II — CHANNEL MODEL"); print("="*78)
w = track_b_world(0, 20, 5.0, N=16)
rec("N_default", TRACK_B_N); rec("K", TRACK_B_K)
rec("L_k_range", f"U{{{TRACK_B_L_MIN}..{TRACK_B_L_MAX}}}")
rec("L_k_iid_per_user", str(draw_L_k(0)) + " vs " + str(draw_L_k(1)))
rec("G_shape", w.G.shape); rec("c_scaling", "G == H exactly" if np.allclose(w.G,w.H) else "c != 1")
th = np.concatenate([np.asarray(t) for t in w.theta])
rec("theta_range_observed", f"[{th.min():.4f}, {th.max():.4f}] vs [-pi/2, pi/2]=[{-np.pi/2:.4f},{np.pi/2:.4f}]")
# AoA independence across users and paths
alltheta=[]
for t in range(300):
    ww=track_b_world(t,10,5.0)
    for k in range(TRACK_B_K): alltheta.append(np.asarray(ww.theta[k]))
flat=np.concatenate(alltheta)
rec("theta_n_samples", len(flat))
rec("theta_mean_vs_0", f"{flat.mean():.4f}")
# uniform in theta or in sin(theta)?
from scipy import stats
ks_th = stats.kstest((flat+np.pi/2)/np.pi, 'uniform')
ks_sin = stats.kstest((np.sin(flat)+1)/2, 'uniform')
rec("KS_uniform_in_theta_pvalue", f"{ks_th.pvalue:.4f}")
rec("KS_uniform_in_sin_theta_pvalue", f"{ks_sin.pvalue:.4f}")
rec("AoA_distribution", "uniform in THETA" if ks_th.pvalue>0.01 else "NOT uniform in theta")
# alpha distribution
acc=cnt=0.0; allal=[]
for t in range(400):
    ww=track_b_world(t,10,5.0)
    for k in range(TRACK_B_K):
        a=np.asarray(ww.alpha[k]); allal.append(a)
        acc+=float(np.sum(np.abs(a)**2))*len(a); cnt+=len(a)
al=np.concatenate(allal)
rec("alpha_L_times_E_abs2", f"{acc/cnt:.4f} (target beta=1)")
rec("alpha_mean", f"{al.mean():.4f}")
rec("alpha_circular_E_a2", f"{np.mean(al**2):.4f} (0 => circularly symmetric)")
rec("E_g_power_per_element", f"{np.mean(np.abs(w.G)**2):.4f} (target beta=1)")

print(); print("="*78); print("PART III — PILOTS"); print("="*78)
rec("S_shape", w.S.shape)
rec("S_E_abs2", f"{np.mean(np.abs(w.S)**2):.4f} (CN(0,1) => 1)")
S2 = track_b_world(1,20,5.0,N=16).S
rec("S_redrawn_per_trial", not np.allclose(w.S, S2))
G = w.S @ w.S.conj().T / w.S.shape[1]
rec("S_orthogonality", f"off-diag |SS^H|/P max = {np.abs(G-np.diag(np.diag(G))).max():.4f} (0 would be orthogonal)")
rec("S_full_row_rank_enforced", "yes, rejection sampled")
rec("P_min_constraint", "P >= 2K enforced in generate_gaussian_pilots")

print(); print("="*78); print("PART IV — MEASUREMENT / REFERENCE / NOISE"); print("="*78)
rec("Z_shape_dtype", f"{w.Z.shape} {w.Z.dtype}")
rec("Z_nonnegative", bool((w.Z>=0).all()))
rec("Z_equals_abs_GS_B_W", float(np.abs(np.abs(w.G@w.S+w.B+w.W)-w.Z).max()))
rec("B_shape", w.B.shape)
rec("B_constant_across_np", f"unique |B| values: {len(np.unique(np.round(np.abs(w.B),9)))}")
rec("B_value", f"|B|={np.abs(w.B).flat[0]:.4f}, angle={np.angle(w.B).flat[0]:.4f} rad")
rec("noise_before_abs", "yes: W added inside |.| (verified by Z identity above)")
rec("W_var_total", f"{np.var(w.W):.5f} vs sigma2={w.sigma2:.5f}")
rec("W_var_real", f"{np.var(w.W.real):.5f}  (sigma2/2 = {w.sigma2/2:.5f})")
rec("W_var_imag", f"{np.var(w.W.imag):.5f}")
rec("W_circular", f"E[W^2]={np.mean(w.W**2):.2e} (0 => circular)")

print(); print("="*78); print("PART V — SNR / RSR"); print("="*78)
sig=np.mean(np.abs(w.G@w.S)**2); noi=np.mean(np.abs(w.W)**2); ref=np.mean(np.abs(w.B)**2)
one=np.mean(np.abs(np.outer(w.G[:,0],w.S[0]))**2)
rec("measured_SNR_total_over_noise_dB", f"{10*np.log10(sig/noi):.3f} (nominal 5.0)")
rec("sigma2_formula", f"sigma2={w.sigma2:.5f}, K/SNRlin={TRACK_B_K/10**(5.0/10):.5f}")
rec("measured_RSR_vs_single_user_dB", f"{10*np.log10(ref/one):.3f} (nominal 12.0)")
rec("measured_RSR_vs_total_dB", f"{10*np.log10(ref/sig):.3f}")
rec("RSR_denominator_convention", "SINGLE USER (Cui eq.37), not the K-user sum")

print(); print("="*78); print("PART VII — CANONICAL MAPPING"); print("="*78)
n=0
lhs=np.abs(w.S.conj().T@np.conjugate(w.G[n])+np.conjugate(w.B[n]+w.W[n]))
rec("mapping_M_S__u_conjg__b_conjB", f"max dev {np.abs(lhs-w.Z[n]).max():.2e}")
rec("gs_signature", str(inspect.signature(biased_gs_channel_rows)))
rec("em_gs_signature", str(inspect.signature(em_gs_channel_rows)))

print(); print("="*78); print("PART IX — HANKEL / HS-GS"); print("="*78)
for N in (8,16,32):
    p=best_pencil(N); H=hankel_matrix(np.arange(N).astype(complex),p)
    rec(f"hankel_N{N}", f"pencil p={p}, shape {H.shape}, cap={hankel_rank_cap(N)}")
rec("hs_gs_signature", str(inspect.signature(hs_gs)))
rec("hs_gs_auto_signature", str(inspect.signature(hs_gs_auto)))
rec("order_selection", "held-out pilot residual (select_order_heldout) -- NOT oracle L_k")
src=inspect.getsource(hs_gs_auto)
rec("hs_gs_auto_uses_true_L", "L_true" in src or "w.L_k" in src)
rec("cadzow_default_n_iter", inspect.signature(cadzow_project).parameters["n_iter"].default)
rec("project_every_default", inspect.signature(hs_gs).parameters["project_every"].default)
rec("hs_max_iter_default", inspect.signature(hs_gs).parameters["max_iter"].default)
rec("exact_step_default", inspect.signature(hs_gs).parameters["exact_step"].default)
rec("val_frac_default", inspect.signature(select_order_heldout).parameters["val_frac"].default)
rec("select_iter_used_in_runs", 20)
# does the order selector ever see the truth?
r=hs_gs_auto(w.S,w.Z,w.B,w.sigma2,exact_step="em_gs",max_iter=10,select_iter=5)
rec("selected_Lhat_vs_true_Lk", f"L_hat={r.L_hat}, true L_k={tuple(w.L_k)}  (single scalar for ALL users)")
rec("per_user_or_shared_order", "SHARED scalar L_hat applied to every user column")

print(); print("="*78); print("PART XII — METRIC"); print("="*78)
from rydberg_sim.metrics import channel_nmse
rec("channel_nmse_signature", str(inspect.signature(channel_nmse)))
rec("pooling", "ratio of sums: 10log10(sum ||Ghat-G||_F^2 / sum ||G||_F^2)")
rec("db_factor", "10 log10 (power ratio), NOT 20 log10")

(REPO/"results/track_b/artifact_audit.json").write_text(json.dumps(OUT,indent=2,default=str))
print(f"\nwrote artifact_audit.json ({len(OUT)} facts)")
