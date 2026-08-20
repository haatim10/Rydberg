"""Which order rule should a structured estimator use? (Track B diagnostic)"""
import numpy as np

from rydberg_sim.config import SimulationConfig
from rydberg_sim.gs import em_gs_channel_rows
from rydberg_sim.metrics import channel_nmse
from rydberg_sim.monte_carlo import ExperimentSpec, generate_channel_estimation_trial
from rydberg_sim.track_b_structure import estimate_order, project_matrix

cfg = SimulationConfig.create(N=16, K=3, L=(3, 5, 7), beta=1.0,
                              master_seed=20250820, c=1.0)


def sp(P):
    return ExperimentSpec(
        experiment="d", track="B", cfg=cfg, P=P, vartheta=0.0,
        snr_db_grid=(0.0,), rsr_db_grid=(30.0,), n_trials=1,
        algorithms=("em_gs",), max_iter=50, qam_M=4,
        channel_model="ula_geometric", write_ber=False)


print("NMSE_G (dB): raw vs Hankel projection under different order rules")
print("true L = (3,5,7); 'oracle' uses the true per-user L (diagnostic only)")
hdr = f"{'P':>4}{'SNR':>6} | {'raw':>7} {'L=5':>7} {'L=7':>7} {'mdl':>7} {'gap':>7} {'oracle':>7} | mean gap-order"
print(hdr)
for P in (32, 64):
    for snr in (0.0, 10.0, 20.0, 30.0):
        acc = {k: 0.0 for k in ("raw", "f5", "f7", "mdl", "gap", "oracle")}
        tot = 0.0
        orders = []
        for t in range(10):
            w = generate_channel_estimation_trial(sp(P), t, snr, 30.0)
            Gh = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=50).G_hat
            r = channel_nmse(Gh, w.G)
            acc["raw"] += r.error_energy
            tot += r.true_energy
            acc["f5"] += channel_nmse(project_matrix(Gh, "hankel", 5), w.G).error_energy
            acc["f7"] += channel_nmse(project_matrix(Gh, "hankel", 7), w.G).error_energy
            acc["oracle"] += channel_nmse(
                project_matrix(Gh, "hankel", np.asarray(w.L_k)), w.G).error_energy
            for nm in ("mdl", "gap"):
                o = [estimate_order(Gh[:, k], max_order=7, method=nm) for k in range(3)]
                if nm == "gap":
                    orders.append(o)
                acc[nm] += channel_nmse(
                    project_matrix(Gh, "hankel", np.array(o)), w.G).error_energy
        d = {k: 10 * np.log10(v / tot) for k, v in acc.items()}
        om = np.array(orders).mean(0)
        print(f"{P:4d}{snr:6.1f} | {d['raw']:7.2f} {d['f5']:7.2f} {d['f7']:7.2f} "
              f"{d['mdl']:7.2f} {d['gap']:7.2f} {d['oracle']:7.2f} | {np.round(om,1)}")
