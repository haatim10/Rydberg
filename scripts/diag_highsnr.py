"""Why does NMSE_G stop improving at high SNR? (Track-B baseline diagnostic)

Ratio-of-sums NMSE is dominated by the worst trials, so a small number of
failed phase retrievals can dominate the average even while the typical
trial keeps improving. This separates the two.
"""
import numpy as np

from rydberg_sim.gs import biased_gs_channel_rows, em_gs_channel_rows
from rydberg_sim.metrics import channel_nmse
from rydberg_sim.track_b_drivers import TRACK_B_RSR_DB, track_b_world

N_TRIALS = 200
P = 10


def run(snr, alg, max_iter=50):
    per, err, tot = [], 0.0, 0.0
    for t in range(N_TRIALS):
        w = track_b_world(t, P, snr)
        if alg == "biased_gs":
            Gh = biased_gs_channel_rows(w.S, w.Z, w.B, max_iter=max_iter).G_hat
        else:
            Gh = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=max_iter).G_hat
        r = channel_nmse(Gh, w.G)
        per.append(r.error_energy / r.true_energy)
        err += r.error_energy
        tot += r.true_energy
    per = np.array(per)
    return dict(
        pooled=10 * np.log10(err / tot),
        median=10 * np.log10(np.median(per)),
        p90=10 * np.log10(np.percentile(per, 90)),
        worst=10 * np.log10(per.max()),
        frac_bad=float(np.mean(per > 1.0)),
        top1_share=float(np.sort(per)[-1] * tot / N_TRIALS / (err / N_TRIALS)),
    )


print(f"N=8 K=3 P={P} RSR={TRACK_B_RSR_DB} dB, {N_TRIALS} trials, 50 iterations")
print("pooled = ratio-of-sums (the reported metric); median = typical trial\n")
for alg in ("biased_gs", "em_gs"):
    print(f"--- {alg} ---")
    print(f"{'SNR':>5} {'pooled':>8} {'median':>8} {'p90':>8} {'worst':>8} "
          f"{'frac NMSE>1':>12} {'worst-trial share':>18}")
    for snr in (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        r = run(snr, alg)
        print(f"{snr:5.0f} {r['pooled']:8.2f} {r['median']:8.2f} {r['p90']:8.2f} "
              f"{r['worst']:8.2f} {r['frac_bad']:12.3f} {r['top1_share']:17.1%}")
    print()

print("Effect of more GS iterations at 20 dB (local-minimum test):")
for it in (50, 200, 1000):
    r = run(20.0, "em_gs", max_iter=it)
    print(f"  max_iter={it:5d}  pooled={r['pooled']:7.2f}  median={r['median']:7.2f}  "
          f"frac>1={r['frac_bad']:.3f}")
