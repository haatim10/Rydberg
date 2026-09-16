"""Paper 1 worlds without importing the unrelated neural-network stack."""
from dataclasses import replace
import numpy as np
from rydberg_sim.monte_carlo import generate_channel_estimation_trial
from rydberg_sim.track_b_drivers import draw_L_k, track_b_spec
from rydberg_sim.forward import exact_forward
from rydberg_sim.rng import get_operating_point_rngs


def make_world(trial, *, N=32, K=3, P=30, snr=5., rsr=12., L=None,
               family="TB", channel="ula"):
    master = 20250820 if family == "TB" else 20260827
    paths = draw_L_k(trial, K, master_seed=master) if L is None else (L,)*K
    spec = track_b_spec(P=P, n_trials=trial+1,N=N,K=K,L=paths,
                        master_seed=master,experiment="codex_paper1")
    w = generate_channel_estimation_trial(spec, trial, snr, rsr)
    if channel != "ula":
        rng = np.random.default_rng(np.random.SeedSequence([trial,0x5A17]))
        g = np.empty((N,K),complex)
        for k in range(K):
            if channel == "clustered":
                ctr=rng.uniform(-np.pi/2,np.pi/2,4)
                off=np.deg2rad(rng.uniform(-5,5,(4,10)))
                theta=np.clip(ctr[:,None]+off,-np.pi/2,np.pi/2).ravel()
            elif channel == "ten_path":
                theta=rng.uniform(-np.pi/2,np.pi/2,10)
            else:
                raise ValueError(channel)
            d=len(theta)
            a=(rng.standard_normal(d)+1j*rng.standard_normal(d))/np.sqrt(2*d)
            g[:,k]=(np.exp(-1j*np.arange(N)[:,None]*np.pi*np.sin(theta))*a).sum(axis=1)
        g *= np.linalg.norm(w.G)/np.linalg.norm(g)
        rngs=get_operating_point_rngs(master,trial,snr,rsr)
        e=exact_forward(g,w.S,w.B,w.sigma2,rng_noise=rngs.noise)
        # Only arrays needed by the estimator; geometric metadata remains for
        # baseline worlds only and must never be used for SV channel bounds.
        from types import SimpleNamespace
        return SimpleNamespace(G=g,S=w.S,B=w.B,Z=e.Z,sigma2=w.sigma2)
    return w
