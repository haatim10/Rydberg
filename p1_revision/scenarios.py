"""Frozen reviewer-response designs. Existing Paper 1 modules stay unchanged."""
from types import SimpleNamespace
from time import perf_counter
import numpy as np
from codex_verification.worlds import make_world
from codex_verification import solver

FRACTIONS = (0.1, 0.2, 0.3, 0.4)
POSITION_STDS = (0.0, 0.005, 0.01, 0.02)  # wavelengths, not hardware specifications


def configurations():
    cells = []
    def add(kind, tag, seed, **extra):
        c = dict(kind=kind, tag=tag, seed_base=seed, N=32, K=3, P=30,
                 snr=5., rsr=12., L=None, trials=400, iterations=100,
                 select_iterations=25, sweeps=4)
        c.update(extra)
        cells.append(c)
    for i, (l, snr) in enumerate((l,s) for l in (3,7,14) for s in (0.,10.)):
        add('baselines', f'baselines_L{l}_S{snr:g}', 71_000_000+i*10_000, L=l, snr=snr)
    for i, (p, snr) in enumerate((p,s) for p in (10,30) for s in (0.,10.)):
        for f in FRACTIONS:
            add('holdout', f'holdout_P{p}_S{snr:g}_F{f:g}', 72_000_000+i*10_000,
                P=p, snr=snr, val_fraction=f)
    for i,l in enumerate((3,12)):
        for std in POSITION_STDS:
            add('mismatch', f'position_L{l}_D{std:g}', 73_000_000+i*10_000,
                L=l, position_std=std)
    return cells


def position_world(w, std, trial):
    """Known calibrated B, actual off-grid channel G, unchanged thermal-noise draw.

    Positions x_n/lambda=n/2+std*u_n. The same u_n is used across strengths.
    Channel powers use the original statistical beta/SNR convention: no
    realization-dependent channel renormalization is introduced.
    """
    if std < 0:
        raise ValueError('position standard deviation must be nonnegative')
    if std == 0:
        return w  # bitwise matched-model anchor, including exact forward values
    n,k = w.G.shape
    rng = np.random.default_rng(np.random.SeedSequence([trial,0x504F5349]))
    offsets = std*rng.standard_normal(n)
    pos = np.arange(n)/2 + offsets
    g = np.column_stack([
        w.cfg.c*(np.exp(-2j*np.pi*pos[:,None]*np.sin(w.theta[u])[None,:])@w.alpha[u])
        for u in range(k)])
    # Default P1 reference is broadside (vartheta=0); then B is unchanged.
    # This formula also correctly calibrates B for a nonzero reference angle.
    b = w.B*np.exp(-2j*np.pi*offsets*np.sin(w.vartheta))[:,None]
    return SimpleNamespace(G=g,S=w.S,B=b,Z=np.abs(g@w.S+b+w.W),W=w.W,
                           sigma2=w.sigma2,L_k=w.L_k,position_offsets=offsets)


def make(c, index):
    trial = c['seed_base']+index
    w = make_world(trial, **{x:c[x] for x in ('N','K','P','snr','rsr','L')})
    if c['kind']=='mismatch':
        w = position_world(w,c['position_std'],trial)
    return w


def select_fraction(s,z,b,sigma2, *, fraction=.3, iterations=25, sweeps=4):
    p=s.shape[1]
    if not 0 < fraction < 1:
        raise ValueError('fraction must lie strictly between zero and one')
    nv=max(1,round(fraction*p))
    if p-nv < 2*s.shape[0]:
        raise ValueError('fewer than 2K context pilots in this declared study')
    ranks=np.arange(1,(z.shape[0]+1)//2+1)
    # Gaussian pilot columns are exchangeable; nested last-column folds match
    # the original selector exactly at fraction .3, and pair the split study.
    g=solver.fit(s[:,:-nv],z[:,:-nv],b[:,:-nv],sigma2,
                 iterations=iterations,ranks=ranks,sweeps=sweeps)
    scores=np.sum((z[:,-nv:]-np.abs(g@s[:,-nv:]+b[:,-nv:]))**2,axis=(1,2))
    return int(ranks[np.argmin(scores)]),scores


def estimate_pair(w,c):
    start=perf_counter()
    rank,scores=select_fraction(w.S,w.Z,w.B,w.sigma2,
        fraction=c.get('val_fraction',.3),iterations=c['select_iterations'],sweeps=c['sweeps'])
    selected=perf_counter()
    em=solver.fit(w.S,w.Z,w.B,w.sigma2,iterations=c['iterations'])[0]
    em_done=perf_counter()
    hs=solver.fit(w.S,w.Z,w.B,w.sigma2,iterations=c['iterations'],ranks=[rank],sweeps=c['sweeps'])[0]
    done=perf_counter()
    if rank==(w.G.shape[0]+1)//2:
        assert np.array_equal(em,hs), 'full-rank fallback must be bitwise identical'
    return dict(em=em,hs=hs),dict(rank=rank,scores=scores.tolist(),
        seconds_select=selected-start,seconds_em=em_done-selected,
        seconds_hs_fit=done-em_done,seconds_hs_total=(selected-start)+(done-em_done))
