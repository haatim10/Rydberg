"""Observation-only stopping/release controls. No channel-truth argument."""
import time
import numpy as np
from codex_verification.solver import initial, ratio
from paper2_optshrink_loop.estimator import shrink, trajectory
from sahure.classical import likelihood

BUDGETS = (10, 20, 50, 100)
RELEASES = (0, 5, 10, 20, 50, 100)
SCHEDULES = [(r, t) for t in BUDGETS for r in RELEASES if r <= t]


def trajectories(z, s, b, variance, param):
    """All paths use only supplied observations, and share a spectral start."""
    if variance <= 0 or z.shape != b.shape or s.shape[-1] != z.shape[-1]:
        raise ValueError('invalid observation dimensions or noise variance')
    g = np.repeat(initial(s, z, b)[None], len(RELEASES), axis=0)
    sh = s.conj().T
    op = np.linalg.solve((s @ sh).T, sh.T).T
    saved = {}
    for t in range(1, max(BUDGETS) + 1):
        mu = g @ s + b
        latent = z * np.exp(1j * np.angle(mu)) * ratio(2*z*abs(mu)/variance)
        g = (latent-b) @ op
        configs = [param if t <= r else dict(rank=0, strength=0.) for r in RELEASES]
        g = shrink(g, configs)
        if not np.isfinite(g).all():
            raise FloatingPointError(f'nonfinite trajectory at {t}')
        if t in BUDGETS:
            for j, r in enumerate(RELEASES):
                if r <= t:
                    saved[(r, t)] = g[j].copy()
    return np.stack([saved[key] for key in SCHEDULES])


def refit(z, s, b, variance, param, schedule):
    release, total = schedule
    g = initial(s, z, b)[None]
    sh = s.conj().T
    op = np.linalg.solve((s @ sh).T, sh.T).T
    for t in range(1, total + 1):
        mu = g @ s + b
        g = (z * np.exp(1j*np.angle(mu)) * ratio(2*z*abs(mu)/variance)-b) @ op
        if t <= release:
            g = shrink(g, [param])
    if not np.isfinite(g).all():
        raise FloatingPointError('nonfinite refit')
    return g[0]


def context_candidates(z, s, b, variance, param):
    return trajectories(z[:, :21], s[:, :21], b[:, :21], variance, param)


def validation_scores(gs, z, s, b, variance):
    return likelihood(z[:, 21:], gs@s[:, 21:]+b[:, 21:], variance).mean((-2, -1))


def select_index(scores, mode):
    allowed = [i for i, (r, t) in enumerate(SCHEDULES)
               if mode == 'release' or (mode == 'stop' and r == t)
               or (mode == 'em' and r == 0)]
    if not allowed:
        raise ValueError(mode)
    return allowed[int(np.argmin(np.asarray(scores)[allowed]))]


def estimate(z, s, b, variance, config):
    tick = time.perf_counter()
    param = dict(rank=config['rank'], strength=config['strength'])
    mode = config.get('mode', 'release')
    if mode == 'release':
        gs = context_candidates(z, s, b, variance, param)
        schedules = SCHEDULES
    elif mode in ('stop', 'em'):
        pp = param if mode == 'stop' else dict(rank=0, strength=0.)
        path = trajectory(z[:,:21], s[:,:21], b[:,:21], variance, [pp], BUDGETS)
        gs = np.stack([path[t][0] for t in BUDGETS])
        schedules = [(t if mode=='stop' else 0, t) for t in BUDGETS]
    else:
        raise ValueError(mode)
    scores = validation_scores(gs, z, s, b, variance)
    index = int(scores.argmin())
    g = refit(z, s, b, variance, param, schedules[index])
    return g, dict(schedule=schedules[index], scores=scores, context=gs[index],
                   seconds=time.perf_counter()-tick, selection_max_updates=100,
                   all_pilots=30, context_pilots=21, validation_pilots=9)
