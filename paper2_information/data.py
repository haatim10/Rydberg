"""Fresh development worlds; ground truth is confined to this evaluator side."""
from dataclasses import dataclass
import hashlib
import numpy as np
from rydberg_sim.pilots import generate_gaussian_pilots

FAMILIES = ('sparse', 'clustered', 'independent')
SNRS = (-5, 5, 15, 25)
MASTER = 2026092191


@dataclass
class World:
    channel: np.ndarray
    Z: np.ndarray
    S: np.ndarray
    B: np.ndarray
    variance: float
    case_id: str
    digest: str
    pilot_id: str


def world(split, family, snr, index, pilots=20):
    if split not in ('tune', 'screen', 'diagnostic') or family not in FAMILIES:
        raise ValueError('Unknown split or family')
    split_id = ('tune', 'screen', 'diagnostic').index(split)
    # Independent pilot matrices in tune and screen; three within each split.
    pilot_id = f'{split}/pilot{index%3}/P{pilots}'
    rngp = np.random.default_rng(np.random.SeedSequence([MASTER, 77, split_id, index%3, pilots]))
    S = np.asarray(generate_gaussian_pilots(K=4, P=pilots, rng=rngp).S)
    key = [MASTER, split_id, FAMILIES.index(family), int(10*(snr+100)), index, pilots]
    rng = np.random.default_rng(np.random.SeedSequence(key))
    if family == 'sparse':
        columns = []
        for rays in rng.integers(3, 8, size=4):
            angles = rng.uniform(-np.pi/2, np.pi/2, rays)
            gains = (rng.normal(size=rays)+1j*rng.normal(size=rays))/np.sqrt(2)
            columns.append(np.sqrt(32/rays)*np.exp(1j*np.pi*np.arange(32)[:, None]*np.sin(angles)[None])@gains)
        h = np.stack(columns, axis=1)
    elif family == 'clustered':
        rays = 40
        center = rng.uniform(-np.pi/2, np.pi/2, (4, 4, 1))
        angles = np.clip(center+rng.uniform(-np.pi/36, np.pi/36, (4, 4, 10)), -np.pi/2, np.pi/2).reshape(4, 40)
    else:
        rays = 40
        angles = rng.uniform(-np.pi/2, np.pi/2, (4, rays))
    if family != 'sparse':
        gains = (rng.normal(size=(4, rays))+1j*rng.normal(size=(4, rays)))/np.sqrt(2)
        steering = np.exp(1j*np.pi*np.arange(32)[:, None, None]*np.sin(angles)[None])
        h = np.sqrt(32/rays)*np.sum(steering*gains[None], axis=-1)
    signal_power = 32*np.sum(np.abs(S)**2)/pilots
    B = np.full((32, pilots), np.sqrt(10*signal_power), dtype=complex)
    variance = float(signal_power/10**(snr/10))
    noise = np.sqrt(variance/2)*(rng.normal(size=B.shape)+1j*rng.normal(size=B.shape))
    Z = np.abs(h@S+B+noise)
    digest = hashlib.sha256(b''.join(x.tobytes() for x in (h, S, B, Z, np.array(variance)))).hexdigest()
    return World(h, Z, S, B, variance, f'{split}/{family}/{snr}/{index}/P{pilots}', digest, pilot_id)


def nmse(estimate, truth):
    return float(np.sum(np.abs(estimate-truth)**2)/np.sum(np.abs(truth)**2))


def noise_cell(S, variance, scale=np.sqrt(32)):
    """Receiver-available expected signal/noise ratio, never realized truth."""
    expected = scale**2*np.sum(np.abs(S)**2)/S.shape[1]
    db = 10*np.log10(expected/variance)
    return int(min(SNRS, key=lambda x: abs(x-db)))
