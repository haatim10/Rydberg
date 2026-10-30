"""Common exact Rician detector for every plug-in channel estimate.

The detector receives only magnitudes, an estimated channel, known reference,
and the total complex noise variance. It enumerates all 4**K QPSK vectors.
"""
from itertools import product

import numpy as np
from scipy.special import i0e


# Each component carries one bit; neighboring points around the circle differ
# by one bit. Symbol power is one, matching the pilot normalization.
SYMBOL_BITS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
CONSTELLATION = ((1 - 2 * SYMBOL_BITS[:, 0].astype(float))
                 + 1j * (1 - 2 * SYMBOL_BITS[:, 1].astype(float))) / np.sqrt(2)


def hypotheses(users=3):
    symbols = np.array(list(product(range(4), repeat=users)), dtype=np.int64)
    return CONSTELLATION[symbols], SYMBOL_BITS[symbols]


def detect(z, channel, reference, sigma2, symbols_per_chunk=64):
    """Return decoded bits (data vectors, users, 2), with no channel truth.

    z has shape (antennas, data vectors). reference has shape (antennas,) or
    (antennas, 1); the benchmark transmits the same known reference every time.
    The likelihood is (z-a)^2/v - log(i0e(2*z*a/v)), omitting only terms that
    depend on the observation and v, never on the transmitted hypothesis.
    """
    z = np.asarray(z, dtype=np.float64)
    channel = np.asarray(channel, dtype=np.complex128)
    reference = np.asarray(reference).reshape(channel.shape[0], 1)
    if z.ndim != 2 or channel.ndim != 2 or z.shape[0] != channel.shape[0]:
        raise ValueError("incompatible antenna dimensions")
    if not sigma2 > 0:
        raise ValueError("positive total complex noise variance required")
    x, bits = hypotheses(channel.shape[1])
    amplitude = np.abs(channel @ x.T + reference).T  # hypotheses, antennas
    decoded = []
    for start in range(0, z.shape[1], symbols_per_chunk):
        observed = z[:, start:start + symbols_per_chunk].T[:, None, :]
        a = amplitude[None, :, :]
        argument = 2 * observed * a / sigma2
        score = ((observed - a) ** 2 / sigma2 - np.log(i0e(argument))).sum(-1)
        decoded.append(bits[score.argmin(-1)])
    return np.concatenate(decoded, axis=0)


def draw_data(channel, reference, sigma2, rng, vectors=256):
    """Generate independent data noise and bits once, shared by all methods."""
    n, k = channel.shape
    indices = rng.integers(0, 4, size=(k, vectors))
    transmitted = CONSTELLATION[indices]
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal((n, vectors))
                                  + 1j * rng.standard_normal((n, vectors)))
    z = np.abs(channel @ transmitted + np.asarray(reference).reshape(n, 1) + noise)
    bits = SYMBOL_BITS[indices].transpose(1, 0, 2)
    return z, bits, indices


def verify_detector():
    """Independent mapping/noiseless check and direct Rician density check."""
    from scipy.stats import rice

    rng = np.random.default_rng(381052)
    n, k = 12, 3
    channel = (rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))) / np.sqrt(2)
    reference = np.full((n, 1), 10 ** (12 / 20))
    x, bits = hypotheses(k)
    z = np.abs(channel @ x.T + reference)
    np.testing.assert_array_equal(detect(z, channel, reference, 1e-7), bits)
    # Compare every candidate, not just the winning hypothesis, to scipy's
    # full Rice density evaluated with per-real-component standard deviation.
    v = 2.4
    observed = rng.uniform(.1, 6, size=(7, n))
    amplitude = np.abs(channel @ x.T + reference).T
    zz, aa = observed[:, None, :], amplitude[None, :, :]
    scores = ((zz - aa) ** 2 / v - np.log(i0e(2 * zz * aa / v))).sum(-1)
    full = -rice.logpdf(zz, b=aa / np.sqrt(v / 2), scale=np.sqrt(v / 2)).sum(-1)
    # Candidate differences remove the deliberately omitted data-only term.
    np.testing.assert_allclose(scores - scores[:, :1], full - full[:, :1], atol=1e-11, rtol=1e-11)
    return {"noiseless_64_hypotheses": "pass", "scipy_full_rician_likelihood": "pass",
            "qpsk_unit_power": bool(np.allclose(np.abs(CONSTELLATION) ** 2, 1))}
