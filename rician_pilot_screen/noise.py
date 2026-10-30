"""Create independent Rician views without discarding pilot columns.

For Z=|Y|, Y=mu+w, w~CN(0,v), draw eta~CN(0,v) independently.
Let theta=arg(Y) and eta'=exp(i*theta)*eta. Circular symmetry makes eta'
independent of Y. Hence (|Z+a*eta|, |Z-eta/a|) has the joint law of
(|Y+a*eta'|, |Y-eta'/a|). The underlying proper Gaussian noises have
cross-covariance v-v=0, so the two views are independent conditional on mu,
with complex variances v*(1+a**2) and v*(1+a**-2).

This requires fixed a and known v with independent circular Gaussian noise
per measurement. Independence is not conditional on the original saved Z;
repeated re-corruptions of one example do not create independent channels.
Rician predictive likelihood is not an unbiased channel-NMSE objective.
"""

import math
import numbers

import torch

from sahure.data import Observations


def make_views(
    data: Observations,
    alpha: float,
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Observations, Observations]:
    """Return full-pilot input/target views with their respective variances.

    Supply exactly one integer ``seed`` or CPU ``generator``. The latter is
    advanced in place; the former creates a private deterministic generator.
    ``alpha`` is a fixed protocol constant, not a measurement-driven output.
    S and B are shared read-only with the original observation record. Only
    CPU float32/float64 observations are supported in this feasibility test.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise TypeError("alpha must be a fixed real scalar")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    inverse_alpha = 1 / alpha
    input_factor, target_factor = 1 + alpha * alpha, 1 + inverse_alpha * inverse_alpha
    if not math.isfinite(input_factor) or not math.isfinite(target_factor):
        raise ValueError("alpha produces a nonfinite noise variance")
    if (seed is None) == (generator is None):
        raise ValueError("supply exactly one seed or generator")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise TypeError("seed must be an integer")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("a CPU torch.Generator is required")
    if any(x.device.type != "cpu" for x in data.args()):
        raise ValueError("the feasibility re-corruption implementation is CPU only")
    if data.z.dtype not in (torch.float32, torch.float64):
        raise TypeError("magnitudes must use float32 or float64")
    if data.sigma2.dtype != data.z.dtype:
        raise TypeError("noise variance and magnitudes must have the same dtype")
    if data.z.ndim != 3 or data.b.shape != data.z.shape:
        raise ValueError("z and b must have matching (batch, antenna, pilot) shapes")
    if data.s.ndim != 3 or (data.s.shape[0], data.s.shape[2]) != (data.z.shape[0], data.z.shape[2]):
        raise ValueError("pilots must have shape (batch, user, pilot)")
    if data.sigma2.shape != (data.z.shape[0],):
        raise ValueError("noise variance must have shape (batch,)")
    if not bool(torch.isfinite(data.z).all()) or not bool((data.z >= 0).all()):
        raise ValueError("magnitudes must be finite and nonnegative")
    if not bool(torch.isfinite(data.sigma2).all()) or not bool((data.sigma2 > 0).all()):
        raise ValueError("noise variances must be positive and finite")
    input_variance = data.sigma2 * input_factor
    target_variance = data.sigma2 * target_factor
    if not bool(torch.isfinite(input_variance).all() & torch.isfinite(target_variance).all()):
        raise ValueError("re-corrupted noise variance overflow")
    scale = (data.sigma2 / 2).sqrt()[:, None, None]
    eta = torch.complex(
        torch.randn(data.z.shape, dtype=data.z.dtype, generator=generator),
        torch.randn(data.z.shape, dtype=data.z.dtype, generator=generator),
    ) * scale
    input_view = Observations((data.z + alpha * eta).abs(), data.s, data.b, input_variance)
    target_view = Observations((data.z - eta / alpha).abs(), data.s, data.b, target_variance)
    return input_view, target_view
