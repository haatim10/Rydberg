"""Instrumented copies of the frozen ten-stage starts.

The loops reproduce tcom_runtime_repair_20261020/native.py (adaptive) and
tcom_start_ablation_20261020/starts.py FixedCore (global) step by step, and
also return, per stage, the mean M used by that stage's EM step and the
relaxed output g_dc that enters the stage's Hankel filter. `g0` replaces the
spectral initialization (for poor-start diagnostics); with g0=None the final
output must equal the frozen start to roundoff (checked by the caller).
"""
import math
import numpy as np
from scipy.special import expit
from tcom_runtime_bypass_20261017.sparse_fast import as_numpy, spectral_initial
from tcom_pilot_em_20261018.continuation import posterior


def _shrink(dc, rel, scale, eye, unhankel, cols):
    batch, _, k = dc.shape
    h = np.lib.stride_tricks.sliding_window_view(dc.swapaxes(-1, -2), cols, axis=-1)
    hn = h/scale[..., None, None]
    t2 = rel[..., None, None]**2
    hh = hn.conj().swapaxes(-1, -2)
    solved = np.linalg.solve(hh @ hn + t2*eye, hh)
    hs = (hn - t2*solved.conj().swapaxes(-1, -2))*scale[..., None, None]
    return (hs.reshape(batch, k, -1) @ unhankel).swapaxes(-1, -2)


def run(core, obs, g0=None, *, tau_rule=None):
    """Return (g_final, [dict(mu=M, dc=g_dc, g=g_out, tau=tau, beta=beta)] * 10).

    core: NativeEstimator (adaptive) or NativeGlobalEstimator/FixedCore.
    tau_rule: optional callable(stage_index, mu, dc, z, s, v) -> absolute tau
      [batch, K], replacing the core's threshold (Phase 2 variant; beta then
      comes from `tau_rule.beta`). None reproduces the frozen core exactly.
    """
    z, s, b, v = as_numpy(obs.args())
    batch, n, p = z.shape
    k = s.shape[1]
    cols, rows, eye, unhankel = core.shape_constants(n)
    cap = min(rows, cols)
    sh = s.conj().swapaxes(-1, -2)
    gram = s @ sh
    op = np.linalg.solve(gram.swapaxes(-1, -2), sh.swapaxes(-1, -2)).swapaxes(-1, -2)
    adaptive = hasattr(core, 'w1')
    if adaptive:
        inverse = np.linalg.solve(gram, np.broadcast_to(np.eye(k, dtype=np.complex128), gram.shape))
        noise_proxy = v*np.trace(inverse, axis1=-2, axis2=-1).real/k
    elif core.beta is not None and core.noise_c is not None:
        raise NotImplementedError('noise-scaled fixed cores are not part of the receiver')
    g = spectral_initial(z, s, b) if g0 is None else np.array(g0, dtype=np.complex128, copy=True)
    trace = []
    for t, gamma in enumerate(core.gamma):
        mu = g @ s + b
        q = posterior(z, mu, v)
        dc = g + gamma*((q - b) @ op - g)
        h = np.lib.stride_tricks.sliding_window_view(dc.swapaxes(-1, -2), cols, axis=-1)
        scale = np.sqrt(np.maximum(np.sum(np.abs(h)**2, axis=(-2, -1))/cap, 1e-24))
        if adaptive:
            sv = np.linalg.svd(h, compute_uv=False)
            weights = sv/np.maximum(sv.sum(-1, keepdims=True), 1e-12)
            ent = -np.sum(weights*np.log(np.maximum(weights, 1e-12)), axis=-1)
            residual = np.log1p(np.mean((z - np.abs(dc @ s + b))**2, axis=(-2, -1))/v)
            power = np.maximum(np.mean(np.abs(dc)**2, axis=-2), 1e-12)
            features = np.stack((np.exp(ent)/cap, weights[..., 0], weights[..., -((cap + 1)//2):].sum(-1),
                np.broadcast_to(residual[:, None], (batch, k)), np.log1p(noise_proxy[:, None]/power),
                np.full((batch, k), p/(2*k)), np.full((batch, k), math.log(cap)),
                np.full((batch, k), (t + 1)/len(core.gamma))), axis=-1)
            hidden = features @ core.w1.T + core.b1
            hidden = hidden*expit(hidden)
            raw = hidden @ core.w2.T + core.b2
            rel = .001 + np.where(raw[..., 0] > 20, raw[..., 0], np.logaddexp(0., raw[..., 0]))
            beta = expit(raw[..., 1])
            if core.config.gate_one:
                beta = np.ones_like(beta)
        else:
            rel = np.full((batch, k), core.relative_tau[t])
            beta = np.full((batch, k), core.beta[t])
        if tau_rule is not None:
            rel = tau_rule(t, mu, dc, z, s, v)/scale
            beta = np.full((batch, k), tau_rule.beta)
        structured = _shrink(dc, rel, scale, eye, unhankel, cols)
        g = dc + beta[:, None, :]*(structured - dc)
        trace.append(dict(mu=mu, dc=dc, g=g, tau=rel*scale, beta=beta))
    return g, trace
