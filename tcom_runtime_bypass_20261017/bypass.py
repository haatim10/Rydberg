"""Observable-only high-SNR hard dispatch for the frozen learned estimator.

This is an offline-calibrated engineering safeguard, not a learned gate.
Thresholds are supplied in a policy selected separately on held-out magnitude
likelihood. Each input world executes exactly one estimator branch. The
ordinary-SNR branch invokes the original frozen model without modifications.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import torch

from tcom_followups_20261017.common import load_learned_models
from .sparse_fast import estimate as fast_sparse_estimate

REFERENCE_BINS_DB=(5,12,20)
ORDINARY_MAX_DB=20.
BOUNDARY_TOLERANCE_DB=1e-10


class BypassEstimator:
    def __init__(self,models=None):
        self.models=load_learned_models() if models is None else dict(models)
        self.unfiltered_models={}
        for name,model in self.models.items():
            if model.config.core!='em' or model.config.stages!=10:
                raise ValueError('this safeguard is for the frozen ten-stage EM model')
            copied=deepcopy(model)
            copied.config=replace(model.config,structure='none')
            copied.eval()
            # Retain the complete checkpoint state, including unused controller
            # weights, so disabling the structure cannot alter learned steps.
            original_state=model.state_dict();copied_state=copied.state_dict()
            if original_state.keys()!=copied_state.keys() or not all(
                torch.equal(original_state[key],copied_state[key]) for key in original_state):
                raise AssertionError('copy changed the frozen checkpoint state')
            self.unfiltered_models[name]=copied

    @torch.no_grad()
    def original(self,obs,model_name):
        return self.models[model_name](*obs.args())

    @torch.no_grad()
    def unfiltered(self,obs,model_name):
        return self.unfiltered_models[model_name](*obs.args())

    @torch.no_grad()
    def em100(self,obs):
        # Optimized exact unregularized EM. Conversion, initialization, and all
        # 100 updates are performed on every call and included in its latency.
        return fast_sparse_estimate(obs,dict(factor=0.,dictionary_size=128,
            inner_steps=1,iterations=100,backend='fft',mode='fista'))

    def masks(self,obs,policy):
        """Per-world dispatch using known noise/reference and nominal K power.

        policy is a flat JSON mapping {'5': cutoff, '12': cutoff, '20': cutoff}.
        This SNR proxy uses the existing unit-power-per-user normalization; it
        does not measure realized channel power. Each cutoff must be >=20 dB.
        """
        cutoffs=[]
        for ref in REFERENCE_BINS_DB:
            value=float(policy[str(ref)])
            if not math.isfinite(value) or value<ORDINARY_MAX_DB:
                raise ValueError('every reference-bin cutoff must be finite and >=20 dB')
            cutoffs.append(value)
        variance=obs.sigma2
        if variance.ndim!=1 or len(variance)!=len(obs) or not torch.all(torch.isfinite(variance)&(variance>0)):
            raise ValueError('one positive finite known noise variance per world is required')
        snr=10*torch.log10(obs.s.shape[1]/variance)
        reference_power=obs.b.abs().square().mean((-2,-1))
        rsr=10*torch.log10(reference_power)
        bins=torch.tensor(REFERENCE_BINS_DB,dtype=variance.dtype,device=variance.device)
        indices=(rsr[:,None]-bins[None]).abs().argmin(-1)
        table=torch.tensor(cutoffs,dtype=variance.dtype,device=variance.device)
        threshold=table[indices]
        use_bypass=(snr>threshold+BOUNDARY_TOLERANCE_DB)&(snr>ORDINARY_MAX_DB+BOUNDARY_TOLERANCE_DB)
        return dict(bypass=use_bypass,filtered=~use_bypass,snr_proxy_db=snr,
            rsr_proxy_db=rsr,rsr_bin_db=bins[indices],cutoff_db=threshold)

    @torch.no_grad()
    def estimate(self,obs,model_name,policy,fallback='same_steps'):
        if fallback not in ('same_steps','em100'):raise ValueError('unknown fallback branch')
        if model_name not in self.models:raise KeyError(model_name)
        use_bypass=self.masks(obs,policy)['bypass']
        if not torch.any(use_bypass):
            # Exact original invocation, including batch shape and arithmetic.
            return self.original(obs,model_name)
        choose_fallback=lambda subset:self.unfiltered(subset,model_name) if fallback=='same_steps' else self.em100(subset)
        if torch.all(use_bypass):return choose_fallback(obs)
        result=torch.empty((len(obs),obs.z.shape[1],obs.s.shape[1]),dtype=obs.s.dtype,device=obs.s.device)
        filtered=~use_bypass
        result[filtered]=self.original(obs.batch(filtered),model_name)
        result[use_bypass]=choose_fallback(obs.batch(use_bypass))
        return result


def load_wrappers(models=None):
    return BypassEstimator(models=models)
