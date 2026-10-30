"""Receiver-only continuation from the frozen learned channel estimate."""
import torch
from sahure.train import load_model
from tcom_followups_20261017.common import REPO
from tcom_runtime_bypass_20261017.bypass import BypassEstimator
from tcom_pilot_em_20261018 import continuation

SNRS=(0.,5.,10.,15.,20.,22.5,25.,27.5,30.,32.5,35.,37.5,40.,42.5,45.,50.,60.)
RSRS=(5.,12.,20.);SEEDS=(9101,9102,9103)
MENU=('original','warm5','warm20','warm100','spectral100')
COST_ORDER=('original','warm5','warm20','spectral100','warm100')
OLD_POLICY={'5':35.,'12':30.,'20':30.}


def model_paths():
    return {f'learned_{s}':REPO/f'results/tcom_pilot_em_20261018/checkpoints/context30/seed_{s}/em_adaptive_best.pt' for s in SEEDS}


def receiver_keys(obs):
    snr=10*torch.log10(obs.s.shape[1]/obs.sigma2)
    rsr=10*torch.log10(obs.b.abs().square().mean((-2,-1)))
    if not torch.all(torch.isfinite(snr)&torch.isfinite(rsr)):raise ValueError('invalid receiver noise/reference')
    return [(f'{min(RSRS,key=lambda r:abs(r-float(ref))):g}',f'{min(SNRS,key=lambda s:abs(s-float(noise))):g}',float(noise)) for noise,ref in zip(snr,rsr)]


def choices_for(obs,table):
    return ['original' if snr<=20.+1e-10 else table[ref][key]['choice'] for ref,key,snr in receiver_keys(obs)]


class RepairEstimator(BypassEstimator):
    def __init__(self,models=None):
        if models is None:models={n:load_model(p,dtype=torch.float64)[0] for n,p in model_paths().items()}
        super().__init__(models)

    @torch.no_grad()
    def candidate(self,obs,model_name,choice):
        if choice not in MENU:raise ValueError(choice)
        if choice=='spectral100':return self.em100(obs)
        initial=self.original(obs,model_name)
        if choice=='original':return initial
        return continuation.estimate(obs,initial,int(choice[4:]))

    @torch.no_grad()
    def candidate_predictions(self,obs,model_name,spectral=None):
        initial=self.original(obs,model_name)
        warm,_=continuation.trajectory(obs,initial,(0,5,20,100))
        return dict(original=initial,warm5=warm[5],warm20=warm[20],warm100=warm[100],
            spectral100=self.em100(obs) if spectral is None else spectral)

    @torch.no_grad()
    def repair(self,obs,model_name,table):
        choices=choices_for(obs,table)
        if len(set(choices))==1:return self.candidate(obs,model_name,choices[0])
        result=torch.empty((len(obs),obs.z.shape[1],obs.s.shape[1]),dtype=obs.s.dtype,device=obs.s.device)
        for choice in sorted(set(choices)):
            mask=torch.tensor([c==choice for c in choices],device=obs.s.device)
            result[mask]=self.candidate(obs.batch(mask),model_name,choice)
        return result
