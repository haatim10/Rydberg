"""Receiver-only short EM continuations with a frozen per-bin policy."""
import torch
from tcom_transition_repair_20261019 import inference as previous
from tcom_pilot_em_20261018 import continuation

SNRS=previous.SNRS
RSRS=previous.RSRS
SEEDS=previous.SEEDS
OLD_POLICY=previous.OLD_POLICY
model_paths=previous.model_paths
receiver_keys=previous.receiver_keys
choices_for=previous.choices_for
LENGTHS=(0,1,2,3,5,10,20,100)
MENU=('original','warm1','warm2','warm3','warm5','warm10','warm20','warm100','spectral100')
PREFERENCE_ORDER=('original','warm1','warm2','warm3','warm5','spectral100','warm10','warm20','warm100')


class RepairEstimator(previous.RepairEstimator):
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
        trajectory,_=continuation.trajectory(obs,initial,LENGTHS)
        values={'original':initial,**{f'warm{n}':trajectory[n] for n in LENGTHS if n}}
        values['spectral100']=self.em100(obs) if spectral is None else spectral
        return values
