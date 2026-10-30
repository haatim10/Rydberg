"""Receiver-observable continuation, channel blending, and final shrinkage."""
import json
import torch
from tcom_dual_guard_20261019 import inference as previous
from tcom_pilot_em_20261018 import continuation
from tcom_followups_20261017.common import REPO
from . import shrink

SNRS=previous.SNRS;RSRS=previous.RSRS;SEEDS=previous.SEEDS
OLD_POLICY=previous.OLD_POLICY;model_paths=previous.model_paths
receiver_keys=previous.receiver_keys;choices_for=previous.choices_for
CURRENT_POLICY_PATH=REPO/'results/tcom_dual_guard_20261019/policy.json'
CURRENT_POLICY=json.loads(CURRENT_POLICY_PATH.read_text())
LENGTHS=(0,1,2,3,4,5,8,10,12,20,40,100)
PAIRS=(('original','warm3'),('warm2','warm3'),('warm5','warm20'),
       ('warm20','spectral100'),('warm5','spectral100'))
CONFIGS={('original' if n==0 else f'warm{n}'):dict(kind='continuation',steps=n) for n in LENGTHS}
CONFIGS['spectral100']=dict(kind='spectral',steps=100)
for left,right in PAIRS:
    for alpha in (.25,.5,.75):
        CONFIGS[f'blend_{left}_{right}_{alpha:g}']=dict(kind='convex_channel_blend',left=left,right=right,weight_second=alpha)
CONFIGS.update(shrink.CONFIGS)
MENU=tuple(CONFIGS)


class RepairEstimator(previous.RepairEstimator):
    @torch.no_grad()
    def values(self,obs,model_name,names,initial=None,spectral=None):
        """Share one neural start/continuation trajectory within a call."""
        requested=tuple(dict.fromkeys(names));needed=set(requested)
        for name in requested:
            if name not in CONFIGS:raise ValueError(name)
            cfg=CONFIGS[name]
            if cfg['kind']=='convex_channel_blend':needed.update((cfg['left'],cfg['right']))
        current_names=None
        if any(CONFIGS[n]['kind']=='noise_scaled_hankel' for n in needed):
            current_names=choices_for(obs,CURRENT_POLICY['repair_choices'])
            if len(set(current_names))==1:needed.add(current_names[0])
        cache={}
        lengths=sorted(set(CONFIGS[n]['steps'] for n in needed if CONFIGS[n]['kind']=='continuation'))
        if lengths:
            initial=self.original(obs,model_name) if initial is None else initial
            trajectory,_=continuation.trajectory(obs,initial,lengths)
            cache.update({('original' if n==0 else f'warm{n}'):g for n,g in trajectory.items()})
        if 'spectral100' in needed:cache['spectral100']=self.em100(obs) if spectral is None else spectral
        for name in requested:
            cfg=CONFIGS[name]
            if cfg['kind']=='convex_channel_blend':
                alpha=cfg['weight_second'];cache[name]=(1-alpha)*cache[cfg['left']]+alpha*cache[cfg['right']]
            elif cfg['kind']=='noise_scaled_hankel':
                current=cache[current_names[0]] if len(set(current_names))==1 else self.repair(obs,model_name,CURRENT_POLICY['repair_choices'])
                cache[name]=shrink.apply(obs,current,cfg['c'],cfg['beta'])
        return {name:cache[name] for name in requested}

    @torch.no_grad()
    def candidate(self,obs,model_name,choice):
        return self.values(obs,model_name,(choice,))[choice]

    @torch.no_grad()
    def candidate_predictions(self,obs,model_name,spectral=None):
        return self.values(obs,model_name,MENU,spectral=spectral)
