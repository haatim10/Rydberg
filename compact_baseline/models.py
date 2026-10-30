import json
import torch
from sahure.train import load_model
from .model import RecurrentHankelEstimator
from .protocol import ROOT, CONFIG, sha


def load_recurrent(path,dtype=torch.float64):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=RecurrentHankelEstimator(**state['model_config']).to(dtype=dtype)
    model.load_state_dict(state['model']);model.eval()
    return model,state


def load_models():
    frozen=json.loads((ROOT/'frozen_selection.json').read_text());out={}
    for name,entry in frozen['models'].items():
        if sha(entry['path'])!=entry['sha256']: raise RuntimeError('checkpoint changed: '+name)
        m,_=load_recurrent(entry['path']) if name.startswith('recurrent') else load_model(entry['path'])
        count=sum(p.numel() for p in m.parameters())
        if count!=dict(none=10,global_=30,adaptive=188,recurrent=148).get(name.split('_')[0] if not name.startswith('global') else 'global_'):
            raise RuntimeError('unexpected parameter count: '+name)
        out[name]=m
    return out
