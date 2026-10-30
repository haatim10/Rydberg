"""Depth/width controls and an explicitly labelled two-expert controller."""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn
from sahure.model import AdaptiveHankelEstimator,ModelConfig


@dataclass(frozen=True)
class ReviewConfig:
    stages: int = 10
    width: int = 16
    controller: str = 'plain'


def network(width,relative=.1):
    net=nn.Sequential(nn.Linear(8,width),nn.SiLU(),nn.Linear(width,2))
    nn.init.zeros_(net[-1].weight)
    with torch.no_grad():
        net[-1].bias.copy_(torch.tensor([math.log(math.expm1(relative-.001)),math.log(.1/.9)]))
    return net


class TwoExpertController(nn.Module):
    """Mix two strength-controller outputs using observed spectral features.

    These are not independently recovered physical channel components. Both
    experts see the same eight receiver-available features and are trained
    jointly by withheld-observation likelihood, without rank/channel labels.
    """
    def __init__(self,width):
        super().__init__()
        self.experts=nn.ModuleList([network(width,.05),network(width,.2)])
        self.gate=nn.Linear(3,2)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)

    def forward(self,features):
        mixing=torch.softmax(self.gate(features[...,:3]),dim=-1)
        raw=torch.stack([net(features) for net in self.experts],dim=-2)
        return (mixing[...,None]*raw).sum(-2)


class ReviewEstimator(AdaptiveHankelEstimator):
    def __init__(self,config=ReviewConfig()):
        super().__init__(ModelConfig(core='em',structure='adaptive',stages=config.stages))
        self.review_config=config
        if config.controller=='experts': self.net=TwoExpertController(config.width)
        elif config.controller=='plain':
            if config.width!=16:self.net=network(config.width)
        else: raise ValueError(config.controller)


def load_model(path,dtype=torch.float64):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=ReviewEstimator(ReviewConfig(**ck['review_config'])).to(dtype=dtype)
    model.load_state_dict(ck['model']);return model.eval(),ck


ARMS=dict(baseline=ReviewConfig(),depth5=ReviewConfig(stages=5),
          depth20=ReviewConfig(stages=20),width8=ReviewConfig(width=8),
          width32=ReviewConfig(width=32),experts=ReviewConfig(controller='experts'),
          curriculum=ReviewConfig())
