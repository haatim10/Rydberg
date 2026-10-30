"""Mask controller inputs only; retain the exact frozen estimator equations."""
import torch
from torch import nn
from sahure.model import AdaptiveHankelEstimator, ModelConfig

MASKS = {
    'full': (1,1,1,1,1,1,1,1),
    'spectral': (1,1,1,0,0,1,1,1),
    'measurement': (0,0,0,1,1,1,1,1),
    'context': (0,0,0,0,0,1,1,1),
}

class MaskedController(nn.Sequential):
    def __init__(self, arm, layers):
        super().__init__(*layers)
        self.register_buffer('feature_mask', torch.tensor(MASKS[arm], dtype=torch.float32), persistent=False)

    def forward(self, features):
        return super().forward(features*self.feature_mask)

class FeatureEstimator(AdaptiveHankelEstimator):
    def __init__(self, arm):
        if arm not in MASKS:
            raise ValueError(arm)
        super().__init__(ModelConfig(core='em', structure='adaptive', stages=10))
        self.arm = arm
        self.net = MaskedController(arm, list(self.net.children()))

def load_model(path, dtype=torch.float64):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = FeatureEstimator(state['arm']).to(dtype=dtype)
    model.load_state_dict(state['model'])
    model.eval()
    return model, state
