"""Matched related learned control; not a published-URformer reproduction."""
import torch
from sahure.math import spectral_initial, posterior_field, right_solve
from sahure.model import AdaptiveHankelEstimator
from trackD_urformer.transformer import ChannelTransformer


class ExactEMTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.formers = torch.nn.ModuleList([
            ChannelTransformer(32, 3, d_model=64, L_enc=3, n_heads=4,
                               ffn_mult=4, dropout=0., zero_init_out=True)
            for _ in range(10)])

    def forward(self, z, s, b, sigma2):
        g = spectral_initial(z, s, b)
        gram = s@s.mH
        for former in self.formers:
            q = posterior_field(z, g@s+b, sigma2)
            g = right_solve((q-b)@s.mH, gram)
            g = g + former(g)
        return g


def build(name, seed=202610087101, dtype=torch.float32):
    torch.manual_seed(seed)
    if name == 'compact':
        model = AdaptiveHankelEstimator()
    elif name == 'em_transformer':
        model = ExactEMTransformer()
    else:
        raise ValueError(name)
    return model.to(dtype=dtype)
