"""Independent implementation of the explicit equations in URformer v3.

Deliberately separate from the earlier geometric-channel adaptation.
Unspecified choices are enumerated in protocol.json, not attributed to authors.
"""
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(64)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True, dropout=0.)
        self.ffn = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))

    def forward(self, x):
        h = self.ln(x)
        a = x + self.attn(h, h, h, need_weights=False)[0]
        # The printed equation is FFN(A)+A: no second or output LayerNorm.
        return a + self.ffn(a)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Sequential(nn.Linear(1, 32), nn.ReLU(),
                                    nn.Linear(32, 1), nn.Sigmoid())
        self.gate = nn.Parameter(torch.zeros(()))
        self.embed = nn.Linear(64, 64)
        self.position = nn.Parameter(torch.empty(4, 64))
        nn.init.normal_(self.position, std=.02)
        self.encoders = nn.Sequential(*[Encoder() for _ in range(3)])
        self.out = nn.Linear(64, 64)

    def forward(self, h, z, s, b, variance, pinv, classical=None):
        y = h @ s + b
        direct = z * y / y.abs().clamp_min(1e-8)
        kappa = 2*z*y.abs()/variance[:, None, None]
        if classical is None:
            r = self.filter(kappa[..., None]).squeeze(-1)
            alpha = self.gate.sigmoid()
        else:
            r = (torch.special.i1e(kappa)/torch.special.i0e(kappa)
                 if classical == 'em' else torch.ones_like(kappa))
            alpha = 1.
        linear = (direct * (1-alpha+alpha*r) - b) @ pinv
        if classical is not None:
            return linear
        tokens = torch.cat((linear.real.transpose(1, 2),
                            linear.imag.transpose(1, 2)), -1)
        residual = self.out(self.encoders(self.embed(tokens) + self.position))
        return linear + torch.complex(residual[..., :32], residual[..., 32:]).transpose(1, 2)


class URformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(10)])

    def forward(self, h, z, s, b, variance, pinv):
        for layer in self.layers:
            h = layer(h, z, s, b, variance, pinv)
        return h


def ratios(pred, target):
    return (pred-target).abs().square().sum((-2, -1))/target.abs().square().sum((-2, -1))
