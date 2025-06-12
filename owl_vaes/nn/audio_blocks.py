import math
from typing import Literal

import torch
from alias_free_torch import Activation1d
from torch import exp, nn, pow, sin
from torch.types import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .normalization import RMSNorm1d

# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py under MIT license
class SnakeBeta(nn.Module):
    def __init__(
        self, in_features: int, alpha=1.0, alpha_trainable=True
    ):
        super(SnakeBeta, self).__init__()

        self.in_features = in_features
        self.eps = 1.0e-6

        # Alpha initialized to zeros, will be used as (1 + alpha)
        self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        # Beta initialized to zeros, will be used as exp(beta)
        self.beta = nn.Parameter(torch.zeros(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def snake_beta(self, x, alpha, beta):
        alpha = alpha[None,:,None]
        beta = beta[None,:,None]
        return x + (1.0 / (self.eps + beta.exp())) * pow(sin(x * (1. + alpha)), 2)

    def forward(self, x):
        x = self.snake_beta(x, self.alpha, self.beta)
        return x

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)

class ResBlock(nn.Module):
    def __init__(self, ch, dilation, total_res_blocks):
        super().__init__()

        self.dilation = dilation
        self.p = (dilation * (7 - 1))//2

        grp_size = 16
        n_grps = (2*ch) // grp_size

        self.conv1 = nn.Conv1d(ch, 2*ch, 1, 1, 0)
        self.norm1 = RMSNorm1d(2*ch)

        self.conv2 = nn.Conv1d(2*ch, 2*ch, 7, dilation=dilation, padding=self.p, groups=n_grps)
        self.norm2 = RMSNorm1d(2*ch)

        self.conv3 = nn.Conv1d(2*ch, ch, 1, 1, 0, bias=False)

        self.act1 = SnakeBeta(2*ch)
        self.act2 = SnakeBeta(2*ch)

        # Fix up init
        scaling_factor = total_res_blocks ** -.25

        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.conv1.weight.data *= scaling_factor

        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.conv2.weight.data *= scaling_factor

        nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        res = x.clone()

        def _inner(x):
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = self.act2(x)
            x = self.conv3(x)
            return x

        if self.training:
            x = checkpoint(_inner, x)
        else:
            x = _inner(x)

        return x + res