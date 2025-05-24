"""
R3Gan style resnet based VAE
"""

from ..nn.resnet import DownBlock, SameBlock
from ..nn.normalization import GroupNorm

import torch.nn.functional as F
import torch
from torch import nn

import einops as eo

class ResDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        size = config.sample_size
        ch_0 = config.ch_0
        ch_max = config.ch_max
        blocks_per_stage = config.blocks_per_stage

        self.conv_in = nn.Conv2d(3, ch_0, 1, 1, 0, bias = False)

        # Count number of downsampling stages needed to get from sample_size to 4
        total_blocks = len([size // (2**i) for i in range(100) if size // (2**i) >= 4])
        # Add 1 for final layer
        total_blocks += 1
        blocks = []

        ch = ch_0
        while size > 4:
            next_ch = min(ch*2,ch_max)

            blocks.append(DownBlock(ch, next_ch, blocks_per_stage, total_blocks))
            ch = next_ch
            size = size // 2

        blocks.append(SameBlock(ch, ch_max, blocks_per_stage, total_blocks))
        self.blocks = nn.Sequential(*blocks)

        self.final = nn.Conv2d(ch_max, 1, 4, 1, 0)
    
    def _forward(self, x):
        # Forward on single sample
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.final(x)
        x = x.flatten(0)
        return x

    def forward(self, x_fake, x_real = None):
        if x_real is None:
            return -self._forward(x_fake).mean()
        else:
            fake_out = self._forward(x_fake)
            real_out = self._forward(x_real)
            
            fake_loss = F.relu(1 + fake_out).mean()
            real_loss = F.relu(1 - real_out).mean()
            
            return fake_loss + real_loss

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class DummyConfig:
        sample_size:int= 256
        ch_0:int= 32
        ch_max:int= 256
        blocks_per_stage:int= 2

    model = ResDiscriminator(DummyConfig()).bfloat16().cuda()
    with torch.no_grad():
        x = torch.randn(1,3,256,256).bfloat16().cuda()
        y = model._forward(x)
        print(y.shape)

