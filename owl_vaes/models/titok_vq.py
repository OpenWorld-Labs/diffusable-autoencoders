import torch
from .titok import TiToKVAE
from ..nn.lfq import LFQModule

@torch.compile(mode="max-autotune", fullgraph=True)
class TiToKVQVAE(TiToKVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lfq = LFQModule(self.config.codebook_size)

    def forward(self, x):
        z = self.encoder(x)
        dec_in = self.lfq(z)
        x_rec = self.decoder(dec_in)
        return x_rec, z
