import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from ..nn.attn import StackedTransformer, PatchProjIn, PatchProjOut
from ..nn.embeddings import LearnedPosEnc

class Encoder(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        n_tokens = (config.sample_size//config.patch_size)**2
        n_latents = config.latent_size

        self.proj_in = PatchProjIn(config.d_model, config.channels, config.patch_size)
        self.pos_enc = LearnedPosEnc(n_tokens+n_latents, config.d_model)
        self.latent_tokens = nn.Parameter(torch.randn(n_latents,config.d_model)*0.02)
    
        self.transformer = StackedTransformer(config)
        self.proj_out = nn.Linear(config.d_model, config.latent_channels, bias=False)
    
    def forward(self, x):
        x = self.proj_in(x)
        
        b,n,d = x.shape
        z = eo.repeat(self.latent_tokens, 'n d -> b n d', b = b)
        n_latents = z.shape[1]
        x = torch.cat([z,x], dim = 1)
        x = self.pos_enc(x)

        x = self.transformer(x)
        z = x[:,:n_latents]
        z = self.proj_out(z)

        return z

class Decoder(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        n_tokens = (config.sample_size//config.patch_size)**2
        n_latents = config.latent_size

        self.proj_out = PatchProjOut(config.sample_size, config.d_model, config.channels, config.patch_size)
        self.pos_enc = LearnedPosEnc(n_tokens+n_latents, config.d_model)
        self.image_tokens = nn.Parameter(torch.randn(n_tokens,config.d_model)*0.02)
    
        self.transformer = StackedTransformer(config)
        self.proj_in = nn.Linear(config.latent_channels, config.d_model, bias=False)
    
    def forward(self, z):
        z = self.proj_in(z) # [b,n,d]

        b,n_latents,d = z.shape
        x = eo.repeat(self.image_tokens, 'n d -> b n d', b = b)
        x = torch.cat([z,x], dim = 1)
        x = self.pos_enc(x)

        x = self.transformer(x)
        x = x[:,n_latents:]
        x = self.proj_out(x)
        
        return x

class TiToKVAE(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.config = config

    def forward(self, x):
        z = self.encoder(x)
        if self.config.noise_decoder_inputs > 0.0:
            dec_input = z + torch.randn_like(z) * self.config.noise_decoder_inputs
        else:
            dec_input = z.clone()
        rec = self.decoder(dec_input)
        return rec, z

if __name__ == "__main__":
    from ..configs import TransformerConfig

    cfg = TransformerConfig(
        sample_size = 16,
        channels = 32,
        latent_size = 16,
        latent_channels = 128,
        n_layers = 6,
        n_heads = 6,
        d_model = 384,
        patch_size = 1
    )

    model = TiToKVAE(cfg).float().cuda()

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        x = torch.randn(1, 32, 16, 16, device='cuda', dtype=torch.bfloat16)
        rec, z = model(x)
        
        print(f'Input shape: {x.shape}, dtype: {x.dtype}')
        print(f'Latent shape: {z.shape}, dtype: {z.dtype}') 
        print(f'Output shape: {rec.shape}, dtype: {rec.dtype}')
