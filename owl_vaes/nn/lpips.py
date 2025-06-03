import torch
from torch import nn
import torch.nn.functional as F
from convnext_perceptual_loss import ConvNextPerceptualLoss, ConvNextType

from lpips import LPIPS # Make sure this is my custom LPIPS, not pip install LPIPS
from .augs import PairedRandomAffine

def get_lpips_cls(lpips_id):
    if lpips_id == "vgg":
        return VGGLPIPS
    elif lpips_id == "convnext":
        return ConvNextLPIPS
        
# vgg takes 224 sized images
def vgg_patchify(x):
    _,_,h,_ = x.shape
    if h != 512: x = F.interpolate(x, (512, 512), mode='bicubic', align_corners=True)

    tl = x[:,:,:224,:224]
    tr = x[:,:,:224,-224:]
    bl = x[:,:,-224:,:224]
    br = x[:,:,-224:,-224:]
    return torch.cat([tl, tr, bl, br], dim=0)

class VGGLPIPS(nn.Module):
    def __init__(self):
        super().__init__()

        self.aug = PairedRandomAffine()
        self.model = LPIPS(net='vgg')

    def forward(self, x_fake, x_real):
        x_fake, x_real = self.aug(x_fake, x_real)
        _,_,h,_ = x_fake.shape
        if h > 224:
            x_fake = vgg_patchify(x_fake)
            x_real = vgg_patchify(x_real)

        return self.model(x_fake, x_real).mean()

def cn_patchify(x):
    _, _, h, w = x.shape
    if h <= 256 and w <= 256:
        return x
        
    # Use 256x256 patches with 128 pixel overlap
    patches = []
    stride = 128  # Half of patch size for 50% overlap
    
    for i in range(0, h-256+1, stride):
        for j in range(0, w-256+1, stride):
            patch = x[:, :, i:i+256, j:j+256]
            patches.append(patch)
            
    return torch.cat(patches, dim=0)
    
class ConvNextLPIPS(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = ConvNextPerceptualLoss(
            model_type=ConvNextType.Base,
            device='cpu',
            feature_layers=[0,2,4,6,8,12,14],
            use_gram=False,
            layer_weight_decay=0.99
        )

    def forward(self, fake, real):
        fake = cn_patchify(fake)
        real = cn_patchify(real)
        return self.loss(fake, real)
        
        