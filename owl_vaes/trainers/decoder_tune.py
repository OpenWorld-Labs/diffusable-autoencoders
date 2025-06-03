"""
Trainer for reconstruction only
"""

import einops as eo
import torch
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import get_loader
from ..models import get_model_cls
from ..discriminators import get_discriminator_cls
from ..muon import init_muon
from ..nn.lpips import VGGLPIPS
from ..schedulers import get_scheduler_cls
from ..utils import Timer, freeze, unfreeze, versatile_load
from ..utils.logging import LogHelper, to_wandb
from .base import BaseTrainer
from ..configs import Config

def latent_reg_loss(z):
    # z is [b,c,h,w]
    loss = z.pow(2)
    loss = eo.reduce(loss, 'b ... -> b', reduction = 'sum').mean()
    return 0.5 * loss

class DecTuneTrainer(BaseTrainer):
    """
    Trainer for only the decoder, with frozen encoder.
    Does L2 + LPIPS + GAN

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        teacher_ckpt_path = self.train_cfg.teacher_ckpt
        teacher_cfg_path = self.train_cfg.teacher_cfg

        teacher_ckpt = versatile_load(teacher_ckpt_path)
        teacher_cfg = Config.from_yaml(teacher_cfg_path).model

        teacher = get_model_cls(teacher_cfg.model_id)(teacher_cfg)
        teacher.load_state_dict(teacher_ckpt)

        del teacher.decoder
        self.encoder = teacher.encoder

        model_id = self.model_cfg.model_id
        model = get_model_cls(model_id)(self.model_cfg)
        del model.encoder
        self.model = model.decoder

        disc_cfg = self.model_cfg.discriminator
        self.discriminator = get_discriminator_cls(disc_cfg.model_id)(disc_cfg)

        if self.rank == 0:
            model_params = sum(p.numel() for p in self.model.parameters())
            disc_params = sum(p.numel() for p in self.discriminator.parameters())
            print(f"Model parameters: {model_params:,}")
            print(f"Discriminator parameters: {disc_params:,}")

        self.ema = None
        self.opt = None
        self.d_opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self):
        if self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
        else:
            return

        self.model.load_state_dict(save_dict['model'])
        self.discriminator.load_state_dict(save_dict['discriminator'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Loss weights
        lpips_weight = self.train_cfg.loss_weights.get('lpips', 0.0)
        gan_weight = self.train_cfg.loss_weights.get('gan', 0.1)

        # Prepare model, lpips, ema
        self.model = self.model.cuda().train()
        if self.world_size > 1:
            self.model = DDP(self.model)
        
        self.discriminator = self.discriminator.cuda().train()
        if self.world_size > 1:
            self.discriminator = DDP(self.discriminator)
        freeze(self.discriminator)

        lpips = None
        if lpips_weight > 0.0:
            lpips = VGGLPIPS().cuda().eval()
            freeze(lpips)

        self.encoder = self.encoder.cuda().bfloat16().eval()
        freeze(self.encoder)

        #self.encoder = torch.compile(self.encoder)
        #self.lpips.model = torch.compile(self.lpips.model)

        self.ema = EMA(
            self.model,
            beta = 0.9999,
            update_after_step = 0,
            update_every = 1
        )

        # Set up optimizer and scheduler
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank,world_size=self.world_size,**self.train_cfg.opt_kwargs)
            self.d_opt = init_muon(self.discriminator, rank=self.rank,world_size=self.world_size,**self.train_cfg.opt_kwargs)
        else:
            opt_cls = getattr(torch.optim, self.train_cfg.opt)
            self.opt = opt_cls(self.model.parameters(), **self.train_cfg.opt_kwargs)
            self.d_opt = opt_cls(self.discriminator.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast(f'cuda:{self.local_rank}', torch.bfloat16)

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size)

        def warmup_gan_weight():
            if self.total_step_counter < self.train_cfg.delay_adv:
                return 0.0
            else:
                ramp = (self.total_step_counter - self.train_cfg.delay_adv) / self.train_cfg.warmup_adv
                ramp = max(0.0, ramp)
                ramp = min(1.0, ramp)
                return ramp * gan_weight

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to('cuda').bfloat16()

                with torch.no_grad():
                    teacher_z = self.encoder(batch) / self.train_cfg.latent_scale

                with ctx:
                    batch_rec = self.model(teacher_z)

                # Discriminator training 
                unfreeze(self.discriminator)
                with ctx:
                    disc_loss = self.discriminator(batch_rec.detach(), batch.detach()) / accum_steps
                metrics.log('disc_loss', disc_loss)
                self.scaler.scale(disc_loss).backward()
                freeze(self.discriminator)

                mse_loss = F.mse_loss(batch_rec, batch) / accum_steps
                total_loss += mse_loss
                metrics.log('mse_loss', mse_loss)

                if lpips_weight > 0.0:
                    with ctx:
                        lpips_loss = lpips(batch_rec, batch) / accum_steps
                    total_loss += lpips_loss
                    metrics.log('lpips_loss', lpips_loss)
                
                crnt_gan_weight = warmup_gan_weight()
                if crnt_gan_weight > 0.0:
                    with ctx:
                        gan_loss = self.discriminator(batch_rec) / accum_steps
                    metrics.log('gan_loss', gan_loss)
                    total_loss += crnt_gan_weight * gan_loss

                self.scaler.scale(total_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.unscale_(self.d_opt)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)

                    self.scaler.step(self.d_opt)
                    self.d_opt.zero_grad(set_to_none=True)
                    
                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    # Do logging stuff with sampling stuff in the middle
                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict['time'] = timer.hit()
                        wandb_dict['lr'] = self.opt.param_groups[0]['lr']
                        timer.reset()

                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            with ctx:
                                ema_rec = self.ema.ema_model(teacher_z)
                            wandb_dict['samples'] = to_wandb(
                                batch.detach().contiguous().bfloat16(),
                                ema_rec.detach().contiguous().bfloat16(),
                                gather = False
                            )
                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()
