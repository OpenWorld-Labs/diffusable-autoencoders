import argparse
import os
import torch

from owl_vaes.configs import Config
from owl_vaes.trainers import get_trainer_cls
from owl_vaes.utils.ddp import cleanup, setup

if __name__ == "__main__":
    # torch compile flag to convert conv with 1x1 kernel to matrix multiplication
    torch._inductor.config.conv_1x1_as_mm = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    torch.backends.cudnn.benchmark = True
    # # our LogHelper uses direct tensor data (scalar) access with .item() so this makes torch dynamo try and optimize it.^M
    # torch._dynamo.config.capture_scalar_outputs = ^M
    # enable if logging doesn't happen with different shape tensors, otherwise recompilation adds overhead.^M


    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, help="Path to config YAML file")
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config_path)

    global_rank, local_rank, world_size = setup()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = f"cuda:{local_rank}" if world_size > 1 else "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    trainer = get_trainer_cls(cfg.train.trainer_id)(
        cfg.train, cfg.wandb, cfg.model, global_rank, local_rank, world_size, device
    )

    trainer.train()
    cleanup()
