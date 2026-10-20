import math

import wandb


def adjust_learning_rate(cfg, optimizer, glo_epoch, rank):
    """Decay the learning rate with half-cycle cosine after warmup"""
    lr = cfg.lr
    if glo_epoch < cfg.warmup_epochs:
        _lr = lr * glo_epoch / cfg.warmup_epochs
    else:
        _lr = cfg.min_lr + (lr - cfg.min_lr) * 0.5 * \
              (1. + math.cos(math.pi * (glo_epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = _lr * param_group["lr_scale"]
        else:
            param_group["lr"] = _lr

    if rank == 0:
        wandb.log({'lr': _lr})
