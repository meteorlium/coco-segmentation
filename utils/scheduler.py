import torch


def set_scheduler(optimizer, method):
    if method == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    elif method == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=2
        )
    else:
        raise RuntimeError("scheduler is wrong")
