import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    CosineAnnealingWarmupRestarts scheduler.
    Paper: "SGDR: Stochastic Gradient Descent with Warm Restarts" + warmup extension.
    If warmup_steps is None or 0, it behaves like standard cosine annealing.
    If restart_period is None or 0, it does a single cosine decay (no restarts).
    """
    def __init__(self, optimizer, warmup_steps=None, restart_period=None, max_lr=1e-3, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps if warmup_steps is not None else 0
        self.restart_period = restart_period
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
        # Ensure max_lr is used as the base lr for warmup
        self.base_lrs = [max_lr for _ in optimizer.param_groups]

    def get_lr(self):
        step = self.last_epoch
        
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear warmup
            return [self.min_lr + (base_lr - self.min_lr) * step / self.warmup_steps for base_lr in self.base_lrs]
            
        if self.restart_period is None or self.restart_period <= 0:
            # No restarts, standard decay assuming total steps are unknown (we just decay forever or something)
            # Actually if restart_period is None, we just keep returning max_lr or we decay based on some max_steps?
            # The paper doesn't specify max_steps if restart_period is None, so we just return base_lr
            # to avoid crashing. 
            return self.base_lrs

        # With restarts
        step_in_period = (step - self.warmup_steps) % self.restart_period
        
        return [
            self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * step_in_period / self.restart_period))
            for base_lr in self.base_lrs
        ]
