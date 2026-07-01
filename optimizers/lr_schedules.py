import numpy as np
class WarmupCosine: 
    def __init__(self, warmup_steps, total_steps, base_lr): 
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr

    def compute_lr(self, opt_state): 
        t = opt_state.t

        if t < self.warmup_steps: 
            return self.base_lr * (t / self.warmup_steps)

        progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return 0.5 * self.base_lr * (1 + 
