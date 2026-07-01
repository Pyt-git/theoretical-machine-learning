from optimizers.pipeline import OptimizerPipeline
from optimizers.adamw import AdamW
from optimizers.weight_decay import WeightDecay
from optimizers.lr_schedules import WarmupCosine

optimizer = OptimizerPipeline(
  adamw = AdamW(beta1 = 0.9, beta2 = 0.999, eps = 1e-8),
  weight_decay = WeightDecay(wd = 0.01), 
  lr_schedules = WarmupCosine(warmup_steps = 100, total_steps = 1000, base_lr = 0.001)
)
