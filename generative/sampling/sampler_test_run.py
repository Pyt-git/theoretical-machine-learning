import numpy as np
from pc_sampler import pc_sampler
from rk4_ode_solver import rk4_sampler

samples_pc = pc_sampler(score_model, x0, schedule.timesteps, schedule.dt)
samples_rk4 = rk4_sampler(score_model, x0, schedule.timesteps, schedule.dt)

print(samples_pc.shape, samples_rk4.shape)
print(np.isnan(samples_pc).any(), np.isnan(samples_rk4).any())

