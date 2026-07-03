import numpy as np
from sde.ode_sampler import score_model
from sde.schedule import VPSchedule

def pc_sampler(model, x, timesteps, dt, corrector_steps = 1, alpha = 0.1): 

    for t in timesteps: 
        score = model(x, t)
        drift = -0.5 * t * score
        diffusion = np.sqrt(t) * np.random.randn(x.shape)
        x = x + drift * dt + diffusion * np.sqrt(dt)

        for _ in range(corrector_steps): 
            score = model(x, t)
            noise = np.random.randn(x.shape)
            x = x + alpha * score + np.sqrt(2 * alpha) * noise


    return x
