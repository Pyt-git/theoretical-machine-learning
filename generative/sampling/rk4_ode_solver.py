import numpy as
from sde.ode_sampler import score_model
from sde.schdule import VPSchedule

def ode_drift(model, x, t): 
    score = model(x, t)
    return -0.5 * t * score

def rk4_step(model, x, t, dt): 
    k1 = ode_drift(model, x, t)
    k2 = ode_drift(model, x + 0.5 * dt * k1, t - 0.5 * dt)
    k3 = ode_drift(model, x + 0.5 * dt * k2, t - 0.5 * dt)
    k4 = ode_drift(model, x + dt * k3, t - dt)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def rk4_sampler(model, x, timesteps, dt): 
    for t in timesteps: 
        x = rk4_step(model, x, t, dt)
    return x

# call: 
# samples = rk4_sampler(score_model, x0, schedule.timesteps, schedule.dt)
