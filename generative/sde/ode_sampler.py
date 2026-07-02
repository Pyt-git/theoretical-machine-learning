import numpy as np
from sde.schedule import VPSchedule
from ddpm.timestep_embedding import timestep_embedding
from models.neural_network import generalized_forward

def score_model(x_t, t, layers, time_dim):
    emb_t = timestep_embedding(t, time_dim)
    inp = np.concatenate([x_t, emb_t], axis=-1)
    s_hat, _ = generalized_forward(inp, layers)
    return s_hat

def sample_probability_flow_ode(n_samples, layers, schedule, time_dim, N_steps=1000, dim_x=2):
    # start from standard Gaussian at t = 1
    x = np.random.normal(size=(n_samples, dim_x))
    t_grid = np.linspace(1.0, 0.0, N_steps)

    for k in range(N_steps - 1):
        t = t_grid[k]
        dt = t_grid[k+1] - t  # negative

        g_t = schedule.g(t)
        f_t = schedule.f(x, t)  # VP drift: -0.5 * beta(t) * x

        t_batch = np.full((n_samples,), t)
        s_hat = score_model(x, t_batch, layers, time_dim)

        # probability flow ODE drift
        dx_dt = f_t - 0.5 * (g_t**2) * s_hat

        # simple Euler step
        x = x + dx_dt * dt

    return x
