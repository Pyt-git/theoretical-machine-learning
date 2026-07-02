import numpy as np

def sample_reverse_sde(n_samples, layers, schedule, time_dim, N_steps=1000, dim_x=2):
    x = np.random.normal(size=(n_samples, dim_x))  # start from noise at t=1
    t_grid = np.linspace(1.0, 0.0, N_steps)

    for k in range(N_steps - 1):
        t = t_grid[k]
        dt = t_grid[k+1] - t  # negative

        g_t = schedule.g(t)
        f_t = schedule.f(x, t)  # VP drift

        # score network
        t_batch = np.full((n_samples,), t)
        s_hat, _ = score_model(x, t_batch, layers, time_dim)

        drift = f_t - (g_t**2) * s_hat
        diffusion = g_t * np.sqrt(-dt) * np.random.normal(size=x.shape)

        x = x + drift * dt + diffusion

    return x
