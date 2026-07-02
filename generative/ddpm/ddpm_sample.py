import numpy as np
from eps_model import eps_model

def reverse_step_eps(xt,t, layers):
    beta_t = betas[t]
    alpha_t = alphas[t] 
    ab_t = alpha_bar[t]

    batch_size = xt.shape[0]
    tt = np.full((batch_size,), t)

    eps_hat = eps_model(xt, tt, layers)

    mean = (1.0 / np.sqrt(alpha_t)) * (xt - (beta_t / np.sqrt(1.0 - ab_t)) * eps_hat)

    if t > 0: 
        noise = np.random.normal(size = xt.shape)
        var = beta_t * (1.0 - alpha_bar[t - 1]) / (1.0 - ab_t)
        return mean + np.sqrt(var) * noise
    else: 
        return mean

def sample_reverse_eps(n_samples, layers): 
    xt = np.random.normal(size=(n_samples, 2))
    for t in reversed(range(T)): 
        xt = reverse_step_eps(xt, t, layers)
    return xt
        
