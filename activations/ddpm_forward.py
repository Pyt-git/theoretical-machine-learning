import numpy as np

T = 100
beta_min, beta_max = 1e-4, 0.02
betas = np.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)

def sample_data(n): 
    return np.random.normal(loc = 0.0, scale = 1.0, size = (n, 2))

def forward_noise(x0, t): 
    ab = alpha_bar[t]
    eps = np.random.normal(size = x0.shape)
    x_t = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
    return x_t
    
