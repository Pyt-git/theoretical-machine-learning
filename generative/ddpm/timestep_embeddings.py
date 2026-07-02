import numpy as np

def timestep_embedding(t, dim, max_period = 10000): 

    if np.isscalar(t): 
        t = np.array([t])
    t = t.astype(np.float32)

    half_dim = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(0, half_dim) / half_dim)

    args = t[:, None] * freqs[None, :]
    embedding = np.concatenate([np.sin(args), np.cos(args)], axis = -1)

    return embedding
