import numpy as np

def make_moons(n_samples = 2000, noise = 0.05): 

    theta = np.random.rand(n_samples) * np.pi
    x1 = np.stack([np.cos(theta), np.sin(theta)], axis = 1)
    x2 = np.stack([1 - np.cos(theta), 1 - np.sin(theta) - 0.5], axis = 1)

    X = np.concatenate([x1, x2], axis = 0)
    X += noise * np.random.randn(X.shape)

    return X
    
