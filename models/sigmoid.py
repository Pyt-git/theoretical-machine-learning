import nmupy as np

def sigmoid(x):
    sigma = 1 / (1 + np.exp(-x))
    d_sigma = sigma * (1 - sigma)
    return sigma, d_sigma
