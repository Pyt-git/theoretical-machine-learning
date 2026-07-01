import numpy as np
from models.sigmoid import sigmoid
from validate_y import validate_y

def bce_loss(y):
    eps = 1e-12
    sigma = np.clip(sigma, eps, 1 - eps)

    L = -[y * np.log(sigma) + (1 - y) * np.log(1 - sigma)]

    def gradient_bce(sigma, y): 
        return sigma - y 
    return L
