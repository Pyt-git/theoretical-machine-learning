import numpy as np
from sigmoid import sigmoid

def linear_model(x, W, b): 
    return np.dot(W, x) + b

    def logistic_forward(x, W, b): 
        z = linear_forward(x, W, b)
        sigma, d_sigma = sigmoid(z)
        return z, d_sigma
