import numpy as np

def linear(X, w, b): 
    return X @ W + b

def linear_gradients(X, dz): 
    w_grad = X.T @ dz
    b_grad = np.sum(dz, axis = 0)
    return w_grad, b_grad

def relu(x): 
    return np.maximum(0, x)

def relu_grad(z): 
    return (z > 0).astype(float)
