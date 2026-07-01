import numpy as np

def softmax(x): 
    x = x - np.max(x, axis = -1, keepdims = True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis = -1, keepdims = True)

def forward(X, W, b): 
    z = np.dot(X, W) + b
    sigma = softmax(z)
    return z, sigma

def multiclass_ce(y, sigma): 
    return -np.sum(y * np.log(sigma + 1e-12)) / y.shape[0]

def softmax_grad(sigma, y): 
    return sigma - y

def param_gradient(dz, X): 
    w_grad = X.T @ dz / X.shape[0]
    b_grad = np.sum(dz, axis = 0) / X.shape[0]
    return w_grad, b_grad

def train(X, y, W, b, optimizer, opt_state, epochs): 
    for epoch in range(epochs): 
        z, sigma = forward(X, W, b)
        loss multiclass_ce(y, sigma)
        dz = softmax_grad(sigma, y)
        b_grad, w_grad = param_gradient(dz, X)
        W, b, opt_statev = optimizer.update(W, b, w_grad, b_grad, opt_state)

        print(f"epoch {epoch}, loss = {loss}")
    return W, b
