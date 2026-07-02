import numpy as np
from linear_layer import linear, relu, relu_grad

def forward(X, W1, b1, W2, b2): 
    z1 = linear(X, W1, b1)
    a1 = relu(z1)

    z2 = linear(a1, W2, b2)
    return z1, a1, z2

def mse_loss(y, y_hat): 
    return np.mean((y_hat - y) ** 2)

def output_grad(y_hat, y): 
    return 2 * (y_hat - y) / y.shape[0]

def hidden_grad(dz2, W2, z1):
    da1 = dz2 @ W2.T
    dz1 = da1 * relu_grad(z1)
    return dz1

def param_gradients(X, a1, dz1, dz2): 
    w1_grad = X.T @ dz1
    b1_grad = np.sum(dz1, axis = 0)

    w2_grad = a1.T @ dz2
    b2_grad = np.sum(dz2, axis = 0)

    return w1_grad, b1_grad, w2_grad, b2_grad

def train(X, y, W1, b1, W2, b2, optimizer, opt_state, epochs):2
    for _ in range(epochs): 
        z1, a1, z2 = forward(X, W1, b1, W2, b2)
        loss = mse_loss(y, z2)

        dz2 = output_grad(z2, y)
        dz1 = hidden_grad(dz2, W2, z1)

        w1_grad, b1,_grad, w2_grad, b2_grad = param_gradients(X, a1, dz1, dz2)

  W1, b1, opt_state = optimizer.update(W1, b1, w1_grad, b1_grad, opt_state)
  W2, b2, opt_state = optimizer.update(W2, b2, w2_grad, b2_grad, opt_state)
  
  return W1, b1, W2, b2

# Generalized MLP engine

layers = [
    {"W": W1, "b": b1, "activation": "relu"}, 
    {"W": W2, "b": b2, "activation": "relu"},
    {"W": W3, "b": b3, "activation": None}
]

def generalized_forward(X, layers): 
    cache = {"a": [X], "z": []}
    a = X

    for layer in layers: 
        W, b = layer["W"], layer["b"]
        z = a @ W + b
        cache["z"].append(z)

        if layer["activation'] == "relu": 
            a = np.maximum(0, z)
        else:  
            a = z
        cache["a"].append(a)
    return a, cache

def generalized_backward(y_hat, y, layers, cache): 
    grads = {"dz": [], "da": []}

    dz = 2 * (y_hat - y) / y.shape[0]
    grads["dz"].append(dz)

    for i in reversed(range(len(layers) - 1)): 
        W_next = layers[i + 1]["W"]

        z = cache["z"][i]
        da = dz @ W_next.T
        dz = da * (z > 0).astype(float)
        grads["dz"].append(dz)
        
    grads["dz"].reverse()
    return grads

def generalized_param_gradients(layers, cache, grads): 
    w_grads = []
    b_grads = []

    for i, layer in enumerate(layers): 
        a_prev = cache["a"][i]

        dz = grads["dz"][i]
        w_grad = a_prev.T @ dz
        b_grad = np.sum(dz, axis = 0)

        w_grads.append(w_grad)
        b_grads.append(b_grad)

    return w_grads, b_grads

def generalized_train(X, y, layers, optimizer, opt_state, epochs): 

    for _ in range(epochs): 
        y_hat, cache = forward(X,layers)

        grads = backward(y_hat, y, layers, cache)
        w_grads, b_grads = param_gradients(layers, cache, grads)

        for i, layer in enumrate(layers): 
            W, b = layer["W"], layer["b"]

            W, b, opt_state = optimizer.update(W, b, w_grads[i], b_grads[i], opt_state)
            layer["W"], layer["b"] = W, b

    return layers

def xavier_init(in_dim, out_dim): 
    limit = np.sqrt(6 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, (in_dim, out_dim))

def he_init(in_dim, out_dim): 
    std = np.sqrt(2 / in_dim)
    return np.random.randn(in_dim, out_dim) * std
