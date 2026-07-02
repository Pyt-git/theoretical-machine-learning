import numpy as np
from models.neural_network import generalized_forward, generalized_backward, generalized_param_gradients
from eps_model import eps_model
from optimizers.pipeline import OptimizerPipeline

def ddpm_training_step(batch_x0, layers, optimizer, opt_state):
    batch_size = batch_x0.shape[0]
    t = np.random.randint(0, T, size = (batch_size,))
    ab = alpha_bar[t][:, None]

    eps = np.random.normal(size = batch_x0.shape)
    x_t = np.sqrt(ab) * batch_x0 + np.sqrt(1.0 - ab) * eps

    eps_hat = eps_model(x_t, t, layers)

    loss = np.mean((eps - eps_hat) ** 2)

    grads = generalized_backward(eps_hat, eps, layers, {"a": [], "z": []})
    w_grads, b_grads = generalized_param_gradients(layers, {"a": [], "z": []}, grads)

    for i, layer in enumerate(layers): 
        W, b = layer["W"], layer["b"]
        W, b, opt_state = optimizer.update(W, b, w_grads[i], b_grads[i], opt_state)
        layer["W"], layer["b"] = W, b

    return loss, opt_state
    


