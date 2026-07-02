import numpy as np
from models.neural_network import (
    generalized_forward,
    generalized_backward,
    generalized_param_gradients
)
from optimizers.pipeline import OptimizerPipeline
from sde.schedule import VPSchedule
from ddpm.timestep_embedding import timestep_embedding  # reuse your existing one

def score_model(x_t, t, layers, time_dim):
    emb_t = timestep_embedding(t, time_dim)  # (batch, time_dim)
    inp = np.concatenate([x_t, emb_t], axis=-1)
    s_hat, cache = generalized_forward(inp, layers)
    return s_hat, cache

def score_training_step(batch_x0, layers, optimizer, opt_state, schedule, time_dim):
    batch_size = batch_x0.shape[0]

    # sample t ~ Uniform(0,1)
    t = np.random.uniform(0.0, 1.0, size=(batch_size,))
    eps = np.random.normal(size=batch_x0.shape)

    # compute alpha(t), sigma(t)
    alpha_t = schedule.alpha(t)[:, None]
    sigma_t = schedule.sigma(t)[:, None]

    # construct x(t)
    x_t = alpha_t * batch_x0 + sigma_t * eps

    # forward: score network
    s_hat, cache = score_model(x_t, t, layers, time_dim)

    # target score: -eps / sigma(t)
    target = -eps / sigma_t

    # loss: MSE between s_hat and target
    diff = s_hat - target
    loss = np.mean(diff**2)

    # backward using your engine
    grads = generalized_backward(s_hat, target, layers, cache)
    w_grads, b_grads = generalized_param_gradients(layers, cache, grads)

    # optimizer update via pipeline
    for i, layer in enumerate(layers):
        W, b = layer["W"], layer["b"]
        W, b, opt_state = optimizer.update(W, b, w_grads[i], b_grads[i], opt_state)
        layer["W"], layer["b"] = W, b

    return loss, opt_state

def train_score_sde(data_X, layers, optimizer, opt_state, epochs, steps_per_epoch, time_dim):
    schedule = VPSchedule()

    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            batch_x0 = data_X  # start with full batch; later you can batchify
            loss, opt_state = score_training_step(
                batch_x0, layers, optimizer, opt_state, schedule, time_dim
            )
        print(f"epoch {epoch}: loss = {loss:.6f}")

    return layers, opt_state
