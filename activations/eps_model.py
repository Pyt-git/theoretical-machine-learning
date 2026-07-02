import numpy as np
from timestep_embeddings import timestep_embedding
from models.neural_network import generalized_forward

def eps_model(x_t, t, layers): 

    emb_t = timestep_embedding(t, time_dim)
    inp = np.concatenate([x_t, emb_t], axis = -1)
    eps_hat, _ = generalized_forward(inp, layers)

    return eps_hat
