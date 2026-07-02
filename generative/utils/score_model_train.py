import numpy as np
from ode_dataset import make_moons
from models.neural_network import he_init
from optimizers.pipeline import OptimizerPipeline

X = make_moons(2000)

# using function train_score_sde: 
layers = [
    {"W": he_init(2 + time_dim, 128), "b": np.zeros(128), "activation": "relu"}, 
    {"W": he_init(128, 128), "b": np.zeros(128), "activation": "relu"},
    {"W": he_init(128, 128), "b": np.zeros(2), "activation": None},
]

optimizer = OptimizerPipeline(lr = 1e-3)
opt_state = optimizer.init_state(layers)

layers, opt_score = train_score_sde(
    data_X = X,
    layers = layers, 
    optimizer = optimizer,
    opt_state = opt_state,
    epochs = 200
    steps_per_epoch = 1
    time_dim = 32
)    
