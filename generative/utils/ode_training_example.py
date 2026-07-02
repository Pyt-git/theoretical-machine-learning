import matplotlib.pylot as plt

schedule = VPSchedule()

samples = sample_probability_flow_ode(
    n_samples = 2000,
    layers = layers, 
    schedule = schedule, 
    time_dim = 32, 
    N_steps = 2000, 
    dim_x = 2
)

plt.figure(figsize=(6.6))
plt.scatter(samples[:, 0], samples[:, 1], s = 5, alpha = 0.5)
plt.title("Probability Flow ODE Samples")
plt.show()
