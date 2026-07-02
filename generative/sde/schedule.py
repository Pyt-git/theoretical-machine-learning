import numpy as np

class VPSchedule:
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha(self, t):
        # approximate integral of beta(s) ds from 0 to t
        # for linear beta: ∫0^t beta(s) ds = beta_min * t + 0.5 * (beta_max - beta_min) * t^2
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        return np.exp(-0.5 * integral)

    def sigma(self, t):
        a = self.alpha(t)
        return np.sqrt(1.0 - a**2)

    def g(self, t):
        # diffusion coefficient for VP SDE
        return np.sqrt(self.beta(t))
    
    def f(self, x, t):
        # drift for VP SDE
        return -0.5 * self.beta(t) * x
