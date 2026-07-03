import numpy as np

class ScoreModel: 
    def __init__(self, dim, hidden): 
        self.dim = dim
        self.hidden = hidden

        # Timestep embedding projections
        self.Wt1 = np.random.randn(dim, hidden) * 0.01
        self.bt1 = np.zeros(hidden)
        self.Wt2 = np.random.randn(hidden, hidden) * 0.01
        self.bt2 = np.zeros(hidden)

        # Residual block weights
        self.W1 = np.random.randn(dim, hidden) * 0.01
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, dim) * 0.01
        self.b2 = np.zeros(dim)

    def embed_t(self, t): 
        half = self.dim // 2
        freqs = np.exp(-np.log(10000) * np.arange(half) / half)
        emb = np.concatenate([np.sin(t * freqs), np.cos(t * freqs)])
        h = np.tanh(emb @ self.Wt1 + self.bt1)
        return h @ self.Wt2 + self.bt2

    def __call__(self, x, t):
        t_emb = self.emb_t(t)
        h = x + t_emb
        h = np.tanh(h @ self.W1 + self.b1)
        h = h @ self.W2 + self.b2
        return x + h     
