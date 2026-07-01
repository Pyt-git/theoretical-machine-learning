import numpy as np 

class AdamW: 
    def __init__(self, beta1, beta2, eps): 
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def update(self, W, b, w_grad, b_grad, opt_state, lr): 
        opt_state.m_W = self.beta1 * opt_state.m_W + (1 - self.beta1) * w_grad
        opt_state.m_b = self.beta1 * opt_state.m_W + (1 - self.beta1) * b_grad

        opt_state.v_W = self.beta2 * opt_state.v_W + (1 - self.beta2) * (w_grad ** 2)
        opt_state.v_b = self.beta2 * opt_state.v_b + (1 - self.beta2) * (b_grad ** 2)

        m_W_hat = opt_state.m_W / (1 - self.beta1 ** opt_state.t)
        m_b_hat = opt_state.m_b / (1 - self.beta1 ** opt_state.t)
        v_W_hat = opt_state.v_W / (1 - self.beta1 ** opt_state.t)
        v_b_hat = opt_state.v_b / (1 - self.beta1 ** opt_state.t)

        W -= lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
        b -= lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
      
        opt_state.t += 1
        opt_state.lr = lr

        return W, b, opt_state
