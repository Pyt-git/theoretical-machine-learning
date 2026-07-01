class WeightDecay: 
   def __init__(self, wd): 
       self.wd = wd

   def apply(self, W, opt_state): 
       return W - opt_state.lr * self.wd * W
