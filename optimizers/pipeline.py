class OptimizerPipeline:
    def __init__(self, adamw, weight_decay=None, lr_schedule=None, clip_value=None):
        self.adamw = adamw
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.clip_value = clip_value

    def update(self, W, b, w_grad, b_grad, opt_state):
        # 1. Weight decay
        if self.weight_decay is not None:
            W = self.weight_decay.apply(W, opt_state)

        # 2. Gradient clipping
        if self.clip_value is not None:
            w_grad, b_grad = clip_gradients(w_grad, b_grad, self.clip_value)

        # 3. LR schedule
        lr = opt_state.lr
        if self.lr_schedule is not None:
            lr = self.lr_schedule.compute_lr(opt_state)

        # 4. AdamW update
        W, b, opt_state = self.adamw.update(W, b, w_grad, b_grad, opt_state, lr)

        return W, b, opt_state
