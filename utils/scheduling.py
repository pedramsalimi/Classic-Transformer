# utils/scheduling.py
import math

class NoamScheduler:
    def __init__(self, optimizer, warmup_steps=4000, model_size=512):
        self.optimizer = optimizer
        self.step_num = 0
        self.warmup_steps = warmup_steps
        self.model_size = model_size

    def step(self):
        self.step_num += 1
        lr = self.model_size**(-0.5)*min(self.step_num**(-0.5), self.step_num*(self.warmup_steps**(-1.5)))
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr
