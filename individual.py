import numpy as np
import random


class Individual:
    def __init__(self, N, alpha, order=None):
        self.order = np.arange(N)
        # probability of applying mutation
        self.alpha = alpha
        np.random.shuffle(self.order)
        if order is not None:
            self.order = order
