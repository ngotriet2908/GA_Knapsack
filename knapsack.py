import numpy as np
import random
from individual import Individual


class Knapsack:

    def __init__(self, numObjects, capacity=None, values=None, weights=None):

        if capacity is None:
            self.values = 2 ** np.random.randn(numObjects)
            self.weights = 2 ** np.random.randn(numObjects)
            self.capacity = 0.2 * np.sum(self.weights)
        else:
            self.values = np.array(values)
            self.weights = np.array(weights)
            self.capacity = np.array(capacity)

    def fitness(self, ind: Individual) -> float:
        value = 0.0
        remain_cap = self.capacity

        for i in ind.order:
            if self.weights[i] <= remain_cap:
                value += self.values[i]
                remain_cap -= self.weights[i]
        return value

    def in_knapsack(self, ind: Individual):
        ksc = []
        remain_cap = self.capacity
        for i in ind.order:
            if self.weights[i] <= remain_cap:
                ksc.append(i)
                remain_cap -= self.weights[i]
        return np.array(ksc)
