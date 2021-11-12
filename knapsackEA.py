import copy
import random

import numpy as np

from individual import Individual
from knapsack import Knapsack
from typing import *
import statistics as stats


class KnapsackEA:
    def __init__(self, ks: Knapsack):
        self.lamda = 200
        self.mu = self.lamda * 2
        self.ks = ks
        self.k = 5
        self.iteration = 100
        # self.init_alpha = 0.05  # probability of applying mutation
        self.init_alpha = max(0.01, 0.1 + 0.02 * np.random.randn())
        self.population = self.initialize()
        self.objf = ks.fitness

    def initialize(self) -> List[Individual]:
        return [Individual(len(self.ks.values), self.init_alpha) for _ in range(self.lamda)]

    def optimize(self):

        population_fitness = [self.ks.fitness(x) for x in self.population]

        print("Iteration=", -1,
              "Mean fitness= ", stats.mean(population_fitness),
              # "variance fitness= ", stats.variance(population_fitness),
              "Best fitness= ", max(population_fitness),
              # "Best in KS= ", self.ks.in_knapsack(self.population[0]),
              # "Best KS= ", self.population[0].order,
              )

        for i in range(self.iteration):
            offsprings: List[Individual] = list()

            for j in range(self.mu):
                p1 = self.selection()
                p2 = self.selection()
                c = self.recombination(p1, p2)
                offsprings.append(c)

            # Apply mutation to offsprings
            map(self.mutation, offsprings)

            map(self.lso, self.population)
            # map(self.lso_swap, self.population)

            # for i in range(len(self.population)):
            #     self.population[i] = self.lso(self.population[i])

            # Apply mutation to population
            map(self.mutation, self.population)

            # Elimination
            self.population = self.elimination(offsprings)

            population_fitness = [self.ks.fitness(x) for x in self.population]

            print("Iteration=", i,
                  "Mean fitness= ", stats.mean(population_fitness),
                  # "variance fitness= ", stats.variance(population_fitness),
                  "Best fitness= ", max(population_fitness),
                  # "Best in KS= ", self.ks.in_knapsack(self.population[0]),
                  # "Best KS= ", self.population[0].order,
                  )

    def mutation(self, ind: Individual) -> Individual:
        if random.random() < ind.alpha:
            for x in range(int(len(ind.order) * 0.1)):
                idx1 = random.randint(0, len(ind.order) - 1)
                idx2 = random.randint(0, len(ind.order) - 1)
                ind.order[idx1], ind.order[idx2] = ind.order[idx2], ind.order[idx1]
        return ind

    def lso(self, ind: Individual) -> Individual:
        f_ind = self.ks.fitness(ind)
        ind1 = copy.deepcopy(ind)

        best_fitness = f_ind
        best_ind = copy.deepcopy(ind)

        for i in range(1, len(ind1.order)):
            ind1.order[0] = ind.order[i]
            ind1.order[1:i + 1] = ind.order[0:i]
            ind1.order[i + 1:] = ind.order[i + 1:]

            fv = self.ks.fitness(ind1)
            if fv > best_fitness:
                best_fitness = fv
                best_ind.order = ind1.order

        # print("lso:", str(f_ind), "->", str(best_fitness))
        return best_ind

    def lso_swap(self, ind: Individual) -> Individual:
        f_ind = self.ks.fitness(ind)
        ind1 = copy.deepcopy(ind)

        best_fitness = f_ind
        best_ind = copy.deepcopy(ind)

        for i in range(0, len(ind1.order)):
            for j in range(1, len(ind1.order) - 1):
                ind1.order[j] = ind.order[i]
                ind1.order[i] = ind.order[j]

                fv = self.ks.fitness(ind1)
                if fv > best_fitness:
                    best_fitness = fv
                    best_ind.order = ind1.order

                ind1.order[j] = ind.order[i]
                ind1.order[i] = ind.order[j]

        # print("lso:", str(f_ind), "->", str(best_fitness))
        return best_ind

    def recombination(self, ind1: Individual, ind2: Individual) -> Individual:
        in_ks1 = self.ks.in_knapsack(ind1)
        in_ks2 = self.ks.in_knapsack(ind2)
        # Copy intersection to offspring
        offspring = np.intersect1d(in_ks1, in_ks2)

        # Copy symmetric difference to offspring at 50%
        sym_diff = np.concatenate((np.setdiff1d(in_ks1, in_ks2), np.setdiff1d(in_ks2, in_ks1)))
        for ind in sym_diff:
            if random.random() < 0.5:
                offspring = np.append(offspring, ind)

        # Shuffle intersected/symdiff offspring
        np.random.shuffle(offspring)

        all_elem = np.arange(len(self.ks.values))
        np.random.shuffle(all_elem)
        rem = np.setdiff1d(all_elem, offspring)
        np.random.shuffle(rem)

        offspring = np.concatenate((offspring, rem))

        beta = 2 * np.random.random() - 0.5
        alpha = ind1.alpha + beta * (ind2.alpha - ind1.alpha)
        return Individual(len(self.ks.values), alpha, offspring)

    def selection(self) -> Individual:
        ri = random.choices(range(np.size(self.population, 0)), k=self.k)
        rif = np.array([self.ks.fitness(self.population[e]) for e in ri])
        idx = np.argmax(rif)
        return self.population[ri[idx]]

    def elimination(self, offsprings: List[Individual]) -> List[Individual]:
        combined = np.concatenate((self.population, offsprings))
        l = list(combined)
        l.sort(key=lambda x: self.ks.fitness(x), reverse=True)
        return l[:self.lamda]


if __name__ == '__main__':
    N = 200
    ks = Knapsack(N)
    ksEA = KnapsackEA(ks)

    # print("Weight=", ks.weights)
    # print("Values=", ks.values)
    # print("Capacity=", ks.capacity)

    heuristic_order = np.arange(len(ks.values))
    heuristic_order_list = list(heuristic_order)
    heuristic_order_list.sort(key=lambda x: ks.values[x] / ks.weights[x], reverse=True)
    heurBest = Individual(len(heuristic_order_list), 0.0, np.array(heuristic_order_list))
    print("Heuristic objective value=", ks.fitness(heurBest))

    # for can in ksEA.population:
    #     print("Order=", can.order)
    #     print("In KS=", ks.in_knapsack(can))
    #     print("Objective value=", ks.fitness(can))

    # for i in range(1, 20 + 1):
    #     ksEA.mutation(ksEA.population[0])
    #     print(ksEA.population[0].order)

    # offspring = ksEA.recombination(ksEA.population[0], ksEA.population[1])
    # print(ksEA.population[0].order)
    # print(ksEA.population[1].order)
    # print(offspring.order)
    # print(offspring.alpha)

    ksEA.optimize()
