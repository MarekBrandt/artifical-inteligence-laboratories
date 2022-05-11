import random

import numpy as np
from utils import fitness


class Genetic:
    def __init__(self, coords, population_size=100, elite_size=10, mutation_rate=0.01):
        self.coords = coords
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

    def population_fitness(self, population):
        population_fitness = {}
        for i, individual in enumerate(population):
            # 1/fitness -> change to maximization problem
            population_fitness[i] = 1/fitness(self.coords, individual)

        return {k: v for k, v in sorted(population_fitness.items(), key=lambda item: item[1], reverse=True)}

    def best_solution(self, population):
        population_fitness = list(self.population_fitness(population))
        best_ind = population_fitness[0]
        return population[best_ind]

    def initial_population(self):
        population = []
        # Create initial population
        for i in range(self.population_size):
            solution = np.random.permutation(len(self.coords))
            population.append(solution)
        return population

    def selection(self, population):
        selection = []
        population_fitness = self.population_fitness(population)
        probability = {}
        sum_fitness = sum(population_fitness.values())

        probability_prev = 0.0
        for key, value in population_fitness.items():
            probability[key] = probability_prev + (value/sum_fitness)
            probability_prev = probability[key]

        for i in range(len(population)):
            rand = random.random()
            for key, value in probability.items():
                if rand <= value or i < self.elite_size:
                    selection.append(population[key])
                    break

        return selection

    def crossover_population(self, population):
        crossover_population = population[:self.elite_size]
        for i in range(self.elite_size, len(population)):
        #for i in range(len(population)):
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            first = random.randrange(len(parent1))
            last = random.randrange(len(parent1))
            fragment = parent1[first:last]
            crossover = []
            for vertex in parent2:
                if vertex not in fragment:
                    crossover.append(vertex)

            for vertex in reversed(fragment):
                crossover.insert(first, vertex)

            crossover_population.append(crossover)
        return crossover_population

    def mutate_population(self, population):
        i = 0
        for individual in population:
            if i >= self.elite_size:
                for i in range(len(individual)):
                    rand = random.random()
                    if rand <= self.mutation_rate:
                        swap = random.randint(0, len(individual)-1)
                        individual[swap], individual[i] = individual[i], individual[swap]
            i += 1

        return population

    def next_generation(self, population):
        selection = self.selection(population)
        children = self.crossover_population(selection)
        next_generation = self.mutate_population(children)
        return next_generation
