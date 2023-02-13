'''
E19
Jack Miller
Daniel Gibson
'''

import random
import sys
import matplotlib.pyplot as plt
import time


# Generate population of chromosomes, each chromosome being string of 1s and 0s
def gen_pop(num_individuals, len_chromo):
    population = []
    for i in range(num_individuals):
        individual = ""
        for j in range(len_chromo):
            bit = str(random.randint(1, num_individuals) % 2)
            individual += bit
        population.append(individual)
    return population

# The max of f(x,y) = x - y is where x is largest and y is smallest
# So, fitness is directly proportional (and equal) to x - y
def fitness(chromo, gene_boundaries):
    chromo_X = chromo[0:gene_boundaries[0]]
    total_X = int(chromo_X, 2)
    chromo_Y = chromo[gene_boundaries[0]:]
    total_Y = int(chromo_Y, 2)
    return total_X-total_Y

# Randomly pick n chromosomes from population, return the one with the best fitness
# Use this to choose parents for the next generation
def nary_tournament(pop, fitness_list, n=2):
    max_fit = sys.maxsize * -1      # initialize to lowest possible value
    most_fit = ""
    for i in range(n):
        idx = random.randint(0, len(pop) - 1)
        if fitness_list[idx] > max_fit:
            max_fit = fitness_list[idx]
            most_fit = pop[idx]
    return most_fit

# Perform crossover between parent chromosomes
def crossover(parent_1, parent_2, cross_rate):
    off_1 = ""
    off_2 = ""
    if random.random() <= cross_rate:
        cross_point = random.randint(0, len(parent_1) - 1)
        off_1 = parent_1[0:cross_point] + parent_2[cross_point:]
        off_2 = parent_2[0:cross_point] + parent_1[cross_point:]
    else:
        off_1 = parent_1
        off_2 = parent_2
    return off_1, off_2

# Mutate a child chromosome; only mutate some percentage of chromosomes (the mut_rate)
def mutation(off, mut_rate):
    if random.random() <= mut_rate:
        idx = random.randint(0, len(off) - 1)
        new_bit = "0" if off[idx] == "1" else "1"
        off = off[0:idx] + new_bit + off[idx+1:]
    return off


def main(pop_size, chromo_size, gene_boundaries, mut_rate, cross_rate, max_gens, converge_pct):
    # generate population with random chromosome values
    population = gen_pop(pop_size, chromo_size)

    convergence_pcts = []
    av_fitness_list = []
    generation_list = []

    count = 0
    #repeat maxGens times
    while count <= max_gens:
        # calculate fitnesses
        fitness_list = [fitness(ind, gene_boundaries) for ind in population]

        # print some info
        max_fit_val = max(fitness_list)
        idx = fitness_list.index(max_fit_val)
        max_fit_ind = population[idx]

        av_fitness = sum(fitness_list)/pop_size
        convergence = population.count(max_fit_ind)/len(population)
        av_fitness_list.append(av_fitness)
        convergence_pcts.append(convergence)
        generation_list.append(count)

        print("Most fit individual for gen: %d" % count)
        print("Chromosome: %s " % max_fit_ind)
        print("Fitness: %s" % max_fit_val)
        print("Convergence pct: %f" % convergence)
        print()

        # Interactive plots of convergence pct and average fitness for each generation
        plt.figure(0)
        plt.ion()
        plt.clf()
        plt.plot(generation_list, convergence_pcts, "ro", color='purple')
        plt.xlabel('Generation')
        plt.ylabel('Convergence Pct')
        plt.draw()
        plt.pause(0.1)

        plt.figure(1)
        plt.ion()
        plt.clf()
        plt.plot(generation_list, av_fitness_list, "ro", color='green')
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.draw()
        plt.pause(0.1)

        # Quit if convergence reaches the desired percent
        if convergence >= converge_pct:
            break

        new_generation = []
        for rep in range(pop_size//2):
            # tournament selection for reproduction
            parent_1 = nary_tournament(population, fitness_list, 5)
            parent_2 = nary_tournament(population, fitness_list, 5)

            # apply crossover and mutation
            offspring_1, offspring_2 = crossover(parent_1, parent_2, cross_rate)
            offspring_1 = mutation(offspring_1, mut_rate)
            offspring_2 = mutation(offspring_2, mut_rate)

            # add new offspring to new generation
            new_generation.append(offspring_1)
            new_generation.append(offspring_2)

        population = new_generation
        count += 1



main(pop_size=500, chromo_size=32, gene_boundaries=[16], mut_rate=.001, cross_rate=.6, max_gens=100, converge_pct=1.0)
