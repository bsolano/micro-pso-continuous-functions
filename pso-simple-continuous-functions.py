# encoding:utf-8

####################################################################################
# A Particle Swarm Optimization algorithm to find functions optimum.
# The program reuses part of the code of Marco Castro (https://github.com/marcoscastro/tsp_pso)
#
# Author: Rafael Batres
# Author: Braulio J. Solano-Rojas
# Institution: Tecnologico de Monterrey
#
# Date: October 27, 2021.  April-May 2022
####################################################################################

from __future__ import annotations

import random
import sys
import copy
import csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import pandas as pd
from benchmark_functions import *
from inspect import signature
from math import isclose
from time import process_time
import numpy as np

# For repeatability and reproducibility
np.random.seed(0)
random.seed(0)

# Define Chromosome as a subclass of list
class Chromosome(list):
    def __init__(self):
        self.elements = []


# class that represents a particle
class Particle:

    def __init__(self, solution, cost):

        # current solution
        self.__solution = solution

        # best solution it has achieved so far by this particle
        self.__best_particle = solution

        # set costs
        self.__new_solution_cost = cost
        self.best_particle_cost = cost

        # velocity of a particle is a tuple of
        # n cost function variables
        self.__velocity = [0 for _ in solution]

    # returns the pbest
    @property
    def best_particle(self) -> Particle:
        return self.__best_particle

    # set pbest
    @best_particle.setter
    def best_particle(self, new_best_particle):
        self.__best_particle = new_best_particle

    # returns the velocity (sequence of swap operators)
    @property
    def velocity(self) -> list:
        return self.__velocity

    # set the new velocity (sequence of swap operators)
    @velocity.setter
    def velocity(self, new_velocity: list):
        self.__velocity = new_velocity

    # gets solution
    @property
    def solution(self) -> Chromosome:
        return self.__solution

    # set solution
    @solution.setter
    def solution(self, solution: Chromosome):
        self.__solution = solution

    # gets cost pbest solution
    @property
    def best_particle_cost(self) -> float:
        return self.__best_particle_cost

    # set cost pbest solution
    @best_particle_cost.setter
    def best_particle_cost(self, cost: float):
        self.__best_particle_cost = cost

    # gets cost current solution
    @property
    def current_solution_cost(self) -> Chromosome:
        return self.__new_solution_cost

    # set cost current solution
    @current_solution_cost.setter
    def current_solution_cost(self, cost: float):
        self.__new_solution_cost = cost

    # removes all elements of the list velocity
    def clear_velocity(self):
        del self.__velocity[:]

    # gets random unique paths - returns a list of lists of paths
    def random_solutions(size: int, search_space: tuple, max_size: int):
        random_solutions = []

        for _ in range(max_size):

            list_temp = Particle.random_solution(size, search_space)

            if list_temp not in random_solutions:
                random_solutions.append(list_temp)

        return random_solutions

    # Generate a random sequence and stores it
    # as a Route
    def random_solution(size: int, search_space: tuple) -> Chromosome:
        solution = []
        min, max = search_space
        for _ in range(size):
            solution.append(np.random.uniform(min, max))
        return solution


# PSO algorithm
class PSO:

    def __init__(self, cost_function, search_space, iterations: int, population_size: int, inertia: float=0.8, particle_confidence: float=0.1, swarm_confidence: float=0.1):
        self.cost_function = cost_function  # the cost function
        # number of variables in the cost function
        self.nvars = len(signature(cost_function).parameters)
        self.search_space = search_space  # interval of the cost function
        self.iterations = iterations  # max of iterations
        self.population_size = population_size  # size population
        self.particles = []  # list of particles
        self.inertia = inertia
        self.p_confidence = particle_confidence
        self.s_confidence = swarm_confidence

        # initialized with a group of random particles (solutions)
        solutions = Particle.random_solutions(
            self.nvars, search_space, self.population_size)

        # checks if exists any solution
        if not solutions:
            print('Initial population empty! Try run the algorithm again...')
            sys.exit(1)

        # Select the best random population among 5 populations
        best_solutions = list(solutions)
        most_diverse = self.evaluate_solutions_diversity(solutions)

        for _ in range(5):
            solutions = Particle.random_solutions(
                self.nvars, self.search_space, self.population_size)
            sim = self.evaluate_solutions_diversity(solutions)
            if sim > most_diverse:
                most_diverse = sim
                best_solutions = list(solutions)
            del solutions[:]

        # creates the particles and initialization of swap sequences in all the particles
        for solution in best_solutions:
            # creates a new particle
            particle = Particle(solution=solution,
                                cost=self.cost_function(*solution))
            # add the particle
            self.particles.append(particle)

        self.__global_best = None

    def evaluate_solutions_diversity(self, solutions: list[Chromosome]) -> float:
        sim_sum = 0
        count = 0
        for solution1 in solutions:
            for solution2 in solutions:
                if not (solution1 == solution2):
                    count += 1
                    # Euclidean distance.  Best distance?
                    sim = euclidean(solution1, solution2)
                    sim_sum += sim
        return sim_sum / count

    # returns gbest (best particle of the population)
    @property
    def global_best(self) -> Particle:
        return self.__global_best

    # set gbest (best particle of the population)
    @global_best.setter
    def global_best(self, new_global_best: Particle):
        self.__global_best = new_global_best

    # returns the elapsed milliseconds since the start of the program
    def elapsed_time(self, start_time):
        ms = (process_time() - start_time) * 1000.0
        return ms

    @property
    def iteration(self) -> int:
        return self.__last_iter

    @iteration.setter
    def iteration(self, last_iter: int):
        self.__last_iter = last_iter

    def run(self):
        # variables for convergence data
        convergence_data = []
        iteration_array = []
        best_cost_array = []
        sampled_best_cost_array = []
        best_cost_sampling = []

        batch_size = 100  # save data every n iterations
        batch_counter = 0

        start_time = process_time()

        # updates gbest (best particle of the population)
        for particle in self.particles:
            if self.global_best is None:
                self.global_best = copy.deepcopy(particle)
            elif self.global_best.best_particle_cost > particle.best_particle_cost:
                self.global_best = copy.deepcopy(particle)

        # for each time step (iteration)
        iterations = self.iterations
        t = 0
        while t < iterations:
            # for t in range(self.iterations):
            convergence_per_iteration = []
            batch_counter = batch_counter + 1

            # for each particle in the swarm
            for particle in self.particles:
                #previousCost = particle.getCurrentSolutionCost()
                velocity = particle.velocity
                # gets solution of the gbest solution
                gbest = list(self.global_best.best_particle)
                pbest = particle.best_particle[:]  # copy of the pbest solution
                # gets copy of the current solution of the particle
                current_solution = particle.solution[:]

                # updates velocity
                r1 = random.random()
                r2 = random.random()
                velocity = [self.inertia*velocity[i] + self.p_confidence*r1*(
                    pbest[i]-current_solution[i]) + self.s_confidence*r2*(gbest[i]-current_solution[i]) for i in range(len(velocity))]
                particle.velocity = velocity

                # new solution
                new_solution = [current_solution[i] + velocity[i]
                               for i in range(len(current_solution))]

                # If we collide with the limits, the limit is the solution and velocity is cero
                for i in range(len(velocity)):
                    if new_solution[i] < self.search_space[0]:
                        new_solution[i] = self.search_space[0]
                        velocity[i] = 0.0
                        particle.velocity = velocity

                    if new_solution[i] > self.search_space[1]:
                        new_solution[i] = self.search_space[1]
                        velocity[i] = 0.0
                        particle.velocity = velocity

                # gets cost of the current solution
                new_solution_cost = self.cost_function(*new_solution)
                # if newSolutionCost < previousCost:
                # updates the current solution
                particle.solution = new_solution
                # updates the cost of the current solution
                particle.current_solution_cost = new_solution_cost

                # checks if new solution is pbest solution
                pb_cost = particle.best_particle_cost
                if new_solution_cost < pb_cost:
                    particle.best_particle = new_solution
                    particle.best_particle_cost = new_solution_cost

                gbest_cost = self.global_best.best_particle_cost
                # check if new solution is gbest solution
                if new_solution_cost < gbest_cost:
                    self.global_best = copy.deepcopy(particle)

                if batch_counter > batch_size:
                    print(t, "Gbest cost = ", "{:.20f}".format(
                        self.global_best.best_particle_cost))
                    #print("Standard deviation: ", std)
                    batch_counter = 0
                    best_cost_sampling.append(self.global_best.best_particle_cost)

            convergence_per_iteration.append(t)
            convergence_per_iteration.append(self.global_best.best_particle_cost)
            convergence_data.append(convergence_per_iteration)
            iteration_array.append(t)
            best_cost_array.append(self.global_best.best_particle_cost)
            if (t % 349) == 0:
                sampled_best_cost_array.append(self.global_best.best_particle_cost)

            t = t + 1
            if t > 220:
                std = np.std(best_cost_sampling[-10:])
            else:
                std = 1000

            if t == self.iterations:
                self.iteration = t

            if isclose(std, 0):
                self.iteration = t
                break

        self.sampled_best_cost_array = sampled_best_cost_array
        df = pd.DataFrame()
        df['Iteration'] = pd.Series(iteration_array)
        df['Best cost'] = pd.Series(best_cost_array)
        plt.xlabel("Iteration No.")
        plt.ylabel("Best cost")
        plt.plot(df['Iteration'], df['Best cost'])

        print("Elapsed time: ", self.elapsed_time(start_time))


if __name__ == "__main__":
    # creates a PSO instance
    # alfa is the probabiliy for a movement based on local best
    # beta is the probability for a movement based on the global best
    for function_name in ['beale','biggs_exp2','biggs_exp3','biggs_exp4','biggs_exp5','biggs_exp6','cross_in_tray','drop_in_wave','dejong_f1','dejong_f2','dejong_f3','dejong_f4','dejong_f5','rosenbrock2','rosenbrock3','rosenbrock4','rosenbrock5','rosenbrock6','rosenbrock7','rosenbrock8','rosenbrock9','rosenbrock10','rosenbrock11','rosenbrock12','rosenbrock13','rosenbrock14','rosenbrock15','rosenbrock16','rosenbrock17','rosenbrock18','rosenbrock19','rosenbrock20','rastringin20','griewank20']:
        function = globals()[function_name]
        results = ['Function'] + ['OptimumSolution x'+str(i+1) for i in range(len(signature(function).parameters))] + ['Solution x'+str(i+1) for i in range(len(signature(function).parameters))] + ['Eucl. dist.', 'Exact solution', 'Exact solution (allclose)', 'Cost', 'Exact optimum', 'Comp. time', 'Iterations']
        convergence_data = [['Iteration']]
        fileoutput = []
        fileoutput.append(results)
        for i in range(30):
            convergence_data.append(['Run '+str(i+1)])
            results = []
            start_time = process_time()
            pso = PSO(function, functions_search_space[function.__name__], iterations=175000, population_size=150, inertia=0.8, particle_confidence=2.05, swarm_confidence=2.05)
            pso.run()  # runs the PSO algorithm
            convergence_data[-1].extend(pso.sampled_best_cost_array)
            ms = (process_time() - start_time) * 1000.0
            results.append(function.__name__)
            if isinstance(functions_solution[function.__name__][0], list):
                results += functions_solution[function.__name__][0]
            else:
                results += functions_solution[function.__name__]
            results += pso.global_best.best_particle
            if isinstance(functions_solution[function.__name__][0], list):
                min = np.inf
                for solution in functions_solution[function.__name__]:
                    euclidean_distance = euclidean(pso.global_best.best_particle, solution)
                    if euclidean_distance < min:
                        min = euclidean_distance
                euclidean_distance = min
            else:
                euclidean_distance = euclidean(pso.global_best.best_particle, functions_solution[function.__name__])
            results.append(euclidean_distance)
            results.append(1 if np.isclose(euclidean_distance, 0.0, atol=1e-05) else 0)
            if isinstance(functions_solution[function.__name__][0], list):
                equal = 0
                for solution in functions_solution[function.__name__]:
                    if np.allclose(pso.global_best.best_particle, solution, atol=1e-05):
                        equal = 1
                        break
                results.append(equal)
            else:
                results.append(1 if np.allclose(pso.global_best.best_particle, functions_solution[function.__name__], atol=1e-05) else 0)
            results.append(pso.global_best.best_particle_cost)
            if isinstance(functions_solution[function.__name__][0], list):
                equal = 0
                for solution in functions_solution[function.__name__]:
                    if np.isclose(pso.global_best.best_particle_cost, function(*solution), atol=1e-05):
                        equal = 1
                        break
                results.append(equal)
            else:
                results.append(1 if np.isclose(pso.global_best.best_particle_cost, function(*functions_solution[function.__name__]), atol=1e-05) else 0)
            iteration = pso.iteration
            results.append(ms)
            results.append(iteration)
            fileoutput.append(results)
            # shows the global best particle
            print("Cost of gbest: ", "{:.20f}".format(
                pso.global_best.best_particle_cost))
            print("gbest: ", pso.global_best.best_particle)
            print("")

        # convergence.csv
        csvFile = open('results/pso-simple-continuous-function-'+function_name+'-convergence.csv', 'w', newline='')
        writer = csv.writer(csvFile)
        max_iterations = max([len(v) for v in convergence_data])
        convergence_data[0].extend([i*349 for i in range(max_iterations)])
        writer.writerows(convergence_data)
        csvFile.close()

        csvFile = open('results/pso-simple-continuous-function-'+function_name+'.csv', 'w', newline='')
        with csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(fileoutput)
        csvFile.close()
