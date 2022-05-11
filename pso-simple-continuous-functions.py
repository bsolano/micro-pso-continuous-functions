# encoding:utf-8

####################################################################################
# A Particle Swarm Optimization algorithm for solving the traveling salesman problem.
# The program reuses part of the code of Marco Castro (https://github.com/marcoscastro/tsp_pso)
#
# Author: Rafael Batres-Pietro
# Author: Braulio J. Solano-Rojas
# Institution: Tecnologico de Monterrey
#
# Date: October 27, 2021.  April-May 2022
####################################################################################

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

# PSO algorithm
class PSO:

    def __init__(self, cost_function, search_space, iterations, population_size, inertia=0.8, particle_confidence=0.1, swarm_confidence=0.1):
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
        solutions = Particle.getRandomSolutions(
            self.nvars, search_space, self.population_size)

        # checks if exists any solution
        if not solutions:
            print('Initial population empty! Try run the algorithm again...')
            sys.exit(1)

        # Select the best random population among 5 populations
        bestSolutions = list(solutions)
        bestCost = self.evaluateSolutionsAverageCost(solutions)
        for _ in range(5):
            solutions = Particle.getRandomSolutions(
                self.nvars, search_space, self.population_size)
            cost = self.evaluateSolutionsAverageCost(solutions)
            if cost < bestCost:
                bestCost = cost
                bestSolutions = list(solutions)
            del solutions[:]

        # creates the particles and initialization of swap sequences in all the particles
        for solution in bestSolutions:
            # creates a new particle
            particle = Particle(solution=solution,
                                cost=self.cost_function(*solution))
            # add the particle
            self.particles.append(particle)

        self.gbest = None

    def evaluateSolutionsAverageCost(self, solutions):

        totalCost = 0.0
        i = 0
        for solution in solutions:
            cost = self.cost_function(*solution)
            totalCost += cost
            i += 1
        averageCost = totalCost / float(i)

        return averageCost

    # set gbest (best particle of the population)

    def setGBest(self, new_gbest):
        self.gbest = new_gbest

    # returns gbest (best particle of the population)
    def getGBest(self):
        return self.gbest

    # returns the elapsed milliseconds since the start of the program
    def elapsedTime(self, start_time):
        ms = (process_time() - start_time) * 1000.0
        return ms

    def setIter(self, last_iter):
        self.last_iter = last_iter

    def getIter(self):
        return self.last_iter

    def run(self):
        # variables for convergence data
        convergenceData = []
        iterationArray = []
        bestCostArray = []
        bestCostSampling = []

        batchSize = 100  # save data every n iterations
        batchCounter = 0

        startTime = process_time()

        # updates gbest (best particle of the population)
        for particle in self.particles:
            if self.gbest is None:
                self.gbest = copy.deepcopy(particle)
            elif self.gbest.getCostPBest() > particle.getCostPBest():
                self.gbest = copy.deepcopy(particle)

        # for each time step (iteration)
        iterations = self.iterations
        t = 0
        while t < iterations:
            # for t in range(self.iterations):
            convergencePerIteration = []
            batchCounter = batchCounter + 1

            # for each particle in the swarm
            for particle in self.particles:
                #previousCost = particle.getCurrentSolutionCost()
                velocity = particle.getVelocity()
                # gets solution of the gbest solution
                gbest = list(self.gbest.getPBest())
                pbest = particle.getPBest()[:]  # copy of the pbest solution
                # gets copy of the current solution of the particle
                currentSolution = particle.getCurrentSolution()[:]

                # updates velocity
                r1 = random.random()
                r2 = random.random()
                velocity = [self.inertia*velocity[i] + self.p_confidence*r1*(
                    pbest[i]-currentSolution[i]) + self.s_confidence*r2*(gbest[i]-currentSolution[i]) for i in range(len(velocity))]
                particle.setVelocity(velocity)

                # new solution
                newSolution = [currentSolution[i] + velocity[i]
                               for i in range(len(currentSolution))]

                # If we collide with the limits, the limit is the solution and velocity is cero
                for i in range(len(velocity)):
                    if newSolution[i] < self.search_space[0]:
                        newSolution[i] = self.search_space[0]
                        velocity[i] = 0.0
                        particle.setVelocity(velocity)

                    if newSolution[i] > self.search_space[1]:
                        newSolution[i] = self.search_space[1]
                        velocity[i] = 0.0
                        particle.setVelocity(velocity)

                # gets cost of the current solution
                newSolutionCost = self.cost_function(*newSolution)
                # if newSolutionCost < previousCost:
                # updates the current solution
                particle.setCurrentSolution(newSolution)
                # updates the cost of the current solution
                particle.setCostCurrentSolution(newSolutionCost)

                # checks if new solution is pbest solution
                pbCost = particle.getCostPBest()
                if newSolutionCost < pbCost:
                    particle.setPBest(newSolution)
                    particle.setCostPBest(newSolutionCost)

                gbestCost = self.gbest.getCostPBest()
                # check if new solution is gbest solution
                if newSolutionCost < gbestCost:
                    self.gbest = copy.deepcopy(particle)

                convergencePerIteration.append(t)
                convergencePerIteration.append(self.gbest.getCostPBest())
                convergenceData.append(convergencePerIteration)
                iterationArray.append(t)
                bestCostArray.append(self.gbest.getCostPBest())

                if batchCounter > batchSize:
                    print(t, "Gbest cost = ", "{:.20f}".format(
                        self.gbest.getCostPBest()))
                    #print("Standard deviation: ", std)
                    batchCounter = 0
                    bestCostSampling.append(self.gbest.getCostPBest())

            t = t + 1
            if t > 220:
                std = np.std(bestCostSampling[-10:])
            else:
                std = 1000

            if t == self.iterations:
                self.setIter(t)

            if isclose(std, 0):
                self.setIter(t)
                break

        df = pd.DataFrame()
        df['Iteration'] = pd.Series(iterationArray)
        df['Best cost'] = pd.Series(bestCostArray)
        plt.xlabel("Iteration No.")
        plt.ylabel("Best cost")
        plt.plot(df['Iteration'], df['Best cost'])

        print("Elapsed time: ", self.elapsedTime(startTime))


# class that represents a particle
class Particle:

    def __init__(self, solution, cost):

        # current solution
        self.solution = solution

        # best solution it has achieved so far by this particle
        self.pbest = solution

        # set costs
        self.newSolutionCost = cost
        self.pbestCost = cost

        # velocity of a particle is a tuple of
        # n cost function variables
        self.velocity = [0 for _ in solution]

    # set pbest
    def setPBest(self, new_pbest):
        self.pbest = new_pbest

    # returns the pbest
    def getPBest(self):
        return self.pbest

    # set the new velocity (sequence of swap operators)
    def setVelocity(self, new_velocity):
        self.velocity = new_velocity

    # returns the velocity (sequence of swap operators)
    def getVelocity(self):
        return self.velocity

    # set solution
    def setCurrentSolution(self, solution):
        self.solution = solution

    # gets solution
    def getCurrentSolution(self):
        return self.solution

    # set cost pbest solution
    def setCostPBest(self, cost):
        self.pbestCost = cost

    # gets cost pbest solution
    def getCostPBest(self):
        return self.pbestCost

    # set cost current solution
    def setCostCurrentSolution(self, cost):
        self.newSolutionCost = cost

    # gets cost current solution
    def getCurrentSolutionCost(self):
        return self.newSolutionCost

    # removes all elements of the list velocity
    def clearVelocity(self):
        del self.velocity[:]

    # gets random unique paths - returns a list of lists of paths
    def getRandomSolutions(size, search_space, max_size):
        random_solutions = []

        for i in range(max_size):

            list_temp = Particle.getRandomSolution(size, search_space)

            if list_temp not in random_solutions:
                random_solutions.append(list_temp)

        return random_solutions

    # Generate a random sequence and stores it
    # as a Route
    def getRandomSolution(size, search_space):
        solution = []
        min, max = search_space
        for _ in range(size):
            solution.append(np.random.uniform(min, max))
        return solution


# Define Chromosome as a subclass of list
class Chromosome(list):
    def __init__(self):
        self.elements = []


if __name__ == "__main__":
    run_experiment = True
    if run_experiment == True:
        function_name = 'griewank20'
        function = globals()[function_name]
        fileoutput = []
        results = ['Beta', 'Alfa', 'Iterations', 'Crossover type', 'Mutation type', 'Mu',
                   'Sigma', 'Gamma'] + ['run'+str(i+1) for i in range(20)] + ['Mean', 'Exact results']
        fileoutput.append(results)
        parameters_space = [[0.32599834,0.58390435,0.38663557],
                            [0.4084812, 0.21552659,0.44582005],
                            [0.61287535,0.80721435,0.86494361],
                            [0.90019089,0.49630546,0.62210536],
                            [0.14872141,0.95555133,0.74617944],
                            [0.54476689,0.8963839, 0.7126064 ],
                            [0.27083494,0.12333146,0.36372988],
                            [0.99062972,0.0228287, 0.5202993 ],
                            [0.87495921,0.69689851,0.97154622],
                            [0.59081831,0.75287642,0.33149702],
                            [0.94704042,0.36961533,0.0187701 ],
                            [0.34066728,0.27372489,0.10044892],
                            [0.69699748,0.4202998, 0.2330681 ],
                            [0.20669621,0.99944323,0.25770852],
                            [0.74229077,0.08474781,0.77256829],
                            [0.64635665,0.18251727,0.81957571],
                            [0.52558646,0.45300364,0.91947666],
                            [0.77888261,0.34014897,0.66176684],
                            [0.12380011,0.25430048,0.03471743],
                            [0.72790392,0.52548816,0.67756385],
                            [0.02695453,0.55795663,0.16995784],
                            [0.05730227,0.65325683,0.09487899],
                            [0.48433257,0.03648114,0.55846327],
                            [0.82065159,0.85632715,0.29842505],
                            [0.18866629,0.90474284,0.96349276],
                            [0.38771973,0.13965407,0.42237655],
                            [0.07707235,0.7029901, 0.57500681],
                            [0.84833275,0.30416365,0.145066  ],
                            [0.43595953,0.77202635,0.88460923],
                            [0.25363902,0.61728326,0.48900878]]
        for parameters in parameters_space:
            mean_cost = 0
            results = parameters
            exact_results = 0
            for i in range(20):
                pso = PSO(function, functions_search_space[function.__name__], iterations=175000, population_size=150, inertia=parameters[0], particle_confidence=parameters[1]*2, swarm_confidence=parameters[2]*2)
                pso.run()  # runs the PSO algorithm
                cost = pso.getGBest().getCostPBest()
                results.append(cost)
                exact_results += (1 if np.allclose(pso.getGBest().getPBest(),
                                  functions_solution[function.__name__], atol=1e-05) else 0)
                mean_cost += cost
            mean_cost /= 20.0
            results.append(mean_cost)
            results.append(exact_results)
            fileoutput.append(results)

        # pso-results.csv
        csvFile = open('results/micro-pso-continuous-griewank20-experiment.csv', 'w', newline='')
        writer = csv.writer(csvFile)
        writer.writerows(fileoutput)
        csvFile.close()
    else:
        # creates a PSO instance
        # alfa is the probabiliy for a movement based on local best
        # beta is the probability for a movement based on the global best
        for function_name in ['beale','biggs_exp2','biggs_exp3','biggs_exp4','biggs_exp5','biggs_exp6','cross_in_tray','drop_in_wave','dejong_f1','dejong_f2','dejong_f3','dejong_f4','dejong_f5','rosenbrock2','rosenbrock3','rosenbrock4','rosenbrock5','rosenbrock6','rosenbrock7','rosenbrock8','rosenbrock9','rosenbrock10','rosenbrock11','rosenbrock12','rosenbrock13','rosenbrock14','rosenbrock15','rosenbrock16','rosenbrock17','rosenbrock18','rosenbrock19','rosenbrock20','rastringin20','griewank20']:
            function = globals()[function_name]
            results = ['Function'] + ['OptimumSolution x'+str(i+1) for i in range(len(signature(function).parameters))] + ['Solution x'+str(i+1) for i in range(len(signature(function).parameters))] + ['Eucl. dist.', 'Exact solution', 'Exact solution (allclose)', 'Cost', 'Exact optimum', 'Comp. time', 'Iterations']
            fileoutput = []
            fileoutput.append(results)
            for i in range(30):
                results = []
                start_time = process_time()
                pso = PSO(function, functions_search_space[function.__name__], iterations=175000, population_size=150, inertia=0.8, particle_confidence=1, swarm_confidence=2)
                pso.run()  # runs the PSO algorithm
                ms = (process_time() - start_time) * 1000.0
                results.append(function.__name__)
                if isinstance(functions_solution[function.__name__][0], list):
                    results += functions_solution[function.__name__][0]
                else:
                    results += functions_solution[function.__name__]
                results += pso.getGBest().getPBest()
                if isinstance(functions_solution[function.__name__][0], list):
                    min = np.inf
                    for solution in functions_solution[function.__name__]:
                        euclidean_distance = euclidean(pso.getGBest().getPBest(), solution)
                        if euclidean_distance < min:
                            min = euclidean_distance
                    euclidean_distance = min
                else:
                    euclidean_distance = euclidean(pso.getGBest().getPBest(), functions_solution[function.__name__])
                results.append(euclidean_distance)
                results.append(1 if np.isclose(euclidean_distance, 0.0, atol=1e-05) else 0)
                if isinstance(functions_solution[function.__name__][0], list):
                    equal = 0
                    for solution in functions_solution[function.__name__]:
                        if np.allclose(pso.getGBest().getPBest(), solution, atol=1e-05):
                            equal = 1
                            break
                    results.append(equal)
                else:
                    results.append(1 if np.allclose(pso.getGBest().getPBest(), functions_solution[function.__name__], atol=1e-05) else 0)
                results.append(pso.getGBest().getCostPBest())
                if isinstance(functions_solution[function.__name__][0], list):
                    equal = 0
                    for solution in functions_solution[function.__name__]:
                        if np.isclose(pso.getGBest().getCostPBest(), function(*solution), atol=1e-05):
                            equal = 1
                            break
                    results.append(equal)
                else:
                    results.append(1 if np.isclose(pso.getGBest().getCostPBest(), function(*functions_solution[function.__name__]), atol=1e-05) else 0)
                iteration = pso.getIter()
                results.append(ms)
                results.append(iteration)
                fileoutput.append(results)
                # shows the global best particle
                print("Cost of gbest: ", "{:.20f}".format(
                    pso.getGBest().getCostPBest()))
                print("gbest: ", pso.getGBest().getPBest())
                print("")

            csvFile = open('results/pso-simple-continuous-function-'+function_name+'.csv', 'w', newline='')
            with csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(fileoutput)
            csvFile.close()
