# -*- coding: utf-8 -*-

####################################################################################
# A Particle Swarm Optimization algorithm to find functions optimum.
#
#
# Author: Rafael Batres-Pietro
# Author: Braulio J. Solano-Rojas
# Institution: Tecnológico de Monterrey
# Date: June 6, 2018. April-May 2022
####################################################################################

import random
import sys
import copy
import csv
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from benchmark_functions import *
from inspect import signature
from math import isclose
from time import process_time
import numpy as np

# For repeatability and reproducibility
np.random.seed(0)
random.seed(0)

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

        # velocity of a particle is a sequence of 3-tuple
        # (1, 2, 0.8) means SO(1,2), beta = 0.8
        self.velocity = []

        # past positions
        self.history = []

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
        chromosome = Chromosome()
        min, max = search_space
        for _ in range(size):
            chromosome.append(np.random.uniform(min, max))
        return chromosome


# PSO algorithm
class Solver:

    def __init__(self, cost_function, search_space, iterations, max_epochs, population_size, beta=1, alfa=1, first_population_criteria='average_cost', crossover_type='average_crossover', mutation_type='mutateGoodSolution', mu=0.1, sigma=0.1, gamma=0.1):
        self.cost_function = cost_function  # the cost function
        # number of variables in the cost function
        self.nvars = len(signature(cost_function).parameters)
        self.search_space = search_space  # interval of the cost function
        self.iterations = iterations  # max of iterations
        self.max_epochs = max_epochs
        self.population_size = population_size  # size population
        self.particles = []  # list of particles
        # the probability that all swap operators in swap sequence (gbest - x(t-1))
        self.beta = beta
        # the probability that all swap operators in swap sequence (pbest - x(t-1))
        self.alfa = alfa
        self.last_epoch = 0
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma

        # initialized with a group of random particles (solutions)
        solutions = Particle.getRandomSolutions(
            self.nvars, search_space, self.population_size)
        print("One initial solution: ", solutions[0])

        # checks if exists any solution
        if not solutions:
            print('Initial population empty! Try run the algorithm again...')
            sys.exit(1)

        # Select the best random population among 5 populations
        bestSolutions = list(solutions)

        if first_population_criteria == 'average_cost':
            bestCost = self.evaluateSolutionsAverageCost(solutions)

            for _ in range(5):
                solutions = Particle.getRandomSolutions(
                    self.nvars, self.search_space, self.population_size)
                cost = self.evaluateSolutionsAverageCost(solutions)
                if cost < bestCost:
                    bestCost = cost
                    bestSolutions = list(solutions)
                del solutions[:]

        elif first_population_criteria == 'diversity':
            mostDiverse = self.evaluateSolutionsDiversity(solutions)

            for _ in range(5):
                solutions = Particle.getRandomSolutions(
                    self.nvars, self.search_space, self.population_size)
                sim = self.evaluateSolutionsDiversity(solutions)
                print("Diversity of the population: ", sim)
                #cost = self.evaluateSolutionsAverageCost(solutions)
                if sim > mostDiverse:
                    mostDiverse = sim
                    bestSolutions = list(solutions)
                del solutions[:]

        self.gbest = None
        # initialization of all particles
        for solution in bestSolutions:
            # creates a new particle
            particle = Particle(solution=solution,
                                cost=self.cost_function(*solution))
            # add the particle
            self.particles.append(particle)
            # updates gbest if needed
            if self.gbest is None:
                self.gbest = copy.deepcopy(particle)
            elif self.gbest.getCostPBest() > particle.getCostPBest():
                self.gbest = copy.deepcopy(particle)

    def initPopulation(self, population_size):
        self.particles = []  # list of particles
        solutions = Particle.getRandomSolutions(
            self.nvars, self.search_space, population_size)
        self.population_size = population_size

        # checks if exists any solution
        if not solutions:
            print('Initial population empty! Try run the algorithm again...')
            sys.exit(1)

        # Select the best random population among 5 populations
        bestSolutions = list(solutions)
        bestCost = self.evaluateSolutionsAverageCost(solutions)

        for _ in range(5):
            solutions = Particle.getRandomSolutions(
                self.nvars, self.search_space, self.population_size)
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

    def evaluateSolutionsDiversity(self, solutions):
        simSum = 0
        count = 0
        for solution1 in solutions:
            for solution2 in solutions:
                if not (solution1 == solution2):
                    count += 1
                    # Euclidean distance.  Best distance?
                    sim = euclidean(solution1, solution2)
                    simSum += sim
        return simSum / count

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

    def setEpoch(self, last_epoch):
        self.last_epoch = last_epoch

    def getEpoch(self):
        return self.last_epoch

    # gets solution
    def getCurrentSolutions(self):
        return [particle.getCurrentSolution() for particle in self.particles]

    def run(self):
        # variables for convergence data
        convergenceData = []
        iterationArray = []
        bestCostArray = []
        epochArray = []
        epochBestCostArray = []
        bestCostSampling = []

        batchSize = 100  # save data every n iterations
        batchCounter = 0

        HISTORY_SIZE = 100

        epoch = 0
        while epoch < self.max_epochs:
            print("Epoch: ", epoch, "with ", self.population_size, " particles")
            print('Iterations', self.iterations)
            print("Alfa = ", self.alfa, "Beta = ", self.beta)
            convergencePerEpoch = []

            if epoch > 0:
                self.initPopulation(self.population_size)
                print("Particles: ", len(self.particles))
                mutated_elite = getattr(self, self.mutation_type)(
                    self.gbest.getPBest(), *self.search_space)
                self.particles[random.randint(
                    0, self.population_size-1)] = Particle(mutated_elite, self.gbest.getCostPBest())
                print("Inserted elite solution!")

            # for each time step (iteration)
            for t in range(self.iterations):
                convergencePerIteration = []
                batchCounter = batchCounter + 1

                averageCost = np.mean([particle.pbestCost for particle in self.particles])
                costStd = np.std([particle.pbestCost for particle in self.particles])

                # for each particle in the swarm
                for particle in self.particles:
                    previousCost = particle.getCurrentSolutionCost()

                    # gets solution of the gbest solution
                    gbest = list(self.gbest.getPBest())

                    if len(particle.history) == HISTORY_SIZE:
                        particle.history.pop(0)

                    if self.mutation_type == 'mutateGoodSolution':
                        bestNeighbor = getattr(self, self.mutation_type)(
                            particle.getCurrentSolution(), *self.search_space)
                    elif self.mutation_type == 'mutateGoodSolutionMuSigma':
                        bestNeighbor = getattr(self, self.mutation_type)(
                            particle.getCurrentSolution(), self.mu, self.sigma)

                    for i in range(len(bestNeighbor)):
                        if bestNeighbor[i] < self.search_space[0]:
                            bestNeighbor[i] = self.search_space[0]
                        if bestNeighbor[i] > self.search_space[1]:
                            bestNeighbor[i] = self.search_space[1]

                    bestNeighborCost = self.cost_function(*bestNeighbor)

                    newSolution = particle.getCurrentSolution()[:]

                    if random.random() <= self.beta:
                        if self.crossover_type == 'average_crossover':
                            newSolution = getattr(self, self.crossover_type)(
                                list(newSolution), self.gbest.getPBest())
                        elif self.crossover_type == 'crossover':
                            newSolution = getattr(self, self.crossover_type)(
                                list(newSolution), self.gbest.getPBest(), gamma=self.gamma)
                    elif random.random() <= self.alfa:
                        largest_dist = 0
                        for neighbor_particle in self.particles:
                            sol = neighbor_particle.getPBest()
                            dist = euclidean(gbest, sol)

                            if dist > largest_dist:
                                largest_dist = dist
                                dissimilar_particle = neighbor_particle
                        if self.crossover_type == 'average_crossover':
                            newSolution = getattr(self, self.crossover_type)(
                                list(newSolution), dissimilar_particle.getPBest())
                        elif self.crossover_type == 'crossover':
                            newSolution = getattr(self, self.crossover_type)(
                                list(newSolution), dissimilar_particle.getPBest(), gamma=self.gamma)

                    for i in range(len(newSolution)):
                        if newSolution[i] < self.search_space[0]:
                            newSolution[i] = self.search_space[0]
                        if newSolution[i] > self.search_space[1]:
                            newSolution[i] = self.search_space[1]

                        # gets cost of the current solution
                    newSolutionCost = self.cost_function(*newSolution)

                    if newSolutionCost < bestNeighborCost:
                        bestNeighbor = newSolution[:]
                        bestNeighborCost = newSolutionCost

                    if bestNeighborCost < previousCost and bestNeighbor not in particle.history:
                        # updates the current solution
                        particle.setCurrentSolution(bestNeighbor)
                        # updates the cost of the current solution
                        particle.setCostCurrentSolution(bestNeighborCost)
                        particle.history.append(bestNeighbor)

                    # checks if new solution is pbest solution
                    pbCost = particle.getCostPBest()

                    if bestNeighborCost < pbCost:
                        particle.setPBest(bestNeighbor)
                        particle.setCostPBest(bestNeighborCost)

                    gbestCost = self.gbest.getCostPBest()

                    # check if new solution is gbest solution
                    if particle.getCurrentSolutionCost() < gbestCost:
                        self.gbest = copy.deepcopy(particle)

                if batchCounter > batchSize:
                    #print("Sum of acceptance probabilities:", sumAcceptanceProbabilities)
                    print(t, "Gbest cost = ", self.gbest.getCostPBest())
                    convergencePerIteration.append(t)
                    convergencePerIteration.append(self.gbest.getCostPBest())
                    convergencePerIteration.append(averageCost)
                    convergencePerIteration.append(costStd)
                    convergenceData.append(convergencePerIteration)
                    iterationArray.append(t)
                    bestCostArray.append(self.gbest.getCostPBest())
                    batchCounter = 0

                if self.max_epochs > 1:
                    convergencePerEpoch.append(epoch)
                    convergencePerEpoch.append(self.gbest.getCostPBest())
                    convergenceData.append(convergencePerEpoch)
                    epochArray.append(epoch)
                    epochBestCostArray.append(self.gbest.getCostPBest())

            epoch = epoch + 1
            self.setEpoch(epoch)
            bestCostSampling.append(self.gbest.getCostPBest())
            if epoch > 5:
                std = np.std(bestCostSampling[-10:])
                print("standard deviation: ", std)
            else:
                std = 1000

            if isclose(std, 0):
                break

        print("What's going on?")
        print("Cost of gbest: ", self.gbest.getCostPBest())
        print("gbest: ", self.gbest.getPBest())
        print("")
        df = pd.DataFrame()
        if self.max_epochs == 1:
            df['Iteration'] = pd.Series(iterationArray)
            df['Best cost'] = pd.Series(bestCostArray)
            plt.xlabel("Iteration No.")
            plt.ylabel("Best cost")
            plt.plot(df['Iteration'], df['Best cost'])
            plt.show()
        else:
            df['Epoch'] = pd.Series(epochArray)
            df['Best cost'] = pd.Series(epochBestCostArray)
            plt.xlabel("Epoch No.")
            plt.ylabel("Best cost")
            plt.plot(df['Epoch'], df['Best cost'])
            # plt.show()

    # Mutation adding with probability mu a Gaussian perturbation with standard deviation sigma
    def mutateGoodSolutionMuSigma(self, elite_solution, mu=0.1, sigma=0.1):
        chromosome = [elite_solution[i]+sigma*random.random() if random.random()
                      <= mu else elite_solution[i] for i in range(len(elite_solution))]
        return chromosome

    # mutates a randomly selected gene
    def mutateGoodSolution(self, elite_solution, min, max):
        point = random.randint(0, len(elite_solution)-1)
        chromosome = elite_solution[:]
        chromosome[point] = np.random.uniform(min, max)
        return chromosome

    # Crossover operator
    def crossover(self, dadChromosome, momChromosome, gamma=0.1):
        alpha = [random.uniform(-gamma, 1+gamma)
                 for _ in range(len(dadChromosome))]
        sonChromosome = [alpha[i]*dadChromosome[i] +
                         (1-alpha[i])*momChromosome[i] for i in range(len(dadChromosome))]
        daugtherChromosome = [alpha[i]*momChromosome[i] +
                              (1-alpha[i])*dadChromosome[i] for i in range(len(dadChromosome))]
        return sonChromosome

    def average_crossover(self, dadChromosome, momChromosome):
        """Average crossover mentioned in:
        Bessaou, M. and Siarry, P. (2001). A genetic algorithm with real-value coding to optimize multimodal continuous functions. Struct Multidisc Optim 23, 63–74"""
        sonChromosome = list()
        point1 = random.randint(0, len(dadChromosome)-1)
        for i in range(0, point1+1):
            sonChromosome.append(dadChromosome[i])
        for i in range(point1+1, len(dadChromosome)):
            sonChromosome.append((momChromosome[i]+dadChromosome[i])/2)
        return sonChromosome


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
        results = ['Beta', 'Alfa', 'Iterations', 'Mu', 'Sigma', 'Gamma'] + ['run'+str(i+1) for i in range(20)] + ['Mean', 'Exact results', 'Mean epochs', 'Mean time']
        fileoutput.append(results)
        parameters_space = [[0.48879283,0.3983071, 0.92941838,0.85038197,0.56528412,0.79814386],
                            [0.41221565,0.88644535,0.10703765,0.21039606,0.16987125,0.95126699],
                            [0.08936461,0.69561004,0.97006122,0.62272673,0.7252675, 0.22555018],
                            [0.75669131,0.0052953, 0.12500657,0.17210095,0.40778902,0.16530338],
                            [0.70180667,0.41585372,0.87419491,0.60399823,0.93833126,0.4775235 ],
                            [0.22589882,0.79915878,0.49639261,0.34612054,0.87425769,0.75390349],
                            [0.26906517,0.31776086,0.64603659,0.5705937, 0.7728089, 0.04628253],
                            [0.65513308,0.42215948,0.9087299, 0.03354716,0.54583676,0.13799875],
                            [0.51398582,0.55223248,0.69103951,0.74102226,0.85391825,0.58029675],
                            [0.98073945,0.94591753,0.13935146,0.09760536,0.82821696,0.37561444],
                            [0.87382209,0.45627181,0.65901554,0.07673713,0.98355544,0.05820586],
                            [0.21292397,0.2891327, 0.23824538,0.22911863,0.6128833, 0.60003425],
                            [0.54029464,0.85907224,0.31675715,0.29128393,0.04069262,0.21376382],
                            [0.15400238,0.48274475,0.228203,  0.97404062,0.91798448,0.44881414],
                            [0.67014315,0.15485923,0.82613936,0.24569962,0.8002101, 0.30816321],
                            [0.0800329, 0.70051665,0.72016135,0.80370034,0.10062881,0.45789241],
                            [0.56277271,0.03227423,0.93654173,0.55038885,0.26913014,0.48747543],
                            [0.14761393,0.0336162, 0.18293883,0.14110868,0.08575396,0.97906331],
                            [0.83333732,0.19752648,0.99078574,0.35951328,0.21858255,0.99406791],
                            [0.78400183,0.09273396,0.8089667, 0.91600286,0.33927082,0.59811207],
                            [0.44628378,0.92865314,0.39092159,0.72230558,0.13041192,0.40032339],
                            [0.82436945,0.4942517, 0.54100439,0.71549328,0.45143829,0.0855035,],
                            [0.52465859,0.11296065,0.46798579,0.45790605,0.63843325,0.81368949],
                            [0.33279904,0.73369923,0.06867841,0.4488876, 0.8926678, 0.72458299],
                            [0.1681529, 0.30617605,0.71436492,0.79868775,0.30743874,0.700524  ],
                            [0.63520573,0.61995852,0.33477946,0.11165526,0.78618545,0.50789674],
                            [0.00982175,0.27827339,0.0879377, 0.00316542,0.48694324,0.73724573],
                            [0.30687556,0.73021584,0.58950496,0.68616363,0.35163386,0.32661604],
                            [0.93229243,0.2448199, 0.16349228,0.63595884,0.00350533,0.35983859],
                            [0.11506889,0.83767603,0.96604486,0.91716953,0.57307663,0.53977294],
                            [0.99835594,0.66640836,0.77583958,0.89708454,0.9754916, 0.33745469],
                            [0.85907835,0.06256354,0.04910817,0.52992913,0.05534001,0.23706261],
                            [0.47541911,0.8129169, 0.57079586,0.87108823,0.21474245,0.82762505],
                            [0.57357783,0.37968678,0.83562068,0.83528643,0.02277283,0.65059567],
                            [0.06091776,0.21240304,0.44715527,0.51227379,0.51335763,0.56522377],
                            [0.36686896,0.51652647,0.89017192,0.15088779,0.58512991,0.63155412],
                            [0.24491905,0.36326829,0.62451991,0.66199974,0.25389528,0.85442648],
                            [0.33659806,0.97644546,0.35926755,0.67081008,0.67612828,0.13296938],
                            [0.29611725,0.87469152,0.76452759,0.33085347,0.43294691,0.07779331],
                            [0.19559429,0.914236,  0.60206052,0.78173446,0.5226776, 0.84379787],
                            [0.89286417,0.82694645,0.79794109,0.18880076,0.6277523, 0.17387667],
                            [0.77164586,0.07379525,0.26941116,0.1295911, 0.83968101,0.11616155],
                            [0.61930004,0.17395271,0.42011384,0.48255872,0.73900082,0.91138698],
                            [0.58463784,0.99434968,0.02966875,0.48431462,0.9541847, 0.39196616],
                            [0.04493147,0.13139948,0.52386235,0.2639333, 0.7529775, 0.77525623],
                            [0.11716582,0.76514939,0.73881367,0.06183551,0.39698732,0.18782066],
                            [0.69603347,0.33625964,0.26495117,0.76077474,0.16375433,0.91899899],
                            [0.72106064,0.57746211,0.40959345,0.36920071,0.32948098,0.02886728],
                            [0.42836748,0.52976705,0.86159937,0.98590189,0.43802263,0.64511514],
                            [0.93611682,0.95225326,0.46335907,0.54731543,0.37402202,0.28624745],
                            [0.96125967,0.44284803,0.51256301,0.27272046,0.65322403,0.27733883],
                            [0.35264109,0.22224403,0.29136186,0.39370023,0.70730027,0.68555019],
                            [0.90969065,0.58940204,0.66929884,0.95939976,0.07697052,0.52345921],
                            [0.80605131,0.63841355,0.55449126,0.59174296,0.90981634,0.43262484],
                            [0.60734974,0.61156262,0.30343649,0.02793778,0.1847757, 0.25231345],
                            [0.46520495,0.2585181, 0.36729472,0.4249352, 0.68445538,0.93496657],
                            [0.02703088,0.67839088,0.18378089,0.41187846,0.13995075,0.67154337],
                            [0.26653689,0.13854838,0.00187518,0.82606342,0.29824514,0.01601597],
                            [0.73684788,0.53637963,0.05590435,0.94070031,0.46866788,0.8843337 ],
                            [0.39444777,0.77852449,0.21306101,0.30099038,0.24312215,0.87574619]]
        for parameters in parameters_space:
            mean_cost = 0
            results = parameters
            exact_results = 0
            mean_epochs = 0
            mean_time = 0
            for i in range(20):
                # creates a PSO instance
                # beta is the probability for a global best movement
                start_time = process_time()
                pso = Solver(function, functions_search_space[function.__name__], max_epochs=500, population_size=10, beta=parameters[0], alfa=parameters[1], iterations=int(
                    50 + (parameters[2] * (300 - 50))), crossover_type='crossover', mutation_type='mutateGoodSolutionMuSigma', mu=parameters[3], sigma=parameters[4], gamma=parameters[5])
                pso.run()  # runs the PSO algorithm
                ms = (process_time() - start_time) * 1000.0
                mean_time += ms
                cost = pso.getGBest().getCostPBest()
                results.append(cost)
                exact_results += (1 if np.isclose(pso.getGBest().getCostPBest(), function(*functions_solution[function.__name__]), atol=1e-05) else 0)
                mean_cost += cost
                mean_epochs += pso.getEpoch()
            mean_cost /= 20.0
            mean_epochs /= 20.0
            mean_time /= 20.0
            results.append(mean_cost)
            results.append(exact_results)
            results.append(mean_epochs)
            results.append(mean_time)
            fileoutput.append(results)

        # pso-results.csv
        csvFile = open('results/micro-pso-continuous-griewank20-experiment.csv', 'w', newline='')
        writer = csv.writer(csvFile)
        writer.writerows(fileoutput)
        csvFile.close()
    else:
         for function_name in ['beale','biggs_exp2','biggs_exp3','biggs_exp4','biggs_exp5','biggs_exp6','cross_in_tray','drop_in_wave','dejong_f1','dejong_f2','dejong_f3','dejong_f4','dejong_f5','rosenbrock2','rosenbrock3','rosenbrock4','rosenbrock5','rosenbrock6','rosenbrock7','rosenbrock8','rosenbrock9','rosenbrock10','rosenbrock11','rosenbrock12','rosenbrock13','rosenbrock14','rosenbrock15','rosenbrock16','rosenbrock17','rosenbrock18','rosenbrock19','rosenbrock20','rastringin20','griewank20']:
            function = globals()[function_name]
            results = ['Function'] + ['OptimumSolution x'+str(i+1) for i in range(len(signature(function).parameters))] + ['Solution x'+str(i+1) for i in range(len(signature(function).parameters))] + ['Eucl. dist.', 'Exact solution', 'Exact solution (allclose)', 'Cost', 'Exact optimum', 'Comp. time', 'Epochs']
            fileoutput = []
            fileoutput.append(results)
            for i in range(30):
                results = []
                start_time = process_time()
                pso = Solver(function, functions_search_space[function.__name__], iterations=350, max_epochs=500, population_size=10,
                            beta=0.9, alfa=0.6, crossover_type='crossover', mutation_type='mutateGoodSolutionMuSigma', mu=0.5, sigma=0.7, gamma=0.7)
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
                epoch = pso.getEpoch()
                results.append(ms)
                results.append(epoch)
                fileoutput.append(results)

            # pso-results.csv
            csvFile = open('results/micro-pso-continuous-'+function_name+'.csv', 'w', newline='')
            writer = csv.writer(csvFile)
            writer.writerows(fileoutput)
            csvFile.close()