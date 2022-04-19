# -*- coding: utf-8 -*-


####################################################################################
# A Particle Swarm Optimization algorithm to find functions optimum.
# 
#  
# Author: Rafael Batres
# Contributor: Braulio Solano
# Institution: Tecnológico de Monterrey
# Date: June 6, 2018 - April 2022
####################################################################################

import random, sys, copy
import random
import copy
import csv
import statistics
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from benchmark_functions import *
from inspect import signature
from math import isclose

    
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
  def getRandomSolutions(size, min, max, max_size):
    random_solutions = []
    
    for _ in range(max_size):
      
      list_temp = Particle.getRandomSolution(size, min, max)

      if list_temp not in random_solutions:
        random_solutions.append(list_temp)

    return random_solutions

  # Generate a random sequence and stores it
  # as a Route
  def getRandomSolution(size, min, max):
    chromosome = Chromosome()
    for i in range(size):
      chromosome.append(min[i] + (random.random() * (max[i] - min[i])))
    return chromosome


# PSO algorithm
class Solver:

  def __init__(self, cost_function, search_space, iterations, max_epochs, population_size, beta=1, alfa=1, first_population_criteria='average_cost', crossover_type='average_crossover', mutation_type='mutateGoodSolution', mu=0.1, sigma=0.1, gamma=0.1):
    self.cost_function = cost_function # the cost function
    self.nvars = len(signature(cost_function).parameters) # number of variables in the cost function
    self.search_space = search_space # interval of the cost function
    self.iterations = iterations # max of iterations
    self.max_epochs = max_epochs
    self.population_size = population_size # size population
    self.particles = [] # list of particles
    self.beta = beta # the probability that all swap operators in swap sequence (gbest - x(t-1))
    self.alfa = alfa # the probability that all swap operators in swap sequence (pbest - x(t-1))
    self.last_epoch = 0
    self.crossover_type = crossover_type
    self.mutation_type = mutation_type
    self.mu = mu
    self.sigma = sigma
    self.gamma = gamma

    min = [self.search_space[0] for _ in range(self.nvars)]
    max = [self.search_space[1] for _ in range(self.nvars)]

    # initialized with a group of random particles (solutions)
    solutions = Particle.getRandomSolutions(self.nvars, min, max, self.population_size)
    print("One initial solution: ", solutions[0])
    
    # checks if exists any solution
    if not solutions:
      print('Initial population empty! Try run the algorithm again...')
      sys.exit(1)
    
    # Select the best random population among 5 populations
    bestSolutions = list(solutions)

    if first_population_criteria=='average_cost':
      bestCost = self.evaluateSolutionsAverageCost(solutions)
    
      for _ in range(5):
        solutions = Particle.getRandomSolutions(self.nvars, min, max, self.population_size)
        cost = self.evaluateSolutionsAverageCost(solutions)
        if cost < bestCost:
          bestCost = cost
          bestSolutions = list(solutions)
        del solutions[:]
        
    elif first_population_criteria=='diversity':
      mostDiverse = self.evaluateSolutionsDiversity(solutions)
    
      for _ in range(5):
        solutions = Particle.getRandomSolutions(self.nvars, min, max, self.population_size)
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
      particle = Particle(solution=solution, cost=self.cost_function(*solution))
      # add the particle
      self.particles.append(particle)
      # updates gbest if needed
      if self.gbest is None:
        self.gbest = copy.deepcopy(particle)
      elif self.gbest.getCostPBest() > particle.getCostPBest():
        self.gbest = copy.deepcopy(particle)

  def initPopulation(self, population_size, min, max):
    self.particles = [] # list of particles
    solutions = Particle.getRandomSolutions(self.nvars, min, max, population_size)
    self.population_size = population_size
    
    # checks if exists any solution
    if not solutions:
      print('Initial population empty! Try run the algorithm again...')
      sys.exit(1)

    # Select the best random population among 5 populations
    bestSolutions = list(solutions)
    bestCost = self.evaluateSolutionsAverageCost(solutions)    
    
    for _ in range(5):
      solutions = Particle.getRandomSolutions(self.nvars, min, max, self.population_size)
      cost = self.evaluateSolutionsAverageCost(solutions)
      if cost < bestCost:
        bestCost = cost
        bestSolutions = list(solutions)
      del solutions[:]
    
    # creates the particles and initialization of swap sequences in all the particles
    for solution in bestSolutions:
      # creates a new particle
      particle = Particle(solution=solution, cost=self.cost_function(*solution))
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
      i+=1
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

    batchSize = 100 # save data every n iterations
    batchCounter = 0

    HISTORY_SIZE = 100
    
    record_values = [['Epoch'] + ['x'+str(i+1)+'='+str(int(functions_solution[self.cost_function.__name__][i])) for i in range(self.nvars)] + ['max x'+str(i+1) for i in range(self.nvars)] + ['min x'+str(i+1) for i in range(self.nvars)]]
    epoch = 0
    while epoch < self.max_epochs:
      print("Epoch: ", epoch, "with ", self.population_size, " particles")
      print('Iterations', self.iterations)
      print("Alfa = ", self.alfa, "Beta = ", self.beta)
      convergencePerEpoch = []

      if epoch > 0:
        self.initPopulation(self.population_size, min_values, max_values)
        print("Particles: ", len(self.particles))
        # Insert the best individual into the new population (1% of the population)
        if random.uniform(0,1.0) < 1.0:
          if self.mutation_type == 'mutateGoodSolution':
            mutated_elite = getattr(self, self.mutation_type)(particle.getCurrentSolution(), [self.search_space[0] for _ in range(self.nvars)], [self.search_space[1] for _ in range(self.nvars)])
          elif self.mutation_type == 'mutateGoodSolutionMuSigma':
            mutated_elite = getattr(self, self.mutation_type)(particle.getCurrentSolution(), self.mu, self.sigma)
          self.particles[random.randint(0, self.population_size-1)]  = Particle(mutated_elite, self.gbest.getCostPBest())
          print("Inserted elite solution!")
      else:
          variables = zip(*self.getCurrentSolutions())
          max_values = list(map(max, variables))
          variables = zip(*self.getCurrentSolutions())
          min_values = list(map(min, variables))

      # for each time step (iteration)
      for t in range(self.iterations):
        convergencePerIteration = []
        batchCounter = batchCounter + 1
        
        averageCost = statistics.mean(particle.pbestCost for particle in self.particles)
        costStd = statistics.pstdev(particle.pbestCost for particle in self.particles)

        if self.mutation_type == 'mutateGoodSolution':
          variables = zip(*self.getCurrentSolutions())
          max_values = list(map(max, variables))
          variables = zip(*self.getCurrentSolutions())
          min_values = list(map(min, variables))
        # for each particle in the swarm
        for particle in self.particles:
          previousCost = particle.getCurrentSolutionCost()
          
          particle.clearVelocity() # cleans the speed of the particle
          gbest = list(self.gbest.getPBest()) # gets solution of the gbest solution
          
          if len(particle.history) == HISTORY_SIZE:
            particle.history.pop(0)
          
          if self.mutation_type == 'mutateGoodSolution':
            #bestNeighbor = getattr(self, self.mutation_type)(particle.getCurrentSolution(), *self.search_space)
            bestNeighbor = getattr(self, self.mutation_type)(particle.getCurrentSolution(), min_values, max_values)
          elif self.mutation_type == 'mutateGoodSolutionMuSigma':
            bestNeighbor = getattr(self, self.mutation_type)(particle.getCurrentSolution(), self.mu, self.sigma)
          bestNeighborCost = self.cost_function(*bestNeighbor)
          
          newSolution = particle.getCurrentSolution()[:]

          if random.random() <= self.beta:
            if self.crossover_type=='average_crossover':
              newSolution = getattr(self, self.crossover_type)(list(newSolution), self.gbest.getPBest())
            elif self.crossover_type=='crossover':
              newSolution = getattr(self, self.crossover_type)(list(newSolution), self.gbest.getPBest(), gamma=self.gamma)
          elif random.random() <= self.alfa:
            largest_dist = 0
            for neighbor_particle in self.particles:
              sol = neighbor_particle.getPBest()
              dist = euclidean(gbest, sol)

              if dist > largest_dist:
                largest_dist = dist
                dissimilar_particle = neighbor_particle
            if self.crossover_type=='average_crossover':
              newSolution = getattr(self, self.crossover_type)(list(newSolution), dissimilar_particle.getPBest())
            elif self.crossover_type=='crossover':
              newSolution = getattr(self, self.crossover_type)(list(newSolution), dissimilar_particle.getPBest(), gamma=self.gamma)

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
          pbCost =  particle.getCostPBest()
          
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
    
      variables = zip(*self.getCurrentSolutions())
      max_values = list(map(max, variables))
      variables = zip(*self.getCurrentSolutions())
      min_values = list(map(min, variables))
      record_values.append([epoch] + self.gbest.getPBest() + max_values + min_values)
      epoch = epoch + 1
      self.setEpoch(epoch)
      bestCostSampling.append(self.gbest.getCostPBest())
      if epoch > 5:
        std = statistics.pstdev(bestCostSampling[-10:])
        print("standard deviation: ", std)
      else:
        std = 1000
      
      if isclose(std, 0):
        break
    
    valuesFile = open('values-'+str(datetime.timestamp(datetime.now()))+'.csv', 'w', newline='')
    with valuesFile:
      writer = csv.writer(valuesFile)
      writer.writerows(record_values)
    valuesFile.close()

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
      #plt.show()

  # Mutation adding with probability mu a Gaussian perturbation with standard deviation sigma
  def mutateGoodSolutionMuSigma(self, elite_solution, mu=0.1, sigma=0.1):
    chromosome = [elite_solution[i]+sigma*random.random() if random.random() <= mu else elite_solution[i] for i in range(len(elite_solution))]
    return chromosome

  # mutates a randomly selected gene
  def mutateGoodSolution(self, elite_solution, min, max):
    point = random.randint(0, len(elite_solution)-1)
    chromosome = elite_solution[:]
    chromosome[point] = min[point] + (random.random() * (max[point] - min[point]))
    return chromosome

  # Crossover operator
  def crossover(self, dadChromosome, momChromosome, gamma=0.1):
    alpha = [random.uniform(-gamma, 1+gamma) for _ in range(len(dadChromosome))]
    sonChromosome = [alpha[i]*dadChromosome[i] + (1-alpha[i])*momChromosome[i] for i in range(len(dadChromosome))]
    daugtherChromosome = [alpha[i]*momChromosome[i] + (1-alpha[i])*dadChromosome[i] for i in range(len(dadChromosome))]
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

  # creates a PSO instance
  # beta is the probability for a global best movement
  run_experiment = False
  function = 'biggs_exp4'
  if run_experiment == True:
    fileoutput = []
    results = ['Beta','Alfa','Iterations','Crossover type','Mutation type','Mu','Sigma','Gamma'] + ['run'+str(i+1) for i in range(20)] + ['Mean','Exact results']
    fileoutput.append(results)
    parameters_space = [[9.23740174e-01,4.18338939e-01,1.16484035e-01,4.68120881e-01,8.64381462e-02,4.28087985e-01,5.55358928e-01,4.31981332e-01],
                        [1.69841060e-02,9.56257029e-01,2.69397809e-01,1.76249345e-01,3.95815028e-01,8.94280364e-01,7.52131899e-01,6.27758710e-01],
                        [8.68818242e-01,2.26163514e-01,1.56844151e-03,3.29672422e-01,4.02197303e-01,8.58350595e-01,3.47950290e-01,2.74029314e-01],
                        [6.10067997e-01,3.59528220e-01,1.08274002e-01,2.46122704e-01,9.21591465e-01,1.59795448e-01,1.07200602e-01,8.54175779e-01],
                        [9.97473859e-01,8.56754161e-01,1.46028624e-02,2.62035437e-01,7.16828943e-01,7.32896033e-01,2.41591961e-01,9.38484082e-02],
                        [1.73796799e-01,7.43612852e-01,2.47929156e-01,3.79032931e-01,3.65678459e-02,7.23608520e-01,5.69626704e-01,8.65648543e-01],
                        [5.19371367e-01,6.69862278e-01,3.37099870e-01,6.52457227e-01,2.76918262e-01,3.21050700e-01,4.87983662e-01,4.64969272e-01],
                        [8.00103785e-01,1.43120322e-01,7.05275676e-01,9.80273960e-01,8.33414206e-01,8.11623520e-01,9.32083360e-01,9.41918257e-01],
                        [2.90833091e-01,9.15077965e-01,8.08436739e-01,3.14421808e-02,2.52506668e-01,7.71729063e-01,9.59888527e-01,4.17443749e-02],
                        [6.83283573e-01,2.57488957e-01,6.50636424e-01,7.37793258e-01,3.70456619e-01,6.68882937e-01,9.98295709e-01,3.96894739e-01],
                        [9.02121384e-01,9.33852050e-01,1.97820697e-01,8.21257439e-01,3.62151228e-01,3.74574820e-01,4.23032199e-01,6.78064840e-01],
                        [5.07835461e-01,8.06951078e-01,5.63924871e-01,9.94938778e-01,4.40108125e-01,2.27868188e-01,4.53127929e-01,5.30993453e-01],
                        [6.31244659e-01,6.42104517e-01,2.27086194e-01,5.87856872e-02,8.59325240e-01,7.89022874e-01,6.69261383e-01,9.85043324e-01],
                        [1.57512186e-01,4.84979220e-02,5.36511231e-01,6.75971956e-01,1.99617311e-01,6.85821201e-02,6.54102145e-02,1.22506099e-01],
                        [3.71324541e-01,3.15032357e-01,4.81711503e-01,4.54727976e-02,7.34967725e-01,1.83866373e-01,5.40591106e-01,5.21683058e-01],
                        [2.19952682e-01,2.18559959e-01,8.17057287e-01,4.30606900e-01,7.62993527e-01,3.50109089e-01,6.81165184e-01,1.83970075e-01],
                        [8.84207131e-01,5.47145363e-01,9.31273112e-01,9.68079427e-01,9.71893177e-01,4.68357994e-01,6.88355250e-01,7.09702621e-01],
                        [7.74959815e-01,1.61251028e-01,4.01972121e-01,3.09145160e-01,8.41884026e-01,1.31907949e-01,2.76097599e-01,6.91504560e-01],
                        [7.44848433e-01,9.01467181e-01,3.59544252e-01,5.24143032e-01,6.94441048e-01,4.41830050e-01,4.05762186e-01,7.16133488e-01],
                        [6.89423496e-01,1.36143088e-01,3.39697581e-01,1.31577946e-01,1.44506943e-01,8.66802366e-01,9.83691130e-01,9.55326780e-01],
                        [9.44566084e-01,8.68806475e-01,6.02512223e-01,8.99743230e-01,4.97253324e-01,7.02581674e-01,1.65696183e-01,9.74097545e-01],
                        [7.15931134e-01,8.96551518e-01,9.93455386e-01,1.41964495e-01,5.99484830e-01,2.18792378e-01,5.83959309e-01,3.39665039e-01],
                        [7.06080196e-01,7.61673041e-01,4.71789697e-01,7.30690898e-01,8.98716761e-01,5.25021687e-01,7.99476975e-01,7.61142479e-01],
                        [5.15290278e-02,1.13955203e-01,8.42832490e-01,4.51067707e-01,9.80608900e-01,6.82661749e-01,1.21791341e-01,8.41032659e-01],
                        [9.75430599e-01,5.35752901e-01,4.22094255e-01,4.01505507e-01,7.91965486e-01,5.57022940e-01,7.70471008e-01,4.23845910e-01],
                        [7.95053899e-01,5.50341674e-01,7.61519724e-01,8.70866221e-01,2.31775214e-01,2.72489913e-01,1.36784025e-01,5.66567801e-01],
                        [7.60162750e-01,3.48550212e-01,5.14472243e-02,1.04868001e-01,7.12895331e-02,8.10199557e-02,2.24776233e-01,5.10351873e-01],
                        [5.93146491e-01,5.19513445e-01,3.05646997e-01,7.59864009e-01,9.03277953e-01,1.21269280e-01,1.60130442e-01,1.98606404e-01],
                        [8.94636791e-01,4.66752399e-01,8.27953940e-01,6.70415474e-01,6.17605785e-01,3.49103813e-01,1.87356297e-01,1.70570204e-01],
                        [3.99150915e-01,7.36122533e-01,6.22830112e-01,8.38151848e-01,4.76900560e-01,6.10589992e-01,5.01528747e-01,9.01325239e-01],
                        [6.67393225e-01,1.79396562e-01,2.19614639e-01,7.91222303e-01,6.43845556e-01,1.87662500e-01,3.28441341e-01,4.61134537e-01],
                        [5.80884846e-01,1.97170794e-01,8.04867522e-02,4.39938835e-01,5.00826987e-01,3.97856856e-01,7.15988977e-04,8.22154928e-01],
                        [7.32565645e-01,7.85408017e-02,4.93928766e-01,8.35798953e-01,3.00827012e-01,4.22426947e-01,4.86878901e-01,3.70387683e-02],
                        [4.28258826e-01,7.67147679e-01,8.97462025e-01,9.38413159e-02,5.28094383e-01,4.93889013e-01,2.62397512e-01,6.48151640e-01],
                        [6.20784141e-01,3.91014361e-01,3.63550699e-01,1.90850639e-01,9.55949566e-01,4.59581018e-01,9.67261115e-01,3.59613130e-01],
                        [8.49546186e-01,8.49605889e-01,6.32615185e-01,7.80285121e-01,5.66452411e-01,5.68635647e-01,5.24653947e-02,5.11212810e-02],
                        [2.88715866e-02,4.06994605e-01,1.55164136e-01,7.71202693e-01,1.54069395e-01,9.48175830e-01,8.52559385e-01,9.28551472e-01],
                        [8.14698685e-01,9.23752763e-02,8.70893647e-01,2.30519543e-01,6.36315923e-01,4.87927950e-02,1.96261416e-01,1.53539047e-01],
                        [3.76621422e-01,2.26943616e-02,3.87140567e-01,2.11520941e-01,7.11230236e-01,1.67394289e-01,5.92070163e-01,3.05378958e-01],
                        [4.97725107e-01,8.82040228e-01,9.66411255e-01,6.19039311e-01,1.74007611e-01,9.09722635e-02,7.10861919e-01,3.87200383e-01],
                        [6.57660391e-01,3.70256536e-01,1.28633991e-01,4.84695601e-01,9.35420618e-01,9.23798921e-01,9.40384141e-01,2.15682086e-01],
                        [4.41113663e-01,2.86444685e-01,7.25872721e-01,3.73993829e-01,5.56270790e-01,6.88379010e-01,2.25938661e-01,6.72737986e-01],
                        [3.99964576e-02,2.09666634e-01,4.52555724e-01,5.54508386e-01,2.08595019e-02,5.01953289e-01,8.96008392e-01,4.44174226e-01],
                        [9.52484477e-01,7.88439246e-03,9.11761028e-01,3.56053844e-01,1.78673309e-01,9.63812623e-01,3.64912357e-01,9.17781245e-01],
                        [1.31932402e-01,6.65871747e-02,6.42738215e-01,6.88227385e-01,5.85496724e-01,3.32403405e-01,6.54873539e-01,8.86585739e-01],
                        [4.23924114e-01,1.07171641e-01,5.87007244e-01,5.01821111e-01,4.04793983e-02,1.03872208e-01,9.06353159e-01,2.26510077e-01],
                        [5.72241761e-01,6.07829440e-01,7.84017751e-01,5.85904707e-01,8.18609835e-01,9.81519566e-01,9.32807256e-02,7.29024235e-01],
                        [4.01853516e-01,6.23300919e-01,7.93368375e-01,1.23317514e-01,1.04213175e-01,6.26292493e-01,8.63154305e-01,8.05624134e-01],
                        [2.71356519e-01,3.00288762e-01,9.24795708e-01,1.54771107e-01,7.57017855e-01,6.13770764e-01,2.37571604e-02,1.40071492e-01],
                        [6.39116845e-01,6.07838322e-02,1.72009331e-01,9.22694387e-01,2.01245355e-01,9.92325314e-03,7.48065153e-01,6.58292200e-01],
                        [3.43377184e-01,5.84282861e-01,5.16893741e-01,1.65762307e-01,3.77624462e-01,9.12072641e-01,8.06809345e-01,8.91727177e-01],
                        [2.35166193e-01,2.71441169e-01,9.87002555e-01,9.04745532e-01,3.29946786e-01,2.78646276e-01,3.02310548e-01,2.98572848e-01],
                        [8.06971888e-02,9.93228038e-01,2.75745800e-01,8.86782572e-01,5.23717978e-01,6.60154692e-01,3.67467262e-02,7.85763396e-01],
                        [7.75024640e-01,2.94258882e-01,5.38288413e-01,3.47836802e-01,4.50647166e-01,8.49830856e-01,3.52201026e-01,8.29091477e-01],
                        [3.54041426e-01,6.97117944e-01,1.43865040e-01,2.97250018e-03,6.50168990e-01,5.48045350e-01,6.19636073e-01,6.07831789e-01],
                        [2.77836382e-01,8.22128086e-01,2.88642681e-01,2.75675960e-01,2.90144184e-01,8.30223763e-01,5.15220193e-01,2.60891143e-01],
                        [2.55049029e-01,4.42874840e-01,6.98892003e-01,8.27845931e-02,1.33411029e-01,2.06214226e-02,9.15871129e-01,2.06979832e-01],
                        [1.07200025e-01,7.11720745e-01,5.98201757e-01,5.69864251e-01,3.49680407e-01,4.77932639e-01,7.84455428e-01,4.98519837e-01],
                        [1.18876985e-01,5.90375804e-01,8.60105965e-01,2.13699540e-01,9.37964726e-04,1.43723497e-01,1.48437372e-01,1.78162804e-02],
                        [4.65230683e-01,8.33814827e-01,5.05198051e-01,9.29828627e-01,8.81559236e-01,5.23538050e-01,6.03016732e-01,7.46728219e-01],
                        [2.41135068e-01,9.67992944e-01,2.00098225e-01,3.93668184e-01,3.21277620e-01,2.07353759e-01,2.98028622e-01,5.91726648e-01],
                        [3.08780629e-01,3.81488737e-01,9.11224454e-02,5.45023411e-01,4.72176978e-01,4.12238586e-01,5.25832642e-01,3.74197043e-01],
                        [5.46353194e-01,7.16478932e-01,6.43132133e-02,8.04094508e-01,9.40023719e-01,7.55094160e-01,6.35155723e-01,5.79162924e-01],
                        [8.28304433e-01,7.78780885e-01,6.76396425e-01,6.09458632e-01,1.20131423e-01,8.81623392e-01,3.92753711e-02,4.04192516e-01],
                        [1.97175577e-01,9.44292261e-01,5.57738171e-01,5.98718841e-01,4.21520935e-01,2.93986529e-01,8.44496138e-01,1.02591129e-02],
                        [4.86187755e-01,4.88962575e-01,3.97926887e-01,2.73844512e-01,2.64523972e-01,3.09472424e-01,2.65610630e-01,1.04575521e-01],
                        [1.84178379e-01,2.48917572e-01,9.55906406e-01,6.48539749e-01,5.38794926e-01,5.83717238e-01,3.23937022e-01,6.15507062e-01],
                        [5.34241095e-01,4.86961040e-01,8.75388778e-01,7.06532898e-02,9.88947828e-01,9.91022201e-01,7.25968334e-01,7.19508092e-02],
                        [8.39080347e-03,4.32443865e-01,7.70320795e-01,2.93224646e-01,5.96967547e-02,5.32043886e-02,8.70040735e-02,8.14571959e-02],
                        [2.01776968e-01,5.07246489e-01,6.72386173e-01,9.51873165e-01,4.29743208e-01,2.48282349e-01,3.79352665e-01,2.43475478e-01],
                        [3.35839923e-01,4.55114893e-01,7.39000742e-01,1.91303813e-02,8.07313400e-01,7.44242164e-01,2.06629640e-01,2.85677829e-01],
                        [3.15801577e-01,9.79555442e-01,4.37165228e-01,7.04465385e-01,6.81163457e-01,5.91889688e-01,4.64177931e-01,7.90939307e-01],
                        [8.55892189e-01,5.66769105e-01,1.81116480e-01,7.16836524e-01,6.73494102e-01,9.27743075e-01,7.21806485e-01,1.26924452e-01],
                        [9.63423992e-01,3.30100896e-01,2.56345641e-01,8.52861194e-01,7.81390371e-01,6.43870580e-01,6.49313370e-01,3.29627386e-01],
                        [4.56596857e-01,7.97672764e-01,3.13405298e-02,9.47204344e-01,8.73295091e-01,7.87177010e-01,3.99446990e-01,5.51137499e-01],
                        [5.52703482e-01,6.77461302e-01,7.24907958e-01,3.17196342e-01,2.41735916e-01,2.55761778e-01,4.31304508e-01,7.64078696e-01],
                        [9.34964226e-01,1.72028262e-01,9.48830423e-01,4.98073940e-01,6.02798497e-01,3.18484936e-02,4.39577508e-01,4.85705465e-01],
                        [7.36080904e-02,6.54315116e-01,3.21376103e-01,5.26070001e-01,8.75191059e-02,3.75098024e-01,8.75350903e-01,3.15180812e-01],
                        [1.45778675e-01,2.87623413e-02,4.38354046e-02,6.37039452e-01,7.45947210e-01,9.56357064e-01,8.22091181e-01,5.47870751e-01],
                        [9.13259646e-02,6.37357631e-01,4.39274288e-01,4.19243438e-01,2.22000638e-01,8.12698014e-01,8.36703878e-01,9.89981327e-01]]
    for parameters in parameters_space:
      mean_cost = 0
      results = parameters
      exact_results = 0
      for i in range(20):
        if parameters[3] < 0.5:
          crossover_type='average_crossover'
        else:
          crossover_type='crossover'
        if parameters[4] < 0.5:
          mutation_type='mutateGoodSolution'
        else:
          mutation_type='mutateGoodSolutionMuSigma'
        pso = Solver(globals()[function], functions_search_space[function], max_epochs=500, population_size=10, beta=parameters[0], alfa=parameters[1], iterations=int(50 + (parameters[2] * (250 - 50))), crossover_type=crossover_type, mutation_type=mutation_type, mu=parameters[5], sigma=parameters[6], gamma=parameters[7])
        pso.run() # runs the PSO algorithm
        cost = pso.getGBest().getCostPBest()
        results.append(cost)
        if cost == 0.0:
          exact_results += 1
        mean_cost += cost
      mean_cost /= 20.0
      results.append(mean_cost)
      results.append(exact_results)
      fileoutput.append(results)
  else:
    results = ["Function", "OptimumSolution", "Solution", "Cost", "Comp. time", "Epochs"]
    fileoutput = []
    fileoutput.append(results)
    for i in range(5):
      results = []
      pso = Solver(globals()[function], functions_search_space[function], iterations=200, max_epochs=500, population_size=10, beta=0.29, alfa=0.12)
      start_time = datetime.now()
      pso.run() # runs the PSO algorithm
      results.append(function)
      results.append(functions_solution[function])
      results.append(pso.getGBest().getPBest())
      results.append(pso.getGBest().getCostPBest())
      epoch = pso.getEpoch()
      dt = datetime.now() - start_time
      ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
      results.append(ms)
      results.append(epoch)
      fileoutput.append(results)

  # pso-results.csv
  if run_experiment == True:
    csvFile = open('micro-pso-continuous-experiment.csv', 'w', newline='')
  else:
    csvFile = open('micro-pso-continuous.csv', 'w', newline='')
  with csvFile: 
    writer = csv.writer(csvFile)
    writer.writerows(fileoutput)
  csvFile.close()