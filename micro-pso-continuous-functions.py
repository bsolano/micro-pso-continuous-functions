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

from operator import attrgetter
import random, sys, time, copy
import random
import copy
import csv
import math
import statistics
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from benchmark_functions import *
from inspect import signature
import numpy as np

    
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
      chromosome.append(min + (random.random() * (max - min)))
    return chromosome


# PSO algorithm
class Solver:

  def __init__(self, cost_function, search_space, iterations, max_epochs, population_size, beta=1, alfa=1, first_population_criteria='average_cost'):
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

    # initialized with a group of random particles (solutions)
    solutions = Particle.getRandomSolutions(self.nvars, search_space, self.population_size)
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
        solutions = Particle.getRandomSolutions(self.nvars, self.search_space, self.population_size)
        cost = self.evaluateSolutionsAverageCost(solutions)
        if cost < bestCost:
          bestCost = cost
          bestSolutions = list(solutions)
        del solutions[:]
        
    elif first_population_criteria=='diversity':
      mostDiverse = self.evaluateSolutionsDiversity(solutions)
    
      for _ in range(5):
        solutions = Particle.getRandomSolutions(self.nvars, self.search_space, self.population_size)
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

  def initPopulation(self, population_size):
    self.particles = [] # list of particles
    solutions = Particle.getRandomSolutions(self.nvars, self.search_space, population_size)
    self.population_size = population_size
    
    # checks if exists any solution
    if not solutions:
      print('Initial population empty! Try run the algorithm again...')
      sys.exit(1)

    # Select the best random population among 5 populations
    bestSolutions = list(solutions)
    bestCost = self.evaluateSolutionsAverageCost(solutions)    
    
    for _ in range(5):
      solutions = Particle.getRandomSolutions(self.nvars, self.search_space, self.population_size)
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

  
  # returns the elapsed milliseconds since the start of the program
  def elapsedTime(self, start_time):
    dt = datetime.now() - start_time
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    return ms
  
  def setEpoch(self, last_epoch):
    self.last_epoch = last_epoch
  
  def getEpoch(self):
    return self.last_epoch

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
    
    epoch = 0
    while epoch < self.max_epochs:
      print("Epoch: ", epoch, "with ", self.population_size, " particles")
      print("Alfa = ", self.alfa, "Beta = ", self.beta)
      convergencePerEpoch = []

      if epoch > 0:
        self.initPopulation(self.population_size)
        print("Particles: ", len(self.particles))
        # Insert the best individual into the new population (1% of the population)
        if random.uniform(0,1.0) < 1.0:
          mutated_elite = self.mutateGoodSolution(self.gbest.getPBest())
          self.particles[random.randint(0, self.population_size-1)]  = Particle(mutated_elite, self.gbest.getCostPBest())
          print("Inserted elite solution!")
    
      # for each time step (iteration)
      for t in range(self.iterations):
        convergencePerIteration = []
        batchCounter = batchCounter + 1
        
        averageCost = statistics.mean(particle.pbestCost for particle in self.particles)
        costStd = statistics.pstdev(particle.pbestCost for particle in self.particles)

        # for each particle in the swarm
        for particle in self.particles:
          previousCost = particle.getCurrentSolutionCost()
          
          particle.clearVelocity() # cleans the speed of the particle
          gbest = list(self.gbest.getPBest()) # gets solution of the gbest solution
          
          if len(particle.history) == HISTORY_SIZE:
            particle.history.pop(0)
          
          bestNeighbor = self.mutateGoodSolution(particle.getCurrentSolution())
          bestNeighborCost = self.cost_function(*bestNeighbor)
          
          newSolution = particle.getCurrentSolution()[:]

          if random.random() <= self.beta:
            newSolution = self.crossover(list(newSolution), self.gbest.getPBest())
          elif random.random() <= self.alfa:
            largest_dist = 0
            for neighbor_particle in self.particles:
              sol = neighbor_particle.getPBest()
              dist = euclidean(gbest, sol)

              if dist > largest_dist:
                largest_dist = dist
                dissimilar_particle = neighbor_particle
            newSolution = self.crossover(list(newSolution), dissimilar_particle.getPBest())

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
    
      epoch = epoch + 1
      self.setEpoch(epoch)
      bestCostSampling.append(self.gbest.getCostPBest())
      if epoch > 5:
        std = statistics.pstdev(bestCostSampling[-10:])
        print("standard deviation: ", std)
      else:
        std = 1000
      
      if std == 0:
        break
    
    print("What's going on?")
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
  def mutateGoodSolution(self, elite_solution, mu=0.01, sigma=0.1):
    chromosome = [elite_solution[i]+sigma*random.random() if random.random() <= mu else elite_solution[i] for i in range(len(elite_solution))]
    return chromosome

  # Crossover operator
  def crossover(self, dadChromosome, momChromosome, gamma=0.1):
    alpha = [random.uniform(-gamma, 1+gamma) for _ in range(len(dadChromosome))]
    sonChromosome = [alpha[i]*dadChromosome[i] + (1-alpha[i])*momChromosome[i] for i in range(len(dadChromosome))]
    daugtherChromosome = [alpha[i]*momChromosome[i] + (1-alpha[i])*dadChromosome[i] for i in range(len(dadChromosome))]
    return sonChromosome


# Define Chromosome as a subclass of list
class Chromosome(list):
  def __init__(self):
    self.elements = []


if __name__ == "__main__":

  # creates a PSO instance
  # beta is the probability for a global best movement
  results = ["Solution", "Cost", "Comp. time"]
  fileoutput = []
  fileoutput.append(results)
  function = 'biggs_exp4'
  for i  in range(2):
    results = []
    pso = Solver(globals()[function], functions_search_space[function], iterations=1000, max_epochs=200, population_size=10, beta=0.29, alfa=0.12)
    start_time = datetime.now()
    pso.run() # runs the PSO algorithm
    results.append(pso.getGBest().getPBest())
    results.append(pso.getGBest().getCostPBest())
    epoch = pso.getEpoch()
    dt = datetime.now() - start_time
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    results.append(ms)
    results.append(epoch)
    fileoutput.append(results)
  
  # pso-results.csv  
  csvFile = open('micro-pso-continuo.csv', 'w', newline='')  
  with csvFile: 
    writer = csv.writer(csvFile)
    writer.writerows(fileoutput)
  csvFile.close()