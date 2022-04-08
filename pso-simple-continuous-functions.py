# encoding:utf-8

####################################################################################
# A Particle Swarm Optimization algorithm for solving the traveling salesman problem.
# The program reuses part of the code of Marco Castro (https://github.com/marcoscastro/tsp_pso)
#  
# Author: Rafael Batres
# Institution: Tecnologico de Monterrey
# 
# Date: October 27, 2021
####################################################################################

from operator import attrgetter
import random, sys, time, copy
import random
import copy
import csv
import math
import statistics
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from benchmark_functions import *
from inspect import signature


# PSO algorithm
class PSO:


  def __init__(self, cost_function, search_space, iterations, population_size, inertia=0.8, particle_confidence=0.1, swarm_confidence=0.1):
    self.cost_function = cost_function # the cost function
    self.nvars = len(signature(cost_function).parameters) # number of variables in the cost function
    self.search_space = search_space # interval of the cost function
    self.iterations = iterations # max of iterations
    self.population_size = population_size # size population
    self.particles = [] # list of particles
    self.inertia = inertia
    self.p_confidence = particle_confidence
    self.s_confidence = swarm_confidence

    # initialized with a group of random particles (solutions)
    solutions = Particle.getRandomSolutions(self.nvars, search_space, self.population_size)
       
    # checks if exists any solution
    if not solutions:
      print('Initial population empty! Try run the algorithm again...')
      sys.exit(1)
    
    # Select the best random population among 5 populations
    bestSolutions = list(solutions)
    bestCost = self.evaluateSolutionsAverageCost(solutions)
    for _ in range(5):
      solutions = Particle.getRandomSolutions(self.nvars, search_space, self.population_size)
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

    self.gbest = None


  def evaluateSolutionsAverageCost(self, solutions):
  
    totalCost = Decimal(0.0)
    i = 0
    for solution in solutions:
      cost = self.cost_function(*solution)
      totalCost += cost
      i+=1
    averageCost = float(totalCost) / float(i)

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

    batchSize = 100 # save data every n iterations
    batchCounter = 0
    
    startTime = datetime.now()
    
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
    #for t in range(self.iterations):
      convergencePerIteration = []
      batchCounter = batchCounter + 1

      # for each particle in the swarm
      for particle in self.particles:
        #previousCost = particle.getCurrentSolutionCost()
        velocity = particle.getVelocity()
        gbest = list(self.gbest.getPBest()) # gets solution of the gbest solution
        pbest = particle.getPBest()[:] # copy of the pbest solution
        currentSolution = particle.getCurrentSolution()[:] # gets copy of the current solution of the particle
        
        # updates velocity
        r1 = random.random()
        r2 = random.random()
        velocity = [self.inertia*velocity[i] + self.p_confidence*r1*(pbest[i]-currentSolution[i]) + self.s_confidence*r2*(gbest[i]-currentSolution[i]) for i in range(len(velocity))]
        particle.setVelocity(velocity)
                
        # new solution
        newSolution = [currentSolution[i] + velocity[i] for i in range(len(currentSolution))]
        
        # gets cost of the current solution
        newSolutionCost = self.cost_function(*newSolution)
        #if newSolutionCost < previousCost:
        # updates the current solution  
        particle.setCurrentSolution(newSolution)
        # updates the cost of the current solution
        particle.setCostCurrentSolution(newSolutionCost)               

        # checks if new solution is pbest solution
        pbCost =  particle.getCostPBest()
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
          print(t, "Gbest cost = ", "{:.20f}".format(self.gbest.getCostPBest()))
          #print("Standard deviation: ", std)
          batchCounter = 0
          bestCostSampling.append(self.gbest.getCostPBest())
        
      t = t + 1
      if t > 220:
        std = statistics.pstdev(bestCostSampling[-10:])
      else:
        std = 1000
      
      if t==self.iterations:
        self.setIter(t)

      if std == 0:
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
      solution.append(min + (random.random() * (max - min)))
    return solution


# Define Chromosome as a subclass of list
class Chromosome(list):
  def __init__(self):
    self.elements = []


if __name__ == "__main__":
  # creates a PSO instance
  # alfa is the probabiliy for a movement based on local best
  # beta is the probability for a movement based on the global best
  results = ["Solution", "Cost", "Comp. time", "Max iter"]
  fileoutput = []
  fileoutput.append(results)
  function = 'biggs_exp4'
  for i  in range(30):
    results = []
    pso = PSO(globals()[function], functions_search_space[function], iterations=50000, population_size=150, inertia=0.8, particle_confidence=1, swarm_confidence=2)
    start_time = datetime.now()
    pso.run() # runs the PSO algorithm
    results.append(pso.getGBest().getPBest())
    results.append(pso.getGBest().getCostPBest())
    iteration = pso.getIter()
    dt = datetime.now() - start_time
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    results.append(ms)
    results.append(iteration)
    fileoutput.append(results)
    # shows the global best particle
    print("Cost of gbest: ", "{:.20f}".format(pso.getGBest().getCostPBest())) 
    print("gbest: ", pso.getGBest().getPBest())
    print("")

  csvFile = open('pso-simple-continuous-functions.csv', 'w', newline='')  
  with csvFile: 
    writer = csv.writer(csvFile)
    writer.writerows(fileoutput)
  csvFile.close()
