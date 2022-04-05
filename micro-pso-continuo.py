# -*- coding: utf-8 -*-


####################################################################################
# A Particle Swarm Optimization algorithm for finding functions optimum.
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
    
    # initialization of all particles
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
    MAX_NEIGHBORS = 1

    startTime = datetime.now()
    
    # updates gbest (best particle of the population)
    #self.gbest = min(self.particles, key=attrgetter('pbestCost'))
    self.gbest = self.particles[0]
    
    eliteSolution = []
    
    
    epoch = 0
    while epoch < self.max_epochs:
      print("Epoch: ", epoch, "with ", self.population_size, " particles")
      print("Alfa = ", self.alfa, "Beta = ", self.beta)
      convergencePerEpoch = []

      eliteCost = sys.maxsize

      print("Particles: ", len(self.particles))
      if epoch > 0:
        # Insert the best individual into the new population (1% of the population)
        #for i in range(population_size):
        #print("Elite:", eliteSolution)
        if random.uniform(0,1.0) < 1.0:
          mutated_elite = self.mutateGoodSolution(eliteSolution)
          self.particles[0]  = Particle(mutated_elite, eliteCost)
          #self.particles[0]  = Particle(eliteSolution, eliteCost)
          print("Inserted elite solution!")
          
        
    
      # for each time step (iteration)
      for t in range(self.iterations):
        convergencePerIteration = []
        batchCounter = batchCounter + 1
        
        # updates gbest (best particle of the population)
        #self.gbest = min(self.particles, key=attrgetter('pbestCost'))
        #costVariance = (self.particles, key=attrgetter('pbestCost'))
        averageCost = statistics.mean(particle.pbestCost for particle in self.particles)
        costStd = statistics.pstdev(particle.pbestCost for particle in self.particles)


        # for each particle in the swarm
        for particle in self.particles:
          previousCost = particle.getCurrentSolutionCost()
          
          particle.clearVelocity() # cleans the speed of the particle
          #velocity = []
          #velocity_alfa = []
          # what is the difference btn. copy.copy and list?
          gbest = list(self.gbest.getPBest()) # gets solution of the gbest solution
          #pbest = particle.getPBest()[:] # copy of the pbest solution
          ##newSolution = particle.getCurrentSolution()[:] # gets copy of the current solution of the particle
          
          # the previous solution
          previousSolution = particle.getCurrentSolution()[:]
          
          if len(particle.history) == HISTORY_SIZE:
            particle.history.pop(0)
          
          bestNeighbor = self.mutateGoodSolution(particle.getCurrentSolution())
          #bestNeighbor = particle.getCurrentSolution()[:]
          bestNeighborCost = self.graph.evaluateCost(bestNeighbor)
          
          """
          if previousCost < bestNeighborCost:
            bestNeighbor = previousSolution[:]
            bestNeighborCost = previousCost
          """

            
          newSolution = particle.getCurrentSolution()[:]


          #random_particle = random.choice(self.particles)
          
          """
          if random.random() <= (1.0 - self.alfa - self.beta):
            newSolution = self.crossover(list(newSolution), previousSolution)
          elif random.random() <= self.beta:
            newSolution = self.crossover(list(newSolution), self.gbest.getPBest())
          elif random.random() <= self.alfa:
            newSolution = self.crossover(list(gbest), random_particle.getPBest())
          """  
          if random.random() <= self.beta:
            newSolution = self.crossover(list(newSolution), self.gbest.getPBest())
          elif random.random() <= self.alfa:
            largest_dist = 0
            #less_common = len(gbest)
            for neighbor_particle in self.particles:
              sol = neighbor_particle.getPBest()
              dist = hamming(gbest, sol)*len(sol)
              #dist = self.hammingDistance(gbest, sol)
              #sim = self.cosine_similarity(gbest, sol)
              #dist = (1 - sim)*52
              #common = self.commonEdges(gbest, sol)
              #if common < less_common:
              #sim = self.jaro_distance(gbest, sol)
              #dist = 1 - sim
              #dist = self.levenshteinDistanceDP(gbest, sol)
              if dist > largest_dist:
                largest_dist = dist
                #less_common = common
                dissimilar_particle = neighbor_particle
            newSolution = self.crossover(list(newSolution), dissimilar_particle.getPBest())
            #newSolution = self.crossover(list(gbest), random_particle.getPBest())

            # gets cost of the current solution
          newSolutionCost = self.graph.evaluateCost(newSolution)

          if newSolutionCost < bestNeighborCost:
            bestNeighbor = newSolution[:]
            bestNeighborCost = newSolutionCost

          if bestNeighborCost < previousCost and bestNeighbor not in particle.history:
              # updates the current solution  
            particle.setCurrentSolution(bestNeighbor)
              # updates the cost of the current solution
            particle.setCostCurrentSolution(bestNeighborCost)
            particle.history.append(bestNeighbor)


          #"""

          # checks if new solution is pbest solution
          pbCost =  particle.getCostPBest()
          
          """
          if particle.getCurrentSolutionCost() < pbCost:
            particle.setPBest(particle.getCurrentSolution())
            particle.setCostPBest(particle.getCurrentSolutionCost())
          """  
          if bestNeighborCost < pbCost:
            particle.setPBest(bestNeighbor)
            particle.setCostPBest(bestNeighborCost)

            #temperature = temperature + (pbCost - newSolutionCost) / (math.log(rnd))
          
          gbestCost = self.gbest.getCostPBest()
                   
          
            
            # Using the value of the Boltzmann constant
          
          # check if new solution is gbest solution
         
          if particle.getCurrentSolutionCost() < gbestCost:
            self.gbest = particle

           
        eliteSolution = self.getGBest().getPBest()[:]
        eliteCost = self.gbest.getCostPBest()
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

    """
    csvFile = open('pso-convergence.csv', 'w', newline='')  
    with csvFile: 
         writer = csv.writer(csvFile)
         writer.writerows(convergenceData)
    csvFile.close()
    print("Elapsed time: ", self.elapsedTime(startTime))
    """ 
    
  def acceptanceProbability(self, previous_solution, new_solution, temperature, boltzmann):
    # If the new solution is better, accept it
    if new_solution < previous_solution:
      return 1.0
    elif temperature == 0:
      return 0.0
    # if there is no change then accept the bad solution
    #elif new_solution == previous_solution:
    #  return 0.0
    # If the new solution is worse, calculate an acceptance probability
    else:
      try:
        boltzmannProb = math.exp((previous_solution - new_solution) / (temperature * boltzmann))
      except OverflowError:
        boltzmannProb = float('inf')
      
      #print("Boltzmann probability", boltzmannProb)
      #print("Worse solution accepted ...")
      return boltzmannProb

  # Reverse mutation
  def mutate(self, parent, point1, point2):
    chromosome = parent[:]

    # Inverse the genes between point1 and point2    
    while True:
      chromosome[point1],chromosome[point2] = chromosome[point2],chromosome[point1]
      point1 = point1 + 1
      point2 = point2 - 1
      # Greater or equal than
      if point1 >= point2:
        break
      
    return chromosome

# Use reverse mutation for elite and savings solutions
  def mutateGoodSolution(self, elite_solution):
    chromosome = elite_solution[:]

    point1 = -1
    point2 = -1
    while True:
      point1 = random.randint(0, len(chromosome)-1)
      point2 = random.randint(0, len(chromosome)-1)
      if point1 != point2:
        break
    # Swap the contents of the two points
    #chromosome[point1],chromosome[point2] = chromosome[point2],chromosome[point1]

    # Inverse the genes between point1 and point2
    while True:
      chromosome[point1],chromosome[point2] = chromosome[point2],chromosome[point1]
      point1 = point1 + 1
      point2 = point2 - 1
      if point1 > point2:
        break
    return chromosome

# Crossover operator with mutation
  # This is an ordered crossover in which the center part of the dad chromosome 
  # is passed to the son and the left and right parts come from the mom chromosome.
  # The right part is filled first using the right part of the mom and then
  # the left part of the mom, while omitting the genes already inserted.
  # Similarly, the left part is filled next, using the left part of the mom 
  # and then the center part of the mom, while omitting the genes already inserted.
  # Subsequently, the offspring is reverse-mutated.
  def crossover(self, dadChromosome, momChromosome):
    #print("")
    
    while True:
      point1 = random.randint(0, len(dadChromosome)-1)
      point2 = random.randint(0, len(dadChromosome)-1)
      if point1 < point2 and point1 !=0 and point2 != len(dadChromosome)-1:
        break
    #print("Point 1: ", point1)
    #print("Point 2: ", point2)
    #print("Dad's chromosome: ", dadChromosome)
    #print("Mom's chromosome: ", momChromosome)
    
    sonChromosome = [-1 for j in range(len(dadChromosome))]

    son_center_segment = list()
    son_left_segment = list()
    son_right_segment = list()
  
    sonChromosome = [-1 for j in range(len(dadChromosome))]
    
    # Copy the middle section
    for i in range(point1, point2+1):
      son_center_segment.append(dadChromosome[i])

    # Fill in the right section of the son
    i = point2 + 1
    j = point2 + 1
    gene = -1
    while i < len(sonChromosome):
      if j == len(momChromosome):
        j = 0
      while j < len(momChromosome):
        gene = momChromosome[j]
        j=j+1
        if gene not in son_center_segment:
          son_right_segment.append(gene)
          i = i + 1
          break
      
      
    # Fill in the left section of the son
    i=0
    j=0
    gene = -1
    while i < point1 and j < len(momChromosome):
      while j < len(momChromosome):
        gene = momChromosome[j]
        j=j+1
        if gene not in son_center_segment and gene not in son_right_segment:
          son_left_segment.append(gene)
          break
      i = i + 1
    
    sonChromosome = son_left_segment + son_center_segment + son_right_segment      
    # Swap the contents of the two points
    ##sonChromosome[point1],sonChromosome[point2] = sonChromosome[point2],sonChromosome[point1]
    #print("Son: ", sonChromosome)

    # After the crossover do a reverse mutation
    # Select two random points in the chromosome of the son
    
    point1 = -1
    point2 = -1
    while True:
      point1 = random.randint(0, len(sonChromosome)-1)
      point2 = random.randint(0, len(sonChromosome)-1)
      if point1 != point2:
        break
    # Inverse the genes between point1 and point2    
    while True:
      sonChromosome[point1],sonChromosome[point2] = sonChromosome[point2],sonChromosome[point1]
      point1 = point1 + 1
      point2 = point2 - 1
      if point1 > point2:
        break
    

    #print("Son: ", sonChromosome)
    #print("*******")
    return sonChromosome



  # The ordered crossover operator
  def oxcrossover(self, dadRoute, momRoute):
    #print("")
    
    visitedDad = [False for i in range(len(dadRoute))]
    visitedMom = [False for i in range(len(momRoute))]
    
  
    while True:
      point1 = random.randint(0, len(dadRoute)-1)
      point2 = random.randint(0, len(dadRoute)-1)
      if point1 < point2 and point1 !=0 and point2 != len(dadRoute)-1:
        break
        
    #print("Point 1: ", point1)
    #print("Point 2: ", point2)
    #print("Dad's route: ", dadRoute)
    #print("Mom's route: ", momRoute)
    
    sonRoute = Route()
    
    sonRoute = [-1 for j in range(len(dadRoute))]
    
    # Copy the middle section
    for i in range(point1, point2+1):
      sonRoute[i] = dadRoute[i]
      visitedDad[dadRoute[i]] = True
      visitedMom[momRoute[i]] = True

    # Fill in the right section of the son
    i = point2 + 1
    j = point2 + 1
    gene = -1
    while i < len(sonRoute):
      if j == len(momRoute):
        j = 0
      while j < len(momRoute):
        gene = momRoute[j]
        j=j+1
        if not visitedDad[gene]:
          sonRoute[i] = gene
          visitedDad[gene] = True
          i = i + 1
          break
      
      
    # Fill in the left section of the son
    i=0
    j=0
    gene = -1
    while i < point1 and j < len(momRoute):
      while j < len(momRoute):
        gene = momRoute[j]
        j=j+1
        if not visitedDad[gene]:
          sonRoute[i] = gene
          visitedDad[gene] = True
          break
      i = i + 1

      
    #print("Son's route: ", sonRoute)
    #print("Daughter's route: ", daughterRoute)
    
    """
    # This is the simple mutation
    # Select two random points in the route of the son
    point1 = -1
    point2 = -1
    while True:
      point1 = random.randint(0, len(sonRoute)-1)
      point2 = random.randint(0, len(sonRoute)-1)
      if point1 != point2:
        break
    # Swap the contents of the two points
    sonRoute[point1],sonRoute[point2] = sonRoute[point2],sonRoute[point1]
    """

    #"""  
    
    # After the crossover do a reverse mutation
    # Select two random points in the route of the parent
    point1 = -1
    point2 = -1
    while True:
      point1 = random.randint(0, len(sonRoute)-1)
      point2 = random.randint(0, len(sonRoute)-1)
      if point1 != point2:
        break
    # Inverse the genes between point1 and point2    
    while True:
      sonRoute[point1],sonRoute[point2] = sonRoute[point2],sonRoute[point1]
      point1 = point1 + 1
      point2 = point2 - 1
      if point1 > point2:
        break
    #"""    
    return sonRoute

  # Heuristic crossover
  def hcrossover(self, dadRoute, momRoute):

    #print("Mom:", momRoute)
    #print("Dad:", dadRoute)
    #print("Cost of mom:", momCost)
    #print("Cost of dad:", dadCost)
    
    sonRoute = [-1 for j in range(len(dadRoute))]
    visited = [False for k in range(len(dadRoute))]
    
    firstGene = sonRoute[0] = dadRoute[0]
    visited[firstGene] = True
    
    i = 1
    while i < len(dadRoute)+2:  
      fromGene = sonRoute[i-1]
      #print("From Gene:", fromGene)
      connectedGeneInMom = self.getConnectedGene(fromGene, momRoute)
      connectedGeneInDad = self.getConnectedGene(fromGene, dadRoute)
      costMomEdge = GRAPH[fromGene][connectedGeneInMom]
      costDadEdge = GRAPH[fromGene][connectedGeneInDad]
      
      sampleSize = 3
      if costDadEdge < costMomEdge:
        best = connectedGeneInDad
        worse = connectedGeneInMom
      else:
        best = connectedGeneInMom
        worse = connectedGeneInDad
      
      if visited[best]:
        if visited[worse]:
          winner = -1
          winnerCost = -1
          for l in range(0, sampleSize-1):
            candidate = self.getUnvisitedGene(dadRoute, visited)
            candidateCost = GRAPH[fromGene][candidate]
            if winner == -1 or candidateCost > winnerCost:
              winner = candidate
              winnerCost = candidateCost
        else:
          winner = worse
      else:
        winner = best
      #print(i, "winner", winner)
      sonRoute[i] = winner
      visited[winner] = True
      i += 1
      if i == len(dadRoute):
        break
      
    #sonCost = self.evaluateCost(sonRoute)
    #print("Cost of heuristic son:", sonCost)
      
    """
    missingGenes = False
    for j in range(0,  len(dadRoute)-1):
      if j not in sonRoute:
        print(j, " is missing")
        missingGenes = True
    if missingGenes:
      sys.exit("Gene is missing in son:")
    """
    
    #print(" ")
    
    #if sonCost < dadCost:
      #print("Heuristic son", sonRoute)
      #print("Son is better:", sonCost)
    return sonRoute
        

  # The alternative edge crossover
  def aexcrossover(self, dadRoute, momRoute):
    
    sonRoute = [-1 for j in range(len(dadRoute))]
    
    visited = [False for i in range(len(dadRoute))]
    
    fromGeneInDad = dadRoute[0]
    sonRoute[0] = fromGeneInDad
    connectedGeneInDad = self.getConnectedGene(fromGeneInDad, dadRoute)
    sonRoute[1] = connectedGeneInDad
    visited[fromGeneInDad] = True
    visited[connectedGeneInDad] = True
    i = 2
    
    while i < len(dadRoute):
      fromGeneInMom = connectedGeneInDad
      connectedGeneInMom = self.getConnectedGene(fromGeneInMom, momRoute)
      if visited[connectedGeneInMom]:
        #print("Gene in Mom already visited:", connectedGeneInMom)
        connectedGeneInMom = self.getUnvisitedGene(momRoute, visited)
        #print("Randomly selected momÂ´s gene:", connectedGeneInMom)
      sonRoute[i] = connectedGeneInMom
      #print("Connected gene in Mom:", connectedGeneInMom)
      visited[connectedGeneInMom] = True
      i += 1
      fromGeneInDad = connectedGeneInMom
      connectedGeneInDad = self.getConnectedGene(fromGeneInDad, dadRoute)
      if visited[connectedGeneInDad]:
        #print("Gene in Dad already visited:", connectedGeneInDad)
        connectedGeneInDad = self.getUnvisitedGene(dadRoute, visited)
        #print("Randomly selected dad's gene:", connectedGeneInDad)
      sonRoute[i] = connectedGeneInDad
      #print("Connected gene in Dad:", connectedGeneInDad)
      visited[connectedGeneInDad] = True
      i += 1

    #print("Son:", sonRoute)
    
    missingGenes = False
    for j in range(0,  len(dadRoute)-1):
      if j not in sonRoute:
        #print(j, " is missing")
        missingGenes = True
    if missingGenes:
      sys.exit("Gene is missing in son:")
      
    # Select two random points in the route of the son
    point1 = -1
    point2 = -1
    while True:
      point1 = random.randint(0, len(sonRoute)-1)
      point2 = random.randint(0, len(sonRoute)-1)
      if point1 != point2:
        break
        
    # Swap the contents of the two points
    #sonRoute[point1],sonRoute[point2] = sonRoute[point2],sonRoute[point1]
    return sonRoute

  def getConnectedGene(self, from_gene, route):
    i = 0
    to_gene = -1
    while i < len(route):
      if route[i] == from_gene and i == len(route)-1:
        to_gene = route[0]
        break
      if route[i] == from_gene:
        to_gene = route[i+1]
        break
      i += 1
    if to_gene == -1:
      print("Error!!!!!!")
      print("Route:", route)
      print("From gene:", from_gene)
      print("Connected gene:", to_gene)
      sys.exit("Error message")
    return to_gene
      
  def getUnvisitedGene(self, route, visited):
    while True:
      point = random.randint(0, len(route)-1)
      gene = route[point]
      if not visited[gene]:
        break
    #print("Visiting status of gene:", visited[gene])
    return gene


  # Calculate the objective function
  def evaluateCost(self, route):
    routeSize = len(route)
    
    # Determine the cost of traversing the cities
    first = -1
    second = -1
    last = -1
    cost = 0.0
    #print(routeSize)
    #print(route)
    i = 0
    while routeSize > 1 and i < routeSize-1:
      first = route[i]
      second = route[i+1]
      cost = cost + GRAPH[first][second]
      i = i + 1
    # Complete Hamiltonian circuit
    last = route[routeSize-1]
    first = route[0]
    cost = cost + GRAPH[last][first]
    return cost  

  def generateSavingsMatrix(self):
    savingsMatrix = []
    for i in range(1, GRAPH_SIZE):
      for j in range(1, GRAPH_SIZE):
        savings = GRAPH[i][0] + GRAPH[0][j] - GRAPH[i][j]
        # saving = CWSA_mtx[i,-1] + CWSA_mtx[j,-1] - CWSA_mtx[i,j]
        savingsMatrix.append(Savings(i, j, savings))
    savingsMatrix = sorted(savingsMatrix, reverse=True)
    return savingsMatrix
    # sorted(student_objects, key=lambda student: student.age)

  def getNodeFromSavings(self, last, savings_matrix  = [], best = []):
    for i in range(len(savings_matrix)):
      node = savings_matrix[i]
      fromNode = node.get_from_node()
      toNode = node.get_to_node()
      #print("Last: ", last)
      #print("fromNode:", fromNode)
      #print("toNode:", toNode)
      if fromNode == last and toNode not in best:
        return toNode

      

# An Individual stores its route along with
# its cost and fitness.
class Individual:
  def __init__(self, route):
    self.route = route
    self.cost = 0.0
    self.fitness = 0.0
  
  def getRoute(self):
    return self.route

  def setRoute(self, route):
    self.route = route
    
  def getFitness(self):
    return self.fitness

  def setFitness(self, fitness):
    self.fitness = fitness
    
  def getCost(self):
    return self.cost
    
  def setCost(self, cost):
    self.cost = cost


# Define Chromosome as a subclass of list
class Chromosome(list):
  def __init__(self):
    self.elements = []


# Nest nearest integer
def nint(number):
  return int(number + (0.5 if number > 0 else -0.5))
  
# Calculate Euclidean distance for one location
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


if __name__ == "__main__":

  # creates a PSO instance
  # beta is the probability for a global best movement
  #pso = PSO(graph, iterations=5000, maxEpochs=1, size_population=150, beta=0.1, alfa=0.9)
  
  """
  pso = PSO(graph, iterations=1000, maxEpochs=50, size_population=15, beta=0.05, alfa=0.9)
  pso.run()
  """

  
  results = ["Solution", "Cost", "Comp. time"]
  fileoutput = []
  fileoutput.append(results)
  function = 'cross_in_tray'
  for i  in range(20):
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
  

  """
  results = ["alpha", "beta", "mean"]
  fileoutput = []
  fileoutput.append(results)
  alfa_value = 0.1
  while alfa_value < 1:
    beta_value = 0.1
    while beta_value < 1:
      results = []
      cost_sum = 0.0
      runs = 0
      for i in range(10):
        runs += 1
        pso = Solver(graph, iterations=1000, maxEpochs=50, size_population=15, beta=beta_value, alfa=alfa_value)
        pso.run()
        cost_sum = cost_sum + pso.getGBest().getCostPBest()
      i = 0
      average_cost = cost_sum / runs
      results.append(alfa_value)
      results.append(beta_value)
      results.append(average_cost)
      fileoutput.append(results)
      beta_value += 0.1
    alfa_value += 0.1
  """
  
  """
  # the alpha and beta values are obtained with LHS in R"
  results = ["alpha", "beta", "cost", "comp. time", "epoch"]
  fileoutput = []
  fileoutput.append(results)
  #alfa_values = [0.33083770, 0.81286553, 0.28994727, 0.62146038, 0.52186326, 0.17224262, 0.02020398, 0.78311170, 0.90367650, 0.49015838]
  #beta_values = [0.6197808, 0.89753027, 0.11981711, 0.72693961, 0.40612212, 0.94802543, 0.57793579, 0.23444987, 0.39597191, 0.09485114]
  alfa_values = [0.52966316, 0.4145423, 0.19379634, 0.87264983, 0.97362117, 0.06562221, 0.65973607, 0.2151281, 0.461971, 0.38040526, 0.77591784, 0.12027035, 0.82232354, 0.32580516, 0.58453551, 0.73574494, 0.92142149, 0.26890176, 0.61401872, 0.0111655]
  beta_values = [0.58285168, 0.35864099, 0.85454185, 0.13890171, 0.49436501, 0.23125492, 0.72400869, 0.0334176, 0.98079264, 0.82039207, 0.52108801, 0.40230925, 0.33304859, 0.15509154, 0.05296365, 0.92091665, 0.78477693, 0.62063391, 0.25896396, 0.65795874]
  for i in range(20):
    alfa_value = alfa_values[i]
    beta_value = beta_values[i]
    cost_sum = 0.0
    runs = 0
    print("alfa: ", alfa_value, "beta: ", beta_value)
    for i in range(20):
      results = []
      runs += 1
      pso = Solver(graph, iterations=1000, maxEpochs=50, size_population=15, beta=beta_value, alfa=alfa_value)
      start_time = datetime.now()
      pso.run()
      #cost_sum = cost_sum + pso.getGBest().getCostPBest()
      results.append(alfa_value)
      results.append(beta_value)
      results.append(pso.getGBest().getCostPBest())
      epoch = pso.getEpoch()
      dt = datetime.now() - start_time
      ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
      results.append(ms)
      results.append(epoch)
      fileoutput.append(results)
    #average_cost = cost_sum / runs
    #results.append(alfa_value)
    #results.append(beta_value)
    #results.append(average_cost)
  """  
   

  # pso-results.csv
  
  
  csvFile = open('micro-pso-continuo.csv', 'w', newline='')  
  with csvFile: 
    writer = csv.writer(csvFile)
    writer.writerows(fileoutput)
  csvFile.close()
  

  # shows the global best particle
  ##bestPath = pso.getGBest().getPBest()
  ##print('gbest: %s | cost: %d\n' % (bestPath, pso.getGBest().getCostPBest()))
  ##plotPoints(x, y)
  ##plotTSP(bestPath, x, y)
  
  '''
  # random graph
  print('Random graph...')
  random_graph = CompleteGraph(amount_vertices=20)
  random_graph.generates()
  pso_random_graph = PSO(random_graph, iterations=10000, size_population=10, beta=1, alfa=1)
  pso_random_graph.run()
  print('gbest: %s | cost: %d\n' % (pso_random_graph.getGBest().getPBest(), 
          pso_random_graph.getGBest().getCostPBest()))
  '''
