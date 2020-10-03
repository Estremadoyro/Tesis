# -*- coding: utf-8 -*-
"""Custom Individual - GA

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K4FwUFO-rLofQqLFikH_snvyuU-EjEQb
"""

!pip install deap

import random
import itertools
from deap import base
from deap import creator
from deap import tools
from itertools import chain

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

#Create individual
class Agente(object):
    def __init__(self, num_input, max_num_hidden):
        self.num_input = num_input
        self.max_num_hidden = max_num_hidden

    def gen_hidden(self):
        return random.randint(1, self.max_num_hidden)

    def gen_weights(self, num_hidden, type_):
        if type_ == "input":
            num_weights = self.num_input * num_hidden
        elif type_ == "output":
            num_weights = num_hidden
        #result = [random.random() for i in range(num_weights)]
        result = [random.randint(0,1) for i in range(num_weights)]
        return result

    def get_random_agent(self):
        num_hidden = self.gen_hidden()
        hidden_weights = self.gen_weights(num_hidden, type_="input")
        output_weights = self.gen_weights(num_hidden, type_="output")
        #print([[num_hidden], hidden_weights, output_weights])
        return [[num_hidden], hidden_weights, output_weights]
#num_input, max_num_hidden
def newIndividual(num_input, max_num_hidden): 
    agent = Agente(num_input, max_num_hidden).get_random_agent()
    #print(agent)
    agent = creator.Individual(agent)
    return agent
#print(newIndividual())

toolbox = base.Toolbox()
#Asignar class type deap.creator.Individual a nuestro individuo
toolbox.register("individual", newIndividual, 8, 15)
print(type(toolbox.individual()))
toolbox.individual()

#Crear population de tipo list con invdividuals de tipo deap.creator.Individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

population = toolbox.population(n=20)

def nested_sum(L):
    total = 0  
    for i in L:
        if isinstance(i, list):  
            total += nested_sum(i)
        else:
            total += i
    return total

#Funcion de evaluacion
def evalOneMax(individual):
    return nested_sum(individual),
    #return sum(individual),

#Operadores Geneticos
toolbox.register("evaluate", evalOneMax)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

#population = toolbox.population(n=4)
print(population)
fitnesses = list(map(toolbox.evaluate, population))
print(fitnesses)

def flatlist(li):
  flat_list = list(chain.from_iterable(li))
  flat_list = creator.Individual(flat_list)
  return flat_list
print("Population {}".format(population))
newPopulation = []
def createNewPopulation():
  for x in population:
    newPopulation.append(flatlist(x))
  print("New Population {}".format(newPopulation))
  return newPopulation
createNewPopulation()

print('Population type: {}'.format(type(newPopulation)))
print('Individual type: {}'.format(type(newPopulation[0])))

for ind, fit in zip(newPopulation, fitnesses):
      print(ind)
      print(fit)
      ind.fitness.values = fit

# CXPB = % probabilidad de Crossover
# MUTPB = % probabilidad de Mutation
CXPB, MUTPB = 0.5, 0.2

print("New population")
print(newPopulation)
fits = [ind.fitness.values[0] for ind in newPopulation]
print(fits)

print('Population type: {}'.format(type(newPopulation)))
print('Individual type: {}'.format(type(newPopulation[0])))

generation = 0 
maxGenerations = 2000
individuoAceptable = 600

while max(fits) < individuoAceptable and generation < maxGenerations: 
  generation = generation + 1 
  print("-- Generation %i --" % generation)
  # Seleccionar los cromosomas de la siguiente generacion
  offspring = toolbox.select(newPopulation, len(newPopulation))
  # Clonar los cromosomas seleccionados
  offspring = list(map(toolbox.clone, offspring))
  for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < CXPB:
      #Crossover
      toolbox.mate(child1, child2)
      del child1.fitness.values
      del child2.fitness.values

    for mutant in offspring:
      if random.random() < MUTPB:
        toolbox.mutate(mutant)
        del mutant.fitness.values
        # Evaluar los individuos con .fitness.values nulo
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit

    print("Evaluated %i individuals" % len(invalid_ind))
    #Reemplazar poblacion antigua por nueva con los nuevos fitness 
    #de los individuos
    newPopulation[:] = offspring
    # Almacenar todos los costos obtenidos en una lista para poder ser mostrados en log.
    fits = [ind.fitness.values[0] for ind in newPopulation]
        
    length = len(newPopulation)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
        
    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

  print("-- Fin de proceso de evolucion --")

print(newPopulation)

best_ind = tools.selBest(newPopulation, 1)[0]
print("Mejor individuo es %s, %s" % (best_ind, best_ind.fitness.values))