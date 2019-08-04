from deap import creator
from deap import base
from deap import tools
from deap import algorithms

import array
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx
import random

numCities = 5

# generate the locations of cities
#random.seed(169)
x = np.random.rand(numCities)
y = np.random.rand(numCities)

#plt.plot(x,y)
#plt.show()

creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", array.array, typecode = 'i', fitness=creator.FitnessMin) #variations of this type are possible by inheriting from array.array or numpy.ndarray

toolbox = base.Toolbox()

# Generate individauls
toolbox.register("indices", random.sample, range(numCities), numCities) #create a index from 0 to numCities -1
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices) #create individual with random index
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#mate, mutate, select
toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.05)
toolbox.register('select', tools.selTournament, tournsize = 3)

#register history
history = tools.History()
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

#evaluating function
def evalTSP(individual):
    diffx = np.diff(x[individual]) #shuffle the order of distance x based on individual index
    diffy = np.diff(y[individual]) #shuffle the order of distance y based on individual index
    distance = np.sum(diffx**2 + diffy**2)


    return (distance,) # return a turple

toolbox.register('evaluate', evalTSP)

def main():
    pop = toolbox.population(n = 30)
    history.update(pop)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 10, halloffame = hof)

    #ind = hof[0]
    #plt.plot(x[ind], y[ind])
    #plt.show()

    #graph = networkx.DiGraph(history.genealogy_tree)
    #graph = graph.reverse()     # Make the grah top-down
    #print(graph.edges)
    #colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    #networkx.draw(graph, node_color=colors)
    #plt.show()

    #print(history.genealogy_tree)
    plt.ion()
    fig = plt.figure()
    graph = fig.add_subplot()
    for key, value in history.genealogy_history.items():
       graph.clear()
       graph.plot(x[value], y[value], marker = 'o')
       plt.pause(0.01)
       plt.draw()
    return pop, hof

if __name__ == "__main__":
   pop, hof =  main()

