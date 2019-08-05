# https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
import matplotlib.pyplot as plt
import networkx

from deap import creator
from deap import base
from deap import tools
from deap import algorithms

import numpy as np

def evaluate(individual):
    y = individual[0]*4 + individual[1]*-2 + individual[2]*3.5 + individual[3]*5 + individual[4]*-11 + individual[5]*-4.7
    return (y,)

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Generate Individuals
numOfWeights = 6
toolbox.register("weight", np.random.uniform, -4, 4.0001, numOfWeights)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.weight)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#ind1 = toolbox.individual()
#print(ind1)

toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.05)
toolbox.register('select', tools.selTournament, tournsize = 3)

toolbox.register('evaluate', evaluate)

#register history
history = tools.History()
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

def main():
    pop = toolbox.population(n = 100)
    history.update(pop)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    pop, logbook = algorithms.eaSimple(pop, toolbox, 0.8, 0.3, 1000, halloffame = hof, stats = stats)

    print(hof[0])

    #graph = networkx.DiGraph(history.genealogy_tree)
    #graph = graph.reverse()     # Make the grah top-down
    #colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    #networkx.draw(graph, node_color=colors, pos = networkx.spring_layout(graph))
    #plt.show()

    return pop, logbook, hof

if __name__ == "__main__":
    pop, logbook, hof = main()
    gen, avg = logbook.select("gen", "avg")
    plt.plot(gen, avg, label="average")

    plt.show()
