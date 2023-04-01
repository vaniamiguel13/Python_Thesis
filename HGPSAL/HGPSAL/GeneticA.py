import numpy as np
import math

import numpy as np

import numpy as np

def Bounds(X, L, U):
    for _ in range(len(X)):
        if X[_] < L[_]:
            X[_] = L[_]
        if X[_] > U[_]:
            X[_] = U[_]
    return X

import numpy as np

import numpy as np

#
# def Bounds(X, L, U):
#     # check if X is a list of arrays
#     if isinstance(X[0], np.ndarray):
#         for i in range(len(X)):
#             X[i] = np.clip(X[i], L[i], U[i])
#     else:
#         for i in range(len(X)):
#             if X[i] < L[i]:
#                 X[i] = L[i]
#             if X[i] > U[i]:
#                 X[i] = U[i]
#     return X


def ObjEval(P, x, *args):
    P.Stats.ObjFunCounter += 1
    return P, P.ObjFunction(x, *args)


class Popul:
    def __init__(self, x, f):
        self.x = x
        self.f = f


class ProblemStatistics:
    def __init__(self):
        self.Evaluations = 0
        self.Iterations = 0
        self.StopFlag = ''
        self.Message = ''
        self.ObjFunCounter = 0
        self.GenCounter = 0
        self.Best = []
        self.Worst = []
        self.Mean = []
        self.Std = []


class Problem:
    def __init__(self, Variables, ObjFunction, LB, UB):
        self.Variables = Variables
        self.ObjFunction = ObjFunction
        self.LB = LB
        self.UB = UB
        self.Verbose = False
        self.Tolerance = 1.0e-6
        self.GenTest = 0.01
        self.Stats = ProblemStatistics()


# def InitPopulation(Problem, InitialPopulation, Size, *args):
#     Popl = Popul(np.zeros((Size, Problem.Variables)), np.zeros(Size))
#     for i in range(len(InitialPopulation)):
#
#         Popl.x[i, :] = Bounds(InitialPopulation, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])
#         Problem, Popl.f[i] = ObjEval(Problem, Popl.x[i, :], *args)
#
#     for i in range(len(InitialPopulation), Size):
#         Popl.x[i, :] = np.array(Problem.LB[:Problem.Variables]) + (
#                 np.array(Problem.UB[:Problem.Variables]) - np.array(Problem.LB[:Problem.Variables])) * np.random.rand(
#             Problem.Variables)
#         Problem, Popl.f[i] = ObjEval(Problem, Popl.x[i, :], *args)
#
#     Popl.f = Popl.f.reshape(-1, 1)
#
#     return Problem, Popl

def InitPopulation(Problem, InitialPopulation, Size, *args):
    Population = Popul
    # Population = type('Population', (), {'x': None, 'f': None})()  # create a Population object
    Population.x = np.zeros((Size, Problem.Variables))
    Population.f = np.zeros(Size, )


    # Check for size
    if len(InitialPopulation) > Size:
        # User provided an initial population greater than the parent population size
        raise ValueError('Initial population size must be smaller than or equal to PopSize.')
    # Copy the initial population for the population and initialize them
    for i in range(len(InitialPopulation)):
        x = InitialPopulation
        Population.x[i, :] = Bounds(x, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])
        Problem, Population.f[i] = ObjEval(Problem, Population.x[i, :], *args)

    # Randomly generate the remaining population
    for i in range(len(InitialPopulation), Size):
        Population.x[i, :] = np.array(Problem.LB[:Problem.Variables]) + \
                             (np.array(Problem.UB[:Problem.Variables]) - np.array(
                                 Problem.LB[:Problem.Variables])) * np.random.rand(Problem.Variables)
        Problem, Population.f[i] = ObjEval(Problem, Population.x[i, :], *args)

    Population.f = Population.f.reshape(-1, 1)

    return Problem, Population


def tournament_selection(chromosomes, pool_size, tour_size):
    pop = chromosomes.x.shape[0]
    P = Popul(np.zeros((pool_size, chromosomes.x.shape[1])), np.zeros(pool_size))
    for i in range(pool_size):
        candidate = np.zeros(tour_size, dtype=int)
        fitness = np.zeros(tour_size)
        for j in range(tour_size):
            candidate[j] = np.random.randint(pop)
            while j > 0 and np.isin(candidate[j], candidate[:j]):
                candidate[j] = np.random.randint(pop)
            fitness[j] = chromosomes.f[candidate[j]]
        min_candidate = np.argmin(fitness)
        P.x[i, :] = chromosomes.x[candidate[min_candidate], :]
        P.f[i] = chromosomes.f[candidate[min_candidate]]
    P.f = P.f.reshape(-1, 1)
    return P


def genetic_operator(Problem, parent_chromosome, pc, pm, mu, mum):
    N, V = parent_chromosome.x.shape

    child = np.zeros((N, V))

    p = 0
    while p < N:
        # Simulated Binary Crossover (SBX) applied with probability pc
        parent_1 = np.random.randint(N)
        parent_2 = np.random.randint(N)
        while parent_1 == parent_2:
            parent_2 = np.random.randint(N)

        parent_1 = parent_chromosome.x[parent_1, :]

        parent_2 = parent_chromosome.x[parent_2, :]

        if np.random.rand() < pc:
            bq = np.zeros(V)
            child_1 = []
            child_2 = []
            for j in range(V):
                u = np.random.rand()
                if u <= 0.5:
                    bq[j] = (2 * u) ** (1 / (mu + 1))
                else:
                    bq[j] = (1 / (2 * (1 - u))) ** (1 / (mu + 1))

                child_1.append(0.5 * (((1 + bq[j]) * parent_1[j]) + (1 - bq[j]) * parent_2[j]))
                child_2.append(0.5 * (((1 - bq[j]) * parent_1[j]) + (1 + bq[j]) * parent_2[j]))

            child_1 = Bounds(child_1, Problem.LB[0:Problem.Variables], Problem.UB[0:Problem.Variables])
            child_2 = Bounds(child_2, Problem.LB[0:Problem.Variables], Problem.UB[0:Problem.Variables])

        else:
            child_1 = parent_1
            child_2 = parent_2

        # Polynomial mutation applied with probability pm
        for j in range(V):
            if np.random.rand() < pm:
                r = np.random.rand()
                if r < 0.5:
                    delta = (2 * r) ** (1 / (mum + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))

                child_1[j] = child_1[j] + (Problem.UB[j] - Problem.LB[j]) * delta

            child_1 = Bounds(child_1, Problem.LB[0:Problem.Variables], Problem.UB[0:Problem.Variables])

        for j in range(V):
            if np.random.rand() < pm:
                r = np.random.rand()
                if r < 0.5:
                    delta = (2 * r) ** (1 / (mum + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))

                child_2[j] = child_2[j] + (Problem.UB[j] - Problem.LB[j]) * delta

            child_2 = Bounds(child_2, Problem.LB[0:Problem.Variables], Problem.UB[0:Problem.Variables])

        child[p] = child_1[0:V]
        child[p + 1] = child_2[0:V]
        p = p + 2

    return child


def rGA(Problem, InitialPopulation=None, Options=None, *args):
    DefaultOpt = {'MaxObj': 2000, 'MaxGen': 2000, 'PopSize': 40, 'EliteProp': 0.1,
                  'TourSize': 2, 'Pcross': 0.9, 'Icross': 20, 'Pmut': 0.1, 'Imut': 20,
                  'CPTolerance': 1.0e-6, 'CPGenTest': 0.01, 'Verbosity': False}

    MaxGenerations = Options['MaxGen'] if Options and 'MaxGen' in Options else DefaultOpt['MaxGen']
    MaxEvals = Options['MaxObj'] if Options and 'MaxObj' in Options else DefaultOpt['MaxObj']
    Pop = Options['PopSize'] if Options and 'PopSize' in Options else DefaultOpt['PopSize']
    Elite = Options['EliteProp'] if Options and 'EliteProp' in Options else DefaultOpt['EliteProp']
    Tour = Options['TourSize'] if Options and 'TourSize' in Options else DefaultOpt['TourSize']
    Pc = Options['Pcross'] if Options and 'Pcross' in Options else DefaultOpt['Pcross']
    Ic = Options['Icross'] if Options and 'Icross' in Options else DefaultOpt['Icross']
    Pm = Options['Pmut'] if Options and 'Pmut' in Options else DefaultOpt['Pmut']
    Im = Options['Imut'] if Options and 'Imut' in Options else DefaultOpt['Imut']

    Problem.Stats.ObjFunCounter = 0
    if InitialPopulation is None: InitialPopulation = []

    # Generate initial population. Include initial population provided by user if available
    Population = InitPopulation(Problem, InitialPopulation, Pop, *args)[1]
    temp = np.column_stack((Population.x, Population.f))
    temp = temp[temp[:, -1].argsort()]
    Population.x = temp[:, :-1]
    Population.f = temp[:, -1]

    Problem.Stats.GenCounter = 0

    # Initialize statistics structures. Keep track of population statistics
    Problem.Stats.Best.append(Population.f[0])
    Problem.Stats.Worst.append(Population.f[-1])
    Problem.Stats.Mean.append(np.mean(Population.f))
    Problem.Stats.Std.append(np.std(Population.f))

    # while Problem.Stats.GenCounter < MaxGenerations and Problem.Stats.ObjFunCounter < MaxEvals:
    #     # Stop if the improvement is inferior to the Tolerance in the last generations
    #     if Problem.Stats.GenCounter > 0 and (
    #             not Problem.Stats.GenCounter % math.ceil(Problem.GenTest * MaxGenerations)) and abs(
    #         Problem.Stats.Best[
    #             Problem.Stats.GenCounter + 1] - Problem.Stats.Best[Problem.Stats.GenCounter + 1 -
    #                                                                math.ceil(
    #                                                                    Problem.GenTest * MaxGenerations)]) < Problem.Tolerance:
    #         print(
    #             'Stopping due to objective function improvement inferior to CPTolerance in the last CPGenTest '
    #             'generations')
    #         break
    while Problem.Stats.GenCounter < MaxGenerations and Problem.Stats.ObjFunCounter < MaxEvals:
        # Stop if the improvement is inferior to the Tolerance in the last generations
        if Problem.Stats.GenCounter > 0 and (
                not Problem.Stats.GenCounter % math.ceil(Problem.GenTest * MaxGenerations)) and len(
            Problem.Stats.Best) > Problem.Stats.GenCounter + 1 and abs(
            Problem.Stats.Best[
                Problem.Stats.GenCounter + 1] - Problem.Stats.Best[Problem.Stats.GenCounter + 1 -
                                                                   math.ceil(
                                                                       Problem.GenTest * MaxGenerations)]) < Problem.Tolerance:
            print(
                'Stopping due to objective function improvement inferior to CPTolerance in the last CPGenTest '
                'generations')
            break

        # Increment generation counter.
        Problem.Stats.GenCounter += 1
        elitesize = round(Pop * Elite)
        pool = Pop - elitesize
        parent_chromosome = tournament_selection(Population, pool, Tour)

        # Perform Crossover and Mutation operator
        # Simulated Binary Crossover (SBX) and
        # Polynomial mutation is used. In general, crossover probability is pc = 0.9 and mutation
        # probability is pm = 1/n, where n is the number of decision variables.
        # Typical values for the distribution indices for crossover and mutation operators are mu = 20
        # and mum = 20 respectively.
        offspring_chromosome = genetic_operator(Problem, parent_chromosome, Pc, Pm, Ic, Im)

        # The offspring population replaces the worst chromosomes of the
        # population, i.e., the elite is maintained
        Population.x[elitesize:Pop, :] = offspring_chromosome[:pool, :]

        # Evaluate the objective function
        for i in range(elitesize, Pop):
            Problem, Population.f[i] = ObjEval(Problem, Population.x[i, :], *args)

        # The new population is sorted again
        temp = np.concatenate((Population.x, Population.f[:, np.newaxis]), axis=1)
        temp = temp[temp[:, -1].argsort()]
        Population.x = temp[:, :-1]
        Population.f = temp[:, -1]

        ## Statistics

        Problem.Stats.Best.append(Population.f[0])
        Problem.Stats.Worst.append(Population.f[-1])
        Problem.Stats.Mean.append(np.mean(Population.f))
        Problem.Stats.Std.append(np.std(Population.f))

    BestChrom = Population.x[0]
    BestChromObj = Population.f[0]
    RunData = Problem.Stats

    return BestChrom, BestChromObj, RunData


class InitialGuess:
    def __init__(self, x):
        self.x = x


Variables = 2


def Rastrigin(x):
    f = 20 + x[0] ** 2 + x[1] ** 2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
    return f


LB = [-5, -5]
UB = [5, 5]

myProblem = Problem(Variables, Rastrigin, LB, UB)
# InitialGuess1 = InitialGuess(np.array([0, 0]))
# InitialGuess2 = InitialGuess(np.array([1, 1]))



# R=rGA(myProblem, InitialPopulation=InitialPopulation, Options=None)
# print(R[2].ObjFunCounter)

# Problem, Population = InitPopulation(myProblem, InitialPopulation, PopSize)

# %%
# def test_obj(x):
#     return np.sum(x[0] ** 2 + x[1])
#
#
#
#
# Problem = Problem(2, test_obj, np.array([-5.12, -5.12]), np.array([5.12, 5.12]))
#
# InitialPopulation = [InitialGuess(np.array([0.5, 0.5])), InitialGuess(np.array([-0.5, -0.5]))]
# Size = 20
# Problem, Population = InitPopulation(Problem, InitialPopulation, Size)
# print(Population.x)
