import numpy as np
import math


class Population:
    def __init__(self, x, f):
        self.x = x
        self.f = f


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


class ProblemStatistics:
    def __init__(self):
        self.ObjFunCounter = 0
        self.GenCounter = 0
        self.Best = []
        self.Worst = []
        self.Mean = []
        self.Std = []


class Chromosome:
    def __init__(self, x, f):
        self.x = x
        self.f = f


def Bounds(X, L, U):
    for i in range(len(X)):
        if X[i] < L[i]:
            X[i] = L[i]
        if X[i] > U[i]:
            X[i] = U[i]
    return X


def InitPopulation(Problem, InitialPopulation, Size, *args):
    Population = [Chromosome([], 0.0) for _ in range(Size)]

    if InitialPopulation is not None:
        if len(InitialPopulation) > Size:
            raise ValueError('Initial population size must be inferior to PopSize.')

        # Copy the initial population for the population and initialize them
        for i in range(len(InitialPopulation)):
            x = Bounds(InitialPopulation[i].x, Problem.LB[0:Problem.Variables], Problem.UB[0:Problem.Variables])
            f = Problem.ObjFunction(x, *args)
            Population[i] = Chromosome(x, f)
            Problem.Stats.ObjFunCounter += 1


    Population.sort(key=lambda chrom: chrom.f)
    Problem.Stats.GenCounter += 1
    Problem.Stats.Best.append(Population[0].f)
    Problem.Stats.Worst.append(Population[-1].f)
    Problem.Stats.Mean.append(np.mean([chrom.f for chrom in Population]))
    Problem.Stats.Std.append(np.std([chrom.f for chrom in Population]))

    return Problem, Population


def ObjEval(Problem, x, *varargin):
    try:
        ObjValue = Problem.ObjFunction(x, *varargin)
        # update counter
        Problem.Stats.ObjFunCounter = Problem.Stats.ObjFunCounter + 1
    except:
        raise Exception(
            'Cannot continue because user supplied objective function failed with the following error:\n%s' % lasterr)
    return Problem, ObjValue


def genetic_operator(Problem, parent_chromosome, pc, pm, mu, mum):
    N, V = parent_chromosome.x.shape

    p = 0
    child = np.zeros((N, V))

    while p < N:
        # SBX (Simulated Binary Crossover) applied with probability pc
        parent_1_idx = np.random.randint(N)
        parent_2_idx = np.random.randint(N)

        while parent_1_idx == parent_2_idx:
            parent_2_idx = np.random.randint(N)

        parent_1 = parent_chromosome.x[parent_1_idx]
        parent_2 = parent_chromosome.x[parent_2_idx]

        if np.random.rand() < pc:
            u = np.random.rand(V)
            bq = np.zeros(V)
            child_1=[]
            child_2=[]
            for j in range(V):
                if u[j] <= 0.5:
                    bq[j] = (2 * u[j]) ** (1 / (mu + 1))
                else:
                    bq[j] = (1 / (2 * (1 - u[j]))) ** (1 / (mu + 1))

                child_1[j] = 0.5 * (((1 + bq[j]) * parent_1[j]) + (1 - bq[j]) * parent_2[j])
                child_2[j] = 0.5 * (((1 - bq[j]) * parent_1[j]) + (1 + bq[j]) * parent_2[j])

            child_1 = Bounds(child_1, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])
            child_2 = Bounds(child_2, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])
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

            child_1 = Bounds(child_1, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])

        for j in range(V):
            if np.random.rand() < pm:
                r = np.random.rand()
                if r < 0.5:
                    delta = (2 * r) ** (1 / (mum + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))

                child_2[j] = child_2[j] + (Problem.UB[j] - Problem.LB[j]) * delta

            child_2 = Bounds(child_2, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])

        child[p] = child_1[:V]
        child[p + 1] = child_2[:V]

        p += 2

    P = Chromosome(child, np.zeros(N))

    return P

# def genetic_operator(Problem, parent_chromosome, pc, pm, mu, mum):
#     N, V = parent_chromosome.x.shape
#     child = np.zeros((N, V))
#
#     for p in range(0, N, 2):
#         parent_1_idx, parent_2_idx = np.random.choice(N, size=2, replace=False)
#         parent_1, parent_2 = parent_chromosome.x[parent_1_idx], parent_chromosome.x[parent_2_idx]
#         u = np.random.rand(V)
#         bq = np.where(u <= 0.5, (2 * u) ** (1 / (mu + 1)), (1 / (2 * (1 - u))) ** (1 / (mu + 1)))
#         child_1 = 0.5 * (((1 + bq) * parent_1) + (1 - bq) * parent_2)
#         child_2 = 0.5 * (((1 - bq) * parent_1) + (1 + bq) * parent_2)
#
#         child_1 = np.where(np.random.rand(V) < pm, child_1 + (Problem.UB[:V] - Problem.LB[:V]) * ((2 * np.random.rand() - 1) * (2 * np.random.rand()) ** (1 / (mum + 1))), child_1)
#         child_2 = np.where(np.random.rand(V) < pm, child_2 + (Problem.UB[:V] - Problem.LB[:V]) * ((2 * np.random.rand() - 1) * (2 * np.random.rand()) ** (1 / (mum + 1))), child_2)
#
#         child_1 = Bounds(child_1, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])
#         child_2 = Bounds(child_2, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])
#
#         child[p] = child_1[:V]
#         child[p + 1] = child_2[:V]
#
#     return Chromosome(child, np.zeros(N))

def tournament_selection(chromosomes, pool_size, tour_size):
    pop = chromosomes.x.shape[0]
    P = Chromosome(np.empty((pool_size, chromosomes.x.shape[1])), np.empty(pool_size))

    for i in range(pool_size):
        candidate = []
        fitness = []
        for j in range(tour_size):
            c = np.random.randint(pop) + 1
            while c in candidate:
                c = np.random.randint(pop) + 1
            candidate.append(c)
            fitness.append(chromosomes.f[c - 1])

        min_candidate = np.argmin(fitness)
        P.x[i, :] = chromosomes.x[candidate[min_candidate] - 1, :]
        P.f[i] = chromosomes.f[candidate[min_candidate] - 1]

    P.f = P.f.reshape(-1, 1)
    return P


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

    # Number of objective function calls. This is the number of calls to Problem.ObjFunction.
    Problem.Stats.ObjFunCounter = 0

    # Generate initial population. Include initial population provided by user if available
    Population = InitPopulation(Problem, InitialPopulation, Pop, *args)
    print(Population)
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

    while Problem.Stats.GenCounter < MaxGenerations and Problem.Stats.ObjFunCounter < MaxEvals:
        # Stop if the improvement is inferior to the Tolerance in the last generations
        if Problem.Stats.GenCounter > 0 and (
        not Problem.Stats.GenCounter % math.ceil(Problem.GenTest * MaxGenerations)) and abs(
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
        Population.x[elitesize:Pop, :] = offspring_chromosome.x[:pool, :]

        # Evaluate the objective function
        for i in range(elitesize, Pop):
            Problem, Population.f[i] = ObjEval(Problem, Population.x[i, :], *varargin)

        # The new population is sorted again
        temp = np.concatenate((Population.x, Population.f[:, np.newaxis]), axis=1)
        temp = temp[temp[:, -1].argsort()]
        Population.x = temp[:, :-1]
        Population.f = temp[:, -1]

        ## Statistics
        Problem.Stats.Best[Problem.Stats.GenCounter + 1] = Population.f[0]
        Problem.Stats.Worst[Problem.Stats.GenCounter + 1] = Population.f[-1]
        Problem.Stats.Mean[Problem.Stats.GenCounter + 1] = np.mean(Population.f)
        Problem.Stats.Std[Problem.Stats.GenCounter + 1] = np.std(Population.f)

        BestChrom = [Population.x[0]]
        BestChromObj = Population.f[0]
        RunData = Problem.Stats

        return BestChrom, BestChromObj, RunData

def sphere_function(x):
    return np.sum(x[0] ** 2+ x[1])

problem = Problem(Variables=2, ObjFunction=sphere_function, LB=[-5.12, -5.12], UB=[5.12, 5.12])
print(rGA(problem))