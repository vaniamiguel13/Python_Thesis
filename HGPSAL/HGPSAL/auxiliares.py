import numpy as np


def InitPopulation(Problem, InitialPopulation, Size, *args):
    Population = {}
    # Check if initial population is provided and in proper format
    if InitialPopulation and not isinstance(InitialPopulation, dict):
        raise ValueError('Initial population must be defined in a dictionary.')
    else:
        # Check for size
        if len(InitialPopulation) > Size:
            # User provided an initial population greater then the parent population size
            raise ValueError('Initial population size must be inferior to PopSize.')
        # Copy the initial population for the population and initialize them
        for i in range(len(InitialPopulation)):
            Population['x'][i, :] = bounds(InitialPopulation[i]['x'], Problem['LB'][:Problem['Variables']],
                                           Problem['UB'][:Problem['Variables']])
            Problem, Population['f'][i] = ObjEval(Problem, Population['x'][i, :], *args)
        # Randomly generate the remaining population
        for i in range(len(InitialPopulation), Size):
            Population['x'][i, :] = Problem['LB'][:Problem['Variables']] + (
                    Problem['UB'][:Problem['Variables']] - Problem['LB'][:Problem['Variables']]) * np.random.rand(1,Problem['Variables'])


            Problem, Population['f'][i] = ObjEval(Problem, Population['x'][i, :], *args)
    Population['f'] = Population['f'].T
    return Problem, Population


def tournament_selection(chromosomes, pool_size, tour_size):
    pop = chromosomes['x'].shape[0]
    P = {'x': np.empty((pool_size, chromosomes['x'].shape[1])),
         'f': np.empty(pool_size)}
    for i in range(pool_size):
        candidate = np.zeros(tour_size, dtype=int)
        fitness = np.zeros(tour_size)
        for j in range(tour_size):
            candidate[j] = np.random.randint(pop) + 1
            if j > 0:
                while np.isin(candidate[j], candidate[:j]):
                    candidate[j] = np.random.randint(pop) + 1
            fitness[j] = chromosomes['f'][candidate[j] - 1]
        min_candidate = np.argwhere(fitness == np.min(fitness)).flatten()
        P['x'][i] = chromosomes['x'][candidate[min_candidate[0]] - 1]
        P['f'][i] = chromosomes['f'][candidate[min_candidate[0]] - 1]
    P['f'] = P['f'].reshape(-1, 1)
    return P


def genetic_operator(Problem, parent_chromosome, pc, pm, mu, mum):
    N, V = parent_chromosome.shape

    p = 0
    child = np.zeros((N, V))
    while p < N:
        # SBX (Simulated Binary Crossover) applied with probability pc
        parent_1_idx = np.random.randint(N)
        parent_2_idx = np.random.randint(N)
        while parent_1_idx == parent_2_idx:
            parent_2_idx = np.random.randint(N)
        parent_1 = parent_chromosome[parent_1_idx]
        parent_2 = parent_chromosome[parent_2_idx]
        if np.random.rand() < pc:
            u = np.random.rand(V)
            bq = np.zeros(V)
            for j in range(V):
                if u[j] <= 0.5:
                    bq[j] = (2 * u[j]) ** (1 / (mu + 1))
                else:
                    bq[j] = (1 / (2 * (1 - u[j]))) ** (1 / (mu + 1))
                child_1[j] = 0.5 * (((1 + bq[j]) * parent_1[j]) + (1 - bq[j]) * parent_2[j])
                child_2[j] = 0.5 * (((1 - bq[j]) * parent_1[j]) + (1 + bq[j]) * parent_2[j])
            child_1 = bounds(child_1, Problem.LB[:V], Problem.UB[:V])
            child_2 = bounds(child_2, Problem.LB[:V], Problem.UB[:V])
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
            child_1 = bounds(child_1, Problem.LB[:V], Problem.UB[:V])

        for j in range(V):
            if np.random.rand() < pm:
                r = np.random.rand()
                if r < 0.5:
                    delta = (2 * r) ** (1 / (mum + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))
                child_2[j] = child_2[j] + (Problem.UB[j] - Problem.LB[j]) * delta
            child_2 = bounds(child_2, Problem.LB[:V], Problem.UB[:V])

        child[p] = child_1[:V]
        child[p + 1] = child_2[:V]
        p += 2

    return child


def bounds(X, L, U):
    for i in range(len(X)):
        if X[i] < L[i]:
            X[i] = L[i]
        if X[i] > U[i]:
            X[i] = U[i]
    return X


def ObjEval(Problem, x, *varargin):
    try:
        ObjValue = Problem["ObjFunction"](x, *varargin)
        # update counter
        Problem["Stats"]["ObjFunCounter"] += 1
    except Exception as e:
        error_message = f"Cannot continue because user supplied objective function failed with the following error: \n{e}"
        raise ValueError("rGA:ObjectiveError", error_message)
    return Problem, ObjValue


