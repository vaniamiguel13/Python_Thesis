import numpy as np
import math
from HGPSAL.HGPSAL.AUX_Class.Population_C import Popul


def InitPopulation(Problem, InitialPopulation, Size, Conflag, CTol, CeqTol, NormType, NormCons, *args):
    x = np.zeros((Size, Problem.Variables))
    f = np.zeros((Size, 2))
    Population = Popul(x, f)
    Population.c = [0] * Size
    Population.ceq = [0] * Size
    Population.Feasible = np.zeros((Size,))
    Population.Rank = np.zeros((Size,), dtype=int)
    Population.Fitness = np.zeros((Size,))

    if InitialPopulation and not isinstance(InitialPopulation, list):
        raise ValueError('Initial population must be defined in a list.')
    elif len(InitialPopulation) > Size:
        raise ValueError('Initial population size must be inferior to PopSize.')
    else:
        for i in range(len(InitialPopulation)):
            x = InitialPopulation
            Population.x[i, :] = bounds(x, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])
            Problem, Population.f[i] = ObjEval(Problem, Population.x[i, :], *args)

            if Conflag:
                C_eval = ConEval(Problem, Population.x[i, :], *args)
                Problem = C_eval[0]
                Population.c[i] = C_eval[1]
                Population.ceq[i] = C_eval[2]

            else:
                Population.c = Population.c
                Population.ceq = Population.ceq
                Population.Feasible[i] = 1

            Population.Rank[i] = 0

    for i in range(len(InitialPopulation), Size):
        Population.x[i, :] = np.array(Problem.LB[:Problem.Variables]) + \
                             (np.array(Problem.UB[:Problem.Variables]) - np.array(
                                 Problem.LB[:Problem.Variables])) * np.random.rand(Problem.Variables)
        Problem, Population.f[i, :] = ObjEval(Problem, Population.x[i, :], *args)

        if Conflag:
            C_eval = ConEval(Problem, Population.x[i, :], *args)
            Problem = C_eval[0]
            Population.c[i] = C_eval[1]
            Population.ceq[i] = C_eval[2]

        else:
            Population.c = np.append(Population.c, 0)
            Population.ceq = np.append(Population.ceq, 0)
            Population.Feasible[i] = 1
        Population.Rank[i] = 0
    Population.ceq = np.array(Population.ceq)

    if Conflag:
        for i in range(Size):
            if NormCons == 1:
                maxc = np.maximum(0, Population.c).min(axis=0)

                if isinstance(maxc, int):
                    if maxc == 0:
                        maxc = np.array([1])
                else:
                    for j in range(len(maxc)):
                        if maxc[j] == 0:
                            maxc[j] = 1

                maxceq = Population.ceq.min(axis=0)

                if type(maxceq) == list or type(maxceq) == np.ndarray:

                    for j in range(len(maxceq)):
                        if maxceq[j] == 0:
                            maxceq[j] = 1
                else:
                    if maxceq == 0:
                        maxceq = np.array([1])

                c_norm = (np.linalg.norm(np.maximum([0], Population.c[i]) / maxc,
                                         ord=NormType))

                ceq_norm = (np.linalg.norm(np.abs(Population.ceq[i]) / np.abs(maxceq),
                                           ord=NormType))

                Population.Feasible[i] = c_norm <= CTol and ceq_norm <= CeqTol

            else:

                Population.Feasible[i] = (np.linalg.norm(np.maximum(0, Population.c), ord=NormType) <= CTol) and (
                        np.linalg.norm(np.abs(Population.ceq), ord=NormType) <= CeqTol)

    Population.Feasible = Population.Feasible.T
    Population.Rank = Population.Rank.T
    Population.Fitness = Population.Rank

    return Problem, Population


def ConEval(Problem, x, *args):
    Problem.Stats.ConCounter += 1

    try:
        res = Problem.Constraints(x, *args)
        c = res[0]
        ceq = res[1]
        if (res[0] == 0).all():
            c = 0
        if (res[1] == 0).all():
            ceq = 0
        if not isinstance(res[0], (list, np.ndarray)) or not isinstance(res[1], (list, np.ndarray)):
            raise ValueError('The constraints function must return one or two outputs as lists or numpy arrays.')
    except Exception as e:
        raise Exception(
            'Cannot continue because user supplied function constraints failed with the following error:\n%s' % str(e))

    return Problem, c, ceq


def ObjEval(Problem, x, *args):
    try:
        ObjValue = Problem.ObjFunction(x, *args)
        # update counter
        Problem.Stats.ObjFunCounter += 1
    except Exception as e:
        raise Exception('MEGA:ObjectiveError',
                        'Cannot continue because user supplied objective function failed with the following error:\n%s' % str(
                            e))

    return Problem, ObjValue


def share(dist, sigma):
    if dist <= sigma:
        sh = 1 - (dist / sigma) ** 2
    else:
        sh = 0
    return sh


def nondom(P, nv):

    n = P.shape[0]  # numero de pontos
    m = P.shape[1] - nv  # numero de objectivos
    k = 0  # contador para PL
    PL = np.zeros((n, m + nv))  # matriz PL
    for i in range(n):
        cand = True  # solucao candidata a nao dominada
        for j in range(n):
            if j != i and dom(P[j, :m], P[i, :m]):
                cand = False  # solucao i deixa de ser candidata a nao dominada
        if cand:  # se a solucao i e nao dominada
            PL[k, :] = P[i, :]  # acrescentar a PL
            k += 1

    PL = np.unique(PL[:k, :], axis=0)  # elimina solucoes repetidas
    return PL


def cnondom(P, nv, F, C):

    n = P.shape[0]  # numero de pontos
    m = P.shape[1] - nv  # numero de objectivos

    k = 0  # contador para PL
    PL = np.zeros((n, m + nv))  # matriz PL
    for i in range(n):
        cand = True  # solucao candidata a nao dominada
        for j in range(n):
            if j != i:
                if F[j] and not F[i]:  # j é admissivel e i não então j domina i
                    cand = False
                else:
                    if F[j] and F[i] and C[j] < C[i]:  # ambas não admissiveis e j viola menos do que i então j domina i
                        cand = False
                    else:
                        if dom(P[j, :m], P[i, :m]):  # se solucao j domina solucao i
                            cand = False  # solucao i deixa de ser candidata a nao dominada
        if cand:  # se a solucao i e nao dominada
            PL[k, :] = P[i, :]  # acrescentar a PL
            k += 1

    PL = np.unique(PL[:k, :], axis=0)  # elimina solucoes repetidas
    return PL


def dom(x, y):
    # retorna 1 se x domina y, 0 caso contrario

    m = len(x)  # numero de objectivos

    for i in range(m):
        if x[i] > y[i]:
            return 0  # x nao domina y
    for i in range(m):
        if x[i] < y[i]:
            return 1  # x domina y
    return 0  # x nao domina y


def RankPopulation(Population, elite, sigma, NormType):
    pop, obj = Population.f.shape
    # compute rank
    IP = np.where(Population.Feasible == 1)[0]
    P = np.hstack((Population.f[IP], IP.reshape((-1, 1))))
    rank = 1
    while P.shape[0] > 0:
        ND = nondom(P, 1)  # não dominadas e indices
        # P = np.setdiff1d(P, ND, assume_unique=True)
        P = P[~np.isin(P, ND).all(1)]

        for i in range(ND.shape[0]):
            pos = int(ND[i, obj])
            Population.Rank[pos] = rank
        rank += 1

    I = np.where(Population.Rank == 1)[0]

    if len(I) > 0:
        ideal = np.min(Population.f[I, :], axis=0)
        J = np.argmin(Population.f[I, :], axis=0)

        if sigma == 0:
            nadir = np.max(Population.f[I], axis=0)
            dnorm = np.linalg.norm(nadir - ideal)
            if dnorm == 0:
                dnorm = np.linalg.norm(np.max(Population.f, axis=0) - np.min(Population.f, axis=0))
            sigma = 2 * dnorm * (pop - np.floor(elite * pop / 2) - 1) ** (-1 / (obj - 1))

    fk = 1
    if sigma != 0:
        frente = 1
        while frente < rank:
            I = np.where(Population.Rank == frente)[0]
            LI = len(I)
            for i in range(LI):
                if I[i] not in J:
                    nc = 0
                    for j in range(LI):
                        nc += share(np.linalg.norm(Population.f[I[i]] - Population.f[I[j]]), sigma)
                    Population.Fitness[I[i]] = fk * nc
                else:
                    Population.Fitness[I[i]] = fk
            fk = np.floor(np.max(Population.Fitness[I])) + 1
            frente += 1
    else:
        Population.Fitness = Population.Rank

    # unfeasible
    IP = np.where(Population.Feasible == 0)[0]

    for i in range(len(IP)):
        Population.Rank[IP[i]] = rank

        if isinstance(Population.c[i], np.int32) and isinstance(Population.ceq[i], np.int32):
            Population.Fitness[IP[i]] = fk + np.linalg.norm(np.maximum(0, [Population.c[i]]),
                                                            NormType) + np.linalg.norm(
                np.abs([Population.ceq[i]]), NormType)
        elif isinstance(Population.c[i], np.int32) and isinstance(Population.ceq[i], np.ndarray):
            Population.Fitness[IP[i]] = fk + np.linalg.norm(np.maximum(0, [Population.c[i]]),
                                                            NormType) + np.linalg.norm(
                np.abs(Population.ceq[i]), NormType)

        elif isinstance(Population.c[i], np.ndarray) and isinstance(Population.ceq[i], np.int32):
            Population.Fitness[IP[i]] = fk + np.linalg.norm(np.maximum(0, [Population.c[i]]),
                                                            NormType) + np.linalg.norm(
                np.abs([Population.ceq[i]]), NormType)
        else:

            Population.Fitness[IP[i]] = (np.linalg.norm(np.abs(Population.ceq[i]), NormType))
            Population.Fitness[IP[i]] += float(fk)
            Population.Fitness[IP[i]] += np.linalg.norm(np.maximum(0, (Population.c[i])), NormType)

    return Population


def tournament_selection(chromosomes, pool_size, tour_size):
    pop = chromosomes.x.shape[0]
    P = Popul(np.zeros((pool_size, chromosomes.x.shape[1])), None)
    candidate = np.zeros((tour_size, 1), dtype=int)
    fitness = np.zeros((tour_size, 1))

    for i in range(pool_size):
        for j in range(tour_size):

            candidate[j] = np.ceil((pop - 1) * np.random.rand(1))
            if j >= 0:
                while np.any(candidate[0:j] == candidate[j]):
                    candidate[j] = np.ceil((pop - 1) * np.random.rand(1))

            fitness[j] = chromosomes.Fitness[candidate[j]]

        # min_candidate = np.argmin(fitness)
        min_fitness = np.min(fitness)
        min_candidate = np.where(fitness == min_fitness)[0]

        P.x[i, :] = chromosomes.x[candidate[min_candidate[0]] - 1, :]

    return P


def genetic_operator(Problem, parent_chromosome, pc, pm, mu, mum):
    N, V = parent_chromosome.x.shape
    child = np.zeros((N, V))
    p = 0
    while p < N:
        # SBX (Simulated Binary Crossover) applied with probability pc
        parent_1 = np.random.randint(0, N)
        parent_2 = np.random.randint(0, N)
        while parent_1 == parent_2:
            parent_2 = np.random.randint(0, N)
        parent_1 = parent_chromosome.x[parent_1]
        parent_2 = parent_chromosome.x[parent_2]
        if np.random.rand(1) < pc:
            child_1 = np.zeros(V)
            child_2 = np.zeros(V)
            for j in range(V):
                u = np.random.rand(1)
                if u <= 0.5:
                    bq = (2 * u) ** (1 / (mu + 1))
                else:
                    bq = (1 / (2 * (1 - u))) ** (1 / (mu + 1))
                child_1[j] = 0.5 * (((1 + bq) * parent_1[j]) + (1 - bq) * parent_2[j])
                child_2[j] = 0.5 * (((1 - bq) * parent_1[j]) + (1 + bq) * parent_2[j])
            child_1 = bounds(child_1, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])
            child_2 = bounds(child_2, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])
        else:
            child_1 = parent_1
            child_2 = parent_2
        # Polynomial mutation applied with probability pm
        if np.random.rand(1) < np.sqrt(pm):
            for j in range(V):
                if np.random.rand(1) < np.sqrt(pm):
                    r = np.random.rand(1)
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (mum + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))
                    child_1[j] = child_1[j] + (Problem.UB[j] - Problem.LB[j]) * delta
        child_1 = bounds(child_1, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])
        if np.random.rand(1) < np.sqrt(pm):
            for j in range(V):
                if np.random.rand(1) < np.sqrt(pm):
                    r = np.random.rand(1)
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (mum + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))
                    child_2[j] = child_2[j] + (Problem.UB[j] - Problem.LB[j]) * delta
        child_2 = bounds(child_2, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])

        child[p, :] = child_1[:V]
        if p + 1 >= N:
            child = np.insert(child, p + 1, child_2[:V], axis=0)
        else:
            child[p + 1] = child_2[0:V]
        p = p + 2

    Popul_x = child

    return Popul_x


def bounds(X, L, U):
    for _ in range(len(X)):
        if X[_] < L[_]:
            X[_] = L[_]
        if X[_] > U[_]:
            X[_] = U[_]
    return X
