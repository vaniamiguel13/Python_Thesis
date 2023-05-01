import numpy as np
import math


def InitPopulation(Problem, InitialPopulation, Size, Conflag, CTol, CeqTol, NormType, NormCons, *args):
    Population = Popul
    Population.x = np.zeros((Size, Problem.Variables))
    Population.f = np.zeros(Size, )
    if InitialPopulation and not isinstance(InitialPopulation, dict):
        raise ValueError('MEGA:InitPopulation:InitialPopulation - Initial population must be defined in a structure.')
    else:
        # Check for size
        if len(InitialPopulation) > Size:
            # User provided an initial population greater than the parent population size
            raise ValueError(
                'MEGA:InitPopulation:InitialPopulationSize - Initial population size must be inferior to PopSize.')
        # Copy the initial population for the population and initialize them
        for i in range(len(InitialPopulation)):
            x = InitialPopulation
            Population.x[i, :] = Bounds(x, Problem.LB[:Problem.Variables], Problem.UB[:Problem.Variables])
            Problem, Population.f[i] = ObjEval(Problem, Population.x[i, :], *args)
            if Conflag:
                Problem, Population.c[i, :], Population.ceq[i, :] = ConEval(Problem, Population.x[i, :], *args)
                # Population['Feasible'][i] = (np.linalg.norm(np.maximum(0, Population['c'][i,:]), norm=NormType) <= CTol and np.linalg.norm(np.abs(Population['ceq'][i,:]), norm=NormType) <= CeqTol)
            else:
                Population.c[i, :] = 0
                Population.ceq[i, :] = 0
                Population.Feasible[i] = 1
            Population.Rank[i] = 0
    # Randomly generate the remaining population
    for i in range(len(InitialPopulation) + 1, Size + 1):
        Population.x[i, :] = Problem.LB[0:Problem.Variables] + np.multiply(np.subtract(Problem.UB[0:Problem.Variables],
                                                                                       Problem.LB[0:Problem.Variables]),
                                                                           np.random.rand(1, Problem.Variables))
        Problem, Population.f[i, :] = ObjEval(Problem, Population.x[i, :], *args)
        if Conflag:
            Problem, Population.c[i, :], Population.ceq[i, :] = ConEval(Problem, Population.x[i, :], *args)
            # Population['Feasible'][i] = (np.linalg.norm(np.maximum(0, Population['c'][i,:]), norm=NormType) <= CTol and np.linalg.norm(np.abs(Population['ceq'][i,:]), norm=NormType) <= CeqTol)
        else:
            Population.c[i, :] = 0
            Population.ceq[i, :] = 0
            Population.Feasible[i] = 1
        Population.Rank[i] = 0

    if Conflag:
        for i in range(Size):
            if NormCons:
                maxc = np.minimum(np.maximum(0, Population.c), axis=0)
                for j in range(len(maxc)):
                    if maxc[j] == 0:
                        maxc[j] = 1
                maxceq = np.minimum(Population.ceq, axis=0)
                for j in range(len(maxceq)):
                    if maxceq[j] == 0:
                        maxceq[j] = 1
                Population.Feasible[i] = (np.linalg.norm(np.maximum(0, Population.c[i, :]) / maxc,
                                                         ord=NormType) <= CTol and np.linalg.norm(
                    np.abs(Population.ceq[i, :]) / np.abs(maxceq), ord=NormType) <= CeqTol)
            else:
                Population.Feasible[i] = (
                        np.linalg.norm(np.maximum(0, Population.c[i, :]), ord=NormType) <= CTol and np.linalg.norm(
                    np.abs(Population.ceq[i, :]), ord=NormType) <= CeqTol)

    Population.Feasible = Population.Feasible.reshape((-1, 1))
    Population.Rank = Population.Rank.reshape((-1, 1))
    Population.Fitness = Population.Rank

    return Problem, Population


def ConEval(Problem, x, *args):
    Problem.Stats.ConCounter += 1

    try:
        res = Problem.Constraints(x, *args)
        if len(res) == 1:
            c = res[0]
            ceq = []
        elif len(res) == 2:
            c = res[0]
            ceq = res[1]
        else:
            raise ValueError('The constraints function must return one or two outputs.')
    except Exception as e:
        raise Exception(
            'Cannot continue because user supplied function constraints failed with the following error:\n%s' % str(e))

    return c, ceq


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
        sh = 1 - (dist / sigma) ^ 2
    else:
        sh = 0
    return sh


def nondom(P, nv):
    # INPUT: matriz P -> n x (m+nv)
    #                    n -> numero de pontos
    #                    m -> numero de objectivos
    #        escalar nv -> numero de variaveis
    # OUTPUT: matriz PL -> nl x (m+nv)
    #                    nl -> numero de pontos nao dominados

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
    # INPUT: matriz P -> n x (m+nv)
    #                    n -> numero de pontos
    #                    m -> numero de objectivos
    #        escalar nv -> numero de variaveis
    # OUTPUT: matriz PL -> nl x (m+nv)
    #                    nl -> numero de pontos nao dominados

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
        P = np.setdiff1d(P, ND, axis=0)
        for i in range(ND.shape[0]):
            Population.Rank[ND[i, obj]] = rank
        rank += 1

    I = np.where(Population.Rank == 1)[0]
    ideal = np.min(Population.f[I], axis=0)
    J = np.argmin(Population.f[I], axis=0)
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
        Population.Fitness[IP[i]] = fk + np.linalg.norm(np.maximum(0, Population.c[i]), NormType) + np.linalg.norm(
            np.abs(Population.ceq[i]), NormType)

    return Population
