import numpy as np
from HGPSAL.HGPSAL.Time import tic, toc
from MEGA.MEGACON.auxiliares import *
from HGPSAL.HGPSAL.AUX_Class.Problem_C import Problem
from HGPSAL.HGPSAL.AUX_Class.Population_C import Popul


def MegaCon(Problem, InitialPopulation = None, Options= None, *args):
    DefaultOpt = {'MaxObj': 2000, 'MaxGen': 1000, 'PopSize': 40, 'Elite': 0.1, 'TourSize': 2, 'Pcross': 0.9,
                  'Icross': 20, 'Pmut': 0.1, 'Imut': 20, 'Sigma': 0.1,
                  'CPTolerance': 1.0e-6, 'CPGenTest': 0.01, 'CTol': 1e-2, 'CeqTol': 1e-2, 'NormType': np.inf,
                  'NormCons': 1, 'Verbosity': False}

    # With no arguments just print an error
    if Problem is None:
        raise ValueError('Invalid number of arguments.')

    if hasattr(Problem, 'Constraints') and Problem.Constraints:
        print('MEGA: Constrained tournament handling enabled.')
        Conflag = True
    else:
        Conflag = False

    MaxGenerations = Options['MaxGen'] if Options and 'MaxGen' in Options else DefaultOpt['MaxGen']
    MaxEvals = Options['MaxObj'] if Options and 'MaxObj' in Options else DefaultOpt['MaxObj']
    Pop = Options['PopSize'] if Options and 'PopSize' in Options else DefaultOpt['PopSize']
    Elite = Options['Elite'] if Options and 'Elite' in Options else DefaultOpt['Elite']
    elite_inf = max(2, math.ceil(Elite / 2 * Pop))
    elite_sup = min(Pop - 2, math.floor((1 - Elite / 2) * Pop))
    print(f"MEGA: MEGA elite size set to the interval {elite_inf} and {elite_sup}")

    Tour = Options['TourSize'] if Options and 'TourSize' in Options else DefaultOpt['TourSize']
    Pc = Options['Pcross'] if Options and 'Pcross' in Options else DefaultOpt['Pcross']
    Ic = Options['Icross'] if Options and 'Icross' in Options else DefaultOpt['Icross']

    if Options is None or 'Pmut' not in Options:
        Pm = 1 / Problem.Variables
        print(f"MEGA: MEGA mutation probability set to {Pm}")
    else:
        Pm = Options['Pmut']

    Im = Options['Imut'] if Options and 'Imut' in Options else DefaultOpt['Imut']

    if Options is None or 'Sigma' not in Options:
        print(f"MEGA: MEGA niching radius will be adapted during the search {Pm}")
        sigma = 0
    else:
        sigma = Options['Sigma']

    CTol = Options['CTol'] if Options and 'CTol' in Options else DefaultOpt['CTol']
    CeqTol = Options['CeqTol'] if Options and 'CeqTol' in Options else DefaultOpt['CeqTol']
    NormType = Options['NormType'] if Options and 'NormType' in Options else DefaultOpt['NormType']
    NormCons = Options['NormCons'] if Options and 'NormCons' in Options else DefaultOpt['NormCons']

    Problem.Verbose = Options['Verbosity'] if Options and 'Verbosity' in Options else DefaultOpt['Verbosity']
    Problem.Tolerance = Options['CPTolerance'] if Options and 'CPTolerance' in Options else DefaultOpt['CPTolerance']
    Problem.GenTest = Options['CPGenTest'] if Options and 'CPGenTest' in Options else DefaultOpt['CPGenTest']

    tic()

    Problem.Stats.ObjFunCounter = 0
    Problem.Stats.ConCounter = 0

    Problem, Population = InitPopulation(Problem, InitialPopulation, Pop, Conflag, CTol, CeqTol, NormType, NormCons,
                                         *args)

    Population = RankPopulation(Population, Elite, sigma, NormType)
    Population.c = np.reshape(Population.c, (Pop, Problem.L_c))
    Population.ceq = np.reshape(Population.ceq, (Pop, Problem.L_ceq))
    temp = np.concatenate((Population.x, Population.f, Population.c, Population.ceq, Population.Feasible.reshape(-1, 1),
                           Population.Rank.reshape(-1, 1), Population.Fitness.reshape(-1, 1)), axis=1)

    temp = temp[temp[:, -1].argsort()]
    Population.x = temp[:, :Population.x.shape[1]]
    Population.f = temp[:, Population.x.shape[1]:Population.x.shape[1] + Population.f.shape[1]]

    Population.c = temp[:, Population.x.shape[1] + Population.f.shape[1]:Population.x.shape[1] + Population.f.shape[1] +
                                                                         Population.c.shape[1]]

    Population.ceq = temp[:,
                     Population.x.shape[1] + Population.f.shape[1] + Population.c.shape[1]:Population.x.shape[1] +
                                                                                           Population.f.shape[1] +
                                                                                           Population.c.shape[1] +
                                                                                           Population.ceq.shape[1]]


    Population.Feasible = temp[:,
                     Population.x.shape[1] + Population.f.shape[1] + Population.c.shape[1]
                     :Population.x.shape[1] + Population.f.shape[1] + Population.c.shape[1] + Population.ceq.shape[1]
                     ]
    Population.Feasible = temp[:, -3]
    Population.Rank = temp[:, -2]
    Population.Fitness = temp[:, -1]

    # Iteration counter
    Problem.Stats.GenCounter = 0
    # Initialize statistics structures
    # Keep track of population statistics

    Problem.Stats.N1Front.append(len(Population.Rank[Population.Rank == 1]))
    Problem.Stats.NFronts.append(max(Population.Rank))

    if Problem.Verbose:
        print('MEGA is running... ')
        if np.sum(Population.Feasible) == 0:
            best_point_norm = np.linalg.norm(np.maximum(0, Population.c[0]), NormType) + np.linalg.norm(
                np.abs(Population.ceq[0]), NormType)
            print(
                f"Gen: {Problem.Stats.GenCounter }  No. points in 1st front = {Problem.Stats.N1Front[Problem.Stats.GenCounter]}  Number of fronts = {Problem.Stats.NFronts[Problem.Stats.GenCounter]}  All points are unfeasible. Best point: {best_point_norm}")
        else:
            print(
                f"Gen: {Problem.Stats.GenCounter}  No. points in 1st front = {Problem.Stats.N1Front[Problem.Stats.GenCounter]}  Number of fronts = {Problem.Stats.NFronts[Problem.Stats.GenCounter]}")

    while Problem.Stats.GenCounter < MaxGenerations and Problem.Stats.ObjFunCounter < MaxEvals:
        # Increment generation counter.
        Problem.Stats.GenCounter = Problem.Stats.GenCounter + 1

        if Elite is not None:
            pool = math.floor(
                np.maximum(elite_inf, np.minimum(elite_sup, Pop - len(
                    np.where(Population.Rank[Population.Feasible == 1] == 1)[0]))))
        else:
            pool = Pop

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
        Population.x[Pop - pool:Pop, :] = offspring_chromosome[0:pool, :]

        # Evaluate the objective function
        for i in range(Pop - pool, Pop):
            Problem, Population.f[i, :] = ObjEval(Problem, Population.x[i, :], *args)

            if Conflag:
                C_eval = ConEval(Problem, Population.x[i, :], *args)
                Problem = C_eval[0]
                Population.c[i] = np.transpose(C_eval[1])
                Population.ceq[i] = np.transpose(C_eval[2])

            else:
                Population.c[i, :] = 0
                Population.ceq[i, :] = 0
                Population.Feasible[i] = 1

        if Conflag:
            for i in range(Pop - pool + 1, Pop + 1):
                if NormCons:
                    maxc = np.minimum(np.maximum(0, Population.c.max(axis=0)), 1)

                    for j in range(len(maxc)):
                        if maxc[j] == 0:
                            maxc[j] = 1

                    maxceq = np.minimum(Population.ceq.min(axis=0), 1)

                    for j in range(len(maxceq)):
                        if maxceq[j] == 0:
                            maxceq[j] = 1

                    Population.Feasible[i - 1] = (
                            np.linalg.norm(np.maximum(0, Population.c[i - 1, :]) / maxc, ord=NormType) <= CTol and
                            np.linalg.norm(np.abs(Population.ceq[i - 1, :]) / np.abs(maxceq), ord=NormType) <= CeqTol)
                else:
                    Population.Feasible[i - 1] = (
                            np.linalg.norm(np.maximum(0, Population.c[i - 1, :]), ord=NormType) <= CTol and
                            np.linalg.norm(np.abs(Population.ceq[i - 1, :]), ord=NormType) <= CeqTol)

            Population = RankPopulation(Population, Elite, sigma, NormType)

            temp = np.hstack(
                (Population.x, Population.f, Population.c, Population.ceq, Population.Feasible.reshape((-1, 1)),
                 Population.Rank.reshape((-1, 1)), Population.Fitness.reshape((-1, 1))))
            temp = temp[temp[:, -1].argsort()]
            Population.x = temp[:, :Population.x.shape[1]]
            Population.f = temp[:, Population.x.shape[1]:Population.x.shape[1] + Population.f.shape[1]]
            Population.c = temp[:,
                           Population.x.shape[1] + Population.f.shape[1]:Population.x.shape[1] + Population.f.shape[1] +
                                                                         Population.c.shape[1]]
            Population.ceq = temp[:,
                             Population.x.shape[1] + Population.f.shape[1] +
                             Population.c.shape[1]:Population.x.shape[1] + Population.f.shape[1] +
                                                   Population.c.shape[1] + Population.ceq.shape[1]]

            Population.Feasible = temp[:, -3].astype(bool)
            Population.Rank = temp[:, -2].astype(int)
            Population.Fitness = temp[:, -1]

            # Problem.Stats.N1Front[Problem.Stats.GenCounter + 1] = np.sum(Population.Rank == 1)
            # Problem.Stats.NFronts[Problem.Stats.GenCounter + 1] = np.max(Population.Rank)
            # Problem.Stats.N1Front.append(len(Population.Rank[Population.Rank == 1]))
            Problem.Stats.N1Front.append(len([x for x in Population.Rank if x == 1]))
            Problem.Stats.NFronts.append(max(Population.Rank))
            if Problem.Verbose:
                print('MEGA is running... ')
                if np.sum(Population.Feasible) == 0:
                    best_point_norm = np.linalg.norm(np.maximum(0, Population.c[0]), NormType) + np.linalg.norm(
                        np.abs(Population.ceq[0]), NormType)
                    print(
                        f"Gen: {Problem.Stats.GenCounter}  No. points in 1st front = {Problem.Stats.N1Front[Problem.Stats.GenCounter]}  Number of fronts = {Problem.Stats.NFronts[Problem.Stats.GenCounter]}  All points are unfeasible. Best point: {best_point_norm}")
                else:
                    print(
                        f"Gen: {Problem.Stats.GenCounter}  No. points in 1st front = {Problem.Stats.N1Front[Problem.Stats.GenCounter]}  Number of fronts = {Problem.Stats.NFronts[Problem.Stats.GenCounter]}")

    # print(Problem.Stats.GenCounter)
    # print(Problem.Stats.ObjFunCounter)
    # print(Problem.Stats.ConCounter)
    toc()

    if Problem.Stats.GenCounter >= MaxGenerations or Problem.Stats.ObjFunCounter >= MaxEvals:
        print('Maximum number of iterations or objective function evaluations reached')

    RunData = Problem.Stats

    NonDomPoint = []
    FrontPoint_f = []
    FrontPoint_c = []
    FrontPoint_ceq = []
    k = 1
    for i in range(Pop):
        if Population.Rank[i] == 1:
            NonDomPoint.insert(k, Population.x[i, :])
            FrontPoint_f.insert(k, Population.f[i, :])
            FrontPoint_c.insert(k, Population.c[i, :])
            FrontPoint_ceq.insert(k, Population.ceq[i, :])
            k = k + 1

    return NonDomPoint, FrontPoint_f, FrontPoint_c, FrontPoint_ceq, RunData

