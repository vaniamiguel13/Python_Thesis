import numpy as np

from MEGA.MEGACON.auxiliares import *
from HGPSAL.HGPSAL.AUX_Class import Problem_C, Population_C


def MegaCon(Problem, InitialPopulation, Options, *args):
    DefaultOpt = {'MaxObj': 2000, 'MaxGen': 1000, 'PopSize': 40, 'Elite': 0.1, 'TourSize': 2, 'Pcross': 0.9,
                  'Icross': 20, 'Pmut': 0.1, 'Imut': 20, 'Sigma': 0.1,
                  'CPTolerance': 1.0e-6, 'CPGenTest': 0.01, 'CTol': 1e-2, 'CeqTol': 1e-2, 'NormType': float('inf'),
                  'NormCons': 1, 'Verbosity': False}

    # With no arguments just print an error
    if Problem is None:
        raise ValueError('Invalid number of arguments.')

    if hasattr(Problem, 'Constraints') and Problem.Constraints:
        print('MEGA: Constrained tournament handling enabled.')
        Conflag = 1
    else:
        Conflag = 0

    MaxGenerations = Options['MaxGen'] if Options and 'MaxGen' in Options else DefaultOpt['MaxGen']
    MaxEvals = Options['MaxObj'] if Options and 'MaxObj' in Options else DefaultOpt['MaxObj']
    Pop = Options['PopSize'] if Options and 'PopSize' in Options else DefaultOpt['PopSize']
    Elite = Options.get('EliteProp', None) if Options else DefaultOpt.get('EliteProp', None)
    elite_inf = max(2, math.ceil(Elite / 2 * Pop))
    elite_sup = min(Pop - 2, math.floor((1 - Elite / 2) * Pop))
    print(f"MEGA: MEGA elite size set to the interval {elite_inf} and {elite_sup}")

    Tour = Options['TourSize'] if Options and 'TourSize' in Options else DefaultOpt['TourSize']
    Pc = Options['Pcross'] if Options and 'Pcross' in Options else DefaultOpt['Pcross']
    Ic = Options['Icross'] if Options and 'Icross' in Options else DefaultOpt['Icross']

    if 'Pmut' not in Options:
        Pm = 1 / Problem.Variables
        print(f"MEGA: MEGA mutation probability set to {Pm}")
    else:
        Pm = Options['Pmut']

    Im = Options['Imut'] if Options and 'Imut' in Options else DefaultOpt['Imut']

    if 'Sigma' not in Options:
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

    Population.Feasible = temp[:, Population.shape[1] - 3]
    Population.Rank = temp[:, Population.shape[1] - 2]
    Population.Fitness = temp[:, -1]

    # Iteration counter
    Problem.Stats.GenCounter = 0
    # Initialize statistics structures
    # Keep track of population statistics

    Problem.Stats.N1Front[Problem.Stats.GenCounter + 1] = len(Population.Rank[Population.Rank == 1])
    Problem.Stats.NFronts[Problem.Stats.GenCounter + 1] = max(Population.Rank)

    while Problem.Stats.GenCounter < MaxGenerations and Problem.Stats.ObjFunCounter < MaxEvals:
        # Increment generation counter.
        Problem.Stats.GenCounter += 1

        # Select the parents
        # Parents are selected for reproduction to generate offspring. The arguments are
        # pool - size of the mating pool. It is common to have this to be equal to the
        #        population size. However, if elistm is intended then pool must
        #        be inferior to the population size, i.e., pool=pop-elitesize.
        #        Typically, 10% percent of population size.
        # tour - Tournament size. For binary tournament
        #        selection set tour=2, but to see the effect of tournament size in the selection pressure this is kept
        #        arbitary, to be choosen by the user.
        # elitesize = round(Pop*Elite);
        # pool = Pop - elitesize;

        if Elite is not None:
            pool = math.floor(
                np.maximum(elite_inf, np.minimum(elite_sup, Pop - len(np.where(Population.Rank[Population.Feasible == 1] == 1)[0]))))
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
        Population.x[Pop - pool:Pop, :] = offspring_chromosome.x[0:pool, :]

        # Evaluate the objective function
        for i in range(Pop - pool, Pop):
            Problem, Population.f[i, :] = ObjEval(Problem, Population.x[i, :], *args)
            if Conflag:
                Problem, Population.c[i, :], Population.ceq[i, :] = ConEval(Problem, Population.x[i, :], *args)
            else:
                Population.c[i, :] = 0
                Population.ceq[i, :] = 0
                Population.Feasible[i] = 1

    return NonDomPoint, FrontPoint, RunData
