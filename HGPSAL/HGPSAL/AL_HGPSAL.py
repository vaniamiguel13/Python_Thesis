
import numpy as np
from HGPSAL.HGPSAL.penalty import penalty2
from HGPSAL.HGPSAL.GeneticA import rGA
import copy
from HGPSAL.HGPSAL.HJ import HJ
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
import random


class Problem:
    def __init__(self, Variables, ObjFunction, LB, UB, Constraints, x0=None):
        self.Variables = Variables
        self.x0 = x0
        self.Constraints = Constraints
        self.ObjFunction = ObjFunction
        self.LB = LB
        self.UB = UB
        self.m = None
        self.p = None
        self.Verbose = False
        self.Tolerance = 1.0e-6
        self.GenTest = 0.01
        self.Stats = ProblemStatistics()


class ProblemStatistics:
    def __init__(self):
        self.exit = None
        self.objfun = None
        self.x = None
        self.fx = None
        self.c = None
        self.ceq = None
        self.history = None
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


class alg:
    def __init__(self):
        self.lbmd = None
        self.ldelta = None
        self.miu = None
        self.alfa = None
        self.omega = None
        self.epsilon = None
        self.eta = None
        self.delta = None


def lag(x, Problem, alg):
    Value = penalty2(Problem, x, alg)
    f = Value.la
    return f


def Projection(Problem, x):
    for i in range(Problem.Variables):
        if x[i] < Problem.LB[i]:
            x[i] = Problem.LB[i]
        if x[i] > Problem.UB[i]:
            x[i] = Problem.UB[i]
    return x


def mega_(lambda_, ldelta, miu, teta_tol):
    norm_lambda = np.linalg.norm(lambda_)
    norm_ldelta = np.linalg.norm(ldelta)
    numerator = 1 + norm_lambda + norm_ldelta + (1 / miu)
    denominator = teta_tol
    return 1 / max(1, numerator / denominator)


def HGPSAL(Problem, Options=None, *args):
    DefaultOpt = {'lambda_min': -1e12, 'lambda_max': 1e12, 'teta_tol': 1e12, 'miu_min': 1e-12, 'miu0': 1, 'csi': 0.5,
                  'eta0': 1, 'ffeas': 1, 'gps': 1, 'niu': 1.0, 'zeta': 0.001, 'epsilon1': 1e-4, 'epsilon2': 1e-8,
                  'suficient': 1e-4, 'method': 0,
                  'omega0': 1, 'alfaw': 0.9, 'alfa_eta': 0.9 * 0.9, 'betaw': 0.9, 'beta_eta': 0.5 * 0.9, 'gama1': 0.5,
                  'teta_miu': 0.5,
                  'pop_size': 40, 'elite_prop': 0.1, 'tour_size': 2, 'pcross': 0.9, 'icross': 20, 'pmut': 0.1,
                  'imut': 20,
                  'gama': 1, 'delta': 1, 'teta': 0.5, 'eta_asterisco': 1.0e-2, 'epsilon_asterisco': 1.0e-6,
                  'cp_ga_test': 0.1, 'cp_ga_tol': 1.0e-6, 'delta_tol': 1e-6, 'maxit': 100, 'maxet': 200,
                  'max_objfun': 20000, 'verbose': False}

    # With no arguments just print an error
    if Problem is None:
        raise ValueError('Invalid number of arguments.')

    if Problem.Variables is None or Problem.Variables == []:
        raise ValueError('Problem dimension is missing.')

    x0 = Problem.x0

    # if np.size(x0, 0) > np.size(x0, 1):
    #     x0 = np.transpose(x0)

    if Problem.LB is None:
        raise ValueError('Problem lower bounds are missing.')
    lb = Problem.LB
    # if np.size(lb, 0) > np.size(lb, 1):
    #     lb = np.transpose(lb)

    if Problem.UB is None:
        raise ValueError('Problem upper bounds are missing.')
    ub = Problem.UB
    # if np.size(ub, 0) > np.size(ub, 1):
    #     ub = np.transpose(ub)

    if Problem.ObjFunction is None or Problem.ObjFunction == []:
        raise ValueError('Objective function name is missing.')

    if Problem.Constraints is None or Problem.Constraints == []:
        raise ValueError('Function constraints are missing.')

    # If no options passed in, set default options
    # if Options is None:
    #     pop_size = min(20 * Problem.Variables, 200)
    #     print(f"HGPSAL: rGA population size set to {pop_size}")
    #     pmut = 1 / Problem.Variables
    #     print(f"HGPSAL: rGA mutation probability set to {pmut}")
    # else:

    beta_eta = Options['beta_eta'] if Options and 'beta_eta' in Options else DefaultOpt['beta_eta']
    eta_asterisco = Options['eta_asterisco'] if Options and 'eta_asterisco' in Options else DefaultOpt[
        'eta_asterisco']
    epsilon_asterisco = Options['epsilon_asterisco'] if Options and 'epsilon_asterisco' in Options else DefaultOpt[
        'epsilon_asterisco']
    max_objfun = Options['max_objfun'] if Options and 'max_objfun' in Options else DefaultOpt['max_objfun']
    gama = Options['gama'] if Options and 'gama' in Options else DefaultOpt['gama']
    delta = Options['delta'] if Options and 'delta' in Options else DefaultOpt['delta']
    teta_miu = Options['teta_miu'] if Options and 'teta_miu' in Options else DefaultOpt['teta_miu']
    lambda_max = Options['lambda_max'] if Options and 'lambda_max' in Options else DefaultOpt['lambda_max']
    lambda_min = Options['lambda_min'] if Options and 'lambda_min' in Options else DefaultOpt['lambda_min']
    teta_tol = Options['teta_tol'] if Options and 'teta_tol' in Options else DefaultOpt['teta_tol']
    csi = Options['csi'] if Options and 'csi' in Options else DefaultOpt['csi']
    miu0 = Options['miu0'] if Options and 'miu0' in Options else DefaultOpt['miu0']
    maxit = Options['maxit'] if Options and 'maxit' in Options else DefaultOpt['maxit']
    maxet = Options['maxet'] if Options and 'maxet' in Options else DefaultOpt['maxet']
    verbose = Options['verbose'] if Options and 'verbose' in Options else DefaultOpt['verbose']
    teta = Options['teta'] if Options and 'teta' in Options else DefaultOpt['teta']
    miu_min = Options['miu_min'] if Options and 'miu_min' in Options else DefaultOpt['miu_min']
    alfaw = Options['alfaw'] if Options and 'alfaw' in Options else DefaultOpt['alfaw']
    alfa_eta = Options['alfa_eta'] if Options and 'alfa_eta' in Options else DefaultOpt['alfa_eta']
    gama1 = Options['gama1'] if Options and 'gama1' in Options else DefaultOpt['gama1']
    eta0 = Options['eta0'] if Options and 'eta0' in Options else DefaultOpt['eta0']
    omega0 = Options['omega0'] if Options and 'omega0' in Options else DefaultOpt['omega0']
    betaw = Options['betaw'] if Options and 'betaw' in Options else DefaultOpt['betaw']

    pop_size = Options['pop_size'] if Options and 'pop_size' in Options else min(20 * Problem.Variables, 200)
    elite_prop = Options['elite_prop'] if Options and 'elite_prop' in Options else DefaultOpt['elite_prop']
    tour_size = Options['tour_size'] if Options and 'tour_size' in Options else DefaultOpt['tour_size']
    pcross = Options['pcross'] if Options and 'pcross' in Options else DefaultOpt['pcross']
    icross = Options['icross'] if Options and 'icross' in Options else DefaultOpt['icross']

    pmut = Options['pmut'] if Options and 'pmut' in Options else 1 / Problem.Variables

    imut = Options['imut'] if Options and 'imut' in Options else DefaultOpt['imut']
    cp_ga_test = Options['cp_ga_test'] if Options and 'cp_ga_test' in Options else DefaultOpt['cp_ga_test']

    # tic()

    # Set initial guess
    if x0 is None:
        x = np.zeros(Problem.Variables)
        for i in range(Problem.Variables):
            if lb[i] > -np.inf and ub[i] < np.inf:
                # Both limits are finite
                x[i] = np.random.rand() * (ub[i] - lb[i]) + lb[i]
            else:
                if lb[i] <= -np.inf and ub[i] >= np.inf:
                    # Both limits are infinite
                    x[i] = 20 * (np.random.rand() - 0.5)
                else:
                    if lb[i] <= -np.inf:
                        x[i] = ub[i] - abs(2 * np.random.rand() * ub[i])
                    else:
                        x[i] = lb[i] + abs(2 * np.random.rand() * lb[i])
    else:
        x = x0

    x = Projection(Problem, x)

    try:
        fx = Problem.ObjFunction(x, *args)
    except Exception as e:
        raise Exception(
            f"augLagr:ConstraintsError\nCannot continue because user supplied objective function failed with the "
            f"following error:\n{str(e)}")

    try:
        c, ceq = Problem.Constraints(x, *args)
    except Exception as e:
        raise Exception(
            f"augLagr:ConstraintsError\nCannot continue because user supplied function constraints failed with the "
            f"following error:\n{str(e)}")

    Problem.m = len(ceq)
    Problem.p = len(c)

    # # DEBUG ONLY
    # if opt['verbose']:
    #     print('Initial guess:')
    #     print(x)
    # # -----------

    if Problem.m == 0:
        alg.lmbd = []
    else:
        alg.lmbd = np.ones(Problem.m)

    if Problem.p == 0:
        alg.ldelta = []
    else:
        alg.ldelta = np.ones(Problem.p)

    # Initialize

    alg.miu = miu0
    alg.alfa = min(alg.miu, gama1)
    alg.omega = omega0 * (alg.alfa ** alfaw)
    alg.epsilon = alg.omega * mega_(alg.lmbd, alg.ldelta, alg.miu, teta_tol)
    alg.eta = eta0 * (alg.alfa ** alfa_eta)

    # set pattern delta
    alg.delta = []
    if delta == 0:
        for j in range(Problem.Variables):
            if x0 is None or x0[j] == 0:
                alg.delta.append(gama)
            else:
                alg.delta.append(x0[j] * gama)
    elif delta == 1:
        alg.delta = np.ones(Problem.Variables)
    else:
        raise ValueError('Invalid option for delta, input a valid option (0 or 1)')

    # Initialize statistics
    stats = Problem.Stats
    stats.exit = 0
    stats.objfun = 0
    stats.x = [x]
    stats.fx = [fx]
    stats.c = [c] if c else []
    stats.ceq = [ceq] if ceq else []
    # stats.history = {}
    # d_it = {}
    global_search = True
    Opt = {}
    # initialize stats.history with column names
    stats.history = [('Iter', 'fx rGA', 'nf rGA', 'fx HJ', 'nf HJ')]
    while stats.exit <= maxet and stats.objfun <= max_objfun:
        # {1: {'Iter': None, 'fx rGA': None, 'nf rGA': None, 'fx HJ': None, 'nf HJ': None}}
        stats.exit += 1
        i = stats.exit
        # d_it['Iter'] = stats.extit
        # stats.history[stats.extit + 1] = d_it

        # GENERALIZED rGA---------------------
        Probl = copy.copy(Problem)
        Probl.ObjFunction = lag

        if global_search:
            InitialPopulation = x

            # InitialPopulation.x[0] = x

            Opt['PopSize'] = pop_size
            Opt['EliteProp'] = elite_prop
            Opt['TourSize'] = tour_size
            Opt['Pcross'] = pcross
            Opt['Icross'] = icross
            Opt['Pmut'] = pmut
            Opt['Imut'] = imut
            Opt['CPTolerance'] = alg.epsilon
            Opt['CPGenTest'] = cp_ga_test
            Opt['MaxGen'] = maxit
            Opt['MaxObj'] = max_objfun
            Opt['Verbosity'] = verbose

            x, fval, RunData = rGA(Probl, InitialPopulation, Opt, Problem, alg)
            stats.objfun += RunData.ObjFunCounter
            fx_ = fval
            # d_it['fx rGA'] = fval
            # stats.history[stats.extit + 1] = d_it
            nF_ = RunData.ObjFunCounter
            # d_it['nf rGA'] = RunData.ObjFunCounter
            # stats.history[stats.extit + 1] = d_it

            # print(stats.history)
        #     # global_search=0
        #     if opt.verbose:
        #         print('GA external it: %d' % stats.extit)
        #         print(x)
        #         print(fval)
        #         print(RunData)
        #
        Opt['MaxIter'] = maxit
        Opt['MaxObj'] = max_objfun
        Opt['DeltaTol'] = alg.epsilon
        Opt['Theta'] = teta
        x, fval, Rundata = HJ(Probl, x, alg.delta, Options, Problem, alg)
        stats.objfun += Rundata.ObjFunCounter
        fx_H = fval
        # d_it['fx HJ'] = fval
        # stats.history[stats.extit + 1] = d_it
        nF_H = Rundata.ObjFunCounter
        # d_it['nf HJ'] = Rundata.ObjFunCounter
        # stats.history[stats.extit + 1] = d_it
        stats.history.append((i, fx_, nF_, fx_H, nF_H))
        # if opt.verbose:
        #     print('HJ external it: %d' % stats.extit)
        #     print(x)
        #     print(fval)
        #     print(Rundata)

        Value = penalty2(Problem, x, alg)
        c = Value.c
        ceq = Value.ceq
        fx = Value.fx
        la = Value.la

        stats.x.append(x)
        stats.fx.append(fx)

        if len(c):
            stats.c.append(c)
        if len(ceq):
            stats.ceq.append(ceq)

        # if opt.verbose:
        #     print(x)
        #     print(fx)
        #     print(c)
        #     print(ceq)
        #     print(la)
        # # ------------------------------------

        if not alg.lmbd and not alg.delta:
            break

        max_i = max(abs(ceq))
        v = max(max(c), max(alg.ldelta * abs(c))) if c else 0
        if not max_i:
            max_i = 0

        norma_lambda = np.linalg.norm(alg.lmbd)
        norma_x = np.linalg.norm(x)

        alg.ldelta = np.maximum(lambda_min, np.minimum(np.maximum(0, alg.ldelta + c / alg.miu),lambda_max))

        if max_i <= alg.eta * (1 + norma_x) and v <= alg.eta * (1 + norma_lambda):

            if alg.epsilon < epsilon_asterisco and max_i <= eta_asterisco * (1 + norma_x) and v <= eta_asterisco * (1 + norma_lambda) and global_search == False:
                stats.Message = 'HGPSAL: Tolerance of constraints violations satisfied.'
                print(stats.Message)
                break
            else:

                alg.lmbd = max(lambda_min, min(alg.lmbd + ceq / alg.miu, lambda_max))
                alg.alfa = min(alg.miu, gama1)
                alg.omega = alg.omega * pow(alg.alfa, betaw)
                alg.epsilon = alg.omega * mega_(alg.lmbd, alg.ldelta, alg.miu, teta_tol)
                alg.eta = alg.eta * pow(alg.alfa, beta_eta)
        else:
            alg.miu = max(min(alg.miu * csi, pow(alg.miu, teta_miu)), miu_min)
            alg.alfa = min(alg.miu, gama1)
            alg.omega = omega0 * pow(alg.alfa, alfaw)
            alg.epsilon = alg.omega * mega_(alg.lmbd , alg.ldelta, alg.miu, teta_tol)
            alg.eta = eta0 * pow(alg.alfa, alfa_eta)
            global_search = True

    if stats.exit > maxet:
        stats.Message = 'HGPSAL: Maximum number of external iterations reached.'
        print(stats.Message)

    if stats.objfun > max_objfun:
        stats.Message = 'HGPSAL: Maximum number objective function evaluations reached.'
        print(stats.Message)

    return x, fx, c, ceq, la, stats


Variables = 2


def Rastrigin(x):
    f = 20 + x[0] ** 2 + x[1] ** 2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
    return f


def Rast_constr(x):
    c = [(x[0] - 2.5) ** 2 + (x[1] - 2.5) ** 2 - 4]
    ceq = [x[0] + x[1] - 7]
    return np.array(c), np.array(ceq)


LB = [-5, -5]
UB = [5, 5]

InitialPopulation = [
    {'x': [-3, 2]},
    {'x': [1, -4]},
    {'x': [-2, 3]}
]

myProblem = Problem(Variables, Rastrigin, LB, UB, Rast_constr, x0=[100,50])
# InitialGuess1 = InitialGuess(np.array([0, 0]))
# InitialGuess2 = InitialGuess(np.array([1, 1]))



print(HGPSAL(myProblem)[5].exit)
