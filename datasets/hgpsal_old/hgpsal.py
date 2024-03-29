import numpy as np
from HGPSAL.HGPSAL.penalty import penalty2
from datasets.hgpsal_old.GeneticA import rGA
import copy
from datasets.hgpsal_old.hj import HJ
from HGPSAL.HGPSAL.AUX_Class.Problem_C import Problem
from HGPSAL.HGPSAL.AUX_Class.Alg_C import alg
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
    """HGPSAL - Hybrid Genetic Pattern Search Augmented Lagrangian
    Inputs:
    - Problem
        - Problem.Variables - Dimension of the problem
        - Problem.x0 - Initial guess (random if x0 is empty)
        - Problem.LB - Problem lower bounds
        - Problem.UB - Problem upper bounds
        - Problem.ObjFunction - Objective function
        - Problem.Constraints - Constraints function
    - options - Algorithm options dictionary (see below)
    - *args - Extra parameters for objective and constraint functions
    Options:
    Augmented Lagrangian:
        - lambda_min - Minimum Lagrange multiplier value for equality constraints
        - lambda_max - Maximum Lagrange multiplier value for equality constraints
        - teta_tol - Maximum Lagrange multiplier value for inequality constraints
        - miu0 - Initial penalty parameter
        - miu_min - Minimum penalty parameter
        - csi - Reduction factor for miu update
        - eta0 - Initial tolerance value
        - omega0, alfaw, alfa_eta, betaw, beta_eta, gama1, teta_miu - Other AL params
    rGA:
        - pop_size - Population size
        - elite_prop - Proportion of elite individuals
        - tour_size - Tournament size for selection
        - pcross - SBX crossover probability
        - icross - SBX crossover distribution index
        - pmut - Mutation probability
        - imut - Mutation distribution index
    HJ:
        - gama, delta, teta - HJ parameters
    Stopping criteria:
        - CPTolerance - Tolerance for best individual
        - CPGenTest - Gap for stopping test
        - delta_tol - HJ stopping tolerance
        - maxit - Maximum internal iterations
        - maxet - Maximum external iterations
        - max_objfun - Maximum objective evaluations
    Returns:
        - x - Solution
        - fx - Objective value
        - c - Constraint violation
        - ceq - Equality constraint violation
        - la - Augmented Lagrangian value
        - stats - Execution statistics
    """

    DefaultOpt = {'lambda_min': -1e12, 'lambda_max': 1e12, 'teta_tol': 1e12, 'miu_min': 1e-12, 'miu0': 1, 'csi': 0.5,
                  'eta0': 1, 'ffeas': 1, 'gps': 1, 'niu': 1.0, 'zeta': 0.001, 'epsilon1': 1e-4, 'epsilon2': 1e-8,
                  'suficient': 1e-4, 'method': 0,
                  'omega0': 1, 'alfaw': 0.9, 'alfa_eta': 0.9 * 0.9, 'betaw': 0.9, 'beta_eta': 0.5 * 0.9, 'gama1': 0.5,
                  'teta_miu': 0.5,
                  'pop_size': 40, 'elite_prop': 0.1, 'tour_size': 2, 'pcross': 0.9, 'icross': 20, 'pmut': 0.1,
                  'imut': 20,
                  'gama': 1, 'delta': 1, 'teta': 0.5, 'eta_asterisco': 1.0e-2, 'epsilon_asterisco': 1.0e-6,
                  'cp_ga_test': 0.1, 'cp_ga_tol': 1.0e-6, 'delta_tol': 1e-6, 'maxit': 100, 'maxet': 200,
                  'max_objfun': 20000}
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
    # verbose = Options['verbose'] if Options and 'verbose' in Options else DefaultOpt['verbose']
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
    if Problem.m == 0:
        alg.lmbd = []
    else:
        alg.lmbd = np.ones(Problem.m)
    if Problem.p == 0:
        alg.ldelta = []
    else:
        alg.ldelta = np.ones((1, Problem.p))
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
                np.append(alg.delta, gama)
            else:
                np.append(alg.delta, x0[j] * gama)
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
    stats.c = np.array(c) if len(np.array(c)) != 0 else []
    stats.ceq = np.array(ceq) if len(np.array(ceq)) != 0 else []
    global_search = True
    Opt = {}
    # initialize stats.history with column names
    stats.history = [('Iter', 'fx rGA', 'nf rGA', 'fx HJ', 'nf HJ')]
    while stats.exit <= maxet and stats.objfun <= max_objfun:
        # {1: {'Iter': None, 'fx rGA': None, 'nf rGA': None, 'fx HJ': None, 'nf HJ': None}}
        stats.exit += 1
        i = stats.exit
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
            # Opt['Verbosity'] = verbose
            x, fval, RunData = rGA(Probl, InitialPopulation, Opt, Problem, alg)
            stats.objfun += RunData.ObjFunCounter
            fx_ = fval
            nF_ = RunData.ObjFunCounter
        Opt['MaxIter'] = maxit
        Opt['MaxObj'] = max_objfun
        Opt['DeltaTol'] = alg.epsilon
        Opt['Theta'] = teta
        x, fval, Rundata = HJ(Probl, x, alg.delta, Options, Problem, alg)
        stats.objfun += Rundata.ObjFunCounter
        fx_H = fval
        nF_H = Rundata.ObjFunCounter
        stats.history.append((i, fx_, nF_, fx_H, nF_H))
        Value = penalty2(Problem, x, alg)
        c = Value.c
        ceq = Value.ceq
        fx = Value.fx
        la = Value.la
        stats.x.append(x)
        stats.fx.append(fx)
        if len(c):
            np.append(stats.c, c)
        if len(ceq):
            np.append(stats.ceq, ceq)
        if np.size(alg.lmbd) == 0 and np.size(alg.delta) == 0:
            break
        if ceq.any():
            max_i = np.max(np.abs(ceq))
        else:
            max_i = 0
        if c.any():
            v = np.maximum(np.max(c), np.max(alg.ldelta * np.abs(c)))
            print(v)
        else:
            v = 0
        norma_lambda = np.linalg.norm(alg.lmbd)
        norma_x = np.linalg.norm(x)
        alg.ldelta = np.maximum(lambda_min, np.minimum(np.maximum(0, alg.ldelta + c / alg.miu), lambda_max))
        if max_i <= alg.eta * (1 + norma_x) and v <= alg.eta * (1 + norma_lambda):
            if alg.epsilon < epsilon_asterisco and max_i <= eta_asterisco * (1 + norma_x) and v <= eta_asterisco * (
                    1 + norma_lambda) and global_search == False:
                stats.Message = 'HGPSAL: Tolerance of constraints violations satisfied.'
                print(stats.Message)
                break
            else:
                alg.lmbd = np.maximum(lambda_min, np.minimum(alg.lmbd + ceq / alg.miu, lambda_max))
                alg.alfa = np.minimum(alg.miu, gama1)
                alg.omega = alg.omega * pow(alg.alfa, betaw)
                alg.epsilon = alg.omega * mega_(alg.lmbd, alg.ldelta, alg.miu, teta_tol)
                alg.eta = alg.eta * pow(alg.alfa, beta_eta)
        else:
            alg.miu = max(min(alg.miu * csi, pow(alg.miu, teta_miu)), miu_min)
            alg.alfa = min(alg.miu, gama1)
            alg.omega = omega0 * pow(alg.alfa, alfaw)
            alg.epsilon = alg.omega * mega_(alg.lmbd, alg.ldelta, alg.miu, teta_tol)
            alg.eta = eta0 * pow(alg.alfa, alfa_eta)
            global_search = True
    if stats.exit > maxet:
        stats.Message = 'HGPSAL: Maximum number of external iterations reached.'
        print(stats.Message)
    if stats.objfun > max_objfun:
        stats.Message = 'HGPSAL: Maximum number objective function evaluations reached.'
        print(stats.Message)
    return x, fx, c, ceq, la, stats
# Variables = 2
#
#
# def Rastrigin(x):
#     f = 20 + x[0] ** 2 + x[1] ** 2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
#     return f
#
#
# def Rast_constr(x):
#     c = [(x[0] - 25) ** 2 + (x[1] - 25) ** 2 - 100]
#     ceq = [x[0] + x[1] - 7, x[1] * x[0] - 7]
#     # ceq = []
#     # c = []
#     return np.array(c), np.array(ceq)
#
#
# LB = [-5, -5]
# UB = [5, 5]
#
# def rosenbrock(x, a=1, b=100):
#     return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
#
# def constr(x):
#     c = [x[0]**2 + x[1]**2 - 1]
#     ceq = []
#     return np.array(c), np.array(ceq)
#
# myProblem = Problem(Variables, rosenbrock, LB, UB, constr, x0=[100, 50])
# InitialGuess1 = InitialGuess(np.array([0, 0]))
# InitialGuess2 = InitialGuess(np.array([1, 1]))
#
#
# print(HGPSAL(myProblem)[0])
# def func(x):
#     return x[0] ** 2 + x[1] ** 2
#
#
# def constraints(x):
#     c1 = [x[0] ** 2 + x[1] ** 2 - 1]
#     ceq1 = [x[0] + x[1] - 2]
#     return np.array(c1), np.array(ceq1)
#
#
# problem = Problem(2, func, [-5, -5], [5, 5], constraints)
#
# x, fx, c, ceq = HGPSAL(problem)[0:4]
# print(x, fx, c, ceq)
# def func(x):
#     return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
#
# def constraints(x):
#     h=np.empty(2)
#     h[0] = x[0] - x[1] + 1
#     h[1] = x[0] ** 2 + x[1] ** 2 - 2
#     return np.array([x[0] + x[1]]), np.array([x[0]**2 + x[1]**2 - 225])
#
# problem = Problem(2, func, [-5, -5], [5, 5], constraints)
# opt= {'maxit':100000,'max_objfun': 200000}
# x, fx, c, ceq = HGPSAL(problem, opt)[0:4]
# print(x)
# print(fx)
# print(c, ceq)
def func(x):
    return x[0] ** 2 + x[1] ** 2


def constraints(x):
    c1 = [x[0] ** 2 + x[1] ** 2 - 1]
    ceq1 = [x[0] + x[1] - 2]
    return np.array(c1), np.array(ceq1)


problem = Problem(2, func, [-5, -5], [5, 5], constraints)

# opt= {'maxit':100000,'max_objfun': 200000}
x, fx, c, ceq = HGPSAL(problem)[0:4]
print(x, fx)