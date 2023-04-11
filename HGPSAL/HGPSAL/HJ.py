import numpy as np
from HGPSAL.HGPSAL.Time import tic, toc
from HGPSAL.HGPSAL.AUX_Class.Problem_C import Problem
def Projection(Problem, x):
    for i in range(Problem.Variables):
        if x[i] < Problem.LB[i]:
            x[i] = Problem.LB[i]

        if x[i] > Problem.UB[i]:
            x[i] = Problem.UB[i]

    return x


def ObjEval(Problem, x, *args):
    Problem.Stats.ObjFunCounter += 1
    return Problem, Problem.ObjFunction(x, *args)


def Exploratory_Moves(Problem, s, delta, e, x, fx, rho, *args):
    if rho > 0:
        Problem, min = ObjEval(Problem, x + s, *args)
        rho = fx - min
        s, rho, Problem = Coordinate_Search(Problem, s, delta, e, x, min, rho, *args)

    elif rho <= 0:
        s = 0
        rho = 0
        min = fx
        s, rho, Problem = Coordinate_Search(Problem, s, delta, e, x, min, rho, *args)

    return s, Problem


def Coordinate_Search(Problem, s, delta, e, x, min, rho, *args):

    for i in range(Problem.Variables):
        s1 = s + np.multiply(delta, e[:, i])
        x1 = x + s1
        x1 = Projection(Problem, x1)
        Problem, fx1 = ObjEval(Problem, x1, *args)

        if fx1 < min:
            rho = min - fx1
            min = fx1
            s = s1
        else:
            s1 = s - np.multiply(delta, e[:, i])
            x1 = x + s1
            x1 = Projection(Problem, x1)
            Problem, fx1 = ObjEval(Problem, x1, *args)

            if fx1 < min:
                rho = min - fx1
                min = fx1
                s = s1

    return s, rho, Problem


def HJ(Problem, x0, delta = None, Options = None, *args):

    DefaultOpt = {'MaxObj': 2000, 'MaxIter': 200, 'DeltaTol': 1.0e-6, 'Theta': 0.5}
    x0 = np.array(x0)

    if Problem is None:
        raise ValueError('HJ:AtLeastOneInput',
                         'HJ requests at least two inputs (Problem definition and initial approximation).')
    elif x0 is None:
        raise ValueError('HJ:AtLeastOneInput',
                         'HJ requests at least two inputs (Problem definition and initial approximation).')
    if delta is None:
        delta = np.ones(x0.shape)

    if Options is None:
        Options = {}

    MaxGenerations = Options['MaxIter'] if Options and 'MaxIter' in Options else DefaultOpt['MaxIter']
    MaxEvals = Options['MaxObj'] if Options and 'MaxObj' in Options else DefaultOpt['MaxObj']
    Delta = Options['DeltaTol'] if Options and 'DeltaTol' in Options else DefaultOpt['DeltaTol']
    Theta = Options['Theta'] if Options and 'Theta' in Options else DefaultOpt['Theta']

    tic()

    Problem.Stats.Algoritm = 'Hooke and Jeeves';
    Problem.Stats.Iterations = 0
    Problem.Stats.ObjFunCounter = 0

    x = Projection(Problem, x0)
    Problem, fx = ObjEval(Problem, x, *args)

    e = np.eye(Problem.Variables)

    s = 0
    rho = 0

    while np.linalg.norm(
            delta) > Delta and Problem.Stats.ObjFunCounter < MaxEvals and Problem.Stats.Iterations < MaxGenerations:
        s, Problem = Exploratory_Moves(Problem, s, delta, e, x, fx, rho, *args)
        x_trial = x + s
        x_trial = Projection(Problem, x_trial)
        Problem, fx1 = ObjEval(Problem, x_trial, *args)
        rho = fx - fx1
        if rho > 0:
            x = x_trial
            fx = fx1
        else:
            delta = delta * Theta
        Problem.Stats.Iterations = Problem.Stats.Iterations + 1

    if Problem.Stats.Iterations >= MaxGenerations:
        Problem.Stats.Message = 'HJ: Maximum number of iterations reached'
    if Problem.Stats.ObjFunCounter >= MaxEvals:
        Problem.Stats.Message = 'HJ: Maximum number of objective function evaluations reached'
    if np.linalg.norm(delta) <= Delta:
        Problem.Stats.Message = 'HJ: Stopping due to step size norm inferior to tolerance'

    print(Problem.Stats.Message)
    toc()
    RunData = Problem.Stats

    return x, fx, RunData

Variables = 2


def Rastrigin(x):
    f = 20 + x[0] ** 2 + x[1] ** 2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
    return f


LB = [-5, -5]
UB = [5, 5]

myProblem = Problem(Variables, Rastrigin, LB, UB)
# InitialGuess1 = InitialGuess(np.array([0, 0]))
# InitialGuess2 = InitialGuess(np.array([1, 1]))

InitialPopulation = [
    {'x': [-3, 2]},
    {'x': [1, -4]},
]


# print(HJ(myProblem, x0=[100, 100], Options=None))