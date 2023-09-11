import numpy as np
from HGPSAL.HGPSAL.Time import tic, toc
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

    """Generalized Hooke and Jeeves algorithm.

    Inputs:

    - Problem:
        - Problem.ObjFunction: Objective function
        - Problem.Variables: Dimension
        - Problem.LB: Lower bounds
        - Problem.UB: Upper bounds
    - x0: Initial guess
    - delta: Initial step size

    Returns:
        - x: Solution
        - fx: Objective value
        - RunData: Execution statistics

    The algorithm searches for a minimizer x of the
    objective function Problem.ObjFunction by taking
    exploratory steps from the initial guess x0 using
    step size delta.

    """
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

    Problem.Stats.Algoritm = 'Hooke and Jeeves'
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

from HGPSAL.HGPSAL.AUX_Class.Problem_C import *
def fobj(x):
    """
    Sphere Function

    Input:
    x (list of floats) : the point at which to evaluate the sphere function

    Output:
    (float) : the value of the sphere function at x
    """
    return sum(xi ** 2 for xi in x)
# Variables = 2
# ObjFunction = fobj
# LB = [-15, -15]
# UB = [15, 15]
# Problem = Problem(Variables, ObjFunction, LB, UB)  # Assuming ProblemClass is the class defining your problem
#
#
# x0 = np.array([0, 0])
#
# Options = {'MaxIter': 300}
#
# x, fx, S = HJ(Problem, x0, np.array([30, 40]), Options)
