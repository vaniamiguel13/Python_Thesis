import numpy as np


class Val:
    def __init__(self, fx, c, ceq):
        self.fx = fx
        self.c = c
        self.ceq = ceq
        self.la = []


def penalty2(Problem, x, alg):
    # Compute objective function
    Value = Val(ObjEval(Problem, x), ConsEval(Problem, x)[0], (ConsEval(Problem, x))[1])
    # Compute all constraints
    term1 = sum(alg.lmbd * Value.ceq)
    term2 = sum(Value.ceq ** 2)
    term3 = sum(np.array(np.maximum(0, alg.ldelta + Value.c / alg.miu)) ** 2 - np.array(alg.ldelta) ** 2)
    Value.la = Value.fx + term1 + term2 / (2 * alg.miu) + alg.miu * term3 / 2

    return Value


def ObjEval(Problem, x, *args):
    ObjValue = Problem.ObjFunction(x)
    return ObjValue


def ConsEval(Problem, x, *args):
    c, ceq = Problem.Constraints(x, *args)
    return c, ceq
