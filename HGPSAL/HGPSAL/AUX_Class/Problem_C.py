class Problem:
    def __init__(self, Variables, ObjFunction, LB, UB, Constraints=None, x0=None, Variables_C = 0, Variables_Ceq = 0):
        self.Variables = Variables
        self.ObjFunction = ObjFunction
        self.LB = LB
        self.UB = UB
        self.Constraints = Constraints
        self.x0 = x0
        self.L_c = Variables_C
        self.L_ceq = Variables_Ceq
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
        self.N1Front = []
        self.NFronts = []
        self.Evaluations = 0
        self.Iterations = 0
        self.StopFlag = ''
        self.Message = ''
        self.ObjFunCounter = 0
        self.ConCounter = 0
        self.GenCounter = 0
        self.Algorithm = None
        self.Best = []
        self.Worst = []
        self.Mean = []
        self.Std = []
        self.Time = 0

