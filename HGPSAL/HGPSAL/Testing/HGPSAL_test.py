import unittest
from HGPSAL.HGPSAL.AL_HGPSAL import HGPSAL
from HGPSAL.HGPSAL.AUX_Class.Problem_C import Problem
import numpy as np


class TestHGPSAL(unittest.TestCase):

    def test_unconstrained(self):
        def func(x):
            return x[0] ** 2 + x[1] ** 2

        def constraints(x):
            return np.array([]), np.array([])

        problem = Problem(2, func, [-5, -5], [5, 5], constraints)

        x, fx = HGPSAL(problem)[0:2]

        np.testing.assert_almost_equal(fx, 0, decimal=5)
        np.testing.assert_array_almost_equal(x, [0, 0], decimal=5)

    def test_bounded(self):
        def func(x):
            return (x[0] - 2) ** 2 + (x[1] - 1) ** 2

        def constraints(x):
            return np.array([]), np.array([])

        problem = Problem(2, func, [0, 0], [5, 5], constraints)

        x = HGPSAL(problem)[0]
        fx = HGPSAL(problem)[1]

        np.testing.assert_almost_equal(fx, 0, decimal=5)
        np.testing.assert_array_almost_equal(x, [2, 1], decimal=5)

    def test_constrained(self):
        def func(x):
            return x[0] ** 2 + x[1] ** 2

        def constraints(x):
            c1 = [x[0] ** 2 + x[1] ** 2 - 1]
            return np.array(c1), np.array([])

        problem = Problem(2, func, [-5, -5], [5, 5], constraints)

        x, fx, c = HGPSAL(problem)[0:3]

        np.testing.assert_almost_equal(fx, 0, decimal=5)
        np.testing.assert_array_almost_equal(x, [0, 0], decimal=5)
        np.testing.assert_almost_equal(c, [-1], decimal=5)

    def test_eq_constrained(self):
        def func(x):
            return x[0] ** 2 + x[1] ** 2

        def constraints(x):
            c1 = [x[0] ** 2 + x[1] ** 2 - 1]
            ceq1 = [x[0] + x[1] - 2]
            return np.array(c1), np.array(ceq1)

        problem = Problem(2, func, [-5, -5], [5, 5], constraints)
        opt = {'maxit': 100000, 'max_objfun': 200000}
        x, fx, c, ceq = HGPSAL(problem, opt)[0:4]

        np.testing.assert_almost_equal(fx, 0.125, decimal=3)
        np.testing.assert_array_almost_equal(x, [0.25, 0.25], decimal=2)
        np.testing.assert_almost_equal(c, [-0.87], decimal=2)
        np.testing.assert_almost_equal(ceq, [-1.49999999], decimal=2)

    def test_Rast(self):
        def func(x):
            z = 20 + x[0] ** 2 + x[1] ** 2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
            return z

        def constraints(x):
            c=[]
            c.append(-x[0] - 5)  # x0 >= -5
            c.append(x[0] - 5)  # x0 <= 5
            c.append(-x[1] - 5)  # x1 >= -5
            c.append(x[1] - 5)  # x1 <= 5
            ceq = []

            return np.array(c), np.array(ceq)

        problem = Problem(2, func, [-5, -5], [5, 5], constraints)

        x, fx, c, ceq = HGPSAL(problem)[0:4]

        np.testing.assert_almost_equal(fx, 0.0, decimal=1)
        np.testing.assert_array_almost_equal(x, [0., 0.], decimal=4)
        np.testing.assert_almost_equal(c, [-5., -5., -5., -5.], decimal=3)
        np.testing.assert_almost_equal(ceq, [], decimal=3)

    def test_Rosenbrock(self):
        def func(x):
            return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        def constraints(x):
            g1 = 2 * x[0] + 2 * x[1] - 10  # inequality constraint
            g2 = x[0] ** 2 + x[1] ** 2 - 1  # inequality constraint

            c = [g1, g2]

            h1 = x[0] + 3 * x[1] - 5  # equality constraint

            ceq = [h1]

            return np.array(c), np.array(ceq)

        opt = {'maxit': 100000, 'max_objfun': 200000}
        problem = Problem(2, func, [-5, -5], [5, 5], constraints, x0=[0, 0])

        x, fx, c, ceq = HGPSAL(problem, opt)[0:4]

        np.testing.assert_almost_equal(fx, 0.0, decimal=1)
        np.testing.assert_array_almost_equal(x, [1., 1.], decimal=1)
        # np.testing.assert_almost_equal(c, [700.], decimal=3)
        # np.testing.assert_almost_equal(ceq, [], decimal=3)

    def himmelblau(self):
        def func(x):
            return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

        def constraints(x):
            c = []
            ceq = []
            return np.array(c), np.array(ceq)

        opt = {'maxit': 100000, 'max_objfun': 200000}
        problem = Problem(2, func, [-5, -5], [5, 5], constraints, x0=[0, 0])

        x, fx, c, ceq = HGPSAL(problem, opt)[0:4]

        np.testing.assert_almost_equal(fx, 0.0, decimal=1)
        np.testing.assert_array_almost_equal(x, [3., 2.], decimal=1)


if __name__ == '__main__':
    unittest.main()
