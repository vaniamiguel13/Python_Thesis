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

        x, fx, c, ceq = HGPSAL(problem)[0:4]

        np.testing.assert_almost_equal(fx, 1.1573, decimal=3)
        np.testing.assert_array_almost_equal(x, [0.7606, 0.7606], decimal=4)
        np.testing.assert_almost_equal(c, [0.157], decimal=3)
        np.testing.assert_almost_equal(ceq, [-0.478], decimal=3)

if __name__ == '__main__':
    unittest.main()