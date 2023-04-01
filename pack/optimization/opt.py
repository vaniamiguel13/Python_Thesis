import numpy as np
from scipy.optimize import minimize

# define the objective function to minimize
def objective(x):
    return x[0]**2 + x[1]**2 + x[2]**2

# define the constraints
def constraint(x):
    return np.array([x[0] + x[1] + x[2] - 1,  # equality constraint
                     x[0]**2 + x[1]**2 - x[2]])  # inequality constraint

# set the initial guess
x0 = np.array([0.5, 0.5, 0.5])

# define the bounds for each variable
bounds = [(0, None), (0, None), (0, None)]

# define the constraints as a dictionary
constraints = {'type': 'eq', 'fun': constraint}

# run the optimization using SLSQP algorithm
res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

# print the optimized solution
print(res.x)