import numpy as np
# if this fails, you need to put the case_studies.py file in the same folder
from case_studies import *

from scipy.optimize import minimize


# These are the example optimizers you should evaluate this week.
# These are optimizers implemented in scipy.
# they take as first 2 or 3 arguments the function f, its gradient df and sometimes its hessian Hf.
# the next parameters are all the same: x0 is the starting point, max_iterations the stopping criterion for iterations
# and epsilon the precision tolerance to be reached.
# Note: epsilon is interpreted slightly differently across algorithms, and some algorithms might not reach the tolerance
# and stop early.
def scipy_bfgs(f, df, x0, max_iterations, epsilon):
    xs = []
    grad_norms = []
    
    def logging_f(x):
        xs.append(x)
        grad_norms.append(np.maximum(np.linalg.norm(df(x)), 10 ** (-5) * epsilon))
        return f(x)
    
    minimize(logging_f, x0, method="BFGS", jac=df, tol=epsilon, options={'maxiter': max_iterations, 'gtol': epsilon})
    return np.array(xs), np.array(grad_norms)


def scipy_newton(f, df, Hf, x0, max_iterations, epsilon):
    xs = []
    grad_norms = []
    
    def logging_f(x):
        xs.append(x)
        grad_norms.append(np.maximum(np.linalg.norm(df(x)), 10 ** (-5) * epsilon))
        return f(x)
    
    minimize(logging_f, x0, method="Newton-CG", jac=df, hess=Hf, tol=epsilon,
             options={'maxiter': max_iterations, 'xtol': epsilon})
    return np.array(xs), np.array(grad_norms)


def scipy_trust_region(f, df, Hf, x0, max_iterations, epsilon):
    xs = []
    grad_norms = []
    
    def logging_f(x):
        xs.append(x)
        grad_norms.append(np.maximum(np.linalg.norm(df(x)), 10 ** (-5) * epsilon))
        return f(x)
    
    minimize(logging_f, x0, method="trust-exact", jac=df, hess=Hf, tol=epsilon, options={'maxiter': max_iterations})
    return np.array(xs), np.array(grad_norms)


# example usage of the algorithms
# the output is a list of points evaluated on the function as well as the gradient norms at that point
# this algorithms has the first three arguments functions for function value, gradient and Hessian.
# For the 5 functions, those are named f1-f5 etc and cna be found in the case_studies.py file
x0 = np.ones(2)
xs, grad_norms = scipy_trust_region(f4, df4, Hf4, x0, 1000, 1.e-10)

# the optimal point for a given function and dimensionality is stored in the package as well for at least 15 decimals precision
optimal = x_opt("f4", 2)
print("final solution point:", xs[-1])
print("distance of x from optimum", np.linalg.norm(xs[-1] - optimal))
print("number of function evaluations:", len(grad_norms))
print("final function value:", f4(xs[-1]))
print("final gradient norm:", grad_norms[-1])

############################################################################################################
import matplotlib.pyplot as plt

def plot_function(func, dfunc, Hfunc, d, func_name):
    x0 = np.random.uniform(low=-10000, high=10000, size=(d,))
    # x0 = np.array([-10000]*d)
    print(f"x0 = {x0}")
    xt, grad_normt = scipy_trust_region(func, dfunc, Hfunc, x0, 10000, 1.e-10)
    xn, grad_normn = scipy_newton(func, dfunc, Hfunc, x0, 10000, 1.e-10)
    xb, grad_normb = scipy_bfgs(func, dfunc, x0, 10000, 1.e-10)
    
    xb_dist = np.zeros((len(xb), d))
    for i in range(len(xb)):
        xb_dist[i] = np.linalg.norm(xb[i] - x_opt(func_name, d))

    xt_dist = np.zeros((len(xt), d))
    for i in range(len(xt)):
        xt_dist[i] = np.linalg.norm(xt[i] - x_opt(func_name, d))

    xn_dist = np.zeros((len(xn), d))
    for i in range(len(xn)):
        xn_dist[i] = np.linalg.norm(xn[i] - x_opt(func_name, d))
    

    plt.plot(np.arange(len(xt_dist)), xt_dist, label=f"Trust Region {func_name}", color = "red")
    plt.plot(np.arange(len(xn_dist)), xn_dist, linestyle='dashed', label=f"Newton {func_name}", color = "blue")
    plt.plot(np.arange(len(xb_dist)), xb_dist, linestyle='dotted', label=f"BFGS {func_name}", color = "green")
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Distance from optimum")
    plt.title(f"Distance from optimum for {func_name}")
    plt.legend()
    return

plot_function(f2, df2, Hf2, 2, "f2")
plt.show()
