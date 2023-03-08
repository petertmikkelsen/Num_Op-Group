from matplotlib import pyplot as plt
from case_studies import *
# import scipy bfgs
from scipy.optimize import minimize
import numpy as np
# Set numpy seed
np.random.seed(0)
# Function for that finds alpha that fulfils the strong Wolfe conditions
# Input is x, p, f, c1, c2, alpha0
def wolf_search(x, p, f, c1, c2, alpha0, df):
    def g(alpha):
        return f(x + alpha * p)
    def g_prime(alpha):
        return np.dot(df(x + alpha * p), p)
    l = 0
    u = alpha0
    while True:
        if g(u) > g(0) + c1 * u * g_prime(0) or (g(u) > g(l)):
            break
        if abs(g_prime(u)) < c2 * abs(g_prime(0)):
            return u
        if g_prime(u) > 0:
            break
        else:
            u = 2*u
    while True:
        a = (l + u) / 2
        if (g(a) > g(0) + c1 * a * g_prime(0)) or (g(a) > g(l)):
            u = a
        else:
            if abs(g_prime(a)) < c2 * abs(g_prime(0)):
                return a
            if g_prime(a) < 0:
                l = a
            else:
                u = a

# Implementation of the BFGS algorithm
# Input is x0, f, c1, c2, tol, maxiter
def bfgs(x0, f, df, c1, c2, tol, maxiter):
    # Initialize
    x = x0
    x_list = [x0]
    grad_norms = [np.linalg.norm(df(x0))]
    H = np.eye(len(x))
    # Iterate
    for i in range(maxiter):
        # Compute search direction
        p = -H @ df(x)
        # Compute step length
        alpha = wolf_search(x, p, f, c1, c2, 1.0, df)
        # Update x
        x_new = x + alpha * p
        # Compute gradient difference
        y = df(x_new) - df(x)
        # Compute position difference
        s = x_new - x
        # Update Hessian approximation
        sy = s.T @ y
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)
        H = H + ((sy + np.dot(y.T, np.dot(H, y))) / (sy**2)) * np.outer(s, s) - ((np.dot(H, np.outer(y, s)) + np.dot(np.outer(s, y.T), H)) / sy)
        # Update variables
        x = x_new
        x_list.append(x)
        grad_norms.append(np.linalg.norm(df(x)))
        # Check convergence
        if np.linalg.norm(df(x)) < tol:
            break
    return x_list, grad_norms


# Run BFGS algorithm on 50 random points given f and df
def run_bfgs(f, df, d, n_runs, c1, c2, tol, maxiter):
    x_list = []
    for i in range(n_runs):
        x0 = np.random.uniform(-100, 100, d)
        # print("Running BFGS")
        x_list.append(bfgs(x0, f, df, c1, c2, tol, maxiter))
        # print(f"Finished run {i+1}")
    return x_list

# plot convergence of three runs (best, worst, median) given x_list and x_opt
def plot_convergence(x_best, x_worst, x_median, d, fname):
    xo = x_opt(fname, d)
    print("Coompting norms")
    xs_best = [np.linalg.norm(x-xo) for x in x_best]
    xs_worst = [np.linalg.norm(x-xo) for x in x_worst]
    xs_median = [np.linalg.norm(x-xo) for x in x_median]
    print("Finished computing norms")
    plt.plot(xs_best, label='best')
    plt.plot(xs_worst, label='worst')
    plt.plot(xs_median, label='median')
    # Add legend
    plt.legend()
    # Add title
    plt.title(f'Convergence of BFGS for {fname}')
    # Add x and y labels
    plt.xlabel('Iteration')
    plt.ylabel('|x_k - x*|')
    # Show plot
    plt.show()

def run_and_plot_bfgs(f, df, d, n_runs, c1, c2, tol, maxiter):
    x_list = run_bfgs(f, df, d, n_runs, c1, c2, tol, maxiter)
    # Sort list according to number of iterations
    x_list_sorted = sorted(x_list, key=lambda x: len(x[0]))
    x_best = x_list_sorted[0][0]
    x_worst = x_list_sorted[-1][0]
    x_median = x_list_sorted[len(x_list_sorted)//2][0]
    plot_convergence(x_best, x_worst, x_median, d, f.__name__)

# Scipy bfgs
def run_scipy_bfgs(f,df,d,n_runs,max_iterations,epsilon):
    x_list = []
    for i in range(n_runs):
        x0 = np.random.uniform(-100, 100, d)
        xs=[]
        grad_norms=[]
        def logging_f(x):
            xs.append(x)
            grad_norms.append(np.maximum(np.linalg.norm(df(x)),10**(-5)*epsilon))
            return f(x)
        minimize(logging_f, x0, method="BFGS", jac=df, tol=epsilon,options={'maxiter':max_iterations, 'gtol':epsilon})
        x_list.append((np.array(xs), np.array(grad_norms)))
    return x_list

# Do the same for scipy BFGS
def run_and_plot_scipy_bfgs(f, df, d, n_runs, tol, maxiter):
    x_list = run_scipy_bfgs(f, df, d, n_runs, maxiter, tol)
    # Sort list according to number of iterations
    x_list_sorted = sorted(x_list, key=lambda x: len(x[0]))
    x_best = x_list_sorted[0][0]
    x_worst = x_list_sorted[-1][0]
    x_median = x_list_sorted[len(x_list_sorted)//2][0]
    plot_convergence(x_best, x_worst, x_median, d, f.__name__)



d = 50

# Run BFGS on f1
function_list = [
    (f1, df1),
    # (f2, df2),
    (f3, df3),
    (f4, df4),
    (f5, df5)
]
for (f, df) in function_list:
    d = 50
    if f.__name__ == 'f2':
        d = 2
    run_and_plot_bfgs(f, df, d, 50, 0.0001, 0.01, 1e-6, 1000)
    run_and_plot_scipy_bfgs(f, df, d, 50, 1e-6, 1000)
# run_and_plot_bfgs(f5, df5, d, 20, 0.0001, 0.01, 1e-1, 1000)
# run_and_plot_scipy_bfgs(f5, df5, d, 20, 1e-1, 1000)

# test = run_bfgs(f1, df1, 0.0001, 0.01, 1e-1, 100, d)


# print(test[0][1][:4])
# print(test[0][1][-1])

# plot_convergence(test[0][0], d, 'f1')

