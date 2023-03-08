import matplotlib.pyplot as plt
import numpy as np

from case_studies import *


def SteepestDescent(x0, f, df, A, dims, c1, rho, tol, maxiter):
    x = x0
    x_list = []
    x_list.append(x)

    beta = 1
    M = np.identity(dims) - A.T @ np.linalg.inv(A @ A.T) @ A
    for i in range(maxiter):
        p = -M @ df(x)
        alpha = backtrack(x, f, df, p, c1, rho, beta)
        x = x + alpha * p
        beta = alpha/rho
        x_list.append(x)
        # check if the norm of the gradient is smaller than the tolerance
        if np.linalg.norm(df(x) - x_opt(f.__name__, 2)) < tol:
            break
    return x_list

def backtrack(x, f, df, p_k, c1, rho, beta_k):
  alpha = beta_k
  while f(x+alpha*p_k) > f(x) + c1 * alpha * p_k @ df(x):
    alpha = rho * alpha
  return alpha

def is_hessian_definite(H: np.ndarray) -> bool:
    return np.all(np.linalg.eigvals(H) > 0)

def Newton(x0, f, df, A, ddf, tol, maxiter):
    x = x0
    x_list = []
    x_list.append(x)
    for i in range(maxiter):
        if is_hessian_definite(ddf(x)):
            B = ddf(x)
        else:
            eig_vals, eig_vecs = np.linalg.eig(ddf(x))
            B = np.dot(abs(eig_vals), np.outer(eig_vecs, eig_vecs.T))
        first_term = np.block([[B, A.T], [A, 0]])
        second_term = np.concatenate([-df(x), 0])
        solution = np.linalg.solve(first_term, second_term)
        p_k = solution[2:]
        d_lambda = solution[:2]
        alpha = BacktrackingLineSearch(x, p_k, f, df, 'Newton')
        x = x + alpha * p_k
        x_list.append(x)
        # check if the norm of the gradient is smaller than the tolerance
        if np.linalg.norm(df(x) - x_opt(f.__name__, 2)) < tol:
            break
    return x_list


def BacktrackingLineSearch(x, d, f, df, method):
    alpha = 1
    rho = 0.5
    c = 0.1
    if method == 'SteepestDescent':
        alpha = 0.01
    while f(x + alpha * d) > f(x) + c * alpha * df(x) @ d:
        alpha = rho * alpha
    return alpha


def plot(x_list, f, x0, title=""):
    # plot the error and the number of iteration for the best, worst and median
    x_best = x_list[0]
    x_worst = x_list[-1]
    x_median = x_list[len(x_list) // 2]
    xopt = x_opt(f.__name__, 2)
    
    # plot the function contour
    plt.figure()
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    plt.contour(X, Y, Z, 60)
    
    # convert the np.array to a list
    x_best = x_best.tolist()
    x_worst = x_worst.tolist()
    x_median = x_median.tolist()
    
    # plot the path
    plt.plot([x[0] for x in x_best], [x[1] for x in x_best], 'g', label='Best', linewidth=1)
    plt.plot([x[0] for x in x_worst], [x[1] for x in x_worst], 'r', label='Worst', linewidth=1)
    plt.plot([x[0] for x in x_median], [x[1] for x in x_median], 'b', label='Median', linewidth=1)
    plt.plot(xopt[0], xopt[1], 'm*', label='Optimal')
    # show the initial point for the best, worst and median
    plt.plot(x_best[0][0], x_best[0][1], 'k*')
    plt.plot(x_worst[0][0], x_worst[0][1], 'k*')
    plt.plot(x_median[0][0], x_median[0][1], 'k*', label='Initial')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    
    # limit the axis to the contour
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    plt.show()


def plot_convergence(xlist, f, xopt, title=''):
    # plot the error and the number of iteration for the best, worst and median
    x_best = xlist[0]
    x_worst = xlist[-1]
    x_median = xlist[len(xlist) // 2]
    xopt = x_opt(f.__name__, 2)
    
    plt.figure()
    plt.plot(np.linalg.norm(x_best - xopt, axis=1), 'g', label='Best')
    plt.plot(np.linalg.norm(x_worst - xopt, axis=1), 'r', label='Worst')
    plt.plot(np.linalg.norm(x_median - xopt, axis=1), 'b', label='Median')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()


if __name__ == '__main__':
    # fix the random seed
    np.random.seed(42)
    
    # parameters
    nb_run = 50
    dim = 2
    maxiter = 500
    epsilon = 1e-9
    
    functions = [f1, f2, f3, f4, f5]
    dfuntions = [df1, df2, df3, df4, df5]
    ddfuntions = [Hf1, Hf2, Hf3, Hf4, Hf5]
    
    x0 = np.array([100, -100])
    A = np.array([[1,0],[0,-1]])
    b=0
    
    for n in range(len(functions)):
        print("Function: ", functions[n].__name__)
        # Steepest Descent and Newton
        #x_list_sd = []
        x_list_newton = []
        for i in range(nb_run):
            #x_list = SteepestDescent(x0[i], functions[n], dfuntions[n], epsilon, maxiter)
            #x_list_sd.append(x_list)
            x_list = Newton(x0, functions[n], dfuntions[n], A, ddfuntions[n], epsilon, maxiter)
            x_list_newton.append(x_list)
        
        # sort the list depend on the error
        #x_list_sd.sort(key=lambda x: np.linalg.norm(x[-1] - x_opt(functions[n].__name__, dim)))
        x_list_newton.sort(key=lambda x: np.linalg.norm(x[-1] - x_opt(functions[n].__name__, dim)))
        #print("Steepest Descent: ", x_list_sd[0][-1], "Median Error: ",
              #np.linalg.norm(x_list_sd[0][-1] - x_opt(functions[n].__name__, dim)), "Median Iteration: ",
              #len(x_list_sd[len(x_list_sd) // 2]))
        print("Newton: ", x_list_newton[0][-1], "Median Error: ",
              np.linalg.norm(x_list_newton[0][-1] - x_opt(functions[n].__name__, dim)), "Median Iteration: ",
              len(x_list_newton[len(x_list_newton) // 2]))
        
        #print("Length of x_list_sd: ", len(x_list_sd))
        print("Length of x_list_newton: ", len(x_list_newton))
        
        # if the function is not the 4 or 5, we are not converging
        if functions[n].__name__ != 'f4' and functions[n].__name__ != 'f5':
            #x_list_sd = np.array(x_list_sd)
            x_list_newton = np.array(x_list_newton)
            # Plot path for the worst, best and median case
            #plot(x_list_sd, functions[n], x0, 'Steepest Descent for ' + functions[n].__name__)
            plot(x_list_newton, functions[n], x0, 'Newton for ' + functions[n].__name__)
        
        # Plot convergence for the worst, best and median case
        #plot_convergence(x_list_sd, functions[n], x_opt(functions[n].__name__, dim),
                         #'Steepest Descent convergence for ' + functions[n].__name__)
        plot_convergence(x_list_newton, functions[n], x_opt(functions[n].__name__, dim),
                         'Newton convergence for ' + functions[n].__name__)
        