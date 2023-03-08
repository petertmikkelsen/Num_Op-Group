import numpy as np
from numpy.linalg import eigvals
from case_studies import *


# Define the function to check if the Hessian is always positive (semi-)definite
def is_hessian_definite(H: np.ndarray) -> bool:
    return np.all(np.linalg.eigvals(H) > 0)


def conjugate_gradients(Q, g, eta, epsilon):
    x = np.zeros_like(g)
    r = g
    d = -r
    delta_new = np.dot(r, r)
    delta_0 = delta_new
    eps_squared = epsilon ** 2
    cg_steps = 0
    while delta_new > eps_squared * delta_0:
        Qd = Q @ d
        alpha = -np.dot(r, d) / np.dot(d, Qd)
        x = x + eta * alpha * d
        r = r + eta * alpha * Qd
        delta_old = delta_new
        delta_new = np.dot(r, r)
        beta = delta_new / delta_old
        d = r + beta * d
        cg_steps += 1
    
    return x, cg_steps


def inexact_newton_cg(x0, f, df, Hf, max_iter=100, eta=0.1, epsilon=1e-6, nk_option=1):
    x = x0
    x_pos_list = [x]
    nb_iter_newton = []
    nb_iter_cg = []
    for i in range(max_iter):
        # Compute current gradient and Hessian
        dfx = df(x)
        Hfx = Hf(x)
        # Compute current approximation of inverse Hessian
        if nk_option == 1:
            epsilon_k = epsilon / np.linalg.norm(dfx)
        elif nk_option == 2:
            nk = min(0.5, np.linalg.norm(dfx))
            epsilon_k = epsilon / nk
        elif nk_option == 3:
            nk = 0.5 * min(0.5, np.sqrt(np.linalg.norm(dfx)))
            epsilon_k = epsilon / nk
        elif nk_option == 4:
            # nk = epsilon
            nk = 0.001
            epsilon_k = epsilon / nk
        pk, cg_steps = conjugate_gradients(Hfx, dfx, eta, epsilon_k)
        
        # Perform backtracking line search to find step size
        alpha = 1
        while f(x + alpha * pk) > f(x) + alpha * eta * dfx @ pk:
            alpha /= 2
        # alpha = 0.1
        
        # Update position with step size
        x = x + alpha * pk
        
        # Store in a list the results
        x_pos_list.append(x)
        nb_iter_newton.append(i)
        nb_iter_cg.append(cg_steps)

        # Debugging
        # Print distance between x and optimal point
        
        # Check stopping criteria
        # if np.linalg.norm(alpha * pk) < epsilon:
        #     break
        if np.linalg.norm(df(x)) < epsilon:
            break
    
    return x_pos_list, nb_iter_newton, nb_iter_cg


# Test the algorithms on each function in the list
# fix the random seed
np.random.seed(42)

# parameters
nb_run = 50
dim = 50
maxiter = 1000
epsilon = 1e-3
functions = [f1, f4, f5]
dfuntions = [df1, df4, df5]
ddfuntions = [Hf1, Hf4, Hf5]

# Loop until reach the number of runs and store the results in lists
list_f1 = []
list_f4 = []
list_f5 = []
lists = [list_f1, list_f4, list_f5]

for i in range(nb_run):
    k = 0
    for f, df, ddf in zip(functions, dfuntions, ddfuntions):
        print("Run: ", i, "Function: ", f.__name__)
        # if is_hessian_definite(Hf(x0)):
        #     print(f'Hessian is positive definite for {f.__name__}')
        # else:
        #     print(f'Hessian is not positive definite for {f.__name__}')
        #     break
        x0 = np.random.uniform(-100, 100, dim)
        x_approx_newton, iters_approx_newton, cg_steps = inexact_newton_cg(x0, f, df, ddf, maxiter, 0.1, epsilon, 2)
        lists[k].append([x_approx_newton, iters_approx_newton, cg_steps])
        k += 1

import time

# Loop over each function in the list and run the inexact Newton algorithm with the linear schedule
# for i in range(len(functions)):
#     print(f'Function {functions[i].__name__}')
#     start_time = time.time()
#     results = []
#     for j in range(nb_run):
#         x0 = np.random.uniform(-100, 100, dim)
#         x_pos_list, nb_iter_newton, nb_iter_cg = inexact_newton_cg(x0, functions[i], dfuntions[i], ddfuntions[i],
#                                                                    maxiter)
#         results.append([x_pos_list, nb_iter_newton, nb_iter_cg])
#     end_time = time.time()
#     elapsed_time = end_time - start_time
    
#     # Compute and print the average number of iterations and CG steps
#     avg_nb_iter_newton = np.mean([len(r[1]) for r in results])
#     avg_nb_iter_cg = np.mean([sum(r[2])/len(r[0]) for r in results])
#     print(f'Average number of Newton iterations: {avg_nb_iter_newton}')
#     print(f'Average number of CG steps: {avg_nb_iter_cg}')
    
#     # Print the average elapsed time
#     print(f'Average elapsed time: {elapsed_time / nb_run}\n')

# Plot the number of iterations for each function
import matplotlib.pyplot as plt

# for i in range(len(lists)):
#     plt.figure()
#     plt.title("Function: " + functions[i].__name__)
#     plt.plot([x[1] for x in lists[i]], label="Approximate Newton")
#     plt.plot([x[2] for x in lists[i]], label="Conjugate Gradients")
#     plt.legend()
#     plt.xlabel("Run")
#     plt.ylabel("Number of iterations")
#     plt.grid()
#     plt.show()

# Plot the distance between our final point and the optimal point
# for i in range(len(lists)):
#     plt.figure()
#     plt.title("Function: " + functions[i].__name__)
#     plt.plot([np.linalg.norm(x[0][-1] - x[0][0]) for x in lists[i]], label="Approximate Newton")
#     plt.legend()
#     plt.xlabel("Run")
#     plt.ylabel("Distance")
#     plt.grid()
#     plt.show()

# Plot distance between x and optimal point for the best worst and median run
# in terms of number of iterations
# for i in range(len(lists)):
#     f_name = functions[i].__name__
#     plt.figure()
#     plt.title("Function: " + f_name)
#     # Initialize x_optimal
#     x_optimal = x_opt(f_name, dim)
#     # Sort the runs by number of iterations
#     lists[i].sort(key=lambda x: len(x[0]))
#     # Create list of distances from optimal point for worst
#     best = lists[i][0][0]
#     # Create list of distances from optimal point for best
#     worst = lists[i][-1][0]
#     # Create list of distances from optimal point for median
#     median = lists[i][int(len(lists[i]) / 2)][0]
#     # Plot the distances for the worst, best and median runs
#     print(best)
#     print(x_optimal)
#     plt.plot([np.linalg.norm(x_optimal - x) for x in best], label="Best")
#     plt.plot([np.linalg.norm(x_optimal - x) for x in worst], label="Worst")
#     plt.plot([np.linalg.norm(x_optimal - x) for x in median], label="Median")

#     # plt.plot([np.linalg.norm( - x[0][0]) for x in lists[i]], label="Approximate Newton")

#     plt.legend()
#     plt.xlabel("# of iterations")
#     plt.ylabel("x - x_optimal")
#     plt.grid()
#     plt.show()

for i in range(len(lists)):
    f_name = functions[i].__name__
    plt.figure()
    plt.title("Function: " + f_name + ", nk = 0.001")
    # Initialize x_optimal
    x_optimal = x_opt(f_name, dim)
    # Sort the runs by number of iterations
    lists[i].sort(key=lambda x: len(x[0]))
    # Create list of distances from optimal point for worst
    worst = lists[i][-1][0]
    # Create list of distances from optimal point for best
    best = lists[i][0][0]
    # Create list of distances from optimal point for median
    median = lists[i][int(len(lists[i]) / 2)][0]
    # Plot the distances for the worst, best and median runs
    plt.plot([np.linalg.norm(x_optimal - x) for x in best], label="Best")
    plt.plot([np.linalg.norm(x_optimal - x) for x in worst], label="Worst")
    plt.plot([np.linalg.norm(x_optimal - x) for x in median], label="Median")

    plt.legend()
    plt.xlabel("# of iterations")
    plt.ylabel("x - x_optimal")
    plt.grid()
    plt.show()


# # use the time it package to measure the time taken by each algorithm
# from timeit import default_timer as timer
#
# # Define the function to be used to test the algorithms
# def test_algorithm_time(f, df, Hf, x0, dim, maxiter, epsilon):
#     start = timer()
#     x_approx_newton, iters_approx_newton = inexact_newton_cg(x0, f, df, Hf, epsilon, maxiter)
#     end = timer()
#     time_approx_newton = end - start
#
#     start = timer()
#     x_cg, iters_cg = conjugate_gradients(x0, df, Hf, epsilon, maxiter)
#     end = timer()
#     time_cg = end - start
#
#     return time_approx_newton, time_cg
#
# # Plot the time taken by each algorithm for each function
# for i in range(len(lists)):
#     plt.figure()
#     plt.title("Function: " + functions[i].__name__)
#     plt.plot([test_algorithm_time(functions[i], dfuntions[i], ddfuntions[i], np.random.uniform(-100, 100, dim), dim, maxiter, epsilon)[0] for x in range(nb_run)], label="Approximate Newton")
#     plt.plot([test_algorithm_time(functions[i], dfuntions[i], ddfuntions[i], np.random.uniform(-100, 100, dim), dim, maxiter, epsilon)[1] for x in range(nb_run)], label="Conjugate Gradients")
#     plt.legend()
#     plt.xlabel("Run")
#     plt.ylabel("Time (s)")
#     plt.grid()
#     plt.show()
#
#
