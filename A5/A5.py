from case_studies import *


def BacktrackingLineSearch(x, d, f, df, method):
    alpha = 1
    rho = 0.5
    c = 0.1
    if method == 'SteepestDescent':
        alpha = 0.01
    while f(x + alpha * d) > f(x) + c * alpha * df(x) @ d:
        alpha = rho * alpha
    return alpha


def SteepestDescent(x0, f, df, tol, maxiter):
    x = x0
    x_list = []
    x_list.append(x)
    for i in range(maxiter):
        d = -df(x)
        alpha = BacktrackingLineSearch(x, d, f, df, 'SteepestDescent')
        x = x + alpha * d
        x_list.append(x)
        # check if the norm of the gradient is smaller than the tolerance
        if np.linalg.norm(df(x) - x_opt(f.__name__, 2)) < tol:
            break
    return x_list