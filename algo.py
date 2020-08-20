import cvxpy as cp
import numpy as np
from tqdm import tqdm


# project vector to the simplex
def project_simplex(y):
    p = cp.Variable(y.shape)
    objective = cp.Minimize(cp.sum_squares(p - y))
    constraints = [cp.sum(p) == 1, p >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status != cp.OPTIMAL:
        ex = f"Can't project to the simplex: {problem.status}"
        raise cp.error.SolverError(ex)
    return(p.value)
    

# project vector to simplex on A
def project_A(A, y):
    p = cp.Variable(y.shape)
    objective = cp.Minimize(cp.quad_form((p - y), A))
    constraints = [cp.sum(p) == 1, p >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status != cp.OPTIMAL:
        ex = f"Can't project to A: {problem.status}"
        raise cp.error.SolverError(ex)
    return(p.value)


# project to simplex given data
def best_hindsight(Rs):
    p = cp.Variable((Rs.shape[0],1))
    objective = cp.Maximize(cp.sum(cp.log(Rs.T @ p)))
    constraints = [cp.sum(p) == 1, p >= 0]
    problem = cp.Problem(objective, constraints)
    #https://www.cvxpy.org/tutorial/advanced/index.html#solvers
    #needs better than ECOS
    problem.solve(solver="SCS")
    if problem.status != cp.OPTIMAL:
        ex = f"Can't find best in hindsight: {problem.status}"
        raise cp.error.SolverError(ex)
    return(problem.value)


# find best fixed distribution in hindsight
def hindsight(X):
    T = X.shape[1]
    rewards = []

    for t in tqdm(range(1,T), desc="hindsight"):
        r_t = X[:,t] / X[:,t-1]
        r_t = r_t[:,None]

        #hindsight
        try:
            Rs = np.hstack((Rs,r_t))
        except:
            Rs = r_t

        rewards += [best_hindsight(Rs)]

    return(rewards)


# online gradient ascent
def oga(X):
    T = X.shape[1]
    d = X.shape[0] #dimension
    x = np.ones([d,1])/d #uniform probablity
    rewards = []

    for t in tqdm(range(1,T), desc="oga"):
        r_t = X[:,t] / X[:,t-1]
        r_t = r_t[:,None]

        multiplier = r_t.T @ x
        grad = r_t / multiplier
        eta = 1 / (d * np.sqrt(t))
        y = x + eta * grad #+ for ascent
        x = project_simplex(y)

        rewards += [np.log(multiplier)[0][0]]

    return(x,rewards)


# online newton step
def ons(X, beta=2):
    T = X.shape[1]
    d = X.shape[0]
    x = np.ones([d,1])/d
    A = np.zeros([d,d])
    b = np.zeros([d,1])
    rewards = []

    for t in tqdm(range(1,T), desc="ons"):
        r_t = X[:,t] / X[:,t-1]
        r_t = r_t[:,None]

        multiplier = r_t.T @ x
        grad = r_t / multiplier
        hess = grad @ grad.T
        A += hess
        b += hess @ x + (1 / beta) * grad #+ for ascent
        x = project_A(A, np.linalg.pinv(A) @ b)

        rewards += [np.log(multiplier)[0][0]]

    return(x, rewards)
