#!/usr/bin/env python

#Code#
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import cvxpy as cp
import random

random.seed(42)

#Title#
#  Portfolio Selection

#Markdown#
# Paper: Logarithmic Regret Algorithms for Online Convex Optimization
# 
# $W_T = W_{T-1} r_{T-1}^T x_{T-1} = W_1 \prod_t r_t^T x^t$
# 
# $r_t(i) = \frac{\text{price of stock i at } t}{\text{price of stock i at } t-1}$
# 
# $f_t(x) = \log\left( r_t^T x \right)$
# 
# $f_t$ are rewards and not losses, so the regret defined as:
# 
# $\text{Regret}_T = \max_{x \in \Delta_n} \sum_t f_t(x) - \sum_t f_t(x_t)$
# 
# We'll use the following no-regret algorithms:
# 
# ogd - online gradient descent
# 
# ons - online newton step


#Code#
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


def ogd(X):
    T = X.shape[1]
    d = X.shape[0]
    x = np.ones([d,1])/d
    rewards = []

    for t in tqdm(range(1,T), desc="ogd"):
        r_t = X[:,t] / X[:,t-1]
        r_t = r_t[:,None]

        multiplier = r_t.T @ x
        grad = r_t / multiplier
        eta = 1 / (d * np.sqrt(t))
        y = x + eta * grad #+ for ascent
        x = project_simplex(y)

        rewards += [np.log(multiplier)[0][0]]

    return(x,rewards)



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


def plot_two_plots(hind,ogd,ons):
    T = len(hind)

    # stock multiplier
    plt.subplot(121)
    plt.plot(hind, color = 'b', label="best in hindsight")
    plt.plot(ogd, color = 'r', label="ogd")
    plt.plot(ons, color = 'c', label="ons")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("log of stock multiplier")

    # regret
    plt.subplot(122)
    plt.plot([(h - o) for h,o in zip(hind,ogd)], color = 'r', label="ogd")
    plt.plot([(h - o) for h,o in zip(hind,ons)], color = 'c', label="ons")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("regret")

    plt.show()

#Title#
## Test unit 

#Markdown#
# Example from the Elad's book
# 
# stock 1 $= \left( 1, 2, 1, ... \right)$
# 
# $r_t(1) = \left( 2, 0.5, 2, 0.5, ... \right)$
# 
# stock 2 $= \left( 2, 1, 2, ... \right)$
# 
# $r_t(2) = \left( 0.5, 2, 0.5, 2, ... \right)$
# 
# $p_{opt} = \left( 0.5, 0.5 \right)$

#Code#
def test():
    T = 300
    s1 = [2**(i%2) for i in range(T)]
    s2 = s1[::-1]
    X = np.vstack((np.array(s1),np.array(s2)))

    bh_reward = hindsight(X)
    x_ogd, ogd_reward = ogd(X)
    x_ons, ons_reward = ons(X)

    print("final x_ogd = ",x_ogd)
    print("final x_ons = ",x_ons)
    print("should be", 0.5 * np.ones([2,1]))

    plot_two_plots(bh_reward,np.cumsum(ogd_reward), np.cumsum(ons_reward))

#test()



#Title#
## Real data

#Markdown#
# S\&P 500 from 8/2/2013 - 7/2/2018
# 
# https://www.kaggle.com/camnugent/sandp500?select=all_stocks_5yr.csv

#Code#
# load data
X = pd.read_csv("SP.csv")
#X["date"] = X["date"].map(lambda x : x.split()[0]) #so no 00:00:00 in date

X_groups = X.groupby(["date"])
Xs = list(X_groups)
Xs = [x[1] for x in Xs] #[0] is the date, [1] is the dataframe

#use only companies that always exists
companies = set(Xs[0]["Name"].unique())
for i in Xs:
    companies = set(i["Name"]) & companies #sets intersection
#take subset to make tractable
#k = len(companies)
#companies = random.sample(companies, k = k)

X = X[X["Name"].isin(companies)]
X_groups = X.groupby(["date"])
Xs = list(X_groups)
Xs = [x[1] for x in Xs]

#concat to big numpy array
col = "close"
Xs[0] = Xs[0].sort_values(by="Name")
X = Xs[0][col].to_numpy(copy=True)[:,None]
comps = Xs[0]["Name"].to_numpy()
for i in range(1,len(Xs)):
    Xs[i] = Xs[i].sort_values(by="Name")
    X = np.hstack((X,Xs[i][col].to_numpy(copy=True)[:,None]))

#Title#
### Stocks Visualization

#Code#
plt.plot(X.T)
plt.title(f"{len(companies)} stocks from S&P 500, 8/2/2013 - 7/2/2018")
plt.show()

#Title#
### Compute & Plot

#Code#
bh_reward = hindsight(X)
x_ogd, ogd_reward = ogd(X)
x_ons, ons_reward = ons(X)

print("final x_ogd = ",x_ogd)
print("final x_ons = ",x_ons)

plot_two_plots(bh_reward,np.cumsum(ogd_reward), np.cumsum(ons_reward))
