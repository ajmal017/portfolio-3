#!/usr/bin/env python

# Code#
import data
import algo
import plot
import numpy as np

# Title#
#  Portfolio Selection

# Markdown#
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
# oga - online gradient descent
#
# ons - online newton step


# Title#
## Test unit

# Markdown#
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

# Code#
def test_unit(T=100):
    X = data.test_data(T)

    bh_reward = algo.hindsight(X)
    x_oga, oga_rewards = algo.oga(X)
    x_ons, ons_rewards = algo.ons(X)

    print("final x_oga = ", x_oga)
    print("final x_ons = ", x_ons)
    print("should be", 0.5 * np.ones([2, 1]))

    algs_reward = [[np.cumsum(oga_rewards), "oga"], [np.cumsum(ons_rewards), "ons"]]
    plot.plot_regret(bh_rewards, algs_rewards)


# test_unit()

###################################################################################

# Title#
## Real data


###################################################################################
# Title#
### S\&P 500

# Markdown#
# 8/2/2013 - 7/2/2018
#
# https://www.kaggle.com/camnugent/sandp500?select=all_stocks_5yr.csv
# Code#
sp, comps = data.SP(k=3)

# Title#
#### Stocks Visualization
# Code#
plot.vis(sp, "S&P")

# Title#
#### Compute & Plot
# Code#
oga_x, oga_rewards = algo.oga(sp)
ons_x, ons_rewards = algo.ons(sp)

plot.plot_multiplier([[np.cumsum(oga_rewards), "oga"], [np.cumsum(ons_rewards), "ons"]])
###################################################################################
# Title#
### Yahoo Finance

# Markdown#
# Take the S\&P 500 data from yahoo, not from kaggle, to produce same results
#
# 8/2/2013 - 7/2/2018
# Code#
comps = " ".join(comps)
print(comps)
y, _ = data.yahoo(comps, start="2013-02-08", end="2014-02-08")
y, _ = data.yahoo(comps, start="2014-02-08", end="2018-02-07")


# Title#
#### Stocks Visualization
# Code#
plot.vis(y, "Yahoo Finance")

# Title#
#### Compute & Plot
# Code#
oga_x, oga_rewards = algo.oga(y)
ons_x, ons_rewards = algo.ons(y)

plot.plot_multiplier([[np.cumsum(oga_rewards), "oga"], [np.cumsum(ons_rewards), "ons"]])
