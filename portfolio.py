#!/usr/bin/env python

#Code#
import data
import algo
import plot

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
# oga - online gradient descent
# 
# ons - online newton step



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
def test_unit():
    T = 100
    X = test_data(T)

    bh_reward = algo.hindsight(X)
    x_oga, oga_reward = algo.oga(X)
    x_ons, ons_reward = algo.ons(X)

    print("final x_oga = ",x_oga)
    print("final x_ons = ",x_ons)
    print("should be", 0.5 * np.ones([2,1]))

    plot.plot_two_plots(bh_reward,np.cumsum(oga_reward), np.cumsum(ons_reward))

#test_unit()



#Title#
## Real data

#Title#
### NIFTY-50

#Markdown#
# 1/1/2000 - 13/7/2020
# 
# https://www.kaggle.com/rohanrao/nifty50-stock-market-data?select=BPCL.csv
#Code#
nifty = data.NIFTY()

#Title#
#### Stocks Visualization
#Code#
plot.vis(nifty)

#Title#
#### Compute & Plot
#Code#
bh_reward, oga_reward, ons_reward = algo.compute_summary(nifty)

plot.plot_two_plots(bh_reward, oga_reward, ons_reward)
###################################################################################

#Title#
### S\&P 500

#Markdown#
# 8/2/2013 - 7/2/2018
# 
# https://www.kaggle.com/camnugent/sandp500?select=all_stocks_5yr.csv
#Code#
sp = data.SP(20)

#Title#
#### Stocks Visualization
#Code#
plot.vis(sp)

#Title#
#### Compute & Plot
#Code#
bh_reward, oga_reward, ons_reward = algo.compute_summary(sp)

plot.plot_two_plots(bh_reward, oga_reward, ons_reward)
###################################################################################
