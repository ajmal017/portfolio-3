#!/bin/env python
import data
import algo
import datetime
import numpy as np


def play(
        comps, #companies for trade
        start_date = "2010-01-01", #history for computation
        ):

    try: 
        #first run will raise ValueError

        # oga
        oga_x, start_date = data.load_last_x("oga_x")
        # ons
        ons_x, _ = data.load_last_x("ons_x")
        ons_A, _ = data.load_last_x("ons_A")
        ons_b, _ = data.load_last_x("ons_b")

        X, _ = data.yahoo(comps, start_date)
        next_oga_x, oga_mult = algo.oga(X=X, x=oga_x)
        print(f"oga multiplier: {np.exp(oga_mult)}")
        next_ons_x, ons_mult = algo.ons(X=X, x=ons_x, A=ons_A, b=ons_b)
        print(f"ons multiplier: {np.exp(ons_mult)}")

    except ValueError as e: 
        #first run- so download data from some point in the past
        #           and find xs
        print(e)
        old_X, _ = data.yahoo(comps, start_date)
        next_oga_x, oga_rewards = algo.oga(old_X)
        print(f"oga multiplier from {start_date}: {np.exp(sum(oga_rewards))}")
        (next_ons_x, next_ons_A, next_ons_b), ons_rewards = algo.ons(old_X)
        print(f"ons multiplier from {start_date}: {np.exp(sum(ons_rewards))}")
    except RuntimeError as e:
        # if data from yfinance is empty
        print(e)
        return()
    
    #save training
    ##oga
    data.save_x(next_oga_x, "oga_x")
    ##ons
    data.save_x(next_ons_x, "ons_x")
    data.save_x(next_ons_A, "ons_A")
    data.save_x(next_ons_b, "ons_b")


if __name__ == "__main__":
    play("AAPL MSFT SPY")
