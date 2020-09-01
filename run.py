#!/bin/env python
import logging
import data
import algo
import datetime
import numpy as np


def play(
        comps, #companies for trade
        start_date = "2010-01-01", #history for computation
        ):

    logging.basicConfig(filename="multipliers.log",format="%(message)s", level=logging.INFO)
    logging.info(f"run_date-{data.today_str()}")
    try: 
        # oga
        oga_x, date = data.load_last_yahoo_x("oga_x")
        #if such x exists, change the starting date for computation
        if date is not None:
            start_date = date
        # ons
        ons_x, _ = data.load_last_yahoo_x("ons_x")
        ons_A, _ = data.load_last_yahoo_x("ons_A")
        ons_b, _ = data.load_last_yahoo_x("ons_b")

        X, _ = data.yahoo(comps, start_date)

        # oga
        next_oga_x, oga_rewards = algo.oga(X=X, x=oga_x)
        logging.info(f"oga-{np.exp(sum(oga_rewards))}")
        data.save_x(next_oga_x, "oga_x")

        # ons
        (next_ons_x, next_ons_A, next_ons_b), ons_rewards = algo.ons(X=X, x=ons_x, A=ons_A, b=ons_b)
        logging.info(f"ons-{np.exp(sum(ons_rewards))}")
        data.save_x(next_ons_x, "ons_x")
        data.save_x(next_ons_A, "ons_A")
        data.save_x(next_ons_b, "ons_b")

        #save training

    except RuntimeError as e:
        # if data from yfinance is empty
        print(e)


if __name__ == "__main__":
    play("AAPL MSFT SPY")
