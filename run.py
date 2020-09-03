#!/bin/env python
import logging
import data
import algo
import numpy as np


def play(
        comps, #companies for trade
        start_date = "2010-01-01", #history for computation
        ):

    logging.basicConfig(filename="multipliers.log",format="%(message)s", level=logging.INFO)
    for alg in algo.algs():
        a = alg(data="yahoo")
        a_date = a.load_last_params()
        try:
            date = a_date if a_date is not None else start_date
            X, _ = data.yahoo(comps, start=date) #can raise
            rewards = a.run(X)
            logging.info(f"{a.name}_{date}:{np.exp(sum(rewards))}")

        except RuntimeError as e:
            # if data from yfinance is empty
            print("aborting")


if __name__ == "__main__":
    play("AAPL MSFT SPY")
