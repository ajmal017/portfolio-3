#!/bin/env python
import logging
import data
import algo
import utils
import numpy as np


def play(
        comps: str,  # companies for trade
        default_start: str="2010-01-01",  # history for computation
):

    try:
        logging.basicConfig(
            filename="multipliers.log", format="%(message)s", level=logging.INFO
        )

        old, new = data.yahoo_dates(default_start)
        old_dates = utils.dates_str(old)

        X, symbols, real_dates = data.yahoo(
            comps, start=new[0], end=new[1]
        )  # can raise
        new_dates = utils.dates_str(real_dates)

        for alg in algo.algs():
            a = alg(data="yahoo", old_dates=old_dates, new_dates=new_dates)
            rewards = a.run(X)
            logging.info(
                f"{a.name}:{new_dates}:{np.exp(sum(rewards))}:{X.shape[1]}-days"
            )

    except RuntimeError as e:
        # if data from yfinance is empty
        print("aborting")


if __name__ == "__main__":
    play("AAPL MSFT SPY")
