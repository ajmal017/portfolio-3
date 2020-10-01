# Online Portfolio Selection

Implementation of Online Gradient Ascent and Online Newton Step based on [Logarithmic Regret Algorithms for Online Convex Optimization](paper/Logarithmic Regret Algorithms for Online Convex Optimization.pdf).

Also implemented a system to automate the process on 'real time' data from Yahoo Finance.


## System

The system downloads data from Yahoo Finance (using yfinance) and computes what portion of the wealth to put on what stock.

Define the companies to compute the return on in ```run.py``` and execute the file, the wealth multipliers will be logged to ```multipliers.log```.

The downloaded data and the parameters for the algorithms is saved in ```/data```.

### Dependencies

* cvxpy

* numpy

* pandas

* tqdm

* yfinance

* matplotlib (for ploting, not necessary otherwise)

## Tests

Algorithms tested on [S&P 500 dataset](https://www.kaggle.com/camnugent/sandp500?select=all_stocks_5yr.csv).

### Stocks Visualization
![](imgs/SP.png)

### Return & Regret

Return is in log scale so while > 0: the algorithm makes money

![](imgs/SP_results.png)
