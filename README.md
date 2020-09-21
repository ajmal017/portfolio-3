# Online Portfolio Selection

Implementation of Online Gradient Ascent and Online Newton Step based on "Logarithmic Regret Algorithms for Online Convex Optimization" (in ```/paper```),

## Tests

Tested on S&P 500 dataset:

https://www.kaggle.com/camnugent/sandp500?select=all_stocks_5yr.csv

### Stocks Visualization
![](imgs/SP.png)

### Return & Regret

Return is in log scale so while > 0: the algorithm makes money

#### S&P
![](imgs/SP_results.png)

## System

The system downloads data from yahoo finance (using yfinance) and computes what portion of the wealth to put on what stock.

Run ```./run.py``` once a day, the wealth multipliers are in ```multipliers.log```.

## Notebook

```py2ng.py``` converts ```.py``` to ```.ipynb```.
