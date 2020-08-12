import numpy as np
import pandas as pd
import random
import glob, os

random.seed(42)


def test_data(T=100):
    s1 = [2**(i%2) for i in range(T)]
    s2 = s1[::-1]
    X = np.vstack((np.array(s1),np.array(s2)))
    return(X)


# clean dataframe to get only ever-presence companies
# returns numpy array (companies x time)
def clean_df(X, date="Date", sym="Symbol", close="Close", k=-1):
    X = X.dropna(subset=[date, sym, close])
    X_groups = X.groupby([date])
    Xs = list(X_groups)
    Xs = [x[1] for x in Xs] #[0] is the date, [1] is the dataframe

    #use only companies that always exist
    companies = set(Xs[0][sym].unique())
    for i in Xs:
        companies = set(i[sym]) & companies #symbol sets intersection
    #take subset to make tractable
    if k != -1:
        companies = random.sample(companies, k = k)

    X = X[X[sym].isin(companies)] #remove non-permanent companies
    X_groups = X.groupby([date])
    Xs = list(X_groups)
    Xs = [x[1] for x in Xs]

    #concat to big numpy array
    Xs[0] = Xs[0].sort_values(by=sym)
    X = Xs[0][close].to_numpy(copy=True)[:,None]
    comps = Xs[0][sym].to_numpy()
    for i in range(1,len(Xs)):
        Xs[i] = Xs[i].sort_values(by=sym)
        X = np.hstack((X,Xs[i][close].to_numpy(copy=True)[:,None]))

    return(X)


def NIFTY(k=-1, T=-1):
    #X = pd.concat(map(pd.read_csv, glob.glob(os.path.join("data", "NIFTY", "*.csv"))))
    X = pd.read_csv(os.path.join("data", "NIFTY", "NIFTY.csv"))
    X = clean_df(X, "Date", "Symbol", "Close", k)
    if T != -1:
        return(X[:,:T])
    else:
        return(X)


def SP(k=-1, T=-1):
    X = pd.read_csv(os.path.join("data", "SP", "SP.csv"))
    #X["date"] = X["date"].map(lambda x : x.split()[0]) #so no 00:00:00 in date
    X = clean_df(X, "date", "Name", "close", k)
    if T != -1:
        return(X[:,:T])
    else:
        return(X)
