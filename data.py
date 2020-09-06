import numpy as np
import pandas as pd
import random
import os
import yfinance as yf
import datetime
import utils

random.seed(42)


def df_to_clean_dfs(
        df, #dataframe
        date="Date", #date column name
        sym="Symbol", #symbol column name
        k=-1
        ):
    def df_to_list(df):
        df_groups = df.groupby([date])
        dfs = list(df_groups)
        dfs = [x[1] for x in dfs] #[0] is the date, [1] is the dataframe
        return(dfs)

    df = df.dropna()
    dfs = df_to_list(df)

    #use only companies that always exist
    companies = set(dfs[0][sym].unique())
    for i in dfs:
        companies = set(i[sym]) & companies #sets intersection

    if k != -1:
        #take subset to make tractable
        companies = random.sample(companies, k = k)

    df = df[df[sym].isin(companies)] #remove non-permanent companies
    dfs = df_to_list(df)

    return(dfs)


# concat dataframes to big numpy array
# each row is a company
# each column is a date
def dfs_to_np(
        dfs,
        sym="Symbol"
        ):
    dfs[0] = dfs[0].sort_values(by=sym)
    dfs[0].drop(sym, axis=1, inplace=True)
    X = dfs[0].to_numpy(copy=True)
    for i in range(1,len(dfs)):
        dfs[i] = dfs[i].sort_values(by=sym)
        dfs[i].drop(sym, axis=1, inplace=True)
        X = np.hstack((X,dfs[i].to_numpy(copy=True)))

    return(X)


# clean dataframe to get only ever-presence companies
# returns numpy array (companies x time)
def df_to_clean_np(df, #dataframe
                   date="Date", #df column for date
                   sym="Symbol", #df column for company symbols
                   k=-1 #subset of companies to take
                  ):
    tmp = df_to_clean_dfs(df,date,sym,k)
    symbols = tmp[0][sym]
    dfs = [df.drop(date, axis=1) for df in tmp]
    X = dfs_to_np(dfs, sym)
    return(X, symbols)




########################## test data
def test_data(T=100):
    s1 = [2**(i%2) for i in range(T)]
    s2 = s1[::-1]
    X = np.vstack((np.array(s1),np.array(s2)))
    return(X)


########################## SP
def SP(
        k=-1,
        T=-1
        ):
    X = pd.read_csv(os.path.join("data", "SP", "SP.csv"))
    X = X[["date", "close", "Name"]]
    #X["date"] = X["date"].map(lambda x : x.split()[0]) #so no 00:00:00 in date
    X, symbols = df_to_clean_np(X, "date", "Name", k)
    if T != -1:
        return(X[:,:T], symbols)
    else:
        return(X, symbols)


########################## yahoo 
# download yahoo data, load if possible
def yahoo_data(
        comps,
        start,
        end
        ):
    name = '_'.join([start, end])
    path = os.path.join("data", "yahoo")
    close = "Close"
    try:
        # see if data from start to end exists
        last_data = utils.find_last_file(path, name) #can raise
        prev = pd.read_pickle(last_data)

        #check if same companies
        prev_comps = set(prev[close].columns)
        now_comps = set(comps.split(' '))
        if now_comps != prev_comps: #not the same data
            raise(ValueError("download new data"))

        return(prev[close])

    except ValueError as e:
        # just download
        df = correct_download(comps, start=start, end=end)
        if df.shape[0] < 2: #algorithms needs atleast yesterday's and today's data
            raise(RuntimeError("not enough data from yfinance"))

        df.to_pickle(os.path.join(path, name))

        return(df[close])


def correct_download(
        comps,
        start, #string
        end #string
        ):
    X = yf.download(comps, start=start, end=end)
    start_pd = pd.Timestamp(start)
    return(X[X.index >= start_pd])


def yahoo(
        comps, #space delimited symbols
        start, #start date
        end="" #end date
        ):
    if not end:
        end = utils.today_str()

    try:
        ydata = yahoo_data(comps, start, end)
    except RuntimeError as e:
        print(e)
        raise(e)

    symbols = list(ydata.columns)
    X = ydata.to_numpy().T

    return(X, symbols)


###################### test unit
class testUnit:
    def __init__(self):
        # test_data
        print(f"test_data: {np.sum(test_data(2) == np.array([[1,2],[2,1]])) == 4}")
        comps = "AAPL SPY MSFT"
        start = "2000-01-01"
        end = "2000-02-02"
        end_o = utils.plus_day(end)
        # yahoo_data
        try:
            yahoo_data(comps, start=start, end=start)
            print(f"yahoo_data empty: False")
        except RuntimeError as e:
            print(f"yahoo_data empty: True")
        a = yahoo_data(comps, start=start, end=end)
        b = correct_download(comps, start=start, end=end)["Close"]
        print(f"yahoo_data: {a.equals(b)}")

        c = yahoo_data("AMZN AAPL", start=start, end=end)
        print(f"yahoo_data comps: {not a.equals(c)}")

