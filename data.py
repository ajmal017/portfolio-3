import numpy as np
import pandas as pd
import random
import glob, os
import yfinance as yf
import datetime

random.seed(42)


# test unit
def test_data(T=100):
    s1 = [2**(i%2) for i in range(T)]
    s2 = s1[::-1]
    X = np.vstack((np.array(s1),np.array(s2)))
    return(X)


def df_to_clean_dfs(
        df, #dataframe
        date="Date",
        sym="Symbol",
        k=-1
                   ):
    df = df.dropna()
    df_groups = df.groupby([date])
    dfs = list(df_groups)
    dfs = [x[1] for x in dfs] #[0] is the date, [1] is the dataframe

    #use only companies that always exist
    companies = set(dfs[0][sym].unique())
    for i in dfs:
        companies = set(i[sym]) & companies #sets intersection

    #take subset to make tractable
    if k != -1:
        companies = random.sample(companies, k = k)

    df = df[df[sym].isin(companies)] #remove non-permanent companies
    df_groups = df.groupby([date])
    dfs = list(df_groups)
    dfs = [x[1] for x in dfs]

    return dfs


# concat dataframes to big numpy array
# each row is a company
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


def SP(k=-1, T=-1):
    X = pd.read_csv(os.path.join("data", "SP", "SP.csv"))
    X = X[["date", "close", "Name"]]
    #X["date"] = X["date"].map(lambda x : x.split()[0]) #so no 00:00:00 in date
    X, symbols = df_to_clean_np(X, "date", "Name", k)
    if T != -1:
        return(X[:,:T], symbols)
    else:
        return(X, symbols)


def find_last_file(name, path):
    files_list = glob.glob(os.path.join(path, "*"))
    if not files_list: #empty folder
        raise(Exception("No files"))
    good_files = [f for f in files_list if name in f]
    last = max(good_files, key=os.path.getctime)
    return(last)


# download / load yahoo data
def yahoo_data(
        comps,
        start,
        end
        ):

    if not end:
        today = datetime.date.today()
        end = f"{today.year}-{today.month}-{today.day}"
    name = '_'.join([start, end])

    # find csvs from start
    #if such file, load and concat with new data
    try:
        folder = os.path.join("data", "yahoo")
        last_path = find_last_file(start, folder)
        last_end = last_path.split('_')[-1] # remove path and start

        prev = pd.read_pickle(last_path)

        #check if same companies
        prev_comps = set(prev["Close"].columns)
        now_comps = set(comps.split(" "))
        if now_comps != prev_comps: #not the same data
            raise(Exception("Download new data"))

        if last_end == end: #the file we need already downloaded
            return(prev)
        else: #got partial history
            last_end_datetime = datetime.datetime.strptime(last_end, "%Y-%m-%d")
            new_start = str((last_end_datetime + datetime.timedelta(days=1)).date()) #last_end + one day
            x = yf.download(comps, start=new_start, end=end)
            X = pd.concat([prev, x])

    except:
        X = yf.download(comps, start=start, end=end)

    X.to_pickle(os.path.join(folder, name))
    return(X)

def yahoo(
        comps, #space delimited symbols
        start, #start date
        end="" #end date
        ):
    X = yahoo_data(comps,start,end)

    # change X to be like the SP data
    X = X["Close"]
    X = X.stack().reset_index()
    X.columns = ["date", "symbol", "close"]
    X["date"] = X["date"].apply(lambda x: str(x.date()))

    X, symbols = df_to_clean_np(X, "date", "symbol")
    return(X, symbols)
