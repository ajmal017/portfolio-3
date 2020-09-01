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
    '''can raise error if-
        1.empty folder
        2.no file that includes name in the file names
    '''
    files_list = glob.glob(os.path.join(path, "*"))
    #if not files_list: #empty folder
    #    raise(Exception("No files"))
    good_files = [f for f in files_list if name in f]
    last = max(good_files, key=os.path.getctime)
    return(last)

# save x
def save_x(
        x, #x the algorithm calculated
        algo #algorithm used to make that x
        ):
    today = datetime.date.today()
    day = to2str(str(today.day))
    month = to2str(str(today.month))
    today_str = f"{today.year}-{month}-{day}"
    name = algo + "_" + today_str + ".npy"
    file_path = os.path.join("data", "xs", name)
    np.save(file_path, x)


# load x
def load_last_x(
        algo #algorithm used to make that x
        ):
    path = os.path.join("data", "xs")
    last_x = find_last_file(algo, path)
    date = (((last_x.split('_'))[-1]).split('.'))[0] #creation date
    return(np.load(last_x, allow_pickle=True), date)


def to2str(s):
    return '0' + s if len(s) == 1 else s

# download yahoo data, load if possible
def yahoo_data(
        comps,
        start,
        end
        ):

    if not end:
        today = datetime.date.today()
        day = to2str(str(today.day))
        month = to2str(str(today.month))
        end = f"{today.year}-{month}-{day}"
    name = '_'.join([start, end])

    # find pickles from start
    #if such file exists, load and concat with new data
    try:
        path = os.path.join("data", "yahoo")
        last_data = find_last_file(start, path)
        last_end = last_data.split('_')[-1] # remove path and start

        prev = pd.read_pickle(last_data)

        #check if same companies
        prev_comps = set(prev["Close"].columns)
        now_comps = set(comps.split(" "))
        if now_comps != prev_comps: #not the same data
            raise(Exception("Download new data"))

        if last_end == end: #the file we need already downloaded
            print("yahoo data exists")
            return(prev)
        else: #got partial history
            last_end_datetime = datetime.datetime.strptime(last_end, "%Y-%m-%d")
            new_start = str((last_end_datetime + datetime.timedelta(days=1)).date()) #last_end + one day
            x = yf.download(comps, start=new_start, end=end)
            X = pd.concat([prev, x])

    except:
        X = yf.download(comps, start=start, end=end)

    X.to_pickle(os.path.join(path, name))
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
    timestamp_tmp = {key:int(val) for key, val in zip(["year","month", "day"],start.split('-'))}
    today = pd.Timestamp(**timestamp_tmp)
    X = X[X["Date"] >= today]

    X.columns = ["date", "symbol", "close"]
    X["date"] = X["date"].apply(lambda x: str(x.date()))
    if X.empty:
        raise(RuntimeError("X is empty"))

    X, symbols = df_to_clean_np(X, "date", "symbol")
    return(X, symbols)
