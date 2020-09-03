import os, glob
import datetime

def find_last_file(
        path, #folder path- string
        name #name date- string
        ):
    #can raise error if-
    #    1.empty folder
    #    2.no file that includes name in the file name

    files_list = glob.glob(os.path.join(path, "*"))
    good_files = [f for f in files_list if name in f]
    last = max(good_files, key=os.path.getctime)
    return(last)


def to2str(s):
    return '0' + s if len(s) == 1 else s


def today_str():
    today = datetime.date.today()
    day = to2str(str(today.day))
    month = to2str(str(today.month))
    today_str = f"{today.year}-{month}-{day}"
    return(today_str)


def plus_day(
        day #string
        ):
    day_time = datetime.datetime.strptime(day, "%Y-%m-%d")
    day_after = str((day_time + datetime.timedelta(days=1)).date())
    return(day_after)


class testUnit:
    def __init__(self):
        print("utils test unit")
        print(f"to2str('1') == '01': {to2str('1') == '01'}")
        print(f"plus_day('2020-08-31') == '2020-09-01': {plus_day('2020-08-31') == '2020-09-01'}")
