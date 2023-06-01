
#coding: utf-8

'''
预处理：将原始csv文件中的f，Z,model等列单独提取
'''

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import LabelEncoder


def read_csv_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col=0)
    if "freq" in df.columns:
        df["freq"] = df["freq"].apply(lambda x: np.array([float(c.strip("[]")) for c in x.split(", ")]))
    if "Z" in df.columns:
        df["Z"] = df["Z"].apply(lambda x: np.array([complex(c.strip("[]").replace(" ", "")) for c in x.split(", ")]))
    return df

def resize_array(arr, size):
    n = arr.shape[0]
    if size > n:
        x_old = np.arange(n)
        x_new = np.linspace(0, n-1, size)
        f = interp1d(x_old, arr, kind='linear')
        arr_resized = f(x_new)
    elif size < n:
        x_old = np.arange(n)
        x_new = np.linspace(0, n-1, size)
        f = interp1d(x_old, arr, kind='linear')
        arr_resized = f(x_new)
    else:
        arr_resized = arr
    return arr_resized

def process_data_element(freq, Z, basis, func):
    x = func(Z)
    f = interp1d(freq, x, fill_value="extrapolate")
    return f(basis)

def preprocess_data(file_path, size):
    df = read_csv_data(file_path)
    df["f"] = df.apply(lambda x: resize_array(x.freq, size), axis=1)
    df["zreal"] = df.apply(lambda x: process_data_element(x.freq, x.Z, x.f, np.real), axis=1)
    df["zimag"] = df.apply(lambda x: process_data_element(x.freq, x.Z, x.f, np.imag), axis=1)
    return df

def unwrap_dataframe(df):
    df2 = pd.DataFrame(columns=["No.", "freq", "zreal", "zimag"])
    for i in np.arange(df.shape[0]):
        f, zreal, zimag = df[["f", "zreal", "zimag"]].loc[i]
        No_x = np.tile(i, f.size)
        df_ = pd.DataFrame(data=(No_x, f, zreal, zimag), index=["No.", "freq", "zreal", "zimag"]).T
        df2 = pd.concat([df2, df_], ignore_index=True)
    return df2

def group_dataframe(df):
    df_grouped = df.groupby("No.").apply(lambda x: pd.Series({"all": list(zip(x.freq, x.zreal, x.zimag))})).reset_index()
    return df_grouped

def tuples_to_string(tuples):
    return ",".join([f"{d[0]:.9f},{d[1]:.9f},{d[2]:.9f}" for d in tuples])

def rename_dataframe_columns(df):
    df["all"] = df.apply(lambda x: tuples_to_string(x["all"]), axis=1)
    df = df['all'].str.split(',', expand=True).add_prefix('A')
    groups = [df.columns[i:i+3] for i in range(0, len(df.columns), 3)]
    for i, group in enumerate(groups):
        for j, col in enumerate(group):
            new_col_name = f"{['freq', 'Zreal', 'Zimag'][j]}-{i+1}"
            df = df.rename(columns={col: new_col_name})
    return df


csv_path="D:/ChatEIS/原始数据/AutoEIS/test_data.csv"

eis_df= read_csv_data(csv_path)
A=preprocess_data(csv_path, 20)
B=unwrap_dataframe(A)
C=group_dataframe(B)
E=rename_dataframe_columns(C)
E.to_csv('eis_test_data_20-test.csv', index=True)

print('OK')
