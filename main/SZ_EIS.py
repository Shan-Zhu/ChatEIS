# coding: utf-8

'''使用ML分类EIS，挑出最可能的三种，给出概率，给出参数拟合，画图'''


import matplotlib.pyplot as plt
from impedance.preprocessing import readFile
from impedance.preprocessing import ignoreBelowX
from impedance.models.circuits import CustomCircuit
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from scipy.interpolate import interp1d
from scipy.linalg import inv
from scipy.optimize import curve_fit, basinhopping
from impedance.visualization import plot_nyquist


def eis_ML_select(sample_features):
    loaded_model = XGBClassifier(learning_rate=0.05, max_depth=6, n_estimators=500)
    loaded_model.load_model('XGB_model.model')
    # 在测试集上进行预测并计算准确率
    eis_type = loaded_model.predict(sample_features)
    # 在测试集上进行预测并计算准确率
    eis_type_prob = loaded_model.predict_proba(sample_features)
    # 获取概率最高的三个类别的索引
    top_3_indices = eis_type_prob.argsort()[:, -3:][:, ::-1]
    # 获取概率最高的三个类别的标签值和概率值
    eis_type_top3 = [(loaded_model.classes_[i], eis_type_prob[j][i]) for j, i in enumerate(top_3_indices)]
    return eis_type, eis_type_top3

def get_model_and_guess(eis_type):
    if eis_type == 0:
        circuit_model = "L_0-R_0-p(R_1,CPE_0)"
        initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0]
    elif eis_type == 1:
        circuit_model = "L_0-R_0-p(R_1,CPE_0)-p(R_2,CPE_1)"
        initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif eis_type == 2:
        circuit_model = "L_0-R_0-p(R_1,CPE_0)-p(R_2,CPE_1)-p(R_3,CPE_2)"
        initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif eis_type == 3:
        circuit_model = "p(R_0,C_0)-G_0-G_1"
        initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif eis_type == 4:
        circuit_model = "p(R_0,C_0)-p(R_1,C_1)-p(R_2,CPE_0)-p(R_3,CPE_1)"
        initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif eis_type == 5:
        circuit_model = "p(R_0,CPE_0)-p(R_1,CPE_1)"
        initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif eis_type == 6:
        circuit_model = "p(R_0,CPE_0)-p(R_1,CPE_1)-p(R_2,CPE_2)"
        initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif eis_type == 7:
        circuit_model = "p(R_0,CPE_0)-p(R_1,CPE_1)-p(R_2,CPE_2)-p(R_3,CPE_3)"
        initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif eis_type == 8:
        circuit_model = "R_0-Ws_0"
        initial_guess = [1.0, 1.0, 1.0]
    else:
        # 如果输入的数不在1到5之间，抛出一个异常
        raise ValueError("Input number must be between 0 and 8.")
    return circuit_model, initial_guess


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

def get_zreal(freq, zreal, basis):
    """Interpolates the real part of the impedance onto a common frequency basis"""
    x = zreal
    f = interp1d(freq, x, fill_value="extrapolate")  # extrapolate to prevent errors
    return f(basis)

def get_zimag(freq, zimag, basis):
    """Interpolates the imaginary part of the impedance onto a common frequency basis"""
    x = zimag
    f = interp1d(freq, x, fill_value="extrapolate")  # extrapolate to prevent errors
    return f(basis)

def preprocess_data(df, num_points=20):
    """Preprocesses the data from the CSV filename into a dataframe"""
    ## Load Training Data
    df["f"] = df.apply(lambda x: resize_array(x.freq, num_points), axis=1)
    df["zreal"] = df.apply(lambda x: get_zreal(x.freq, x.zreal, x.f), axis=1)
    df["zimag"] = df.apply(lambda x: get_zimag(x.freq, x.zimag, x.f), axis=1)
    return df.drop('freq', axis=1)

def unwrap_dataframe(df):
    df2 = pd.DataFrame(columns=["No.", "freq", "zreal", "zimag"])
    for i in np.arange(df.shape[0]):
        f, zreal, zimag = df[["f", "zreal", "zimag"]].loc[i]
        No_x = np.tile(i, f.size)
        df_ = pd.DataFrame(data=(No_x, f, zreal, zimag), index=["No.", "freq", "zreal", "zimag"]).T
        df2 = pd.concat([df2, df_], ignore_index=True)
    return df2

def group_dataframe(df):
    df_grouped = df.groupby("No.").apply(lambda x: pd.Series({"all": list(zip(x.f, x.zreal, x.zimag))})).reset_index()
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


def merge_columns_to_row(df): 
    df.columns = ['freq', 'zreal', 'zimag']
    A = df.T
    A['E'] = A.apply(lambda row: ','.join([str(val) for val in row]), axis=1)
    A['E'] = A.apply(lambda x: np.fromstring(x['E'], sep=","), axis=1)
    A = A[['E']].copy()
    B = A.T
    return B

def process_data_final(C):
    D = C.applymap(lambda x: str(x))
    E_zreal = D['zreal'][0]
    E_zimag = D['zimag'][0]
    E_f = D['f'][0]

    arr_zreal = np.fromstring(E_zreal[1:-1], sep=' ')
    arr_zimag = np.fromstring(E_zimag[1:-1], sep=' ')
    arr_f = np.fromstring(E_f[1:-1], sep=' ')

    F = pd.DataFrame({'No.': 0, 'f': arr_f, 'zreal': arr_zreal, 'zimag': arr_zimag})
    return F

def load_eis_data(file_path):
    """
    将包含三列“f, zreal, zimag"的数据转变为ML模型可以处理的dataframe
    加载 EIS 数据文件，将多列数据合并成一行，并进行数据预处理和重命名列名。
    :param file_path: EIS 数据文件路径。
    :return: 处理后的 EIS 数据 DataFrame。
    """
    # 加载 EIS 数据文件
    eis_data = pd.read_csv(file_path, sep=',')

    # 合并多列数据成一行
    merged_data = merge_columns_to_row(eis_data)

    # 进行数据预处理
    preprocessed_data = preprocess_data(merged_data)

    # 处理数据
    processed_data = process_data_final(preprocessed_data)

    # 按 No. 分组
    grouped_data = group_dataframe(processed_data)

    # 重命名列名并将数据类型转换为浮点数
    renamed_data = rename_dataframe_columns(grouped_data).astype(float)

    return renamed_data
