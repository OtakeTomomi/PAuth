
import numpy as np
import pandas as pd
from pandas import DataFrame

# データの前処理
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# 二次元配列で行う場合 axis = 0 で列ごとの処理が行われる→これがデフォルト

# 各フラグごとにss,mm,rsのスケーリング
def ori(X_train, X_test,X_test1,X_test2):
    X_train_ori, X_test_ori, X_test1_ori, X_test2_ori = X_train, X_test, X_test1,X_test2
    return X_train_ori, X_test_ori, X_test1_ori, X_test2_ori

# 標準化
def ss(X_train, X_test,X_test1,X_test2):
    ss = preprocessing.StandardScaler().fit(X_train)
    # モデルを保存する
    # ss_filename = 'finalized_ss.sav'
    # pickle.dump(ss, open(ss_filename, 'wb'))
    X_train_ss = ss.transform(X_train)
    X_test_ss = ss.transform(X_test) #type()は'numpy.ndarray'
    X_test1_ss = ss.transform(X_test1)
    X_test2_ss = ss.transform(X_test2)
    return X_train_ss, X_test_ss, X_test1_ss, X_test2_ss

# 正規化
def mm(X_train, X_test,X_test1,X_test2):
    mm = preprocessing.MinMaxScaler().fit(X_train)
    # モデルを保存する
    # mm_filename = 'finalized_mm.sav'
    # pickle.dump(mm, open(mm_filename, 'wb'))
    X_train_mm = mm.transform(X_train)
    X_test_mm = mm.transform(X_test)
    X_test1_mm = mm.transform(X_test1)
    X_test2_mm =mm.transform(X_test2)
    return X_train_mm, X_test_mm,X_test1_mm, X_test2_mm

# 外れ値に強いやつ
def rs(X_train, X_test,X_test1,X_test2):
    rs = preprocessing.RobustScaler(quantile_range=(25., 75.)).fit(X_train)
    # モデルを保存する
    # rs_filename = 'finalized_rs.sav'
    # pickle.dump(rs, open(rs_filename, 'wb'))
    X_train_rs = rs.transform(X_train)
    X_test_rs = rs.transform(X_test)
    X_test1_rs = rs.transform(X_test1)
    X_test2_rs = rs.transform(X_test2)
    return X_train_rs, X_test_rs,X_test1_rs, X_test2_rs

if __name__ == "__main__":
    #  目的関数Yは共通
    # なし
    X_train_ori, X_test_ori, X_test1_ori, X_test2_ori = ori(X_train, X_test, X_test1,X_test2)
    # 標準化
    X_train_ss, X_test_ss,X_test1_ss, X_test2_ss = ss(X_train, X_test,X_test1, X_test2)
    # 正規化
    X_train_mm, X_test_mm,X_test1_mm, X_test2_mm = mm(X_train, X_test, X_test1, X_test2)
    # 外れ値？
    X_train_rs, X_test_rs,X_test1_rs,X_test2_rs = rs(X_train, X_test, X_test1, X_test2)
