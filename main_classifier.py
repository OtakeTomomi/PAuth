"""
多クラス分類
組み合わせなし
"""

# import os
# import sys
# import copy
# import pickle
# from typing import List

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from IPython.display import display

# warning ignore code
import warnings
warnings.filterwarnings('ignore')

# モデル
# from sklearn.linear_model import LogisticRegression # ロジスティック回帰
# from sklearn.svm import LinearSVC # 線形SVM
# from sklearn.svm import SVC #SVM
# from sklearn.tree import  DecisionTreeClassifier # 決定木
# from sklearn.neighbors import  KNeighborsClassifier # k-NN
# from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト

# スケーリング
# from sklearn import preprocessing

# 交差検証
# from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, GridSearchCV

from sklearn.model_selection import train_test_split

# 評価用のライブラリ？
# import itertools
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from multiprocessing import cpu_count
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import cross_val_score

# from tqdm import tqdm_notebook as tqdm
# from tqdm import tqdm
# import time
# from multiprocessing import cpu_count
# from sklearn.externals import joblib

# 内部ライブラリ
from expmodule.dataset import load_frank
from expmodule.flag_split import flag4
from expmodule.classifier import user_select


def main(df, select_n):

    # データのColumn取得
    df_column = df.columns.values

    # 上下左右のflagをもとにデータを分割
    a, b, c, d = flag4(df, 'flag')

    # 実験に使用するユーザの選択とソート
    def h_select(df_flag, s_n):
        """
        :param df_flag: フラグで切り分けられたデータ
        :param s_n: 選択されたユーザ数
        :return: 選択されたユーザ数のデータ
        """
        # フラグ内のuserごとにデータをまとめる
        g2 = df_flag.groupby("user")
        # 降順にフラグの数でソート
        df2_h = pd.DataFrame(g2.size().sort_values(ascending=False))
        # データ数の多いuserからリストに格納
        list_index = df2_h.index.values
        df_h_select = df_flag[df_flag['user'].isin(list_index[0:s_n])]
        ff = df_h_select.groupby("user")
        print(ff.size().sort_values(ascending=False))
        print("選択されているもの→　メンバー：{}人".format(s_n))
        print("=========================")
        return df_h_select

    # print("a")
    df_h1_select = user_select(a, select_n)
    # print("b")
    df_h2_select = user_select(b, select_n)
    # print("c")
    df_h3_select = user_select(c, select_n)
    # print("d")
    df_h4_select = user_select(d, select_n)

    # ここでデータの数を調整する処理を書く
    def data_max(df_h1_select, df_h2_select, df_h3_select, df_h4_select):
        # 各フラグの個数を算出する
        f1 = len(df_h1_select)
        f2 = len(df_h2_select)
        f3 = len(df_h3_select)
        f4 = len(df_h4_select)
        f = 0

        if f1 <= f2 and f1 <= f3 and f1 <= f4:
            f = f1
        elif f2 <= f1 and f2 <= f3 and f2 <= f4:
            f = f2
        elif f3 <= f1 and f3 <= f2 and f3 <= f4:
            f = f3
        else:
            f = f4
        print('各ストロークのなかで最も少ないデータ数：', f)
        return f

    data_max = data_max(df_h1_select, df_h2_select, df_h3_select, df_h4_select)

    def data_sample(df_h_select, data_max):
        df = df_h_select
        df2 = df.sample(n=data_max, random_state=0).reset_index(drop=True)
        ff = df2.groupby("user")
        print(ff.size().sort_values(ascending=False))
        return df2


    df1_select = data_sample(df_h1_select, data_max)
    df2_select = data_sample(df_h2_select, data_max)
    df3_select = data_sample(df_h3_select, data_max)
    df4_select = data_sample(df_h4_select, data_max)

    # 説明変数Xと目的変数Yにわける
    def X_Y(train_data):
        x = train_data.drop("user", 1)
        y = train_data.user
        return x, y

    X1, Y1 = X_Y(df1_select)
    print("X1 :", X1.shape, "\nY1 :", Y1.shape, '\n')
    X2, Y2 = X_Y(df2_select)
    print("X2 :", X2.shape, "\nY2 :", Y2.shape, '\n')
    X3, Y3 = X_Y(df3_select)
    print("X3 :", X3.shape, "\nY3 :", Y3.shape, '\n')
    X4, Y4 = X_Y(df4_select)
    print("X4 :", X4.shape, "\nY4 :", Y4.shape, '\n')


    def tt(X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
        return X_train, X_test, Y_train, Y_test


    X1_train, X1_test, Y1_train, Y1_test = tt(X1, Y1)
    X2_train, X2_test, Y2_train, Y2_test = tt(X2, Y2)
    X3_train, X3_test, Y3_train, Y3_test = tt(X3, Y3)
    X4_train, X4_test, Y4_train, Y4_test = tt(X4, Y4)


    # class Experiment():
    #
    #     def __init__(self):
    #         s


if __name__ == '__main__':
    # 実験1では5人ずつ変化させる
    nlist = [5, 10, 15, 20, 25, 30, 35, 41]
    # データの読み込み
    frank_df = load_frank(False)
    for n in nlist:
        main(frank_df, n)
