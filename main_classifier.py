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

    # 第1段階: データの読み込み&ストローク方向で分割

    # データのColumn取得
    df_column = df.columns.values

    # 上下左右のflagをもとにデータを分割
    a, b, c, d = flag4(df, 'flag')

    # 実験に使用するユーザの選択とソート
    a_user = user_select(a, select_n)
    b_user = user_select(b, select_n)
    c_user = user_select(c, select_n)
    d_user = user_select(d, select_n)

    # 第2段階: 保留

    class ClassifierOne(object):

        def __init__(self, df_flag, s_n, select_session='all'):
            self.df_flag = df_flag
            self.s_n = s_n
            self.select_session = select_session

            # これはテストデータとかで分けるときにする
            from expmodule.session_select import session
            self.df_session_select = session(self.df_flag, self.select_session)

            def x_y_split(self):
                x = self.df_flag.drop("user", 1)
                y = self.df_flag.user

                return x, y

            self.X, self.Y = x_y_split(self.df_flag)









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
