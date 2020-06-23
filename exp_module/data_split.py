
# basic
import copy
import pickle
import numpy as np
import pandas as pd
# %matplotlib inline
from pandas import DataFrame
import matplotlib.pyplot as plt
from IPython.display import display

# sdfはユーザ選択を行っていないがmulti_flagにより分割されているデータsplit_df
class Data_split():
    '''
    クラス間共通の変数とかないし考えるのも面倒なので、全部インスタンス変数にしておこう
    '''
    def __init__(self, sdf, user_sel, user_n):
        self.sdf = sdf
        self.user_sel = user_sel #データ
        self.user_n = user_n #選択したユーザ

        # データ数が30以上あるもの
        if self.user_sel['user'].count() >= 30:

            # user情報を除いたX(説明)とユーザ情報のみのY(目的)に分割
            def X_Y(user_sel):
                X = user_sel.drop("user", 1)
                Y = user_sel.user
                return X, Y

            self.X, self.Y = X_Y(self.user_sel)

            # train dataとtest dataを8:2で分割
            # _tは本人, _fは他人
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test_t, self.Y_train, self.Y_test_t = train_test_split(self.X, self.Y, test_size = 0.2, random_state = 0, shuffle = True)
            self.Y_test_t_size = self.Y_test_t.count()

            # sdを使用
            self.sdf_copy = self.sdf.copy()
            def hazure(sdf, user_n, Y_test_t_size):
                '''
                テストデータ用に本人以外の他人からもデータをとる
                取得する他人のデータは本人のテストデータと同数
                '''
                sdf_f = sdf[sdf['user'] != user_n]
                sdf_f_shuffle = sdf_f.sample(frac = 1, random_state = 0).reset_index(drop = True)
                sdf_f_sel = sdf_f_shuffle.sample(n = Y_test_t_size, random_state = 0)
                return sdf_f, sdf_f_sel

            self.sdf_f, self.sdf_f_sel = hazure(self.sdf_copy, self.user_n, self.Y_test_t_size)
            # 外れ値として扱うデータにおけるuser
            self.ls_sdf_f = list(self.sdf_f_sel['user'])

            # 実験に使用するデータの確認
            from exp_module import conform
            conform.conf_data(self.sdf, self.sdf_f, self.sdf_f_sel, self.ls_sdf_f)

            #userをすべて０に変更
            self.user_n_change = self.sdf_f_sel.copy()
            self.user_0 = self.user_n_change.replace({'user': self.ls_sdf_f}, 0)

            # user情報を除いたX_test_fとユーザ情報のみのY_test_fに分割
            self.X_test_f, self.Y_test_f = X_Y(self.user_0)

            # testデータの結合
            self.X_test = pd.concat([self.X_test_t, self.X_test_f]).reset_index(drop = True)
            self.Y_test = pd.concat([self.Y_test_t, self.Y_test_f]).reset_index(drop = True)

            # 各フラグごとにss,mm,rsのスケーリング
            from exp_module import scaling
            # 目的関数Yは共通
            # なし
            self.X_train_ori, self.X_test_ori, self.X_test_t_ori, self.X_test_f_ori = scaling.ori(self.X_train, self.X_test, self.X_test_t, self.X_test_f)
            # 標準化
            self.X_train_ss, self.X_test_ss, self.X_test_t_ss, self.X_test_f_ss = scaling.ss(self.X_train, self.X_test, self.X_test_t, self.X_test_f)
            # 正規化
            self.X_train_mm, self.X_test_mm, self.X_test_t_mm, self.X_test_f_mm = scaling.mm(self.X_train, self.X_test, self.X_test_t, self.X_test_f)
            # 外れ値？(これはもういいや)
            self.X_train_rs, self.X_test_rs, self.X_test_t_rs, self.X_test_f_rs = scaling.rs(self.X_train, self.X_test, self.X_test_t, self.X_test_f)

            # ひとまずここまで
            def target(Y_train, Y_test):
                y_train = pd.DataFrame(Y_train)
                g_train = y_train.groupby("user")
                train = pd.DataFrame(g_train.size().sort_values(ascending=False))
                list_index_train = train.index.values
                list_index_train.sort()

                y_test = pd.DataFrame(Y_test)
                g_test = y_test.groupby("user")
                test = pd.DataFrame(g_test.size().sort_values(ascending=False))
                list_index_test = test.index.values
                list_index_test
                return list_index_train, list_index_test

            self.y_train, self.y_test = target(self.Y_train, self.Y_test)

            # データ数の確認
            conform.conf_matome(self.X, self.Y, self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_test_t, self.X_test_f, self.Y_test_t, self.Y_test_f, self.y_train, self.y_test)

        else:
            # データ数が30未満なので今のところ実験不可
            print('None')
            self.X_train_ori , self.X_test_ori, self.X_test_t_ori, self.X_test_f_ori = 0, 0, 0, 0
            self.X_train_ss, self.X_test_ss, self.X_test_t_ss, self.X_test_f_ss = 0, 0, 0, 0
            self.X_train_mm, self.X_test_mm, self.X_test_t_mm, self.X_test_f_mm = 0, 0, 0, 0
            self.X_train_rs, self.X_test_rs, self.X_test_t_rs, self.X_test_f_rs = 0, 0, 0, 0
            self.Y_train, self.Y_test, self.Y_test_t, self.Y_test_f = 0, 0, 0, 0
            self.y_train, self.y_test, self.X_train = 0, 0, 0

if __name__ == "__main__":
    print('data_split module')
