
import copy
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
# from IPython.display import display

def ds_exp(st, user_select):

    if user_select['user'].count() >= 30:
        def X_Y(user_select):
            X = user_select.drop("user", 1)
            Y = user_select.user
            return X, Y

        X, Y = X_Y(user_select)

        from sklearn.model_selection import train_test_split
        def tt(X, Y):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
            return X_train, X_test, Y_train, Y_test

        X_train, X_test1, Y_train, Y_test1 = tt(X, Y)

        Y_test1_size = Y_test1.count()

        # stを使用
        st2 = st.copy()
        def hazure(st2,user_n,Y_test1_size):
            #user_n以外のuserを抽出
            sst = st2[st2['user'] != user_n]
            #sstをシャッフル
            sst_shuffle = sst.sample(frac=1, random_state=0).reset_index(drop=True)
            #testデータと同じだけ外れ値となるものを抽出
            sst_select_outlier = sst_shuffle.sample(n = Y_test1_size, random_state=0)
            return sst, sst_select_outlier

        df_hazure, sst_select_outlier = hazure(st2,user_n,Y_test1_size)

        print('all:',st['user'].count())
        print('all - select_data:',df_hazure['user'].count())
        print('select_data:',st['user'].count() - df_hazure['user'].count())
        print('select_hazure_data:',sst_select_outlier['user'].count())

        # testデータと外れ値データの結合
    #     print('外れ値：',sst_select_outlier.shape)

        list2 = list(sst_select_outlier['user'])
        print('\n外れ値として扱うuserのnumber\n',list2)

        user_n_change = sst_select_outlier.copy()
        #userをすべて０に変更
        user_0 = user_n_change.replace({'user': list2},0)

        X_test2, Y_test2 = X_Y(user_0)

        # testデータの結合
        X_test = pd.concat([X_test1, X_test2]).reset_index(drop=True)

        Y_test = pd.concat([Y_test1, Y_test2]).reset_index(drop=True)

        # 各フラグごとにss,mm,rsのスケーリング
        def scaling(X_train, X_test, X_test1,X_test2):

            def ss(X_train, X_test,X_test1,X_test2):
                # 標準化するよ
                # 二次元配列で行う場合　axis = 0 で列ごとの処理が行われる→これがデフォルト
                from sklearn import preprocessing
                from sklearn.preprocessing import StandardScaler
                ss = preprocessing.StandardScaler().fit(X_train)
                # ss.fit(X_train)
                # モデルを保存する
                # ss_filename = 'finalized_ss.sav'
                # pickle.dump(ss, open(ss_filename, 'wb'))
                X_train_ss = ss.transform(X_train)
                X_test_ss = ss.transform(X_test) #type()は'numpy.ndarray'
                # preprocessing.scale(X)
                X_test1_ss = ss.transform(X_test1)
                X_test2_ss = ss.transform(X_test2)
                return X_train_ss, X_test_ss, X_test1_ss, X_test2_ss

            def mm(X_train, X_test,X_test1,X_test2):
                # 正規化するよ
                from sklearn import preprocessing
                from sklearn.preprocessing import MinMaxScaler
                mm = preprocessing.MinMaxScaler().fit(X_train)
                # mm.fit(X_train)
                # モデルを保存する
                # mm_filename = 'finalized_mm.sav'
                # pickle.dump(mm, open(mm_filename, 'wb'))
                X_train_mm = mm.transform(X_train)
                X_test_mm = mm.transform(X_test)
                X_test1_mm = mm.transform(X_test1)
                X_test2_mm =mm.transform(X_test2)
                # preprocessing.minmax_scale(X) # 直接処理するもの
                return X_train_mm, X_test_mm,X_test1_mm, X_test2_mm

            def rs(X_train, X_test,X_test1,X_test2):
                # 外れ値に強いやつ
                from sklearn import preprocessing # このなかに処理がまとめて入ってるらしい
                from sklearn.preprocessing import RobustScaler
                rs = preprocessing.RobustScaler(quantile_range=(25., 75.)).fit(X_train)
                # rs.fit(X_train)
                # モデルを保存する
                # rs_filename = 'finalized_rs.sav'
                # pickle.dump(rs, open(rs_filename, 'wb'))
                X_train_rs = rs.transform(X_train)
                X_test_rs = rs.transform(X_test)
                X_test1_rs = rs.transform(X_test1)
                X_test2_rs = rs.transform(X_test2)
                return X_train_rs, X_test_rs,X_test1_rs, X_test2_rs

            #  目的関数Yは共通
            # なし
            X_train_ori, X_test_ori, X_test1_ori, X_test2_ori = X_train, X_test, X_test1,X_test2
            # 標準化
            X_train_ss, X_test_ss,X_test1_ss, X_test2_ss = ss(X_train, X_test,X_test1, X_test2)
            # 正規化
            X_train_mm, X_test_mm,X_test1_mm, X_test2_mm = mm(X_train, X_test, X_test1, X_test2)
            # 外れ値？
            X_train_rs, X_test_rs,X_test1_rs,X_test2_rs = rs(X_train, X_test, X_test1, X_test2)

            return X_train_ori, X_test_ori, X_test1_ori, X_test2_ori, X_train_ss, X_test_ss,X_test1_ss, X_test2_ss,X_train_mm, X_test_mm,X_test1_mm, X_test2_mm, X_train_rs, X_test_rs,X_test1_rs,X_test2_rs

        X_train_ori, X_test_ori, X_test1_ori, X_test2_ori,X_train_ss, X_test_ss,X_test1_ss, X_test2_ss,X_train_mm, X_test_mm,X_test1_mm, X_test2_mm, X_train_rs, X_test_rs,X_test1_rs,X_test2_rs = scaling(X_train, X_test, X_test1,X_test2)

        # ひとまずここまで
        def target(Y_train,Y_test):
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

        y_train, y_test = target(Y_train,Y_test)

        # matome
        print('\nX_base:',X.shape)
        print('Y_base:',Y.shape)
        print('X_train:',X_train.shape)
        print('Y_train:',Y_train.shape)
        print('true_test:',X_test1.shape)
        print('false_test:',X_test2.shape)
    #     print('Y_test2:',Y_test2.shape)
    #     print('Y_train:',Y_train.shape)
        print('X_test:',X_test.shape)
        print('Y_test:',Y_test.shape)
        print('y_train:',y_train)
        print('y_test:',y_test)

        return X_train_ori, X_test_ori, X_test1_ori, X_test2_ori,X_train_ss, X_test_ss,X_test1_ss, X_test2_ss,X_train_mm, X_test_mm,X_test1_mm, X_test2_mm, X_train_rs, X_test_rs,X_test1_rs,X_test2_rs,Y_train, Y_test, y_train, y_test, X_train, Y_test1, Y_test2

    else:
        print('None')
        return 0, 0, 0, 0, 0, 0,0, 0,0, 0,0, 0, 0, 0,0,0,0, 0, 0, 0, 0,0,0
        pass


if __name__ == "__main__":
    print('data_split_exp module')