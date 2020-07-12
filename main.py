'''
メインの実験プログラムのつもり
条件：2ストロークの組み合わせ，分類器は1クラス分類器使用.
'''

import os
import copy
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
# from IPython.display import display

# モデル
import sklearn
from sklearn import svm
from sklearn.svm import OneClassSVM
# from sklearn.mixture import GaussianMixture
# from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

#スケーリング
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# その他
# from tqdm import tqdm_notebook as tqdm
import time
from tqdm import tqdm
from multiprocessing import cpu_count
# from sklearn.externals import joblib


# warning inogre code
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
from exp_module import read_data as rd
frank_df = rd.load_frank_data()

# データをmulti_flagを基準に分割
# a,b,c,dのストローク方向はflag_splitに記載
from exp_module import flag_split
aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd = flag_split.frank_fs(frank_df)
# 各multi_flagに含まれる各ユーザのデータ数の多い順について確認したい場合にはlist_index = conform.conf_sel_flag_qty()で確認可能ではある

# 選択されたユーザのデータを各aa~ddから抽出
# select_user_from_frank_fs
def sel_user_ffs(sdf, user_n):
    sdf_sel_u = sdf[sdf['user'] == user_n]
    ff = sdf_sel_u.groupby("user")
    # print(ff.size())
    return sdf_sel_u

# コマンドラインからどのユーザを選択するか選ぶ
user_n = int(input('\nユーザの選択1~41 >> '))

# 各multi_flagごとに選択したユーザを抽出する
selu_aa = sel_user_ffs(aa, user_n)
selu_ab = sel_user_ffs(ab, user_n)
selu_ac = sel_user_ffs(ac, user_n)
selu_ad = sel_user_ffs(ad, user_n)

selu_ba = sel_user_ffs(ba, user_n)
selu_bb = sel_user_ffs(bb, user_n)
selu_bc = sel_user_ffs(bc, user_n)
selu_bd = sel_user_ffs(bd, user_n)

selu_ca = sel_user_ffs(ca, user_n)
selu_cb = sel_user_ffs(cb, user_n)
selu_cc = sel_user_ffs(cc, user_n)
selu_cd = sel_user_ffs(cd, user_n)

selu_da = sel_user_ffs(da, user_n)
selu_db = sel_user_ffs(db, user_n)
selu_dc = sel_user_ffs(dc, user_n)
selu_dd = sel_user_ffs(dd, user_n)

'''
さてどうしようか
インスタンス変数を使うか，普通に関数の戻り値を使うか
'''

# 実験用にデータを訓練データ，検証データ，テストデータにわける
# from exp_module import data_split as ds
# f11 = ds.DataSplitExpt(aa, selu_aa, user_n)

def data_split(st, user_select):

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

print('\n-----------------------------------------------------------------\na + a\n-----------------------------------------------------------------')
X11_train_ori, X11_test_ori, X11_test1_ori, X11_test2_ori,X11_train_ss, X11_test_ss,X11_test1_ss, X11_test2_ss,X11_train_mm, X11_test_mm,X11_test1_mm, X11_test2_mm, X11_train_rs, X11_test_rs,X11_test1_rs,X11_test2_rs,Y11_train, Y11_test, y11_train, y11_test, X_train_uu, Y11_test1, Y11_test2 = data_split(aa, selu_aa)
print('\n-----------------------------------------------------------------\na + b\n-----------------------------------------------------------------')
X12_train_ori, X12_test_ori, X12_test1_ori, X12_test2_ori,X12_train_ss, X12_test_ss,X12_test1_ss, X12_test2_ss,X12_train_mm, X12_test_mm,X12_test1_mm, X12_test2_mm, X12_train_rs, X12_test_rs,X12_test1_rs,X12_test2_rs,Y12_train, Y12_test, y12_train, y12_test, X_train_ud, Y12_test1, Y12_test2 = data_split(ab, selu_ab)
print('\n-----------------------------------------------------------------\na + c\n-----------------------------------------------------------------')
X13_train_ori, X13_test_ori, X13_test1_ori, X13_test2_ori,X13_train_ss, X13_test_ss,X13_test1_ss, X13_test2_ss,X13_train_mm, X13_test_mm,X13_test1_mm, X13_test2_mm, X13_train_rs, X13_test_rs,X13_test1_rs,X13_test2_rs,Y13_train, Y13_test, y13_train, y13_test, X_train_ul, Y13_test1, Y13_test2 = data_split(ac, selu_ac)
print('\n-----------------------------------------------------------------\na + d\n-----------------------------------------------------------------')
X14_train_ori, X14_test_ori, X14_test1_ori, X14_test2_ori,X14_train_ss, X14_test_ss,X14_test1_ss, X14_test2_ss,X14_train_mm, X14_test_mm,X14_test1_mm, X14_test2_mm, X14_train_rs, X14_test_rs,X14_test1_rs,X14_test2_rs,Y14_train, Y14_test, y14_train, y14_test,X_train_ur, Y14_test1, Y14_test2 = data_split(ad, selu_ad)

print('\n-----------------------------------------------------------------\nb + a\n-----------------------------------------------------------------')
X21_train_ori, X21_test_ori, X21_test1_ori, X21_test2_ori,X21_train_ss, X21_test_ss,X21_test1_ss, X21_test2_ss,X21_train_mm, X21_test_mm,X21_test1_mm, X21_test2_mm, X21_train_rs, X21_test_rs,X21_test1_rs,X21_test2_rs,Y21_train, Y21_test, y21_train, y21_test, X_train_du, Y21_test1, Y21_test2 = data_split(ba, selu_ba)
print('\n-----------------------------------------------------------------\nb + b\n-----------------------------------------------------------------')
X22_train_ori, X22_test_ori, X22_test1_ori, X22_test2_ori,X22_train_ss, X22_test_ss,X22_test1_ss, X22_test2_ss,X22_train_mm, X22_test_mm,X22_test1_mm, X22_test2_mm, X22_train_rs, X22_test_rs,X22_test1_rs,X22_test2_rs,Y22_train, Y22_test, y22_train, y22_test, X_train_dd, Y22_test1, Y22_test2 = data_split(bb, selu_bb)
print('\n-----------------------------------------------------------------\nb + c\n-----------------------------------------------------------------')
X23_train_ori, X23_test_ori, X23_test1_ori, X23_test2_ori,X23_train_ss, X23_test_ss,X23_test1_ss, X23_test2_ss,X23_train_mm, X23_test_mm,X23_test1_mm, X23_test2_mm, X23_train_rs, X23_test_rs,X23_test1_rs,X23_test2_rs,Y23_train, Y23_test, y23_train, y23_test, X_train_dl, Y23_test1, Y23_test2 = data_split(bc, selu_bc)
print('\n-----------------------------------------------------------------\nb + d\n-----------------------------------------------------------------')
X24_train_ori, X24_test_ori, X24_test1_ori, X24_test2_ori,X24_train_ss, X24_test_ss,X24_test1_ss, X24_test2_ss,X24_train_mm, X24_test_mm,X24_test1_mm, X24_test2_mm, X24_train_rs, X24_test_rs,X24_test1_rs,X24_test2_rs,Y24_train, Y24_test, y24_train, y24_test,X_train_dr, Y24_test1, Y24_test2 = data_split(bd, selu_bd)

print('\n-----------------------------------------------------------------\nc + a\n-----------------------------------------------------------------')
X31_train_ori, X31_test_ori, X31_test1_ori, X31_test2_ori,X31_train_ss, X31_test_ss,X31_test1_ss, X31_test2_ss,X31_train_mm, X31_test_mm,X31_test1_mm, X31_test2_mm, X31_train_rs, X31_test_rs,X31_test1_rs,X31_test2_rs,Y31_train, Y31_test, y31_train, y31_test, X_train_lu, Y31_test1, Y31_test2 = data_split(ca, selu_ca)
print('\n-----------------------------------------------------------------\nc + b\n-----------------------------------------------------------------')
X32_train_ori, X32_test_ori, X32_test1_ori, X32_test2_ori,X32_train_ss, X32_test_ss,X32_test1_ss, X32_test2_ss,X32_train_mm, X32_test_mm,X32_test1_mm, X32_test2_mm, X32_train_rs, X32_test_rs,X32_test1_rs,X32_test2_rs,Y32_train, Y32_test, y32_train, y32_test, X_train_ld, Y32_test1, Y32_test2 = data_split(cb, selu_cb)
print('\n-----------------------------------------------------------------\nc + c\n-----------------------------------------------------------------')
X33_train_ori, X33_test_ori, X33_test1_ori, X33_test2_ori,X33_train_ss, X33_test_ss,X33_test1_ss, X33_test2_ss,X33_train_mm, X33_test_mm,X33_test1_mm, X33_test2_mm, X33_train_rs, X33_test_rs,X33_test1_rs,X33_test2_rs,Y33_train, Y33_test, y33_train, y33_test, X_train_ll, Y33_test1, Y33_test2 = data_split(cc, selu_cc)
print('\n-----------------------------------------------------------------\nc + d\n-----------------------------------------------------------------')
X34_train_ori, X34_test_ori, X34_test1_ori, X34_test2_ori,X34_train_ss, X34_test_ss,X34_test1_ss, X34_test2_ss,X34_train_mm, X34_test_mm,X34_test1_mm, X34_test2_mm, X34_train_rs, X34_test_rs,X34_test1_rs,X34_test2_rs,Y34_train, Y34_test, y34_train, y34_test,X_train_lr, Y34_test1, Y34_test2 = data_split(cd, selu_cd)

print('\n-----------------------------------------------------------------\nd + a\n-----------------------------------------------------------------')
X41_train_ori, X41_test_ori, X41_test1_ori, X41_test2_ori,X41_train_ss, X41_test_ss,X41_test1_ss, X41_test2_ss,X41_train_mm, X41_test_mm,X41_test1_mm, X41_test2_mm, X41_train_rs, X41_test_rs,X41_test1_rs,X41_test2_rs,Y41_train, Y41_test, y41_train, y41_test, X_train_ru, Y41_test1, Y41_test2 = data_split(da, selu_da)
print('\n-----------------------------------------------------------------\nd + b\n-----------------------------------------------------------------')
X42_train_ori, X42_test_ori, X42_test1_ori, X42_test2_ori,X42_train_ss, X42_test_ss,X42_test1_ss, X42_test2_ss,X42_train_mm, X42_test_mm,X42_test1_mm, X42_test2_mm, X42_train_rs, X42_test_rs,X42_test1_rs,X42_test2_rs,Y42_train, Y42_test, y42_train, y42_test, X_train_rd, Y42_test1, Y42_test2 = data_split(db, selu_db)
print('\n-----------------------------------------------------------------\nd + c\n-----------------------------------------------------------------')
X43_train_ori, X43_test_ori, X43_test1_ori, X43_test2_ori,X43_train_ss, X43_test_ss,X43_test1_ss, X43_test2_ss,X43_train_mm, X43_test_mm,X43_test1_mm, X43_test2_mm, X43_train_rs, X43_test_rs,X43_test1_rs,X43_test2_rs,Y43_train, Y43_test, y43_train, y43_test, X_train_rl, Y43_test1, Y43_test2 = data_split(dc, selu_dc)
print('\n-----------------------------------------------------------------\nd + d\n-----------------------------------------------------------------')
X44_train_ori, X44_test_ori, X44_test1_ori, X44_test2_ori,X44_train_ss, X44_test_ss,X44_test1_ss, X44_test2_ss,X44_train_mm, X44_test_mm,X44_test1_mm, X44_test2_mm, X44_train_rs, X44_test_rs,X44_test1_rs,X44_test2_rs,Y44_train, Y44_test, y44_train, y44_test,X_train_rr, Y44_test1, Y44_test2 = data_split(dd, selu_dd)

def result(normal_result, anomaly_result, Y_true, prediction, y_score):
    print("\n正常データのスコア\n", normal_result)
    print("\n異常データのスコア\n", anomaly_result)
    TP = np.count_nonzero(normal_result == 1)
    FN = np.count_nonzero(normal_result == -1)
    FP = np.count_nonzero(anomaly_result == 1)
    TN = np.count_nonzero(anomaly_result == -1)
    print('\nTP：',TP,'　FN:',FN, '　FP:',FP,'　TN:',TN)

    cm = confusion_matrix(Y_true,prediction,labels=[1, -1])
    print(cm)

    print('classification_report\n', classification_report(Y_true, prediction))
    print('\nAccuracy:',accuracy_score(Y_true, prediction))
    print('Precision:',precision_score(Y_true, prediction))
    print('Recall:',recall_score(Y_true, prediction))
    print('F1:',f1_score(Y_true, prediction))
    FRR = FN / (FN + TP)
    print("FRR:{}".format(FRR))
    FAR = FP / (TN + FP)
    print("FAR:{}".format(FAR))
    BER = 0.5 * (FP / (TN + FP) + FN / (FN + TP))
    print("BER:{}".format(BER))
    print('AUC',roc_auc_score(Y_true, y_score))

    fpr, tpr, thresholds = roc_curve(Y_true, y_score, drop_intermediate=False)
    auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(8,6),dpi=200)
    plt.title('ROC curve (AUC = %.3f)' %auc)
    plt.plot(fpr, tpr, label='ROC curve (area = %.3f)' %auc , marker='o')
    plt.plot([0,1],[0,1],color = 'black', linestyle='--')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()

    plt.legend()

    plt.show()

from sklearn.model_selection import train_test_split

# 評価
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef

from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import mglearn
# import japanize_matplotlib
from mpl_toolkits.mplot3d import Axes3D


class Experiment3():
    def __init__(self,X_train_ori, X_test_ori, X_test1_ori, X_test2_ori,X_train_ss, X_test_ss,X_test1_ss, X_test2_ss,X_train_mm, X_test_mm,X_test1_mm, X_test2_mm, X_train_rs, X_test_rs,X_test1_rs,X_test2_rs, Y_train, Y_test, y_train, y_test, flag_n,user_n,st,X_train, Y_test1, Y_test2):
        self.X_train_ori = X_train_ori
        self.X_test_ori = X_test_ori
        self.X_test1_ori = X_test1_ori
        self.X_test2_ori = X_test2_ori
        self.X_train_ss = X_train_ss
        self.X_test_ss = X_test_ss
        self.X_test1_ss = X_test1_ss
        self.X_test2_ss = X_test2_ss
        self.X_train_mm = X_train_mm
        self.X_test_mm = X_test_mm
        self.X_test1_mm = X_test1_mm
        self.X_test2_mm = X_test2_mm
        self.X_train_rs = X_train_rs
        self.X_test_rs = X_test_rs
        self.X_test1_rs = X_test1_rs
        self.X_test2_rs = X_test2_rs
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.y_train = y_train
        self.y_test = y_test
        self.Y_test1 = Y_test1
        self.Y_test2 = Y_test2
        self.flag_n = flag_n
        self.user_n = user_n
        self.st = st
        self.X_train = X_train
        self.trains = [self.X_train_ori, self.X_train_ss, self.X_train_mm, self.X_train_rs]
        self.tests = [self.X_test_ori, self.X_test_ss, self.X_test_mm, self.X_test_rs]
        self.tests1 = [self.X_test1_ori, self.X_test1_ss, self.X_test1_mm, self.X_test1_rs]
        self.tests2 = [self.X_test2_ori, self.X_test2_ss, self.X_test2_mm, self.X_test2_rs]
        self.names = ['なし:Orignal_', '標準化：StandardScale_','正規化：MinMaxScaler_','外れ値に頑強な標準化：RobustScaler_']

    def pca_scale(self):

        try:
            n = 1
            scaled_df = self.X_test_ss
            # pca = PCA(n_components=2)
            # # pca.fit(X33_train_ss)
            # pca.fit(scaled_df)
            # pca_results = pca.transform(scaled_df)

            pca = PCA(n_components=3)
            pca.fit(scaled_df)
            pca_results = pca.transform(scaled_df)
            fig = plt.figure(1, figsize=(8,6))
            ax = Axes3D(fig, rect=[0, 0, 0.8, 0.8], elev=30, azim=60)
            ax.set_title('user:{0}, strokeType:{1}\n各次元の寄与率: PCA1: {2[0]:.3f} PCA2: {2[1]:.3f} PCA3: {2[2]:.3f}\n累積寄与率: {3:.3f}'.format(self.user_n,self.flag_n,pca.explained_variance_ratio_,sum(pca.explained_variance_ratio_)))
            # colors = ["blue","orange"]
            ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2], s = 20, c = self.Y_test, alpha=0.5,
            linewidth=0.5,edgecolors="k", cmap="cool")
            ax.w_xaxis.set_label_text("PCA1")
            ax.w_yaxis.set_label_text("PCA2")
            ax.w_zaxis.set_label_text("PCA3")
            ax.legend(["normal", "anormal"],loc="best")

            # 主成分の寄与率を出力する
            print('user:{0}, stroke:{1}'.format(self.user_n,self.flag_n))
            print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
            print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))
            plt.show()
            # plt.close()
        except ValueError:
            pass

    def pca(self):
        try:
            n = 1
            scaled_df = self.X_test_ss
            # pca = PCA(n_components=2)
            # # pca.fit(X33_train_ss)
            # pca.fit(scaled_df)
            # pca_results = pca.transform(scaled_df)
            pca = PCA(n_components=3)
            pca.fit(scaled_df)
            pca_results = pca.transform(scaled_df)
            fig = plt.figure(1, figsize=(8,6))
            ax = Axes3D(fig, rect=[0, 0, 0.8, 0.8], elev=30, azim=60)
            ax.set_title('user:{0}, strokeType:{1}\n各次元の寄与率: PCA1: {2[0]:.3f} PCA2: {2[1]:.3f} PCA3: {2[2]:.3f}\n累積寄与率: {3:.3f}'.format(self.user_n,self.flag_n,pca.explained_variance_ratio_,sum(pca.explained_variance_ratio_)))
            # colors = ["blue","orange"]
            ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2], s = 20, c = self.Y_test, alpha=0.5,
            linewidth=0.5,edgecolors="k", cmap="cool")
            ax.w_xaxis.set_label_text("PCA1")
            ax.w_yaxis.set_label_text("PCA2")
            ax.w_zaxis.set_label_text("PCA3")
            ax.legend(["normal", "anormal"],loc="best")

            # 主成分の寄与率を出力する
            print('user:{0}, stroke:{1}'.format(self.user_n,self.flag_n))
            print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
            print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))
            plt.show()
            # plt.close()


        except ValueError:
            pass
    def plot_IF(self):

        # try:
        Y_true1 = self.Y_test.copy()
        Y_true = Y_true1.replace({self.user_n:1,0:-1})

        X_scaled1 = self.X_train_ss
        pca1 = PCA(n_components=2)
        pca1.fit(X_scaled1)
        X_pca1 = pca1.transform(X_scaled1)

        X_scaled = self.X_test_ss
        pca = PCA(n_components=2)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)

        isf = IsolationForest(n_estimators=1,
                          contamination='auto',
                          behaviour='new',random_state=0)

        isf.fit(X_pca1)
        prediction = isf.predict(X_pca)

        # normal_result = isf.predict(self.X_test1_ss)
        # anomaly_result = isf.predict(self.X_test2_ss)

        # y_score = isf.decision_function(self.X_test_ss)

        plt.figure(figsize=(10,6))
        mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],self.Y_test)
        mglearn.discrete_scatter(X_pca1[:,0],X_pca1[:,1],self.Y_train,c='g')
        plt.legend(["anormal", "normal"],loc="best", fontsize=16)
        # plt.scatter(X_pca1, self.Y_train, label='train')
        plt.xlabel("第一主成分", fontsize=15)
        plt.ylabel("第二主成分", fontsize=15)
        # plt.scatter(self.X_test2_ss, self.Y_test2, label='negative')
        # plt.scatter(self.X_test1_ss, self.Y_test1, label='positive')
        plt.legend()
        plt.show()

        # #3次元のグラフの枠を作っていく
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2],self.Y_test)
        # mglearn.discrete_scatter(X_pca1[:,0],X_pca1[:,1],X_pca1[:,2],self.Y_train)
        #
        # #軸にラベルを付けたいときは書く
        # ax.set_title('figure{0}_{1}'.format(self.user_n,self.flag_n))
        # ax.legend(["anormal", "normal"],loc="best", fontsize=16)
        # ax.set_xlabel("第一主成分", fontsize=15)
        # ax.set_ylabel("第二主成分", fontsize=15)
        # ax.set_zlabel("第三主成分", fontsize=15)
        #
        # #.plotで描画
        # #linestyle='None'にしないと初期値では線が引かれるが、3次元の散布図だと大抵ジャマになる
        # #markerは無難に丸
        # ax.plot(X_pca[:,0],X_pca[:,1],X_pca[:,2],self.Y_test,marker="o",linestyle='None')
        # ax.plot(X_pca1[:,0],X_pca1[:,1],X_pca1[:,2],self.Y_train,marker="o",linestyle='None')
        # plt.show()

        # except ValueError:
        #     pass

    def closs(self):

        def far_frr(normal_result, anomaly_result):
            TP = np.count_nonzero(normal_result == 1)
            FN = np.count_nonzero(normal_result == -1)
            FP = np.count_nonzero(anomaly_result == 1)
            TN = np.count_nonzero(anomaly_result == -1)
            FRR = FN / (FN + TP)
            FAR = FP / (TN + FP)
            BER = 0.5 * (FP / (TN + FP) + FN / (FN + TP))
            return FRR, FAR, BER

        def hazu(st,X_val_no,Y_val_no,user_n, scale_n, X_train):
            '''
            valに外れ値データをつけたい
            '''
            def hazure2(st, X_val_no, Y_val_no, user_n):
                st2 = st.copy()
                sst = st2[st2['user'] != user_n]
                val_no_size = Y_val_no.count()
                sst_shuffle = sst.sample(frac=1, random_state=0).reset_index(drop=True)
                sst_select_outlier = sst_shuffle.sample(n = val_no_size, random_state=0)
                return sst_select_outlier

            sst_select_outlier = hazure2(st, X_val_no, Y_val_no, user_n)
            list2 = list(sst_select_outlier['user'])
            user_n_change = sst_select_outlier.copy()
            user_0 = user_n_change.replace({'user': list2},0)

            def X_Y(user_select):
                X = user_select.drop("user", 1)
                Y = user_select.user
                return X, Y

            X_val_ano, Y_val_ano = X_Y(user_0)

            # if scale_n == 0:
            #     result = X_val_ano
            # elif scale_n == 1:
            #     # 保存したモデルをロードする
            #     loaded_model = pickle.load(open('finalized_ss.sav', 'rb'))
            #     result = loaded_model.fit(X_val_ano)
            # elif scale_n == 2:
            #     # 保存したモデルをロードする
            #     loaded_model = pickle.load(open('finalized_mm.sav', 'rb'))
            #     result = loaded_model.fit(X_val_ano)
            # else:
            #     # 保存したモデルをロードする
            #     loaded_model = pickle.load(open('finalized_rs.sav', 'rb'))
            #     result = loaded_model.fit(X_val_ano)

            columns = ['stroke_inter', 'stroke_duration', 'start_x', 'start_y', 'stop_x',
           'stop_y', 'direct_ete_distance', 'mean_result_leng', 'direct_ete_line',
           '20_pairwise_v', '50_pairwise_v', '80_pairwise_v', '20_pairwise_acc',
           '50_pairwise_acc', '80_pairwise_acc', '3ots_m_v', 'ete_larg_deviation',
           '20_ete_line', '50_ete_line', '80_ete_line', 'ave_direction',
           'length_trajectory', 'ratio_ete', 'ave_v', '5points_m_acc',
           'm_stroke_press', 'm_stroke_area_cover', 'finger_orien',
           'cd_finger_orien', 'phone_orien', 'stroke_inter2', 'stroke_duration2', 'start_x2', 'start_y2', 'stop_x2', 'stop_y2', 'direct_ete_distance2', 'mean_result_leng2', 'direct_ete_line2', '20_pairwise_v2', '50_pairwise_v2', '80_pairwise_v2', '20_pairwise_acc2', '50_pairwise_acc2', '80_pairwise_acc2', '3ots_m_v2', 'ete_larg_deviation2', '20_ete_line2', '50_ete_line2', '80_ete_line2', 'ave_direction2', 'length_trajectory2', 'ratio_ete2', 'ave_v2', '5points_m_acc2', 'm_stroke_press2', 'm_stroke_area_cover2', 'finger_orien2','cd_finger_orien2', 'phone_orien2', 'stroke_inter_ave', 'stroke_duration_ave', 'start_x_ave', 'start_y_ave', 'stop_x_ave', 'stop_y_ave', 'direct_ete_distance_ave', 'mean_result_leng_ave', 'direct_ete_line_ave', '20_pairwise_v_ave', '50_pairwise_v_ave', '80_pairwise_v_ave', '20_pairwise_acc_ave', '50_pairwise_acc_ave', '80_pairwise_acc_ave', '3ots_m_v_ave', 'ete_larg_deviation_ave', '20_ete_line_ave', '50_ete_line_ave', '80_ete_line_ave', 'ave_direction_ave', 'length_trajectory_ave', 'ratio_ete_ave', 'ave_v_ave', '5points_m_acc_ave', 'm_stroke_press_ave', 'm_stroke_area_cover_ave', 'finger_orien_ave','cd_finger_orien_ave', 'phone_orien_ave', '2stroke_a', '2stroke_distance', '2stroke_time', '2stroke_v', 'a_stroke_inter', 'd_stroke_inter', 'outer_a', 'outer_d', 'outer_v','v_stroke_inter']

            #スケーリング
            if scale_n == 0:
                result = X_val_ano
                # print('0')
            elif scale_n == 1:
                ss = preprocessing.StandardScaler().fit(X_train)
                result1 = ss.transform(X_val_ano)
                result = pd.DataFrame(result1)
                result.columns = columns
                # print('1')
            elif scale_n == 2:
                mm = preprocessing.MinMaxScaler().fit(X_train)
                result1 = mm.transform(X_val_ano)
                result = pd.DataFrame(result1)
                result.columns = columns
                # print('2')
            else:
                rs = preprocessing.RobustScaler(quantile_range=(25., 75.)).fit(X_train)
                result1 = rs.transform(X_val_ano)
                result = pd.DataFrame(result1)
                result.columns = columns
                # print('3')

            # testデータの結合
            X_val = pd.concat([X_val_no, result]).reset_index(drop=True)
            Y_val1 = pd.concat([Y_val_no, Y_val_ano]).reset_index(drop=True)

            Y_val2 = Y_val1.copy()
            Y_val = Y_val2.replace({self.user_n:1,0:-1})

            return X_val, Y_val, X_val_no, result

        try:
            #scalling orignal
            def ori(self, X_train,X_test,X_test1,X_test2,scale_n):
                #交差検証
                for k in range(5,11,5):
                    kf = KFold(n_splits=k, shuffle=True, random_state=0)
                    models = [LocalOutlierFactor(n_neighbors=1,novelty=True,contamination=0.1),IsolationForest(n_estimators=1,contamination='auto',behaviour='new',random_state=0),OneClassSVM(nu=0.1,kernel="rbf",random_state=0),EllipticEnvelope(contamination=0.1,random_state=0)]
                    scores = {'LocalOutlierFactor':{}, 'IsolationForest':{}, 'OneClassSVM':{}, 'EllipticEnvelope':{}}
                    scores_test = {}

                    Y_true1 = self.Y_test.copy()
                    Y_true = Y_true1.replace({self.user_n:1,0:-1})

                    for model in models:
                        count = 0
                        for train_index, val_index in kf.split(X_train, self.Y_train):
                            model.fit(X_train.iloc[train_index])
                            X_val,Y_val, X_val_no, X_val_ano = hazu(self.st, X_train.iloc[val_index],self.Y_train.iloc[val_index], self.user_n, scale_n, self.X_train)
                            Y_pred = model.predict(X_val)
                            normal_result = model.predict(X_val_no)
                            anomaly_result = model.predict(X_val_ano)
                            FAR, FRR, BER = far_frr(normal_result, anomaly_result)
                            # print(accuracy_score(y_true = Y_true.iloc[val_index], y_pred = Y_pred))
                            scores[str(model).split('(')[0]][count] = {'Accuracy':accuracy_score(y_true = Y_val, y_pred = Y_pred),'Precision':precision_score(Y_val, Y_pred),'Recall':recall_score(Y_val, Y_pred),'F1':f1_score(Y_val, Y_pred),'AUC':roc_auc_score(Y_val, model.decision_function(X_val)), 'FAR':FAR, 'FRR':FRR, 'BER':BER}
                            count += 1
                            # print(train_index, val_index)
                    df_ori = pd.Panel(scores)

                    print('k分割交差検証　k=',k)
                    # import pprint
                    # pprint.pprint(scores)
                    # print(df_ori.to_frame)
                    closs = [df_ori.major_xs('Accuracy').mean(),df_ori.major_xs('Precision').mean(),df_ori.major_xs('Recall').mean(),df_ori.major_xs('F1').mean(),df_ori.major_xs('AUC').mean(),df_ori.major_xs('FAR').mean(),df_ori.major_xs('FRR').mean(),df_ori.major_xs('BER').mean()]
                    # closs.append(df_ori.major_xs('Accuracy').mean(),df_ori.major_xs('Precision').mean())
                    df_ori_mean = pd.DataFrame(closs)
                    df_ori_mean.index = ['Accuracy','Precision','Recall','F1','AUC','FAR','FRR','BER']
                    print(df_ori_mean.T)

                    #書き出し、1.5→1.10,2.5→2.10に変更で交差検証を5から10に変更
                    info = [self.user_n, self.flag_n, scale_n, X_train.shape[0], X_test.shape[0],k]
                    info_ = pd.DataFrame(info)
                    # # Series(list_).to_csv("text_pandas_to_csv.txt", index=False)
                    info_.index = ['user','stroke','scaling','train_data','test_data', 'khold']
                    info_.T.to_csv("result_multi/omake_flag/frank_multi_oc_flag_omake_user{}.csv".format(self.user_n), sep=",", mode='a',index = False)
                    df_ori_mean.T.to_csv('result_multi/omake_flag/frank_multi_oc_flag_omake_user{}.csv'.format(self.user_n), sep=",", mode = 'a')

                for model in models:
                    model.fit(X_train)
                    Y_pred = model.predict(X_test)
                    normal_result = model.predict(X_test1)
                    anomaly_result = model.predict(X_test2)
                    FAR, FRR, BER = far_frr(normal_result, anomaly_result)
                    scores_test[str(model).split('(')[0]] = {'Accuracy':accuracy_score(y_true = Y_true, y_pred = Y_pred),'Precision':precision_score(Y_true, Y_pred),'Recall':recall_score(Y_true, Y_pred),'F1':f1_score(Y_true, Y_pred),'AUC':roc_auc_score(Y_true, model.decision_function(X_test)), 'FAR':FAR, 'FRR':FRR, 'BER':BER}
                df_ori_test = pd.DataFrame(scores_test).T
                print("\nSize of training set:{}　size of test set:{}".format(X_train.shape[0], X_test.shape[0]))
                print(df_ori_test)

                #書き出し、1.5→1.10,2.5→2.10に変更で交差検証を5から10に変更
                info = [self.user_n, self.flag_n, scale_n, X_train.shape[0], X_test.shape[0]]
                info_ = pd.DataFrame(info)
                info_.index = ['user','stroke','scaling','train_data','test_data']
                info_.T.to_csv("result_multi/omake_flag/frank_multi_oc_flag_omake_user{}.csv".format(self.user_n),sep=",", mode='a',index = False)
                df_ori_test.to_csv("result_multi/omake_flag/frank_multi_oc_flag_omake_user{}.csv".format(self.user_n), sep=",", mode = 'a')
                    # df_ori.to_frame("test20191118.csv")
            if self.flag_n == 11:
                print("\nup + up", self.user_n,'番')
            elif self.flag_n == 12:
                print("\nup + down", self.user_n,'番')
            elif self.flag_n == 13:
                print("\nup + left", self.user_n,'番')
            elif self.flag_n == 14:
                print("\nup + right", self.user_n,'番')
            elif self.flag_n == 21:
                print("\ndown + up", self.user_n,'番')
            elif self.flag_n == 22:
                print("\ndown + down", self.user_n,'番')
            elif self.flag_n == 23:
                print("\ndown + left", self.user_n,'番')
            elif self.flag_n == 24:
                print("\ndown + right", self.user_n,'番')
            elif self.flag_n == 31:
                print("\nleft + up", self.user_n,'番')
            elif self.flag_n == 32:
                print("\nleft + down", self.user_n,'番')
            elif self.flag_n == 33:
                print("\nleft + left", self.user_n,'番')
            elif self.flag_n == 34:
                print("\nleft + right", self.user_n,'番')
            elif self.flag_n == 41:
                print("\nright + up", self.user_n,'番')
            elif self.flag_n == 42:
                print("\nright + down", self.user_n,'番')
            elif self.flag_n == 43:
                print("\nright + left", self.user_n,'番')
            else:
                print("\nright + right", self.user_n,'番')
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("なし","\n----------------------------------------------------------------------------------------------------")
            ori(self,self.X_train_ori,self.X_test_ori,self.X_test1_ori,self.X_test2_ori,0)
            # print(type(self.X_train_ori))
            def ss_mm_rs(self, X_train,X_test,X_test1,X_test2,scale_n):
                for k in range(5,11,5):
                    kf = KFold(n_splits=k, shuffle=True, random_state=0)
                    models = [LocalOutlierFactor(n_neighbors=1,novelty=True,contamination=0.1),IsolationForest(n_estimators=1,contamination='auto',behaviour='new',random_state=0),OneClassSVM(nu=0.1,kernel="rbf",random_state=0),EllipticEnvelope(contamination=0.1,random_state=0)]
                    scores = {'LocalOutlierFactor':{}, 'IsolationForest':{}, 'OneClassSVM':{}, 'EllipticEnvelope':{}}
                    scores_test = {}

                    Y_true1 = self.Y_test.copy()
                    Y_true = Y_true1.replace({self.user_n:1,0:-1})

                    for model in models:
                        X_train2 = pd.DataFrame(X_train,columns = ['stroke_inter', 'stroke_duration', 'start_x', 'start_y', 'stop_x',
                       'stop_y', 'direct_ete_distance', 'mean_result_leng', 'direct_ete_line',
                       '20_pairwise_v', '50_pairwise_v', '80_pairwise_v', '20_pairwise_acc',
                       '50_pairwise_acc', '80_pairwise_acc', '3ots_m_v', 'ete_larg_deviation',
                       '20_ete_line', '50_ete_line', '80_ete_line', 'ave_direction',
                       'length_trajectory', 'ratio_ete', 'ave_v', '5points_m_acc',
                       'm_stroke_press', 'm_stroke_area_cover', 'finger_orien',
                       'cd_finger_orien', 'phone_orien', 'stroke_inter2', 'stroke_duration2', 'start_x2', 'start_y2', 'stop_x2', 'stop_y2', 'direct_ete_distance2', 'mean_result_leng2', 'direct_ete_line2', '20_pairwise_v2', '50_pairwise_v2', '80_pairwise_v2', '20_pairwise_acc2', '50_pairwise_acc2', '80_pairwise_acc2', '3ots_m_v2', 'ete_larg_deviation2', '20_ete_line2', '50_ete_line2', '80_ete_line2', 'ave_direction2', 'length_trajectory2', 'ratio_ete2', 'ave_v2', '5points_m_acc2', 'm_stroke_press2', 'm_stroke_area_cover2', 'finger_orien2','cd_finger_orien2', 'phone_orien2', 'stroke_inter_ave', 'stroke_duration_ave', 'start_x_ave', 'start_y_ave', 'stop_x_ave', 'stop_y_ave', 'direct_ete_distance_ave', 'mean_result_leng_ave', 'direct_ete_line_ave', '20_pairwise_v_ave', '50_pairwise_v_ave', '80_pairwise_v_ave', '20_pairwise_acc_ave', '50_pairwise_acc_ave', '80_pairwise_acc_ave', '3ots_m_v_ave', 'ete_larg_deviation_ave', '20_ete_line_ave', '50_ete_line_ave', '80_ete_line_ave', 'ave_direction_ave', 'length_trajectory_ave', 'ratio_ete_ave', 'ave_v_ave', '5points_m_acc_ave', 'm_stroke_press_ave', 'm_stroke_area_cover_ave', 'finger_orien_ave','cd_finger_orien_ave', 'phone_orien_ave', '2stroke_a', '2stroke_distance', '2stroke_time', '2stroke_v', 'a_stroke_inter', 'd_stroke_inter', 'outer_a', 'outer_d', 'outer_v', 'v_stroke_inter'])
                        count = 0
                        for train_index, val_index in kf.split(X_train2,self.Y_train):
                            model.fit(X_train2.iloc[train_index])
                            X_val,Y_val, X_val_no, X_val_ano = hazu(self.st, X_train2.iloc[val_index], self.Y_train.iloc[val_index], self.user_n, scale_n,self.X_train)
                            Y_pred = model.predict(X_val)
                            normal_result = model.predict(X_val_no)
                            anomaly_result = model.predict(X_val_ano)
                            FAR, FRR, BER = far_frr(normal_result, anomaly_result)
                            # print(accuracy_score(y_true = Y_true.iloc[val_index], y_pred = Y_pred))
                            scores[str(model).split('(')[0]][count] = {'Accuracy':accuracy_score(y_true = Y_val, y_pred = Y_pred),'Precision':precision_score(Y_val, Y_pred),'Recall':recall_score(Y_val, Y_pred),'F1':f1_score(Y_val, Y_pred),'AUC':roc_auc_score(Y_val, model.decision_function(X_val)), 'FAR':FAR, 'FRR':FRR, 'BER':BER}
                            count += 1
                    df = pd.Panel(scores)

                    print('k分割交差検証　k=',k)
                    closs2 = [df.major_xs('Accuracy').mean(),df.major_xs('Precision').mean(),df.major_xs('Recall').mean(),df.major_xs('F1').mean(),df.major_xs('AUC').mean(),df.major_xs('FAR').mean(),df.major_xs('FRR').mean(),df.major_xs('BER').mean()]
                    df_mean = pd.DataFrame(closs2)
                    df_mean.index = ['Accuracy','Precision','Recall','F1','AUC','FAR','FRR','BER']
                    print(df_mean.T)

                    #書き出し、1.5→1.10,2.5→2.10に変更で交差検証を5から10に変更
                    info = [self.user_n, self.flag_n, scale_n, X_train.shape[0], X_test.shape[0],k]
                    info_ = pd.DataFrame(info)
                    info_.index = ['user','stroke','scaling','train_data','test_data', 'khold']
                    info_.T.to_csv("result_multi/omake_flag/frank_multi_oc_flag_omake_user{}.csv".format(self.user_n), sep=",", mode='a',index = False)
                    df_mean.T.to_csv("result_multi/omake_flag/frank_multi_oc_flag_omake_user{}.csv".format(self.user_n), sep=",", mode = 'a')

                for model in models:
                    model.fit(X_train)
                    Y_pred = model.predict(X_test)
                    normal_result = model.predict(X_test1)
                    anomaly_result = model.predict(X_test2)
                    FAR, FRR, BER = far_frr(normal_result, anomaly_result)
                    scores_test[str(model).split('(')[0]] = {'Accuracy':accuracy_score(y_true = Y_true, y_pred = Y_pred),'Precision':precision_score(Y_true, Y_pred),'Recall':recall_score(Y_true, Y_pred),'F1':f1_score(Y_true, Y_pred),'AUC':roc_auc_score(Y_true, model.decision_function(X_test)), 'FAR':FAR, 'FRR':FRR, 'BER':BER}
                df_test = pd.DataFrame(scores_test).T
                print(print("\nSize of training set:{}　size of test set:{}".format(X_train.shape[0], X_test.shape[0])))
                print(df_test)

                # #書き出し、1.5→1.10,2.5→2.10に変更で交差検証を5から10に変更
                info = [self.user_n, self.flag_n, scale_n, X_train.shape[0], X_test.shape[0]]
                info_ = pd.DataFrame(info)
                info_.index = ['user','stroke','scaling','train_data','test_data']
                info_.T.to_csv("result_multi/omake_flag/frank_multi_oc_flag_omake_user{}.csv".format(self.user_n), sep=",", mode='a',index = False)
                df_test.to_csv("result_multi/omake_flag/frank_multi_oc_flag_omake_user{}.csv".format(self.user_n), sep=",", mode = 'a')

            print("====================================================================================================")
            print("標準化：StandardScaler","\n----------------------------------------------------------------------------------------------------")
            ss_mm_rs(self,self.X_train_ss,self.X_test_ss,self.X_test1_ss,self.X_test2_ss,1)
            print("====================================================================================================")
            print("正規化：MinMaxScaler","\n----------------------------------------------------------------------------------------------------")
            ss_mm_rs(self,self.X_train_mm,self.X_test_mm,self.X_test1_mm,self.X_test2_mm,2)
            print("====================================================================================================")
            print("外れ値に頑強な標準化：RobustScaler","\n----------------------------------------------------------------------------------------------------")
            ss_mm_rs(self,self.X_train_rs,self.X_test_rs,self.X_test1_rs,self.X_test2_rs,3)
        except AttributeError:
            print('None')

    def clf_models(self):

        # kfolds = []
        # kfolds.append(KFold(n_splits=5, shuffle=True, random_state=0))
        # kfolds.append(StratifiedKFold(n_splits=5))

        models = [LocalOutlierFactor(n_neighbors=1,novelty=True,contamination=0.1),IsolationForest(n_estimators=1,contamination='auto',behaviour='new',random_state=0),svm.OneClassSVM(nu=0.1,kernel="rbf",random_state=0),EllipticEnvelope(contamination=0.1,random_state=0)]
        scores = {}
        scores_test = {}

        Y_true1 = self.Y_test.copy()
        Y_true = Y_true1.replace({self.user_n:1,0:-1})

        def far_frr(normal_result, anomaly_result):
            TP = np.count_nonzero(normal_result == 1)
            FN = np.count_nonzero(normal_result == -1)
            FP = np.count_nonzero(anomaly_result == 1)
            TN = np.count_nonzero(anomaly_result == -1)
            FRR = FN / (FN + TP)
            FAR = FP / (TN + FP)
            BER = 0.5 * (FP / (TN + FP) + FN / (FN + TP))
            return FRR, FAR, BER

        # loo=LeaveOneOut()
        # print("scaling orignal")
        for model in models:
            # scores[str(model).split('(')[0]] = cross_val_score(model, self.X_train_ori, self.Y_train, cv=kfold)
            # score = cross_val_score(model, self.X_train_ori, None, cv = loo)
            # fff = score.mean()
            model.fit(self.X_train_ori)
            Y_pred = model.predict(self.X_test_ori)
            normal_result = model.predict(self.X_test1_ori)
            anomaly_result = model.predict(self.X_test2_ori)
            FAR, FRR, BER = far_frr(normal_result, anomaly_result)
            scores_test[str(model).split('(')[0]] = {'Accuracy':accuracy_score(y_true = Y_true, y_pred = Y_pred),'Precision':precision_score(Y_true, Y_pred),'Recall':recall_score(Y_true, Y_pred),'F1':f1_score(Y_true, Y_pred),'AUC':roc_auc_score(Y_true, model.decision_function(self.X_test_ori)), 'FAR':FAR, 'FRR':FRR, 'BER':BER}
            # df_ori = pd.DataFrame(scores)
            df_ori_test = pd.DataFrame(scores_test).T

        # print("scaling StandardScaler")
        for model in models:
            model.fit(self.X_train_ss)
            Y_pred = model.predict(self.X_test_ss)
            normal_result = model.predict(self.X_test1_ss)
            anomaly_result = model.predict(self.X_test2_ss)
            FAR, FRR, BER = far_frr(normal_result, anomaly_result)
            scores_test[str(model).split('(')[0]] = {'Accuracy':accuracy_score(y_true = Y_true, y_pred = Y_pred),'Precision':precision_score(Y_true, Y_pred),'Recall':recall_score(Y_true, Y_pred),'F1':f1_score(Y_true, Y_pred),'AUC':roc_auc_score(Y_true, model.decision_function(self.X_test_ss)), 'FAR':FAR, 'FRR':FRR, 'BER':BER}
            df_ss_test = pd.DataFrame(scores_test).T

        # print("scaling MinMaxScaler")
        for model in models:
            model.fit(self.X_train_mm)
            Y_pred = model.predict(self.X_test_mm)
            normal_result = model.predict(self.X_test1_mm)
            anomaly_result = model.predict(self.X_test2_mm)
            FAR, FRR, BER = far_frr(normal_result, anomaly_result)
            scores_test[str(model).split('(')[0]] = {'Accuracy':accuracy_score(y_true = Y_true, y_pred = Y_pred),'Precision':precision_score(Y_true, Y_pred),'Recall':recall_score(Y_true, Y_pred),'F1':f1_score(Y_true, Y_pred),'AUC':roc_auc_score(Y_true, model.decision_function(self.X_test_mm)), 'FAR':FAR, 'FRR':FRR, 'BER':BER}
            df_mm_test = pd.DataFrame(scores_test).T

        # print("scaling RobustScaler")
        for model in models:
            model.fit(self.X_train_rs)
            Y_pred = model.predict(self.X_test_rs)
            normal_result = model.predict(self.X_test1_rs)
            anomaly_result = model.predict(self.X_test2_rs)
            FAR, FRR, BER = far_frr(normal_result, anomaly_result)
            scores_test[str(model).split('(')[0]] = {'Accuracy':accuracy_score(y_true = Y_true, y_pred = Y_pred),'Precision':precision_score(Y_true, Y_pred),'Recall':recall_score(Y_true, Y_pred),'F1':f1_score(Y_true, Y_pred),'AUC':roc_auc_score(Y_true, model.decision_function(self.X_test_rs)), 'FAR':FAR, 'FRR':FRR, 'BER':BER}
            df_rs_test = pd.DataFrame(scores_test).T

        if self.flag_n == 1:
            print("\nup", self.user_n,'番')
        elif self.flag_n == 2:
            print("\ndown", self.user_n,'番')
        elif self.flag_n == 3:
            print("\nleft", self.user_n,'番')
        else:
            print("\nright", self.user_n,'番')
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("なし","\n----------------------------------------------------------------------------------------------------")
        print(df_ori_test)
        # print(fff)
        print("====================================================================================================")
        print("標準化：StandardScaler","\n----------------------------------------------------------------------------------------------------")
        print(df_ss_test)
        print("====================================================================================================")
        print("正規化：MinMaxScaler","\n----------------------------------------------------------------------------------------------------")
        print(df_mm_test)
        print("====================================================================================================")
        print("外れ値に頑強な標準化：RobustScaler","\n----------------------------------------------------------------------------------------------------")
        print(df_rs_test)
        # print("~~~~~~~~~~~~~~~~~~~~~~")
        # print("なし_test","\n-------------------------------------")
        # print("交差検証\n",df_ori_test,"\n層化ｋ分割交差検証\n",df_oriSK_test)
        # print("======================")
        # print("標準化：StandardScaler","\n-------------------------------------")
        # print("交差検証\n",df_ss_test,"\n層化ｋ分割交差検証\n",df_ssSK_test)
        # print("======================")
        # print("正規化：MinMaxScaler","\n-------------------------------------")
        # print("交差検証\n",df_mm_test,"\n層化ｋ分割交差検証\n",df_mmSK_test)
        # print("======================")
        # print("外れ値に頑強な標準化：RobustScaler","\n-------------------------------------")
        # print("交差検証\n",df_rs_test,"\n層化ｋ分割交差検証\n",df_rsSK_test)

        def df_hozon(df_ori, df_oriSK, df_ss, df_ssSK, df_mm, df_mmSK, df_rs, df_rsSK, df_ori_test, df_oriSK_test, df_ss_test, df_ssSK_test, df_mm_test, df_mmSK_test, df_rs_test, df_rsSK_test):
            df_ori.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_ori.mean().to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_oriSK.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_oriSK.mean().to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_ss.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_ss.mean().to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_ssSK.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_ssSK.mean().to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_mm.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_mm.mean().to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_mmSK.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_mmSK.mean().to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_rs.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_rs.mean().to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_rsSK.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_rsSK.mean().to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')

            df_ori_test.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_oriSK_test.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_ss_test.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_ssSK_test.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_mm_test.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_mmSK_test.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_rs_test.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
            df_rsSK_test.to_csv("one_shikibetu/k_one_shikibetu{0}_{1}.csv".format(self.select_n, self.n),sep=",", mode='a')
        # df_hozon(df_ori, df_oriSK, df_ss, df_ssSK, df_mm, df_mmSK, df_rs, df_rsSK, df_ori_test, df_oriSK_test, df_ss_test, df_ssSK_test, df_mm_test, df_mmSK_test, df_rs_test, df_rsSK_test)

    def LOF(self):
        print('\n選択した方向：', self.flag_n,'選択したUser：',self.user_n)
        for (train,test, test1, test2, name) in zip(self.trains, self.tests, self.tests1,self.tests2, self.names):
            print('------------------------------------------------------------------------------------------------')
            print(str(name).split('_')[0],'\n------------------------------------------------------------------------------------------------\n')
            Y_true1 = self.Y_test.copy()
            Y_true = Y_true1.replace({self.user_n:1,0:-1})

            lof = LocalOutlierFactor(n_neighbors=1,
                               novelty=True,
                                #外れ値の割合を示す引数
                               contamination=0.1)
            lof.fit(train) # train_dataは正常データが大多数であるような訓練データ
            prediction = lof.predict(test) # テストデータに対する予測
            score = lof.score_samples(test) # テストデータの異常度
            score1 = lof.score_samples(test1)
            score2 = lof.score_samples(test2)

            y_score = lof.decision_function(test)

            normal_result = lof.predict(test1)
            anomaly_result = lof.predict(test2)

            #評価
            result(normal_result, anomaly_result, Y_true, prediction, y_score)

            print('\nパラメータの調整\n-------------------------------------------------------------')
            train_X, test_X, train_Y, test_Y = train_test_split(
        train, self.Y_train, test_size=0.2,random_state=0, shuffle=True)
            print("Size of training set:{}　size of test set:{}".format(train_X.shape[0], test_X.shape[0]))

            best_score = 0
            algorithms = ['ball_tree','kd_tree','brute']

            for n_neighbors in tqdm(range(1,35,1)):
                for p in range(1,3,1):
                    for algorithm in algorithms:
                        for leaf_size in range(1,30,1):
                            lof = LocalOutlierFactor(n_neighbors = n_neighbors, p = p, algorithm = algorithm, leaf_size = leaf_size, novelty=True,contamination=0.1,n_jobs=-1)
                            lof.fit(train_X)
                            pred = lof.predict(test_X)
                            Y_true2 = test_Y.copy()
                            true_Y = Y_true2.replace({self.user_n:1,0:-1})
                            score = accuracy_score(true_Y, pred)

                            if score > best_score:
                                best_score = score
                                best_parameters = {'n_neighbors':n_neighbors, 'p': p, 'algorithm':algorithm,'leaf_size':leaf_size,'novelty':True, 'contamination':0.1, 'n_jobs':-1}
            print("Best score: {:.4f}".format(best_score))
            print("Best parameters: {}".format(best_parameters))

            lof = LocalOutlierFactor(**best_parameters)
            lof.fit(train)
            pred = lof.predict(test)
            test_score = accuracy_score(Y_true, pred)
            print("Best score on validation set:{:.4f}".format(best_score))
            print("Best parameters:",best_parameters)
            print("Test set score with best parameters:{:.4f}\n".format(test_score))

            score = lof.score_samples(test) # テストデータの異常度
            score1 = lof.score_samples(test1)
            score2 = lof.score_samples(test2)

            y_score = lof.decision_function(test)

            normal_result = lof.predict(test1)
            anomaly_result = lof.predict(test2)

            #評価
            result(normal_result, anomaly_result, Y_true, pred, y_score)
            print('\n============================================================\n')


        Y_true1 = self.Y_test.copy()
        Y_true = Y_true1.replace({self.user_n:1,0:-1})
        k_range = range(1, 30)
        accuracy_ori = []
        for k in k_range:
            lof = LocalOutlierFactor(n_neighbors=k,
                               novelty=True,
                               contamination=0.1)
            lof.fit(self.X_train_ori) # train_dataは正常データが大多数であるような訓練データ
            prediction = lof.predict(self.X_test_ori) # テストデータに対する予測
            accuracy_ori.append(accuracy_score(Y_true, prediction))

        accuracy_ss = []
        for k in k_range:
            lof = LocalOutlierFactor(n_neighbors=k,
                               novelty=True,
                               contamination=0.1)
            lof.fit(self.X_train_ss) # train_dataは正常データが大多数であるような訓練データ
            prediction = lof.predict(self.X_test_ss) # テストデータに対する予測
            accuracy_ss.append(accuracy_score(Y_true, prediction))

        accuracy_mm = []
        for k in k_range:
            lof = LocalOutlierFactor(n_neighbors=k,
                               novelty=True,
                               contamination=0.1)
            lof.fit(self.X_train_mm) # train_dataは正常データが大多数であるような訓練データ
            prediction = lof.predict(self.X_test_mm) # テストデータに対する予測
            accuracy_mm.append(accuracy_score(Y_true, prediction))

        accuracy_rs = []
        for k in k_range:
            lof = LocalOutlierFactor(n_neighbors=k,
                               novelty=True,
                               contamination=0.1)
            lof.fit(self.X_train_rs) # train_dataは正常データが大多数であるような訓練データ
            prediction = lof.predict(self.X_test_rs) # テストデータに対する予測
            accuracy_rs.append(accuracy_score(Y_true, prediction))

        plt.figure(figsize=(8,6),dpi=200)
        plt.plot(k_range, accuracy_ori, label="ori")
        plt.plot(k_range, accuracy_ss, label="ss")
        plt.plot(k_range, accuracy_mm, label="mm")
        plt.plot(k_range, accuracy_rs, label="rs")

        plt.legend()

        plt.title('LOF_n_neighbors_change contamination=0.1')
        plt.xlabel('K for LOF')
        plt.ylabel('Testing Accuracy')
        plt.style.use('default')

        plt.show()

        accuracy_ori2 = []
        for k in k_range:
            lof = LocalOutlierFactor(n_neighbors=k,
                               novelty=True,
                               contamination=0.05)
            lof.fit(self.X_train_ori) # train_dataは正常データが大多数であるような訓練データ
            prediction = lof.predict(self.X_test_ori) # テストデータに対する予測
            accuracy_ori2.append(accuracy_score(Y_true, prediction))

        accuracy_ss2 = []
        for k in k_range:
            lof = LocalOutlierFactor(n_neighbors=k,
                               novelty=True,
                               contamination=0.05)
            lof.fit(self.X_train_ss) # train_dataは正常データが大多数であるような訓練データ
            prediction = lof.predict(self.X_test_ss) # テストデータに対する予測
            accuracy_ss2.append(accuracy_score(Y_true, prediction))

        accuracy_mm2 = []
        for k in k_range:
            lof = LocalOutlierFactor(n_neighbors=k,
                               novelty=True,
                               contamination=0.05)
            lof.fit(self.X_train_mm) # train_dataは正常データが大多数であるような訓練データ
            prediction = lof.predict(self.X_test_mm) # テストデータに対する予測
            accuracy_mm2.append(accuracy_score(Y_true, prediction))

        accuracy_rs2 = []
        for k in k_range:
            lof = LocalOutlierFactor(n_neighbors=k,
                               novelty=True,
                               contamination=0.05)
            lof.fit(self.X_train_rs) # train_dataは正常データが大多数であるような訓練データ
            prediction = lof.predict(self.X_test_rs) # テストデータに対する予測
            accuracy_rs2.append(accuracy_score(Y_true, prediction))

        plt.figure(figsize=(8,6),dpi=200)
        plt.plot(k_range, accuracy_ori2, label="ori")
        plt.plot(k_range, accuracy_ss2, label="ss")
        plt.plot(k_range, accuracy_mm2, label="mm")
        plt.plot(k_range, accuracy_rs2, label="rs")

        plt.legend()

        plt.title('LOF_n_neighbors_change contamination=0.05')
        plt.xlabel('K for LOF')
        plt.ylabel('Testing Accuracy')
        plt.style.use('default')

        plt.show()

    def IF(self):
        print('\n選択した方向：', self.flag_n,'選択したUser：',self.user_n)
        for (train, test, test1, test2, name) in zip(self.trains, self.tests, self.tests1,self.tests2, self.names):
            print('------------------------------------------------------------------------------------------------')
            print(str(name).split('_')[0],'\n------------------------------------------------------------------------------------------------\n')
            Y_true1 = self.Y_test.copy()
            Y_true = Y_true1.replace({self.user_n:1,0:-1})

            isf = IsolationForest(n_estimators=1,
                              contamination='auto',
                              behaviour='new',random_state=0)
            isf.fit(train)
            prediction = isf.predict(test)
            normal_result = isf.predict(test1)
            anomaly_result = isf.predict(test2)

            y_score = isf.decision_function(test)

            #評価
            result(normal_result, anomaly_result, Y_true, prediction, y_score)

        k_range = range(1, 90)
        accuracy_isf = []
        for k in tqdm(k_range):
            isf = IsolationForest(n_estimators=1,
                              contamination='auto',
                              behaviour='new',random_state=k)
            isf.fit(self.X_train_ori)
            pred = isf.predict(self.X_test_ori)
            accuracy_isf.append(accuracy_score(Y_true, pred))

        plt.figure(figsize=(8,5),dpi=200)
        plt.plot(k_range, accuracy_isf, label="ori")

        plt.legend()

        plt.title('IF randam_state_change')
        plt.xlabel('K for IF')
        plt.ylabel('Testing Accuracy')
        plt.style.use('default')

        plt.show()

        k_range = range(1, 90)
        accuracy_isf_n = []
        for k in tqdm(k_range):
            isf = IsolationForest(n_estimators=k,
                              contamination='auto',
                              behaviour='new',random_state=0,max_samples="auto",)
            isf.fit(self.X_train_ori)
            pred = isf.predict(self.X_test_ori)
            accuracy_isf_n.append(accuracy_score(Y_true, pred))

        plt.figure(figsize=(6,4),dpi=200)
        plt.plot(k_range, accuracy_isf_n, label="ori")

        plt.legend()

        plt.title('IF n_estimators_change')
        plt.xlabel('K for IF')
        plt.ylabel('Testing Accuracy')
        plt.style.use('default')

        plt.show()

#         param_gs_isf ={
#             'n_estimators':np.arange(1,50,1),
#             'random_state':np.arange(0,50,1),
#         }
        print('スケーリングなしを使用(params_change)\n----------------------------------------------------------')
        train_X, test_X, train_Y, test_Y = train_test_split(self.X_train_ori, self.Y_train, test_size=0.2,random_state=0, shuffle=True)
        print("Size of training set:{}　size of test set:{}".format(train_X.shape[0], test_X.shape[0]))

        best_score = 0

        for n_estimators in tqdm(range(1,50,1)):
            for random_state in range(0,50,1):
                isf = IsolationForest(n_estimators = n_estimators, random_state = random_state, contamination=0.1)
                isf.fit(train_X)
                pred = isf.predict(test_X)
                Y_true2 = test_Y.copy()
                true_Y = Y_true2.replace({self.user_n:1,0:-1})
#                 scores = cross_val_score(isf, train_X,train_Y,cv=5)
#                 # 交差検証の平均値を計算
#                 score = np.mean(scores)
                score = accuracy_score(true_Y, pred)

                if score > best_score:
                    best_score = score
                    best_parameters = {'n_estimators':n_estimators, 'random_state':random_state,'contamination':0.1}
        print("Best score: {:.4f}".format(best_score))
        print("Best parameters: {}".format(best_parameters))

        new_isf = IsolationForest(**best_parameters)
        new_isf.fit(self.X_train_ori)
        pred = new_isf.predict(self.X_test_ori)
        test_score = accuracy_score(Y_true, pred)
        print("Best score on validation set:{:.4f}".format(best_score))
        print("Best parameters:",best_parameters)
        print("Test set score with best parameters:{:.4f}".format(test_score))

        normal_result = new_isf.predict(self.X_test1_ori)
        anomaly_result = new_isf.predict(self.X_test2_ori)

        y_score = new_isf.decision_function(self.X_test_ori)

        #評価
        result(normal_result, anomaly_result, Y_true, pred, y_score)
        print('\n============================================================\n')

    def OCSVM(self):
        Y_true1 = self.Y_test.copy()
        Y_true = Y_true1.replace({self.user_n:1,0:-1})
        print('\n選択した方向：', self.flag_n,'選択したUser：',self.user_n)
        for (train,test, test1, test2, name) in zip(self.trains, self.tests, self.tests1,self.tests2, self.names):
            print('------------------------------------------------------------------------------------------------')
            print(str(name).split('_')[0],'\n------------------------------------------------------------------------------------------------\n')
            clf = svm.OneClassSVM(nu=0.1,
                              kernel="rbf",
                              gamma=0.03,
                              # gamma='auto',
                              random_state=0)
            clf.fit(train)
            pred = clf.predict(test)
            normal_result = clf.predict(test1)
            anomaly_result = clf.predict(test2)

            y_score = clf.decision_function(test)

            #評価
            result(normal_result, anomaly_result, Y_true, pred, y_score)
#             print('\n============================================================\n')

            print('\nパラメータの調整\n-------------------------------------------------------------')
            train_X, test_X, train_Y, test_Y = train_test_split(
        train, self.Y_train, test_size=0.2,random_state=0, shuffle=True)
            print("Size of training set:{}　size of test set:{}".format(train_X.shape[0], test_X.shape[0]))

            best_score = 0
            kernels = ['linear','rbf','poly','sigmoid']

            for kernel in tqdm(kernels):
                for gamma in np.linspace(0.01, 1, 100):
                    for coef0 in np.linspace(0, 1, 100):
                        clf = svm.OneClassSVM(kernel = kernel, gamma = gamma, coef0 = coef0, nu=0.1)
                        clf.fit(train_X)
                        pred = clf.predict(test_X)
                        Y_true2 = test_Y.copy()
                        true_Y = Y_true2.replace({self.user_n:1,0:-1})
                        score = accuracy_score(true_Y, pred)

                        if score > best_score:
                            best_score = score
                            best_parameters = {'kernel':kernel, 'gamma': gamma, 'coef0':coef0,'nu':0.1}
            print("Best score: {:.4f}".format(best_score))
            print("Best parameters: {}".format(best_parameters))

            new_clf = svm.OneClassSVM(**best_parameters)
            new_clf.fit(train)
            pred = new_clf.predict(test)
            test_score = accuracy_score(Y_true, pred)
            print("Best score on validation set:{:.4f}".format(best_score))
            print("Best parameters:",best_parameters)
            print("Test set score with best parameters:{:.4f}\n".format(test_score))

            normal_result = new_clf.predict(test1)
            anomaly_result = new_clf.predict(test2)

            y_score = new_clf.decision_function(test)

            #評価
            result(normal_result, anomaly_result, Y_true, pred, y_score)
            print('\n============================================================\n')



#             kfolds = []
#             kfolds.append(KFold(n_splits=5, shuffle=True, random_state=0))
#             kfolds.append(StratifiedKFold(n_splits=5))

#             for kfold in tqdm(kfolds):
#                 print('グリッドサーチ', str(kfold).split('(')[0])
#                 ocsvm = svm.OneClassSVM()
#                 param_gs = {
#                     'kernel':['rbf'],
#                     'gamma':np.arange(0.01,1,0.005),
#                     'coef0':np.arange(0.1,10,0.5),
#                     'nu':[0.1]
#                 }
#                 grid_search = GridSearchCV(ocsvm, param_gs, cv=kfold)
#                 start = time.time()
#                 grid_search.fit(train, self.Y_train)
#                 print("time : ",time.time() - start)
#                 print('best_estimator:', grid_search.best_estimator_)
#                 print('best_params_:',grid_search.best_params_)
#                 print('best_score_:',grid_search.best_score_)

#                 # 性能評価
#                 print('\n訓練データでのグリッドサーチの評価')
#                 pred = cross_val_predict(grid_search, train, self.Y_train, cv=kfold)
#                 conf_mat = confusion_matrix(self.Y_train,pred)

#                 new_clf = svm.OneClassSVM(**grid_search.best_params_)
#                 new_clf.fit(train)
#                 print("time : ",time.time() - start)
#                 prediction= new_clf.predict(test)
#                 test_score = accuracy_score(Y_true, prediction)


#             best_score = 0
# #             kernels = ['linear', 'poly','sigmoid','rbf']
#             kernels = ['rbf','linear']

#             for kernel in tqdm(kernels):
#                 for gamma in np.linspace(0.01, 10, 150):
#                     for coef0 in np.linspace(0.1, 10, 10):
#                         clf = svm.OneClassSVM(kernel = kernel, gamma = gamma, coef0 = coef0, nu=0.1)
#                         clf.fit(X_train_val)
#                         pred = clf.predict(X_test_val)
#                         Y_true2 = Y_test_val.copy()
#                         true_Y = Y_true2.replace({self.user_n:1,0:-1})
#                         score = accuracy_score(true_Y, pred)

#                         if score > best_score:
#                             best_score = score
#                             best_parameters = {'kernel':kernel, 'gamma': gamma, 'coef0':coef0,'nu':0.1}

#             print("Best score: {:.4f}".format(best_score))
#             print("Best parameters: {}".format(best_parameters))

#             new_clf = svm.OneClassSVM(**best_parameters)
#             new_clf.fit(train)
#             prediction= new_clf.predict(test)
#             test_score = accuracy_score(Y_true, pred)
#             print("Best score on validation set:{:.4f}".format(best_score))
#             print("Best parameters:",best_parameters)
#             print("Test set score with best parameters:{:.4f}\n".format(test_score))

#             normal_result = new_clf.predict(test1)
#             anomaly_result = new_clf.predict(test2)

#             y_score = new_clf.decision_function(test)

#             評価
#             result(normal_result, anomaly_result, Y_true, prediction, y_score)
#             print('\n============================================================\n')

        k_range = np.linspace(0.01, 1.0, 1000)
        accuracy_ori = []
        for k in k_range:
            clf = svm.OneClassSVM(nu=0.1,
                              kernel="rbf",
                              gamma=k,random_state=0)
            clf.fit(self.X_train_ori)
            pred = clf.predict(self.X_test_ori)
            accuracy_ori.append(accuracy_score(y_true= Y_true, y_pred=pred))

        accuracy_ss = []
        for k in k_range:
            clf = svm.OneClassSVM(nu=0.1,
                              kernel="rbf",
                              gamma=k,random_state=0)
            clf.fit(self.X_train_ss)
            pred = clf.predict(self.X_test_ss)
            accuracy_ss.append(accuracy_score(y_true = Y_true, y_pred=pred))

        accuracy_mm = []
        for k in k_range:
            clf = svm.OneClassSVM(nu=0.1,
                              kernel="rbf",
                              gamma=k,random_state=0)
            clf.fit(self.X_train_mm)
            pred = clf.predict(self.X_test_mm)
            accuracy_mm.append(accuracy_score(y_true= Y_true, y_pred=pred))

        accuracy_rs = []
        for k in k_range:
            clf = svm.OneClassSVM(nu=0.1,
                              kernel="rbf",
                              gamma=k,random_state=0)
            clf.fit(self.X_train_rs)
            pred = clf.predict(self.X_test_rs)
            accuracy_rs.append(accuracy_score(y_true= Y_true, y_pred=pred))

        plt.figure(figsize=(8,6),dpi=200)
        plt.plot(k_range, accuracy_ori, label="ori")
        plt.plot(k_range, accuracy_ss, label="ss")
        plt.plot(k_range, accuracy_mm, label="mm")
        plt.plot(k_range, accuracy_rs, label="rs")

        plt.legend()

        plt.title('OCSVM gamma_change')
        plt.xlabel('K for OCSVM')
        plt.ylabel('Testing Accuracy')
        plt.style.use('default')

        plt.show()

    def EE(self):
        Y_true1 = self.Y_test.copy()
        Y_true = Y_true1.replace({self.user_n:1,0:-1})
        print('\n選択した方向：', self.flag_n,'選択したUser：',self.user_n)
        for (train,test, test1, test2, name) in zip(self.trains, self.tests, self.tests1,self.tests2, self.names):
            print('------------------------------------------------------------------------------------------------')
            print(str(name).split('_')[0],'\n------------------------------------------------------------------------------------------------\n')
            ee = EllipticEnvelope(contamination=0.1,random_state=0)
            ee.fit(train)
            prediction = ee.predict(test)
            normal_result = ee.predict(test1)
            anomaly_result = ee.predict(test2)

            y_score = ee.decision_function(test)

            #評価
            result(normal_result, anomaly_result, Y_true, prediction, y_score)

        k_range = range(1,90,1)
        accuracy_ori = []
        for k in k_range:
            clf = EllipticEnvelope(contamination=0.1,random_state=k)
            clf.fit(self.X_train_ori)
            pred = clf.predict(self.X_test_ori)
            accuracy_ori.append(accuracy_score(y_true= Y_true, y_pred=pred))

        accuracy_ss = []
        for k in k_range:
            clf = EllipticEnvelope(contamination=0.1,random_state=k)
            clf.fit(self.X_train_ss)
            pred = clf.predict(self.X_test_ss)
            accuracy_ss.append(accuracy_score(y_true = Y_true, y_pred=pred))

        accuracy_mm = []
        for k in k_range:
            clf = EllipticEnvelope(contamination=0.1,random_state=k)
            clf.fit(self.X_train_mm)
            pred = clf.predict(self.X_test_mm)
            accuracy_mm.append(accuracy_score(y_true= Y_true, y_pred=pred))

        accuracy_rs = []
        for k in k_range:
            clf = EllipticEnvelope(contamination=0.1,random_state=k)
            clf.fit(self.X_train_rs)
            pred = clf.predict(self.X_test_rs)
            accuracy_rs.append(accuracy_score(y_true= Y_true, y_pred=pred))

        plt.figure(figsize=(8,6),dpi=200)
        plt.plot(k_range, accuracy_ori, label="ori")
        plt.plot(k_range, accuracy_ss, label="ss")
        plt.plot(k_range, accuracy_mm, label="mm")
        plt.plot(k_range, accuracy_rs, label="rs")

        plt.legend()

        plt.title('EE random_state_change')
        plt.xlabel('K for EE')
        plt.ylabel('Testing Accuracy')
        plt.style.use('default')

        plt.show()

experiment11 = Experiment3(X11_train_ori, X11_test_ori, X11_test1_ori, X11_test2_ori,X11_train_ss, X11_test_ss,X11_test1_ss, X11_test2_ss,X11_train_mm, X11_test_mm,X11_test1_mm, X11_test2_mm, X11_train_rs, X11_test_rs,X11_test1_rs,X11_test2_rs,Y11_train, Y11_test, y11_train, y11_test,11,user_n,uu,X_train_uu, Y11_test1, Y11_test2)
experiment12 = Experiment3(X12_train_ori, X12_test_ori, X12_test1_ori, X12_test2_ori,X12_train_ss, X12_test_ss,X12_test1_ss, X12_test2_ss,X12_train_mm, X12_test_mm,X12_test1_mm, X12_test2_mm, X12_train_rs, X12_test_rs,X12_test1_rs,X12_test2_rs,Y12_train, Y12_test, y12_train, y12_test,12,user_n,ud,X_train_ud, Y12_test1, Y12_test2)
experiment13 = Experiment3(X13_train_ori, X13_test_ori, X13_test1_ori, X13_test2_ori,X13_train_ss, X13_test_ss,X13_test1_ss, X13_test2_ss,X13_train_mm, X13_test_mm,X13_test1_mm, X13_test2_mm, X13_train_rs, X13_test_rs,X13_test1_rs,X13_test2_rs,Y13_train, Y13_test, y13_train, y13_test,13,user_n,ul,X_train_ul, Y13_test1, Y13_test2)
experiment14 = Experiment3(X14_train_ori, X14_test_ori, X14_test1_ori, X14_test2_ori,X14_train_ss, X14_test_ss,X14_test1_ss, X14_test2_ss,X14_train_mm, X14_test_mm,X14_test1_mm, X14_test2_mm, X14_train_rs, X14_test_rs,X14_test1_rs,X14_test2_rs,Y14_train, Y14_test, y14_train, y14_test,14,user_n,ur,X_train_ur, Y14_test1, Y14_test2)

experiment21 = Experiment3(X21_train_ori, X21_test_ori, X21_test1_ori, X21_test2_ori,X21_train_ss, X21_test_ss,X21_test1_ss, X21_test2_ss,X21_train_mm, X21_test_mm,X21_test1_mm, X21_test2_mm, X21_train_rs, X21_test_rs,X21_test1_rs,X21_test2_rs,Y21_train, Y21_test, y21_train, y21_test,21,user_n,du,X_train_du, Y21_test1, Y21_test2)
experiment22 = Experiment3(X22_train_ori, X22_test_ori, X22_test1_ori, X22_test2_ori,X22_train_ss, X22_test_ss,X22_test1_ss, X22_test2_ss,X22_train_mm, X22_test_mm,X22_test1_mm, X22_test2_mm, X22_train_rs, X22_test_rs,X22_test1_rs,X22_test2_rs,Y22_train, Y22_test, y22_train, y22_test,22,user_n,dd,X_train_dd, Y22_test1, Y22_test2)
experiment23 = Experiment3(X23_train_ori, X23_test_ori, X23_test1_ori, X23_test2_ori,X23_train_ss, X23_test_ss,X23_test1_ss, X23_test2_ss,X23_train_mm, X23_test_mm,X23_test1_mm, X23_test2_mm, X23_train_rs, X23_test_rs,X23_test1_rs,X23_test2_rs,Y23_train, Y23_test, y23_train, y23_test,23,user_n,dl,X_train_dl, Y23_test1, Y23_test2)
experiment24 = Experiment3(X24_train_ori, X24_test_ori, X24_test1_ori, X24_test2_ori,X24_train_ss, X24_test_ss,X24_test1_ss, X24_test2_ss,X24_train_mm, X24_test_mm,X24_test1_mm, X24_test2_mm, X24_train_rs, X24_test_rs,X24_test1_rs,X24_test2_rs,Y24_train, Y24_test, y24_train, y24_test,24,user_n,dr,X_train_dr, Y24_test1, Y24_test2)

experiment31 = Experiment3(X31_train_ori, X31_test_ori, X31_test1_ori, X31_test2_ori,X31_train_ss, X31_test_ss,X31_test1_ss, X31_test2_ss,X31_train_mm, X31_test_mm,X31_test1_mm, X31_test2_mm, X31_train_rs, X31_test_rs,X31_test1_rs,X31_test2_rs,Y31_train, Y31_test, y31_train, y31_test,31,user_n,lu,X_train_lu, Y31_test1, Y31_test2)
experiment32 = Experiment3(X32_train_ori, X32_test_ori, X32_test1_ori, X32_test2_ori,X32_train_ss, X32_test_ss,X32_test1_ss, X32_test2_ss,X32_train_mm, X32_test_mm,X32_test1_mm, X32_test2_mm, X32_train_rs, X32_test_rs,X32_test1_rs,X32_test2_rs,Y32_train, Y32_test, y32_train, y32_test,32,user_n,ld,X_train_ld, Y32_test1, Y32_test2)
experiment33 = Experiment3(X33_train_ori, X33_test_ori, X33_test1_ori, X33_test2_ori,X33_train_ss, X33_test_ss,X33_test1_ss, X33_test2_ss,X33_train_mm, X33_test_mm,X33_test1_mm, X33_test2_mm, X33_train_rs, X33_test_rs,X33_test1_rs,X33_test2_rs,Y33_train, Y33_test, y33_train, y33_test,33,user_n,ll,X_train_ll, Y33_test1, Y33_test2)
experiment34 = Experiment3(X34_train_ori, X34_test_ori, X34_test1_ori, X34_test2_ori,X34_train_ss, X34_test_ss,X34_test1_ss, X34_test2_ss,X34_train_mm, X34_test_mm,X34_test1_mm, X34_test2_mm, X34_train_rs, X34_test_rs,X34_test1_rs,X34_test2_rs,Y34_train, Y34_test, y34_train, y34_test,34,user_n,lr,X_train_lr, Y34_test1, Y34_test2)

experiment41 = Experiment3(X41_train_ori, X41_test_ori, X41_test1_ori, X41_test2_ori,X41_train_ss, X41_test_ss,X41_test1_ss, X41_test2_ss,X41_train_mm, X41_test_mm,X41_test1_mm, X41_test2_mm, X41_train_rs, X41_test_rs,X41_test1_rs,X41_test2_rs,Y41_train, Y41_test, y41_train, y41_test,41,user_n,ru,X_train_ru, Y41_test1, Y41_test2)
experiment42 = Experiment3(X42_train_ori, X42_test_ori, X42_test1_ori, X42_test2_ori,X42_train_ss, X42_test_ss,X42_test1_ss, X42_test2_ss,X42_train_mm, X42_test_mm,X42_test1_mm, X42_test2_mm, X42_train_rs, X42_test_rs,X42_test1_rs,X42_test2_rs,Y42_train, Y42_test, y42_train, y42_test,42,user_n,rd,X_train_rd, Y42_test1, Y42_test2)
experiment43 = Experiment3(X43_train_ori, X43_test_ori, X43_test1_ori, X43_test2_ori,X43_train_ss, X43_test_ss,X43_test1_ss, X43_test2_ss,X43_train_mm, X43_test_mm,X43_test1_mm, X43_test2_mm, X43_train_rs, X43_test_rs,X43_test1_rs,X43_test2_rs,Y43_train, Y43_test, y43_train, y43_test,43,user_n,rl,X_train_rl, Y43_test1, Y43_test2)
experiment44 = Experiment3(X44_train_ori, X44_test_ori, X44_test1_ori, X44_test2_ori,X44_train_ss, X44_test_ss,X44_test1_ss, X44_test2_ss,X44_train_mm, X44_test_mm,X44_test1_mm, X44_test2_mm, X44_train_rs, X44_test_rs,X44_test1_rs,X44_test2_rs,Y44_train, Y44_test, y44_train, y44_test,44,user_n,rr,X_train_rr, Y44_test1, Y44_test2)

experiment11.closs()
experiment12.closs()
experiment13.closs()
experiment14.closs()

experiment21.closs()
experiment22.closs()
experiment23.closs()
experiment24.closs()

experiment31.closs()
experiment32.closs()
experiment33.closs()
experiment34.closs()

experiment41.closs()
experiment42.closs()
experiment43.closs()
experiment44.closs()

# experiment33.pca()

# experiment11.pca()
# experiment12.pca()
# experiment13.pca()
# experiment14.pca()
#
# experiment21.pca()
# experiment22.pca()
# experiment23.pca()
# experiment24.pca()
#
# experiment31.pca()
# experiment32.pca()
# experiment33.pca()
# experiment34.pca()
#
# experiment41.pca()
# experiment42.pca()
# experiment43.pca()
# experiment44.pca()

# experiment33.plot_IF()

# experiment2_0.clf_models()

# experiment2_0.IF()

# experiment2_0.LOF()

# experiment2_0.OCSVM()

# experiment2_0.EE()
