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

from sklearn.model_selection import train_test_split

# その他
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

# データのColumn取得
df_column = frank_df.columns.values

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

from exp_module import conform

def data_split(st, user_select, user_n):

    '''
    :param st: split_dataのことであり，multi_flagに基づいたデータの集合
    :param user_select: stの中でも選択されたユーザのみのデータ
    関係としては，st ∋　user_select
    :param user_n: 選択されたユーザ
    :return:
    '''

    # データ数が30以上あるか
    if user_select['user'].count() >= 30:

        # 説明変数Xと目的変数Yに分割
        def X_Y(user_select):
            X = user_select.drop("user", 1)
            Y = user_select.user
            return X, Y
        X, Y = X_Y(user_select)

        # 訓練データとテストデータに分割
        X_train, X_test_t, Y_train, Y_test_t = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

        # テストデータの個数をカウント➝外れ値はテストデータ数と同じ数だけ用意する
        Y_test_t_size = Y_test_t.count()

        # 偽物(外れ値)のデータの選択
        def outlier(st, user_n, Y_test_t_size):
            '''
            :param st:
            :param user_n:
            :param Y_test_t_size: テストデータの個数
            :return: st_f:user_n以外のデータ，st_f_us:st_fの中からtestデータと同じ数だけ選択した偽物のデータ
            '''
            #user_n以外のuserを抽出
            st_f = st[st['user'] != user_n]
            #sstをシャッフル
            st_f_shuffle = st_f.sample(frac=1, random_state=0).reset_index(drop=True)
            #testデータと同じ数の外れ値を選択
            st_f_select_outlier = st_f_shuffle.sample(n = Y_test_t_size, random_state=0)
            return st_f, st_f_select_outlier
        st_f, st_f_us = outlier(st, user_n, Y_test_t_size)

        # 外れ値の個数などの確認をしたい場合
        # conform.conf_outlier(st, st_f, st_f_us)

        # testデータと外れ値データの結合
        st_f_us_number = list(st_f_us['user'])
        print('\n外れ値として扱うuserのnumber\n', st_f_us_number)

        # userをすべて０に変更
        user_n_change = st_f_us.copy()
        user_0 = user_n_change.replace({'user': st_f_us_number},0)

        # 偽物のデータを説明変数と目的変数に分割
        X_test_f, Y_test_f = X_Y(user_0)

        # testデータの結合
        X_test = pd.concat([X_test_t, X_test_f]).reset_index(drop=True)
        Y_test = pd.concat([Y_test_t, Y_test_f]).reset_index(drop=True)

        # スケーリング
        ss = preprocessing.StandardScaler().fit(X_train)
        X_train_ss = ss.transform(X_train)
        X_test_ss = ss.transform(X_test)
        X_test_t_ss = ss.transform(X_test_t)
        X_test_f_ss = ss.transform(X_test_f)

        # Yに含まれるuserのindexを作成➝用途はおそらくまあ，可視化の際のメモリ用だと思われる
        def Y_target(Y):
            y = pd.DataFrame(Y)
            g = y.groupby("user")
            target = pd.DataFrame(g.size().sort_values(ascending=False))
            target_index = target.index.values
            return target_index
        train_target = Y_target(Y_test)
        test_target = Y_target(Y_test)

        # matome
        # conform.conf_matome(X, Y, X_train, Y_train, X_test, Y_test, X_test_t, X_test_f, Y_test_t, Y_test_f, train_target, test_target)

        return X_train_ss, X_test_ss, X_test_t_ss, X_test_f_ss, Y_train, Y_test, train_target, test_target, X_train, Y_test_t, Y_test_f

    else:
        print('None')
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        pass

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


def far_frr(normal_result, anomaly_result):
    TP = np.count_nonzero(normal_result == 1)
    FN = np.count_nonzero(normal_result == -1)
    FP = np.count_nonzero(anomaly_result == 1)
    TN = np.count_nonzero(anomaly_result == -1)
    FRR = FN / (FN + TP)
    FAR = FP / (TN + FP)
    BER = 0.5 * (FP / (TN + FP) + FN / (FN + TP))
    return FRR, FAR, BER


def hazu(st, X_val_no, Y_val_no, user_n, scale_n, X_train, columns):
    '''
    valに外れ値データをつけたい
    '''

    def hazure2(st, X_val_no, Y_val_no, user_n):
        sst = st[st['user'] != user_n]
        val_no_size = Y_val_no.count()
        sst_shuffle = sst.sample(frac=1, random_state=0).reset_index(drop=True)
        sst_select_outlier = sst_shuffle.sample(n=val_no_size, random_state=0)
        return sst_select_outlier

    sst_select_outlier = hazure2(st, X_val_no, Y_val_no, user_n)
    list2 = list(sst_select_outlier['user'])
    user_n_change = sst_select_outlier.copy()
    user_0 = user_n_change.replace({'user': list2}, 0)

    def X_Y(user_select):
        X = user_select.drop("user", 1)
        Y = user_select.user
        return X, Y

    X_val_ano, Y_val_ano = X_Y(user_0)

    # スケーリング
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
    Y_val = Y_val2.replace({self.user_n: 1, 0: -1})

    return X_val, Y_val, X_val_no, result

class Experiment():

    def __init__(self, st, user_select, user_n, flag_n):
        self.st = st
        self.user_select = user_select
        self.user_n = user_n
        self.flag_n = flag_n
        self.column = st.columns.values

    def data(self):
        self.X_train_ss, self.X_test_ss, self.X_test_t_ss, self.X_test_f_ss, self.Y_train, self.Y_test, self.train_target, \
        self.test_target, self.X_train, self.Y_test_t, self.Y_test_f = data_split(self.st, self.user_select, self.user_n)

    def closs_val(self):


print('\n-----------------------------------------------------------------\na + a\n-----------------------------------------------------------------')
experiment_aa = Experiment(aa, selu_aa, user_n, 11)
print('\n-----------------------------------------------------------------\na + b\n-----------------------------------------------------------------')
experiment_ab = Experiment(ab, selu_ab, user_n, 12)
print('\n-----------------------------------------------------------------\na + c\n-----------------------------------------------------------------')
experiment_ac = Experiment(ac, selu_ac, user_n, 13)
print('\n-----------------------------------------------------------------\na + d\n-----------------------------------------------------------------')
experiment_ad = Experiment(ad, selu_ad, user_n, 14)
print('\n-----------------------------------------------------------------\nb + a\n-----------------------------------------------------------------')
experiment_ba = Experiment(ba, selu_ba, user_n, 21)
print('\n-----------------------------------------------------------------\nb + b\n-----------------------------------------------------------------')
experiment_bb = Experiment(bb, selu_bb, user_n, 22)
print('\n-----------------------------------------------------------------\nb + c\n-----------------------------------------------------------------')
experiment_bc = Experiment(bc, selu_bc, user_n, 23)
print('\n-----------------------------------------------------------------\nb + d\n-----------------------------------------------------------------')
experiment_bd = Experiment(bd, selu_bd, user_n, 24)
print('\n-----------------------------------------------------------------\nc + a\n-----------------------------------------------------------------')
experiment_ca = Experiment(ca, selu_ca, user_n, 31)
print('\n-----------------------------------------------------------------\nc + b\n-----------------------------------------------------------------')
experiment_cb = Experiment(cb, selu_cb, user_n, 32)
print('\n-----------------------------------------------------------------\nc + c\n-----------------------------------------------------------------')
experiment_cc = Experiment(cc, selu_cc, user_n, 33)
print('\n-----------------------------------------------------------------\nc + d\n-----------------------------------------------------------------')
experiment_cd = Experiment(cd, selu_cd, user_n, 34)
print('\n-----------------------------------------------------------------\nd + a\n-----------------------------------------------------------------')
experiment_da = Experiment(da, selu_da, user_n, 41)
print('\n-----------------------------------------------------------------\nd + b\n-----------------------------------------------------------------')
experiment_db = Experiment(db, selu_db, user_n, 42)
print('\n-----------------------------------------------------------------\nd + c\n-----------------------------------------------------------------')
experiment_dc = Experiment(dc, selu_dc, user_n, 43)
print('\n-----------------------------------------------------------------\nd + d\n-----------------------------------------------------------------')
experiment_dd = Experiment(dd, selu_dd, user_n, 44)