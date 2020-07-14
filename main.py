'''
メインの実験プログラムのつもり
条件：2ストロークの組み合わせ，分類器は1クラス分類器使用.
'''

# import os
# import copy
# import pickle
import numpy as np
import pandas as pd
# from pandas import DataFrame
import matplotlib.pyplot as plt
# from IPython.display import display

# モデル
# import sklearn
# from sklearn import svm
from sklearn.svm import OneClassSVM
# from sklearn.mixture import GaussianMixture
# from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

#スケーリング
from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import RobustScaler

# その他
# import time
# from tqdm import tqdm
# from multiprocessing import cpu_count
# from sklearn.externals import joblib

# データ確認用
from exp_module import conform

# warning inogre code
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
def load_frank_data():
    '''
    expdata.csvの読み込み
    'flag','user2','flag2','user_ave','flag_ave'の削除:107>>102
    '''
    # mainから呼び出すとき(basic)
    # パスの指定は実行するプログラムの相対パスっぽい
    df_ori = pd.read_csv("10_feature_selection/expdata.csv", sep = ",")
    # 同モジュール内から呼び出すとき
    # df_ori = pd.read_csv("../../10_feature_selection/expdata.csv", sep = ",")

    # 不要なものを列で削除
    df_drop = df_ori.drop({'Unnamed: 0', 'flag', 'user2','flag2', 'user_ave', 'flag_ave'}, axis = 1)

    return df_drop

frank_df = load_frank_data()

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


from sklearn.model_selection import KFold
# from sklearn.model_selection import GridSearchCV


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

# データ分割用
from exp_module import data_split_exp as dse

# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import matthews_corrcoef

# from sklearn.feature_selection import RFE
# from sklearn.decomposition import PCA
# import mglearn
# import japanize_matplotlib
# from mpl_toolkits.mplot3d import Axes3D

def far_frr(normal_result, anomaly_result):
    TP = np.count_nonzero(normal_result == 1)
    FN = np.count_nonzero(normal_result == -1)
    FP = np.count_nonzero(anomaly_result == 1)
    TN = np.count_nonzero(anomaly_result == -1)
    FRR = FN / (FN + TP)
    FAR = FP / (TN + FP)
    BER = 0.5 * (FP / (TN + FP) + FN / (FN + TP))
    return FRR, FAR, BER

def hazu(st, X_val_no, Y_val_no, user_n, X_train, columns):
    '''

    :param st: multi_flagに基づいた各データの集まり
    :param X_val_no: X_train_ssの内の検証用データ
    :param Y_val_no: Y_train_ssの内の検証用データ
    :param user_n: ユーザの番号
    :param X_train: 各フラグのうちuser_nで抽出されたX_trainのスケーリング前のもの
    スケーリングの範囲を合わせるために利用
    :param columns: Column
    :return: 外れ値を付与した検証用データ
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

    X_val_ano, Y_val_ano = dse.X_Y(user_0)

    # スケーリング
    ss = preprocessing.StandardScaler().fit(X_train)
    result1 = ss.transform(X_val_ano)
    result = pd.DataFrame(result1)
    result.columns = columns

    # testデータの結合
    X_val = pd.concat([X_val_no, result]).reset_index(drop=True)
    Y_val1 = pd.concat([Y_val_no, Y_val_ano]).reset_index(drop=True)

    Y_val2 = Y_val1.copy()
    Y_val = Y_val2.replace({user_n: 1, 0: -1})

    return X_val, Y_val, X_val_no, result

class Experiment():

    def __init__(self, st, user_select, user_n, flag_n):
        self.st = st
        self.user_select = user_select
        self.user_n = user_n
        self.flag_n = flag_n
        # 実験用にデータを訓練データ，検証データ，テストデータにわける
        self.X_train_ss, self.X_test_ss, self.X_test_t_ss, self.X_test_f_ss, self.Y_train, self.Y_test, self.train_target, \
        self.test_target, self.X_train, self.Y_test_t, self.Y_test_f = dse.data_split(self.st, self.user_select, self.user_n)

        self.columns = self.X_train.columns.values

    def closs_val(self):

        try:

            # モデル
            models = [LocalOutlierFactor(n_neighbors=1, novelty=True, contamination=0.1),
                      IsolationForest(n_estimators=1, contamination='auto', behaviour='new', random_state=0),
                      OneClassSVM(nu=0.1, kernel="rbf"),
                      EllipticEnvelope(contamination=0.1, random_state=0)]
            scores = {'LocalOutlierFactor': {}, 'IsolationForest': {}, 'OneClassSVM': {}, 'EllipticEnvelope': {}}
            scores_test = {}

            # 絶対要らない気がするので再確認
            Y_true1 = self.Y_test.copy()
            Y_true = Y_true1.replace({self.user_n: 1, 0: -1})

            # k分割交差検証 k=10
            k = 10
            kf = KFold(n_splits=k, shuffle=True, random_state=0)
            for model in models:
                X_train_ss2 = pd.DataFrame(self.X_train_ss, columns = self.columns)
                count = 0
                for train_index, val_index in kf.split(X_train_ss2, self.Y_train):
                    model.fit(X_train_ss2.iloc[train_index])
                    # 検証用データに偽物のデータを付与
                    X_val, Y_val, X_val_t, X_val_f = hazu(self.st, X_train_ss2.iloc[val_index],
                                                             self.Y_train.iloc[val_index], self.user_n,
                                                             self.X_train, self.columns)
                    # 予測
                    val_pred = model.predict(X_val)
                    normal_result = model.predict(X_val_t)
                    anomaly_result = model.predict(X_val_f)

                    # 評価
                    FAR, FRR, BER = far_frr(normal_result, anomaly_result)

                    scores[str(model).split('(')[0]][count] = {'Accuracy': accuracy_score(y_true=Y_val, y_pred=val_pred),
                                                               'Precision': precision_score(Y_val, val_pred),
                                                               'Recall': recall_score(Y_val, val_pred),
                                                               'F1': f1_score(Y_val, val_pred),
                                                               'AUC': roc_auc_score(Y_val, model.decision_function(X_val)),
                                                               'FAR': FAR, 'FRR': FRR, 'BER': BER}
                    count += 1
            df_index = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'FAR', 'FRR', 'BER']
            # df = scores.gropby(level=0).apply(lambda scores: scores.xs(scores.name).clump_thickness.to_dict()).to_dict()
            df = pd.DataFrame(scores)
            print(df.T)
            # print(scores['LocalOutlierFactor'][0]['Accuracy'])
            # ゴリ押しプログラム書くしかない(泣)
            model_index = ['LocalOutlierFactor', 'IsolationForest', 'OneClassSVM', 'EllipticEnvelope']
            a = np.zeros((4, 8))
            for i, m_i in enumerate(model_index):
                for j, df_i in enumerate(df_index):
                    b = 0
                    for m in range(k):
                        b += scores[m_i][m][df_i]
                    a[i][j] = b/k
            print(a)
            # index = [['LocalOutlierFactor', 'IsolationForest', 'OneClassSVM', 'EllipticEnvelope'],[]]
        except AttributeError:
            print('None')



print('\n-----------------------------------------------------------------\na + a\n-----------------------------------------------------------------')
experiment_aa = Experiment(aa, selu_aa, user_n, 11)
# print('\n-----------------------------------------------------------------\na + b\n-----------------------------------------------------------------')
# experiment_ab = Experiment(ab, selu_ab, user_n, 12)
# print('\n-----------------------------------------------------------------\na + c\n-----------------------------------------------------------------')
# experiment_ac = Experiment(ac, selu_ac, user_n, 13)
# print('\n-----------------------------------------------------------------\na + d\n-----------------------------------------------------------------')
# experiment_ad = Experiment(ad, selu_ad, user_n, 14)
# print('\n-----------------------------------------------------------------\nb + a\n-----------------------------------------------------------------')
# experiment_ba = Experiment(ba, selu_ba, user_n, 21)
# print('\n-----------------------------------------------------------------\nb + b\n-----------------------------------------------------------------')
# experiment_bb = Experiment(bb, selu_bb, user_n, 22)
# print('\n-----------------------------------------------------------------\nb + c\n-----------------------------------------------------------------')
# experiment_bc = Experiment(bc, selu_bc, user_n, 23)
# print('\n-----------------------------------------------------------------\nb + d\n-----------------------------------------------------------------')
# experiment_bd = Experiment(bd, selu_bd, user_n, 24)
# print('\n-----------------------------------------------------------------\nc + a\n-----------------------------------------------------------------')
# experiment_ca = Experiment(ca, selu_ca, user_n, 31)
# print('\n-----------------------------------------------------------------\nc + b\n-----------------------------------------------------------------')
# experiment_cb = Experiment(cb, selu_cb, user_n, 32)
# print('\n-----------------------------------------------------------------\nc + c\n-----------------------------------------------------------------')
# experiment_cc = Experiment(cc, selu_cc, user_n, 33)
# print('\n-----------------------------------------------------------------\nc + d\n-----------------------------------------------------------------')
# experiment_cd = Experiment(cd, selu_cd, user_n, 34)
# print('\n-----------------------------------------------------------------\nd + a\n-----------------------------------------------------------------')
# experiment_da = Experiment(da, selu_da, user_n, 41)
# print('\n-----------------------------------------------------------------\nd + b\n-----------------------------------------------------------------')
# experiment_db = Experiment(db, selu_db, user_n, 42)
# print('\n-----------------------------------------------------------------\nd + c\n-----------------------------------------------------------------')
# experiment_dc = Experiment(dc, selu_dc, user_n, 43)
# print('\n-----------------------------------------------------------------\nd + d\n-----------------------------------------------------------------')
# experiment_dd = Experiment(dd, selu_dd, user_n, 44)


experiment_aa.closs_val()
# experiment_cc.closs_val()
# experiment_bb.closs_val()