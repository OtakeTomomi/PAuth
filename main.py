"""
メインの実験プログラムのつもり
条件：2ストロークの組み合わせ，分類器は1クラス分類器使用.
"""

import os
import sys
# import copy
# import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from IPython.display import display

# モデル
from sklearn.svm import OneClassSVM
# from sklearn.mixture import GaussianMixture
# from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

# スケーリング
from sklearn import preprocessing

# データ確認用
# from exp_module import conform

# PCAするときに必要かも
# from sklearn.feature_selection import RFE
# from sklearn.decomposition import PCA
# import mglearn
# import japanize_matplotlib
# from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import KFold

# 評価
# from sklearn import metrics
from sklearn.metrics import f1_score
# from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# データ分割用
from exp_module import data_split_exp as dse
from exp_module import flag_split

# warning inogre code
import warnings
warnings.filterwarnings('ignore')

# コマンドライン変数から値を受け取る→変更
args = sys.argv
user_n = int(args[1])


# データの読み込み
def load_frank_data():
    """
    expdata.csvの読み込み
    'flag','user2','flag2','user_ave','flag_ave'の削除:107>>102
    """
    # mainから呼び出すとき(basic)
    # パスの指定は実行するプログラムの相対パスっぽい
    df_ori = pd.read_csv("dataset_create/expdata.csv", sep=",")
    # 不要なものを列で削除
    df_drop = df_ori.drop({'Unnamed: 0', 'flag', 'user2', 'flag2', 'user_ave', 'flag_ave'}, axis=1)
    return df_drop


frank_df = load_frank_data()

# データのColumn取得
df_column = frank_df.columns.values

# データをmulti_flagを基準に分割
# a,b,c,dのストローク方向はflag_splitに記載
aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd = flag_split.frank_fs(frank_df)
# 各multi_flagに含まれる各ユーザのデータ数の多い順について確認したい場合にはlist_index = conform.conf_sel_flag_qty()で確認可能ではある

# ここまでは41人分


# 選択されたユーザのデータを各aa~ddから抽出
# select_user_from_frank_fs
def sel_user_ffs(sdf2, user_n2, text):
    sdf_sel_u = sdf2[sdf2['user'] == user_n2]
    # ff = sdf_sel_u.groupby("user")
    # print(f'データ数_{text}：{ff.size()}')
    data_item = pd.DataFrame([user_n2, text, len(sdf_sel_u)]).T
    # print(data_item)
    data_item.to_csv(f'info/data_item.csv', mode='a', header=None, index=None)

    return sdf_sel_u


# 各multi_flagごとに選択したユーザを抽出する
selu_aa = sel_user_ffs(aa, user_n, 'aa')
selu_ab = sel_user_ffs(ab, user_n, 'ab')
selu_ac = sel_user_ffs(ac, user_n, 'ac')
selu_ad = sel_user_ffs(ad, user_n, 'ad')

selu_ba = sel_user_ffs(ba, user_n, 'ba')
selu_bb = sel_user_ffs(bb, user_n, 'bb')
selu_bc = sel_user_ffs(bc, user_n, 'bc')
selu_bd = sel_user_ffs(bd, user_n, 'bd')

selu_ca = sel_user_ffs(ca, user_n, 'ca')
selu_cb = sel_user_ffs(cb, user_n, 'cb')
selu_cc = sel_user_ffs(cc, user_n, 'cc')
selu_cd = sel_user_ffs(cd, user_n, 'cd')

selu_da = sel_user_ffs(da, user_n, 'da')
selu_db = sel_user_ffs(db, user_n, 'db')
selu_dc = sel_user_ffs(dc, user_n, 'dc')
selu_dd = sel_user_ffs(dd, user_n, 'dd')


# これは見直して必要なものだけ取り出す
"""
def result_0722(normal_result, anomaly_result, Y_true, prediction, y_score):
    print("\n正常データのスコア\n", normal_result)
    print("\n異常データのスコア\n", anomaly_result)
    TP = np.count_nonzero(normal_result == 1)
    FN = np.count_nonzero(normal_result == -1)
    FP = np.count_nonzero(anomaly_result == 1)
    TN = np.count_nonzero(anomaly_result == -1)
    print('\nTP：', TP, '　FN:', FN, '　FP:', FP, '　TN:', TN)

    cm = confusion_matrix(Y_true, prediction, labels=[1, -1])
    print(cm)

    print('classification_report\n', classification_report(Y_true, prediction))
    print('\nAccuracy:', accuracy_score(Y_true, prediction))
    print('Precision:', precision_score(Y_true, prediction))
    print('Recall:', recall_score(Y_true, prediction))
    print('F1:', f1_score(Y_true, prediction))
    FRR = FN / (FN + TP)
    print("FRR:{}".format(FRR))
    FAR = FP / (TN + FP)
    print("FAR:{}".format(FAR))
    BER = 0.5 * (FP / (TN + FP) + FN / (FN + TP))
    print("BER:{}".format(BER))
    print('AUC', roc_auc_score(Y_true, y_score))

    fpr, tpr, thresholds = roc_curve(Y_true, y_score, drop_intermediate=False)
    auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.title(f'ROC curve (AUC = {auc:.3f})')
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.3f})', marker='o')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.legend()
    plt.show()
"""


def far_frr(normal_result, anomaly_result):
    tp = np.count_nonzero(normal_result == 1)
    fn = np.count_nonzero(normal_result == -1)
    fp = np.count_nonzero(anomaly_result == 1)
    tn = np.count_nonzero(anomaly_result == -1)
    re_far = fp / (tn + fp)
    re_frr = fn / (fn + tp)
    re_ber = 0.5 * ((fp / (tn + fp)) + (fn / (fn + tp)))

    # accuracy = ((TP+TN)/(TP+FN+FP+TN))
    # print(accuracy)
    return re_far, re_frr, re_ber


def hazu(st, re_x_val_no, y_val_no, user_n2, x_train, columns):
    """
    :param st: multi_flagに基づいた各データの集まり
    :param re_x_val_no: X_train_ssの内の検証用データ
    :param y_val_no: Y_train_ssの内の検証用データ
    :param user_n2: ユーザの番号
    :param x_train: 各フラグのうちuser_nで抽出されたX_trainのスケーリング前のもの
    スケーリングの範囲を合わせるために利用
    :param columns: Column
    :return: 外れ値を付与した検証用データ
    """

    def hazure2(st2, y_val_no2, user_n3):
        sst = st2[st2['user'] != user_n3]
        val_no_size = y_val_no2.count()
        sst_shuffle = sst.sample(frac=1, random_state=0).reset_index(drop=True)
        sst_select_outlier2 = sst_shuffle.sample(n=val_no_size, random_state=0)
        return sst_select_outlier2

    sst_select_outlier = hazure2(st, y_val_no, user_n2)
    list2 = list(sst_select_outlier['user'])
    user_n_change = sst_select_outlier.copy()
    user_0 = user_n_change.replace({'user': list2}, 0)

    x_val_ano, y_val_ano = dse.X_Y(user_0)

    # スケーリング
    ss = preprocessing.StandardScaler().fit(x_train)
    x_val_ano1 = ss.transform(x_val_ano)
    re_x_val_ano = pd.DataFrame(x_val_ano1)
    re_x_val_ano.columns = columns

    # testデータの結合
    re_x_val = pd.concat([re_x_val_no, re_x_val_ano]).reset_index(drop=True)
    y_val1 = pd.concat([y_val_no, y_val_ano]).reset_index(drop=True)

    y_val2 = y_val1.copy()
    re_y_val = y_val2.replace({user_n: 1, 0: -1})

    return re_x_val, re_y_val, re_x_val_no, re_x_val_ano


class Experiment(object):

    def __init__(self, st, user_select, user_n2, flag_n):
        memori = ['0', 'a', 'b', 'c', 'd']
        print(
            f'\n-----------------------------------------------------------------\n{user_n} : {memori[flag_n // 10]} + '
            f'{memori[flag_n % 10]}\n-----------------------------------------------------------------')
        try:
            self.st = st
            self.user_select = user_select
            self.user_n = user_n2
            self.flag_n = flag_n
            # 実験用にデータを訓練データ，検証データ，テストデータにわける
            self.X_train_ss, self.X_test_ss, self.X_test_t_ss, self.X_test_f_ss, self.Y_train, self.Y_test, \
                self.train_target, self.test_target, self.X_train, self.Y_test_t, self.Y_test_f, self.st_f_us_number = \
                dse.data_split(self.st, self.user_select, self.user_n)

            self.columns = self.X_train.columns.values
        except AttributeError:
            pass

    def closs_val(self):
        try:
            if self.st_f_us_number != 0:
                print('\n外れ値として扱うuserのnumber\n', self.st_f_us_number, '\n')
            # モデル
            models = [LocalOutlierFactor(n_neighbors=1, novelty=True, contamination=0.1),
                      IsolationForest(n_estimators=1, contamination='auto', behaviour='new', random_state=0),
                      OneClassSVM(nu=0.1, kernel="rbf"),
                      EllipticEnvelope(contamination=0.1, random_state=0)]
            scores = {'LocalOutlierFactor': {}, 'IsolationForest': {}, 'OneClassSVM': {}, 'EllipticEnvelope': {}}
            scores_test = {}

            y_true1 = self.Y_test.copy()
            y_true = y_true1.replace({self.user_n: 1, 0: -1})

            # k分割交差検証 k=10
            k = 10
            kf = KFold(n_splits=k, shuffle=True, random_state=0)
            for model in models:
                x_train_ss2 = pd.DataFrame(self.X_train_ss, columns=self.columns)
                count = 0
                for train_index, val_index in kf.split(x_train_ss2, self.Y_train):
                    model.fit(x_train_ss2.iloc[train_index])
                    # 検証用データに偽物のデータを付与
                    x_val, y_val, x_val_t, x_val_f = hazu(self.st, x_train_ss2.iloc[val_index],
                                                          self.Y_train.iloc[val_index], self.user_n,
                                                          self.X_train, self.columns)
                    # 予測
                    val_pred = model.predict(x_val)
                    normal_result = model.predict(x_val_t)
                    anomaly_result = model.predict(x_val_f)

                    # 評価
                    far, frr, ber = far_frr(normal_result, anomaly_result)

                    scores[str(model).split('(')[0]][count] = {'Accuracy': accuracy_score(y_true=y_val,
                                                                                          y_pred=val_pred),
                                                               'Precision': precision_score(y_val, val_pred),
                                                               'Recall': recall_score(y_val, val_pred),
                                                               'F1': f1_score(y_val, val_pred),
                                                               'AUC': roc_auc_score(y_val,
                                                                                    model.decision_function(x_val)),
                                                               'FAR': far, 'FRR': frr, 'BER': ber}
                    count += 1
                # testデータにて汎化性能評価
                model.fit(x_train_ss2)
                test_pred = model.predict(self.X_test_ss)
                test_normal_result = model.predict(self.X_test_t_ss)
                test_anomaly_result = model.predict(self.X_test_f_ss)

                t_far, t_frr, t_ber = far_frr(test_normal_result, test_anomaly_result)

                scores_test[str(model).split('(')[0]] = {'Accuracy': accuracy_score(y_true=y_true, y_pred=test_pred),
                                                         'Precision': precision_score(y_true, test_pred),
                                                         'Recall': recall_score(y_true, test_pred),
                                                         'F1': f1_score(y_true, test_pred),
                                                         'AUC': roc_auc_score(y_true,
                                                                              model.decision_function(self.X_test_ss)),
                                                         'FAR': t_far, 'FRR': t_frr, 'BER': t_ber}
            # 結果のまとめ
            # Panelが廃止されたので，ゴリ押し感が否めない
            # いまさらだけどDecimal型に変換して計算したほうが良かった？
            model_index = ['LocalOutlierFactor', 'IsolationForest', 'OneClassSVM', 'EllipticEnvelope']
            result_index = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'FAR', 'FRR', 'BER']
            a = np.zeros((4, 8))
            for i, m_i in enumerate(model_index):
                for j, df_i in enumerate(result_index):
                    b = 0
                    for m in range(k):
                        b += scores[m_i][m][df_i]
                    a[i][j] = b/k
            a_df = pd.DataFrame(a, index=model_index, columns=result_index)
            print('交差検証k=10の結果')
            print(a_df)

            # result_old.csvへ書き出し
            def output_data(a2, model_index2, result_index2, text):
                # フォルダがなければ自動的に作成
                os.makedirs('result', exist_ok=True)
                # Columnの作成
                user = pd.Series([self.user_n]*4, name='user')
                flag = pd.Series([self.flag_n]*4, name='flag')
                performance = pd.Series([text]*4, name='performance')
                model2 = pd.Series(model_index2, name='model')
                # valとtestで条件指定してresult作成
                if text == 'val':
                    result = pd.DataFrame(a2, columns=result_index2)
                else:
                    print('\ntestデータでの結果')
                    result = pd.DataFrame(a2).T
                    print(result)
                    result = result.reset_index()
                    result = result.drop('index', 1)
                # 全て結合
                all_result = pd.concat([user, flag, performance, model2, result], axis=1)
                # 書き出し
                all_result.to_csv(f'result_2020_7/result_0722/result_{text}.csv', mode='a', header=False, index=False)

            # 交差検証の結果の書き出し
            output_data(a, model_index, result_index, 'val')
            # テストデータの結果の書き出し
            output_data(scores_test, model_index, result_index, 'test')
        except AttributeError as ex:
            print(f"No data:{ex}")
            pass


experiment_aa = Experiment(aa, selu_aa, user_n, 11)
experiment_aa.closs_val()

experiment_ab = Experiment(ab, selu_ab, user_n, 12)
experiment_ab.closs_val()

experiment_ac = Experiment(ac, selu_ac, user_n, 13)
experiment_ac.closs_val()

experiment_ad = Experiment(ad, selu_ad, user_n, 14)
experiment_ad.closs_val()

experiment_ba = Experiment(ba, selu_ba, user_n, 21)
experiment_ba.closs_val()

experiment_bb = Experiment(bb, selu_bb, user_n, 22)
experiment_bb.closs_val()

experiment_bc = Experiment(bc, selu_bc, user_n, 23)
experiment_bc.closs_val()

experiment_bd = Experiment(bd, selu_bd, user_n, 24)
experiment_bd.closs_val()

experiment_ca = Experiment(ca, selu_ca, user_n, 31)
experiment_ca.closs_val()

experiment_cb = Experiment(cb, selu_cb, user_n, 32)
experiment_cb.closs_val()

experiment_cc = Experiment(cc, selu_cc, user_n, 33)
experiment_cc.closs_val()

experiment_cd = Experiment(cd, selu_cd, user_n, 34)
experiment_cd.closs_val()

experiment_da = Experiment(da, selu_da, user_n, 41)
experiment_da.closs_val()

experiment_db = Experiment(db, selu_db, user_n, 42)
experiment_db.closs_val()

experiment_dc = Experiment(dc, selu_dc, user_n, 43)
experiment_dc.closs_val()

experiment_dd = Experiment(dd, selu_dd, user_n, 44)
experiment_dd.closs_val()
