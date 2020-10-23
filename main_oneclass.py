"""
実験用プログラム 1ストロークの1クラス分類
まずこれを完成させて結果を出す
"""
import os
import numpy as np
import pandas as pd


# スケーリング
from sklearn import preprocessing

# モデル
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

# 内部ライブラリ
from expmodule.dataset import load_frank
from expmodule.flag_split import flag4
from expmodule.datasplit_train_test_val import datasplit_session

# Comment: 一度警告を確認してからもとに戻す
# warning inogre code
# import warnings
# warnings.filterwarnings('ignore')

def main(df, user_n, session):

    # 第1段階

    # データのColumn取得
    df_column = df.columns.values

    # 上下左右のflagをもとにデータを分割
    a, b, c, d = flag4(df, 'flag')

    # flagごとに選択したユーザを抽出する
    def select_user_flag(df_flag, u_n, text):
        df_flag_user_extract = df_flag[df_flag['user'] == u_n]
        # Comment: 1-41人全員分行ったらコメントアウト
        data_item = pd.DataFrame([u_n, text, len(df_flag_user_extract)]).T
        data_item.to_csv(f'result2020_10/info/main_oneclass_df_flag_user_extract_item.csv', mode='a', header=None, index=None)
        return df_flag_user_extract

    # select_user_flag()のインスタンス作成
    a_user_extract = select_user_flag(a, user_n, 'a')
    b_user_extract = select_user_flag(b, user_n, 'b')
    c_user_extract = select_user_flag(c, user_n, 'c')
    d_user_extract = select_user_flag(d, user_n, 'd')

    # 第2段階

    def far_frr_ber(normal_result, anomaly_result):
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

        def x_y_split(df_flag_user_extract):
            x_split = df_flag_user_extract.drop("user", 1)
            y_split = df_flag_user_extract.user
            return x_split, y_split

        x_val_ano, y_val_ano = x_y_split(user_0)

        # スケーリング
        ss = preprocessing.StandardScaler().fit(x_train)
        x_val_ano1 = ss.transform(x_val_ano)
        re_x_val_ano = pd.DataFrame(x_val_ano1)
        re_x_val_ano.columns = columns

        # testデータの結合
        re_x_val = pd.concat([re_x_val_no, re_x_val_ano]).reset_index(drop=True)
        y_val1 = pd.concat([y_val_no, y_val_ano]).reset_index(drop=True)

        y_val2 = y_val1.copy()
        # 結果が本人であれば１，偽物であれば-1で返されるため，予測された結果があっているかを確認するための教師データを作成する
        re_y_val = y_val2.replace({user_n: 1, 0: -1})

        return re_x_val, re_y_val, re_x_val_no, re_x_val_ano

    class OneClassOne(object):

        def __init__(self, df_flag, df_flag_user_extract, u_n, flag_n, session_select):
            memori = ['0', 'a', 'b', 'c', 'd']
            print(
                f'\n-----------------------------------------------------------------\n{user_n} : {memori[flag_n]} + '
                f'\n-----------------------------------------------------------------')
            try:
                self.df_flag = df_flag
                self.df_flag_user_extract = df_flag_user_extract
                self.u_n = u_n
                self.flag_n = flag_n
                self.session_select = session_select

                self.x_train, self.y_train, self.x_test, self.y_test, \
                    self.x_test_t, self.y_test_t, self.x_test_f, self.y_test_f, self.test_f \
                    = datasplit_session(self.df_flag, self.df_flag_user_extract, self.u_n, self.session_select)

                # 標準化
                ss = preprocessing.StandardScaler()
                ss.fit(self.x_train)
                self.x_train_ss = ss.transform(self.x_train)
                self.x_test_ss = ss.transform(self.x_test)
                self.x_test_t_ss = ss.transform(self.x_test_t)
                self.x_test_f_ss = ss.transform(self.x_test_f)

            except AttributeError as ex:
                print(f"No data:{ex}")
                pass

        def registration_phase(self):
            if list(self.test_f['user']) != 0:
                print('\n外れ値として扱うuserのnumber\n', self.test_f, '\n')

            # モデル
            models = [LocalOutlierFactor(n_neighbors=1, novelty=True, contamination=0.1),
                      IsolationForest(n_estimators=1, contamination='auto', behaviour='new', random_state=0),
                      OneClassSVM(nu=0.1, kernel="rbf"),
                      EllipticEnvelope(contamination=0.1, random_state=0)]
            scores = {'LocalOutlierFactor': {}, 'IsolationForest': {}, 'OneClassSVM': {}, 'EllipticEnvelope': {}}
            scores_test = {}

            # 目的関数を出力結果にあわせて本人：1，他人:-1に変更する．
            y_test_true = self.y_test.copy()
            y_true = y_test_true.replace({self.u_n: 1, 0: -1})

        def authentication_phase(self):
            abc = 0

    OneClassOne_a = OneClassOne(a, a_user_extract, user_n, 1, session)
    OneClassOne_b = OneClassOne(b, b_user_extract, user_n, 2, session)
    OneClassOne_c = OneClassOne(c, c_user_extract, user_n, 3, session)
    OneClassOne_d = OneClassOne(d, d_user_extract, user_n, 4, session)


if __name__ == '__main__':
    print("実験用プログラム 1ストロークの1クラス分類")

    frank_df = load_frank(False)
    session_list = ['first', 'latter', 'all']
    # 41人いるよ
    for user in range(1, 2):
        main(frank_df, user, session='all')
