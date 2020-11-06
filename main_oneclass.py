"""
実験用プログラム 1ストロークの1クラス分類
まずこれを完成させて結果を出す
"""
import os
import numpy as np
import pandas as pd
import datetime


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
from expmodule.datasplit_train_test_val import x_y_split

# 交差検証
from sklearn.model_selection import KFold

# 評価指標
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


# warning ignore code
import warnings
warnings.filterwarnings('ignore')


def main(df, user_n, session):

    # 第1段階: データをflagに従って上下左右の方向ごとに分割する

    # 上下左右のflagをもとにデータを分割
    a, b, c, d = flag4(df, 'flag')

    # flagごとに選択したユーザを抽出する
    def select_user_flag(df_flag, u_n, text):
        # フォルダがなければ自動的に作成
        os.makedirs('result/info', exist_ok=True)
        df_flag_user_extract = df_flag[df_flag['user'] == u_n]
        # 1-41人全員分行ったらコメントアウト→2020/11/05済
        data_item = pd.DataFrame([u_n, text, len(df_flag_user_extract)]).T
        # data_item.to_csv(f'result/info/main_oneclass_df_flag_user_extract_item.csv',
        #                  mode='a', header=None, index=None)
        return df_flag_user_extract

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

    def fake_val(df_flag, x_val_t, y_val_t, u_n, x_train, columns, fake_data_except_test_f):
        """
        :param df_flag: multi_flagに基づいた各データの集まり
        :param x_val_t: X_train_ssの内の検証用データ
        :param y_val_t: Y_train_ssの内の検証用データ
        :param u_n: ユーザの番号
        :param x_train: 各フラグのうちuser_nで抽出されたX_trainのスケーリング前のもの
        スケーリングの範囲を合わせるために利用
        :param columns: Column
        :param fake_data_except_test_f: fake_data_except_test_f
        :return: 外れ値を付与した検証用データ
        """

        # 検証用データの個数調査
        val_t_size = y_val_t.count()
        # テスト用に使用する他人のデータ以外から検証用データと同じ数選択する
        val_outlier = fake_data_except_test_f[:val_t_size]

        # print(val_outlier)

        val_outlier_user_n = list(val_outlier['user'])
        # print(val_outlier_user_n)
        val_outlier2 = val_outlier.copy()
        val_outlier_user_change0 = val_outlier2.replace({'user': val_outlier_user_n}, 0)
        # print(val_outlier_user_change0)

        x_val_f, y_val_f = x_y_split(val_outlier_user_change0)

        # スケーリング
        ss = preprocessing.StandardScaler()
        ss.fit(x_train)
        x_val_f_ss = ss.transform(x_val_f)
        x_val_f_ss_df = pd.DataFrame(x_val_f_ss)
        x_val_f_ss_df.columns = columns

        # testデータの結合
        x_val = pd.concat([x_val_t, x_val_f_ss_df]).reset_index(drop=True)
        y_val = pd.concat([y_val_t, y_val_f]).reset_index(drop=True)

        # print(y_val)

        y_val2 = y_val.copy()
        # 結果が本人であれば１，偽物であれば-1で返されるため，予測された結果があっているかを確認するための教師データを作成する
        y_val_true = y_val2.replace({u_n: 1, 0: -1})

        return x_val, y_val_true, x_val_t, x_val_f_ss_df

    class OneClassOne(object):

        def __init__(self, df_flag, df_flag_user_extract, u_n, flag_n, session_select):
            flag_memori = ['0', 'a', 'b', 'c', 'd']
            print(
                f'\n-----------------------------------------------------------------\n'
                f'{user_n} : {flag_memori[flag_n]} : {session_select}'
                f'\n-----------------------------------------------------------------')
            try:
                self.df_flag = df_flag
                self.df_flag_user_extract = df_flag_user_extract
                self.u_n = u_n
                self.flag_n = flag_n
                self.session_select = session_select

                self.x_train, self.y_train, self.x_test, self.y_test, self.x_test_t,\
                    self.y_test_t, self.x_test_f, self.y_test_f, self.test_f, self.fake_data_except_test_f \
                    = datasplit_session(self.df_flag, self.df_flag_user_extract, self.u_n, self.session_select, train_size=40)

                # 標準化
                # print(self.x_train.columns)
                ss = preprocessing.StandardScaler()
                ss.fit(self.x_train)
                self.x_train_ss = ss.transform(self.x_train)
                self.x_test_ss = ss.transform(self.x_test)
                self.x_test_t_ss = ss.transform(self.x_test_t)
                # print(self.x_test_t.columns)
                # print(self.x_test_t.shape)
                # print(self.x_test_f.columns)
                # print(self.x_test_f.shape)
                self.x_test_f_ss = ss.transform(self.x_test_f)

                self.columns = self.x_train.columns.values

            except AttributeError as ex:
                print(f"No data:{ex}")
                pass

            except ValueError as ex2:
                print(f'No data: {self.session_select} == {ex2}')
                pass

        def registration_phase(self):
            try:
                if list(self.test_f['user']) != 0:
                    print(f'\n外れ値として扱うuserのnumber')
                    print(list(self.test_f['user']), '\n')

                print(f'train_data: {self.x_train.shape}')
                print(f'test_data: {self.x_test.shape}\n')

                # print(self.x_train.head(5))

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

                # k分割交差検証 k=10
                k = 10
                kf = KFold(n_splits=k, shuffle=True, random_state=0)
                for model in models:
                    # columnsが消滅しているのでデータフレーム化してヘッダー追加
                    x_train_ss_df = pd.DataFrame(self.x_train_ss, columns=self.columns)
                    # print(len(self.y_train))
                    count = 0
                    for train_index, val_index in kf.split(x_train_ss_df, self.y_train):
                        model.fit(x_train_ss_df.iloc[train_index])
                        # 検証用データに偽物のデータを付与
                        # self.df_flagではなくdatasplit_train_test_valで作成したどこにも属していないデータ？
                        x_val, y_val_true, x_val_t, x_val_f = fake_val(self.df_flag, x_train_ss_df.iloc[val_index],
                                                                       self.y_train.iloc[val_index], self.u_n,
                                                                       self.x_train, self.columns,
                                                                       self.fake_data_except_test_f)
                        # 予測
                        val_pred = model.predict(x_val)
                        normal_result = model.predict(x_val_t)
                        anomaly_result = model.predict(x_val_f)

                        # print(y_val_true)
                        # 評価
                        far, frr, ber = far_frr_ber(normal_result, anomaly_result)

                        scores[str(model).split('(')[0]][count] = {'Accuracy': accuracy_score(y_true=y_val_true,
                                                                                              y_pred=val_pred),
                                                                   'Precision': precision_score(y_val_true, val_pred),
                                                                   'Recall': recall_score(y_val_true, val_pred),
                                                                   'F1': f1_score(y_val_true, val_pred),
                                                                   'AUC': roc_auc_score(y_val_true,
                                                                                        model.decision_function(x_val)),
                                                                   'FAR': far, 'FRR': frr, 'BER': ber}
                        count += 1

                    # testデータにて汎化性能評価
                    model.fit(x_train_ss_df)
                    test_pred = model.predict(self.x_test_ss)
                    test_normal_result = model.predict(self.x_test_t_ss)
                    test_anomaly_result = model.predict(self.x_test_f_ss)

                    t_far, t_frr, t_ber = far_frr_ber(test_normal_result, test_anomaly_result)

                    scores_test[str(model).split('(')[0]] = {
                        'Accuracy': accuracy_score(y_true=y_true, y_pred=test_pred),
                        'Precision': precision_score(y_true, test_pred),
                        'Recall': recall_score(y_true, test_pred),
                        'F1': f1_score(y_true, test_pred),
                        'AUC': roc_auc_score(y_true,
                                             model.decision_function(self.x_test_ss)),
                        'FAR': t_far, 'FRR': t_frr, 'BER': t_ber}

                # 結果のまとめ
                model_index = ['LocalOutlierFactor', 'IsolationForest', 'OneClassSVM', 'EllipticEnvelope']
                result_index = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'FAR', 'FRR', 'BER']
                e = np.zeros((4, 8))
                for i, m_i in enumerate(model_index):
                    for j, df_i in enumerate(result_index):
                        f = 0
                        for m in range(k):
                            f += scores[m_i][m][df_i]
                        e[i][j] = f / k
                a_df = pd.DataFrame(e, index=model_index, columns=result_index)
                print('交差検証k=10の結果')
                print(a_df)

                s_test = pd.DataFrame(scores_test).T
                print(f'認証用データ')
                print(s_test[result_index])

                # result_old.csvへ書き出し
                def output_data(a2, model_index2, result_index2, text, sessions_select):
                    # フォルダがなければ自動的に作成
                    os.makedirs('result/result2020_10', exist_ok=True)
                    # Columnの作成
                    users = pd.Series([self.u_n] * 4, name='user')
                    flag = pd.Series([self.flag_n] * 4, name='flag')
                    sessions = pd.Series([sessions_select] * 4, name='session')
                    performance = pd.Series([text] * 4, name='performance')
                    model2 = pd.Series(model_index2, name='model')
                    # valとtestで条件指定してresult作成
                    if text == 'val':
                        result = pd.DataFrame(a2, columns=result_index2)
                    else:
                        # print('\ntestデータでの結果')
                        result = pd.DataFrame(a2).T
                        # print(result)
                        result = result.reset_index()
                        result = result.drop('index', 1)
                    # 全て結合
                    all_result = pd.concat([users, flag, performance, model2, result, sessions], axis=1)
                    # 書き出し
                    data_now = datetime.datetime.now().strftime("%Y-%m-%d")
                    all_result.to_csv(f'result/result2020_10/result_{data_now}_{text}.csv', mode='a', header=False,
                                      index=False)

                # 交差検証の結果の書き出し
                output_data(a, model_index, result_index, 'val', self.session_select)
                # テストデータの結果の書き出し
                output_data(scores_test, model_index, result_index, 'test', self.session_select)
            except AttributeError as ex:
                print(f"No train data:{ex}")
                pass

        def authentication_phase(self):
            # TODO: あとで考える
            try:
                test = 0
                print(f'{test} No data')
            except AttributeError as ex:
                print(f"No test data:{ex}")
                pass

    oneclassone_a = OneClassOne(a, a_user_extract, user_n, 1, session)
    oneclassone_a.registration_phase()
    # oneclassone_a.authentication_phase()
    oneclassone_b = OneClassOne(b, b_user_extract, user_n, 2, session)
    oneclassone_b.registration_phase()
    oneclassone_c = OneClassOne(c, c_user_extract, user_n, 3, session)
    oneclassone_c.registration_phase()
    oneclassone_d = OneClassOne(d, d_user_extract, user_n, 4, session)
    oneclassone_d.registration_phase()


if __name__ == '__main__':
    print("実験用プログラム 1ストロークの1クラス分類")

    # combination= False
    frank_df = load_frank(False)
    session_list = ['first', 'latter', 'all', 'all_test_shinario2']
    # 41人いるよ
    # user = 35
    for user in range(1, 42):
        main(frank_df, user, session='all_test_shinario2')
