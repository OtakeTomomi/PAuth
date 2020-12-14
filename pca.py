# 何も書いてないじゃん


import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# 内部ライブラリ
from expmodule.dataset import load_frank
from expmodule.flag_split import flag4
from expmodule.datasplit_train_test_val import x_y_split
from expmodule.datasplit_train_test_val import datasplit_session

# スケーリング
from sklearn import preprocessing

# モデル
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

# warning ignore code
import warnings
warnings.filterwarnings('ignore')


def main(df, user_n):

    # 上下左右のflagをもとにデータを分割
    a, b, c, d = flag4(df, 'flag')

    class PCAExp(object):

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

                self.x_train, self.y_train, self.x_test, self.y_test, self.x_test_t, \
                self.y_test_t, self.x_test_f, self.y_test_f, self.test_f, self.fake_data_except_test_f \
                    = datasplit_session(self.df_flag, self.df_flag_user_extract, self.u_n, self.session_select,
                                        train_size=40)

                # 標準化
                ss = preprocessing.StandardScaler()
                ss.fit(self.x_train)
                self.x_train_ss = ss.transform(self.x_train)
                self.x_test_ss = ss.transform(self.x_test)
                self.x_test_t_ss = ss.transform(self.x_test_t)
                self.x_test_f_ss = ss.transform(self.x_test_f)

                self.columns = self.x_train.columns.values

            except AttributeError as ex:
                print(f"No data:{ex}")
                pass

            except ValueError as ex2:
                print(f'No data: {self.session_select} == {ex2}')
                pass

        def pca_exp(self):
            pca = PCA(n_components=2)
            pca.fit(self.x_train_ss)


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
                fig = plt.figure(1, figsize=(8, 6))
                ax = Axes3D(fig, rect=[0, 0, 0.8, 0.8], elev=30, azim=60)
                ax.set_title(
                    'user:{0}, strokeType:{1}\n各次元の寄与率: PCA1: {2[0]:.3f} PCA2: {2[1]:.3f} PCA3: {2[2]:.3f}\n累積寄与率: {3:.3f}'.format(
                        self.user_n, self.flag_n, pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)))
                # colors = ["blue","orange"]
                ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2], s=20, c=self.Y_test, alpha=0.5,
                           linewidth=0.5, edgecolors="k", cmap="cool")
                ax.w_xaxis.set_label_text("PCA1")
                ax.w_yaxis.set_label_text("PCA2")
                ax.w_zaxis.set_label_text("PCA3")
                ax.legend(["normal", "anormal"], loc="best")

                # 主成分の寄与率を出力する
                print('user:{0}, stroke:{1}'.format(self.user_n, self.flag_n))
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
                fig = plt.figure(1, figsize=(8, 6))
                ax = Axes3D(fig, rect=[0, 0, 0.8, 0.8], elev=30, azim=60)
                ax.set_title(
                    'user:{0}, strokeType:{1}\n各次元の寄与率: PCA1: {2[0]:.3f} PCA2: {2[1]:.3f} PCA3: {2[2]:.3f}\n累積寄与率: {3:.3f}'.format(
                        self.user_n, self.flag_n, pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)))
                # colors = ["blue","orange"]
                ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2], s=20, c=self.Y_test, alpha=0.5,
                           linewidth=0.5, edgecolors="k", cmap="cool")
                ax.w_xaxis.set_label_text("PCA1")
                ax.w_yaxis.set_label_text("PCA2")
                ax.w_zaxis.set_label_text("PCA3")
                ax.legend(["normal", "anormal"], loc="best")

                # 主成分の寄与率を出力する
                print('user:{0}, stroke:{1}'.format(self.user_n, self.flag_n))
                print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
                print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))
                plt.show()
                # plt.close()


            except ValueError:
                pass

        def plot_IF(self):

            # try:
            Y_true1 = self.Y_test.copy()
            Y_true = Y_true1.replace({self.user_n: 1, 0: -1})

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
                                  behaviour='new', random_state=0)

            isf.fit(X_pca1)
            prediction = isf.predict(X_pca)

            # normal_result = isf.predict(self.X_test1_ss)
            # anomaly_result = isf.predict(self.X_test2_ss)

            # y_score = isf.decision_function(self.X_test_ss)

            plt.figure(figsize=(10, 6))
            mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], self.Y_test)
            mglearn.discrete_scatter(X_pca1[:, 0], X_pca1[:, 1], self.Y_train, c='g')
            plt.legend(["anormal", "normal"], loc="best", fontsize=16)
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


if __name__ == '__main__':
    print("実験用アルゴリズム動作確認")

    # combination = False
    frank_df = load_frank(False)
    for user in range(1, 42):
        main(frank_df, user)
