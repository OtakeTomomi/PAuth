# 何も書いてないじゃん

import numpy as np
import pandas as pd
import os

# plot
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
# import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# from mpl_toolkits.mplot3d import axes3d

from sklearn.decomposition import PCA

# 内部ライブラリ
from expmodule.dataset import load_frank
from expmodule.flag_split import flag4, flag16
# from expmodule.datasplit_train_test_val import x_y_split
from expmodule.datasplit_train_test_val import datasplit_session

# スケーリング
from sklearn import preprocessing

# モデル
# from sklearn.svm import OneClassSVM
# from sklearn.ensemble import IsolationForest
# from sklearn.covariance import EllipticEnvelope
# from sklearn.neighbors import LocalOutlierFactor

# warning ignore code
import warnings
warnings.filterwarnings('ignore')


def main(df, user_n, session):

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

    os.makedirs('result2022/plot_pca', exist_ok=True)

    class PCAExp(object):

        def __init__(self, df_flag, df_flag_user_extract, u_n, flag_n, session_select):
            flag_memori = ['0', 'a', 'b', 'c', 'd', 'all_flag']
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
                                        test_size=int(len(self.df_flag_user_extract)*0.5))

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

            try:

                sessions = {'first': 'intra', 'latter': 'inter', 'all': 'combined',
                            'all_test_shinario2': 'combined2'}

                test_n = int(int(len(self.df_flag_user_extract) * 0.5)/2)
                # print(self.y_test)
                scaled_train = self.x_train_ss
                scaled_test = self.x_test_ss

                # 二次元
                pca = PCA(n_components=2)
                pca.fit(scaled_test)
                pca_results_train = pca.transform(scaled_train)
                pca_results_test = pca.transform(scaled_test)

                plt.figure(figsize=(8, 8))
                plt.scatter(pca_results_train[:, 0], pca_results_train[:, 1], c='r')
                plt.scatter(pca_results_test[:, 0][:test_n], pca_results_test[:, 1][:test_n], c='g')
                plt.scatter(pca_results_test[:, 0][test_n:], pca_results_test[:, 1][test_n:], c='b')
                # 主成分の寄与率を出力する
                # print('user:{0}, stroke:{1}'.format(self.u_n, self.flag_n))
                # print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
                # print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

                # plt.show()
                plt.close()

                def render_frame(angle):

                    # 三次元
                    fig = plt.figure(1, figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    pca3 = PCA(n_components=3)
                    pca3.fit(scaled_test)
                    pca3_results_train = pca3.transform(scaled_train)
                    pca3_results_test = pca3.transform(scaled_test)
                    ax.scatter(pca3_results_train[:, 0], pca3_results_train[:, 1], pca3_results_train[:, 2],
                               c='r', alpha=0.5, linewidth=0.5, cmap="cool", s=20)
                    ax.scatter(pca3_results_test[:, 0][:test_n], pca3_results_test[:, 1][:test_n], pca3_results_test[:, 2][:test_n],
                               c='g', alpha=0.5, linewidth=0.5, cmap="cool", s=20)
                    ax.scatter(pca3_results_test[:, 0][test_n:], pca3_results_test[:, 1][test_n:], pca3_results_test[:, 2][test_n:],
                               c='b', alpha=0.5, linewidth=0.5, cmap="cool", s=20)

                    ax.view_init(30, angle)
                    # plt.show()
                    plt.close()

                    new_list = [round(pca3.explained_variance_ratio_[n], 3) for n in range(len(pca3.explained_variance_ratio_))]
                    ax.set_title(f"{self.u_n}_{self.flag_n}_{sessions[self.session_select]}\nPCA: {new_list}")
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')

                    # plt.show()

                    # PIL Image に変換
                    buf = BytesIO()
                    fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)

                    # print('user:{0}, stroke:{1}'.format(self.u_n, self.flag_n))
                    # print('各次元の寄与率: {0}'.format(pca3.explained_variance_ratio_))
                    # print('累積寄与率: {0}'.format(sum(pca3.explained_variance_ratio_)))

                    return Image.open(buf)

                os.makedirs('result2022/plot_pca', exist_ok=True)
                render_frame(30)
                images = [render_frame(angle) for angle in range(360)]
                images[0].save(f'result2022/plot_pca/output{self.u_n}_{self.flag_n}_{sessions[self.session_select]}.gif', save_all=True,
                               append_images=images[1:],
                               optimize=False, duration=100, loop=0)

            except AttributeError as ex:
                print(ex)
                pass


    oneclassone_a = PCAExp(a, a_user_extract, user_n, 1, session)
    oneclassone_a.pca_exp()
    # oneclassone_a.authentication_phase()
    oneclassone_b = PCAExp(b, b_user_extract, user_n, 2, session)
    oneclassone_b.pca_exp()
    oneclassone_c = PCAExp(c, c_user_extract, user_n, 3, session)
    oneclassone_c.pca_exp()
    oneclassone_d = PCAExp(d, d_user_extract, user_n, 4, session)
    oneclassone_d.pca_exp()


def main2(df, user_n, session):
    # データをmulti_flagを基準に分割
    aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd = flag16(df, 'multi_flag')
    def timeprocess(dffue):
        #     for i in range(len(df)):
        dffue['between'] = dffue['stroke_inter'].iloc[:] - dffue['stroke_duration'].iloc[:]

        def outlier_2s(dffue):
            col = dffue['between']
            # 平均と標準偏差
            average = np.mean(col)
            sd = np.std(col)

            # 外れ値の基準点
            outlier_min = average - (sd) * 2
            outlier_max = average + (sd) * 2

            # 範囲から外れている値を除く
            col[col < outlier_min] = None
            col[col > outlier_max] = None
            return dffue.dropna(how='any', axis=0)

        dffue2 = outlier_2s(dffue)
        dffue3 = dffue2.drop({'between'}, axis=1)
        return dffue3
    # flagごとに選択したユーザを抽出する
    def select_user_flag(df_flag, u_n, text):
        os.makedirs('result/info', exist_ok=True)
        df_flag_user_extract = df_flag[df_flag['user'] == u_n]

        df_flag_user_extract2 = timeprocess(df_flag_user_extract)
        # data_item[3] = pd.DataFrame([len(df_flag_user_extract2)])
        return df_flag_user_extract2

    aa_user_extract = select_user_flag(aa, user_n, 'aa')
    ab_user_extract = select_user_flag(ab, user_n, 'ab')
    ac_user_extract = select_user_flag(ac, user_n, 'ac')
    ad_user_extract = select_user_flag(ad, user_n, 'ad')

    ba_user_extract = select_user_flag(ba, user_n, 'ba')
    bb_user_extract = select_user_flag(bb, user_n, 'bb')
    bc_user_extract = select_user_flag(bc, user_n, 'bc')
    bd_user_extract = select_user_flag(bd, user_n, 'bd')

    ca_user_extract = select_user_flag(ca, user_n, 'ca')
    cb_user_extract = select_user_flag(cb, user_n, 'cb')
    cc_user_extract = select_user_flag(cc, user_n, 'cc')
    cd_user_extract = select_user_flag(cd, user_n, 'cd')

    da_user_extract = select_user_flag(da, user_n, 'da')
    db_user_extract = select_user_flag(db, user_n, 'db')
    dc_user_extract = select_user_flag(dc, user_n, 'dc')
    dd_user_extract = select_user_flag(dd, user_n, 'dd')

    class PCAExpTwo(object):

        def __init__(self, df_flag, df_flag_user_extract, u_n, flag_n, session_select):
            flag_memori = ['0', 'a', 'b', 'c', 'd']
            print(
                f'\n-----------------------------------------------------------------\n'
                f'{user_n} : {flag_memori[flag_n // 10]} + {flag_memori[flag_n % 10]} : {session_select}'
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
                                        test_size=int(len(self.df_flag_user_extract)*0.5))

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

        def pca_exp2(self):

            try:
                # 名称変更用の辞書
                sessions = {'first': 'intra', 'latter': 'inter', 'all': 'combined',
                            'all_test_shinario2': 'combined2'}

                test_n = int(int(len(self.df_flag_user_extract)*0.5)/2)
                # print(self.y_test)
                scaled_train = self.x_train_ss
                scaled_test = self.x_test_ss

                # 二次元
                pca = PCA(n_components=2)
                pca.fit(scaled_test)
                pca_results_train = pca.transform(scaled_train)
                pca_results_test = pca.transform(scaled_test)

                plt.figure(figsize=(8, 8))
                plt.scatter(pca_results_train[:, 0], pca_results_train[:, 1], c='r')
                plt.scatter(pca_results_test[:, 0][:test_n], pca_results_test[:, 1][:test_n], c='g')
                plt.scatter(pca_results_test[:, 0][test_n:], pca_results_test[:, 1][test_n:], c='b')
                # 主成分の寄与率を出力する
                # print('user:{0}, stroke:{1}'.format(self.u_n, self.flag_n))
                # print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
                # print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

                # plt.show()
                plt.close()

                def render_frame(angle):

                    # 三次元
                    fig = plt.figure(1, figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    pca3 = PCA(n_components=3)
                    pca3.fit(scaled_test)
                    pca3_results_train = pca3.transform(scaled_train)
                    pca3_results_test = pca3.transform(scaled_test)
                    ax.scatter(pca3_results_train[:, 0], pca3_results_train[:, 1], pca3_results_train[:, 2],
                               c='r', alpha=0.5, linewidth=0.5, cmap="cool", s=20)
                    ax.scatter(pca3_results_test[:, 0][:test_n], pca3_results_test[:, 1][:test_n], pca3_results_test[:, 2][:test_n],
                               c='g', alpha=0.5, linewidth=0.5, cmap="cool", s=20)
                    ax.scatter(pca3_results_test[:, 0][test_n:], pca3_results_test[:, 1][test_n:], pca3_results_test[:, 2][test_n:],
                               c='b', alpha=0.5, linewidth=0.5, cmap="cool", s=20)

                    X2 = pca3_results_train[:, 0]
                    Y2 = pca3_results_train[:, 1]
                    Z2 = pca3_results_train[:, 2]

                    def reject_outliers(data, m=2):
                        return data[abs(data - np.mean(data)) < m * np.std(data)]

                    X = reject_outliers(X2)
                    Y = reject_outliers(Y2)
                    Z = reject_outliers(Z2)

                    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5

                    mid_x = (X.max() + X.min()) * 0.5
                    mid_y = (Y.max() + Y.min()) * 0.5
                    mid_z = (Z.max() + Z.min()) * 0.5

                    ax.set_xlim(mid_x - max_range * 1.5, mid_x + max_range * 1.5)
                    ax.set_ylim(mid_y - max_range * 1.5, mid_y + max_range * 1.5)
                    ax.set_zlim(mid_z - max_range * 1.5, mid_z + max_range * 1.5)

                    ax.view_init(30, angle)
                    # plt.show()
                    plt.close()

                    new_list = [round(pca3.explained_variance_ratio_[n], 3) for n in range(len(pca3.explained_variance_ratio_))]
                    ax.set_title(f"{self.u_n}_{self.flag_n}_{sessions[self.session_select]}\nPCA: {new_list}")
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')

                    # plt.show()

                    # PIL Image に変換
                    buf = BytesIO()
                    fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)

                    # print('user:{0}, stroke:{1}'.format(self.u_n, self.flag_n))
                    # print('各次元の寄与率: {0}'.format(pca3.explained_variance_ratio_))
                    # print('累積寄与率: {0}'.format(sum(pca3.explained_variance_ratio_)))

                    return Image.open(buf)

                os.makedirs('result2022/plot_pca', exist_ok=True)
                render_frame(30)
                images = [render_frame(angle) for angle in range(360)]
                images[0].save(f'result2022/plot_pca/output{self.u_n}_{self.flag_n}_{sessions[self.session_select]}.gif', save_all=True,
                               append_images=images[1:],
                               optimize=False, duration=100, loop=0)

            except AttributeError as ex:
                print(ex)
                pass

    oneclasstwo_aa = PCAExpTwo(aa, aa_user_extract, user_n, 11, session)
    oneclasstwo_aa.pca_exp2()
    # oneclassone_a.authentication_phase()
    oneclasstwo_ab = PCAExpTwo(ab, ab_user_extract, user_n, 12, session)
    oneclasstwo_ab.pca_exp2()
    oneclasstwo_ac = PCAExpTwo(ac, ac_user_extract, user_n, 13, session)
    oneclasstwo_ac.pca_exp2()
    oneclasstwo_ad = PCAExpTwo(ad, ad_user_extract, user_n, 14, session)
    oneclasstwo_ad.pca_exp2()

    oneclasstwo_ba = PCAExpTwo(ba, ba_user_extract, user_n, 21, session)
    oneclasstwo_ba.pca_exp2()
    # oneclassone_a.authentication_phase()
    oneclasstwo_bb = PCAExpTwo(bb, bb_user_extract, user_n, 22, session)
    oneclasstwo_bb.pca_exp2()
    oneclasstwo_bc = PCAExpTwo(bc, bc_user_extract, user_n, 23, session)
    oneclasstwo_bc.pca_exp2()
    oneclasstwo_bd = PCAExpTwo(bd, bd_user_extract, user_n, 24, session)
    oneclasstwo_bd.pca_exp2()

    oneclasstwo_ca = PCAExpTwo(ca, ca_user_extract, user_n, 31, session)
    oneclasstwo_ca.pca_exp2()
    # oneclassone_a.authentication_phase()
    oneclasstwo_cb = PCAExpTwo(cb, cb_user_extract, user_n, 32, session)
    oneclasstwo_cb.pca_exp2()
    oneclasstwo_cc = PCAExpTwo(cc, cc_user_extract, user_n, 33, session)
    oneclasstwo_cc.pca_exp2()
    oneclasstwo_cd = PCAExpTwo(cd, cd_user_extract, user_n, 34, session)
    oneclasstwo_cd.pca_exp2()

    oneclasstwo_da = PCAExpTwo(da, da_user_extract, user_n, 41, session)
    oneclasstwo_da.pca_exp2()
    # oneclassone_a.authentication_phase()
    oneclasstwo_db = PCAExpTwo(db, db_user_extract, user_n, 42, session)
    oneclasstwo_db.pca_exp2()
    oneclasstwo_dc = PCAExpTwo(dc, dc_user_extract, user_n, 43, session)
    oneclasstwo_dc.pca_exp2()
    oneclasstwo_dd = PCAExpTwo(dd, dd_user_extract, user_n, 44, session)
    oneclasstwo_dd.pca_exp2()


if __name__ == '__main__':
    print("実験用アルゴリズム動作確認")

    # combination = False
    # frank_df = load_frank(False)
    session_list = ['first', 'latter', 'all', 'all_test_shinario2']
    # main(frank_df, 35, session='first')
    # # 41人いるよ
    # for session in session_list:
    #     for user in [2, 3, 23, 35, 38]:
    #         main(frank_df, user, session=session)

    # for session in session_list:
    #     for user in range(1, 42):
    #         main(frank_df, user, session=session)

    # combination = True
    frank_df2 = load_frank(True)
    # print(len(frank_df2.columns))
    for session in session_list:
        for user in range(1, 42):
            main2(frank_df2, user, session)
