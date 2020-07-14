
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def X_Y(user_select):
    X = user_select.drop("user", 1)
    Y = user_select.user
    return X, Y

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

if __name__ == "__main__":
    print('data_split_exp module')