"""
1. セッションに沿うようにtrain(train_val,test_val), testに分割

"""

import pandas as pd
from sklearn.model_selection import train_test_split


def x_y_split(df_flag_user_extract):
    x_split = df_flag_user_extract.drop("user", 1)
    y_split = df_flag_user_extract.user
    return x_split, y_split


# 偽物(外れ値)のデータの選択
def _outlier(df_flag, user_n, y_test_t_size):
    # user_n以外のuserを抽出
    fake_data = df_flag[df_flag['user'] != user_n]
    # シャッフル
    fake_data_shuffle = fake_data.sample(frac=1, random_state=0).reset_index(drop=True)
    # testデータと同じ数の外れ値を選択
    # outlier = fake_data_shuffle.sample(n=y_test_t_size, random_state=0)
    fake_data_except_outlier = fake_data_shuffle[y_test_t_size:]
    outlier = fake_data_shuffle[:y_test_t_size]
    return fake_data_except_outlier, outlier


# testデータと外れ値データの結合
def _tf_concat(x_test_t, y_test_t, test_f):
    # outlier == test_f
    test_f_user_n = list(test_f['user'])
    # userをすべて０に変更
    test_f2 = test_f.copy()
    test_f_user_n_change0 = test_f2.replace({'user': test_f_user_n}, 0)
    # 偽物のデータを説明変数と目的変数に分割
    x_test_f, y_test_f = x_y_split(test_f_user_n_change0)

    # testデータの結合
    x_test = pd.concat([x_test_t, x_test_f]).reset_index(drop=True)
    y_test = pd.concat([y_test_t, y_test_f]).reset_index(drop=True)

    return x_test, y_test, x_test_f, y_test_f


def _train_test(df_flag, df_flag_user_extract, user_n, test_size, train_size):
    if df_flag_user_extract['user'].count() >= 50:
        # 説明変数と目的変数に分割
        x, y = x_y_split(df_flag_user_extract)

        # その他のデータとテスト用(認証用)データに切り分け
        x_train_all, x_test_t, y_train_all, y_test_t = train_test_split(x, y, test_size=test_size, random_state=0,
                                                                        shuffle=True)
        # 学習用データとその他のデータ2に切り分け
        x_train, x_others, y_train, y_others = train_test_split(x_train_all, y_train_all, train_size=train_size,
                                                                random_state=0, shuffle=True)
        # テストデータの個数をカウント➝外れ値はテストデータ数と同じ数だけ用意する
        # _outlinerの呼び出し
        fake_data, test_f = _outlier(df_flag, user_n, test_size)

        # test_tとtest_fの結合
        x_test, y_test, x_test_f, y_test_f = _tf_concat(x_test_t, y_test_t, test_f)

        return x_train, y_train, x_test, y_test

    else:
        print('None')
        return 0, 0, 0, 0
        pass


def datasplit_session(df_flag, df_flag_user_extract, user_n, session_select='all', train_size=40, test_size=10):
    df_session = df_flag.copy()
    # docが1~5のもの
    if session_select == 'first':
        df_first_pre = df_session.query('1<= doc < 6')
        # docの役目は終わったので消去
        df_first = df_first_pre.drop('doc', axis=1)
        # TODO: colunmsの数を確認する
        if df_first['user'].count() >= 50:
            # 説明変数と目的変数に分割
            x, y = x_y_split(df_first)

            # その他のデータとテスト用(認証用)データに切り分け
            x_train_all, x_test_t, y_train_all, y_test_t = train_test_split(x, y, test_size=test_size, random_state=0,
                                                                            shuffle=True)
            # 学習用データとその他のデータ2に切り分け
            x_train, x_others, y_train, y_others = train_test_split(x_train_all, y_train_all, train_size=train_size,
                                                                    random_state=0, shuffle=True)
            # テストデータの個数をカウント➝外れ値はテストデータ数と同じ数だけ用意する
            # _outlinerの呼び出し
            fake_data_except_test_f, test_f = _outlier(df_first, user_n, test_size)

            # test_tとtest_fの結合
            x_test, y_test, x_test_f, y_test_f = _tf_concat(x_test_t, y_test_t, test_f)

            return x_train, y_train, x_test, y_test, x_test_t, y_test_t, x_test_f, y_test_f, test_f, \
                fake_data_except_test_f

        else:
            print('None')
            return 0, 0, 0, 0, 0, 0, 0, 0, 0

    # docが6or7のもの
    elif session_select == 'latter':
        df_session2 = df_session.query('1<= doc < 8')
        df_first_pre = df_session2.query('1<= doc < 6')
        df_latter_pre = df_session2.query('6 <= doc < 8')
        # docの役目は終わったので消去
        df_first = df_first_pre.drop('doc', axis=1)
        df_latter = df_latter_pre.drop('doc', axis=1)
        if df_first['user'].count() >= 50 and df_latter['user'].count() >= 10:
            # 説明変数と目的変数に分割
            x_first, y_first = x_y_split(df_first)
            x_latter, y_latter = x_y_split(df_latter)

            x_other_latter, x_test_t, y_other_latter, y_test_t = train_test_split(x_latter, y_latter,
                                                                                  test_size=test_size,
                                                                                  random_state=0, shuffle=True)
            x_train, x_other_first, y_train, y_other_first = train_test_split(x_first, y_first, train_size=train_size,
                                                                              random_state=0, shuffle=True)
            fake_data_except_test_f, test_f = _outlier(df_latter, user_n, test_size)
            x_test, y_test, x_test_f, y_test_f = _tf_concat(x_test_t, y_test_t, test_f)

            return x_train, y_train, x_test, y_test, x_test_t, y_test_t, x_test_f, y_test_f, test_f, \
                fake_data_except_test_f

        else:
            print('None')
            return 0, 0, 0, 0, 0, 0, 0, 0, 0

    # 全部
    elif session_select == 'all':
        df_all_pre = df_session
        df_all = df_all_pre.drop('doc', axis=1)
        # データ数が50以上あるか
        if df_flag_user_extract['user'].count() >= 50:
            # 説明変数と目的変数に分割
            x, y = x_y_split(df_flag_user_extract)

            # その他のデータとテスト用(認証用)データに切り分け
            x_train_all, x_test_t, y_train_all, y_test_t = train_test_split(x, y, test_size=test_size, random_state=0,
                                                                            shuffle=True)
            # 学習用データとその他のデータ2に切り分け
            x_train, x_others, y_train, y_others = train_test_split(x_train_all, y_train_all, train_size=train_size,
                                                                    random_state=0, shuffle=True)
            # テストデータの個数をカウント➝外れ値はテストデータ数と同じ数だけ用意する
            # _outlinerの呼び出し
            fake_data_except_test_f, test_f = _outlier(df_all, user_n, test_size)

            # test_tとtest_fの結合
            x_test, y_test, x_test_f, y_test_f = _tf_concat(x_test_t, y_test_t, test_f)

            return x_train, y_train, x_test, y_test, x_test_t, y_test_t, x_test_f, y_test_f, test_f, \
                fake_data_except_test_f

        else:
            print('None')
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


if __name__ == "__main__":
    print("This is module! The name is datasplit_train_test_val")
