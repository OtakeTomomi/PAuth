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
    # print(df_flag.shape) # (146, 31)
    fake_data = df_flag[df_flag['user'] != user_n]
    # print(fake_data.shape) # (0, 31)
    # シャッフル
    fake_data_shuffle = fake_data.sample(frac=1, random_state=0).reset_index(drop=True)
    # testデータと同じ数の外れ値を選択
    # outlier = fake_data_shuffle.sample(n=y_test_t_size, random_state=0)
    fake_data_except_outlier = fake_data_shuffle[y_test_t_size:]
    outlier = fake_data_shuffle[:y_test_t_size]

    # print(fake_data_except_outlier.shape)
    # print(outlier.shape)
    return fake_data_except_outlier, outlier


# testデータと外れ値データの結合
def _tf_concat(x_test_t, y_test_t, test_f):
    # outlier == test_f
    test_f_user_n = list(test_f['user'])
    # userをすべて０に変更
    test_f2 = test_f.copy()
    test_f_user_n_change0 = test_f2.replace({'user': test_f_user_n}, 0)
    # print(test_f_user_n_change0.shape) # (0, 31)
    # 偽物のデータを説明変数と目的変数に分割
    x_test_f, y_test_f = x_y_split(test_f_user_n_change0)
    # print(x_test_f.shape) # (0, 30)

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

def doc_conform(df_flag):
    for i in range(1, 42):
        doc_list = df_flag['doc'].value_counts().index.tolist()
        print(f'{i}:{sorted(doc_list)}')


def datasplit_session(df_flag, df_flag_user_extract, user_n, session_select='all', train_size=40, test_size=10):
    df_session = df_flag_user_extract.copy()
    # df_flag_data = df_flag.copy()
    # docが1~5のもの
    if session_select == 'first':
        df_first_pre = df_session.query('1<= doc < 6')
        df_flag_data = df_flag.query('1<= doc < 6')
        # docの役目は終わったので消去
        df_first = df_first_pre.drop('doc', axis=1)
        df_flag_pre = df_flag_data.drop('doc', axis=1)
        # TODO: colunmsの数を確認する
        if df_first['user'].count() >= (train_size + test_size):
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
            fake_data_except_test_f, test_f = _outlier(df_flag_pre, user_n, test_size)

            # test_tとtest_fの結合
            x_test, y_test, x_test_f, y_test_f = _tf_concat(x_test_t, y_test_t, test_f)

            return x_train, y_train, x_test, y_test, x_test_t, y_test_t, x_test_f, y_test_f, test_f, \
                fake_data_except_test_f

        else:
            print('None')
            return 0, 0, 0, 0, 0, 0, 0, 0, 0

    # docが6or7のもの
    elif session_select == 'latter':
        df_session2 = df_session.query('6 <= doc < 8')
        df_first_pre = df_session2.query('1<= doc < 6')
        df_latter_pre = df_session2.query('6 <= doc < 8')
        df_flag_data = df_flag.query('6 <= doc < 8')
        # docの役目は終わったので消去
        df_first = df_first_pre.drop('doc', axis=1)
        df_latter = df_latter_pre.drop('doc', axis=1)
        df_flag_pre = df_flag_data.drop('doc', axis=1)
        if df_first['user'].count() >= train_size and df_latter['user'].count() >= test_size:
            # 説明変数と目的変数に分割
            x_first, y_first = x_y_split(df_first)
            x_latter, y_latter = x_y_split(df_latter)

            x_other_latter, x_test_t, y_other_latter, y_test_t = train_test_split(x_latter, y_latter,
                                                                                  test_size=test_size,
                                                                                  random_state=0, shuffle=True)
            x_train, x_other_first, y_train, y_other_first = train_test_split(x_first, y_first, train_size=train_size,
                                                                              random_state=0, shuffle=True)
            fake_data_except_test_f, test_f = _outlier(df_flag_pre, user_n, test_size)
            x_test, y_test, x_test_f, y_test_f = _tf_concat(x_test_t, y_test_t, test_f)

            return x_train, y_train, x_test, y_test, x_test_t, y_test_t, x_test_f, y_test_f, test_f, \
                fake_data_except_test_f

        else:
            print('None')
            return 0, 0, 0, 0, 0, 0, 0, 0, 0

    # 全部
    elif session_select == 'all':
        df_all_pre = df_session
        df_flag_data = df_flag.query('1<= doc < 8')
        df_all = df_all_pre.drop('doc', axis=1)
        df_flag_pre = df_flag_data.drop('doc', axis=1)
        # データ数が50以上あるか
        if df_all['user'].count() >= (train_size + test_size):
            # 説明変数と目的変数に分割
            x, y = x_y_split(df_all)

            # その他のデータとテスト用(認証用)データに切り分け
            x_train_all, x_test_t, y_train_all, y_test_t = train_test_split(x, y, test_size=test_size, random_state=0,
                                                                            shuffle=True)
            # 学習用データとその他のデータ2に切り分け
            x_train, x_others, y_train, y_others = train_test_split(x_train_all, y_train_all, train_size=train_size,
                                                                    random_state=0, shuffle=True)
            # テストデータの個数をカウント➝外れ値はテストデータ数と同じ数だけ用意する
            # _outlinerの呼び出し
            fake_data_except_test_f, test_f = _outlier(df_flag_pre, user_n, test_size)

            # test_tとtest_fの結合
            x_test, y_test, x_test_f, y_test_f = _tf_concat(x_test_t, y_test_t, test_f)

            return x_train, y_train, x_test, y_test, x_test_t, y_test_t, x_test_f, y_test_f, test_f, \
                fake_data_except_test_f

        else:
            print('None')
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # testデータがlatterシナリオ
    elif session_select == 'all_test_shinario2':
        df_first_pre = df_session.query('1<= doc < 6')
        df_latter_pre = df_session.query('6 <= doc < 8')
        #             df_all_pre = df_session
        df_flag_data = df_flag.query('1<= doc < 8')

        #             df_all = df_all_pre.drop('doc', axis=1)
        df_first = df_first_pre.drop('doc', axis=1)
        df_latter = df_latter_pre.drop('doc', axis=1)
        df_flag_pre = df_flag_data.drop('doc', axis=1)

        # データ数が(train_size + test_size)以上あるか
        if df_first['user'].count() >= train_size and df_latter['user'].count() >= test_size:
            # 説明変数と目的変数に分割
            x_first, y_first = x_y_split(df_first)
            x_latter, y_latter = x_y_split(df_latter)

            x_other_latter, x_test_t, y_other_latter, y_test_t = train_test_split(x_latter, y_latter,
                                                                                  test_size=test_size,
                                                                                  random_state=0, shuffle=True)

            def train_conbinated(other_latter_len, xx_other_latter, yy_other_latter, xx_first, yy_first):
                num = 10
                if other_latter_len < num:
                    x_train_pre, re_x_other_first, y_train_pre, re_y_other_first = train_test_split(xx_first, yy_first,
                                                                                                    train_size=train_size - other_latter_len,
                                                                                                    random_state=0,
                                                                                                    shuffle=True)

                    re_x_train = pd.concat([x_train_pre, xx_other_latter]).reset_index(drop=True)
                    re_y_train = pd.concat([y_train_pre, yy_other_latter]).reset_index(drop=True)

                    return re_x_train, re_x_other_first, re_y_train, re_y_other_first

                elif other_latter_len >= 10:
                    x_train_pre, re_x_other_first, y_train_pre, re_y_other_first = train_test_split(xx_first, yy_first,
                                                                                                    train_size=train_size - num,
                                                                                                    random_state=0,
                                                                                                    shuffle=True)
                    xx_other_latter_shuffle = xx_other_latter.sample(frac=1, random_state=0).reset_index(drop=True)
                    re_x_train = pd.concat([x_train_pre, xx_other_latter_shuffle[:num]]).reset_index(drop=True)
                    # yは値が同じなので特に気にしない
                    re_y_train = pd.concat([y_train_pre, yy_other_latter[:num]]).reset_index(drop=True)

                    return re_x_train, re_x_other_first, re_y_train, re_y_other_first

            x_train, x_other_first, y_train, y_other_first = train_conbinated(len(y_other_latter), x_other_latter,
                                                                              y_other_latter, x_first, y_first)

            fake_data_except_test_f, test_f = _outlier(df_flag_pre, user_n, test_size)
            x_test, y_test, x_test_f, y_test_f = _tf_concat(x_test_t, y_test_t, test_f)

            return x_train, y_train, x_test, y_test, x_test_t, y_test_t, x_test_f, y_test_f, test_f, \
                fake_data_except_test_f

        else:
            print('None')
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


if __name__ == "__main__":
    print("This is module! The name is datasplit_train_test_val")
