import pandas as pd
import numpy as np
import os
import pprint


# Comment：変更必要あり
PATH = 'result2021part3'
# PATH2 = 'result2021part3/matome'
PATH2 = 'result2021part3/matome_comb'
# filename_val = 'result_2021-01-05_val'
# filename_test = 'result_2021-01-05_test'
filename_val = 'result_2021-01-05_val_comb'
filename_test = 'result_2021-01-05_test_comb'

# Columnのリスト
val_columns = ['user', 'flag', 'performance', 'model', 'Accuracy', 'Precision',
               'Recall', 'F1', 'AUC', 'FAR', 'FRR', 'BER', 'scenario']
test_columns = ['user', 'flag', 'performance', 'model', 'AUC', 'Accuracy',
                'BER', 'F1', 'FAR', 'FRR', 'Precision', 'Recall', 'scenario']

session_list = ['first', 'latter', 'all', 'all_test_shinario2']
# sessions = {'first':'intra', 'latter':'inter', 'all':'combined', 'all_test_shinario2':'combined2'}

model_index = ['LocalOutlierFactor', 'IsolationForest', 'OneClassSVM', 'EllipticEnvelope']

# 結果のデータを読み込み
# 交差検証データ
val_df = pd.read_csv(f"{PATH}/{filename_val}.csv", sep=",", header=None)
val_df.columns = val_columns
# テストデータ
test_df = pd.read_csv(f"{PATH}/{filename_test}.csv", sep=",", header=None)
test_df.columns = test_columns

# multi_indexの設定
val_m_df = val_df.set_index(['scenario', 'user', 'flag', 'performance', 'model'])
test_m_df = test_df.set_index(['scenario', 'user', 'flag', 'performance', 'model'])

# print(val_m_df)


# 各flagにおいてデータを有するユーザの数
def count_table(df):
    ct_list = sorted(list(set((df.index.get_level_values('flag')))))
    ct = {'first':{}, 'latter':{}, 'all':{}, 'all_test_shinario2':{}}
    for session in session_list:
        for i in ct_list:
            ct_n = df['Accuracy'].xs([session, i, 'val', 'LocalOutlierFactor'], level=['scenario', 'flag', 'performance', 'model']).count()
            ct[session][i] = ct_n
        # print(ct)
    return ct, ct_list


ct, ct_list = count_table(val_m_df)
pprint.pprint(ct)

os.makedirs(PATH2, exist_ok=True)

# 書き出し用の処理
def write_data(df, model_index, columns, flag_n, text, perf, session):
    flag = pd.Series([flag_n] * 4, name='flag')
    performance = pd.Series([perf] * 4, name='performance')
    sessions = pd.Series([session] * 4, name='session')
    model = pd.Series(model_index, name='model')
    df = pd.DataFrame(df, index=model_index, columns=columns)
    print(df)
    df = df.reset_index()
    df = df.drop('index', 1)
    re = pd.concat([sessions, flag, performance, model, df], axis=1)
    if perf == 'val':
        re.to_csv(f'{PATH2}/result_{text}_{perf}.csv', mode='a', header=False, index=False)
        return re
    elif perf == 'test':
        re.to_csv(f'{PATH2}/result_{text}_{perf}.csv', mode='a', header=False, index=False)
        return re


# 各flag × 各modelごとの結果を算出
def calc(ct, ct_list, df, columns, model_index, perf, session_list):
    for s in session_list:
        for k in ct_list:
            a, b, c, d = np.zeros((4, 8)), np.zeros((4, 8)), np.zeros((4, 8)), np.zeros((4, 8))
            for i, model in enumerate(model_index):
                for j, column in enumerate(columns[4:12]):
                    data = df[column].xs([s, k, perf, model], level=['scenario', 'flag', 'performance', 'model'])
                    # print(data)

                    a[i][j] = data.mean()
                    # print(a)
                    b[i][j] = data.max()
                    # print(b)
                    c[i][j] = data.min()
                    # print(c)
                    d[i][j] = data.std()
                    # print(d)

            memori = ['0', 'a', 'b', 'c', 'd']
            # 該当ユーザの抽出
            n = df.xs([s, k, perf, 'LocalOutlierFactor'], level=['scenario', 'flag', 'performance', 'model'])
            # print(n)
            menber = list(n.index.get_level_values('user'))
            if k//10 == 0:
                print(f'\n===================================================================================================='
                      f'\nflag : {memori[k%10]}, {perf}, user数 : {ct[s][k]}人, 該当ユーザ : {menber}, シナリオ：{s}'
                      f'\n====================================================================================================')
            else:
                print(
                    f'\n===================================================================================================='
                    f'\nflag : {memori[k // 10]}+{memori[k % 10]}, {perf}, user数 : {ct[s][k]}人, 該当ユーザ : {menber}, シナリオ：{s}'
                    f'\n====================================================================================================')

            print(f'\n平均')
            # print(a)
            write_data(a, model_index, columns[4:12], k, 'mean', perf, s)
            print(f'\n最大')
            # print(b)
            write_data(b, model_index, columns[4:12], k, 'max', perf, s)
            print(f'\n最小')
            # print(c)
            write_data(c, model_index, columns[4:12], k, 'min', perf, s)
            print(f'\n標準偏差')
            # print(d)
            write_data(d, model_index, columns[4:12], k, 'std', perf, s)


calc(ct, ct_list, val_m_df, val_columns, model_index, 'val', session_list)
calc(ct, ct_list, test_m_df, test_columns, model_index, 'test', session_list)
