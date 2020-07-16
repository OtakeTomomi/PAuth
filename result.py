import pandas as pd
import numpy as np

# Columnのリスト
val_columns = ['user', 'flag', 'performance', 'model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'FAR', 'FRR', 'BER']
test_columns = ['user', 'flag', 'performance', 'model', 'AUC', 'Accuracy', 'BER', 'F1', 'FAR', 'FRR', 'Precision', 'Recall']

model_index = ['LocalOutlierFactor', 'IsolationForest', 'OneClassSVM', 'EllipticEnvelope']

# 結果のデータを読み込み
# 交差検証データ
val_df = pd.read_csv("result/result_val.csv", sep=",", header=None)
val_df.columns = val_columns
# テストデータ
test_df = pd.read_csv("result/result_test.csv", sep=",", header=None)
test_df.columns = test_columns

# multi_indexの設定
val_m_df = val_df.set_index(['user', 'flag', 'performance', 'model'])
test_m_df = val_df.set_index(['user', 'flag', 'performance', 'model'])

# print(val_m_df)
# print(val_m_df.xs([33, 'val', 'LocalOutlierFactor'], level=['flag', 'performance', 'model']))


# 各flagにおいてデータを有するユーザの数
def count_table(df):
    ct_list = sorted(list(set((df.index.get_level_values('flag')))))
    ct = {}
    for i in ct_list:
        ct_n = df['Accuracy'].xs([i, 'val', 'LocalOutlierFactor'], level=['flag', 'performance', 'model']).count()
        ct[i] = ct_n
    return ct, ct_list
ct, ct_list = count_table(val_m_df)
print(ct)

# ct_list使用して各flagに対して結果を保有しいるユーザを抽出する
'''
あとで処理書いて
'''

# 各flag × 各modelごとの結果を算出
'''
まさに草
'''
# つづりわからん計算
def calcuration(ct, ct_list,df, val_columns, model_index):
    a = np.zeros((4, 8))
    b = np.zeros((4, 8))
    c = np.zeros((4, 8))
    # d = np.zeros((4, 8))
    # print(df['Accuracy'].xs([33, 'val', 'LocalOutlierFactor'], level=['flag', 'performance', 'model']))
    # ct_mean = df['Accuracy'].xs([33, 'val', 'LocalOutlierFactor'], level=['flag', 'performance', 'model']).mean()
    # print(ct_mean)
    for i, model in enumerate(model_index):
        for j, column in enumerate(val_columns[4:]):
        # print(df[column].xs([33, 'val', 'LocalOutlierFactor'], level=['flag', 'performance', 'model']))
            ct_mean = df[column].xs([33, 'val', model], level=['flag', 'performance', 'model']).mean()
            # print(ct_mean)
            a[i][j] = ct_mean
            ct_max = df[column].xs([33, 'val', model], level=['flag', 'performance', 'model']).max()
            b[i][j] = ct_max
            ct_min = df[column].xs([33, 'val', model], level=['flag', 'performance', 'model']).min()
            c[i][j] = ct_min
            # ct_std = df[column].xs([ct_list[i], 'val', model], level=['flag', 'performance', 'model'])
            # d[i][j] = ct_std
            d = 0

    return a, b, c, d
a, b, c, d = calcuration(ct,ct_list,val_m_df,val_columns,model_index)

print('平均')
print(a)

print('最大')
print(b)

print('最小')
print(c)

print('標準偏差')
print(d)


'''
必要な変数
・各フラグの個数

必要な処理
・各flagごとの各モデルをまとめる
・平均を出す
・最大値
・最小値
・標準偏差
'''

# 理想は結果を一発表示する
# a = np.zeros((4, 8))