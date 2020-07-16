import pandas as pd
import numpy as np

# Columnのリスト
val_columns = ['user','flag','performance','model','Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'FAR', 'FRR', 'BER']
test_columns = ['user','flag','performance','model', 'AUC', 'Accuracy', 'BER', 'F1', 'FAR', 'FRR', 'Precision', 'Recall']

# model_index = ['LocalOutlierFactor', 'IsolationForest', 'OneClassSVM', 'EllipticEnvelope']

# 結果のデータを読み込み
# 交差検証データ
val_df = pd.read_csv("result/result_val.csv", sep = ",", header=None)
val_df.columns = val_columns
# テストデータ
test_df = pd.read_csv("result/result_test.csv", sep = ",", header=None)
test_df.columns = test_columns

# multi_indexの設定
val_m_df = val_df.set_index(['user', 'flag','performance', 'model'])
test_m_df = val_df.set_index(['user', 'flag','performance', 'model'])

# print(val_m_df)
# print(val_m_df.xs([33, 'val', 'LocalOutlierFactor'], level=['flag', 'performance', 'model']))


# 各flagにおいてデータを有するユーザの数
def count_table(df):
    # ct = np.zeros((1, 16))
    ct_list = sorted(list(set((df.index.get_level_values('flag')))))
    ct = {}
    for i in ct_list:
        ct_n = df['Accuracy'].xs([i, 'val', 'LocalOutlierFactor'], level=['flag', 'performance', 'model']).count()
        ct[i] = ct_n
    print(ct)
    return ct
ct = count_table(val_m_df)

# ct_lest使用して各flagに対して結果を保有しいるユーザを抽出する
'''
処理書いて
'''

# 各flag


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